from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import train_final_model as base
import train_spike_probability_gate as gate_mod


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "log"

BASE_SUBMISSION_PATH = DATA_DIR / "submission_spike_gate_aggressive.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
VALIDATION_GATE_PATH = DATA_DIR / "spike_gate_validation_predictions.csv"
SEARCH_GATE_PATH = DATA_DIR / "spike_gate_search_results.csv"
FUTURE_PROMO_FEATURES_PATH = DATA_DIR / "future_promo_calendar_features.csv"

SEARCH_RESULTS_PATH = DATA_DIR / "adaptive_scaling_search_results.csv"
SUBMISSION_BEST_PATH = DATA_DIR / "submission_adaptive_scale_best.csv"
SUBMISSION_SPIKE_PATH = DATA_DIR / "submission_adaptive_scale_spike.csv"
SUBMISSION_CONSERVATIVE_PATH = DATA_DIR / "submission_adaptive_scale_conservative.csv"
SUBMISSION_PLUS_PATH = DATA_DIR / "submission_adaptive_scale_plus.csv"
SUBMISSION_MINUS_PATH = DATA_DIR / "submission_adaptive_scale_minus.csv"

REPORT_PATH = LOG_DIR / "adaptive_scaling_report.txt"
LOG_FILE = LOG_DIR / "adaptive_scaling_layer.log"

DATE_COL = base.DATE_COL
TARGET_COL = base.TARGET_COL
COGS_COL = base.COGS_COL
TRAIN_CUTOFF = base.TRAIN_CUTOFF
RANDOM_STATE = base.RANDOM_STATE

HIGH_RISK_MONTHS = {2, 3, 5, 8}
ALPHAS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
LOW_UPLIFTS = [0.03, 0.05]
MID_UPLIFTS = [0.08, 0.10, 0.12]
HIGH_UPLIFTS = [0.15, 0.18, 0.22, 0.25]
CONTEXT_ALPHAS = [0.05, 0.08, 0.10, 0.12]
CONTEXT_BETAS = [0.00, 0.02, 0.04]
CONTEXT_GAMMAS = [0.00, 0.03, 0.05]
CONTEXT_DELTAS = [0.00, 0.02, 0.04]
CONSERVATIVE_THRESHOLDS = [0.35, 0.40, 0.50, 0.60]
CONSERVATIVE_UPLIFTS = [0.05, 0.08, 0.10, 0.12, 0.15]
NON_SPIKE_GUARD = 1.01
OVERALL_GUARD = 1.01
MEMORY_SIGNAL_WEIGHT = 0.20
CAMPAIGN_SIGNAL_WEIGHT = 0.15
PHASE_PEAK_WEIGHT = 0.20
PHASE_LAST7_WEIGHT = 0.10
PHASE_FIRST7_WEIGHT = 0.05


class Reporter:
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self.lines: list[str] = []

    def emit(self, message: str = "") -> None:
        print(message)
        self.lines.append(message)
        if message:
            self.logger.info(message)

    def emit_frame(self, title: str, frame: pd.DataFrame | pd.Series) -> None:
        self.emit(title)
        if frame.empty:
            self.emit("(empty)")
            return
        self.emit(frame.to_string(index=False))

    def save(self, path: Path = REPORT_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        self.logger.info("Saved report to %s", path)


def setup_logging(log_file: Path = LOG_FILE) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("adaptive_scaling_layer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.info("Logging initialized: %s", log_file)
    return logger


def safe_divide(numerator: Any, denominator: Any, epsilon: float = 1e-6) -> Any:
    if isinstance(numerator, pd.Series) or isinstance(denominator, pd.Series):
        numerator_series = pd.Series(numerator, copy=False)
        denominator_series = pd.Series(denominator, copy=False)
        valid = numerator_series.notna() & denominator_series.notna()
        result = pd.Series(np.nan, index=numerator_series.index, dtype=float)
        result.loc[valid] = numerator_series.loc[valid] / (denominator_series.loc[valid].abs() + epsilon)
        return result

    if pd.isna(numerator) or pd.isna(denominator):
        return np.nan
    return float(numerator) / (abs(float(denominator)) + epsilon)


def load_sample_submission(path: Path = SAMPLE_SUBMISSION_PATH) -> pd.DataFrame:
    return base.load_sample_submission(path)


def load_base_submission(sample_submission: pd.DataFrame, path: Path = BASE_SUBMISSION_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Base submission not found: {path}")
    output = pd.read_csv(path, parse_dates=[DATE_COL], low_memory=False)
    output[DATE_COL] = pd.to_datetime(output[DATE_COL], errors="coerce").dt.normalize()
    validate_submission_frame(output[[DATE_COL, TARGET_COL, COGS_COL]], sample_submission)
    return output[[DATE_COL, TARGET_COL, COGS_COL]].copy()


def load_validation_gate_file(path: Path = VALIDATION_GATE_PATH) -> tuple[pd.DataFrame, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Spike gate validation file not found: {path}")

    df = pd.read_csv(path, parse_dates=[DATE_COL], low_memory=False)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce").dt.normalize()
    if df[DATE_COL].isna().any():
        raise ValueError("Validation gate file contains invalid dates")

    actual_col = next((column for column in ["actual_Revenue", "Revenue", "actual"] if column in df.columns), None)
    base_col = next((column for column in ["base_pred", "predicted_Revenue", "predicted", "best_adjusted_pred"] if column in df.columns), None)
    prob_top10_col = next((column for column in ["prob_top10", "spike_prob_top10"] if column in df.columns), None)
    prob_top15_col = next((column for column in ["prob_top15", "spike_prob_top15"] if column in df.columns), None)

    mapping = {
        "actual_col": actual_col or "",
        "base_col": base_col or "",
        "prob_top10_col": prob_top10_col or "",
        "prob_top15_col": prob_top15_col or "",
    }

    if actual_col is None:
        raise ValueError("Could not infer actual revenue column from spike gate validation file")
    if base_col is None:
        raise ValueError("Could not infer base prediction column from spike gate validation file")

    columns = [DATE_COL, actual_col, base_col]
    if prob_top10_col is not None:
        columns.append(prob_top10_col)
    if prob_top15_col is not None:
        columns.append(prob_top15_col)
    return df[columns].copy(), mapping


def choose_best_config(search_df: pd.DataFrame) -> dict[str, Any]:
    ordered = search_df.sort_values(
        ["accepted", "RMSE", "top10_RMSE", "top10_underprediction", "non_spike_RMSE", "MAE"],
        ascending=[False, True, True, True, True, True],
    )
    return ordered.iloc[0].to_dict()


def choose_aggressive_config(search_df: pd.DataFrame) -> dict[str, Any]:
    best = choose_best_config(search_df)
    same_rule = search_df[
        (search_df["label_name"] == best["label_name"])
        & (search_df["gating_mode"] == best["gating_mode"])
        & (np.isclose(search_df["threshold"], float(best["threshold"])))
        & (search_df["uplift"] > float(best["uplift"]))
    ].sort_values(["uplift", "RMSE"])
    if not same_rule.empty:
        return same_rule.iloc[0].to_dict()
    return best


def reconstruct_aggressive_validation(
    validation_df: pd.DataFrame,
    mapping: dict[str, str],
    aggressive_config: dict[str, Any],
) -> pd.DataFrame:
    actual = pd.to_numeric(validation_df[mapping["actual_col"]], errors="coerce")
    base_pred = pd.to_numeric(validation_df[mapping["base_col"]], errors="coerce")
    label_name = str(aggressive_config["label_name"])
    prob_column = f"prob_{label_name}"
    mapped_prob_column = mapping.get(f"{prob_column}_col", "")
    if not mapped_prob_column:
        mapped_prob_column = mapping["prob_top10_col"] if label_name == "top10" else mapping["prob_top15_col"]
    if not mapped_prob_column:
        raise ValueError(f"Missing probability column for aggressive config label={label_name}")

    probs = pd.to_numeric(validation_df[mapped_prob_column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    gate = gate_mod.gate_values(
        probs,
        str(aggressive_config["gating_mode"]),
        float(aggressive_config["threshold"]),
    )
    adjusted = base_pred.to_numpy(dtype=float) * (1.0 + float(aggressive_config["uplift"]) * gate)
    output = pd.DataFrame(
        {
            DATE_COL: validation_df[DATE_COL],
            "actual_Revenue": actual,
            "base_pred": adjusted,
            "prob_top10": pd.to_numeric(validation_df[mapping["prob_top10_col"]], errors="coerce").fillna(0.0)
            if mapping["prob_top10_col"]
            else 0.0,
            "prob_top15": pd.to_numeric(validation_df[mapping["prob_top15_col"]], errors="coerce").fillna(0.0)
            if mapping["prob_top15_col"]
            else 0.0,
        }
    )
    return output


def build_campaign_strength_lookup(training_context: pd.DataFrame) -> dict[str, float]:
    promo_days = training_context[pd.to_numeric(training_context["calendar_any_promo"], errors="coerce").fillna(0.0) > 0].copy()
    if promo_days.empty:
        return {column: 0.0 for column in gate_mod.CAMPAIGN_FLAG_COLUMNS}

    values: dict[str, float] = {}
    for column in gate_mod.CAMPAIGN_FLAG_COLUMNS:
        active = promo_days[pd.to_numeric(promo_days[column], errors="coerce").fillna(0.0) > 0]
        values[column] = float(active[TARGET_COL].mean()) if not active.empty else 0.0

    max_value = max(values.values()) if values else 0.0
    if max_value <= 0:
        return {key: 0.0 for key in values}
    return {key: float(value / max_value) for key, value in values.items()}


def prepare_scaling_context(
    feature_frame: pd.DataFrame,
    promo_intensity_scale: float,
    campaign_strength_lookup: dict[str, float],
) -> pd.DataFrame:
    output = feature_frame.copy()
    output["month"] = pd.to_numeric(output["month"], errors="coerce").fillna(0).astype(int)
    output["is_high_risk_month"] = output["month"].isin(HIGH_RISK_MONTHS).astype(int)
    output["is_promo"] = pd.to_numeric(output["calendar_any_promo"], errors="coerce").fillna(0.0).clip(0, 1)
    output["promo_intensity"] = (
        pd.to_numeric(output["calendar_avg_discount_value"], errors="coerce").fillna(0.0)
        * pd.to_numeric(output["calendar_active_promo_count"], errors="coerce").fillna(0.0)
    )
    base_norm = (output["promo_intensity"] / max(promo_intensity_scale, 1e-6)).clip(lower=0.0, upper=1.0)

    promo_day_number_series = (
        pd.to_numeric(output["promo_day_number"], errors="coerce")
        if "promo_day_number" in output.columns
        else pd.to_numeric(output.get("promo_progress_ratio", 0.0), errors="coerce").fillna(0.0)
        * pd.to_numeric(output.get("promo_duration", 0.0), errors="coerce").fillna(0.0)
    )
    output["promo_is_first_7_days"] = (
        (promo_day_number_series.fillna(0.0) > 0)
        & (promo_day_number_series.fillna(0.0) <= 7)
    ).astype(int)
    output["promo_is_last_7_days"] = (
        (pd.to_numeric(output["promo_days_remaining"], errors="coerce").fillna(9999.0) >= 0)
        & (pd.to_numeric(output["promo_days_remaining"], errors="coerce").fillna(9999.0) <= 6)
        & (output["is_promo"] > 0)
    ).astype(int)
    progress = pd.to_numeric(output["promo_progress_ratio"], errors="coerce").fillna(0.0)
    output["is_peak_phase"] = ((progress >= 0.3) & (progress <= 0.7) & (output["is_promo"] > 0)).astype(int)

    campaign_signal = np.zeros(len(output), dtype=float)
    for column in gate_mod.CAMPAIGN_FLAG_COLUMNS:
        if column in output.columns:
            campaign_signal += (
                pd.to_numeric(output[column], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0).to_numpy(dtype=float)
                * float(campaign_strength_lookup.get(column, 0.0))
            )
    output["campaign_strength_signal"] = np.clip(campaign_signal, 0.0, 1.5)

    phase_multiplier = (
        1.0
        + PHASE_PEAK_WEIGHT * output["is_peak_phase"].to_numpy(dtype=float)
        + PHASE_LAST7_WEIGHT * output["promo_is_last_7_days"].to_numpy(dtype=float)
        + PHASE_FIRST7_WEIGHT * output["promo_is_first_7_days"].to_numpy(dtype=float)
    )
    output["normalized_promo_intensity"] = (base_norm.to_numpy(dtype=float) * phase_multiplier * (1.0 + CAMPAIGN_SIGNAL_WEIGHT * output["campaign_strength_signal"].to_numpy(dtype=float))).clip(0.0, 1.5)

    lag365_p90 = (
        pd.to_numeric(output["lag365_above_p90"], errors="coerce").fillna(0.0)
        if "lag365_above_p90" in output.columns
        else pd.Series(0.0, index=output.index, dtype=float)
    )
    lag365_p95 = (
        pd.to_numeric(output["lag365_above_p95"], errors="coerce").fillna(0.0)
        if "lag365_above_p95" in output.columns
        else pd.Series(0.0, index=output.index, dtype=float)
    )
    spike_strength = (
        pd.to_numeric(output["spike_strength_365"], errors="coerce").fillna(0.0)
        if "spike_strength_365" in output.columns
        else pd.Series(0.0, index=output.index, dtype=float)
    )
    output["memory_signal"] = np.maximum.reduce(
        [
            lag365_p90.to_numpy(dtype=float),
            lag365_p95.to_numpy(dtype=float),
            np.clip(spike_strength.to_numpy(dtype=float) - 1.0, 0.0, 1.0),
        ]
    )

    for prob_column in ["prob_top10", "prob_top15"]:
        if prob_column in output.columns:
            raw_prob = pd.to_numeric(output[prob_column], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
            output[prob_column] = raw_prob
            output[f"eff_{prob_column}"] = (raw_prob.to_numpy(dtype=float) * (1.0 + MEMORY_SIGNAL_WEIGHT * output["memory_signal"])).clip(0.0, 1.0)

    return output


def build_validation_scaling_frame(reporter: Reporter, logger: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation_gate, mapping = load_validation_gate_file(VALIDATION_GATE_PATH)
    reporter.emit(
        "Validation column mapping: "
        f"actual={mapping['actual_col']}, base={mapping['base_col']}, "
        f"prob_top10={mapping['prob_top10_col'] or 'missing'}, prob_top15={mapping['prob_top15_col'] or 'missing'}"
    )

    search_df = pd.read_csv(SEARCH_GATE_PATH)
    aggressive_config = choose_aggressive_config(search_df)
    reporter.emit(
        "Using aggressive spike-gate base config: "
        f"label={aggressive_config['label_name']}, mode={aggressive_config['gating_mode']}, "
        f"threshold={aggressive_config['threshold']:.2f}, uplift={aggressive_config['uplift']:.2f}"
    )
    validation_base = reconstruct_aggressive_validation(validation_gate, mapping, aggressive_config)

    train_df = base.load_train_data(gate_mod.TRAIN_DATA_PATH)
    historical_static = gate_mod.build_static_features_historical(train_df[DATE_COL], train_df[DATE_COL].min(), logger)
    training_classifier_table = gate_mod.build_classifier_table(train_df, historical_static)
    train_history = train_df[train_df[DATE_COL] < TRAIN_CUTOFF].set_index(DATE_COL)[TARGET_COL]
    thresholds_bundle = gate_mod.spike2.compute_threshold_bundle(train_history)

    validation_recursive = gate_mod.build_recursive_classifier_features(
        prediction_dates=validation_base[DATE_COL],
        static_features=historical_static,
        initial_revenue_history=train_history,
        recursive_revenue_source=validation_base.set_index(DATE_COL)["base_pred"],
        thresholds_bundle=thresholds_bundle,
        feature_columns=gate_mod.CLASSIFIER_FEATURES,
    )

    validation_frame = validation_base.merge(validation_recursive, on=DATE_COL, how="left", validate="one_to_one")
    validation_frame[TARGET_COL] = validation_frame["actual_Revenue"]
    training_context = training_classifier_table[training_classifier_table[DATE_COL] < TRAIN_CUTOFF].copy()
    campaign_lookup = build_campaign_strength_lookup(training_context)

    train_promo_intensity = (
        pd.to_numeric(training_context["calendar_avg_discount_value"], errors="coerce").fillna(0.0)
        * pd.to_numeric(training_context["calendar_active_promo_count"], errors="coerce").fillna(0.0)
    )
    promo_scale = float(train_promo_intensity.quantile(0.95)) if not train_promo_intensity.empty else 1.0
    validation_context = prepare_scaling_context(validation_frame, promo_scale, campaign_lookup)
    return validation_context, pd.DataFrame([aggressive_config])


def train_full_probability_models(
    train_df: pd.DataFrame,
    historical_static: pd.DataFrame,
    reporter: Reporter,
) -> dict[str, dict[str, Any]]:
    classifier_table = gate_mod.build_classifier_table(train_df, historical_static)
    full_models: dict[str, dict[str, Any]] = {}

    for label_name, quantile in gate_mod.LABEL_SPECS.items():
        threshold = float(train_df[TARGET_COL].quantile(quantile))
        table = classifier_table.copy()
        table["label"] = (pd.to_numeric(table[TARGET_COL], errors="coerce") >= threshold).astype(int)
        clean = table.dropna(subset=gate_mod.CLASSIFIER_FEATURES + ["label"]).reset_index(drop=True)
        X_train = clean[gate_mod.CLASSIFIER_FEATURES].copy()
        y_train = clean["label"].copy()
        feature_medians = X_train.median(numeric_only=True)
        reporter.emit(
            f"Training probability classifier {label_name}: rows={len(X_train):,}, positives={int(y_train.sum()):,}, "
            f"threshold={threshold:,.2f}"
        )
        model, model_type = gate_mod.train_classifier(X_train, y_train, reporter)
        full_models[label_name] = {
            "model_object": model,
            "model_type": model_type,
            "feature_medians": feature_medians,
            "threshold": threshold,
        }
    return full_models


def build_future_scaling_frame(reporter: Reporter, logger: logging.Logger) -> pd.DataFrame:
    train_df = base.load_train_data(gate_mod.TRAIN_DATA_PATH)
    sample_submission = base.load_sample_submission(SAMPLE_SUBMISSION_PATH)
    base_submission = load_base_submission(sample_submission, BASE_SUBMISSION_PATH)

    historical_static = gate_mod.build_static_features_historical(train_df[DATE_COL], train_df[DATE_COL].min(), logger)
    full_models = train_full_probability_models(train_df, historical_static, reporter)
    thresholds_bundle = gate_mod.spike2.compute_threshold_bundle(train_df.set_index(DATE_COL)[TARGET_COL])

    future_static = gate_mod.build_static_features_future(sample_submission[DATE_COL], train_df[DATE_COL].min(), logger)
    future_recursive = gate_mod.build_recursive_classifier_features(
        prediction_dates=sample_submission[DATE_COL],
        static_features=future_static,
        initial_revenue_history=train_df.set_index(DATE_COL)[TARGET_COL],
        recursive_revenue_source=base_submission.set_index(DATE_COL)[TARGET_COL],
        thresholds_bundle=thresholds_bundle,
        feature_columns=gate_mod.CLASSIFIER_FEATURES,
    )

    future_frame = sample_submission[[DATE_COL]].merge(
        future_recursive,
        on=DATE_COL,
        how="left",
        validate="one_to_one",
    )
    raw_future_promo = pd.read_csv(FUTURE_PROMO_FEATURES_PATH, parse_dates=[DATE_COL], low_memory=False)
    raw_future_promo[DATE_COL] = pd.to_datetime(raw_future_promo[DATE_COL], errors="coerce").dt.normalize()
    for source_col, target_col in {
        "future_promo_is_first_7_days": "promo_is_first_7_days",
        "future_promo_is_last_7_days": "promo_is_last_7_days",
    }.items():
        if source_col in raw_future_promo.columns:
            future_frame = future_frame.merge(
                raw_future_promo[[DATE_COL, source_col]].rename(columns={source_col: target_col}),
                on=DATE_COL,
                how="left",
                validate="one_to_one",
            )
        else:
            future_frame[target_col] = 0.0

    for label_name, trained in full_models.items():
        X_future = future_recursive[gate_mod.CLASSIFIER_FEATURES].fillna(trained["feature_medians"]).fillna(0.0)
        probs = gate_mod.predict_classifier_proba(trained["model_object"], trained["model_type"], X_future)
        future_frame[f"prob_{label_name}"] = probs

    training_classifier_table = gate_mod.build_classifier_table(train_df, historical_static)
    training_context = training_classifier_table.copy()
    campaign_lookup = build_campaign_strength_lookup(training_context)
    full_promo_intensity = (
        pd.to_numeric(training_context["calendar_avg_discount_value"], errors="coerce").fillna(0.0)
        * pd.to_numeric(training_context["calendar_active_promo_count"], errors="coerce").fillna(0.0)
    )
    promo_scale = float(full_promo_intensity.quantile(0.95)) if not full_promo_intensity.empty else 1.0
    future_frame = prepare_scaling_context(future_frame, promo_scale, campaign_lookup)
    return future_frame


def compute_scale(frame: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    strategy = str(config["strategy"])
    prob_source = str(config["prob_source"])
    prob = pd.to_numeric(frame[prob_source], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0).to_numpy(dtype=float)
    is_promo = pd.to_numeric(frame["is_promo"], errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
    norm_promo = pd.to_numeric(frame["normalized_promo_intensity"], errors="coerce").fillna(0.0).clip(0.0, 1.5).to_numpy(dtype=float)
    high_risk = pd.to_numeric(frame["is_high_risk_month"], errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)

    if strategy == "soft_linear":
        scale = 1.0 + float(config["alpha"]) * prob
    elif strategy == "soft_quadratic":
        scale = 1.0 + float(config["alpha"]) * (prob**2)
    elif strategy == "threshold":
        scale = np.ones(len(frame), dtype=float)
        low_uplift = float(config["low_uplift"])
        mid_uplift = float(config["mid_uplift"])
        high_uplift = float(config["high_uplift"])
        scale = np.where(prob >= 0.35, 1.0 + low_uplift, scale)
        scale = np.where(prob >= 0.50, 1.0 + mid_uplift, scale)
        scale = np.where(prob >= 0.70, 1.0 + high_uplift, scale)
    elif strategy == "context":
        scale = (
            1.0
            + float(config["alpha"]) * prob
            + float(config["beta"]) * is_promo
            + float(config["gamma"]) * norm_promo
            + float(config["delta"]) * high_risk
        )
    elif strategy == "conservative_gate":
        gate = (
            (prob >= float(config["threshold"]))
            & ((is_promo == 1.0) | (high_risk == 1.0))
        ).astype(float)
        scale = 1.0 + float(config["uplift"]) * gate
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return np.maximum(scale, 1.0)


def compute_spike_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = actual - predicted
    top10_threshold = float(np.quantile(actual, 0.90))
    top5_threshold = float(np.quantile(actual, 0.95))
    top10_mask = actual >= top10_threshold
    top5_mask = actual >= top5_threshold
    non_spike_mask = actual < top10_threshold

    def masked_rmse(mask: np.ndarray) -> float:
        return float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan

    return {
        "top10_RMSE": masked_rmse(top10_mask),
        "top10_underprediction": int(np.sum(error[top10_mask] > 0)) if top10_mask.any() else 0,
        "top10_count": int(np.sum(top10_mask)),
        "top5_RMSE": masked_rmse(top5_mask),
        "top5_underprediction": int(np.sum(error[top5_mask] > 0)) if top5_mask.any() else 0,
        "top5_count": int(np.sum(top5_mask)),
        "non_spike_RMSE": masked_rmse(non_spike_mask),
    }


def evaluate_config(
    validation_frame: pd.DataFrame,
    config: dict[str, Any],
    base_metrics: dict[str, float],
) -> dict[str, Any]:
    scale = compute_scale(validation_frame, config)
    base_pred = validation_frame["base_pred"].to_numpy(dtype=float)
    actual = validation_frame["actual_Revenue"].to_numpy(dtype=float)
    scaled_pred = base_pred * scale

    overall = base.evaluate_predictions(pd.Series(actual), scaled_pred)
    spike_metrics = compute_spike_metrics(actual, scaled_pred)

    non_spike_ok = spike_metrics["non_spike_RMSE"] <= base_metrics["non_spike_RMSE"] * NON_SPIKE_GUARD
    overall_ok = overall["RMSE"] <= base_metrics["RMSE"] * OVERALL_GUARD
    accepted = int(non_spike_ok and overall_ok)

    return {
        **config,
        "mean_scale": float(np.mean(scale)),
        "min_scale": float(np.min(scale)),
        "max_scale": float(np.max(scale)),
        "MAE": overall["MAE"],
        "RMSE": overall["RMSE"],
        "R2": overall["R2"],
        **spike_metrics,
        "improves_rmse": int(overall["RMSE"] < base_metrics["RMSE"]),
        "improves_top10_rmse": int(spike_metrics["top10_RMSE"] < base_metrics["top10_RMSE"]),
        "accepted": accepted,
    }


def build_search_space(prob_sources: list[str]) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []

    for prob_source in prob_sources:
        for alpha in ALPHAS:
            configs.append({"strategy": "soft_linear", "prob_source": prob_source, "alpha": alpha})
            configs.append({"strategy": "soft_quadratic", "prob_source": prob_source, "alpha": alpha})

        for low_uplift in LOW_UPLIFTS:
            for mid_uplift in MID_UPLIFTS:
                for high_uplift in HIGH_UPLIFTS:
                    if not (low_uplift < mid_uplift < high_uplift):
                        continue
                    configs.append(
                        {
                            "strategy": "threshold",
                            "prob_source": prob_source,
                            "low_uplift": low_uplift,
                            "mid_uplift": mid_uplift,
                            "high_uplift": high_uplift,
                        }
                    )

        for alpha in CONTEXT_ALPHAS:
            for beta in CONTEXT_BETAS:
                for gamma in CONTEXT_GAMMAS:
                    for delta in CONTEXT_DELTAS:
                        configs.append(
                            {
                                "strategy": "context",
                                "prob_source": prob_source,
                                "alpha": alpha,
                                "beta": beta,
                                "gamma": gamma,
                                "delta": delta,
                            }
                        )

        for threshold in CONSERVATIVE_THRESHOLDS:
            for uplift in CONSERVATIVE_UPLIFTS:
                configs.append(
                    {
                        "strategy": "conservative_gate",
                        "prob_source": prob_source,
                        "threshold": threshold,
                        "uplift": uplift,
                    }
                )

    return configs


def select_best_rmse_config(search_results: pd.DataFrame) -> dict[str, Any]:
    accepted = search_results[search_results["accepted"] == 1].copy()
    pool = accepted if not accepted.empty else search_results.copy()
    ordered = pool.sort_values(["RMSE", "top10_RMSE", "top10_underprediction", "MAE"])
    return ordered.iloc[0].to_dict()


def select_best_spike_config(search_results: pd.DataFrame, base_metrics: dict[str, float]) -> dict[str, Any]:
    pool = search_results[
        (search_results["non_spike_RMSE"] <= base_metrics["non_spike_RMSE"] * NON_SPIKE_GUARD)
        & (search_results["RMSE"] <= base_metrics["RMSE"] * 1.01)
    ].copy()
    if pool.empty:
        pool = search_results.copy()
    ordered = pool.sort_values(["top10_RMSE", "top10_underprediction", "RMSE", "MAE"])
    return ordered.iloc[0].to_dict()


def select_conservative_config(search_results: pd.DataFrame) -> dict[str, Any]:
    accepted = search_results[search_results["accepted"] == 1].copy()
    pool = accepted if not accepted.empty else search_results.copy()
    pool["mean_scale_distance"] = (pool["mean_scale"] - 1.0).abs()
    ordered = pool.sort_values(["mean_scale_distance", "RMSE", "top10_RMSE", "MAE"])
    return ordered.iloc[0].to_dict()


def choose_neighbor_configs(search_results: pd.DataFrame, best_config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    same_family = search_results[
        (search_results["strategy"] == best_config["strategy"])
        & (search_results["prob_source"] == best_config["prob_source"])
    ].copy()
    if same_family.empty:
        return best_config, best_config

    same_family = same_family.sort_values(["mean_scale", "RMSE", "top10_RMSE"]).reset_index(drop=True)
    match_mask = np.isclose(same_family["mean_scale"], float(best_config["mean_scale"])) & np.isclose(
        same_family["RMSE"],
        float(best_config["RMSE"]),
    )
    if not match_mask.any():
        return best_config, best_config

    idx = int(np.where(match_mask)[0][0])
    lower = same_family.iloc[max(0, idx - 1)].to_dict()
    higher = same_family.iloc[min(len(same_family) - 1, idx + 1)].to_dict()
    return lower, higher


def validate_submission_frame(output: pd.DataFrame, sample_submission: pd.DataFrame) -> None:
    if list(output.columns) != [DATE_COL, TARGET_COL, COGS_COL]:
        raise ValueError("Submission columns must be exactly Date, Revenue, COGS")
    if len(output) != len(sample_submission):
        raise ValueError("Submission row count does not match sample submission")
    if not output[DATE_COL].equals(sample_submission[DATE_COL]):
        raise ValueError("Submission Date order does not match sample submission")
    if output[[TARGET_COL, COGS_COL]].isna().any().any():
        raise ValueError("Submission contains missing values")
    if (output[[TARGET_COL, COGS_COL]] < 0).any().any():
        raise ValueError("Submission contains negative Revenue or COGS")


def build_scaled_submission(
    sample_submission: pd.DataFrame,
    base_submission: pd.DataFrame,
    future_frame: pd.DataFrame,
    config: dict[str, Any],
    path: Path,
) -> pd.DataFrame:
    scale = compute_scale(future_frame, config)
    output = sample_submission[[DATE_COL]].copy()
    output[TARGET_COL] = base_submission[TARGET_COL].to_numpy(dtype=float) * scale
    output[COGS_COL] = base_submission[COGS_COL].to_numpy(dtype=float) * scale
    output[TARGET_COL] = output[TARGET_COL].clip(lower=0.0)
    output[COGS_COL] = output[COGS_COL].clip(lower=0.0)
    validate_submission_frame(output, sample_submission)
    output.to_csv(path, index=False)
    return output


def summarize_metric_change(before: float, after: float) -> float:
    return float(after - before)


def run() -> None:
    logger = setup_logging()
    reporter = Reporter(logger)

    reporter.emit("Adaptive Scaling Layer")
    reporter.emit("======================")
    reporter.emit("")

    reporter.emit("1. Load current best base submission and rebuild validation base")
    sample_submission = load_sample_submission(SAMPLE_SUBMISSION_PATH)
    base_submission = load_base_submission(sample_submission, BASE_SUBMISSION_PATH)
    validation_frame, aggressive_meta = build_validation_scaling_frame(reporter, logger)
    del aggressive_meta

    base_metrics = {
        **base.evaluate_predictions(validation_frame["actual_Revenue"], validation_frame["base_pred"].to_numpy(dtype=float)),
        **compute_spike_metrics(
            validation_frame["actual_Revenue"].to_numpy(dtype=float),
            validation_frame["base_pred"].to_numpy(dtype=float),
        ),
    }
    reporter.emit(
        f"Validation base metrics: MAE={base_metrics['MAE']:,.2f} | RMSE={base_metrics['RMSE']:,.2f} | "
        f"R2={base_metrics['R2']:.6f}"
    )
    reporter.emit(
        f"Base top10 RMSE={base_metrics['top10_RMSE']:,.2f} | "
        f"underprediction={base_metrics['top10_underprediction']}/{base_metrics['top10_count']} | "
        f"non-spike RMSE={base_metrics['non_spike_RMSE']:,.2f}"
    )

    prob_sources = [column for column in ["prob_top10", "prob_top15", "eff_prob_top10", "eff_prob_top15"] if column in validation_frame.columns]
    if not prob_sources:
        raise RuntimeError("No spike probability columns available for adaptive scaling search.")
    reporter.emit(f"Available probability sources: {', '.join(prob_sources)}")

    reporter.emit("")
    reporter.emit("2. Search adaptive scaling strategies on validation 2022")
    search_configs = build_search_space(prob_sources)
    results = [evaluate_config(validation_frame, config, base_metrics) for config in search_configs]
    search_results = pd.DataFrame(results).sort_values(
        ["accepted", "RMSE", "top10_RMSE", "top10_underprediction", "non_spike_RMSE", "MAE"],
        ascending=[False, True, True, True, True, True],
    ).reset_index(drop=True)
    search_results.to_csv(SEARCH_RESULTS_PATH, index=False)
    reporter.emit_frame("Top 15 scaling configs:", search_results.head(15))

    reporter.emit("")
    reporter.emit("3. Select best / spike-focused / conservative configs")
    best_config = select_best_rmse_config(search_results)
    spike_config = select_best_spike_config(search_results, base_metrics)
    conservative_config = select_conservative_config(search_results)
    minus_config, plus_config = choose_neighbor_configs(search_results, best_config)

    reporter.emit_frame(
        "Selected configs:",
        pd.DataFrame([best_config, spike_config, conservative_config, minus_config, plus_config])[
            ["strategy", "prob_source", "mean_scale", "RMSE", "top10_RMSE", "non_spike_RMSE", "accepted"]
            + [column for column in ["alpha", "beta", "gamma", "delta", "threshold", "uplift", "low_uplift", "mid_uplift", "high_uplift"] if column in search_results.columns]
        ].drop_duplicates()
    )

    reporter.emit("")
    reporter.emit("4. Recompute future spike probabilities and build future context")
    future_frame = build_future_scaling_frame(reporter, logger)

    reporter.emit("")
    reporter.emit("5. Apply selected scaling rules to current best submission")
    best_submission = build_scaled_submission(sample_submission, base_submission, future_frame, best_config, SUBMISSION_BEST_PATH)
    spike_submission = build_scaled_submission(sample_submission, base_submission, future_frame, spike_config, SUBMISSION_SPIKE_PATH)
    conservative_submission = build_scaled_submission(sample_submission, base_submission, future_frame, conservative_config, SUBMISSION_CONSERVATIVE_PATH)
    plus_submission = build_scaled_submission(sample_submission, base_submission, future_frame, plus_config, SUBMISSION_PLUS_PATH)
    minus_submission = build_scaled_submission(sample_submission, base_submission, future_frame, minus_config, SUBMISSION_MINUS_PATH)
    del best_submission, spike_submission, conservative_submission, plus_submission, minus_submission

    reporter.emit("")
    reporter.emit("6. Final summary")
    best_scale_future = compute_scale(future_frame, best_config)
    best_after = evaluate_config(validation_frame, best_config, base_metrics)
    spike_after = evaluate_config(validation_frame, spike_config, base_metrics)
    conservative_after = evaluate_config(validation_frame, conservative_config, base_metrics)

    reporter.emit(f"Best strategy: {best_config['strategy']}")
    reporter.emit(
        "Best parameters: "
        + ", ".join(
            [
                f"{key}={best_config[key]}"
                for key in ["prob_source", "alpha", "beta", "gamma", "delta", "threshold", "uplift", "low_uplift", "mid_uplift", "high_uplift"]
                if key in best_config and pd.notna(best_config[key])
            ]
        )
    )
    reporter.emit(
        f"Validation RMSE before/after: {base_metrics['RMSE']:,.2f} -> {best_after['RMSE']:,.2f}"
    )
    reporter.emit(
        f"Top 10% RMSE before/after: {base_metrics['top10_RMSE']:,.2f} -> {best_after['top10_RMSE']:,.2f}"
    )
    reporter.emit(
        f"Non-spike RMSE before/after: {base_metrics['non_spike_RMSE']:,.2f} -> {best_after['non_spike_RMSE']:,.2f}"
    )
    reporter.emit(
        f"Mean future scale factor: {np.mean(best_scale_future):.6f}"
    )
    reporter.emit(
        f"Min/Max future scale factor: {np.min(best_scale_future):.6f} / {np.max(best_scale_future):.6f}"
    )
    reporter.emit_frame(
        "Best vs spike-focused vs conservative validation metrics:",
        pd.DataFrame([best_after, spike_after, conservative_after])[
            ["strategy", "prob_source", "RMSE", "MAE", "R2", "top10_RMSE", "top10_underprediction", "top5_RMSE", "non_spike_RMSE", "mean_scale", "min_scale", "max_scale"]
        ],
    )
    reporter.emit(
        "Created submission files: "
        "submission_adaptive_scale_best.csv, submission_adaptive_scale_spike.csv, "
        "submission_adaptive_scale_conservative.csv, submission_adaptive_scale_plus.csv, "
        "submission_adaptive_scale_minus.csv"
    )
    upload_order = [
        SUBMISSION_BEST_PATH.name,
        SUBMISSION_CONSERVATIVE_PATH.name,
        SUBMISSION_PLUS_PATH.name,
        SUBMISSION_MINUS_PATH.name,
        SUBMISSION_SPIKE_PATH.name,
    ]
    reporter.emit(f"Recommended upload order: {upload_order}")
    reporter.emit(
        "Leakage confirmation: this layer only rescales existing predictions using spike probabilities recomputed from the spike-gate classifier pipeline, future-known promo schedule/context, and recursive lagged revenue features built from historical actual revenue plus existing submission predictions. No future actual Revenue/COGS or same-day realized demand was used."
    )

    reporter.save(REPORT_PATH)


if __name__ == "__main__":
    run()
