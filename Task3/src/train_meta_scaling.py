from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import adaptive_scaling_layer as scale_mod
import train_final_model as base
import train_spike_probability_gate as gate_mod


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "log"

VALIDATION_GATE_PATH = DATA_DIR / "spike_gate_validation_predictions.csv"
ADAPTIVE_PLUS_SUBMISSION_PATH = DATA_DIR / "submission_adaptive_scale_plus.csv"
AGGRESSIVE_SUBMISSION_PATH = DATA_DIR / "submission_spike_gate_aggressive.csv"
FUTURE_PROMO_FEATURES_PATH = DATA_DIR / "future_promo_calendar_features.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"

PRUNED_VALIDATION_PATH = DATA_DIR / "pruned_ensemble_validation_predictions.csv"
SPIKE_VALIDATION_PATH = DATA_DIR / "spike_model_validation_predictions.csv"
REGIME_VALIDATION_PATH = DATA_DIR / "promo_regime_validation_predictions.csv"
PRUNED_SUBMISSION_PATH = DATA_DIR / "submission_pruned_ensemble.csv"
SPIKE_SUBMISSION_PATH = DATA_DIR / "submission_spike_aware.csv"
REGIME_SUBMISSION_PATH = DATA_DIR / "submission_promo_regime.csv"

VALIDATION_OUTPUT_PATH = DATA_DIR / "meta_scaling_validation_predictions.csv"
COMPARISON_PATH = DATA_DIR / "meta_scaling_model_comparison.csv"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "meta_scaling_feature_importance.csv"

SUBMISSION_BEST_PATH = DATA_DIR / "submission_meta_scale_best.csv"
SUBMISSION_CONSERVATIVE_PATH = DATA_DIR / "submission_meta_scale_conservative.csv"
SUBMISSION_AGGRESSIVE_PATH = DATA_DIR / "submission_meta_scale_aggressive.csv"

REPORT_PATH = LOG_DIR / "meta_scaling_report.txt"
LOG_FILE = LOG_DIR / "train_meta_scaling.log"

DATE_COL = base.DATE_COL
TARGET_COL = base.TARGET_COL
COGS_COL = base.COGS_COL
TRAIN_CUTOFF = base.TRAIN_CUTOFF
RANDOM_STATE = base.RANDOM_STATE

SCALE_TARGET_MIN = 0.75
SCALE_TARGET_MAX = 1.35
CLIP_RANGES = [(0.97, 1.12), (0.95, 1.15), (0.93, 1.18)]
CONSERVATIVE_CLIP = (0.98, 1.10)
AGGRESSIVE_CLIP = (0.93, 1.18)
RIDGE_ALPHA = 10.0
META_FIRST_PREDICTION_DATE = pd.Timestamp("2022-04-01")


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

    logger = logging.getLogger("train_meta_scaling")
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


def load_submission(path: Path, sample_submission: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Submission not found: {path}")
    frame = pd.read_csv(path, parse_dates=[DATE_COL], low_memory=False)
    frame[DATE_COL] = pd.to_datetime(frame[DATE_COL], errors="coerce").dt.normalize()
    validate_submission_frame(frame[[DATE_COL, TARGET_COL, COGS_COL]], sample_submission)
    return frame[[DATE_COL, TARGET_COL, COGS_COL]].copy()


def load_validation_component(path: Path, pred_name: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=[DATE_COL, pred_name])
    frame = pd.read_csv(path, parse_dates=[DATE_COL], low_memory=False)
    frame[DATE_COL] = pd.to_datetime(frame[DATE_COL], errors="coerce").dt.normalize()
    if "predicted_Revenue" not in frame.columns:
        return pd.DataFrame(columns=[DATE_COL, pred_name])
    return frame[[DATE_COL, "predicted_Revenue"]].rename(columns={"predicted_Revenue": pred_name})


def clip_scale_target(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").clip(lower=SCALE_TARGET_MIN, upper=SCALE_TARGET_MAX)


def compute_spike_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = actual - predicted
    top10_threshold = float(np.quantile(actual, 0.90))
    non_spike_mask = actual < top10_threshold
    top10_mask = actual >= top10_threshold

    def masked_rmse(mask: np.ndarray) -> float:
        return float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan

    return {
        "top10_RMSE": masked_rmse(top10_mask),
        "top10_underprediction": int(np.sum(error[top10_mask] > 0)) if top10_mask.any() else 0,
        "top10_count": int(np.sum(top10_mask)),
        "non_spike_RMSE": masked_rmse(non_spike_mask),
    }


def evaluate_meta_predictions(actual: pd.Series, base_pred: np.ndarray, predicted_scale: np.ndarray) -> dict[str, float]:
    final_pred = np.asarray(base_pred, dtype=float) * np.asarray(predicted_scale, dtype=float)
    overall = base.evaluate_predictions(actual, final_pred)
    spike = compute_spike_metrics(actual.to_numpy(dtype=float), final_pred)
    return {"final_pred": final_pred, **overall, **spike}


def build_validation_contexts(reporter: Reporter, logger: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggressive_context, _ = scale_mod.build_validation_scaling_frame(reporter, logger)

    search_df = pd.read_csv(scale_mod.SEARCH_RESULTS_PATH)
    best_cfg = scale_mod.select_best_rmse_config(search_df)
    _, plus_cfg = scale_mod.choose_neighbor_configs(search_df, best_cfg)
    plus_scale = scale_mod.compute_scale(aggressive_context, plus_cfg)
    plus_pred = aggressive_context["base_pred"].to_numpy(dtype=float) * plus_scale

    plus_context = aggressive_context.copy()
    plus_context["base_pred"] = plus_pred

    pruned_val = load_validation_component(PRUNED_VALIDATION_PATH, "pruned_pred")
    spike_val = load_validation_component(SPIKE_VALIDATION_PATH, "spike_pred")
    regime_val = load_validation_component(REGIME_VALIDATION_PATH, "regime_pred")
    components = pruned_val.merge(spike_val, on=DATE_COL, how="outer").merge(regime_val, on=DATE_COL, how="outer")

    aggressive_context = aggressive_context.merge(components, on=DATE_COL, how="left")
    plus_context = plus_context.merge(components, on=DATE_COL, how="left")
    return aggressive_context.sort_values(DATE_COL).reset_index(drop=True), plus_context.sort_values(DATE_COL).reset_index(drop=True)


def train_full_probability_models(
    train_df: pd.DataFrame,
    historical_static: pd.DataFrame,
    reporter: Reporter,
) -> dict[str, dict[str, Any]]:
    classifier_table = gate_mod.build_classifier_table(train_df, historical_static)
    models: dict[str, dict[str, Any]] = {}
    for label_name, quantile in gate_mod.LABEL_SPECS.items():
        threshold = float(train_df[TARGET_COL].quantile(quantile))
        table = classifier_table.copy()
        table["label"] = (pd.to_numeric(table[TARGET_COL], errors="coerce") >= threshold).astype(int)
        clean = table.dropna(subset=gate_mod.CLASSIFIER_FEATURES + ["label"]).reset_index(drop=True)
        X_train = clean[gate_mod.CLASSIFIER_FEATURES].copy()
        y_train = clean["label"].copy()
        feature_medians = X_train.median(numeric_only=True)
        reporter.emit(
            f"Training full probability model {label_name}: rows={len(X_train):,}, positives={int(y_train.sum()):,}"
        )
        model, model_type = gate_mod.train_classifier(X_train, y_train, reporter)
        models[label_name] = {
            "model_object": model,
            "model_type": model_type,
            "feature_medians": feature_medians,
        }
    return models


def build_future_meta_context(base_submission_path: Path, reporter: Reporter, logger: logging.Logger) -> pd.DataFrame:
    train_df = base.load_train_data(gate_mod.TRAIN_DATA_PATH)
    sample_submission = base.load_sample_submission(SAMPLE_SUBMISSION_PATH)
    base_submission = load_submission(base_submission_path, sample_submission)

    historical_static = gate_mod.build_static_features_historical(train_df[DATE_COL], train_df[DATE_COL].min(), logger)
    future_static = gate_mod.build_static_features_future(sample_submission[DATE_COL], train_df[DATE_COL].min(), logger)
    full_models = train_full_probability_models(train_df, historical_static, reporter)
    thresholds_bundle = gate_mod.spike2.compute_threshold_bundle(train_df.set_index(DATE_COL)[TARGET_COL])

    future_recursive = gate_mod.build_recursive_classifier_features(
        prediction_dates=sample_submission[DATE_COL],
        static_features=future_static,
        initial_revenue_history=train_df.set_index(DATE_COL)[TARGET_COL],
        recursive_revenue_source=base_submission.set_index(DATE_COL)[TARGET_COL],
        thresholds_bundle=thresholds_bundle,
        feature_columns=gate_mod.CLASSIFIER_FEATURES,
    )

    future_frame = sample_submission[[DATE_COL]].merge(future_recursive, on=DATE_COL, how="left", validate="one_to_one")
    future_frame["base_pred"] = base_submission[TARGET_COL].to_numpy(dtype=float)

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
        future_frame[f"prob_{label_name}"] = gate_mod.predict_classifier_proba(
            trained["model_object"],
            trained["model_type"],
            X_future,
        )

    sample_submission = base.load_sample_submission(SAMPLE_SUBMISSION_PATH)
    component_map = [
        (PRUNED_SUBMISSION_PATH, "pruned_pred"),
        (SPIKE_SUBMISSION_PATH, "spike_pred"),
        (REGIME_SUBMISSION_PATH, "regime_pred"),
    ]
    for path, column in component_map:
        if path.exists():
            component = load_submission(path, sample_submission)[[DATE_COL, TARGET_COL]].rename(columns={TARGET_COL: column})
            future_frame = future_frame.merge(component, on=DATE_COL, how="left", validate="one_to_one")

    return future_frame.sort_values(DATE_COL).reset_index(drop=True)


def add_base_prediction_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["base_pred"] = pd.to_numeric(output["base_pred"], errors="coerce").fillna(0.0)
    output["log_base_pred"] = np.log1p(output["base_pred"].clip(lower=0.0))
    output["base_pred_rank_pct"] = output["base_pred"].rank(method="average", pct=True).fillna(0.0)
    output["promo_intensity"] = (
        pd.to_numeric(output["calendar_avg_discount_value"], errors="coerce").fillna(0.0)
        * pd.to_numeric(output["calendar_active_promo_count"], errors="coerce").fillna(0.0)
    )
    output["is_high_risk_month"] = pd.to_numeric(output["month"], errors="coerce").fillna(0).astype(int).isin(scale_mod.HIGH_RISK_MONTHS).astype(int)

    if "prob_top10" in output.columns:
        output["spike_prob"] = pd.to_numeric(output["prob_top10"], errors="coerce").fillna(0.0)
    elif "prob_top15" in output.columns:
        output["spike_prob"] = pd.to_numeric(output["prob_top15"], errors="coerce").fillna(0.0)
    else:
        output["spike_prob"] = 0.0

    if "promo_is_first_7_days" not in output.columns:
        promo_day = (
            pd.to_numeric(output["promo_progress_ratio"], errors="coerce").fillna(0.0)
            * pd.to_numeric(output["promo_duration"], errors="coerce").fillna(0.0)
        )
        output["promo_is_first_7_days"] = ((promo_day > 0) & (promo_day <= 7)).astype(int)
    if "promo_is_last_7_days" not in output.columns:
        days_remaining = pd.to_numeric(output["promo_days_remaining"], errors="coerce").fillna(9999.0)
        is_promo = pd.to_numeric(output["calendar_any_promo"], errors="coerce").fillna(0.0) > 0
        output["promo_is_last_7_days"] = ((days_remaining >= 0) & (days_remaining <= 6) & is_promo).astype(int)

    if "spike_pred" in output.columns and "pruned_pred" in output.columns:
        output["spike_minus_pruned"] = (
            pd.to_numeric(output["spike_pred"], errors="coerce").fillna(0.0)
            - pd.to_numeric(output["pruned_pred"], errors="coerce").fillna(0.0)
        )
        output["base_over_spike"] = scale_mod.safe_divide(output["base_pred"], pd.to_numeric(output["spike_pred"], errors="coerce").fillna(0.0) + 1e-6)
    return output


def build_meta_feature_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feature_candidates = [
        "base_pred",
        "log_base_pred",
        "base_pred_rank_pct",
        "spike_prob",
        "prob_top10",
        "prob_top15",
        "month",
        "day_of_year",
        "day_of_week",
        "is_weekend",
        "is_high_risk_month",
        "calendar_any_promo",
        "calendar_avg_discount_value",
        "calendar_active_promo_count",
        "promo_intensity",
        "promo_progress_ratio",
        "promo_days_remaining",
        "promotion_campaign_index",
        "promo_is_first_7_days",
        "promo_is_last_7_days",
        "revenue_lag_365",
        "lag7_to_roll30_ratio",
        "lag30_to_roll90_ratio",
        "spike_strength_365",
        "volatility_30",
        "volatility_90",
        "pruned_pred",
        "spike_pred",
        "regime_pred",
        "spike_minus_pruned",
        "base_over_spike",
    ] + gate_mod.CAMPAIGN_FLAG_COLUMNS

    feature_columns = [column for column in feature_candidates if column in frame.columns]
    X = frame[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X, feature_columns


def fit_ridge_regression(X: pd.DataFrame, y: pd.Series, alpha: float = RIDGE_ALPHA) -> dict[str, Any]:
    X_values = X.to_numpy(dtype=float)
    y_values = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)

    means = X_values.mean(axis=0)
    stds = X_values.std(axis=0, ddof=0)
    stds[stds == 0] = 1.0
    X_scaled = (X_values - means) / stds

    X_design = np.column_stack([np.ones(len(X_scaled)), X_scaled])
    penalty = np.eye(X_design.shape[1], dtype=float)
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(X_design.T @ X_design + alpha * penalty, X_design.T @ y_values)

    return {
        "model_name": "ridge_regression",
        "model_type": "ridge",
        "coef": beta[1:],
        "intercept": float(beta[0]),
        "means": means,
        "stds": stds,
        "feature_columns": X.columns.tolist(),
    }


def predict_ridge(model: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    columns = model["feature_columns"]
    X_values = X[columns].to_numpy(dtype=float)
    X_scaled = (X_values - model["means"]) / model["stds"]
    return model["intercept"] + X_scaled @ model["coef"]


def fit_huber_regression(X: pd.DataFrame, y: pd.Series) -> dict[str, Any] | None:
    try:
        from sklearn.linear_model import HuberRegressor
    except Exception:
        return None

    model = HuberRegressor(alpha=0.0005, epsilon=1.35, max_iter=300)
    model.fit(X, y)
    return {
        "model_name": "huber_regressor",
        "model_type": "huber",
        "model_object": model,
        "feature_columns": X.columns.tolist(),
    }


def fit_lightgbm_meta(X: pd.DataFrame, y: pd.Series) -> dict[str, Any] | None:
    if not base.lightgbm_available():
        return None
    import lightgbm as lgb

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.03,
        "max_depth": 3,
        "num_leaves": 8,
        "min_data_in_leaf": 40,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": RANDOM_STATE,
        "verbosity": -1,
        "force_col_wise": True,
    }
    dataset = lgb.Dataset(X, label=y, feature_name=X.columns.tolist(), free_raw_data=False)
    model = lgb.train(params=params, train_set=dataset, num_boost_round=250)
    return {
        "model_name": "lightgbm_shallow",
        "model_type": "lightgbm",
        "model_object": model,
        "feature_columns": X.columns.tolist(),
    }


def predict_meta_model(model_info: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    model_type = model_info["model_type"]
    if model_type == "ridge":
        return np.asarray(predict_ridge(model_info, X), dtype=float)
    if model_type == "huber":
        return np.asarray(model_info["model_object"].predict(X[model_info["feature_columns"]]), dtype=float)
    if model_type == "lightgbm":
        return np.asarray(model_info["model_object"].predict(X[model_info["feature_columns"]]), dtype=float)
    raise ValueError(f"Unknown meta model type: {model_type}")


def get_meta_importance(model_info: dict[str, Any]) -> pd.DataFrame:
    if model_info["model_type"] == "ridge":
        importance = np.abs(np.asarray(model_info["coef"], dtype=float))
        return (
            pd.DataFrame(
                {
                    "feature": model_info["feature_columns"],
                    "importance_gain": importance,
                    "importance_split": np.nan,
                }
            )
            .sort_values("importance_gain", ascending=False)
            .reset_index(drop=True)
        )
    if model_info["model_type"] == "huber":
        importance = np.abs(np.asarray(model_info["model_object"].coef_, dtype=float))
        return (
            pd.DataFrame(
                {
                    "feature": model_info["feature_columns"],
                    "importance_gain": importance,
                    "importance_split": np.nan,
                }
            )
            .sort_values("importance_gain", ascending=False)
            .reset_index(drop=True)
        )
    if model_info["model_type"] == "lightgbm":
        model = model_info["model_object"]
        return (
            pd.DataFrame(
                {
                    "feature": model_info["feature_columns"],
                    "importance_gain": model.feature_importance(importance_type="gain"),
                    "importance_split": model.feature_importance(importance_type="split"),
                }
            )
            .sort_values(["importance_gain", "importance_split"], ascending=False)
            .reset_index(drop=True)
        )
    return pd.DataFrame(columns=["feature", "importance_gain", "importance_split"])


def build_clip_key(clip_range: tuple[float, float]) -> str:
    return f"[{clip_range[0]:.2f},{clip_range[1]:.2f}]"


def select_best_meta_config(comparison: pd.DataFrame) -> dict[str, Any]:
    plus_rows = comparison[comparison["base_context"] == "adaptive_plus"].copy()
    accepted = plus_rows[
        (plus_rows["non_spike_RMSE"] <= plus_rows["base_non_spike_RMSE"] * 1.01)
        & (plus_rows["RMSE"] <= plus_rows["base_RMSE"] * 1.01)
    ]
    pool = accepted if not accepted.empty else plus_rows
    ordered = pool.sort_values(["RMSE", "top10_RMSE", "top10_underprediction", "MAE"])
    return ordered.iloc[0].to_dict()


def discover_meta_model_names() -> list[str]:
    model_names = ["ridge_regression"]
    try:
        from sklearn.linear_model import HuberRegressor  # noqa: F401

        model_names.append("huber_regressor")
    except Exception:
        pass
    if base.lightgbm_available():
        model_names.append("lightgbm_shallow")
    return model_names


def fit_meta_model_by_name(model_name: str, X: pd.DataFrame, y: pd.Series) -> dict[str, Any] | None:
    if model_name == "ridge_regression":
        return fit_ridge_regression(X, y, alpha=RIDGE_ALPHA)
    if model_name == "huber_regressor":
        return fit_huber_regression(X, y)
    if model_name == "lightgbm_shallow":
        return fit_lightgbm_meta(X, y)
    raise ValueError(f"Unknown meta model name: {model_name}")


def build_meta_month_folds(dates: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    month_starts = pd.date_range(
        dates.min().replace(day=1),
        dates.max().replace(day=1),
        freq="MS",
    )
    folds: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for month_start in month_starts:
        if month_start < META_FIRST_PREDICTION_DATE:
            continue
        month_end = month_start + pd.offsets.MonthBegin(1)
        folds.append((pd.Timestamp(month_start), pd.Timestamp(month_end)))
    return folds


def generate_time_safe_meta_predictions(
    context: pd.DataFrame,
    target: pd.Series,
    model_names: list[str],
    reporter: Reporter,
    context_name: str,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    dates = pd.to_datetime(context[DATE_COL], errors="coerce").dt.normalize()
    X_all, feature_columns = build_meta_feature_matrix(context)
    folds = build_meta_month_folds(dates)
    reporter.emit(
        f"Time-safe meta folds for {context_name}: "
        f"{[(start.strftime('%Y-%m-%d'), (end - pd.Timedelta(days=1)).strftime('%Y-%m-%d')) for start, end in folds]}"
    )

    predictions: dict[str, np.ndarray] = {}
    final_models: dict[str, dict[str, Any]] = {}

    for model_name in model_names:
        raw_scale = np.ones(len(context), dtype=float)
        model_available = True
        for fold_start, fold_end in folds:
            train_mask = dates < fold_start
            pred_mask = (dates >= fold_start) & (dates < fold_end)
            if not pred_mask.any():
                continue

            X_train = X_all.loc[train_mask, feature_columns].copy()
            y_train = target.loc[train_mask].copy()
            train_valid = ~(X_train.isna().all(axis=1) | y_train.isna())
            X_train = X_train.loc[train_valid].fillna(0.0)
            y_train = y_train.loc[train_valid]

            if len(X_train) < 45:
                continue

            model_info = fit_meta_model_by_name(model_name, X_train, y_train)
            if model_info is None:
                model_available = False
                break

            X_pred = X_all.loc[pred_mask, model_info["feature_columns"]].fillna(0.0)
            raw_scale[pred_mask.to_numpy()] = predict_meta_model(model_info, X_pred)

        if not model_available:
            reporter.emit(f"{model_name} unavailable during time-safe calibration; skipping")
            continue

        full_model = fit_meta_model_by_name(model_name, X_all[feature_columns].fillna(0.0), target.fillna(1.0))
        if full_model is None:
            reporter.emit(f"{model_name} unavailable during full fit; skipping")
            continue

        predictions[model_name] = raw_scale
        final_models[model_name] = full_model

    return predictions, final_models


def run() -> None:
    logger = setup_logging()
    reporter = Reporter(logger)

    reporter.emit("Meta Scaling Layer")
    reporter.emit("==================")
    reporter.emit("")

    reporter.emit("1. Build validation meta-training data")
    aggressive_context, plus_context = build_validation_contexts(reporter, logger)
    aggressive_context = add_base_prediction_features(aggressive_context)
    plus_context = add_base_prediction_features(plus_context)

    aggressive_scale_target = clip_scale_target(
        aggressive_context["actual_Revenue"] / aggressive_context["base_pred"].replace(0, np.nan)
    ).fillna(1.0)
    plus_scale_target = clip_scale_target(
        plus_context["actual_Revenue"] / plus_context["base_pred"].replace(0, np.nan)
    ).fillna(1.0)
    _, feature_columns = build_meta_feature_matrix(plus_context)
    reporter.emit(f"Meta feature count: {len(feature_columns)}")
    reporter.emit(
        f"Aggressive scale target clipped range: {aggressive_scale_target.min():.4f} -> "
        f"{aggressive_scale_target.max():.4f}"
    )
    reporter.emit(
        f"Adaptive-plus scale target clipped range: {plus_scale_target.min():.4f} -> "
        f"{plus_scale_target.max():.4f}"
    )
    reporter.emit(
        f"Time-safe calibration setup: train on prior observed 2022 months, first adjusted month = "
        f"{META_FIRST_PREDICTION_DATE.date()}"
    )

    reporter.emit("")
    reporter.emit("2. Train small meta models")
    model_names = discover_meta_model_names()
    reporter.emit(f"Candidate meta models: {model_names}")
    aggressive_raw_predictions, aggressive_full_models = generate_time_safe_meta_predictions(
        aggressive_context,
        aggressive_scale_target,
        model_names,
        reporter,
        context_name="spike_gate_aggressive",
    )
    plus_raw_predictions, plus_full_models = generate_time_safe_meta_predictions(
        plus_context,
        plus_scale_target,
        model_names,
        reporter,
        context_name="adaptive_plus",
    )
    available_model_names = sorted(set(aggressive_raw_predictions) & set(plus_raw_predictions) & set(plus_full_models))
    reporter.emit(f"Available time-safe meta models: {available_model_names}")

    reporter.emit("")
    reporter.emit("3. Evaluate meta models on validation clip ranges")
    comparison_rows: list[dict[str, Any]] = []
    aggressive_actual = aggressive_context["actual_Revenue"]
    plus_actual = plus_context["actual_Revenue"]
    aggressive_base_metrics = {
        **base.evaluate_predictions(aggressive_actual, aggressive_context["base_pred"].to_numpy(dtype=float)),
        **compute_spike_metrics(
            aggressive_actual.to_numpy(dtype=float),
            aggressive_context["base_pred"].to_numpy(dtype=float),
        ),
    }
    plus_base_metrics = {
        **base.evaluate_predictions(plus_actual, plus_context["base_pred"].to_numpy(dtype=float)),
        **compute_spike_metrics(
            plus_actual.to_numpy(dtype=float),
            plus_context["base_pred"].to_numpy(dtype=float),
        ),
    }

    validation_best_payload: dict[str, Any] = {}
    for model_name in available_model_names:
        raw_aggr_scale = aggressive_raw_predictions[model_name]
        raw_plus_scale = plus_raw_predictions[model_name]
        for clip_range in CLIP_RANGES:
            clipped_aggr = np.clip(raw_aggr_scale, clip_range[0], clip_range[1])
            clipped_plus = np.clip(raw_plus_scale, clip_range[0], clip_range[1])

            aggr_metrics = evaluate_meta_predictions(
                aggressive_actual,
                aggressive_context["base_pred"].to_numpy(dtype=float),
                clipped_aggr,
            )
            plus_metrics = evaluate_meta_predictions(
                plus_actual,
                plus_context["base_pred"].to_numpy(dtype=float),
                clipped_plus,
            )

            comparison_rows.append(
                {
                    "base_context": "spike_gate_aggressive",
                    "model_name": model_name,
                    "clip_min": clip_range[0],
                    "clip_max": clip_range[1],
                    "base_RMSE": aggressive_base_metrics["RMSE"],
                    "base_top10_RMSE": aggressive_base_metrics["top10_RMSE"],
                    "base_non_spike_RMSE": aggressive_base_metrics["non_spike_RMSE"],
                    "mean_scale": float(np.mean(clipped_aggr)),
                    "min_scale": float(np.min(clipped_aggr)),
                    "max_scale": float(np.max(clipped_aggr)),
                    **{key: value for key, value in aggr_metrics.items() if key != "final_pred"},
                }
            )
            comparison_rows.append(
                {
                    "base_context": "adaptive_plus",
                    "model_name": model_name,
                    "clip_min": clip_range[0],
                    "clip_max": clip_range[1],
                    "base_RMSE": plus_base_metrics["RMSE"],
                    "base_top10_RMSE": plus_base_metrics["top10_RMSE"],
                    "base_non_spike_RMSE": plus_base_metrics["non_spike_RMSE"],
                    "mean_scale": float(np.mean(clipped_plus)),
                    "min_scale": float(np.min(clipped_plus)),
                    "max_scale": float(np.max(clipped_plus)),
                    **{key: value for key, value in plus_metrics.items() if key != "final_pred"},
                }
            )

            validation_best_payload[(model_name, clip_range)] = {
                "aggressive_scale": clipped_aggr,
                "plus_scale": clipped_plus,
                "aggressive_pred": aggr_metrics["final_pred"],
                "plus_pred": plus_metrics["final_pred"],
            }

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        ["base_context", "RMSE", "top10_RMSE", "top10_underprediction", "MAE"]
    ).reset_index(drop=True)
    comparison_df.to_csv(COMPARISON_PATH, index=False)
    reporter.emit_frame("Top meta-model configs:", comparison_df.head(15))

    best_config = select_best_meta_config(comparison_df)
    best_model_name = str(best_config["model_name"])
    best_clip = (float(best_config["clip_min"]), float(best_config["clip_max"]))
    payload = validation_best_payload[(best_model_name, best_clip)]

    reporter.emit("")
    reporter.emit("4. Save validation predictions and feature importance")
    validation_output = pd.DataFrame(
        {
            DATE_COL: plus_context[DATE_COL],
            "actual_Revenue": plus_context["actual_Revenue"],
            "spike_gate_aggressive_base_pred": aggressive_context["base_pred"],
            "spike_gate_aggressive_meta_pred": payload["aggressive_pred"],
            "adaptive_plus_base_pred": plus_context["base_pred"],
            "adaptive_plus_meta_scale_target": plus_scale_target,
            "best_meta_scale": payload["plus_scale"],
            "best_meta_pred": payload["plus_pred"],
            "best_model_name": best_model_name,
            "best_clip_min": best_clip[0],
            "best_clip_max": best_clip[1],
        }
    )
    validation_output.to_csv(VALIDATION_OUTPUT_PATH, index=False)

    importance_rows = [
        get_meta_importance(model_info).assign(model_name=model_name)
        for model_name, model_info in plus_full_models.items()
    ]
    importance_output = pd.concat(importance_rows, ignore_index=True)
    importance_output.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    reporter.emit("")
    reporter.emit("5. Build future meta features from adaptive_scale_plus base")
    sample_submission = base.load_sample_submission(SAMPLE_SUBMISSION_PATH)
    future_context = build_future_meta_context(ADAPTIVE_PLUS_SUBMISSION_PATH, reporter, logger)
    future_context = add_base_prediction_features(future_context)
    X_future, _ = build_meta_feature_matrix(future_context)
    base_future_submission = load_submission(ADAPTIVE_PLUS_SUBMISSION_PATH, sample_submission)

    chosen_model = plus_full_models[best_model_name]
    raw_future_scale = predict_meta_model(chosen_model, X_future)

    clip_variant_map = {
        "best": best_clip,
        "conservative": CONSERVATIVE_CLIP,
        "aggressive": AGGRESSIVE_CLIP,
    }
    future_scales = {
        name: np.clip(raw_future_scale, clip_range[0], clip_range[1])
        for name, clip_range in clip_variant_map.items()
    }

    reporter.emit("")
    reporter.emit("6. Save future submissions")
    for name, path in [
        ("best", SUBMISSION_BEST_PATH),
        ("conservative", SUBMISSION_CONSERVATIVE_PATH),
        ("aggressive", SUBMISSION_AGGRESSIVE_PATH),
    ]:
        scale_values = future_scales[name]
        output = sample_submission[[DATE_COL]].copy()
        output[TARGET_COL] = base_future_submission[TARGET_COL].to_numpy(dtype=float) * scale_values
        output[COGS_COL] = base_future_submission[COGS_COL].to_numpy(dtype=float) * scale_values
        output[TARGET_COL] = output[TARGET_COL].clip(lower=0.0)
        output[COGS_COL] = output[COGS_COL].clip(lower=0.0)
        validate_submission_frame(output, sample_submission)
        output.to_csv(path, index=False)

    reporter.emit("")
    reporter.emit("7. Final summary")
    adaptive_plus_best_row = comparison_df[
        (comparison_df["base_context"] == "adaptive_plus")
        & (comparison_df["model_name"] == best_model_name)
        & np.isclose(comparison_df["clip_min"], best_clip[0])
        & np.isclose(comparison_df["clip_max"], best_clip[1])
    ].iloc[0]

    aggressive_best_row = comparison_df[
        (comparison_df["base_context"] == "spike_gate_aggressive")
        & (comparison_df["model_name"] == best_model_name)
        & np.isclose(comparison_df["clip_min"], best_clip[0])
        & np.isclose(comparison_df["clip_max"], best_clip[1])
    ].iloc[0]

    top_meta_features = (
        importance_output[importance_output["model_name"] == best_model_name]
        .sort_values("importance_gain", ascending=False)
        .head(20)
    )
    reporter.emit(f"Best meta model: {best_model_name}")
    reporter.emit(f"Best clip range: {build_clip_key(best_clip)}")
    reporter.emit(
        f"Spike-gate aggressive RMSE before/after: {aggressive_base_metrics['RMSE']:,.2f} -> {aggressive_best_row['RMSE']:,.2f}"
    )
    reporter.emit(
        f"Adaptive-plus RMSE before/after: {plus_base_metrics['RMSE']:,.2f} -> {adaptive_plus_best_row['RMSE']:,.2f}"
    )
    reporter.emit(
        f"Adaptive-plus Top10 RMSE before/after: {plus_base_metrics['top10_RMSE']:,.2f} -> {adaptive_plus_best_row['top10_RMSE']:,.2f}"
    )
    reporter.emit(
        f"Adaptive-plus Non-spike RMSE before/after: {plus_base_metrics['non_spike_RMSE']:,.2f} -> {adaptive_plus_best_row['non_spike_RMSE']:,.2f}"
    )
    reporter.emit(
        f"Future scale mean/min/max: {np.mean(future_scales['best']):.6f} / "
        f"{np.min(future_scales['best']):.6f} / {np.max(future_scales['best']):.6f}"
    )
    reporter.emit_frame("Top meta features:", top_meta_features[["feature", "importance_gain", "importance_split"]])
    reporter.emit(
        "Created submission files: submission_meta_scale_best.csv, submission_meta_scale_conservative.csv, "
        "submission_meta_scale_aggressive.csv"
    )
    reporter.emit(
        "Recommended upload order: "
        f"{[SUBMISSION_BEST_PATH.name, SUBMISSION_CONSERVATIVE_PATH.name, SUBMISSION_AGGRESSIVE_PATH.name]}"
    )
    reporter.emit(
        "Leakage confirmation: the meta model only learns scale factors from forecast-safe context built on lagged recursive revenue features, future-known promo schedule, existing base predictions, and available model probabilities/components. No future actual Revenue/COGS or same-day realized demand was used."
    )

    reporter.save(REPORT_PATH)


if __name__ == "__main__":
    run()
