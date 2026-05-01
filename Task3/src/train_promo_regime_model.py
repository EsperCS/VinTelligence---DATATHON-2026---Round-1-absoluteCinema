from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import train_final_model as base
import train_spike_aware_model as spike


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "log"

FUTURE_PROMO_FEATURES_PATH = DATA_DIR / "future_promo_calendar_features.csv"

VALIDATION_PREDICTIONS_PATH = DATA_DIR / "promo_regime_validation_predictions.csv"
COMPARISON_PATH = DATA_DIR / "promo_regime_model_comparison.csv"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "promo_regime_feature_importance.csv"

SUBMISSION_PATH = DATA_DIR / "submission_promo_regime.csv"
SUBMISSION_BLEND_30_PATH = DATA_DIR / "submission_regime_blend_30.csv"
SUBMISSION_BLEND_50_PATH = DATA_DIR / "submission_regime_blend_50.csv"
SUBMISSION_BLEND_70_PATH = DATA_DIR / "submission_regime_blend_70.csv"
SUBMISSION_BLEND_PUBLICBEST_PATH = DATA_DIR / "submission_regime_blend_publicbest.csv"

PRUNED_VALIDATION_PATH = DATA_DIR / "pruned_ensemble_validation_predictions.csv"
SPIKE_VALIDATION_PATH = DATA_DIR / "spike_model_validation_predictions.csv"
PRUNED_SUBMISSION_PATH = DATA_DIR / "submission_pruned_ensemble.csv"
SPIKE_SUBMISSION_PATH = DATA_DIR / "submission_spike_aware.csv"

REPORT_PATH = LOG_DIR / "promo_regime_model_report.txt"
LOG_FILE = LOG_DIR / "train_promo_regime_model.log"

REGIME_NON_PROMO = "non_promo"
REGIME_PROMO_NORMAL = "promo_normal"
REGIME_PROMO_HIGH = "promo_high"
REGIME_ORDER = [REGIME_NON_PROMO, REGIME_PROMO_NORMAL, REGIME_PROMO_HIGH]

CURRENT_BASELINES = {
    "PRUNED_ENSEMBLE": {"RMSE": 943_731.57},
    "SPIKE_MODEL": {"RMSE": 842_278.60},
}

FUTURE_PROMO_RENAME_MAP = {
    "future_calendar_active_promo_count": "calendar_active_promo_count",
    "future_calendar_any_promo": "calendar_any_promo",
    "future_calendar_avg_discount_value": "calendar_avg_discount_value",
    "future_calendar_max_discount_value": "calendar_max_discount_value",
    "future_calendar_stackable_promo_count": "calendar_stackable_promo_count",
    "future_calendar_has_stackable_promo": "calendar_has_stackable_promo",
    "future_calendar_has_category_specific_promo": "calendar_has_category_specific_promo",
    "future_calendar_percentage_promo_count": "calendar_percentage_promo_count",
    "future_calendar_fixed_promo_count": "calendar_fixed_promo_count",
}


class Reporter:
    """Small helper to print, log, and save a plain-text report."""

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

    logger = logging.getLogger("train_promo_regime_model")
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


def deduplicate_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def load_future_promo_features(path: Path = FUTURE_PROMO_FEATURES_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Future promo calendar features not found: {path}")

    future = pd.read_csv(path, parse_dates=[base.DATE_COL], low_memory=False)
    future[base.DATE_COL] = pd.to_datetime(future[base.DATE_COL], errors="coerce").dt.normalize()
    if future[base.DATE_COL].isna().any():
        raise ValueError("future_promo_calendar_features.csv contains invalid dates")

    missing = [column for column in FUTURE_PROMO_RENAME_MAP if column not in future.columns]
    if missing:
        raise ValueError(f"Missing required future promo columns: {missing}")

    future = future.rename(columns=FUTURE_PROMO_RENAME_MAP)
    keep_columns = [base.DATE_COL] + list(FUTURE_PROMO_RENAME_MAP.values())
    future = future[keep_columns].copy()
    for column in base.PROMOTION_FEATURES:
        if column not in future.columns:
            future[column] = 0.0
    future[base.PROMOTION_FEATURES] = (
        future[base.PROMOTION_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    )
    return future[[base.DATE_COL] + base.PROMOTION_FEATURES].sort_values(base.DATE_COL).reset_index(drop=True)


def build_future_static_features(
    future_dates: pd.Series,
    min_date: pd.Timestamp,
    logger: logging.Logger,
) -> pd.DataFrame:
    calendar = base.build_calendar_features(future_dates, min_date)
    promo = load_future_promo_features(FUTURE_PROMO_FEATURES_PATH)
    inventory = base.build_inventory_asof_features(future_dates, base.INVENTORY_PATH, logger)
    return (
        calendar.merge(promo, on=base.DATE_COL, how="left", validate="one_to_one")
        .merge(inventory, on=base.DATE_COL, how="left", validate="one_to_one")
        .fillna(0)
        .sort_values(base.DATE_COL)
        .reset_index(drop=True)
    )


def compute_promo_intensity(frame: pd.DataFrame) -> pd.Series:
    return (
        pd.to_numeric(frame["calendar_avg_discount_value"], errors="coerce").fillna(0.0)
        * pd.to_numeric(frame["calendar_active_promo_count"], errors="coerce").fillna(0.0)
    )


def compute_intensity_threshold(static_features: pd.DataFrame, train_end_exclusive: pd.Timestamp) -> float:
    promo_days = static_features[
        (static_features[base.DATE_COL] < train_end_exclusive)
        & (pd.to_numeric(static_features["calendar_any_promo"], errors="coerce").fillna(0.0) > 0)
    ].copy()
    if promo_days.empty:
        return 0.0
    intensities = compute_promo_intensity(promo_days)
    return float(intensities.quantile(0.75))


def assign_regimes(static_features: pd.DataFrame, intensity_threshold: float) -> pd.DataFrame:
    output = static_features[[base.DATE_COL] + base.PROMOTION_FEATURES].copy()
    output["promo_intensity"] = compute_promo_intensity(output)
    any_promo = pd.to_numeric(output["calendar_any_promo"], errors="coerce").fillna(0.0) > 0

    output["regime"] = REGIME_NON_PROMO
    output.loc[any_promo, "regime"] = REGIME_PROMO_NORMAL
    output.loc[any_promo & (output["promo_intensity"] >= intensity_threshold), "regime"] = REGIME_PROMO_HIGH
    return output[[base.DATE_COL, "promo_intensity", "regime"]].sort_values(base.DATE_COL).reset_index(drop=True)


def determine_regime_from_row(row: dict[str, float], intensity_threshold: float) -> str:
    any_promo = float(row.get("calendar_any_promo", 0.0) or 0.0) > 0.0
    if not any_promo:
        return REGIME_NON_PROMO

    promo_intensity = (
        float(row.get("calendar_avg_discount_value", 0.0) or 0.0)
        * float(row.get("calendar_active_promo_count", 0.0) or 0.0)
    )
    if promo_intensity >= intensity_threshold:
        return REGIME_PROMO_HIGH
    return REGIME_PROMO_NORMAL


def fallback_regime(requested_regime: str, trained_models: dict[str, dict[str, Any]]) -> str:
    if requested_regime in trained_models:
        return requested_regime
    fallback_order = {
        REGIME_PROMO_HIGH: [REGIME_PROMO_NORMAL, REGIME_NON_PROMO],
        REGIME_PROMO_NORMAL: [REGIME_NON_PROMO, REGIME_PROMO_HIGH],
        REGIME_NON_PROMO: [REGIME_PROMO_NORMAL, REGIME_PROMO_HIGH],
    }
    for regime in fallback_order.get(requested_regime, []):
        if regime in trained_models:
            return regime
    raise ValueError("No trained regime model available for prediction.")


def build_feature_columns(model_table: pd.DataFrame) -> list[str]:
    top_features = spike.load_top_full_features(limit=50)
    return deduplicate_preserve_order(
        [feature for feature in top_features if feature in model_table.columns] + spike.SPIKE_FEATURES
    )


def make_regime_training_matrix(
    model_table: pd.DataFrame,
    feature_columns: list[str],
    regime_name: str,
    train_end_exclusive: pd.Timestamp | None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    table = model_table.copy()
    if train_end_exclusive is not None:
        table = table[table[base.DATE_COL] < train_end_exclusive].copy()
    table = table[table["regime"] == regime_name].copy()

    missing = [column for column in feature_columns if column not in table.columns]
    if missing:
        raise ValueError(f"Missing feature columns for regime model: {missing}")

    clean = table.dropna(subset=feature_columns + [base.TARGET_COL]).reset_index(drop=True)
    if clean.empty:
        raise ValueError(f"No training rows remain for regime={regime_name} after dropping NaNs")

    X = clean[feature_columns].copy()
    y = clean[base.TARGET_COL].copy()
    medians = X.median(numeric_only=True)
    return X, y, clean, medians


def train_regime_models(
    model_table: pd.DataFrame,
    feature_columns: list[str],
    train_end_exclusive: pd.Timestamp | None,
    reporter: Reporter,
    label: str,
) -> dict[str, dict[str, Any]]:
    trained: dict[str, dict[str, Any]] = {}

    for regime_name in REGIME_ORDER:
        X_train, y_train, clean, medians = make_regime_training_matrix(
            model_table=model_table,
            feature_columns=feature_columns,
            regime_name=regime_name,
            train_end_exclusive=train_end_exclusive,
        )
        reporter.emit(
            f"Training {label} | regime={regime_name}: rows={len(X_train):,}, features={len(feature_columns)}"
        )
        model, model_type = spike.train_variant_model(X_train, y_train, reporter, objective="regression")
        trained[regime_name] = {
            "regime": regime_name,
            "model_object": model,
            "model_type": model_type,
            "feature_columns": feature_columns,
            "feature_medians": medians,
            "X_train": X_train,
            "y_train": y_train,
            "train_clean": clean,
        }

    return trained


def recursive_predict_regime(
    trained_models: dict[str, dict[str, Any]],
    prediction_dates: pd.Series,
    static_features: pd.DataFrame,
    initial_revenue_history: pd.Series,
    intensity_threshold: float,
    volatility_threshold: float,
) -> tuple[np.ndarray, pd.Series]:
    static_by_date = static_features.set_index(base.DATE_COL).sort_index()
    history = pd.to_numeric(initial_revenue_history, errors="coerce").sort_index().copy()

    predictions: list[float] = []
    regimes: list[str] = []

    for forecast_date in pd.to_datetime(prediction_dates):
        if forecast_date not in static_by_date.index:
            raise ValueError(f"Missing static features for forecast date {forecast_date.date()}")

        row: dict[str, float] = static_by_date.loc[forecast_date].to_dict()
        row.update(base.compute_revenue_features_from_history(history, forecast_date))
        row.update(spike.compute_spike_features_from_row(row, volatility_threshold))

        desired_regime = determine_regime_from_row(row, intensity_threshold)
        selected_regime = fallback_regime(desired_regime, trained_models)
        trained = trained_models[selected_regime]

        X_row = pd.DataFrame([row], columns=trained["feature_columns"])
        X_row = X_row.apply(pd.to_numeric, errors="coerce").fillna(trained["feature_medians"]).fillna(0.0)

        prediction = float(trained["model_object"].predict(X_row)[0])
        prediction = max(0.0, prediction)
        predictions.append(prediction)
        regimes.append(desired_regime)
        history.loc[forecast_date] = prediction

    return np.asarray(predictions, dtype=float), pd.Series(regimes, index=pd.to_datetime(prediction_dates))


def rmse_from_mask(errors: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return np.nan
    return float(np.sqrt(np.mean(errors[mask] ** 2)))


def evaluate_regime_predictions(
    name: str,
    actual: pd.Series,
    predicted: np.ndarray,
    regimes: pd.Series,
) -> dict[str, Any]:
    base_metrics = spike.evaluate_candidate(name, actual, predicted)
    errors = actual.to_numpy(dtype=float) - np.asarray(predicted, dtype=float)
    regime_values = pd.Series(regimes).reset_index(drop=True)

    output: dict[str, Any] = dict(base_metrics)
    for regime_name in REGIME_ORDER:
        mask = regime_values.eq(regime_name).to_numpy(dtype=bool)
        output[f"{regime_name}_RMSE"] = rmse_from_mask(errors, mask)
        output[f"{regime_name}_count"] = int(mask.sum())
    return output


def save_validation_predictions(
    dates: pd.Series,
    actual: pd.Series,
    predicted: np.ndarray,
    regimes: pd.Series,
    path: Path = VALIDATION_PREDICTIONS_PATH,
) -> pd.DataFrame:
    actual_values = actual.to_numpy(dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    error = actual_values - predicted_values
    output = pd.DataFrame(
        {
            base.DATE_COL: pd.to_datetime(dates).reset_index(drop=True),
            "regime": pd.Series(regimes).reset_index(drop=True),
            "actual_Revenue": actual_values,
            "predicted_Revenue": predicted_values,
            "error": error,
            "abs_error": np.abs(error),
            "pct_error": np.where(actual_values != 0, error / actual_values, np.nan),
        }
    )
    output.to_csv(path, index=False)
    return output


def load_validation_predictions(path: Path, expected_dates: pd.Series) -> pd.DataFrame:
    validation = pd.read_csv(path, parse_dates=[base.DATE_COL], low_memory=False)
    validation[base.DATE_COL] = pd.to_datetime(validation[base.DATE_COL], errors="coerce").dt.normalize()
    required = {base.DATE_COL, "actual_Revenue", "predicted_Revenue"}
    if not required.issubset(validation.columns):
        raise ValueError(f"Validation file missing required columns: {path}")

    aligned = pd.DataFrame({base.DATE_COL: pd.to_datetime(expected_dates).reset_index(drop=True)}).merge(
        validation[[base.DATE_COL, "actual_Revenue", "predicted_Revenue"]],
        on=base.DATE_COL,
        how="left",
        validate="one_to_one",
    )
    if aligned[["actual_Revenue", "predicted_Revenue"]].isna().any().any():
        raise ValueError(f"Validation predictions missing rows after alignment: {path}")
    return aligned


def load_submission(path: Path, sample_submission: pd.DataFrame) -> pd.DataFrame:
    return spike.load_submission(path, sample_submission)


def validate_submission_frame(submission: pd.DataFrame, sample_submission: pd.DataFrame, label: str) -> None:
    if len(submission) != len(sample_submission):
        raise ValueError(f"{label}: row count mismatch")
    if submission.columns.tolist() != [base.DATE_COL, base.TARGET_COL, base.COGS_COL]:
        raise ValueError(f"{label}: invalid columns {submission.columns.tolist()}")
    if not submission[base.DATE_COL].equals(sample_submission[base.DATE_COL]):
        raise ValueError(f"{label}: Date order mismatch")
    if submission[[base.TARGET_COL, base.COGS_COL]].isna().any().any():
        raise ValueError(f"{label}: missing values found")
    if (submission[[base.TARGET_COL, base.COGS_COL]] < 0).any().any():
        raise ValueError(f"{label}: negative values found")


def build_submission(
    sample_submission: pd.DataFrame,
    predictions: np.ndarray,
    cogs_ratio: float,
    path: Path,
) -> pd.DataFrame:
    output = pd.DataFrame({base.DATE_COL: sample_submission[base.DATE_COL].copy()})
    output[base.TARGET_COL] = np.maximum(0.0, np.asarray(predictions, dtype=float))
    output[base.COGS_COL] = np.maximum(0.0, output[base.TARGET_COL] * float(cogs_ratio))
    output = output[[base.DATE_COL, base.TARGET_COL, base.COGS_COL]]
    output.to_csv(path, index=False)
    return output


def build_importance_frame(
    trained_models: dict[str, dict[str, Any]],
    validation_metrics: dict[str, Any],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for regime_name, trained in trained_models.items():
        importance = base.get_feature_importance(
            trained["model_object"],
            trained["model_type"],
            trained["feature_columns"],
            trained["X_train"],
            trained["y_train"],
            validation_metrics["RMSE"],
        ).copy()
        importance.insert(0, "model", regime_name)
        importance["validation_rmse_overall"] = validation_metrics["RMSE"]
        importance["validation_rmse_regime"] = validation_metrics.get(f"{regime_name}_RMSE", np.nan)
        importance["train_rows"] = len(trained["X_train"])
        rows.append(importance)
    if not rows:
        return pd.DataFrame(columns=["model", "feature", "importance_split", "importance_gain"])
    return pd.concat(rows, ignore_index=True)


def compute_blend_validation_metrics(
    actual: pd.Series,
    regimes: pd.Series,
    pruned_pred: np.ndarray,
    spike_pred: np.ndarray,
    regime_pred: np.ndarray,
) -> pd.DataFrame:
    blend_specs = {
        "PUBLIC_BEST_50_50": {"pruned": 0.50, "spike": 0.50, "regime": 0.00},
        "REGIME_BLEND_30": {"pruned": 0.35, "spike": 0.35, "regime": 0.30},
        "REGIME_BLEND_50": {"pruned": 0.25, "spike": 0.25, "regime": 0.50},
        "REGIME_BLEND_70": {"pruned": 0.15, "spike": 0.15, "regime": 0.70},
        "REGIME_BLEND_PUBLICBEST": {"pruned": 0.25, "spike": 0.25, "regime": 0.50},
    }
    rows: list[dict[str, Any]] = []
    for name, weights in blend_specs.items():
        predicted = (
            weights["pruned"] * pruned_pred
            + weights["spike"] * spike_pred
            + weights["regime"] * regime_pred
        )
        metrics = evaluate_regime_predictions(name, actual, predicted, regimes)
        metrics["weight_pruned"] = weights["pruned"]
        metrics["weight_spike"] = weights["spike"]
        metrics["weight_regime"] = weights["regime"]
        rows.append(metrics)
    return pd.DataFrame(rows)


def build_blended_submissions(
    sample_submission: pd.DataFrame,
    regime_submission: pd.DataFrame,
    reporter: Reporter,
) -> dict[str, pd.DataFrame]:
    pruned_submission = load_submission(PRUNED_SUBMISSION_PATH, sample_submission)
    spike_submission = load_submission(SPIKE_SUBMISSION_PATH, sample_submission)

    base_components = {
        "pruned": pruned_submission,
        "spike": spike_submission,
        "regime": regime_submission,
    }

    blend_30 = spike.blend_submissions(
        sample_submission,
        base_components,
        {"pruned": 0.35, "spike": 0.35, "regime": 0.30},
    )
    blend_50 = spike.blend_submissions(
        sample_submission,
        base_components,
        {"pruned": 0.25, "spike": 0.25, "regime": 0.50},
    )
    blend_70 = spike.blend_submissions(
        sample_submission,
        base_components,
        {"pruned": 0.15, "spike": 0.15, "regime": 0.70},
    )

    public_best_component = spike.blend_submissions(
        sample_submission,
        {"pruned": pruned_submission, "spike": spike_submission},
        {"pruned": 0.50, "spike": 0.50},
    )
    publicbest = spike.blend_submissions(
        sample_submission,
        {"public_best": public_best_component, "regime": regime_submission},
        {"public_best": 0.50, "regime": 0.50},
    )

    outputs = {
        str(SUBMISSION_BLEND_30_PATH): blend_30,
        str(SUBMISSION_BLEND_50_PATH): blend_50,
        str(SUBMISSION_BLEND_70_PATH): blend_70,
        str(SUBMISSION_BLEND_PUBLICBEST_PATH): publicbest,
    }
    for path_str, submission in outputs.items():
        path = Path(path_str)
        submission.to_csv(path, index=False)
        validate_submission_frame(submission, sample_submission, path.name)
        reporter.emit(f"Saved validated blended submission: {path}")
    return outputs


def format_metrics_line(metrics: dict[str, Any]) -> str:
    return (
        f"MAE={metrics['MAE']:,.2f} | RMSE={metrics['RMSE']:,.2f} | R2={metrics['R2']:.6f} | "
        f"Top10 RMSE={metrics['top10_RMSE']:,.2f} | Top10 underprediction="
        f"{metrics['top10_underprediction']}/{metrics['top10_count']}"
    )


def run_experiment() -> None:
    logger = setup_logging()
    reporter = Reporter(logger)

    reporter.emit("Promotion-Regime Forecasting Model")
    reporter.emit("================================")
    reporter.emit("")

    reporter.emit("1. Load train data, sample submission, and rebuild forecast-safe features")
    train_df = base.load_train_data(base.TRAIN_DATA_PATH)
    sample_submission = base.load_sample_submission(base.SAMPLE_SUBMISSION_PATH)
    historical_static = base.build_static_features(train_df[base.DATE_COL], train_df[base.DATE_COL].min(), logger)
    future_static = build_future_static_features(
        sample_submission[base.DATE_COL],
        train_df[base.DATE_COL].min(),
        logger,
    )
    spike_table = spike.build_spike_model_table(train_df, historical_static)
    feature_columns = build_feature_columns(spike_table)
    reporter.emit(f"Historical static feature shape: {historical_static.shape}")
    reporter.emit(f"Future static feature shape: {future_static.shape}")
    reporter.emit(f"Spike table shape: {spike_table.shape}")
    reporter.emit(f"Feature count after spike merge/dedupe: {len(feature_columns)}")

    reporter.emit("")
    reporter.emit("2. Build historical promo regimes using training-only intensity threshold")
    validation_intensity_threshold = compute_intensity_threshold(historical_static, base.TRAIN_CUTOFF)
    historical_regimes = assign_regimes(historical_static, validation_intensity_threshold)
    labeled_validation_table = spike_table.merge(
        historical_regimes[[base.DATE_COL, "promo_intensity", "regime"]],
        on=base.DATE_COL,
        how="left",
        validate="one_to_one",
    )
    validation_train_counts = (
        historical_regimes[historical_regimes[base.DATE_COL] < base.TRAIN_CUTOFF]["regime"]
        .value_counts()
        .reindex(REGIME_ORDER, fill_value=0)
        .rename_axis("regime")
        .reset_index(name="count")
    )
    validation_2022_counts = (
        historical_regimes[
            (historical_regimes[base.DATE_COL] >= base.TRAIN_CUTOFF)
            & (historical_regimes[base.DATE_COL] <= base.VALIDATION_END)
        ]["regime"]
        .value_counts()
        .reindex(REGIME_ORDER, fill_value=0)
        .rename_axis("regime")
        .reset_index(name="count")
    )
    reporter.emit(f"Validation promo intensity p75 threshold: {validation_intensity_threshold:,.4f}")
    reporter.emit_frame("Training regime counts (pre-2022):", validation_train_counts)
    reporter.emit_frame("Validation regime counts (2022):", validation_2022_counts)

    reporter.emit("")
    reporter.emit("3. Train 3 regime models and run recursive validation on 2022")
    validation_models = train_regime_models(
        model_table=labeled_validation_table,
        feature_columns=feature_columns,
        train_end_exclusive=base.TRAIN_CUTOFF,
        reporter=reporter,
        label="validation",
    )
    validation_dates = train_df[
        (train_df[base.DATE_COL] >= base.TRAIN_CUTOFF) & (train_df[base.DATE_COL] <= base.VALIDATION_END)
    ][base.DATE_COL]
    validation_actual = train_df.set_index(base.DATE_COL).loc[validation_dates, base.TARGET_COL].reset_index(drop=True)
    validation_history = train_df[train_df[base.DATE_COL] < base.TRAIN_CUTOFF].set_index(base.DATE_COL)[base.TARGET_COL]
    validation_volatility_threshold = spike.compute_fixed_volatility_threshold(validation_history)
    validation_predictions, validation_used_regimes = recursive_predict_regime(
        trained_models=validation_models,
        prediction_dates=validation_dates,
        static_features=historical_static,
        initial_revenue_history=validation_history,
        intensity_threshold=validation_intensity_threshold,
        volatility_threshold=validation_volatility_threshold,
    )
    regime_metrics = evaluate_regime_predictions(
        "PROMO_REGIME_MODEL",
        validation_actual,
        validation_predictions,
        validation_used_regimes,
    )
    reporter.emit(format_metrics_line(regime_metrics))
    reporter.emit(
        f"RMSE by regime | non_promo={regime_metrics['non_promo_RMSE']:,.2f} | "
        f"promo_normal={regime_metrics['promo_normal_RMSE']:,.2f} | "
        f"promo_high={regime_metrics['promo_high_RMSE']:,.2f}"
    )

    validation_output = save_validation_predictions(
        validation_dates,
        validation_actual,
        validation_predictions,
        validation_used_regimes,
        VALIDATION_PREDICTIONS_PATH,
    )

    reporter.emit("")
    reporter.emit("4. Compare against strong baselines and regime blends on validation")
    pruned_validation = load_validation_predictions(PRUNED_VALIDATION_PATH, validation_dates)
    spike_validation = load_validation_predictions(SPIKE_VALIDATION_PATH, validation_dates)

    pruned_metrics = evaluate_regime_predictions(
        "PRUNED_ENSEMBLE",
        validation_actual,
        pruned_validation["predicted_Revenue"].to_numpy(dtype=float),
        validation_used_regimes,
    )
    spike_metrics = evaluate_regime_predictions(
        "SPIKE_MODEL",
        validation_actual,
        spike_validation["predicted_Revenue"].to_numpy(dtype=float),
        validation_used_regimes,
    )

    blend_comparison = compute_blend_validation_metrics(
        actual=validation_actual,
        regimes=validation_used_regimes,
        pruned_pred=pruned_validation["predicted_Revenue"].to_numpy(dtype=float),
        spike_pred=spike_validation["predicted_Revenue"].to_numpy(dtype=float),
        regime_pred=validation_predictions,
    )

    comparison_rows = [pruned_metrics, spike_metrics, regime_metrics]
    comparison_df = pd.concat([pd.DataFrame(comparison_rows), blend_comparison], ignore_index=True)
    comparison_df = comparison_df.sort_values(
        ["RMSE", "top10_RMSE", "top10_underprediction", "MAE"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    comparison_df.to_csv(COMPARISON_PATH, index=False)
    reporter.emit_frame("Validation comparison:", comparison_df)

    reporter.emit("")
    reporter.emit("5. Retrain regime models on full 2012-2022 history and forecast future dates")
    full_intensity_threshold = compute_intensity_threshold(historical_static, train_df[base.DATE_COL].max() + pd.Timedelta(days=1))
    full_regimes = assign_regimes(historical_static, full_intensity_threshold)
    full_labeled_table = spike_table.merge(
        full_regimes[[base.DATE_COL, "promo_intensity", "regime"]],
        on=base.DATE_COL,
        how="left",
        validate="one_to_one",
    )
    full_models = train_regime_models(
        model_table=full_labeled_table,
        feature_columns=feature_columns,
        train_end_exclusive=None,
        reporter=reporter,
        label="full_train",
    )

    future_predictions, future_regimes = recursive_predict_regime(
        trained_models=full_models,
        prediction_dates=sample_submission[base.DATE_COL],
        static_features=future_static,
        initial_revenue_history=train_df.set_index(base.DATE_COL)[base.TARGET_COL],
        intensity_threshold=full_intensity_threshold,
        volatility_threshold=spike.compute_fixed_volatility_threshold(
            train_df.set_index(base.DATE_COL)[base.TARGET_COL]
        ),
    )
    cogs_ratio = base.estimate_cogs_ratio(train_df)
    regime_submission = build_submission(sample_submission, future_predictions, cogs_ratio, SUBMISSION_PATH)
    validate_submission_frame(regime_submission, sample_submission, SUBMISSION_PATH.name)

    future_regime_counts = (
        pd.Series(future_regimes)
        .value_counts()
        .reindex(REGIME_ORDER, fill_value=0)
        .rename_axis("regime")
        .reset_index(name="count")
    )
    reporter.emit(f"Full-train promo intensity p75 threshold: {full_intensity_threshold:,.4f}")
    reporter.emit_frame("Future regime counts:", future_regime_counts)

    reporter.emit("")
    reporter.emit("6. Build regime blends with current best submissions")
    build_blended_submissions(sample_submission, regime_submission, reporter)

    reporter.emit("")
    reporter.emit("7. Save feature importance and final summary")
    importance_df = build_importance_frame(full_models, regime_metrics)
    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    for regime_name in REGIME_ORDER:
        top_importance = (
            importance_df[importance_df["model"] == regime_name]
            .sort_values(["importance_gain", "importance_split"], ascending=False)
            .head(10)
        )
        reporter.emit_frame(f"Top features for {regime_name}:", top_importance)

    regime_beats_spike = regime_metrics["RMSE"] < spike_metrics["RMSE"]
    upload_candidates = comparison_df[
        comparison_df["model"].isin(
            [
                "PROMO_REGIME_MODEL",
                "REGIME_BLEND_30",
                "REGIME_BLEND_50",
                "REGIME_BLEND_70",
                "REGIME_BLEND_PUBLICBEST",
            ]
        )
    ][["model", "RMSE", "MAE", "R2"]]

    reporter.emit("Overall validation metrics:")
    reporter.emit(format_metrics_line(regime_metrics))
    reporter.emit(
        f"RMSE by regime: non_promo={regime_metrics['non_promo_RMSE']:,.2f}, "
        f"promo_normal={regime_metrics['promo_normal_RMSE']:,.2f}, "
        f"promo_high={regime_metrics['promo_high_RMSE']:,.2f}"
    )
    reporter.emit(
        f"Top 10% spike metrics: RMSE={regime_metrics['top10_RMSE']:,.2f}, "
        f"underprediction={regime_metrics['top10_underprediction']}/{regime_metrics['top10_count']}"
    )
    reporter.emit(
        f"Does regime model beat spike model on validation? {regime_beats_spike} "
        f"(regime RMSE={regime_metrics['RMSE']:,.2f} vs spike RMSE={spike_metrics['RMSE']:,.2f})"
    )
    reporter.emit_frame("Suggested upload order from validation:", upload_candidates.sort_values("RMSE"))
    reporter.emit(f"Saved validation predictions: {VALIDATION_PREDICTIONS_PATH}")
    reporter.emit(f"Saved comparison table: {COMPARISON_PATH}")
    reporter.emit(f"Saved feature importance: {FEATURE_IMPORTANCE_PATH}")
    reporter.emit(f"Saved main submission: {SUBMISSION_PATH}")
    reporter.emit(f"Saved blended submissions: {SUBMISSION_BLEND_30_PATH}, {SUBMISSION_BLEND_50_PATH}, {SUBMISSION_BLEND_70_PATH}, {SUBMISSION_BLEND_PUBLICBEST_PATH}")
    reporter.emit(
        "Leakage confirmation: regime assignment uses only promotion calendar features; "
        "models use only lagged Revenue, calendar, promotion schedule, and inventory as-of features; "
        "validation and future forecasting remain fully recursive without future Revenue or future COGS."
    )
    reporter.save(REPORT_PATH)


if __name__ == "__main__":
    run_experiment()
