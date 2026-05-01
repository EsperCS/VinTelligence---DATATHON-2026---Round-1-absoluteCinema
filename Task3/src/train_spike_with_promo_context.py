from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import train_final_model as base
import train_final_model_promo_duration as promo_duration
import train_spike_aware_model as spike1


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "log"

VALIDATION_PREDICTIONS_PATH = DATA_DIR / "spike_promo_context_validation_predictions.csv"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "spike_promo_context_feature_importance.csv"
SUBMISSION_PATH = DATA_DIR / "submission_spike_promo_context.csv"

METRICS_PATH = LOG_DIR / "spike_promo_context_metrics.txt"
LOG_FILE = LOG_DIR / "train_spike_promo_context.log"

CURRENT_SPIKE_RMSE = 842_278.60

PROMO_CONTEXT_BASE_FEATURES = [
    "calendar_any_promo",
    "calendar_active_promo_count",
    "calendar_avg_discount_value",
    "calendar_max_discount_value",
    "promo_day_number",
    "promo_days_remaining",
    "promo_duration",
    "promo_progress_ratio",
]

PROMO_CONTEXT_FEATURES = [
    "promo_intensity",
    "promo_month",
    "promo_day_of_year",
    "promo_early",
    "promo_mid",
    "promo_late",
    "promo_x_lag365",
    "promo_x_spike",
    "promo_x_volatility",
]


class Reporter:
    """Print, log, and persist run messages."""

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

    def save(self, path: Path = METRICS_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        self.logger.info("Saved promo-context metrics to %s", path)


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_spike_with_promo_context")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger


def add_promo_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add richer promotion context features on top of the spike-aware table."""
    output = df.sort_values(base.DATE_COL).reset_index(drop=True).copy()

    output["promo_day_number"] = pd.to_numeric(output.get("promo_avg_day_number", 0), errors="coerce").fillna(0)
    output["promo_days_remaining"] = pd.to_numeric(
        output.get("promo_avg_days_remaining", 0),
        errors="coerce",
    ).fillna(0)
    output["promo_duration"] = pd.to_numeric(output.get("promo_avg_duration_days", 0), errors="coerce").fillna(0)
    output["promo_progress_ratio"] = pd.to_numeric(
        output.get("promo_avg_progress_ratio", 0),
        errors="coerce",
    ).fillna(0)

    output["promo_intensity"] = output["calendar_avg_discount_value"] * output["calendar_active_promo_count"]
    output["promo_month"] = output["month"] * output["calendar_any_promo"]
    output["promo_day_of_year"] = output["day_of_year"] * output["calendar_any_promo"]

    output["promo_early"] = (output["promo_progress_ratio"] < 0.3).astype(int)
    output["promo_mid"] = (
        (output["promo_progress_ratio"] >= 0.3) & (output["promo_progress_ratio"] <= 0.7)
    ).astype(int)
    output["promo_late"] = (output["promo_progress_ratio"] > 0.7).astype(int)

    output["promo_x_lag365"] = output["calendar_any_promo"] * output["revenue_lag_365"]
    output["promo_x_spike"] = output["calendar_any_promo"] * output["lag7_to_roll30_ratio"]
    output["promo_x_volatility"] = output["calendar_any_promo"] * output["volatility_30"]

    return output


def build_promo_context_model_table(
    train_df: pd.DataFrame,
    static_features: pd.DataFrame,
) -> pd.DataFrame:
    base_table = spike1.build_spike_model_table(train_df, static_features)
    return add_promo_context_features(base_table)


def compute_promo_context_features_from_row(row: dict[str, float]) -> dict[str, float]:
    promo_progress_ratio = float(row.get("promo_progress_ratio", 0.0))
    calendar_any_promo = float(row.get("calendar_any_promo", 0.0))
    calendar_active_promo_count = float(row.get("calendar_active_promo_count", 0.0))
    calendar_avg_discount_value = float(row.get("calendar_avg_discount_value", 0.0))

    return {
        "promo_intensity": calendar_avg_discount_value * calendar_active_promo_count,
        "promo_month": float(row.get("month", 0.0)) * calendar_any_promo,
        "promo_day_of_year": float(row.get("day_of_year", 0.0)) * calendar_any_promo,
        "promo_early": float(int(promo_progress_ratio < 0.3)),
        "promo_mid": float(int(0.3 <= promo_progress_ratio <= 0.7)),
        "promo_late": float(int(promo_progress_ratio > 0.7)),
        "promo_x_lag365": calendar_any_promo * float(row.get("revenue_lag_365", np.nan)),
        "promo_x_spike": calendar_any_promo * float(row.get("lag7_to_roll30_ratio", np.nan)),
        "promo_x_volatility": calendar_any_promo * float(row.get("volatility_30", np.nan)),
    }


def recursive_predict_promo_context(
    model: Any,
    model_type: str,
    prediction_dates: pd.Series,
    feature_columns: list[str],
    static_features: pd.DataFrame,
    initial_revenue_history: pd.Series,
    feature_medians: pd.Series,
    volatility_threshold: float,
) -> np.ndarray:
    del model_type
    static_by_date = static_features.set_index(base.DATE_COL).sort_index()
    history = pd.to_numeric(initial_revenue_history, errors="coerce").sort_index().copy()
    predictions: list[float] = []

    for forecast_date in pd.to_datetime(prediction_dates):
        if forecast_date not in static_by_date.index:
            raise ValueError(f"Missing static features for forecast date {forecast_date.date()}")

        row: dict[str, float] = static_by_date.loc[forecast_date].to_dict()
        row["promo_day_number"] = float(row.get("promo_avg_day_number", 0.0))
        row["promo_days_remaining"] = float(row.get("promo_avg_days_remaining", 0.0))
        row["promo_duration"] = float(row.get("promo_avg_duration_days", 0.0))
        row["promo_progress_ratio"] = float(row.get("promo_avg_progress_ratio", 0.0))
        row.update(base.compute_revenue_features_from_history(history, forecast_date))
        row.update(spike1.compute_spike_features_from_row(row, volatility_threshold))
        row.update(compute_promo_context_features_from_row(row))

        X_row = pd.DataFrame([row], columns=feature_columns)
        X_row = X_row.apply(pd.to_numeric, errors="coerce").fillna(feature_medians).fillna(0)

        prediction = float(model.predict(X_row)[0])
        prediction = max(0.0, prediction)
        predictions.append(prediction)
        history.loc[forecast_date] = prediction

    return np.asarray(predictions, dtype=float)


def train_validate_model(
    model_table: pd.DataFrame,
    static_features: pd.DataFrame,
    train_df: pd.DataFrame,
    feature_columns: list[str],
    reporter: Reporter,
) -> dict[str, Any]:
    X_train, y_train, train_clean, feature_medians = spike1.make_training_matrix(
        model_table,
        feature_columns,
        base.TRAIN_CUTOFF,
    )
    reporter.emit(
        f"Training spike+promo-context model: rows={len(X_train):,}, features={len(feature_columns)}"
    )
    model, model_type = spike1.train_variant_model(
        X_train,
        y_train,
        reporter,
        objective="quantile",
        alpha=0.70,
    )

    validation_dates = train_df[
        (train_df[base.DATE_COL] >= base.TRAIN_CUTOFF) & (train_df[base.DATE_COL] <= base.VALIDATION_END)
    ][base.DATE_COL]
    actual = train_df.set_index(base.DATE_COL).loc[validation_dates, base.TARGET_COL].reset_index(drop=True)
    initial_history = train_df[train_df[base.DATE_COL] < base.TRAIN_CUTOFF].set_index(base.DATE_COL)[base.TARGET_COL]
    volatility_threshold = spike1.compute_fixed_volatility_threshold(initial_history)
    predictions = recursive_predict_promo_context(
        model=model,
        model_type=model_type,
        prediction_dates=validation_dates,
        feature_columns=feature_columns,
        static_features=static_features,
        initial_revenue_history=initial_history,
        feature_medians=feature_medians,
        volatility_threshold=volatility_threshold,
    )
    metrics = spike1.evaluate_candidate("SPIKE_PROMO_CONTEXT", actual, predictions)
    return {
        "model": "SPIKE_PROMO_CONTEXT",
        "model_object": model,
        "model_type": model_type,
        "feature_columns": feature_columns,
        "feature_medians": feature_medians,
        "X_train": X_train,
        "y_train": y_train,
        "train_clean": train_clean,
        "validation_dates": validation_dates.reset_index(drop=True),
        "actual": actual,
        "predictions": predictions,
        "metrics": metrics,
    }


def train_full_model(
    model_table: pd.DataFrame,
    feature_columns: list[str],
    reporter: Reporter,
) -> dict[str, Any]:
    X_train, y_train, _, feature_medians = spike1.make_training_matrix(
        model_table,
        feature_columns,
        train_end_exclusive=None,
    )
    reporter.emit(
        f"Retraining spike+promo-context model on all rows: rows={len(X_train):,}, features={len(feature_columns)}"
    )
    model, model_type = spike1.train_variant_model(
        X_train,
        y_train,
        reporter,
        objective="quantile",
        alpha=0.70,
    )
    return {
        "model": "SPIKE_PROMO_CONTEXT",
        "model_object": model,
        "model_type": model_type,
        "feature_columns": feature_columns,
        "feature_medians": feature_medians,
        "X_train": X_train,
        "y_train": y_train,
    }


def forecast_submission(
    trained: dict[str, Any],
    static_features: pd.DataFrame,
    train_df: pd.DataFrame,
    sample_submission: pd.DataFrame,
    path: Path,
) -> pd.DataFrame:
    initial_history = train_df.set_index(base.DATE_COL)[base.TARGET_COL].sort_index()
    volatility_threshold = spike1.compute_fixed_volatility_threshold(initial_history)
    predictions = recursive_predict_promo_context(
        model=trained["model_object"],
        model_type=trained["model_type"],
        prediction_dates=sample_submission[base.DATE_COL],
        feature_columns=trained["feature_columns"],
        static_features=static_features,
        initial_revenue_history=initial_history,
        feature_medians=trained["feature_medians"],
        volatility_threshold=volatility_threshold,
    )
    cogs_ratio = base.estimate_cogs_ratio(train_df)
    return base.build_submission(sample_submission, predictions, cogs_ratio, path)


def save_validation_predictions(result: dict[str, Any]) -> pd.DataFrame:
    actual = result["actual"].to_numpy(dtype=float)
    predicted = np.asarray(result["predictions"], dtype=float)
    error = actual - predicted
    output = pd.DataFrame(
        {
            "Date": result["validation_dates"],
            "actual_Revenue": actual,
            "predicted_Revenue": predicted,
            "error": error,
            "abs_error": np.abs(error),
            "pct_error": np.where(actual != 0, error / actual, np.nan),
        }
    )
    output.to_csv(VALIDATION_PREDICTIONS_PATH, index=False)
    return output


def run() -> None:
    logger = setup_logging()
    reporter = Reporter(logger)

    reporter.emit("Spike Model With Promo Context")
    reporter.emit("==============================")
    reporter.emit("")

    reporter.emit("1. Load base data and build static promo-context features")
    train_df = base.load_train_data(base.TRAIN_DATA_PATH)
    sample_submission = base.load_sample_submission(base.SAMPLE_SUBMISSION_PATH)
    all_dates = pd.Series(
        pd.date_range(train_df[base.DATE_COL].min(), sample_submission[base.DATE_COL].max(), freq="D")
    )
    static_features = promo_duration.build_static_features_with_promo_duration(
        all_dates,
        train_df[base.DATE_COL].min(),
        logger,
    )
    model_table = build_promo_context_model_table(train_df, static_features)
    feature_columns = spike1.deduplicate_preserve_order(
        [feature for feature in spike1.load_top_full_features(limit=50) if feature in model_table.columns]
        + [feature for feature in spike1.SPIKE_FEATURES if feature in model_table.columns]
        + [feature for feature in PROMO_CONTEXT_BASE_FEATURES if feature in model_table.columns]
        + [feature for feature in PROMO_CONTEXT_FEATURES if feature in model_table.columns]
    )
    reporter.emit(f"Static feature table shape: {static_features.shape}")
    reporter.emit(f"Promo-context model table shape: {model_table.shape}")
    reporter.emit(f"Feature count: {len(feature_columns)}")

    reporter.emit("")
    reporter.emit("2. Recursive validation on 2022")
    validation_result = train_validate_model(
        model_table=model_table,
        static_features=static_features,
        train_df=train_df,
        feature_columns=feature_columns,
        reporter=reporter,
    )
    metrics = validation_result["metrics"]
    reporter.emit(
        f"Validation metrics: MAE={metrics['MAE']:,.2f} | RMSE={metrics['RMSE']:,.2f} | R2={metrics['R2']:.6f}"
    )
    reporter.emit(
        f"Top10 spike metrics: RMSE={metrics['top10_RMSE']:,.2f} | "
        f"underprediction={metrics['top10_underprediction']}/{metrics['top10_count']}"
    )
    reporter.emit(
        f"Improvement vs current spike RMSE ({CURRENT_SPIKE_RMSE:,.2f}): "
        f"{CURRENT_SPIKE_RMSE - metrics['RMSE']:,.2f}"
    )

    validation_output = save_validation_predictions(validation_result)
    reporter.emit(f"Saved validation predictions: {VALIDATION_PREDICTIONS_PATH} | shape={validation_output.shape}")

    importance = base.get_feature_importance(
        model=validation_result["model_object"],
        model_type=validation_result["model_type"],
        feature_columns=validation_result["feature_columns"],
        X_ref=validation_result["X_train"],
        y_ref=validation_result["y_train"],
        baseline_rmse=metrics["RMSE"],
    )
    importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    reporter.emit(f"Saved feature importance: {FEATURE_IMPORTANCE_PATH}")
    reporter.emit_frame("Top 20 features:", importance.head(20))

    if metrics["RMSE"] < CURRENT_SPIKE_RMSE:
        reporter.emit("")
        reporter.emit("3. Improvement confirmed, generating submission")
        final_model = train_full_model(
            model_table=model_table,
            feature_columns=feature_columns,
            reporter=reporter,
        )
        submission = forecast_submission(
            trained=final_model,
            static_features=static_features,
            train_df=train_df,
            sample_submission=sample_submission,
            path=SUBMISSION_PATH,
        )
        reporter.emit(
            f"Saved improved submission: {SUBMISSION_PATH} | rows={len(submission):,}"
        )
    else:
        reporter.emit("")
        reporter.emit("3. No RMSE improvement, submission not generated")

    reporter.save(METRICS_PATH)


if __name__ == "__main__":
    run()
