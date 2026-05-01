from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import train_final_model as base
import train_spike_aware_model as spike1


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "log"

SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
PRUNED_SUBMISSION_PATH = DATA_DIR / "submission_pruned_ensemble.csv"
SPIKE_COMPARISON_REFERENCE_PATH = DATA_DIR / "spike_model_comparison.csv"

SUBMISSION_Q70_PATH = DATA_DIR / "submission_spike_v2_q70.csv"
SUBMISSION_Q75_PATH = DATA_DIR / "submission_spike_v2_q75.csv"
SUBMISSION_Q65_PATH = DATA_DIR / "submission_spike_v2_q65.csv"
SUBMISSION_RESIDUAL_PATH = DATA_DIR / "submission_spike_v2_residual.csv"
BLEND_60_PATH = DATA_DIR / "submission_spike_v2_blend_60.csv"
BLEND_50_PATH = DATA_DIR / "submission_spike_v2_blend_50.csv"
BLEND_40_PATH = DATA_DIR / "submission_spike_v2_blend_40.csv"

COMPARISON_PATH = DATA_DIR / "spike_v2_model_comparison.csv"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "spike_v2_feature_importance.csv"
VALIDATION_PREDICTIONS_PATH = DATA_DIR / "spike_v2_validation_predictions.csv"

REPORT_PATH = LOG_DIR / "spike_v2_model_report.txt"
LOG_FILE = LOG_DIR / "train_spike_v2_model.log"

EPSILON = 1e-6
TOP_FEATURE_LIMIT = 50

SPIKE_V2_FEATURES = [
    "spike_strength_7_30",
    "spike_strength_14_30",
    "spike_strength_30_90",
    "spike_strength_365",
    "lag7_above_p80",
    "lag7_above_p90",
    "lag7_above_p95",
    "lag30_above_p80",
    "lag30_above_p90",
    "lag365_above_p80",
    "lag365_above_p90",
    "lag365_above_p95",
    "momentum_7_vs_30",
    "momentum_14_vs_60",
    "momentum_30_vs_90",
    "momentum_ratio_7_30",
    "spike_strength_365_x_month",
    "spike_strength_365_x_day_of_year",
    "lag365_above_p90_x_month",
    "lag365_above_p90_x_day_of_week",
    "promo_x_lag365_spike",
    "promo_discount_x_spike_strength365",
    "promo_count_x_spike_strength365",
]

REFERENCE_SPIKE_DEFAULTS = {
    "model": "CURRENT_SPIKE_MODEL",
    "MAE": 623_974.93,
    "RMSE": 842_278.60,
    "R2": 0.746779,
    "top10_RMSE": 1_567_649.30,
    "top10_mean_error": 1_040_226.36,
    "top10_underprediction": 28,
    "top10_count": 37,
    "top5_RMSE": 1_695_363.57,
    "top5_underprediction": 15,
    "top5_count": 19,
}


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

    def save(self, path: Path = REPORT_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        self.logger.info("Saved spike V2 report to %s", path)


def setup_logging(log_file: Path = LOG_FILE) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train_spike_v2_model")
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


def expanding_quantile(series: pd.Series, q: float) -> pd.Series:
    shifted = pd.to_numeric(series, errors="coerce").shift(1)
    return shifted.expanding(min_periods=1).quantile(q)


def load_reference_current_spike_metrics() -> dict[str, Any]:
    if SPIKE_COMPARISON_REFERENCE_PATH.exists():
        comparison = pd.read_csv(SPIKE_COMPARISON_REFERENCE_PATH)
        match = comparison[comparison["model"] == "SPIKE_VARIANT_B_QUANTILE"]
        if not match.empty:
            row = match.iloc[0].to_dict()
            row["model"] = "CURRENT_SPIKE_MODEL"
            return row
    return REFERENCE_SPIKE_DEFAULTS.copy()


def add_historical_spike_v2_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extend Spike V1 features with V2 normalized strength and extreme-risk features."""
    output = spike1.add_historical_spike_features(df).copy()

    p80_history = expanding_quantile(output[base.TARGET_COL], 0.80)
    p90_history = expanding_quantile(output[base.TARGET_COL], 0.90)
    p95_history = expanding_quantile(output[base.TARGET_COL], 0.95)

    output["spike_strength_7_30"] = (
        (output["lag_7"] - output["rolling_mean_30"]) / (output["revenue_roll_std_30"] + EPSILON)
    )
    output["spike_strength_14_30"] = (
        (output["lag_14"] - output["rolling_mean_30"]) / (output["revenue_roll_std_30"] + EPSILON)
    )
    output["spike_strength_30_90"] = (
        (output["lag_30"] - output["revenue_roll_mean_90"]) / (output["revenue_roll_std_90"] + EPSILON)
    )
    output["spike_strength_365"] = (
        (output["revenue_lag_365"] - output["revenue_roll_mean_365"]) / (output["revenue_roll_std_365"] + EPSILON)
    )

    output["lag7_above_p80"] = np.where(p80_history.notna(), (output["lag_7"] > p80_history).astype(int), np.nan)
    output["lag7_above_p90"] = np.where(p90_history.notna(), (output["lag_7"] > p90_history).astype(int), np.nan)
    output["lag7_above_p95"] = np.where(p95_history.notna(), (output["lag_7"] > p95_history).astype(int), np.nan)
    output["lag30_above_p80"] = np.where(p80_history.notna(), (output["lag_30"] > p80_history).astype(int), np.nan)
    output["lag30_above_p90"] = np.where(p90_history.notna(), (output["lag_30"] > p90_history).astype(int), np.nan)
    output["lag365_above_p80"] = np.where(
        p80_history.notna(),
        (output["revenue_lag_365"] > p80_history).astype(int),
        np.nan,
    )
    output["lag365_above_p90"] = np.where(
        p90_history.notna(),
        (output["revenue_lag_365"] > p90_history).astype(int),
        np.nan,
    )
    output["lag365_above_p95"] = np.where(
        p95_history.notna(),
        (output["revenue_lag_365"] > p95_history).astype(int),
        np.nan,
    )

    output["momentum_7_vs_30"] = output["lag_7"] - output["lag_30"]
    output["momentum_14_vs_60"] = output["lag_14"] - output["revenue_lag_60"]
    output["momentum_30_vs_90"] = output["lag_30"] - output["revenue_lag_90"]
    output["momentum_ratio_7_30"] = output["lag_7"] / (output["lag_30"] + EPSILON)

    output["spike_strength_365_x_month"] = output["spike_strength_365"] * output["month"]
    output["spike_strength_365_x_day_of_year"] = output["spike_strength_365"] * output["day_of_year"]
    output["lag365_above_p90_x_month"] = output["lag365_above_p90"] * output["month"]
    output["lag365_above_p90_x_day_of_week"] = output["lag365_above_p90"] * output["day_of_week"]

    promo_any = output["calendar_any_promo"] if "calendar_any_promo" in output.columns else 0
    promo_discount = output["calendar_avg_discount_value"] if "calendar_avg_discount_value" in output.columns else 0
    promo_count = output["calendar_active_promo_count"] if "calendar_active_promo_count" in output.columns else 0
    output["promo_x_lag365_spike"] = promo_any * output["lag365_above_p90"]
    output["promo_discount_x_spike_strength365"] = promo_discount * output["spike_strength_365"]
    output["promo_count_x_spike_strength365"] = promo_count * output["spike_strength_365"]

    return output


def build_spike_v2_model_table(train_df: pd.DataFrame, static_features: pd.DataFrame) -> pd.DataFrame:
    table = base.build_historical_model_table(train_df, static_features, include_business_lag365=False)
    return add_historical_spike_v2_features(table)


def compute_threshold_bundle(history: pd.Series) -> dict[str, float]:
    ordered = pd.to_numeric(history, errors="coerce").dropna().sort_index()
    if ordered.empty:
        return {
            "volatility_threshold": 0.0,
            "p80": 0.0,
            "p90": 0.0,
            "p95": 0.0,
        }

    return {
        "volatility_threshold": spike1.compute_fixed_volatility_threshold(ordered),
        "p80": float(ordered.quantile(0.80)),
        "p90": float(ordered.quantile(0.90)),
        "p95": float(ordered.quantile(0.95)),
    }


def compute_spike_v2_features_from_row(row: dict[str, float], thresholds: dict[str, float]) -> dict[str, float]:
    features = spike1.compute_spike_features_from_row(row, thresholds["volatility_threshold"])

    spike_strength_7_30 = (row.get("lag_7", np.nan) - row.get("rolling_mean_30", np.nan)) / (
        row.get("revenue_roll_std_30", np.nan) + EPSILON
    )
    spike_strength_14_30 = (row.get("lag_14", np.nan) - row.get("rolling_mean_30", np.nan)) / (
        row.get("revenue_roll_std_30", np.nan) + EPSILON
    )
    spike_strength_30_90 = (row.get("lag_30", np.nan) - row.get("revenue_roll_mean_90", np.nan)) / (
        row.get("revenue_roll_std_90", np.nan) + EPSILON
    )
    spike_strength_365 = (
        row.get("revenue_lag_365", np.nan) - row.get("revenue_roll_mean_365", np.nan)
    ) / (row.get("revenue_roll_std_365", np.nan) + EPSILON)

    lag7 = row.get("lag_7", np.nan)
    lag30 = row.get("lag_30", np.nan)
    lag365 = row.get("revenue_lag_365", np.nan)

    features.update(
        {
            "spike_strength_7_30": spike_strength_7_30,
            "spike_strength_14_30": spike_strength_14_30,
            "spike_strength_30_90": spike_strength_30_90,
            "spike_strength_365": spike_strength_365,
            "lag7_above_p80": float(int(pd.notna(lag7) and lag7 > thresholds["p80"])) if pd.notna(lag7) else np.nan,
            "lag7_above_p90": float(int(pd.notna(lag7) and lag7 > thresholds["p90"])) if pd.notna(lag7) else np.nan,
            "lag7_above_p95": float(int(pd.notna(lag7) and lag7 > thresholds["p95"])) if pd.notna(lag7) else np.nan,
            "lag30_above_p80": float(int(pd.notna(lag30) and lag30 > thresholds["p80"])) if pd.notna(lag30) else np.nan,
            "lag30_above_p90": float(int(pd.notna(lag30) and lag30 > thresholds["p90"])) if pd.notna(lag30) else np.nan,
            "lag365_above_p80": float(int(pd.notna(lag365) and lag365 > thresholds["p80"])) if pd.notna(lag365) else np.nan,
            "lag365_above_p90": float(int(pd.notna(lag365) and lag365 > thresholds["p90"])) if pd.notna(lag365) else np.nan,
            "lag365_above_p95": float(int(pd.notna(lag365) and lag365 > thresholds["p95"])) if pd.notna(lag365) else np.nan,
            "momentum_7_vs_30": lag7 - lag30 if pd.notna(lag7) and pd.notna(lag30) else np.nan,
            "momentum_14_vs_60": (
                row.get("lag_14", np.nan) - row.get("revenue_lag_60", np.nan)
                if pd.notna(row.get("lag_14", np.nan)) and pd.notna(row.get("revenue_lag_60", np.nan))
                else np.nan
            ),
            "momentum_30_vs_90": (
                lag30 - row.get("revenue_lag_90", np.nan)
                if pd.notna(lag30) and pd.notna(row.get("revenue_lag_90", np.nan))
                else np.nan
            ),
            "momentum_ratio_7_30": lag7 / (lag30 + EPSILON) if pd.notna(lag7) and pd.notna(lag30) else np.nan,
            "spike_strength_365_x_month": spike_strength_365 * row.get("month", np.nan),
            "spike_strength_365_x_day_of_year": spike_strength_365 * row.get("day_of_year", np.nan),
            "lag365_above_p90_x_month": (
                float(int(pd.notna(lag365) and lag365 > thresholds["p90"])) * row.get("month", np.nan)
            ),
            "lag365_above_p90_x_day_of_week": (
                float(int(pd.notna(lag365) and lag365 > thresholds["p90"])) * row.get("day_of_week", np.nan)
            ),
            "promo_x_lag365_spike": row.get("calendar_any_promo", 0.0)
            * float(int(pd.notna(lag365) and lag365 > thresholds["p90"])),
            "promo_discount_x_spike_strength365": row.get("calendar_avg_discount_value", 0.0) * spike_strength_365,
            "promo_count_x_spike_strength365": row.get("calendar_active_promo_count", 0.0) * spike_strength_365,
        }
    )
    return features


def make_training_matrix(
    model_table: pd.DataFrame,
    feature_columns: list[str],
    train_end_exclusive: pd.Timestamp | None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    return spike1.make_training_matrix(model_table, feature_columns, train_end_exclusive)


def recursive_predict_v2(
    model: Any,
    model_type: str,
    prediction_dates: pd.Series,
    feature_columns: list[str],
    static_features: pd.DataFrame,
    initial_revenue_history: pd.Series,
    feature_medians: pd.Series,
    thresholds: dict[str, float],
) -> np.ndarray:
    del model_type
    static_by_date = static_features.set_index(base.DATE_COL).sort_index()
    history = pd.to_numeric(initial_revenue_history, errors="coerce").sort_index().copy()
    predictions: list[float] = []

    for forecast_date in pd.to_datetime(prediction_dates):
        if forecast_date not in static_by_date.index:
            raise ValueError(f"Missing static features for forecast date {forecast_date.date()}")

        row = static_by_date.loc[forecast_date].to_dict()
        row.update(base.compute_revenue_features_from_history(history, forecast_date))
        row.update(compute_spike_v2_features_from_row(row, thresholds))

        X_row = pd.DataFrame([row], columns=feature_columns)
        X_row = X_row.apply(pd.to_numeric, errors="coerce").fillna(feature_medians).fillna(0)
        prediction = float(model.predict(X_row)[0])
        prediction = max(0.0, prediction)
        predictions.append(prediction)
        history.loc[forecast_date] = prediction

    return np.asarray(predictions, dtype=float)


def train_quantile_variant(
    variant_name: str,
    alpha: float,
    model_table: pd.DataFrame,
    static_features: pd.DataFrame,
    train_df: pd.DataFrame,
    feature_columns: list[str],
    reporter: Reporter,
) -> dict[str, Any]:
    X_train, y_train, train_clean, feature_medians = make_training_matrix(
        model_table,
        feature_columns,
        base.TRAIN_CUTOFF,
    )
    reporter.emit(
        f"Training {variant_name}: rows={len(X_train):,}, features={len(feature_columns)}, alpha={alpha:.2f}"
    )
    model, model_type = spike1.train_variant_model(
        X_train,
        y_train,
        reporter,
        objective="quantile",
        alpha=alpha,
    )

    validation_dates = train_df[
        (train_df[base.DATE_COL] >= base.TRAIN_CUTOFF) & (train_df[base.DATE_COL] <= base.VALIDATION_END)
    ][base.DATE_COL]
    actual = train_df.set_index(base.DATE_COL).loc[validation_dates, base.TARGET_COL].reset_index(drop=True)
    initial_history = train_df[train_df[base.DATE_COL] < base.TRAIN_CUTOFF].set_index(base.DATE_COL)[base.TARGET_COL]
    thresholds = compute_threshold_bundle(initial_history)
    predictions = recursive_predict_v2(
        model,
        model_type,
        validation_dates,
        feature_columns,
        static_features,
        initial_history,
        feature_medians,
        thresholds,
    )
    metrics = spike1.evaluate_candidate(variant_name, actual, predictions)
    return {
        "model": variant_name,
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
        "thresholds": thresholds,
        "alpha": alpha,
    }


def train_residual_variant(
    model_table: pd.DataFrame,
    static_features: pd.DataFrame,
    train_df: pd.DataFrame,
    base_feature_columns: list[str],
    residual_feature_columns: list[str],
    reporter: Reporter,
) -> dict[str, Any]:
    base_X, base_y, base_clean, base_medians = make_training_matrix(
        model_table,
        base_feature_columns,
        base.TRAIN_CUTOFF,
    )
    reporter.emit(
        f"Training V2D base model: rows={len(base_X):,}, features={len(base_feature_columns)}"
    )
    base_model, base_model_type = spike1.train_variant_model(
        base_X,
        base_y,
        reporter,
        objective="regression",
    )

    train_subset = model_table[model_table[base.DATE_COL] < base.TRAIN_CUTOFF].copy()
    union_features = spike1.deduplicate_preserve_order(base_feature_columns + residual_feature_columns)
    residual_clean = train_subset.dropna(subset=union_features + [base.TARGET_COL]).reset_index(drop=True)
    residual_X = residual_clean[residual_feature_columns].copy()
    residual_medians = residual_X.median(numeric_only=True)
    base_pred_on_residual_train = np.asarray(base_model.predict(residual_clean[base_feature_columns]), dtype=float)
    residual_target = residual_clean[base.TARGET_COL].to_numpy(dtype=float) - base_pred_on_residual_train
    reporter.emit(
        f"Training V2D residual model: rows={len(residual_X):,}, features={len(residual_feature_columns)}"
    )
    residual_model, residual_model_type = spike1.train_variant_model(
        residual_X,
        pd.Series(residual_target),
        reporter,
        objective="regression",
    )

    validation_dates = train_df[
        (train_df[base.DATE_COL] >= base.TRAIN_CUTOFF) & (train_df[base.DATE_COL] <= base.VALIDATION_END)
    ][base.DATE_COL]
    actual = train_df.set_index(base.DATE_COL).loc[validation_dates, base.TARGET_COL].reset_index(drop=True)
    initial_history = train_df[train_df[base.DATE_COL] < base.TRAIN_CUTOFF].set_index(base.DATE_COL)[base.TARGET_COL]
    thresholds = compute_threshold_bundle(initial_history)
    predictions = recursive_predict_residual(
        base_model=base_model,
        base_model_type=base_model_type,
        residual_model=residual_model,
        residual_model_type=residual_model_type,
        prediction_dates=validation_dates,
        base_feature_columns=base_feature_columns,
        residual_feature_columns=residual_feature_columns,
        static_features=static_features,
        initial_revenue_history=initial_history,
        base_feature_medians=base_medians,
        residual_feature_medians=residual_medians,
        thresholds=thresholds,
    )
    metrics = spike1.evaluate_candidate("SPIKE_V2_RESIDUAL", actual, predictions)
    return {
        "model": "SPIKE_V2_RESIDUAL",
        "base_model": base_model,
        "base_model_type": base_model_type,
        "residual_model": residual_model,
        "residual_model_type": residual_model_type,
        "base_feature_columns": base_feature_columns,
        "residual_feature_columns": residual_feature_columns,
        "base_feature_medians": base_medians,
        "residual_feature_medians": residual_medians,
        "base_X_train": base_X,
        "base_y_train": base_y,
        "residual_X_train": residual_X,
        "residual_y_train": pd.Series(residual_target),
        "validation_dates": validation_dates.reset_index(drop=True),
        "actual": actual,
        "predictions": predictions,
        "metrics": metrics,
        "thresholds": thresholds,
        "base_train_clean": base_clean,
        "residual_train_clean": residual_clean,
    }


def recursive_predict_residual(
    base_model: Any,
    base_model_type: str,
    residual_model: Any,
    residual_model_type: str,
    prediction_dates: pd.Series,
    base_feature_columns: list[str],
    residual_feature_columns: list[str],
    static_features: pd.DataFrame,
    initial_revenue_history: pd.Series,
    base_feature_medians: pd.Series,
    residual_feature_medians: pd.Series,
    thresholds: dict[str, float],
) -> np.ndarray:
    del base_model_type, residual_model_type
    static_by_date = static_features.set_index(base.DATE_COL).sort_index()
    history = pd.to_numeric(initial_revenue_history, errors="coerce").sort_index().copy()
    predictions: list[float] = []

    for forecast_date in pd.to_datetime(prediction_dates):
        if forecast_date not in static_by_date.index:
            raise ValueError(f"Missing static features for forecast date {forecast_date.date()}")

        row = static_by_date.loc[forecast_date].to_dict()
        row.update(base.compute_revenue_features_from_history(history, forecast_date))
        row.update(compute_spike_v2_features_from_row(row, thresholds))

        base_row = pd.DataFrame([row], columns=base_feature_columns)
        base_row = base_row.apply(pd.to_numeric, errors="coerce").fillna(base_feature_medians).fillna(0)
        residual_row = pd.DataFrame([row], columns=residual_feature_columns)
        residual_row = residual_row.apply(pd.to_numeric, errors="coerce").fillna(residual_feature_medians).fillna(0)

        base_prediction = float(base_model.predict(base_row)[0])
        residual_prediction = float(residual_model.predict(residual_row)[0])
        final_prediction = max(0.0, base_prediction + residual_prediction)
        predictions.append(final_prediction)
        history.loc[forecast_date] = final_prediction

    return np.asarray(predictions, dtype=float)


def train_full_quantile_variant(
    variant_name: str,
    alpha: float,
    model_table: pd.DataFrame,
    feature_columns: list[str],
    reporter: Reporter,
) -> dict[str, Any]:
    X_train, y_train, _, feature_medians = make_training_matrix(model_table, feature_columns, None)
    reporter.emit(
        f"Retraining {variant_name} on all rows: rows={len(X_train):,}, features={len(feature_columns)}, alpha={alpha:.2f}"
    )
    model, model_type = spike1.train_variant_model(
        X_train,
        y_train,
        reporter,
        objective="quantile",
        alpha=alpha,
    )
    return {
        "model": variant_name,
        "model_object": model,
        "model_type": model_type,
        "feature_columns": feature_columns,
        "feature_medians": feature_medians,
        "X_train": X_train,
        "y_train": y_train,
    }


def train_full_residual_variant(
    model_table: pd.DataFrame,
    base_feature_columns: list[str],
    residual_feature_columns: list[str],
    reporter: Reporter,
) -> dict[str, Any]:
    base_X, base_y, _, base_medians = make_training_matrix(model_table, base_feature_columns, None)
    reporter.emit(
        f"Retraining V2D base model on all rows: rows={len(base_X):,}, features={len(base_feature_columns)}"
    )
    base_model, base_model_type = spike1.train_variant_model(
        base_X,
        base_y,
        reporter,
        objective="regression",
    )

    union_features = spike1.deduplicate_preserve_order(base_feature_columns + residual_feature_columns)
    residual_clean = model_table.dropna(subset=union_features + [base.TARGET_COL]).reset_index(drop=True)
    residual_X = residual_clean[residual_feature_columns].copy()
    residual_medians = residual_X.median(numeric_only=True)
    base_pred_on_train = np.asarray(base_model.predict(residual_clean[base_feature_columns]), dtype=float)
    residual_target = residual_clean[base.TARGET_COL].to_numpy(dtype=float) - base_pred_on_train
    reporter.emit(
        f"Retraining V2D residual model on all rows: rows={len(residual_X):,}, features={len(residual_feature_columns)}"
    )
    residual_model, residual_model_type = spike1.train_variant_model(
        residual_X,
        pd.Series(residual_target),
        reporter,
        objective="regression",
    )
    return {
        "model": "SPIKE_V2_RESIDUAL",
        "base_model": base_model,
        "base_model_type": base_model_type,
        "residual_model": residual_model,
        "residual_model_type": residual_model_type,
        "base_feature_columns": base_feature_columns,
        "residual_feature_columns": residual_feature_columns,
        "base_feature_medians": base_medians,
        "residual_feature_medians": residual_medians,
        "base_X_train": base_X,
        "base_y_train": base_y,
        "residual_X_train": residual_X,
        "residual_y_train": pd.Series(residual_target),
    }


def forecast_quantile_submission(
    trained: dict[str, Any],
    static_features: pd.DataFrame,
    train_df: pd.DataFrame,
    sample_submission: pd.DataFrame,
    path: Path,
) -> pd.DataFrame:
    initial_history = train_df.set_index(base.DATE_COL)[base.TARGET_COL].sort_index()
    thresholds = compute_threshold_bundle(initial_history)
    predictions = recursive_predict_v2(
        trained["model_object"],
        trained["model_type"],
        sample_submission[base.DATE_COL],
        trained["feature_columns"],
        static_features,
        initial_history,
        trained["feature_medians"],
        thresholds,
    )
    cogs_ratio = base.estimate_cogs_ratio(train_df)
    return base.build_submission(sample_submission, predictions, cogs_ratio, path)


def forecast_residual_submission(
    trained: dict[str, Any],
    static_features: pd.DataFrame,
    train_df: pd.DataFrame,
    sample_submission: pd.DataFrame,
    path: Path,
) -> pd.DataFrame:
    initial_history = train_df.set_index(base.DATE_COL)[base.TARGET_COL].sort_index()
    thresholds = compute_threshold_bundle(initial_history)
    predictions = recursive_predict_residual(
        base_model=trained["base_model"],
        base_model_type=trained["base_model_type"],
        residual_model=trained["residual_model"],
        residual_model_type=trained["residual_model_type"],
        prediction_dates=sample_submission[base.DATE_COL],
        base_feature_columns=trained["base_feature_columns"],
        residual_feature_columns=trained["residual_feature_columns"],
        static_features=static_features,
        initial_revenue_history=initial_history,
        base_feature_medians=trained["base_feature_medians"],
        residual_feature_medians=trained["residual_feature_medians"],
        thresholds=thresholds,
    )
    cogs_ratio = base.estimate_cogs_ratio(train_df)
    return base.build_submission(sample_submission, predictions, cogs_ratio, path)


def validate_submission_frame(
    path: Path,
    sample_submission: pd.DataFrame,
) -> dict[str, Any]:
    submission = pd.read_csv(path, parse_dates=[base.DATE_COL])
    submission[base.DATE_COL] = pd.to_datetime(submission[base.DATE_COL], errors="coerce").dt.normalize()
    same_order = submission[base.DATE_COL].reset_index(drop=True).equals(
        sample_submission[base.DATE_COL].reset_index(drop=True)
    )
    exact_columns = list(submission.columns) == [base.DATE_COL, base.TARGET_COL, base.COGS_COL]
    missing_count = int(submission.isna().sum().sum())
    negative_count = int(
        ((pd.to_numeric(submission[base.TARGET_COL], errors="coerce") < 0) |
         (pd.to_numeric(submission[base.COGS_COL], errors="coerce") < 0)).sum()
    )
    return {
        "file": path.name,
        "rows": int(len(submission)),
        "exact_columns": exact_columns,
        "date_order_matches": bool(same_order),
        "missing_values": missing_count,
        "negative_values": negative_count,
    }


def build_importance_frame(
    quantile_models: list[dict[str, Any]],
    residual_model: dict[str, Any],
    validation_results: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for trained in quantile_models:
        model_name = trained["model"]
        validation_rmse = validation_results[model_name]["metrics"]["RMSE"]
        importance = base.get_feature_importance(
            trained["model_object"],
            trained["model_type"],
            trained["feature_columns"],
            trained["X_train"],
            trained["y_train"],
            validation_rmse,
        ).copy()
        importance.insert(0, "component", "model")
        importance.insert(0, "model", model_name)
        frames.append(importance)

    residual_rmse = validation_results["SPIKE_V2_RESIDUAL"]["metrics"]["RMSE"]
    base_importance = base.get_feature_importance(
        residual_model["base_model"],
        residual_model["base_model_type"],
        residual_model["base_feature_columns"],
        residual_model["base_X_train"],
        residual_model["base_y_train"],
        residual_rmse,
    ).copy()
    base_importance.insert(0, "component", "base")
    base_importance.insert(0, "model", "SPIKE_V2_RESIDUAL")
    frames.append(base_importance)

    residual_importance = base.get_feature_importance(
        residual_model["residual_model"],
        residual_model["residual_model_type"],
        residual_model["residual_feature_columns"],
        residual_model["residual_X_train"],
        residual_model["residual_y_train"],
        residual_rmse,
    ).copy()
    residual_importance.insert(0, "component", "residual")
    residual_importance.insert(0, "model", "SPIKE_V2_RESIDUAL")
    frames.append(residual_importance)

    return pd.concat(frames, ignore_index=True)


def save_validation_predictions(
    result: dict[str, Any],
    path: Path = VALIDATION_PREDICTIONS_PATH,
) -> pd.DataFrame:
    actual = result["actual"].to_numpy(dtype=float)
    predicted = np.asarray(result["predictions"], dtype=float)
    error = actual - predicted
    output = pd.DataFrame(
        {
            base.DATE_COL: result["validation_dates"],
            "actual_Revenue": actual,
            "predicted_Revenue": predicted,
            "selected_model": result["model"],
            "error": error,
            "abs_error": np.abs(error),
            "pct_error": np.where(actual != 0, error / actual, np.nan),
        }
    )
    output.to_csv(path, index=False)
    return output


def emit_metrics(reporter: Reporter, title: str, metrics: dict[str, Any]) -> None:
    reporter.emit(title)
    reporter.emit(f"MAE={metrics['MAE']:,.2f} | RMSE={metrics['RMSE']:,.2f} | R2={metrics['R2']:.6f}")
    reporter.emit(
        f"Top10 RMSE={metrics['top10_RMSE']:,.2f} | Top10 underprediction="
        f"{metrics['top10_underprediction']}/{metrics['top10_count']}"
    )
    reporter.emit(
        f"Top5 RMSE={metrics['top5_RMSE']:,.2f} | Top5 underprediction="
        f"{metrics['top5_underprediction']}/{metrics['top5_count']}"
    )


def run_experiment() -> None:
    logger = setup_logging()
    reporter = Reporter(logger)

    reporter.emit("Spike V2 Forecasting Model")
    reporter.emit("=========================")
    reporter.emit("")

    reporter.emit("1. Load safe base data")
    train_df = base.load_train_data(base.TRAIN_DATA_PATH)
    sample_submission = base.load_sample_submission(SAMPLE_SUBMISSION_PATH)
    all_dates = pd.Series(
        pd.date_range(train_df[base.DATE_COL].min(), sample_submission[base.DATE_COL].max(), freq="D")
    )
    static_features = base.build_static_features(all_dates, train_df[base.DATE_COL].min(), logger)
    model_table = build_spike_v2_model_table(train_df, static_features)

    base_pruned_features = [
        feature for feature in spike1.load_top_full_features(limit=TOP_FEATURE_LIMIT)
        if feature in model_table.columns
    ]
    v2_feature_columns = spike1.deduplicate_preserve_order(
        base_pruned_features + spike1.SPIKE_FEATURES + SPIKE_V2_FEATURES
    )
    reporter.emit(f"Static feature table shape: {static_features.shape}")
    reporter.emit(f"Spike V2 model table shape: {model_table.shape}")
    reporter.emit(f"Base pruned features: {len(base_pruned_features)}")
    reporter.emit(f"Spike V2 feature columns: {len(v2_feature_columns)}")

    reporter.emit("")
    reporter.emit("2. Load current reference metrics")
    pruned_actual, pruned_pred = spike1.load_baseline_validation()
    current_pruned_metrics = spike1.evaluate_candidate("CURRENT_PRUNED_ENSEMBLE", pruned_actual, pruned_pred)
    current_spike_metrics = load_reference_current_spike_metrics()
    emit_metrics(reporter, "Current pruned ensemble:", current_pruned_metrics)
    emit_metrics(reporter, "Current spike model:", current_spike_metrics)

    reporter.emit("")
    reporter.emit("3. Train V2 variants with recursive validation 2022")
    validation_results: dict[str, dict[str, Any]] = {}

    v2a = train_quantile_variant(
        "SPIKE_V2_Q70",
        0.70,
        model_table,
        static_features,
        train_df,
        v2_feature_columns,
        reporter,
    )
    validation_results[v2a["model"]] = v2a
    emit_metrics(reporter, "V2A metrics:", v2a["metrics"])

    v2b = train_quantile_variant(
        "SPIKE_V2_Q75",
        0.75,
        model_table,
        static_features,
        train_df,
        v2_feature_columns,
        reporter,
    )
    validation_results[v2b["model"]] = v2b
    emit_metrics(reporter, "V2B metrics:", v2b["metrics"])

    v2c = train_quantile_variant(
        "SPIKE_V2_Q65",
        0.65,
        model_table,
        static_features,
        train_df,
        v2_feature_columns,
        reporter,
    )
    validation_results[v2c["model"]] = v2c
    emit_metrics(reporter, "V2C metrics:", v2c["metrics"])

    v2d = train_residual_variant(
        model_table,
        static_features,
        train_df,
        base_pruned_features,
        v2_feature_columns,
        reporter,
    )
    validation_results[v2d["model"]] = v2d
    emit_metrics(reporter, "V2D metrics:", v2d["metrics"])

    comparison_rows = [
        current_pruned_metrics,
        current_spike_metrics,
        v2a["metrics"],
        v2b["metrics"],
        v2c["metrics"],
        v2d["metrics"],
    ]
    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        ["RMSE", "top10_RMSE", "top10_underprediction", "MAE"],
        ascending=[True, True, True, True],
    )
    comparison_df.to_csv(COMPARISON_PATH, index=False)

    best_variant_name = comparison_df[
        comparison_df["model"].isin(["SPIKE_V2_Q70", "SPIKE_V2_Q75", "SPIKE_V2_Q65", "SPIKE_V2_RESIDUAL"])
    ].iloc[0]["model"]
    best_variant = validation_results[best_variant_name]
    save_validation_predictions(best_variant, VALIDATION_PREDICTIONS_PATH)
    reporter.emit(f"Best V2 model by validation RMSE: {best_variant_name}")

    reporter.emit("")
    reporter.emit("4. Retrain best-effort full V2 models for future submissions")
    full_v2a = train_full_quantile_variant("SPIKE_V2_Q70", 0.70, model_table, v2_feature_columns, reporter)
    full_v2b = train_full_quantile_variant("SPIKE_V2_Q75", 0.75, model_table, v2_feature_columns, reporter)
    full_v2c = train_full_quantile_variant("SPIKE_V2_Q65", 0.65, model_table, v2_feature_columns, reporter)
    full_v2d = train_full_residual_variant(model_table, base_pruned_features, v2_feature_columns, reporter)

    submission_q70 = forecast_quantile_submission(full_v2a, static_features, train_df, sample_submission, SUBMISSION_Q70_PATH)
    submission_q75 = forecast_quantile_submission(full_v2b, static_features, train_df, sample_submission, SUBMISSION_Q75_PATH)
    submission_q65 = forecast_quantile_submission(full_v2c, static_features, train_df, sample_submission, SUBMISSION_Q65_PATH)
    submission_residual = forecast_residual_submission(full_v2d, static_features, train_df, sample_submission, SUBMISSION_RESIDUAL_PATH)

    importance_df = build_importance_frame([full_v2a, full_v2b, full_v2c], full_v2d, validation_results)
    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    reporter.emit("")
    reporter.emit("5. Build blends around the best V2 variant")
    pruned_submission = spike1.load_submission(PRUNED_SUBMISSION_PATH, sample_submission)
    variant_submission_lookup = {
        "SPIKE_V2_Q70": submission_q70,
        "SPIKE_V2_Q75": submission_q75,
        "SPIKE_V2_Q65": submission_q65,
        "SPIKE_V2_RESIDUAL": submission_residual,
    }
    variant_submission_name_lookup = {
        "SPIKE_V2_Q70": SUBMISSION_Q70_PATH.name,
        "SPIKE_V2_Q75": SUBMISSION_Q75_PATH.name,
        "SPIKE_V2_Q65": SUBMISSION_Q65_PATH.name,
        "SPIKE_V2_RESIDUAL": SUBMISSION_RESIDUAL_PATH.name,
    }
    best_variant_submission = variant_submission_lookup[best_variant_name]

    blend_60 = spike1.blend_submissions(
        sample_submission,
        {"PRUNED": pruned_submission, "V2": best_variant_submission},
        {"PRUNED": 0.40, "V2": 0.60},
    )
    blend_60.to_csv(BLEND_60_PATH, index=False)
    blend_50 = spike1.blend_submissions(
        sample_submission,
        {"PRUNED": pruned_submission, "V2": best_variant_submission},
        {"PRUNED": 0.50, "V2": 0.50},
    )
    blend_50.to_csv(BLEND_50_PATH, index=False)
    blend_40 = spike1.blend_submissions(
        sample_submission,
        {"PRUNED": pruned_submission, "V2": best_variant_submission},
        {"PRUNED": 0.60, "V2": 0.40},
    )
    blend_40.to_csv(BLEND_40_PATH, index=False)

    validation_frames = pd.DataFrame(
        [
            validate_submission_frame(SUBMISSION_Q70_PATH, sample_submission),
            validate_submission_frame(SUBMISSION_Q75_PATH, sample_submission),
            validate_submission_frame(SUBMISSION_Q65_PATH, sample_submission),
            validate_submission_frame(SUBMISSION_RESIDUAL_PATH, sample_submission),
            validate_submission_frame(BLEND_60_PATH, sample_submission),
            validate_submission_frame(BLEND_50_PATH, sample_submission),
            validate_submission_frame(BLEND_40_PATH, sample_submission),
        ]
    )

    reporter.emit("")
    reporter.emit("6. Final summary")
    reporter.emit_frame("V2 comparison table:", comparison_df)
    reporter.emit(
        "Spike-specific improvement vs old spike model: "
        f"Top10 RMSE delta={current_spike_metrics['top10_RMSE'] - best_variant['metrics']['top10_RMSE']:,.2f}, "
        f"Top10 underprediction delta={current_spike_metrics['top10_underprediction'] - best_variant['metrics']['top10_underprediction']}"
    )
    reporter.emit_frame(
        f"Top 30 feature importance for {best_variant_name}:",
        importance_df[importance_df["model"] == best_variant_name]
        .sort_values(["importance_gain", "importance_split"], ascending=False)
        .head(30),
    )
    reporter.emit_frame("Submission validation checks:", validation_frames)
    reporter.emit(f"Saved comparison file: {COMPARISON_PATH}")
    reporter.emit(f"Saved feature importance file: {FEATURE_IMPORTANCE_PATH}")
    reporter.emit(f"Saved validation predictions file: {VALIDATION_PREDICTIONS_PATH}")
    reporter.emit(
        "Created submissions: "
        f"{SUBMISSION_Q70_PATH.name}, {SUBMISSION_Q75_PATH.name}, {SUBMISSION_Q65_PATH.name}, "
        f"{SUBMISSION_RESIDUAL_PATH.name}, {BLEND_60_PATH.name}, {BLEND_50_PATH.name}, {BLEND_40_PATH.name}"
    )
    reporter.emit(
        "Recommended first upload order: "
        f"{BLEND_50_PATH.name} -> {BLEND_60_PATH.name} -> {BLEND_40_PATH.name} -> "
        f"{variant_submission_name_lookup[best_variant_name]}"
    )
    reporter.emit(
        "Leakage confirmation: Spike V2 uses only lagged Revenue, fixed thresholds from training history, "
        "calendar, promotion schedule, and inventory as-of features. Validation and future prediction remain recursive."
    )
    reporter.save(REPORT_PATH)


if __name__ == "__main__":
    run_experiment()
