from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import train_final_model as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "log"

SUBMISSION_PATH = DATA_DIR / "submission_promo_duration.csv"
VALIDATION_PREDICTIONS_PATH = DATA_DIR / "final_promo_duration_validation_predictions.csv"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "final_promo_duration_feature_importance.csv"
METRICS_PATH = LOG_DIR / "final_promo_duration_metrics.txt"
LOG_FILE = LOG_DIR / "train_final_model_promo_duration.log"

PREVIOUS_FINAL_MODEL_A = {
    "MAE": 695_337.54,
    "RMSE": 985_315.32,
    "R2": 0.653472,
}

PROMO_DURATION_FEATURES = [
    "promo_avg_duration_days",
    "promo_max_duration_days",
    "promo_min_duration_days",
    "promo_avg_day_number",
    "promo_min_day_number",
    "promo_max_day_number",
    "promo_avg_days_remaining",
    "promo_min_days_remaining",
    "promo_max_days_remaining",
    "promo_avg_progress_ratio",
    "promo_min_progress_ratio",
    "promo_max_progress_ratio",
    "promo_is_first_3_days",
    "promo_is_last_3_days",
    "promo_is_first_7_days",
    "promo_is_last_7_days",
    "promo_short_count",
    "promo_medium_count",
    "promo_long_count",
    "promo_discount_duration_intensity",
    "promo_discount_progress_intensity",
    "promo_end_urgency_intensity",
]

EXTENDED_PROMOTION_FEATURES = base.PROMOTION_FEATURES + PROMO_DURATION_FEATURES
MODEL_A_FEATURES = (
    base.CALENDAR_FEATURES
    + base.REVENUE_FEATURES
    + EXTENDED_PROMOTION_FEATURES
    + base.INVENTORY_FEATURES
)
MODEL_B_FEATURES = MODEL_A_FEATURES + base.BUSINESS_LAG365_FEATURES


class RunReporter:
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

    def save_metrics(self, path: Path = METRICS_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        self.logger.info("Saved metrics report to %s", path)


def setup_logging(log_file: Path = LOG_FILE) -> logging.Logger:
    """Configure simple file logging for the promo-duration run."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train_final_model_promo_duration")
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


def get_feature_columns(model_variant: str) -> list[str]:
    """Return feature list for promo-duration Model A or B."""
    if model_variant == "A":
        return MODEL_A_FEATURES.copy()
    if model_variant == "B":
        return MODEL_B_FEATURES.copy()
    raise ValueError(f"Unknown model variant: {model_variant}")


def _safe_category_specific(series: pd.Series) -> pd.Series:
    text = series.astype("string")
    return text.notna() & text.str.strip().ne("")


def build_promotion_calendar_with_duration(
    dates: pd.Series,
    promotions_path: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Build promotion-calendar plus forecast-safe duration/phase features."""
    calendar = pd.DataFrame({base.DATE_COL: pd.to_datetime(dates).sort_values().unique()})
    for feature in EXTENDED_PROMOTION_FEATURES:
        calendar[feature] = 0.0

    if not promotions_path.exists():
        logger.warning("promotions.csv not found at %s; promotion features are zero", promotions_path)
        return calendar

    promotions = pd.read_csv(promotions_path, low_memory=False)
    required = {"promo_id", "start_date", "end_date"}
    if not required.issubset(promotions.columns):
        logger.warning("promotions.csv missing required columns; promotion features are zero")
        return calendar

    promotions["start_date"] = pd.to_datetime(promotions["start_date"], errors="coerce").dt.normalize()
    promotions["end_date"] = pd.to_datetime(promotions["end_date"], errors="coerce").dt.normalize()
    promotions = promotions.dropna(subset=["start_date", "end_date"]).copy()
    promotions["duration_days"] = (promotions["end_date"] - promotions["start_date"]).dt.days + 1
    promotions = promotions[promotions["duration_days"] > 0].copy()

    if promotions.empty:
        return calendar

    promotions["discount_value"] = pd.to_numeric(
        promotions.get("discount_value", 0),
        errors="coerce",
    ).fillna(0)
    promotions["stackable_flag_numeric"] = (
        base._stackable_to_int(promotions["stackable_flag"])
        if "stackable_flag" in promotions.columns
        else 0
    )
    promotions["category_specific"] = (
        _safe_category_specific(promotions["applicable_category"]).astype(int)
        if "applicable_category" in promotions.columns
        else 0
    )
    if "promo_type" in promotions.columns:
        promo_type = promotions["promo_type"].astype(str).str.lower()
        promotions["percentage_promo"] = promo_type.eq("percentage").astype(int)
        promotions["fixed_promo"] = promo_type.eq("fixed").astype(int)
    else:
        promotions["percentage_promo"] = 0
        promotions["fixed_promo"] = 0

    rows: list[dict[str, Any]] = []
    min_date = calendar[base.DATE_COL].min()
    max_date = calendar[base.DATE_COL].max()

    for row in promotions.itertuples(index=False):
        start_date = max(row.start_date, min_date)
        end_date = min(row.end_date, max_date)
        if start_date > end_date:
            continue

        for active_date in pd.date_range(start_date, end_date, freq="D"):
            promo_day_number = (active_date - row.start_date).days + 1
            promo_days_remaining = (row.end_date - active_date).days
            progress_ratio = promo_day_number / row.duration_days
            rows.append(
                {
                    base.DATE_COL: active_date,
                    "promo_id": row.promo_id,
                    "discount_value": row.discount_value,
                    "stackable_flag_numeric": row.stackable_flag_numeric,
                    "category_specific": row.category_specific,
                    "percentage_promo": row.percentage_promo,
                    "fixed_promo": row.fixed_promo,
                    "duration_days": row.duration_days,
                    "promo_day_number": promo_day_number,
                    "promo_days_remaining": promo_days_remaining,
                    "promo_progress_ratio": progress_ratio,
                    "promo_is_first_3_days": int(promo_day_number <= 3),
                    "promo_is_last_3_days": int(promo_days_remaining <= 2),
                    "promo_is_first_7_days": int(promo_day_number <= 7),
                    "promo_is_last_7_days": int(promo_days_remaining <= 6),
                    "promo_short": int(row.duration_days <= 7),
                    "promo_medium": int(8 <= row.duration_days <= 30),
                    "promo_long": int(row.duration_days > 30),
                }
            )

    if not rows:
        return calendar

    expanded = pd.DataFrame(rows)
    daily = (
        expanded.groupby(base.DATE_COL, as_index=False)
        .agg(
            calendar_active_promo_count=("promo_id", "nunique"),
            calendar_avg_discount_value=("discount_value", "mean"),
            calendar_max_discount_value=("discount_value", "max"),
            calendar_stackable_promo_count=("stackable_flag_numeric", "sum"),
            calendar_has_stackable_promo=("stackable_flag_numeric", "max"),
            calendar_has_category_specific_promo=("category_specific", "max"),
            calendar_percentage_promo_count=("percentage_promo", "sum"),
            calendar_fixed_promo_count=("fixed_promo", "sum"),
            promo_avg_duration_days=("duration_days", "mean"),
            promo_max_duration_days=("duration_days", "max"),
            promo_min_duration_days=("duration_days", "min"),
            promo_avg_day_number=("promo_day_number", "mean"),
            promo_min_day_number=("promo_day_number", "min"),
            promo_max_day_number=("promo_day_number", "max"),
            promo_avg_days_remaining=("promo_days_remaining", "mean"),
            promo_min_days_remaining=("promo_days_remaining", "min"),
            promo_max_days_remaining=("promo_days_remaining", "max"),
            promo_avg_progress_ratio=("promo_progress_ratio", "mean"),
            promo_min_progress_ratio=("promo_progress_ratio", "min"),
            promo_max_progress_ratio=("promo_progress_ratio", "max"),
            promo_is_first_3_days=("promo_is_first_3_days", "max"),
            promo_is_last_3_days=("promo_is_last_3_days", "max"),
            promo_is_first_7_days=("promo_is_first_7_days", "max"),
            promo_is_last_7_days=("promo_is_last_7_days", "max"),
            promo_short_count=("promo_short", "sum"),
            promo_medium_count=("promo_medium", "sum"),
            promo_long_count=("promo_long", "sum"),
        )
    )
    daily["calendar_any_promo"] = (daily["calendar_active_promo_count"] > 0).astype(int)
    daily["promo_discount_duration_intensity"] = (
        daily["calendar_avg_discount_value"]
        * daily["calendar_active_promo_count"]
        * daily["promo_avg_duration_days"]
    )
    daily["promo_discount_progress_intensity"] = (
        daily["calendar_avg_discount_value"]
        * daily["calendar_active_promo_count"]
        * daily["promo_avg_progress_ratio"]
    )
    daily["promo_end_urgency_intensity"] = (
        daily["calendar_avg_discount_value"] * daily["promo_is_last_3_days"]
    )

    calendar = calendar.drop(columns=EXTENDED_PROMOTION_FEATURES).merge(
        daily,
        on=base.DATE_COL,
        how="left",
    )
    for feature in EXTENDED_PROMOTION_FEATURES:
        calendar[feature] = calendar[feature].fillna(0)

    return calendar[[base.DATE_COL] + EXTENDED_PROMOTION_FEATURES]


def build_static_features_with_promo_duration(
    dates: pd.Series,
    min_date: pd.Timestamp,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Build calendar, promo-duration calendar, and inventory as-of features."""
    calendar = base.build_calendar_features(dates, min_date)
    promo = build_promotion_calendar_with_duration(dates, base.PROMOTIONS_PATH, logger)
    inventory = base.build_inventory_asof_features(dates, base.INVENTORY_PATH, logger)
    return (
        calendar.merge(promo, on=base.DATE_COL, how="left", validate="one_to_one")
        .merge(inventory, on=base.DATE_COL, how="left", validate="one_to_one")
        .fillna(0)
    )


def validate_model_variant(
    variant: str,
    model_table: pd.DataFrame,
    static_features: pd.DataFrame,
    train_df: pd.DataFrame,
    reporter: RunReporter,
) -> dict[str, Any]:
    """Train before 2022 and recursively validate one day at a time in 2022."""
    feature_columns = get_feature_columns(variant)
    X_train, y_train, train_clean, feature_medians = base.make_training_matrix(
        model_table,
        feature_columns,
        base.TRAIN_CUTOFF,
    )
    reporter.emit(
        f"Training promo-duration Model {variant}: rows={len(X_train):,}, "
        f"features={len(feature_columns)}"
    )
    model, model_type = base.train_model(X_train, y_train, reporter)

    validation_dates = train_df[
        (train_df[base.DATE_COL] >= base.TRAIN_CUTOFF)
        & (train_df[base.DATE_COL] <= base.VALIDATION_END)
    ][base.DATE_COL]
    actual = train_df.set_index(base.DATE_COL).loc[validation_dates, base.TARGET_COL]
    initial_history = train_df[train_df[base.DATE_COL] < base.TRAIN_CUTOFF].set_index(base.DATE_COL)[
        base.TARGET_COL
    ]
    business_maps = base.build_business_source_maps(train_df[train_df[base.DATE_COL] < base.TRAIN_CUTOFF])

    predictions = base.recursive_predict(
        model=model,
        model_type=model_type,
        prediction_dates=validation_dates,
        feature_columns=feature_columns,
        static_features=static_features,
        initial_revenue_history=initial_history,
        business_maps=business_maps,
        feature_medians=feature_medians,
        include_business_lag365=(variant == "B"),
    )
    metrics = base.evaluate_predictions(actual, predictions)
    return {
        "variant": variant,
        "model": model,
        "model_type": model_type,
        "feature_columns": feature_columns,
        "feature_medians": feature_medians,
        "train_clean": train_clean,
        "validation_dates": validation_dates.reset_index(drop=True),
        "actual": actual.reset_index(drop=True),
        "predictions": predictions,
        "metrics": metrics,
    }


def choose_model_variant(result_a: dict[str, Any], result_b: dict[str, Any]) -> dict[str, Any]:
    """Prefer safe Model A unless B clearly improves recursive validation."""
    metrics_a = result_a["metrics"]
    metrics_b = result_b["metrics"]
    b_mae_improvement = (metrics_a["MAE"] - metrics_b["MAE"]) / metrics_a["MAE"]
    b_is_better = (
        b_mae_improvement >= 0.005
        and metrics_b["RMSE"] <= metrics_a["RMSE"]
        and metrics_b["R2"] >= metrics_a["R2"]
    )
    return result_b if b_is_better else result_a


def train_final_selected_model(
    selected_variant: str,
    model_table: pd.DataFrame,
    reporter: RunReporter,
) -> dict[str, Any]:
    """Train selected promo-duration model on all usable 2012-2022 rows."""
    feature_columns = get_feature_columns(selected_variant)
    X_all, y_all, train_clean, feature_medians = base.make_training_matrix(
        model_table,
        feature_columns,
        train_end_exclusive=None,
    )
    reporter.emit("")
    reporter.emit("5. Final training")
    reporter.emit(
        f"Training selected Model {selected_variant} on all usable rows: "
        f"rows={len(X_all):,}, features={len(feature_columns)}"
    )
    model, model_type = base.train_model(X_all, y_all, reporter)
    return {
        "variant": selected_variant,
        "model": model,
        "model_type": model_type,
        "feature_columns": feature_columns,
        "feature_medians": feature_medians,
        "X_all": X_all,
        "y_all": y_all,
        "train_clean": train_clean,
    }


def compare_with_previous(metrics: dict[str, float], reporter: RunReporter) -> dict[str, float | bool]:
    """Compare selected promo-duration model against previous final Model A."""
    mae_change = metrics["MAE"] - PREVIOUS_FINAL_MODEL_A["MAE"]
    rmse_change = metrics["RMSE"] - PREVIOUS_FINAL_MODEL_A["RMSE"]
    r2_change = metrics["R2"] - PREVIOUS_FINAL_MODEL_A["R2"]
    mae_pct = mae_change / PREVIOUS_FINAL_MODEL_A["MAE"] * 100
    rmse_pct = rmse_change / PREVIOUS_FINAL_MODEL_A["RMSE"] * 100
    improved = (
        metrics["MAE"] < PREVIOUS_FINAL_MODEL_A["MAE"]
        and metrics["RMSE"] < PREVIOUS_FINAL_MODEL_A["RMSE"]
        and metrics["R2"] > PREVIOUS_FINAL_MODEL_A["R2"]
    )

    reporter.emit("")
    reporter.emit("Comparison with previous final Model A")
    reporter.emit(
        f"Previous Model A - MAE={PREVIOUS_FINAL_MODEL_A['MAE']:,.2f}, "
        f"RMSE={PREVIOUS_FINAL_MODEL_A['RMSE']:,.2f}, R2={PREVIOUS_FINAL_MODEL_A['R2']:.6f}"
    )
    reporter.emit(
        f"Promo-duration model - MAE={metrics['MAE']:,.2f}, "
        f"RMSE={metrics['RMSE']:,.2f}, R2={metrics['R2']:.6f}"
    )
    reporter.emit(f"MAE change: {mae_change:,.2f} ({mae_pct:.2f}%)")
    reporter.emit(f"RMSE change: {rmse_change:,.2f} ({rmse_pct:.2f}%)")
    reporter.emit(f"R2 change: {r2_change:.6f}")
    reporter.emit("Promotion duration features improved model: " + ("yes" if improved else "no"))

    return {
        "mae_change": mae_change,
        "rmse_change": rmse_change,
        "r2_change": r2_change,
        "mae_pct": mae_pct,
        "rmse_pct": rmse_pct,
        "improved": improved,
    }


def save_validation_predictions(selected: dict[str, Any], path: Path) -> pd.DataFrame:
    """Save selected model validation predictions."""
    output = pd.DataFrame(
        {
            base.DATE_COL: selected["validation_dates"],
            "actual_Revenue": selected["actual"],
            "predicted_Revenue": selected["predictions"],
            "selected_model": selected["variant"],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    return output


def build_submission(
    sample_submission: pd.DataFrame,
    revenue_predictions: np.ndarray,
    cogs_ratio: float,
    path: Path,
) -> pd.DataFrame:
    """Create promo-duration submission without overwriting the current final submission."""
    submission = sample_submission[[base.DATE_COL]].copy()
    submission[base.TARGET_COL] = np.maximum(0.0, revenue_predictions)
    submission[base.COGS_COL] = np.maximum(0.0, submission[base.TARGET_COL] * cogs_ratio)
    submission = submission[[base.DATE_COL, base.TARGET_COL, base.COGS_COL]]
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    return submission


def emit_metrics(title: str, metrics: dict[str, float], reporter: RunReporter) -> None:
    reporter.emit(title)
    reporter.emit(f"MAE: {metrics['MAE']:,.2f}")
    reporter.emit(f"RMSE: {metrics['RMSE']:,.2f}")
    reporter.emit(f"R2: {metrics['R2']:.6f}")


def run_training() -> None:
    logger = setup_logging()
    reporter = RunReporter(logger)

    reporter.emit("Final Model With Promotion Duration Features")
    reporter.emit("============================================")
    reporter.emit("")
    reporter.emit("1. Load data")

    train_df = base.load_train_data(base.TRAIN_DATA_PATH)
    sample_submission = base.load_sample_submission(base.SAMPLE_SUBMISSION_PATH)
    min_date = train_df[base.DATE_COL].min()
    all_dates = pd.Series(
        pd.date_range(train_df[base.DATE_COL].min(), sample_submission[base.DATE_COL].max(), freq="D")
    )

    reporter.emit(f"Loaded train data: {base.TRAIN_DATA_PATH} | shape={train_df.shape}")
    reporter.emit(
        f"Train date range: {train_df[base.DATE_COL].min().date()} -> "
        f"{train_df[base.DATE_COL].max().date()}"
    )
    reporter.emit(
        "Forecast date range: "
        f"{sample_submission[base.DATE_COL].min().date()} -> "
        f"{sample_submission[base.DATE_COL].max().date()}"
    )

    reporter.emit("")
    reporter.emit("2. Build static features with promotion duration/phase")
    static_features = build_static_features_with_promo_duration(all_dates, min_date, logger)
    reporter.emit(f"Static feature table shape: {static_features.shape}")
    reporter.emit(f"Promotion duration features created ({len(PROMO_DURATION_FEATURES)}): {PROMO_DURATION_FEATURES}")

    reporter.emit("")
    reporter.emit("3. Build candidate model tables")
    table_a = base.build_historical_model_table(train_df, static_features, include_business_lag365=False)
    table_b = base.build_historical_model_table(train_df, static_features, include_business_lag365=True)
    dropped_unsafe = [column for column in base.UNSAFE_SAME_DAY_COLUMNS if column in train_df.columns]
    reporter.emit(f"Dropped/blocked unsafe same-day features: {dropped_unsafe}")
    reporter.emit(f"Model A features: {len(MODEL_A_FEATURES)}")
    reporter.emit(f"Model B features: {len(MODEL_B_FEATURES)}")

    reporter.emit("")
    reporter.emit("4. Validation backtest - recursive 2022 forecast")
    result_a = validate_model_variant("A", table_a, static_features, train_df, reporter)
    result_b = validate_model_variant("B", table_b, static_features, train_df, reporter)
    emit_metrics("Model A validation metrics:", result_a["metrics"], reporter)
    emit_metrics("Model B validation metrics:", result_b["metrics"], reporter)

    selected = choose_model_variant(result_a, result_b)
    selected_variant = selected["variant"]
    reporter.emit(f"Selected model: {selected_variant}")
    comparison = compare_with_previous(selected["metrics"], reporter)

    validation_output = save_validation_predictions(selected, VALIDATION_PREDICTIONS_PATH)
    reporter.emit(f"Saved validation predictions: {VALIDATION_PREDICTIONS_PATH}")
    reporter.emit(f"Validation prediction shape: {validation_output.shape}")

    selected_table = table_b if selected_variant == "B" else table_a
    final_model = train_final_selected_model(selected_variant, selected_table, reporter)

    reporter.emit("")
    reporter.emit("6. Recursive future forecast")
    initial_history = train_df.set_index(base.DATE_COL)[base.TARGET_COL].sort_index()
    business_maps = base.build_business_source_maps(train_df)
    revenue_predictions = base.recursive_predict(
        model=final_model["model"],
        model_type=final_model["model_type"],
        prediction_dates=sample_submission[base.DATE_COL],
        feature_columns=final_model["feature_columns"],
        static_features=static_features,
        initial_revenue_history=initial_history,
        business_maps=business_maps,
        feature_medians=final_model["feature_medians"],
        include_business_lag365=(selected_variant == "B"),
    )
    cogs_ratio = base.estimate_cogs_ratio(train_df)
    submission = build_submission(sample_submission, revenue_predictions, cogs_ratio, SUBMISSION_PATH)
    reporter.emit(f"Estimated COGS/Revenue ratio from latest 365 train days: {cogs_ratio:.6f}")
    reporter.emit(f"Saved promo-duration submission: {SUBMISSION_PATH}")

    reporter.emit("")
    reporter.emit("7. Feature importance")
    importance = base.get_feature_importance(
        final_model["model"],
        final_model["model_type"],
        final_model["feature_columns"],
        final_model["X_all"],
        final_model["y_all"],
        selected["metrics"]["RMSE"],
    )
    importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    top30 = importance.head(30)
    reporter.emit_frame("Top 30 feature importances:", top30)
    reporter.emit(f"Saved feature importance: {FEATURE_IMPORTANCE_PATH}")

    reporter.emit("")
    reporter.emit("8. Final summary")
    emit_metrics("Selected recursive validation metrics:", selected["metrics"], reporter)
    reporter.emit(
        "Metric improvement vs previous final Model A: "
        + ("yes" if comparison["improved"] else "no")
    )
    reporter.emit_frame("Top 30 feature importances:", top30)
    reporter.emit(f"Promotion duration features created: {PROMO_DURATION_FEATURES}")
    reporter.emit(
        "Promotion duration features improved the model: "
        + ("yes" if comparison["improved"] else "no")
    )
    reporter.emit(f"submission_promo_duration.csv rows: {len(submission):,}")
    reporter.emit(
        "Leakage confirmation: no order_items, same-day demand, returns, reviews, future Revenue, "
        "or future COGS were used. Promotion duration/phase features come only from promotions.csv schedules."
    )
    reporter.emit(f"Promo-duration submission path: {SUBMISSION_PATH}")

    reporter.save_metrics(METRICS_PATH)


if __name__ == "__main__":
    run_training()
