from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import build_promo_2023_features as promo_builder
import train_final_model as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "log"

SALES_PATH = DATA_DIR / "sales.csv"
PROMOTIONS_PATH = DATA_DIR / "promotions.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
FUTURE_PROMO_REFERENCE_PATH = DATA_DIR / "future_promo_calendar_features.csv"
CURRENT_BEST_SUBMISSION_PATH = DATA_DIR / "submission_blend_direct_15_cogs8900.csv"
SEGMENT_SUBMISSION_PATH = DATA_DIR / "submission_m5_segment_bottomup.csv"
FINAL_MICRO_VALIDATION_PATH = DATA_DIR / "final_micro_calibration_validation_predictions.csv"
DIRECT_SEASONAL_VALIDATION_PATH = DATA_DIR / "direct_seasonal_validation_predictions.csv"
DIRECT_SEASONAL_IMPORTANCE_PATH = DATA_DIR / "direct_seasonal_feature_importance.csv"
M5_VALIDATION_PATH = DATA_DIR / "m5_multilevel_validation_predictions.csv"

SYNTHETIC_PROMOTIONS_PATH = DATA_DIR / "synthetic_promotions_2023_2024.csv"
FUTURE_PROMO_KNOWN_PATH = DATA_DIR / "future_promo_known_features.csv"
SUBMISSION_PROMO_KNOWN_PATH = DATA_DIR / "submission_promo_known.csv"
VALIDATION_PREDICTIONS_PATH = DATA_DIR / "promo_known_validation_predictions.csv"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "promo_known_feature_importance.csv"
MODEL_COMPARISON_PATH = DATA_DIR / "promo_known_model_comparison.csv"
REPORT_PATH = LOG_DIR / "promo_known_pipeline_report.txt"
LOG_FILE = LOG_DIR / "train_promo_known_pipeline.log"

BLEND_OUTPUTS = {
    0.10: DATA_DIR / "submission_promo_known_blend_10.csv",
    0.20: DATA_DIR / "submission_promo_known_blend_20.csv",
    0.30: DATA_DIR / "submission_promo_known_blend_30.csv",
    0.40: DATA_DIR / "submission_promo_known_blend_40.csv",
    0.50: DATA_DIR / "submission_promo_known_blend_50.csv",
}

THREE_WAY_OUTPUTS = {
    "801010": DATA_DIR / "submission_promo_segment_blend_801010.csv",
    "702010": DATA_DIR / "submission_promo_segment_blend_702010.csv",
    "701020": DATA_DIR / "submission_promo_segment_blend_701020.csv",
    "602020": DATA_DIR / "submission_promo_segment_blend_602020.csv",
}

DATE_COL = base.DATE_COL
TARGET_COL = base.TARGET_COL
COGS_COL = base.COGS_COL
RANDOM_STATE = base.RANDOM_STATE

FOLDS = [
    ("fold_1", pd.Timestamp("2019-06-30"), pd.Timestamp("2019-07-01"), pd.Timestamp("2020-12-31")),
    ("fold_2", pd.Timestamp("2020-06-30"), pd.Timestamp("2020-07-01"), pd.Timestamp("2021-12-31")),
    ("fold_3", pd.Timestamp("2021-06-30"), pd.Timestamp("2021-07-01"), pd.Timestamp("2022-12-31")),
]

CAMPAIGN_FEATURE_MAP = {
    "spring sale": "spring_sale",
    "mid-year sale": "mid_year_sale",
    "fall launch": "fall_launch",
    "year-end sale": "year_end_sale",
    "urban blowout": "urban_blowout",
    "rural special": "rural_special",
}
CAMPAIGN_FEATURES = list(CAMPAIGN_FEATURE_MAP.values())

CALENDAR_FEATURES = [
    "day_of_week",
    "day_of_year",
    "week_of_year",
    "month",
    "year",
    "is_weekend",
    "is_month_start",
    "is_month_end",
]
PROMO_FEATURES = [
    "calendar_any_promo",
    "calendar_active_promo_count",
    "calendar_avg_discount_value",
    "calendar_max_discount_value",
    "calendar_stackable_promo_count",
    "promo_duration",
    "promo_day_number",
    "promo_progress_ratio",
    "promo_days_remaining",
    "promo_is_first_7_days",
    "promo_is_last_7_days",
    "promotion_campaign_index",
    "campaign_intensity",
    "discount_x_progress",
    "discount_x_days_remaining",
] + CAMPAIGN_FEATURES
LAG_FEATURES = [
    "lag_7",
    "lag_14",
    "lag_30",
    "lag_90",
    "lag_180",
    "lag_365",
    "rolling_mean_7",
    "rolling_mean_30",
    "rolling_mean_90",
    "rolling_mean_365",
]
INTERACTION_FEATURES = [
    "lag365_x_discount",
    "lag365_x_campaign_active",
    "rolling30_x_promo_active",
] + [f"day_of_year_x_{column}" for column in CAMPAIGN_FEATURES]
MODEL_FEATURES = CALENDAR_FEATURES + PROMO_FEATURES + LAG_FEATURES + INTERACTION_FEATURES


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
        if getattr(frame, "empty", False):
            self.emit("(empty)")
            return
        self.emit(frame.to_string(index=False))

    def save(self, path: Path = REPORT_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        self.logger.info("Saved report to %s", path)


def setup_logging(log_file: Path = LOG_FILE) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_promo_known_pipeline")
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


def safe_divide(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or abs(float(denominator)) < 1e-9:
        return np.nan
    return float(numerator) / float(denominator)


def normalize_campaign_name(value: Any) -> str:
    return str(value).strip().lower()


def stackable_to_int(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    return text.isin(["1", "true", "yes", "y"]).astype(int)


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


def robust_weighted_average(values: list[tuple[float, float]]) -> float:
    clean = [(weight, value) for weight, value in values if pd.notna(value) and np.isfinite(value)]
    if not clean:
        return np.nan
    total_weight = sum(weight for weight, _ in clean)
    if total_weight <= 0:
        return np.nan
    return float(sum(weight * value for weight, value in clean) / total_weight)


def compute_metrics(actual: pd.Series, predicted: np.ndarray, promo_mask: pd.Series) -> dict[str, float]:
    metrics = base.evaluate_predictions(actual, predicted)
    actual_np = actual.to_numpy(dtype=float)
    pred_np = np.asarray(predicted, dtype=float)
    errors = actual_np - pred_np
    top10_threshold = float(np.quantile(actual_np, 0.90))
    top10_mask = actual_np >= top10_threshold
    non_promo_mask = ~promo_mask.to_numpy(dtype=bool)
    promo_np = promo_mask.to_numpy(dtype=bool)

    metrics["top10_RMSE"] = float(np.sqrt(np.mean(errors[top10_mask] ** 2))) if top10_mask.any() else np.nan
    metrics["top10_underprediction"] = int(np.sum(errors[top10_mask] > 0)) if top10_mask.any() else 0
    metrics["promo_day_RMSE"] = float(np.sqrt(np.mean(errors[promo_np] ** 2))) if promo_np.any() else np.nan
    metrics["non_promo_RMSE"] = float(np.sqrt(np.mean(errors[non_promo_mask] ** 2))) if non_promo_mask.any() else np.nan
    return metrics


def load_sales(path: Path = SALES_PATH) -> pd.DataFrame:
    sales = pd.read_csv(path, parse_dates=[DATE_COL], low_memory=False)
    sales[DATE_COL] = pd.to_datetime(sales[DATE_COL], errors="coerce").dt.normalize()
    sales = sales.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    sales[TARGET_COL] = pd.to_numeric(sales[TARGET_COL], errors="coerce")
    sales[COGS_COL] = pd.to_numeric(sales[COGS_COL], errors="coerce")
    sales["year"] = sales[DATE_COL].dt.year.astype(int)
    sales["month"] = sales[DATE_COL].dt.month.astype(int)
    sales["day_of_year"] = sales[DATE_COL].dt.dayofyear.astype(int)
    return sales


def load_sample_submission(path: Path = SAMPLE_SUBMISSION_PATH) -> pd.DataFrame:
    sample = pd.read_csv(path, parse_dates=[DATE_COL], low_memory=False)
    sample[DATE_COL] = pd.to_datetime(sample[DATE_COL], errors="coerce").dt.normalize()
    return sample.sort_values(DATE_COL).reset_index(drop=True)


def load_promotions(path: Path = PROMOTIONS_PATH) -> pd.DataFrame:
    promotions = promo_builder.load_promotions(path)
    promotions["duration_days"] = (promotions["end_date"] - promotions["start_date"]).dt.days + 1
    promotions["campaign_index"] = (
        promotions.sort_values(["source_year", "start_date", "end_date", "promo_id"])
        .groupby("source_year")
        .cumcount()
        .add(1)
    )
    return promotions


def build_daily_promo_known_features(dates: pd.Series, promotions: pd.DataFrame) -> pd.DataFrame:
    calendar = pd.DataFrame({DATE_COL: pd.to_datetime(dates).sort_values().unique()})
    feature_columns = [
        "calendar_any_promo",
        "calendar_active_promo_count",
        "calendar_avg_discount_value",
        "calendar_max_discount_value",
        "calendar_stackable_promo_count",
        "promo_duration",
        "promo_day_number",
        "promo_progress_ratio",
        "promo_days_remaining",
        "promo_is_first_7_days",
        "promo_is_last_7_days",
        "promotion_campaign_index",
    ] + CAMPAIGN_FEATURES
    for column in feature_columns:
        calendar[column] = 0.0

    if promotions.empty:
        output = calendar.copy()
        output["campaign_intensity"] = 0.0
        output["discount_x_progress"] = 0.0
        output["discount_x_days_remaining"] = 0.0
        return output

    promos = promotions.copy()
    promos["discount_value"] = pd.to_numeric(promos["discount_value"], errors="coerce").fillna(0.0)
    promos["stackable_numeric"] = stackable_to_int(promos["stackable_flag"])

    min_date = calendar[DATE_COL].min()
    max_date = calendar[DATE_COL].max()
    rows: list[dict[str, Any]] = []

    for row in promos.itertuples(index=False):
        active_start = max(row.start_date, min_date)
        active_end = min(row.end_date, max_date)
        if active_start > active_end:
            continue

        campaign_values = {feature: 0 for feature in CAMPAIGN_FEATURES}
        campaign_feature = CAMPAIGN_FEATURE_MAP.get(normalize_campaign_name(row.promo_name_base))
        if campaign_feature is not None:
            campaign_values[campaign_feature] = 1

        duration_days = int(getattr(row, "duration_days", (row.end_date - row.start_date).days + 1))
        for active_date in pd.date_range(active_start, active_end, freq="D"):
            promo_day_number = (active_date - row.start_date).days + 1
            promo_days_remaining = (row.end_date - active_date).days
            rows.append(
                {
                    DATE_COL: active_date,
                    "promo_id": row.promo_id,
                    "discount_value": float(row.discount_value),
                    "stackable_numeric": int(row.stackable_numeric),
                    "promo_duration": duration_days,
                    "promo_day_number": promo_day_number,
                    "promo_progress_ratio": promo_day_number / max(duration_days, 1),
                    "promo_days_remaining": promo_days_remaining,
                    "promo_is_first_7_days": int(promo_day_number <= 7),
                    "promo_is_last_7_days": int(promo_days_remaining <= 6),
                    "promotion_campaign_index": int(getattr(row, "campaign_index", 0)),
                    **campaign_values,
                }
            )

    if not rows:
        output = calendar.copy()
        output["campaign_intensity"] = 0.0
        output["discount_x_progress"] = 0.0
        output["discount_x_days_remaining"] = 0.0
        return output

    expanded = pd.DataFrame(rows)
    daily = expanded.groupby(DATE_COL, as_index=False).agg(
        calendar_active_promo_count=("promo_id", "nunique"),
        calendar_avg_discount_value=("discount_value", "mean"),
        calendar_max_discount_value=("discount_value", "max"),
        calendar_stackable_promo_count=("stackable_numeric", "sum"),
        promo_duration=("promo_duration", "mean"),
        promo_day_number=("promo_day_number", "mean"),
        promo_progress_ratio=("promo_progress_ratio", "mean"),
        promo_days_remaining=("promo_days_remaining", "mean"),
        promo_is_first_7_days=("promo_is_first_7_days", "max"),
        promo_is_last_7_days=("promo_is_last_7_days", "max"),
        promotion_campaign_index=("promotion_campaign_index", "max"),
        **{feature: (feature, "max") for feature in CAMPAIGN_FEATURES},
    )
    daily["calendar_any_promo"] = (daily["calendar_active_promo_count"] > 0).astype(int)
    output = calendar.drop(columns=feature_columns).merge(daily, on=DATE_COL, how="left")
    for column in feature_columns:
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0.0)
    output["campaign_intensity"] = output["calendar_avg_discount_value"] * output["calendar_active_promo_count"]
    output["discount_x_progress"] = output["calendar_avg_discount_value"] * output["promo_progress_ratio"]
    output["discount_x_days_remaining"] = output["calendar_avg_discount_value"] * output["promo_days_remaining"]
    return output


def merge_future_reference_features(custom: pd.DataFrame, path: Path, logger: logging.Logger) -> pd.DataFrame:
    if not path.exists():
        return custom

    reference = pd.read_csv(path, parse_dates=[DATE_COL], low_memory=False)
    reference[DATE_COL] = pd.to_datetime(reference[DATE_COL], errors="coerce").dt.normalize()
    rename_map = {
        "future_calendar_any_promo": "calendar_any_promo",
        "future_calendar_active_promo_count": "calendar_active_promo_count",
        "future_calendar_avg_discount_value": "calendar_avg_discount_value",
        "future_calendar_max_discount_value": "calendar_max_discount_value",
        "future_calendar_stackable_promo_count": "calendar_stackable_promo_count",
        "future_promo_avg_duration_days": "promo_duration",
        "future_promo_avg_day_number": "promo_day_number",
        "future_promo_avg_progress_ratio": "promo_progress_ratio",
        "future_promo_avg_days_remaining": "promo_days_remaining",
        "future_promo_is_first_7_days": "promo_is_first_7_days",
        "future_promo_is_last_7_days": "promo_is_last_7_days",
        "future_promotion_campaign_index": "promotion_campaign_index",
    }
    keep_columns = [DATE_COL] + [column for column in rename_map if column in reference.columns]
    reference = reference[keep_columns].rename(columns=rename_map)

    merged = custom.set_index(DATE_COL).copy()
    ref_indexed = reference.set_index(DATE_COL)
    for column in ref_indexed.columns:
        if column in merged.columns:
            merged.loc[ref_indexed.index, column] = pd.to_numeric(ref_indexed[column], errors="coerce").fillna(
                merged.loc[ref_indexed.index, column]
            )
        else:
            merged[column] = pd.to_numeric(ref_indexed[column], errors="coerce")

    merged = merged.reset_index()
    merged["campaign_intensity"] = merged["calendar_avg_discount_value"] * merged["calendar_active_promo_count"]
    merged["discount_x_progress"] = merged["calendar_avg_discount_value"] * merged["promo_progress_ratio"]
    merged["discount_x_days_remaining"] = merged["calendar_avg_discount_value"] * merged["promo_days_remaining"]
    logger.info("Merged existing future_promo_calendar_features.csv as reference aggregate layer")
    return merged


def build_static_context(dates: pd.Series, promo_features: pd.DataFrame, min_date: pd.Timestamp) -> pd.DataFrame:
    calendar = base.build_calendar_features(dates, min_date)
    keep_calendar = [DATE_COL] + CALENDAR_FEATURES
    static = calendar[keep_calendar].merge(promo_features, on=DATE_COL, how="left", validate="one_to_one")
    return static.fillna(0.0)


def add_safe_lag_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.sort_values(DATE_COL).reset_index(drop=True).copy()
    revenue = pd.to_numeric(output[TARGET_COL], errors="coerce")
    shifted = revenue.shift(1)
    output["lag_7"] = revenue.shift(7)
    output["lag_14"] = revenue.shift(14)
    output["lag_30"] = revenue.shift(30)
    output["lag_90"] = revenue.shift(90)
    output["lag_180"] = revenue.shift(180)
    output["lag_365"] = revenue.shift(365)
    output["rolling_mean_7"] = shifted.rolling(window=7, min_periods=7).mean()
    output["rolling_mean_30"] = shifted.rolling(window=30, min_periods=30).mean()
    output["rolling_mean_90"] = shifted.rolling(window=90, min_periods=90).mean()
    output["rolling_mean_365"] = shifted.rolling(window=365, min_periods=365).mean()
    return output


def add_interaction_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["lag365_x_discount"] = output["lag_365"] * output["calendar_avg_discount_value"]
    output["lag365_x_campaign_active"] = output["lag_365"] * output["calendar_any_promo"]
    output["rolling30_x_promo_active"] = output["rolling_mean_30"] * output["calendar_any_promo"]
    for campaign_column in CAMPAIGN_FEATURES:
        output[f"day_of_year_x_{campaign_column}"] = output["day_of_year"] * output[campaign_column]
    return output


def build_training_table(sales: pd.DataFrame, promo_features: pd.DataFrame) -> pd.DataFrame:
    static = build_static_context(sales[DATE_COL], promo_features, sales[DATE_COL].min())
    table = sales[[DATE_COL, TARGET_COL, COGS_COL]].merge(static, on=DATE_COL, how="left", validate="one_to_one")
    table = add_safe_lag_features(table)
    table = add_interaction_features(table)
    return table


def train_regressor(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[Any, str]:
    if base.lightgbm_available():
        import lightgbm as lgb

        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.03,
            "max_depth": 6,
            "num_leaves": 31,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "min_data_in_leaf": 20,
            "seed": RANDOM_STATE,
            "verbosity": -1,
            "force_col_wise": True,
        }
        dataset = lgb.Dataset(X_train, label=y_train, feature_name=X_train.columns.tolist(), free_raw_data=False)
        model = lgb.train(params=params, train_set=dataset, num_boost_round=500)
        return model, "lightgbm"

    try:
        from sklearn.ensemble import GradientBoostingRegressor
    except Exception as exc:
        raise ImportError("Neither LightGBM nor sklearn fallback is available") from exc

    model = GradientBoostingRegressor(
        learning_rate=0.03,
        n_estimators=500,
        max_depth=3,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model, "gradient_boosting"


def predict_regressor(model: Any, model_type: str, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(X), dtype=float)


def prepare_training_matrix(training_table: pd.DataFrame, train_end: pd.Timestamp) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    subset = training_table.loc[training_table[DATE_COL] <= train_end].copy()
    subset = subset.dropna(subset=["lag_365", "rolling_mean_365"]).copy()
    X_train = subset[MODEL_FEATURES].apply(pd.to_numeric, errors="coerce")
    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)
    y_train = pd.to_numeric(subset[TARGET_COL], errors="coerce")
    mask = ~y_train.isna()
    return X_train.loc[mask].reset_index(drop=True), y_train.loc[mask].reset_index(drop=True), medians


def build_prediction_features(history: pd.DataFrame, target_date: pd.Timestamp, static_row: pd.Series) -> dict[str, float]:
    past = history.loc[history[DATE_COL] < target_date, [DATE_COL, TARGET_COL]].copy()
    past = past.sort_values(DATE_COL).reset_index(drop=True)
    value_map = past.set_index(DATE_COL)[TARGET_COL]

    def get_lag(days: int) -> float:
        ref_date = target_date - pd.Timedelta(days=days)
        if ref_date in value_map.index:
            return float(pd.to_numeric(value_map.loc[ref_date], errors="coerce"))
        return np.nan

    def get_recent_mean(window: int) -> float:
        if len(past) < window:
            return np.nan
        values = pd.to_numeric(past[TARGET_COL].tail(window), errors="coerce")
        return float(values.mean()) if len(values) == window else np.nan

    row = {feature: float(pd.to_numeric(static_row.get(feature, 0.0), errors="coerce")) for feature in CALENDAR_FEATURES + PROMO_FEATURES}
    row["lag_7"] = get_lag(7)
    row["lag_14"] = get_lag(14)
    row["lag_30"] = get_lag(30)
    row["lag_90"] = get_lag(90)
    row["lag_180"] = get_lag(180)
    row["lag_365"] = get_lag(365)
    row["rolling_mean_7"] = get_recent_mean(7)
    row["rolling_mean_30"] = get_recent_mean(30)
    row["rolling_mean_90"] = get_recent_mean(90)
    row["rolling_mean_365"] = get_recent_mean(365)
    row["lag365_x_discount"] = row["lag_365"] * row["calendar_avg_discount_value"] if pd.notna(row["lag_365"]) else np.nan
    row["lag365_x_campaign_active"] = row["lag_365"] * row["calendar_any_promo"] if pd.notna(row["lag_365"]) else np.nan
    row["rolling30_x_promo_active"] = row["rolling_mean_30"] * row["calendar_any_promo"] if pd.notna(row["rolling_mean_30"]) else np.nan
    for campaign_column in CAMPAIGN_FEATURES:
        row[f"day_of_year_x_{campaign_column}"] = row["day_of_year"] * row[campaign_column]
    return row


def recursive_predict(
    model: Any,
    model_type: str,
    medians: pd.Series,
    history: pd.DataFrame,
    future_static: pd.DataFrame,
) -> pd.DataFrame:
    history_frame = history[[DATE_COL, TARGET_COL]].copy().sort_values(DATE_COL).reset_index(drop=True)
    static_index = future_static.set_index(DATE_COL).sort_index()
    rows: list[dict[str, Any]] = []

    for target_date in future_static[DATE_COL]:
        static_row = static_index.loc[target_date]
        feature_row = build_prediction_features(history_frame, target_date, static_row)
        X = pd.DataFrame([feature_row], columns=MODEL_FEATURES).apply(pd.to_numeric, errors="coerce").fillna(medians)
        prediction = float(np.clip(predict_regressor(model, model_type, X)[0], 0.0, None))
        rows.append({DATE_COL: target_date, "predicted_Revenue": prediction})
        history_frame = pd.concat(
            [history_frame, pd.DataFrame({DATE_COL: [target_date], TARGET_COL: [prediction]})],
            ignore_index=True,
        )

    return pd.DataFrame(rows)


def extract_feature_importance(model: Any, model_type: str) -> pd.DataFrame:
    if model_type == "lightgbm":
        features = model.feature_name()
        gain = model.feature_importance(importance_type="gain")
        split = model.feature_importance(importance_type="split")
        importance = pd.DataFrame(
            {"feature": features, "importance_gain": gain.astype(float), "importance_split": split.astype(float)}
        )
        return importance.sort_values("importance_gain", ascending=False).reset_index(drop=True)
    return pd.DataFrame({"feature": MODEL_FEATURES, "importance_gain": np.nan, "importance_split": np.nan})


def summarize_synthetic_promotions(synthetic: pd.DataFrame, even_source_year: int) -> pd.DataFrame:
    summary = (
        synthetic.assign(target_year=synthetic["start_date"].dt.year.where(synthetic["start_date"].dt.year == 2023, 2024))
        .groupby("target_year", as_index=False)
        .agg(
            campaign_count=("promo_id", "count"),
            start_date=("start_date", "min"),
            end_date=("end_date", "max"),
        )
    )
    summary["source_pattern_year"] = summary["target_year"].map({2023: 2021, 2024: even_source_year})
    return summary[["target_year", "source_pattern_year", "campaign_count", "start_date", "end_date"]]


def build_current_best_validation_2022() -> pd.DataFrame | None:
    if not FINAL_MICRO_VALIDATION_PATH.exists() or not DIRECT_SEASONAL_VALIDATION_PATH.exists():
        return None

    current_base = pd.read_csv(FINAL_MICRO_VALIDATION_PATH, parse_dates=[DATE_COL], low_memory=False)
    current_base[DATE_COL] = pd.to_datetime(current_base[DATE_COL], errors="coerce").dt.normalize()
    if "current_base_pred" not in current_base.columns or "actual_Revenue" not in current_base.columns:
        return None

    direct_validation = pd.read_csv(DIRECT_SEASONAL_VALIDATION_PATH, parse_dates=[DATE_COL], low_memory=False)
    direct_validation[DATE_COL] = pd.to_datetime(direct_validation[DATE_COL], errors="coerce").dt.normalize()
    target_type = "log_ratio"
    if DIRECT_SEASONAL_IMPORTANCE_PATH.exists():
        importance = pd.read_csv(DIRECT_SEASONAL_IMPORTANCE_PATH, low_memory=False)
        if "best_target_type" in importance.columns:
            non_null = importance["best_target_type"].dropna()
            if not non_null.empty:
                target_type = str(non_null.iloc[0])

    direct_subset = direct_validation.loc[
        (direct_validation["fold"] == "fold_3") & (direct_validation["target_type"] == target_type),
        [DATE_COL, "predicted_Revenue"],
    ].rename(columns={"predicted_Revenue": "direct_pred"})

    merged = current_base.merge(direct_subset, on=DATE_COL, how="inner", validate="one_to_one")
    if merged.empty:
        return None

    merged["current_best_pred"] = 0.85 * pd.to_numeric(merged["current_base_pred"], errors="coerce") + 0.15 * pd.to_numeric(
        merged["direct_pred"], errors="coerce"
    )
    return merged[[DATE_COL, "actual_Revenue", "current_best_pred"]].copy()


def load_segment_validation_2022() -> pd.DataFrame | None:
    if not M5_VALIDATION_PATH.exists():
        return None
    frame = pd.read_csv(M5_VALIDATION_PATH, parse_dates=[DATE_COL], low_memory=False)
    frame[DATE_COL] = pd.to_datetime(frame[DATE_COL], errors="coerce").dt.normalize()
    if "segment_recursive_sum" not in frame.columns:
        return None
    subset = frame.loc[frame["fold"] == "fold_3", [DATE_COL, "actual_Revenue", "segment_recursive_sum"]].copy()
    subset = subset.rename(columns={"segment_recursive_sum": "segment_pred"})
    return subset


def score_candidate(
    candidate_name: str,
    actual: pd.Series,
    predicted: np.ndarray,
    promo_mask: pd.Series,
    scope: str,
) -> dict[str, Any]:
    metrics = compute_metrics(actual, predicted, promo_mask)
    return {"candidate": candidate_name, "scope": scope, **metrics}


def build_submission(dates: pd.Series, revenue: pd.Series | np.ndarray, ratio: float = 0.8900) -> pd.DataFrame:
    revenue_series = pd.Series(np.asarray(revenue, dtype=float))
    output = pd.DataFrame(
        {
            DATE_COL: pd.to_datetime(dates).reset_index(drop=True),
            TARGET_COL: revenue_series.clip(lower=0.0),
            COGS_COL: revenue_series.clip(lower=0.0) * ratio,
        }
    )
    return output


def run() -> None:
    logger = setup_logging()
    reporter = Reporter(logger)
    reporter.emit("Promo-Known Forecasting Pipeline")
    reporter.emit("==============================")
    reporter.emit("")

    sales = load_sales()
    sample_submission = load_sample_submission()
    promotions = load_promotions()

    reporter.emit("1. Build synthetic future promotions")
    synthetic_promotions, even_source_year = promo_builder.build_synthetic_promotions(promotions)
    synthetic_promotions.to_csv(SYNTHETIC_PROMOTIONS_PATH, index=False, date_format="%Y-%m-%d")
    synthetic_summary = summarize_synthetic_promotions(synthetic_promotions, even_source_year)
    reporter.emit_frame("Synthetic promotion summary:", synthetic_summary)

    reporter.emit("")
    reporter.emit("2. Build daily promo-known features")
    historical_promo_features = build_daily_promo_known_features(sales[DATE_COL], promotions)
    future_promo_features = build_daily_promo_known_features(sample_submission[DATE_COL], synthetic_promotions)
    future_promo_features = merge_future_reference_features(future_promo_features, FUTURE_PROMO_REFERENCE_PATH, logger)
    future_promo_features.to_csv(FUTURE_PROMO_KNOWN_PATH, index=False, date_format="%Y-%m-%d")
    reporter.emit_frame(
        "Future promo-known feature sample:",
        future_promo_features.head(10)[
            [DATE_COL, "calendar_any_promo", "calendar_active_promo_count", "calendar_avg_discount_value", "promo_progress_ratio"]
        ],
    )

    reporter.emit("")
    reporter.emit("3. Train promo-aware model with long-horizon validation")
    training_table = build_training_table(sales, historical_promo_features)
    fold_rows: list[dict[str, Any]] = []
    validation_rows: list[pd.DataFrame] = []
    fold_predictions_lookup: dict[str, pd.DataFrame] = {}

    for fold_name, train_end, valid_start, valid_end in FOLDS:
        reporter.emit(f"{fold_name}: train <= {train_end.date()}, validate {valid_start.date()} -> {valid_end.date()}")
        X_train, y_train, medians = prepare_training_matrix(training_table, train_end)
        model, model_type = train_regressor(X_train, y_train)

        history = sales.loc[sales[DATE_COL] <= train_end, [DATE_COL, TARGET_COL]].copy()
        fold_static = build_static_context(
            sales.loc[(sales[DATE_COL] >= valid_start) & (sales[DATE_COL] <= valid_end), DATE_COL],
            historical_promo_features.loc[
                (historical_promo_features[DATE_COL] >= valid_start) & (historical_promo_features[DATE_COL] <= valid_end)
            ].copy(),
            sales[DATE_COL].min(),
        )
        fold_predictions = recursive_predict(model, model_type, medians, history, fold_static)
        actual = sales.loc[(sales[DATE_COL] >= valid_start) & (sales[DATE_COL] <= valid_end), [DATE_COL, TARGET_COL]].copy()
        merged = actual.merge(fold_predictions, on=DATE_COL, how="left", validate="one_to_one")
        merged = merged.merge(
            historical_promo_features[[DATE_COL, "calendar_any_promo", "campaign_intensity"]],
            on=DATE_COL,
            how="left",
            validate="one_to_one",
        )
        merged["fold"] = fold_name
        merged["error"] = merged[TARGET_COL] - merged["predicted_Revenue"]
        merged["abs_error"] = merged["error"].abs()
        validation_rows.append(merged)
        fold_predictions_lookup[fold_name] = merged[[DATE_COL, TARGET_COL, "predicted_Revenue", "calendar_any_promo"]].copy()

        metrics = score_candidate(
            candidate_name="promo_known_model",
            actual=merged[TARGET_COL],
            predicted=merged["predicted_Revenue"].to_numpy(dtype=float),
            promo_mask=merged["calendar_any_promo"].fillna(0).astype(int),
            scope=fold_name,
        )
        fold_rows.append(metrics)

    validation_predictions = pd.concat(validation_rows, ignore_index=True)
    validation_predictions.to_csv(VALIDATION_PREDICTIONS_PATH, index=False, date_format="%Y-%m-%d")
    fold_metrics = pd.DataFrame(fold_rows)
    avg_metrics = pd.DataFrame(
        [
            {
                "candidate": "promo_known_model",
                "scope": "average_folds",
                "MAE": fold_metrics["MAE"].mean(),
                "RMSE": fold_metrics["RMSE"].mean(),
                "R2": fold_metrics["R2"].mean(),
                "top10_RMSE": fold_metrics["top10_RMSE"].mean(),
                "top10_underprediction": fold_metrics["top10_underprediction"].mean(),
                "promo_day_RMSE": fold_metrics["promo_day_RMSE"].mean(),
                "non_promo_RMSE": fold_metrics["non_promo_RMSE"].mean(),
            }
        ]
    )
    reporter.emit_frame("Fold metrics:", fold_metrics)
    reporter.emit_frame("Average fold metrics:", avg_metrics)

    reporter.emit("")
    reporter.emit("4. Train final promo-aware model on full history")
    X_final, y_final, final_medians = prepare_training_matrix(training_table, sales[DATE_COL].max())
    final_model, final_model_type = train_regressor(X_final, y_final)
    final_importance = extract_feature_importance(final_model, final_model_type)
    final_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    reporter.emit_frame("Top 30 features:", final_importance.head(30))

    future_static = build_static_context(sample_submission[DATE_COL], future_promo_features, sales[DATE_COL].min())
    future_predictions = recursive_predict(
        final_model,
        final_model_type,
        final_medians,
        sales[[DATE_COL, TARGET_COL]].copy(),
        future_static,
    )
    submission_promo_known = build_submission(sample_submission[DATE_COL], future_predictions["predicted_Revenue"], ratio=0.8900)
    validate_submission_frame(submission_promo_known, sample_submission)
    submission_promo_known.to_csv(SUBMISSION_PROMO_KNOWN_PATH, index=False, date_format="%Y-%m-%d")

    reporter.emit("")
    reporter.emit("5. Blend with current best and optional segment bottom-up")
    current_best_submission = pd.read_csv(CURRENT_BEST_SUBMISSION_PATH, parse_dates=[DATE_COL], low_memory=False)
    current_best_submission[DATE_COL] = pd.to_datetime(current_best_submission[DATE_COL], errors="coerce").dt.normalize()
    if not current_best_submission[DATE_COL].equals(sample_submission[DATE_COL]):
        raise ValueError("Current best submission Date order does not match sample submission")
    current_best_revenue = pd.to_numeric(current_best_submission[TARGET_COL], errors="coerce")
    promo_known_revenue = pd.to_numeric(submission_promo_known[TARGET_COL], errors="coerce")

    created_files = [str(SUBMISSION_PROMO_KNOWN_PATH)]
    for weight, output_path in BLEND_OUTPUTS.items():
        revenue = (1.0 - weight) * current_best_revenue + weight * promo_known_revenue
        output = build_submission(sample_submission[DATE_COL], revenue, ratio=0.8900)
        validate_submission_frame(output, sample_submission)
        output.to_csv(output_path, index=False, date_format="%Y-%m-%d")
        created_files.append(str(output_path))

    segment_submission_exists = SEGMENT_SUBMISSION_PATH.exists()
    if segment_submission_exists:
        segment_submission = pd.read_csv(SEGMENT_SUBMISSION_PATH, parse_dates=[DATE_COL], low_memory=False)
        segment_submission[DATE_COL] = pd.to_datetime(segment_submission[DATE_COL], errors="coerce").dt.normalize()
        if not segment_submission[DATE_COL].equals(sample_submission[DATE_COL]):
            raise ValueError("Segment bottom-up submission Date order does not match sample submission")
        segment_revenue = pd.to_numeric(segment_submission[TARGET_COL], errors="coerce")
        three_way_specs = {
            "801010": (0.80, 0.10, 0.10),
            "702010": (0.70, 0.20, 0.10),
            "701020": (0.70, 0.10, 0.20),
            "602020": (0.60, 0.20, 0.20),
        }
        for key, (w_current, w_promo, w_segment) in three_way_specs.items():
            revenue = (
                w_current * current_best_revenue
                + w_promo * promo_known_revenue
                + w_segment * segment_revenue
            )
            output = build_submission(sample_submission[DATE_COL], revenue, ratio=0.8900)
            validate_submission_frame(output, sample_submission)
            output.to_csv(THREE_WAY_OUTPUTS[key], index=False, date_format="%Y-%m-%d")
            created_files.append(str(THREE_WAY_OUTPUTS[key]))

    comparison_rows = pd.concat([fold_metrics, avg_metrics], ignore_index=True)

    current_best_validation = build_current_best_validation_2022()
    segment_validation = load_segment_validation_2022()
    best_blend_candidate = None

    if current_best_validation is not None:
        promo_fold3 = fold_predictions_lookup["fold_3"].rename(columns={TARGET_COL: "actual_Revenue"})
        blend_eval = current_best_validation.merge(
            promo_fold3[[DATE_COL, "predicted_Revenue", "calendar_any_promo"]],
            on=DATE_COL,
            how="inner",
            validate="one_to_one",
        ).rename(columns={"predicted_Revenue": "promo_known_pred"})
        actual_series = pd.to_numeric(blend_eval["actual_Revenue"], errors="coerce")
        promo_mask = blend_eval["calendar_any_promo"].fillna(0).astype(int)

        candidate_rows = [
            score_candidate(
                candidate_name="current_best_2022_analog",
                actual=actual_series,
                predicted=blend_eval["current_best_pred"].to_numpy(dtype=float),
                promo_mask=promo_mask,
                scope="fold_3_2022",
            ),
            score_candidate(
                candidate_name="promo_known_fold_3",
                actual=actual_series,
                predicted=blend_eval["promo_known_pred"].to_numpy(dtype=float),
                promo_mask=promo_mask,
                scope="fold_3_2022",
            ),
        ]

        for weight in sorted(BLEND_OUTPUTS):
            candidate_pred = (1.0 - weight) * blend_eval["current_best_pred"] + weight * blend_eval["promo_known_pred"]
            candidate_rows.append(
                score_candidate(
                    candidate_name=f"current_best_plus_promo_{int(weight * 100):02d}",
                    actual=actual_series,
                    predicted=candidate_pred.to_numpy(dtype=float),
                    promo_mask=promo_mask,
                    scope="fold_3_2022",
                )
            )

        if segment_validation is not None:
            segment_eval = blend_eval.merge(
                segment_validation[[DATE_COL, "segment_pred"]],
                on=DATE_COL,
                how="inner",
                validate="one_to_one",
            )
            three_way_specs = {
                "blend_801010": (0.80, 0.10, 0.10),
                "blend_702010": (0.70, 0.20, 0.10),
                "blend_701020": (0.70, 0.10, 0.20),
                "blend_602020": (0.60, 0.20, 0.20),
            }
            for name, (w_current, w_promo, w_segment) in three_way_specs.items():
                candidate_pred = (
                    w_current * segment_eval["current_best_pred"]
                    + w_promo * segment_eval["promo_known_pred"]
                    + w_segment * segment_eval["segment_pred"]
                )
                candidate_rows.append(
                    score_candidate(
                        candidate_name=name,
                        actual=pd.to_numeric(segment_eval["actual_Revenue"], errors="coerce"),
                        predicted=candidate_pred.to_numpy(dtype=float),
                        promo_mask=segment_eval["calendar_any_promo"].fillna(0).astype(int),
                        scope="fold_3_2022",
                    )
                )

        candidate_frame = pd.DataFrame(candidate_rows).sort_values("RMSE").reset_index(drop=True)
        comparison_rows = pd.concat([comparison_rows, candidate_frame], ignore_index=True, sort=False)
        best_blend_candidate = candidate_frame.iloc[0].to_dict()
        reporter.emit_frame("2022 blend candidate metrics:", candidate_frame)

    comparison_rows.to_csv(MODEL_COMPARISON_PATH, index=False)

    reporter.emit("")
    reporter.emit("6. Final summary")
    reporter.emit(f"Created files: {', '.join(created_files)}")
    if best_blend_candidate is not None:
        reporter.emit(
            "Best blend candidate by validation: "
            f"{best_blend_candidate['candidate']} | RMSE={best_blend_candidate['RMSE']:,.2f}, "
            f"MAE={best_blend_candidate['MAE']:,.2f}, R2={best_blend_candidate['R2']:.6f}"
        )
    else:
        reporter.emit("Best blend candidate by validation: unavailable (benchmark analog missing)")

    upload_order = [
        "submission_promo_known_blend_10.csv",
        "submission_promo_known_blend_20.csv",
        "submission_promo_known.csv",
        "submission_promo_known_blend_30.csv",
    ]
    if segment_submission_exists:
        upload_order = [
            "submission_promo_segment_blend_801010.csv",
            "submission_promo_known_blend_10.csv",
            "submission_promo_segment_blend_702010.csv",
            "submission_promo_known.csv",
            "submission_promo_known_blend_20.csv",
        ]
    reporter.emit(f"Recommended upload order: {upload_order}")
    reporter.emit(
        "Leakage safety confirmation: the promo-known pipeline only uses historical Revenue, known historical promotions, "
        "synthetic future promo schedules derived from promotions.csv, and safe lag>=7 recursive features. "
        "No future actual Revenue/COGS, no same-day realized demand, and no external data are used."
    )
    reporter.save()


if __name__ == "__main__":
    run()
