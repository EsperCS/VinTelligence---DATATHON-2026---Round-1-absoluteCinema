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

TRAIN_DATA_PATH = DATA_DIR / "daily_feature_table.csv"
PROMOTIONS_PATH = DATA_DIR / "promotions.csv"
FUTURE_PROMO_FEATURES_PATH = DATA_DIR / "future_promo_calendar_features.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"

PRUNED_VALIDATION_PATH = DATA_DIR / "pruned_ensemble_validation_predictions.csv"
SPIKE_VALIDATION_PATH = DATA_DIR / "spike_model_validation_predictions.csv"
PRUNED_SUBMISSION_PATH = DATA_DIR / "submission_pruned_ensemble.csv"
SPIKE_SUBMISSION_PATH = DATA_DIR / "submission_spike_aware.csv"

SUBMISSION_DEEP_PATH = DATA_DIR / "submission_deep_feature.csv"
SUBMISSION_ENSEMBLE_PATH = DATA_DIR / "submission_deep_feature_ensemble.csv"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "deep_feature_importance.csv"
VALIDATION_PREDICTIONS_PATH = DATA_DIR / "deep_feature_validation_predictions.csv"

REPORT_PATH = LOG_DIR / "deep_feature_report.txt"
LOG_FILE = LOG_DIR / "train_deep_feature_model.log"

DATE_COL = base.DATE_COL
TARGET_COL = base.TARGET_COL
COGS_COL = base.COGS_COL
TRAIN_CUTOFF = base.TRAIN_CUTOFF
VALIDATION_END = base.VALIDATION_END
RANDOM_STATE = base.RANDOM_STATE

EPSILON = 1e-6
TOP_FEATURE_MIN = 60
TOP_FEATURE_MAX = 80
CORRELATION_THRESHOLD = 0.995
WEIGHT_STEP = 0.05

HISTORICAL_PROMO_CONTEXT_COLUMNS = [
    "promo_duration",
    "promo_day_number",
    "promo_days_remaining",
    "promo_progress_ratio",
    "promotion_campaign_index",
]
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
    "future_promo_avg_duration_days": "promo_duration",
    "future_promo_avg_day_number": "promo_day_number",
    "future_promo_avg_days_remaining": "promo_days_remaining",
    "future_promo_avg_progress_ratio": "promo_progress_ratio",
    "future_promotion_campaign_index": "promotion_campaign_index",
}

BASE_SAFE_FEATURES = [
    "day_of_week",
    "day_of_year",
    "week_of_year",
    "month",
    "quarter",
    "year",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "is_quarter_start",
    "is_quarter_end",
    "time_index",
    "post_2019_flag",
    "years_since_start",
    "years_since_2019",
    "post_2019_time_index",
    "lag_7",
    "lag_14",
    "lag_30",
    "revenue_lag_60",
    "revenue_lag_90",
    "revenue_lag_180",
    "revenue_lag_365",
    "rolling_mean_7",
    "rolling_mean_30",
    "revenue_roll_mean_14",
    "revenue_roll_mean_60",
    "revenue_roll_mean_90",
    "revenue_roll_mean_180",
    "revenue_roll_mean_365",
    "revenue_roll_std_30",
    "revenue_roll_std_90",
    "revenue_roll_std_365",
    "calendar_active_promo_count",
    "calendar_any_promo",
    "calendar_avg_discount_value",
    "calendar_max_discount_value",
    "calendar_stackable_promo_count",
    "calendar_has_stackable_promo",
    "calendar_has_category_specific_promo",
    "calendar_percentage_promo_count",
    "calendar_fixed_promo_count",
    "promo_duration",
    "promo_day_number",
    "promo_days_remaining",
    "promo_progress_ratio",
    "promotion_campaign_index",
    "inv_stockout_rate",
    "inv_avg_fill_rate",
    "inv_avg_days_of_supply",
    "inv_avg_sell_through_rate",
    "inv_reorder_rate",
    "inv_overstock_rate",
]

DEEP_FEATURES = [
    "revenue_lag_1",
    "revenue_lag_3",
    "rev_growth_1_7",
    "rev_growth_7_30",
    "rev_growth_30_90",
    "short_trend",
    "rolling_std_7",
    "rolling_std_30",
    "rolling_std_90",
    "volatility_7",
    "volatility_30",
    "volatility_90",
    "volatility_ratio",
    "lag365_above_p90",
    "lag365_above_p95",
    "spike_strength_365",
    "discount_intensity",
    "promo_pressure",
    "promo_urgency",
    "seasonal_phase",
    "seasonal_phase_cos",
]

INTERACTION_FEATURES = [
    "revenue_lag_365_x_calendar_avg_discount_value",
    "revenue_lag_365_x_calendar_any_promo",
    "rev_growth_7_30_x_calendar_any_promo",
    "rev_growth_1_7_x_calendar_avg_discount_value",
    "volatility_30_x_calendar_any_promo",
    "day_of_year_x_calendar_any_promo",
    "seasonal_phase_x_calendar_avg_discount_value",
    "lag365_above_p90_x_calendar_any_promo",
    "spike_strength_365_x_calendar_avg_discount_value",
]

CANDIDATE_FEATURES = BASE_SAFE_FEATURES + DEEP_FEATURES + INTERACTION_FEATURES


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

    logger = logging.getLogger("train_deep_feature_model")
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


def safe_divide(numerator: Any, denominator: Any, epsilon: float = EPSILON) -> Any:
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


def deduplicate_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def load_train_data(path: Path = TRAIN_DATA_PATH) -> pd.DataFrame:
    return base.load_train_data(path)


def load_sample_submission(path: Path = SAMPLE_SUBMISSION_PATH) -> pd.DataFrame:
    return base.load_sample_submission(path)


def load_promotions(path: Path = PROMOTIONS_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    promotions = pd.read_csv(path, low_memory=False)
    promotions["start_date"] = pd.to_datetime(promotions["start_date"], errors="coerce").dt.normalize()
    promotions["end_date"] = pd.to_datetime(promotions["end_date"], errors="coerce").dt.normalize()
    promotions = promotions.dropna(subset=["start_date", "end_date"]).copy()
    if promotions.empty:
        return promotions

    promotions["source_year"] = promotions["start_date"].dt.year.astype(int)
    promotions["discount_value"] = pd.to_numeric(promotions.get("discount_value", 0), errors="coerce").fillna(0.0)
    promotions["duration_days"] = (promotions["end_date"] - promotions["start_date"]).dt.days + 1
    promotions = promotions.sort_values(["source_year", "start_date", "end_date"]).reset_index(drop=True)
    promotions["campaign_index"] = promotions.groupby("source_year").cumcount() + 1
    return promotions


def build_historical_promo_context(dates: pd.Series, promotions_path: Path = PROMOTIONS_PATH) -> pd.DataFrame:
    calendar = pd.DataFrame({DATE_COL: pd.to_datetime(dates).sort_values().unique()})
    for column in HISTORICAL_PROMO_CONTEXT_COLUMNS:
        calendar[column] = 0.0

    promotions = load_promotions(promotions_path)
    if promotions.empty:
        return calendar

    min_date = calendar[DATE_COL].min()
    max_date = calendar[DATE_COL].max()
    rows: list[dict[str, Any]] = []
    for row in promotions.itertuples(index=False):
        active_start = max(row.start_date, min_date)
        active_end = min(row.end_date, max_date)
        if active_start > active_end:
            continue

        for active_date in pd.date_range(active_start, active_end, freq="D"):
            promo_day_number = (active_date - row.start_date).days + 1
            promo_days_remaining = (row.end_date - active_date).days
            rows.append(
                {
                    DATE_COL: active_date,
                    "promo_duration": float(row.duration_days),
                    "promo_day_number": float(promo_day_number),
                    "promo_days_remaining": float(promo_days_remaining),
                    "promo_progress_ratio": float(promo_day_number / row.duration_days),
                    "promotion_campaign_index": float(row.campaign_index),
                }
            )

    if not rows:
        return calendar

    expanded = pd.DataFrame(rows)
    aggregated = (
        expanded.groupby(DATE_COL, as_index=False)
        .agg(
            promo_duration=("promo_duration", "mean"),
            promo_day_number=("promo_day_number", "mean"),
            promo_days_remaining=("promo_days_remaining", "mean"),
            promo_progress_ratio=("promo_progress_ratio", "mean"),
            promotion_campaign_index=("promotion_campaign_index", "mean"),
        )
    )
    return calendar.drop(columns=HISTORICAL_PROMO_CONTEXT_COLUMNS).merge(aggregated, on=DATE_COL, how="left").fillna(0.0)


def load_future_promo_context(path: Path = FUTURE_PROMO_FEATURES_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Future promo feature file not found: {path}")

    future = pd.read_csv(path, parse_dates=[DATE_COL], low_memory=False)
    future[DATE_COL] = pd.to_datetime(future[DATE_COL], errors="coerce").dt.normalize()
    if future[DATE_COL].isna().any():
        raise ValueError("future_promo_calendar_features.csv contains invalid dates")

    missing = [column for column in FUTURE_PROMO_RENAME_MAP if column not in future.columns]
    if missing:
        raise ValueError(f"Missing future promo columns: {missing}")

    future = future.rename(columns=FUTURE_PROMO_RENAME_MAP)
    keep_columns = [DATE_COL] + list(FUTURE_PROMO_RENAME_MAP.values())
    future = future[keep_columns].copy()

    numeric_columns = [column for column in future.columns if column != DATE_COL]
    future[numeric_columns] = future[numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return future.sort_values(DATE_COL).reset_index(drop=True)


def build_historical_static_features(train_df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    dates = train_df[DATE_COL]
    min_date = train_df[DATE_COL].min()
    calendar = base.build_calendar_features(dates, min_date)
    promo = base.build_promotion_calendar(dates, PROMOTIONS_PATH, logger)
    promo_context = build_historical_promo_context(dates, PROMOTIONS_PATH)
    inventory = base.build_inventory_asof_features(dates, base.INVENTORY_PATH, logger)
    return (
        calendar.merge(promo, on=DATE_COL, how="left", validate="one_to_one")
        .merge(promo_context, on=DATE_COL, how="left", validate="one_to_one")
        .merge(inventory, on=DATE_COL, how="left", validate="one_to_one")
        .fillna(0.0)
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )


def build_future_static_features(future_dates: pd.Series, min_date: pd.Timestamp, logger: logging.Logger) -> pd.DataFrame:
    calendar = base.build_calendar_features(future_dates, min_date)
    promo_context = load_future_promo_context(FUTURE_PROMO_FEATURES_PATH)
    inventory = base.build_inventory_asof_features(future_dates, base.INVENTORY_PATH, logger)
    return (
        calendar.merge(promo_context, on=DATE_COL, how="left", validate="one_to_one")
        .merge(inventory, on=DATE_COL, how="left", validate="one_to_one")
        .fillna(0.0)
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )


def add_deep_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    output = df.sort_values(DATE_COL).reset_index(drop=True).copy()
    shifted_revenue = pd.to_numeric(output[TARGET_COL], errors="coerce").shift(1)

    output["revenue_lag_1"] = output[TARGET_COL].shift(1)
    output["revenue_lag_3"] = output[TARGET_COL].shift(3)

    output["rolling_std_7"] = shifted_revenue.rolling(window=7, min_periods=7).std()
    output["rolling_std_30"] = shifted_revenue.rolling(window=30, min_periods=30).std()
    output["rolling_std_90"] = shifted_revenue.rolling(window=90, min_periods=90).std()

    output["rev_growth_1_7"] = safe_divide(output["revenue_lag_1"], output["lag_7"])
    output["rev_growth_7_30"] = safe_divide(output["lag_7"], output["rolling_mean_30"])
    output["rev_growth_30_90"] = safe_divide(output["lag_30"], output["revenue_roll_mean_90"])
    output["short_trend"] = output["revenue_lag_3"] - output["lag_7"]

    output["volatility_7"] = output["rolling_std_7"]
    output["volatility_30"] = output["rolling_std_30"]
    output["volatility_90"] = output["rolling_std_90"]
    output["volatility_ratio"] = safe_divide(output["volatility_7"], output["volatility_30"])

    p90_history = shifted_revenue.expanding(min_periods=1).quantile(0.90)
    p95_history = shifted_revenue.expanding(min_periods=1).quantile(0.95)
    output["lag365_above_p90"] = np.where(
        output["revenue_lag_365"].notna() & p90_history.notna(),
        (output["revenue_lag_365"] >= p90_history).astype(int),
        np.nan,
    )
    output["lag365_above_p95"] = np.where(
        output["revenue_lag_365"].notna() & p95_history.notna(),
        (output["revenue_lag_365"] >= p95_history).astype(int),
        np.nan,
    )
    output["spike_strength_365"] = safe_divide(output["revenue_lag_365"], output["revenue_roll_mean_365"])

    output["discount_intensity"] = output["calendar_avg_discount_value"] * output["calendar_active_promo_count"]
    output["promo_pressure"] = output["promo_progress_ratio"] * output["calendar_avg_discount_value"]
    output["promo_urgency"] = np.where(
        output["calendar_any_promo"] > 0,
        1.0 / (pd.to_numeric(output["promo_days_remaining"], errors="coerce").fillna(0.0) + 1.0),
        0.0,
    )

    phase = 2.0 * np.pi * output["day_of_year"] / 365.0
    output["seasonal_phase"] = np.sin(phase)
    output["seasonal_phase_cos"] = np.cos(phase)

    output["revenue_lag_365_x_calendar_avg_discount_value"] = (
        output["revenue_lag_365"] * output["calendar_avg_discount_value"]
    )
    output["revenue_lag_365_x_calendar_any_promo"] = (
        output["revenue_lag_365"] * output["calendar_any_promo"]
    )
    output["rev_growth_7_30_x_calendar_any_promo"] = output["rev_growth_7_30"] * output["calendar_any_promo"]
    output["rev_growth_1_7_x_calendar_avg_discount_value"] = (
        output["rev_growth_1_7"] * output["calendar_avg_discount_value"]
    )
    output["volatility_30_x_calendar_any_promo"] = output["volatility_30"] * output["calendar_any_promo"]
    output["day_of_year_x_calendar_any_promo"] = output["day_of_year"] * output["calendar_any_promo"]
    output["seasonal_phase_x_calendar_avg_discount_value"] = (
        output["seasonal_phase"] * output["calendar_avg_discount_value"]
    )
    output["lag365_above_p90_x_calendar_any_promo"] = (
        output["lag365_above_p90"] * output["calendar_any_promo"]
    )
    output["spike_strength_365_x_calendar_avg_discount_value"] = (
        output["spike_strength_365"] * output["calendar_avg_discount_value"]
    )
    return output


def build_deep_model_table(train_df: pd.DataFrame, static_features: pd.DataFrame) -> pd.DataFrame:
    table = base.build_historical_model_table(train_df, static_features, include_business_lag365=False)
    return add_deep_historical_features(table)


def compute_threshold_bundle(history: pd.Series) -> dict[str, float]:
    ordered = pd.to_numeric(history, errors="coerce").dropna().sort_index()
    if ordered.empty:
        return {"p90": 0.0, "p95": 0.0}
    return {
        "p90": float(ordered.quantile(0.90)),
        "p95": float(ordered.quantile(0.95)),
    }


def compute_extra_revenue_features_from_history(history: pd.Series, forecast_date: pd.Timestamp) -> dict[str, float]:
    past_history = pd.to_numeric(history, errors="coerce")
    past_history = past_history[past_history.index < forecast_date].sort_index()

    def tail_stat(window: int, fn: str) -> float:
        values = past_history.tail(window)
        if len(values) != window:
            return np.nan
        if fn == "mean":
            return float(values.mean())
        return float(values.std(ddof=1))

    return {
        "revenue_lag_1": float(past_history.get(forecast_date - pd.Timedelta(days=1), np.nan)),
        "revenue_lag_3": float(past_history.get(forecast_date - pd.Timedelta(days=3), np.nan)),
        "rolling_std_7": tail_stat(7, "std"),
        "rolling_std_30": tail_stat(30, "std"),
        "rolling_std_90": tail_stat(90, "std"),
    }


def compute_deep_features_from_row(row: dict[str, float], thresholds: dict[str, float]) -> dict[str, float]:
    lag1 = row.get("revenue_lag_1", np.nan)
    lag3 = row.get("revenue_lag_3", np.nan)
    lag7 = row.get("lag_7", np.nan)
    lag30 = row.get("lag_30", np.nan)
    lag365 = row.get("revenue_lag_365", np.nan)
    roll30 = row.get("rolling_mean_30", np.nan)
    roll90 = row.get("revenue_roll_mean_90", np.nan)
    roll365 = row.get("revenue_roll_mean_365", np.nan)
    std7 = row.get("rolling_std_7", np.nan)
    std30 = row.get("rolling_std_30", np.nan)
    std90 = row.get("rolling_std_90", np.nan)
    promo_any = float(row.get("calendar_any_promo", 0.0) or 0.0)
    promo_discount = float(row.get("calendar_avg_discount_value", 0.0) or 0.0)
    promo_count = float(row.get("calendar_active_promo_count", 0.0) or 0.0)
    promo_days_remaining = float(row.get("promo_days_remaining", 0.0) or 0.0)
    promo_progress = float(row.get("promo_progress_ratio", 0.0) or 0.0)
    day_of_year = float(row.get("day_of_year", 0.0) or 0.0)
    phase = 2.0 * np.pi * day_of_year / 365.0

    rev_growth_1_7 = safe_divide(lag1, lag7)
    rev_growth_7_30 = safe_divide(lag7, roll30)
    rev_growth_30_90 = safe_divide(lag30, roll90)
    volatility_ratio = safe_divide(std7, std30)
    lag365_above_p90 = float(int(pd.notna(lag365) and lag365 >= thresholds["p90"])) if pd.notna(lag365) else np.nan
    lag365_above_p95 = float(int(pd.notna(lag365) and lag365 >= thresholds["p95"])) if pd.notna(lag365) else np.nan
    spike_strength_365 = safe_divide(lag365, roll365)
    seasonal_phase = float(np.sin(phase))

    return {
        "rev_growth_1_7": rev_growth_1_7,
        "rev_growth_7_30": rev_growth_7_30,
        "rev_growth_30_90": rev_growth_30_90,
        "short_trend": lag3 - lag7 if pd.notna(lag3) and pd.notna(lag7) else np.nan,
        "volatility_7": std7,
        "volatility_30": std30,
        "volatility_90": std90,
        "volatility_ratio": volatility_ratio,
        "lag365_above_p90": lag365_above_p90,
        "lag365_above_p95": lag365_above_p95,
        "spike_strength_365": spike_strength_365,
        "discount_intensity": promo_discount * promo_count,
        "promo_pressure": promo_progress * promo_discount,
        "promo_urgency": 1.0 / (promo_days_remaining + 1.0) if promo_any > 0 else 0.0,
        "seasonal_phase": seasonal_phase,
        "seasonal_phase_cos": float(np.cos(phase)),
        "revenue_lag_365_x_calendar_avg_discount_value": lag365 * promo_discount if pd.notna(lag365) else np.nan,
        "revenue_lag_365_x_calendar_any_promo": lag365 * promo_any if pd.notna(lag365) else np.nan,
        "rev_growth_7_30_x_calendar_any_promo": rev_growth_7_30 * promo_any if pd.notna(rev_growth_7_30) else np.nan,
        "rev_growth_1_7_x_calendar_avg_discount_value": (
            rev_growth_1_7 * promo_discount if pd.notna(rev_growth_1_7) else np.nan
        ),
        "volatility_30_x_calendar_any_promo": std30 * promo_any if pd.notna(std30) else np.nan,
        "day_of_year_x_calendar_any_promo": day_of_year * promo_any,
        "seasonal_phase_x_calendar_avg_discount_value": seasonal_phase * promo_discount,
        "lag365_above_p90_x_calendar_any_promo": (
            lag365_above_p90 * promo_any if pd.notna(lag365_above_p90) else np.nan
        ),
        "spike_strength_365_x_calendar_avg_discount_value": (
            spike_strength_365 * promo_discount if pd.notna(spike_strength_365) else np.nan
        ),
    }


def build_candidate_feature_list(model_table: pd.DataFrame) -> list[str]:
    blocked = {DATE_COL, TARGET_COL, COGS_COL}
    blocked.update(base.UNSAFE_SAME_DAY_COLUMNS)
    return [
        feature
        for feature in deduplicate_preserve_order(CANDIDATE_FEATURES)
        if feature in model_table.columns and feature not in blocked
    ]


def train_lightgbm_variant(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    objective: str = "regression",
    alpha: float | None = None,
) -> Any:
    import lightgbm as lgb

    params = {
        "objective": objective,
        "metric": "rmse",
        "learning_rate": 0.025,
        "max_depth": 6,
        "num_leaves": 31,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "feature_fraction": 0.9,
        "seed": RANDOM_STATE,
        "verbosity": -1,
        "force_col_wise": True,
    }
    if objective == "quantile":
        params["metric"] = "quantile"
        params["alpha"] = 0.70 if alpha is None else alpha

    dataset = lgb.Dataset(
        X_train,
        label=y_train,
        feature_name=X_train.columns.tolist(),
        free_raw_data=False,
    )
    return lgb.train(params=params, train_set=dataset, num_boost_round=1800)


def make_training_matrix(
    model_table: pd.DataFrame,
    feature_columns: list[str],
    train_end_exclusive: pd.Timestamp | None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    table = model_table.copy()
    if train_end_exclusive is not None:
        table = table[table[DATE_COL] < train_end_exclusive].copy()

    clean = table.dropna(subset=feature_columns + [TARGET_COL]).reset_index(drop=True)
    X = clean[feature_columns].copy()
    y = clean[TARGET_COL].copy()
    medians = X.median(numeric_only=True)
    return X, y, clean, medians


def get_feature_importance(model: Any, feature_columns: list[str]) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "importance_split": model.feature_importance(importance_type="split"),
                "importance_gain": model.feature_importance(importance_type="gain"),
            }
        )
        .sort_values(["importance_gain", "importance_split"], ascending=False)
        .reset_index(drop=True)
    )


def select_features(
    model_table: pd.DataFrame,
    candidate_features: list[str],
    reporter: Reporter,
) -> tuple[list[str], pd.DataFrame]:
    X_train, y_train, clean, _ = make_training_matrix(model_table, candidate_features, TRAIN_CUTOFF)
    reporter.emit(
        f"Feature selection warm-up: rows={len(X_train):,}, candidate_features={len(candidate_features)}"
    )
    warm_model = train_lightgbm_variant(X_train, y_train, objective="regression")
    importance = get_feature_importance(warm_model, candidate_features)
    importance["selected_after_correlation"] = 0

    ranked = importance[importance["importance_gain"] > 0]["feature"].tolist()
    if not ranked:
        ranked = importance["feature"].tolist()

    numeric_frame = clean[ranked].apply(pd.to_numeric, errors="coerce")
    corr = numeric_frame.corr().abs()

    selected: list[str] = []
    for feature in ranked:
        if feature not in corr.columns:
            continue
        if any(corr.loc[feature, kept] >= CORRELATION_THRESHOLD for kept in selected if kept in corr.columns):
            continue
        selected.append(feature)
        if len(selected) >= TOP_FEATURE_MAX:
            break

    if len(selected) < TOP_FEATURE_MIN:
        for feature in ranked:
            if feature not in selected:
                selected.append(feature)
            if len(selected) >= min(TOP_FEATURE_MAX, len(ranked)):
                break

    selected = selected[:TOP_FEATURE_MAX]
    importance.loc[importance["feature"].isin(selected), "selected_after_correlation"] = 1
    return selected, importance


def compute_spike_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    actual_values = actual.to_numpy(dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    error = actual_values - predicted_values

    top10_threshold = float(np.quantile(actual_values, 0.90))
    top5_threshold = float(np.quantile(actual_values, 0.95))
    top10_mask = actual_values >= top10_threshold
    top5_mask = actual_values >= top5_threshold
    non_spike_mask = actual_values < top10_threshold

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


def evaluate_candidate(name: str, actual: pd.Series, predictions: np.ndarray) -> dict[str, Any]:
    overall = base.evaluate_predictions(actual, predictions)
    spike_metrics = compute_spike_metrics(actual, predictions)
    return {"model": name, **overall, **spike_metrics}


def recursive_predict_deep(
    model: Any,
    prediction_dates: pd.Series,
    feature_columns: list[str],
    static_features: pd.DataFrame,
    initial_revenue_history: pd.Series,
    feature_medians: pd.Series,
    thresholds: dict[str, float],
) -> np.ndarray:
    static_by_date = static_features.set_index(DATE_COL).sort_index()
    history = pd.to_numeric(initial_revenue_history, errors="coerce").sort_index().copy()
    predictions: list[float] = []

    for forecast_date in pd.to_datetime(prediction_dates):
        if forecast_date not in static_by_date.index:
            raise ValueError(f"Missing static features for forecast date {forecast_date.date()}")

        row: dict[str, float] = static_by_date.loc[forecast_date].to_dict()
        row.update(base.compute_revenue_features_from_history(history, forecast_date))
        row.update(compute_extra_revenue_features_from_history(history, forecast_date))
        row.update(compute_deep_features_from_row(row, thresholds))

        X_row = pd.DataFrame([row], columns=feature_columns)
        X_row = X_row.apply(pd.to_numeric, errors="coerce").fillna(feature_medians).fillna(0.0)

        prediction = float(model.predict(X_row)[0])
        prediction = max(0.0, prediction)
        predictions.append(prediction)
        history.loc[forecast_date] = prediction

    return np.asarray(predictions, dtype=float)


def validate_variant(
    variant_name: str,
    model_table: pd.DataFrame,
    static_features: pd.DataFrame,
    train_df: pd.DataFrame,
    feature_columns: list[str],
    objective: str,
    alpha: float | None,
    reporter: Reporter,
) -> dict[str, Any]:
    X_train, y_train, train_clean, feature_medians = make_training_matrix(model_table, feature_columns, TRAIN_CUTOFF)
    reporter.emit(
        f"Training {variant_name}: rows={len(X_train):,}, features={len(feature_columns)}, objective={objective}"
    )
    model = train_lightgbm_variant(X_train, y_train, objective=objective, alpha=alpha)

    validation_dates = train_df[
        (train_df[DATE_COL] >= TRAIN_CUTOFF) & (train_df[DATE_COL] <= VALIDATION_END)
    ][DATE_COL]
    actual = train_df.set_index(DATE_COL).loc[validation_dates, TARGET_COL].reset_index(drop=True)
    initial_history = train_df[train_df[DATE_COL] < TRAIN_CUTOFF].set_index(DATE_COL)[TARGET_COL]
    thresholds = compute_threshold_bundle(initial_history)
    predictions = recursive_predict_deep(
        model=model,
        prediction_dates=validation_dates,
        feature_columns=feature_columns,
        static_features=static_features,
        initial_revenue_history=initial_history,
        feature_medians=feature_medians,
        thresholds=thresholds,
    )
    metrics = evaluate_candidate(variant_name, actual, predictions)
    return {
        "model": variant_name,
        "model_object": model,
        "feature_columns": feature_columns,
        "feature_medians": feature_medians,
        "train_clean": train_clean,
        "X_train": X_train,
        "y_train": y_train,
        "validation_dates": validation_dates.reset_index(drop=True),
        "actual": actual,
        "predictions": predictions,
        "metrics": metrics,
        "thresholds": thresholds,
        "importance": get_feature_importance(model, feature_columns),
    }


def retrain_full_variant(
    variant_name: str,
    model_table: pd.DataFrame,
    feature_columns: list[str],
    objective: str,
    alpha: float | None,
    reporter: Reporter,
) -> dict[str, Any]:
    X_train, y_train, train_clean, feature_medians = make_training_matrix(model_table, feature_columns, None)
    reporter.emit(f"Retraining {variant_name} on all rows: rows={len(X_train):,}, features={len(feature_columns)}")
    model = train_lightgbm_variant(X_train, y_train, objective=objective, alpha=alpha)
    thresholds = compute_threshold_bundle(train_clean.set_index(DATE_COL)[TARGET_COL])
    return {
        "model": variant_name,
        "model_object": model,
        "feature_columns": feature_columns,
        "feature_medians": feature_medians,
        "train_clean": train_clean,
        "X_train": X_train,
        "y_train": y_train,
        "thresholds": thresholds,
        "importance": get_feature_importance(model, feature_columns),
    }


def load_validation_frame(path: Path, model_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Validation prediction file not found: {path}")
    frame = pd.read_csv(path, parse_dates=[DATE_COL], low_memory=False)
    frame[DATE_COL] = pd.to_datetime(frame[DATE_COL], errors="coerce").dt.normalize()
    if "actual_Revenue" not in frame.columns or "predicted_Revenue" not in frame.columns:
        raise ValueError(f"{path} must contain Date, actual_Revenue, predicted_Revenue")
    return frame[[DATE_COL, "actual_Revenue", "predicted_Revenue"]].rename(
        columns={"predicted_Revenue": model_name}
    )


def search_three_model_ensemble(
    actual: pd.Series,
    pruned_pred: np.ndarray,
    spike_pred: np.ndarray,
    deep_pred: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    units = int(round(1.0 / WEIGHT_STEP))
    for pruned_unit in range(units + 1):
        for spike_unit in range(units + 1 - pruned_unit):
            deep_unit = units - pruned_unit - spike_unit
            w_pruned = pruned_unit * WEIGHT_STEP
            w_spike = spike_unit * WEIGHT_STEP
            w_deep = deep_unit * WEIGHT_STEP
            blended = w_pruned * pruned_pred + w_spike * spike_pred + w_deep * deep_pred
            metrics = evaluate_candidate("ensemble", actual, blended)
            rows.append(
                {
                    "weight_pruned": w_pruned,
                    "weight_spike": w_spike,
                    "weight_deep": w_deep,
                    **metrics,
                }
            )
    return pd.DataFrame(rows).sort_values(["RMSE", "MAE", "top10_RMSE"]).reset_index(drop=True)


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


def save_submission(
    sample_submission: pd.DataFrame,
    predictions: np.ndarray,
    cogs_ratio: float,
    path: Path,
) -> pd.DataFrame:
    output = sample_submission[[DATE_COL]].copy()
    output[TARGET_COL] = np.maximum(0.0, np.asarray(predictions, dtype=float))
    output[COGS_COL] = np.maximum(0.0, output[TARGET_COL] * cogs_ratio)
    validate_submission_frame(output, sample_submission)
    output.to_csv(path, index=False)
    return output


def blend_submissions(
    sample_submission: pd.DataFrame,
    pruned_submission: pd.DataFrame,
    spike_submission: pd.DataFrame,
    deep_submission: pd.DataFrame,
    weights: dict[str, float],
    path: Path,
) -> pd.DataFrame:
    output = sample_submission[[DATE_COL]].copy()
    output[TARGET_COL] = (
        weights["pruned"] * pruned_submission[TARGET_COL]
        + weights["spike"] * spike_submission[TARGET_COL]
        + weights["deep"] * deep_submission[TARGET_COL]
    )
    output[COGS_COL] = (
        weights["pruned"] * pruned_submission[COGS_COL]
        + weights["spike"] * spike_submission[COGS_COL]
        + weights["deep"] * deep_submission[COGS_COL]
    )
    output[TARGET_COL] = output[TARGET_COL].clip(lower=0.0)
    output[COGS_COL] = output[COGS_COL].clip(lower=0.0)
    validate_submission_frame(output, sample_submission)
    output.to_csv(path, index=False)
    return output


def load_submission(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Submission file not found: {path}")
    frame = pd.read_csv(path, parse_dates=[DATE_COL], low_memory=False)
    frame[DATE_COL] = pd.to_datetime(frame[DATE_COL], errors="coerce").dt.normalize()
    return frame[[DATE_COL, TARGET_COL, COGS_COL]].copy()


def run() -> None:
    logger = setup_logging()
    reporter = Reporter(logger)

    reporter.emit("Deep Feature Revenue Model")
    reporter.emit("==========================")
    reporter.emit("")

    reporter.emit("1. Load historical data and build safe static features")
    train_df = load_train_data(TRAIN_DATA_PATH)
    sample_submission = load_sample_submission(SAMPLE_SUBMISSION_PATH)
    historical_static = build_historical_static_features(train_df, logger)
    future_static = build_future_static_features(sample_submission[DATE_COL], train_df[DATE_COL].min(), logger)
    model_table = build_deep_model_table(train_df, historical_static)
    candidate_features = build_candidate_feature_list(model_table)
    reporter.emit(f"Historical rows: {len(train_df):,}")
    reporter.emit(f"Candidate feature count before selection: {len(candidate_features)}")

    reporter.emit("")
    reporter.emit("2. Deep feature mining + selection")
    selected_features, warm_importance = select_features(model_table, candidate_features, reporter)
    reporter.emit(f"Selected feature count after importance/correlation pruning: {len(selected_features)}")
    reporter.emit_frame(
        "Top 25 warm-up feature gains:",
        warm_importance.head(25)[["feature", "importance_gain", "selected_after_correlation"]],
    )

    reporter.emit("")
    reporter.emit("3. Train recursive validation variants")
    result_a = validate_variant(
        variant_name="DEEP_FEATURE_STANDARD",
        model_table=model_table,
        static_features=historical_static,
        train_df=train_df,
        feature_columns=selected_features,
        objective="regression",
        alpha=None,
        reporter=reporter,
    )
    result_b = validate_variant(
        variant_name="DEEP_FEATURE_QUANTILE_Q70",
        model_table=model_table,
        static_features=historical_static,
        train_df=train_df,
        feature_columns=selected_features,
        objective="quantile",
        alpha=0.70,
        reporter=reporter,
    )
    best_deep = min([result_a, result_b], key=lambda item: (item["metrics"]["RMSE"], item["metrics"]["MAE"]))

    comparison_frame = pd.DataFrame([result_a["metrics"], result_b["metrics"]]).sort_values("RMSE").reset_index(drop=True)
    reporter.emit_frame("Deep model comparison:", comparison_frame)

    reporter.emit("")
    reporter.emit("4. Compare against existing pruned / spike baselines")
    pruned_validation = load_validation_frame(PRUNED_VALIDATION_PATH, "pruned_pred")
    spike_validation = load_validation_frame(SPIKE_VALIDATION_PATH, "spike_pred")
    baseline_frame = (
        pruned_validation.merge(spike_validation, on=[DATE_COL, "actual_Revenue"], how="inner", validate="one_to_one")
    )
    baseline_actual = baseline_frame["actual_Revenue"]
    pruned_metrics = evaluate_candidate("PRUNED_ENSEMBLE", baseline_actual, baseline_frame["pruned_pred"].to_numpy(dtype=float))
    spike_metrics = evaluate_candidate("SPIKE_MODEL", baseline_actual, baseline_frame["spike_pred"].to_numpy(dtype=float))
    reporter.emit_frame(
        "Baseline comparison:",
        pd.DataFrame([pruned_metrics, spike_metrics]).sort_values("RMSE").reset_index(drop=True),
    )

    reporter.emit("")
    reporter.emit("5. Search new ensemble: deep + pruned + spike")
    deep_validation = pd.DataFrame(
        {
            DATE_COL: result_a["validation_dates"],
            "actual_Revenue": result_a["actual"],
            "deep_pred": best_deep["predictions"],
        }
    )
    ensemble_frame = (
        baseline_frame.merge(deep_validation, on=[DATE_COL, "actual_Revenue"], how="inner", validate="one_to_one")
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )
    ensemble_search = search_three_model_ensemble(
        actual=ensemble_frame["actual_Revenue"],
        pruned_pred=ensemble_frame["pruned_pred"].to_numpy(dtype=float),
        spike_pred=ensemble_frame["spike_pred"].to_numpy(dtype=float),
        deep_pred=ensemble_frame["deep_pred"].to_numpy(dtype=float),
    )
    best_ensemble_row = ensemble_search.iloc[0].to_dict()
    reporter.emit_frame("Top ensemble weights:", ensemble_search.head(10))

    reporter.emit("")
    reporter.emit("6. Retrain best deep variant on full 2012-2022 data")
    full_deep = retrain_full_variant(
        variant_name=best_deep["model"],
        model_table=model_table,
        feature_columns=selected_features,
        objective="quantile" if best_deep["model"] == "DEEP_FEATURE_QUANTILE_Q70" else "regression",
        alpha=0.70 if best_deep["model"] == "DEEP_FEATURE_QUANTILE_Q70" else None,
        reporter=reporter,
    )

    future_predictions = recursive_predict_deep(
        model=full_deep["model_object"],
        prediction_dates=sample_submission[DATE_COL],
        feature_columns=full_deep["feature_columns"],
        static_features=future_static,
        initial_revenue_history=train_df.set_index(DATE_COL)[TARGET_COL],
        feature_medians=full_deep["feature_medians"],
        thresholds=full_deep["thresholds"],
    )
    cogs_ratio = base.estimate_cogs_ratio(train_df)
    deep_submission = save_submission(sample_submission, future_predictions, cogs_ratio, SUBMISSION_DEEP_PATH)

    pruned_submission = load_submission(PRUNED_SUBMISSION_PATH)
    spike_submission = load_submission(SPIKE_SUBMISSION_PATH)
    ensemble_weights = {
        "pruned": float(best_ensemble_row["weight_pruned"]),
        "spike": float(best_ensemble_row["weight_spike"]),
        "deep": float(best_ensemble_row["weight_deep"]),
    }
    ensemble_submission = blend_submissions(
        sample_submission=sample_submission,
        pruned_submission=pruned_submission,
        spike_submission=spike_submission,
        deep_submission=deep_submission,
        weights=ensemble_weights,
        path=SUBMISSION_ENSEMBLE_PATH,
    )
    del ensemble_submission

    reporter.emit("")
    reporter.emit("7. Save validation predictions and feature importance")
    best_ensemble_validation_pred = (
        best_ensemble_row["weight_pruned"] * ensemble_frame["pruned_pred"].to_numpy(dtype=float)
        + best_ensemble_row["weight_spike"] * ensemble_frame["spike_pred"].to_numpy(dtype=float)
        + best_ensemble_row["weight_deep"] * ensemble_frame["deep_pred"].to_numpy(dtype=float)
    )
    validation_output = pd.DataFrame(
        {
            DATE_COL: ensemble_frame[DATE_COL],
            "actual_Revenue": ensemble_frame["actual_Revenue"],
            "deep_standard_pred": result_a["predictions"],
            "deep_quantile_pred": result_b["predictions"],
            "deep_best_pred": best_deep["predictions"],
            "pruned_pred": ensemble_frame["pruned_pred"],
            "spike_pred": ensemble_frame["spike_pred"],
            "ensemble_pred": best_ensemble_validation_pred,
        }
    )
    validation_output.to_csv(VALIDATION_PREDICTIONS_PATH, index=False)

    importance_output = pd.concat(
        [
            warm_importance.assign(stage="warmup_selection", variant="warmup"),
            result_a["importance"].assign(stage="validation", variant=result_a["model"]),
            result_b["importance"].assign(stage="validation", variant=result_b["model"]),
            full_deep["importance"].assign(stage="full_train", variant=full_deep["model"]),
        ],
        ignore_index=True,
    )
    importance_output["selected_feature"] = importance_output["feature"].isin(selected_features).astype(int)
    importance_output.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    reporter.emit("")
    reporter.emit("8. Final summary")
    deep_rmse_improvement_vs_pruned = pruned_metrics["RMSE"] - best_deep["metrics"]["RMSE"]
    deep_rmse_improvement_vs_spike = spike_metrics["RMSE"] - best_deep["metrics"]["RMSE"]
    ensemble_rmse_improvement_vs_pruned = pruned_metrics["RMSE"] - float(best_ensemble_row["RMSE"])
    ensemble_rmse_improvement_vs_spike = spike_metrics["RMSE"] - float(best_ensemble_row["RMSE"])

    reporter.emit_frame(
        "Top 30 selected features from full-train model:",
        full_deep["importance"].head(30)[["feature", "importance_gain", "importance_split"]],
    )

    new_feature_hits = [
        feature
        for feature in full_deep["importance"]["feature"].head(30).tolist()
        if feature in (DEEP_FEATURES + INTERACTION_FEATURES)
    ]
    reporter.emit(f"Best deep variant: {best_deep['model']}")
    reporter.emit(
        "Deep variant metrics: "
        f"MAE={best_deep['metrics']['MAE']:,.2f} | "
        f"RMSE={best_deep['metrics']['RMSE']:,.2f} | "
        f"R2={best_deep['metrics']['R2']:.6f}"
    )
    reporter.emit(
        "Deep variant spike metrics: "
        f"top10_RMSE={best_deep['metrics']['top10_RMSE']:,.2f} | "
        f"top10_underprediction={best_deep['metrics']['top10_underprediction']}/{best_deep['metrics']['top10_count']} | "
        f"non_spike_RMSE={best_deep['metrics']['non_spike_RMSE']:,.2f}"
    )
    reporter.emit(
        "RMSE improvement vs pruned baseline: "
        f"{deep_rmse_improvement_vs_pruned:,.2f}"
    )
    reporter.emit(
        "RMSE improvement vs spike baseline: "
        f"{deep_rmse_improvement_vs_spike:,.2f}"
    )
    reporter.emit(
        "Best ensemble weights: "
        f"pruned={ensemble_weights['pruned']:.2f}, "
        f"spike={ensemble_weights['spike']:.2f}, "
        f"deep={ensemble_weights['deep']:.2f}"
    )
    reporter.emit(
        "Best ensemble metrics: "
        f"MAE={best_ensemble_row['MAE']:,.2f} | "
        f"RMSE={best_ensemble_row['RMSE']:,.2f} | "
        f"R2={best_ensemble_row['R2']:.6f}"
    )
    reporter.emit(
        "Best ensemble spike metrics: "
        f"top10_RMSE={best_ensemble_row['top10_RMSE']:,.2f} | "
        f"top10_underprediction={int(best_ensemble_row['top10_underprediction'])}/{int(best_ensemble_row['top10_count'])} | "
        f"non_spike_RMSE={best_ensemble_row['non_spike_RMSE']:,.2f}"
    )
    reporter.emit(
        "Ensemble RMSE improvement vs pruned baseline: "
        f"{ensemble_rmse_improvement_vs_pruned:,.2f}"
    )
    reporter.emit(
        "Ensemble RMSE improvement vs spike baseline: "
        f"{ensemble_rmse_improvement_vs_spike:,.2f}"
    )
    reporter.emit(
        "New features that actually helped (appearing in top 30): "
        + (", ".join(new_feature_hits) if new_feature_hits else "none from the new deep-feature block")
    )
    reporter.emit(f"Saved deep submission: {SUBMISSION_DEEP_PATH}")
    reporter.emit(f"Saved deep ensemble submission: {SUBMISSION_ENSEMBLE_PATH}")
    reporter.emit(
        "Recommended submission file: "
        + ("submission_deep_feature_ensemble.csv" if best_ensemble_row["RMSE"] <= best_deep["metrics"]["RMSE"] else "submission_deep_feature.csv")
    )
    reporter.emit(
        "Leakage confirmation: the model only uses calendar, promotion schedule/context, inventory as-of, and lagged/rolling Revenue features computed from observed history plus recursive predictions. No same-day realized demand, future Revenue, or future COGS is used."
    )

    reporter.save(REPORT_PATH)


if __name__ == "__main__":
    run()
