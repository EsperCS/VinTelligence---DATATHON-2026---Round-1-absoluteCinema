from __future__ import annotations

import logging
import re
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
SYNTHETIC_PROMOTIONS_PATH = DATA_DIR / "synthetic_promotions_2023_2024.csv"
FUTURE_PROMO_FEATURES_PATH = DATA_DIR / "future_promo_calendar_features.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"

BASE_SUBMISSION_PATH = DATA_DIR / "submission_spike_gate_aggressive.csv"
SPIKE_GATE_VALIDATION_PATH = DATA_DIR / "spike_gate_validation_predictions.csv"
SPIKE_GATE_SEARCH_PATH = DATA_DIR / "spike_gate_search_results.csv"

SUBMISSION_SAFE_10_PATH = DATA_DIR / "submission_safe_feature_10.csv"
SUBMISSION_SAFE_15_PATH = DATA_DIR / "submission_safe_feature_15.csv"
SUBMISSION_SAFE_20_PATH = DATA_DIR / "submission_safe_feature_20.csv"

FEATURE_IMPORTANCE_PATH = DATA_DIR / "safe_feature_importance.csv"
VALIDATION_PREDICTIONS_PATH = DATA_DIR / "safe_feature_validation_predictions.csv"
REPORT_PATH = LOG_DIR / "safe_feature_report.txt"
LOG_FILE = LOG_DIR / "train_safe_feature_model.log"

DATE_COL = base.DATE_COL
TARGET_COL = base.TARGET_COL
COGS_COL = base.COGS_COL
TRAIN_CUTOFF = base.TRAIN_CUTOFF
VALIDATION_END = base.VALIDATION_END
RANDOM_STATE = base.RANDOM_STATE

EPSILON = 1e-6
TOP_FEATURE_MIN = 38
TOP_FEATURE_MAX = 55
CORRELATION_THRESHOLD = 0.995
SAFE_BLEND_WEIGHTS = [0.10, 0.15, 0.20]

CAMPAIGN_FLAG_MAP = {
    "spring sale": "is_spring_sale",
    "mid-year sale": "is_midyear_sale",
    "fall launch": "is_fall_launch",
    "year-end sale": "is_year_end_sale",
    "urban blowout": "is_urban_blowout",
    "rural special": "is_rural_special",
}
CAMPAIGN_FLAG_COLUMNS = list(CAMPAIGN_FLAG_MAP.values())

PROMO_CONTEXT_COLUMNS = [
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
] + CAMPAIGN_FLAG_COLUMNS

SAFE_NEW_FEATURES = [
    "trend_7_30",
    "trend_30_90",
    "trend_90_365",
    "volatility_30",
    "volatility_90",
    "volatility_365",
    "volatility_ratio_30_90",
    "spike_strength_365",
    "lag365_above_p90",
    "lag365_above_p95",
    "last_year_same_campaign_revenue",
    "avg_campaign_revenue_last_year",
    "campaign_strength_index",
    "is_early_phase",
    "is_peak_phase",
    "is_late_phase",
    "seasonal_phase",
    "seasonal_phase_cos",
] + [f"campaign_memory_{column}" for column in CAMPAIGN_FLAG_COLUMNS]

SAFE_INTERACTIONS = [
    "trend_30_90_x_calendar_any_promo",
    "spike_strength_365_x_calendar_avg_discount_value",
    "seasonal_phase_x_calendar_avg_discount_value",
    "volatility_30_x_calendar_any_promo",
]

CANDIDATE_FEATURES = BASE_SAFE_FEATURES + SAFE_NEW_FEATURES + SAFE_INTERACTIONS


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

    logger = logging.getLogger("train_safe_feature_model")
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


def normalize_campaign_name(name: Any) -> str:
    text = str(name).strip()
    text = re.sub(r"\s+\d{4}$", "", text)
    return text.lower().strip()


def load_promotions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    promotions = pd.read_csv(path, low_memory=False)
    promotions["start_date"] = pd.to_datetime(promotions["start_date"], errors="coerce").dt.normalize()
    promotions["end_date"] = pd.to_datetime(promotions["end_date"], errors="coerce").dt.normalize()
    promotions = promotions.dropna(subset=["start_date", "end_date"]).copy()
    if promotions.empty:
        return promotions

    promotions["promo_name_base"] = promotions.get("promo_name", "").map(normalize_campaign_name)
    promotions["source_year"] = promotions["start_date"].dt.year.astype(int)
    promotions["duration_days"] = (promotions["end_date"] - promotions["start_date"]).dt.days + 1
    promotions = promotions.sort_values(["source_year", "start_date", "end_date"]).reset_index(drop=True)
    promotions["campaign_index"] = promotions.groupby("source_year").cumcount() + 1
    return promotions


def build_campaign_flag_calendar(dates: pd.Series, promotions: pd.DataFrame) -> pd.DataFrame:
    calendar = pd.DataFrame({DATE_COL: pd.to_datetime(dates).sort_values().unique()})
    for column in CAMPAIGN_FLAG_COLUMNS:
        calendar[column] = 0.0

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
        flag_name = CAMPAIGN_FLAG_MAP.get(str(row.promo_name_base).lower())
        if flag_name is None:
            continue
        for active_date in pd.date_range(active_start, active_end, freq="D"):
            rows.append({DATE_COL: active_date, flag_name: 1.0})

    if not rows:
        return calendar

    expanded = pd.DataFrame(rows)
    aggregated = expanded.groupby(DATE_COL, as_index=False).max()
    return calendar.drop(columns=CAMPAIGN_FLAG_COLUMNS).merge(aggregated, on=DATE_COL, how="left").fillna(0.0)


def build_historical_promo_context(dates: pd.Series, promotions: pd.DataFrame) -> pd.DataFrame:
    calendar = pd.DataFrame({DATE_COL: pd.to_datetime(dates).sort_values().unique()})
    for column in PROMO_CONTEXT_COLUMNS:
        calendar[column] = 0.0

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
            day_number = (active_date - row.start_date).days + 1
            days_remaining = (row.end_date - active_date).days
            rows.append(
                {
                    DATE_COL: active_date,
                    "promo_duration": float(row.duration_days),
                    "promo_day_number": float(day_number),
                    "promo_days_remaining": float(days_remaining),
                    "promo_progress_ratio": float(day_number / row.duration_days),
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
    return calendar.drop(columns=PROMO_CONTEXT_COLUMNS).merge(aggregated, on=DATE_COL, how="left").fillna(0.0)


def load_future_promo_features(path: Path) -> pd.DataFrame:
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
    promotions = load_promotions(PROMOTIONS_PATH)
    calendar = base.build_calendar_features(train_df[DATE_COL], train_df[DATE_COL].min())
    promo = base.build_promotion_calendar(train_df[DATE_COL], PROMOTIONS_PATH, logger)
    promo_context = build_historical_promo_context(train_df[DATE_COL], promotions)
    campaign_flags = build_campaign_flag_calendar(train_df[DATE_COL], promotions)
    inventory = base.build_inventory_asof_features(train_df[DATE_COL], base.INVENTORY_PATH, logger)
    return (
        calendar.merge(promo, on=DATE_COL, how="left", validate="one_to_one")
        .merge(promo_context, on=DATE_COL, how="left", validate="one_to_one")
        .merge(campaign_flags, on=DATE_COL, how="left", validate="one_to_one")
        .merge(inventory, on=DATE_COL, how="left", validate="one_to_one")
        .fillna(0.0)
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )


def build_future_static_features(future_dates: pd.Series, min_date: pd.Timestamp, logger: logging.Logger) -> pd.DataFrame:
    synthetic_promotions = load_promotions(SYNTHETIC_PROMOTIONS_PATH)
    calendar = base.build_calendar_features(future_dates, min_date)
    promo = load_future_promo_features(FUTURE_PROMO_FEATURES_PATH)
    campaign_flags = build_campaign_flag_calendar(future_dates, synthetic_promotions)
    inventory = base.build_inventory_asof_features(future_dates, base.INVENTORY_PATH, logger)
    return (
        calendar.merge(promo, on=DATE_COL, how="left", validate="one_to_one")
        .merge(campaign_flags, on=DATE_COL, how="left", validate="one_to_one")
        .merge(inventory, on=DATE_COL, how="left", validate="one_to_one")
        .fillna(0.0)
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )


def add_safe_features_historical(df: pd.DataFrame) -> pd.DataFrame:
    output = df.sort_values(DATE_COL).reset_index(drop=True).copy()
    shifted_revenue = pd.to_numeric(output[TARGET_COL], errors="coerce").shift(1)
    p90_history = shifted_revenue.expanding(min_periods=1).quantile(0.90)
    p95_history = shifted_revenue.expanding(min_periods=1).quantile(0.95)

    output["trend_7_30"] = safe_divide(output["rolling_mean_7"], output["rolling_mean_30"])
    output["trend_30_90"] = safe_divide(output["rolling_mean_30"], output["revenue_roll_mean_90"])
    output["trend_90_365"] = safe_divide(output["revenue_roll_mean_90"], output["revenue_roll_mean_365"])

    output["volatility_30"] = output["revenue_roll_std_30"]
    output["volatility_90"] = output["revenue_roll_std_90"]
    output["volatility_365"] = output["revenue_roll_std_365"]
    output["volatility_ratio_30_90"] = safe_divide(output["volatility_30"], output["volatility_90"])

    output["spike_strength_365"] = safe_divide(output["revenue_lag_365"], output["revenue_roll_mean_365"])
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

    campaign_any_flag = output[CAMPAIGN_FLAG_COLUMNS].max(axis=1) if CAMPAIGN_FLAG_COLUMNS else output["calendar_any_promo"]
    output["last_year_same_campaign_revenue"] = output["revenue_lag_365"] * campaign_any_flag
    active_count = pd.to_numeric(output["calendar_active_promo_count"], errors="coerce").replace(0, np.nan)
    output["avg_campaign_revenue_last_year"] = safe_divide(output["last_year_same_campaign_revenue"], active_count)
    output["campaign_strength_index"] = safe_divide(output["last_year_same_campaign_revenue"], output["revenue_roll_mean_365"])
    for column in CAMPAIGN_FLAG_COLUMNS:
        output[f"campaign_memory_{column}"] = output["revenue_lag_365"] * output[column]

    progress = pd.to_numeric(output["promo_progress_ratio"], errors="coerce")
    promo_active = pd.to_numeric(output["calendar_any_promo"], errors="coerce").fillna(0.0) > 0
    output["is_early_phase"] = np.where(promo_active & progress.notna(), (progress < 0.3).astype(int), 0.0)
    output["is_peak_phase"] = np.where(
        promo_active & progress.notna(),
        ((progress >= 0.3) & (progress <= 0.7)).astype(int),
        0.0,
    )
    output["is_late_phase"] = np.where(promo_active & progress.notna(), (progress > 0.7).astype(int), 0.0)

    phase = 2.0 * np.pi * output["day_of_year"] / 365.0
    output["seasonal_phase"] = np.sin(phase)
    output["seasonal_phase_cos"] = np.cos(phase)

    output["trend_30_90_x_calendar_any_promo"] = output["trend_30_90"] * output["calendar_any_promo"]
    output["spike_strength_365_x_calendar_avg_discount_value"] = (
        output["spike_strength_365"] * output["calendar_avg_discount_value"]
    )
    output["seasonal_phase_x_calendar_avg_discount_value"] = (
        output["seasonal_phase"] * output["calendar_avg_discount_value"]
    )
    output["volatility_30_x_calendar_any_promo"] = output["volatility_30"] * output["calendar_any_promo"]
    return output


def build_safe_model_table(train_df: pd.DataFrame, static_features: pd.DataFrame) -> pd.DataFrame:
    table = base.build_historical_model_table(train_df, static_features, include_business_lag365=False)
    return add_safe_features_historical(table)


def compute_threshold_bundle(history: pd.Series) -> dict[str, float]:
    ordered = pd.to_numeric(history, errors="coerce").dropna().sort_index()
    if ordered.empty:
        return {"p90": 0.0, "p95": 0.0}
    return {
        "p90": float(ordered.quantile(0.90)),
        "p95": float(ordered.quantile(0.95)),
    }


def compute_safe_features_from_row(row: dict[str, float], thresholds: dict[str, float]) -> dict[str, float]:
    trend_7_30 = safe_divide(row.get("rolling_mean_7"), row.get("rolling_mean_30"))
    trend_30_90 = safe_divide(row.get("rolling_mean_30"), row.get("revenue_roll_mean_90"))
    trend_90_365 = safe_divide(row.get("revenue_roll_mean_90"), row.get("revenue_roll_mean_365"))

    volatility_30 = row.get("revenue_roll_std_30", np.nan)
    volatility_90 = row.get("revenue_roll_std_90", np.nan)
    volatility_365 = row.get("revenue_roll_std_365", np.nan)
    volatility_ratio_30_90 = safe_divide(volatility_30, volatility_90)

    lag365 = row.get("revenue_lag_365", np.nan)
    spike_strength_365 = safe_divide(lag365, row.get("revenue_roll_mean_365"))

    campaign_any_flag = float(max(row.get(column, 0.0) or 0.0 for column in CAMPAIGN_FLAG_COLUMNS))
    active_promo_count = row.get("calendar_active_promo_count", np.nan)
    last_year_same_campaign_revenue = lag365 * campaign_any_flag if pd.notna(lag365) else np.nan
    avg_campaign_revenue_last_year = safe_divide(last_year_same_campaign_revenue, active_promo_count)
    campaign_strength_index = safe_divide(last_year_same_campaign_revenue, row.get("revenue_roll_mean_365"))

    progress = row.get("promo_progress_ratio", np.nan)
    promo_any = float(row.get("calendar_any_promo", 0.0) or 0.0)
    day_of_year = float(row.get("day_of_year", 0.0) or 0.0)
    phase = 2.0 * np.pi * day_of_year / 365.0
    features = {
        "trend_7_30": trend_7_30,
        "trend_30_90": trend_30_90,
        "trend_90_365": trend_90_365,
        "volatility_30": volatility_30,
        "volatility_90": volatility_90,
        "volatility_365": volatility_365,
        "volatility_ratio_30_90": volatility_ratio_30_90,
        "spike_strength_365": spike_strength_365,
        "lag365_above_p90": float(int(pd.notna(lag365) and lag365 >= thresholds["p90"])) if pd.notna(lag365) else np.nan,
        "lag365_above_p95": float(int(pd.notna(lag365) and lag365 >= thresholds["p95"])) if pd.notna(lag365) else np.nan,
        "last_year_same_campaign_revenue": last_year_same_campaign_revenue,
        "avg_campaign_revenue_last_year": avg_campaign_revenue_last_year,
        "campaign_strength_index": campaign_strength_index,
        "is_early_phase": float(int(promo_any > 0 and pd.notna(progress) and progress < 0.3)),
        "is_peak_phase": float(int(promo_any > 0 and pd.notna(progress) and 0.3 <= progress <= 0.7)),
        "is_late_phase": float(int(promo_any > 0 and pd.notna(progress) and progress > 0.7)),
        "seasonal_phase": float(np.sin(phase)),
        "seasonal_phase_cos": float(np.cos(phase)),
        "trend_30_90_x_calendar_any_promo": trend_30_90 * promo_any if pd.notna(trend_30_90) else np.nan,
        "spike_strength_365_x_calendar_avg_discount_value": (
            spike_strength_365 * float(row.get("calendar_avg_discount_value", 0.0) or 0.0)
            if pd.notna(spike_strength_365)
            else np.nan
        ),
        "seasonal_phase_x_calendar_avg_discount_value": float(np.sin(phase))
        * float(row.get("calendar_avg_discount_value", 0.0) or 0.0),
        "volatility_30_x_calendar_any_promo": volatility_30 * promo_any if pd.notna(volatility_30) else np.nan,
    }
    for column in CAMPAIGN_FLAG_COLUMNS:
        features[f"campaign_memory_{column}"] = lag365 * float(row.get(column, 0.0) or 0.0) if pd.notna(lag365) else np.nan
    return features


def build_candidate_feature_list(model_table: pd.DataFrame) -> list[str]:
    blocked = {DATE_COL, TARGET_COL, COGS_COL}
    blocked.update(base.UNSAFE_SAME_DAY_COLUMNS)
    return [
        feature
        for feature in deduplicate_preserve_order(CANDIDATE_FEATURES)
        if feature in model_table.columns and feature not in blocked
    ]


def train_light_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    reporter: Reporter,
) -> tuple[Any, str]:
    if base.lightgbm_available():
        import lightgbm as lgb

        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.03,
            "max_depth": 6,
            "num_leaves": 24,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "feature_fraction": 0.85,
            "min_data_in_leaf": 25,
            "seed": RANDOM_STATE,
            "verbosity": -1,
            "force_col_wise": True,
        }
        dataset = lgb.Dataset(
            X_train,
            label=y_train,
            feature_name=X_train.columns.tolist(),
            free_raw_data=False,
        )
        model = lgb.train(params=params, train_set=dataset, num_boost_round=1200)
        reporter.logger.info("Trained LightGBM light model rows=%s features=%s", len(X_train), X_train.shape[1])
        return model, "lightgbm"

    try:
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError as exc:
        raise ImportError("LightGBM unavailable and sklearn GradientBoostingRegressor not installed.") from exc

    model = GradientBoostingRegressor(
        learning_rate=0.03,
        n_estimators=800,
        max_depth=3,
        subsample=0.85,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    reporter.logger.info("Trained GradientBoostingRegressor rows=%s features=%s", len(X_train), X_train.shape[1])
    return model, "gradient_boosting"


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
    feature_medians = X.median(numeric_only=True)
    return X, y, clean, feature_medians


def get_feature_importance(model: Any, model_type: str, feature_columns: list[str]) -> pd.DataFrame:
    if model_type == "lightgbm":
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

    if hasattr(model, "feature_importances_"):
        return (
            pd.DataFrame(
                {
                    "feature": feature_columns,
                    "importance_split": np.nan,
                    "importance_gain": np.asarray(model.feature_importances_, dtype=float),
                }
            )
            .sort_values("importance_gain", ascending=False)
            .reset_index(drop=True)
        )

    return pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_split": np.nan,
            "importance_gain": np.nan,
        }
    )


def select_features(
    model_table: pd.DataFrame,
    candidate_features: list[str],
    reporter: Reporter,
) -> tuple[list[str], pd.DataFrame]:
    X_train, y_train, clean, _ = make_training_matrix(model_table, candidate_features, TRAIN_CUTOFF)
    reporter.emit(
        f"Warm-up feature selection: rows={len(X_train):,}, candidate_features={len(candidate_features)}"
    )
    warm_model, warm_type = train_light_model(X_train, y_train, reporter)
    importance = get_feature_importance(warm_model, warm_type, candidate_features)
    importance["selected_after_correlation"] = 0

    ranked = importance[importance["importance_gain"].fillna(0) > 0]["feature"].tolist()
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
    top10_mask = actual_values >= top10_threshold
    non_spike_mask = actual_values < top10_threshold

    def masked_rmse(mask: np.ndarray) -> float:
        return float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan

    return {
        "top10_RMSE": masked_rmse(top10_mask),
        "top10_underprediction": int(np.sum(error[top10_mask] > 0)) if top10_mask.any() else 0,
        "top10_count": int(np.sum(top10_mask)),
        "non_spike_RMSE": masked_rmse(non_spike_mask),
    }


def evaluate_candidate(name: str, actual: pd.Series, predictions: np.ndarray) -> dict[str, Any]:
    overall = base.evaluate_predictions(actual, predictions)
    spike_metrics = compute_spike_metrics(actual, predictions)
    return {"model": name, **overall, **spike_metrics}


def recursive_predict_safe(
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
        row.update(compute_safe_features_from_row(row, thresholds))

        X_row = pd.DataFrame([row], columns=feature_columns)
        X_row = X_row.apply(pd.to_numeric, errors="coerce").fillna(feature_medians).fillna(0.0)

        prediction = float(model.predict(X_row)[0])
        prediction = max(0.0, prediction)
        predictions.append(prediction)
        history.loc[forecast_date] = prediction

    return np.asarray(predictions, dtype=float)


def gate_values(probabilities: np.ndarray, mode: str, threshold: float) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    if mode == "hard":
        return (probs >= threshold).astype(float)
    return np.where(probs >= threshold, probs, 0.0)


def choose_aggressive_config(search_df: pd.DataFrame) -> dict[str, Any]:
    search_df = search_df.copy().sort_values(
        ["accepted", "RMSE", "top10_RMSE", "top10_underprediction", "non_spike_RMSE", "MAE"],
        ascending=[False, True, True, True, True, True],
    )
    best = search_df.iloc[0]
    same_rule = search_df[
        (search_df["label_name"] == best["label_name"])
        & (search_df["gating_mode"] == best["gating_mode"])
        & (np.isclose(search_df["threshold"], float(best["threshold"])))
        & (search_df["uplift"] > float(best["uplift"]))
    ].sort_values(["uplift", "RMSE"])
    if not same_rule.empty:
        return same_rule.iloc[0].to_dict()
    return best.to_dict()


def load_base_validation_predictions() -> pd.DataFrame:
    if not SPIKE_GATE_VALIDATION_PATH.exists():
        raise FileNotFoundError(f"Missing spike gate validation predictions: {SPIKE_GATE_VALIDATION_PATH}")
    if not SPIKE_GATE_SEARCH_PATH.exists():
        raise FileNotFoundError(f"Missing spike gate search results: {SPIKE_GATE_SEARCH_PATH}")

    validation = pd.read_csv(SPIKE_GATE_VALIDATION_PATH, parse_dates=[DATE_COL], low_memory=False)
    validation[DATE_COL] = pd.to_datetime(validation[DATE_COL], errors="coerce").dt.normalize()
    search_df = pd.read_csv(SPIKE_GATE_SEARCH_PATH)
    aggressive = choose_aggressive_config(search_df)
    label_name = str(aggressive["label_name"])
    prob_column = f"prob_{label_name}"
    if prob_column not in validation.columns:
        raise ValueError(f"Missing probability column {prob_column} in spike gate validation file")

    gate = gate_values(
        validation[prob_column].to_numpy(dtype=float),
        str(aggressive["gating_mode"]),
        float(aggressive["threshold"]),
    )
    adjusted_pred = validation["base_pred"].to_numpy(dtype=float) * (1.0 + float(aggressive["uplift"]) * gate)
    return pd.DataFrame(
        {
            DATE_COL: validation[DATE_COL],
            "actual_Revenue": pd.to_numeric(validation["actual_Revenue"], errors="coerce"),
            "base_pred": adjusted_pred,
            "base_label": label_name,
            "base_mode": aggressive["gating_mode"],
            "base_threshold": aggressive["threshold"],
            "base_uplift": aggressive["uplift"],
        }
    )


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
        raise FileNotFoundError(f"Base submission not found: {path}")
    output = pd.read_csv(path, parse_dates=[DATE_COL], low_memory=False)
    output[DATE_COL] = pd.to_datetime(output[DATE_COL], errors="coerce").dt.normalize()
    validate_submission_frame(output[[DATE_COL, TARGET_COL, COGS_COL]], sample_submission)
    return output[[DATE_COL, TARGET_COL, COGS_COL]].copy()


def save_blend_submission(
    sample_submission: pd.DataFrame,
    base_submission: pd.DataFrame,
    safe_revenue_pred: np.ndarray,
    safe_cogs_ratio: float,
    safe_weight: float,
    path: Path,
) -> pd.DataFrame:
    safe_frame = sample_submission[[DATE_COL]].copy()
    safe_frame[TARGET_COL] = np.maximum(0.0, np.asarray(safe_revenue_pred, dtype=float))
    safe_frame[COGS_COL] = np.maximum(0.0, safe_frame[TARGET_COL] * safe_cogs_ratio)

    output = sample_submission[[DATE_COL]].copy()
    output[TARGET_COL] = (1.0 - safe_weight) * base_submission[TARGET_COL] + safe_weight * safe_frame[TARGET_COL]
    output[COGS_COL] = (1.0 - safe_weight) * base_submission[COGS_COL] + safe_weight * safe_frame[COGS_COL]
    output[TARGET_COL] = output[TARGET_COL].clip(lower=0.0)
    output[COGS_COL] = output[COGS_COL].clip(lower=0.0)
    validate_submission_frame(output, sample_submission)
    output.to_csv(path, index=False)
    return output


def run() -> None:
    logger = setup_logging()
    reporter = Reporter(logger)

    reporter.emit("Safe Feature Revenue Model")
    reporter.emit("==========================")
    reporter.emit("")

    reporter.emit("1. Load data and rebuild safe static features")
    train_df = base.load_train_data(TRAIN_DATA_PATH)
    sample_submission = base.load_sample_submission(SAMPLE_SUBMISSION_PATH)
    historical_static = build_historical_static_features(train_df, logger)
    future_static = build_future_static_features(sample_submission[DATE_COL], train_df[DATE_COL].min(), logger)
    model_table = build_safe_model_table(train_df, historical_static)
    candidate_features = build_candidate_feature_list(model_table)
    reporter.emit(f"Historical rows: {len(train_df):,}")
    reporter.emit(f"Candidate safe feature count: {len(candidate_features)}")

    reporter.emit("")
    reporter.emit("2. Select stable feature subset")
    selected_features, warm_importance = select_features(model_table, candidate_features, reporter)
    reporter.emit(f"Selected stable feature count: {len(selected_features)}")
    reporter.emit_frame(
        "Top 25 warm-up features:",
        warm_importance.head(25)[["feature", "importance_gain", "selected_after_correlation"]],
    )

    reporter.emit("")
    reporter.emit("3. Train LIGHT safe model and validate recursively on 2022")
    X_train, y_train, train_clean, feature_medians = make_training_matrix(model_table, selected_features, TRAIN_CUTOFF)
    model, model_type = train_light_model(X_train, y_train, reporter)
    validation_dates = train_df[
        (train_df[DATE_COL] >= TRAIN_CUTOFF) & (train_df[DATE_COL] <= VALIDATION_END)
    ][DATE_COL]
    actual = train_df.set_index(DATE_COL).loc[validation_dates, TARGET_COL].reset_index(drop=True)
    initial_history = train_df[train_df[DATE_COL] < TRAIN_CUTOFF].set_index(DATE_COL)[TARGET_COL]
    thresholds = compute_threshold_bundle(initial_history)
    safe_validation_pred = recursive_predict_safe(
        model=model,
        prediction_dates=validation_dates,
        feature_columns=selected_features,
        static_features=historical_static,
        initial_revenue_history=initial_history,
        feature_medians=feature_medians,
        thresholds=thresholds,
    )
    safe_metrics = evaluate_candidate("SAFE_FEATURE_MODEL", actual, safe_validation_pred)
    reporter.emit_frame("Safe model validation metrics:", pd.DataFrame([safe_metrics]))

    reporter.emit("")
    reporter.emit("4. Reconstruct current best base validation and blend lightly")
    base_validation = load_base_validation_predictions().sort_values(DATE_COL).reset_index(drop=True)
    base_metrics = evaluate_candidate(
        "CURRENT_BASE_AGGRESSIVE",
        base_validation["actual_Revenue"],
        base_validation["base_pred"].to_numpy(dtype=float),
    )
    reporter.emit_frame("Base aggressive validation metrics:", pd.DataFrame([base_metrics]))

    validation_output = pd.DataFrame(
        {
            DATE_COL: base_validation[DATE_COL],
            "actual_Revenue": base_validation["actual_Revenue"],
            "base_pred": base_validation["base_pred"],
            "safe_pred": safe_validation_pred,
        }
    )

    blend_rows: list[dict[str, Any]] = []
    for safe_weight in SAFE_BLEND_WEIGHTS:
        blend_name = f"BLEND_SAFE_{int(round(safe_weight * 100)):02d}"
        blended = (
            (1.0 - safe_weight) * base_validation["base_pred"].to_numpy(dtype=float)
            + safe_weight * safe_validation_pred
        )
        validation_output[f"blend_{int(round(safe_weight * 100)):02d}"] = blended
        metrics = evaluate_candidate(blend_name, base_validation["actual_Revenue"], blended)
        metrics["safe_weight"] = safe_weight
        blend_rows.append(metrics)

    blend_comparison = pd.DataFrame(blend_rows).sort_values(["RMSE", "MAE"]).reset_index(drop=True)
    reporter.emit_frame("Light blends vs base:", blend_comparison)

    reporter.emit("")
    reporter.emit("5. Retrain safe model on full 2012-2022 and build future blends")
    X_all, y_all, train_clean_all, feature_medians_all = make_training_matrix(model_table, selected_features, None)
    full_model, full_model_type = train_light_model(X_all, y_all, reporter)
    del full_model_type
    full_thresholds = compute_threshold_bundle(train_clean_all.set_index(DATE_COL)[TARGET_COL])
    safe_future_pred = recursive_predict_safe(
        model=full_model,
        prediction_dates=sample_submission[DATE_COL],
        feature_columns=selected_features,
        static_features=future_static,
        initial_revenue_history=train_df.set_index(DATE_COL)[TARGET_COL],
        feature_medians=feature_medians_all,
        thresholds=full_thresholds,
    )

    base_submission = load_submission(BASE_SUBMISSION_PATH, sample_submission)
    safe_cogs_ratio = base.estimate_cogs_ratio(train_df)
    save_blend_submission(
        sample_submission=sample_submission,
        base_submission=base_submission,
        safe_revenue_pred=safe_future_pred,
        safe_cogs_ratio=safe_cogs_ratio,
        safe_weight=0.10,
        path=SUBMISSION_SAFE_10_PATH,
    )
    save_blend_submission(
        sample_submission=sample_submission,
        base_submission=base_submission,
        safe_revenue_pred=safe_future_pred,
        safe_cogs_ratio=safe_cogs_ratio,
        safe_weight=0.15,
        path=SUBMISSION_SAFE_15_PATH,
    )
    save_blend_submission(
        sample_submission=sample_submission,
        base_submission=base_submission,
        safe_revenue_pred=safe_future_pred,
        safe_cogs_ratio=safe_cogs_ratio,
        safe_weight=0.20,
        path=SUBMISSION_SAFE_20_PATH,
    )

    reporter.emit("")
    reporter.emit("6. Save artifacts")
    validation_output.to_csv(VALIDATION_PREDICTIONS_PATH, index=False)

    full_importance = get_feature_importance(full_model, "lightgbm" if base.lightgbm_available() else "gradient_boosting", selected_features)
    importance_output = pd.concat(
        [
            warm_importance.assign(stage="warmup_selection"),
            full_importance.assign(stage="full_train", selected_after_correlation=1),
        ],
        ignore_index=True,
    )
    importance_output["selected_feature"] = importance_output["feature"].isin(selected_features).astype(int)
    importance_output.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    reporter.emit("")
    reporter.emit("7. Final summary")
    best_blend = blend_comparison.iloc[0].to_dict()
    best_weight = float(best_blend["safe_weight"])
    safe_only_vs_base_rmse = base_metrics["RMSE"] - safe_metrics["RMSE"]
    best_blend_vs_base_rmse = base_metrics["RMSE"] - float(best_blend["RMSE"])
    best_blend_vs_base_spike = base_metrics["top10_RMSE"] - float(best_blend["top10_RMSE"])
    best_blend_vs_base_non_spike = base_metrics["non_spike_RMSE"] - float(best_blend["non_spike_RMSE"])

    useful_new_features = [
        feature
        for feature in full_importance["feature"].head(30).tolist()
        if feature in (SAFE_NEW_FEATURES + SAFE_INTERACTIONS)
    ]

    reporter.emit_frame(
        "Top 30 final safe-model features:",
        full_importance.head(30)[["feature", "importance_gain", "importance_split"]],
    )
    reporter.emit(
        "Useful new stable features: "
        + (", ".join(useful_new_features) if useful_new_features else "none from the new safe block reached top 30")
    )
    reporter.emit(
        f"Safe model standalone metrics: MAE={safe_metrics['MAE']:,.2f} | RMSE={safe_metrics['RMSE']:,.2f} | "
        f"R2={safe_metrics['R2']:.6f}"
    )
    reporter.emit(
        f"RMSE improvement vs current base (safe-only): {safe_only_vs_base_rmse:,.2f}"
    )
    reporter.emit(
        f"Best blend weight: base={1.0 - best_weight:.2f} | safe={best_weight:.2f}"
    )
    reporter.emit(
        f"Best blend metrics: MAE={best_blend['MAE']:,.2f} | RMSE={best_blend['RMSE']:,.2f} | "
        f"R2={best_blend['R2']:.6f}"
    )
    reporter.emit(
        f"RMSE improvement vs current base (best blend): {best_blend_vs_base_rmse:,.2f}"
    )
    reporter.emit(
        f"Spike RMSE change vs current base: {best_blend_vs_base_spike:,.2f}"
    )
    reporter.emit(
        f"Non-spike RMSE change vs current base: {best_blend_vs_base_non_spike:,.2f}"
    )

    recommended_file = {
        0.10: SUBMISSION_SAFE_10_PATH.name,
        0.15: SUBMISSION_SAFE_15_PATH.name,
        0.20: SUBMISSION_SAFE_20_PATH.name,
    }.get(best_weight, SUBMISSION_SAFE_10_PATH.name)
    reporter.emit(f"Recommended submission: {recommended_file}")
    reporter.emit(
        "Leakage confirmation: this branch uses only medium-term lagged/rolling Revenue, stable promo schedule/context, campaign flags, inventory as-of, and recursive prediction history. No lag < 7, no same-day realized demand, and no future actual Revenue/COGS were used."
    )

    reporter.save(REPORT_PATH)


if __name__ == "__main__":
    run()
