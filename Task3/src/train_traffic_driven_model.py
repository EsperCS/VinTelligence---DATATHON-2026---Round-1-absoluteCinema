from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import train_final_model as base
import train_promo_known_pipeline as promo_known


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "log"

SALES_PATH = DATA_DIR / "sales.csv"
WEB_TRAFFIC_PATH = DATA_DIR / "web_traffic.csv"
INVENTORY_PATH = DATA_DIR / "inventory.csv"
PROMOTIONS_PATH = DATA_DIR / "promotions.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
FUTURE_PROMO_KNOWN_PATH = DATA_DIR / "future_promo_known_features.csv"
CURRENT_BEST_SUBMISSION_PATH = DATA_DIR / "submission_blend_direct_15_cogs8900.csv"
CURRENT_BEST_VALIDATION_PATH = DATA_DIR / "final_micro_calibration_validation_predictions.csv"
DIRECT_SEASONAL_VALIDATION_PATH = DATA_DIR / "direct_seasonal_validation_predictions.csv"
DIRECT_SEASONAL_IMPORTANCE_PATH = DATA_DIR / "direct_seasonal_feature_importance.csv"
SEGMENT_VALIDATION_PATH = DATA_DIR / "m5_multilevel_validation_predictions.csv"
OPTIONAL_PROMO_SUBMISSION_PATH = DATA_DIR / "submission_promo_known.csv"
OPTIONAL_SEGMENT_SUBMISSION_PATH = DATA_DIR / "submission_m5_segment_bottomup.csv"
OPTIONAL_BASE_SUBMISSION_PATH = DATA_DIR / "submission_cogs_ratio_8900.csv"

CORRELATION_PATH = DATA_DIR / "traffic_lead_lag_correlation.csv"
MODEL_COMPARISON_PATH = DATA_DIR / "traffic_driven_model_comparison.csv"
VALIDATION_PREDICTIONS_PATH = DATA_DIR / "traffic_driven_validation_predictions.csv"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "traffic_driven_feature_importance.csv"
REPORT_PATH = LOG_DIR / "traffic_driven_model_report.txt"
LOG_FILE = LOG_DIR / "train_traffic_driven_model.log"

SUBMISSION_SEASONAL_PATH = DATA_DIR / "submission_traffic_driven_seasonal.csv"
SUBMISSION_CONSERVATIVE_PATH = DATA_DIR / "submission_traffic_driven_conservative.csv"
SUBMISSION_HIGH_PATH = DATA_DIR / "submission_traffic_driven_high_demand.csv"

BLEND_OUTPUTS = {
    0.05: DATA_DIR / "submission_traffic_blend_05.csv",
    0.10: DATA_DIR / "submission_traffic_blend_10.csv",
    0.15: DATA_DIR / "submission_traffic_blend_15.csv",
    0.20: DATA_DIR / "submission_traffic_blend_20.csv",
    0.25: DATA_DIR / "submission_traffic_blend_25.csv",
}
HIGH_BLEND_OUTPUTS = {
    0.10: DATA_DIR / "submission_traffic_high_blend_10.csv",
    0.15: DATA_DIR / "submission_traffic_high_blend_15.csv",
}

DATE_COL = base.DATE_COL
TARGET_COL = base.TARGET_COL
COGS_COL = base.COGS_COL
RANDOM_STATE = base.RANDOM_STATE

VALIDATION_2022 = ("validation_2022", pd.Timestamp("2021-12-31"), pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"))
LONG_FOLDS = [
    ("fold_1", pd.Timestamp("2019-06-30"), pd.Timestamp("2019-07-01"), pd.Timestamp("2020-12-31")),
    ("fold_2", pd.Timestamp("2020-06-30"), pd.Timestamp("2020-07-01"), pd.Timestamp("2021-12-31")),
    ("fold_3", pd.Timestamp("2021-06-30"), pd.Timestamp("2021-07-01"), pd.Timestamp("2022-12-31")),
]
ALL_SCOPES = [VALIDATION_2022] + LONG_FOLDS

HIGH_RISK_MONTHS = {2, 3, 5, 8}
TRAFFIC_SOURCES = ["organic_search", "paid_search", "email_campaign", "social_media", "direct", "referral"]

CALENDAR_FEATURES = ["day_of_week", "day_of_year", "month", "week_of_year", "is_weekend"]
REVENUE_SAFE_FEATURES = [
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
TRAFFIC_FEATURES = [
    "sessions_lag_1",
    "sessions_lag_2",
    "sessions_lag_3",
    "sessions_lag_7",
    "sessions_lag_14",
    "unique_visitors_lag_1",
    "unique_visitors_lag_2",
    "unique_visitors_lag_3",
    "unique_visitors_lag_7",
    "page_views_lag_1",
    "page_views_lag_2",
    "page_views_lag_3",
    "page_views_lag_7",
    "sessions_roll_mean_3",
    "sessions_roll_mean_7",
    "sessions_roll_mean_14",
    "sessions_roll_mean_30",
    "sessions_roll_std_7",
    "sessions_roll_std_30",
    "page_views_roll_mean_7",
    "unique_visitors_roll_mean_7",
    "sessions_growth_1_7",
    "sessions_growth_3_14",
    "traffic_spike_125",
    "traffic_spike_150",
    "traffic_acceleration",
    "pageview_per_session_lag_1",
    "visitor_to_session_ratio_lag_1",
    "engagement_index",
    "source_diversity_count_lag_1",
    "avg_bounce_rate_lag_1",
    "avg_session_duration_sec_lag_1",
]
SOURCE_TRAFFIC_FEATURES = []
for source in TRAFFIC_SOURCES:
    SOURCE_TRAFFIC_FEATURES.extend([f"{source}_sessions_lag_1", f"{source}_sessions_lag_7", f"{source}_sessions_roll_mean_7"])

PROMO_FEATURES = [
    "calendar_any_promo",
    "calendar_avg_discount_value",
    "calendar_active_promo_count",
    "promo_progress_ratio",
    "promo_days_remaining",
    "campaign_intensity",
    "discount_x_progress",
    "discount_x_days_remaining",
] + promo_known.CAMPAIGN_FEATURES

INVENTORY_BASE_FEATURES = [
    "inv_avg_days_of_supply",
    "inv_min_days_of_supply",
    "inv_avg_sell_through_rate",
    "inv_stockout_rate",
    "inv_reorder_rate",
    "inv_fill_rate_mean",
]
INVENTORY_DERIVED_FEATURES = [
    "low_stock_flag",
    "stock_pressure",
    "sessions_growth_x_promo_active",
    "traffic_spike125_x_promo_active",
    "traffic_spike125_x_discount",
    "low_stock_x_traffic_spike125",
    "stock_pressure_x_traffic_growth",
]

MODEL_FEATURE_SETS = {
    "A_baseline_no_traffic": CALENDAR_FEATURES + PROMO_FEATURES + REVENUE_SAFE_FEATURES,
    "B_traffic_only": CALENDAR_FEATURES + PROMO_FEATURES + TRAFFIC_FEATURES + SOURCE_TRAFFIC_FEATURES,
    "C_traffic_plus_revenue": CALENDAR_FEATURES + PROMO_FEATURES + REVENUE_SAFE_FEATURES + TRAFFIC_FEATURES + SOURCE_TRAFFIC_FEATURES,
    "D_traffic_revenue_inventory_carry": CALENDAR_FEATURES + PROMO_FEATURES + REVENUE_SAFE_FEATURES + TRAFFIC_FEATURES + SOURCE_TRAFFIC_FEATURES + INVENTORY_BASE_FEATURES + INVENTORY_DERIVED_FEATURES,
    "E_traffic_revenue_inventory_monthavg": CALENDAR_FEATURES + PROMO_FEATURES + REVENUE_SAFE_FEATURES + TRAFFIC_FEATURES + SOURCE_TRAFFIC_FEATURES + INVENTORY_BASE_FEATURES + INVENTORY_DERIVED_FEATURES,
}
CORRECTION_FEATURES = [
    "base_pred",
    "base_pred_rank_pct",
    "base_pred_log1p",
    "calendar_any_promo",
    "calendar_avg_discount_value",
    "calendar_active_promo_count",
    "promo_progress_ratio",
    "promo_days_remaining",
    "campaign_intensity",
    "sessions_growth_1_7",
    "sessions_growth_3_14",
    "traffic_spike_125",
    "traffic_spike_150",
    "traffic_acceleration",
    "engagement_index",
    "low_stock_flag",
    "stock_pressure",
    "lag_7",
    "lag_30",
    "lag_365",
    "rolling_mean_30",
    "day_of_week",
    "day_of_year",
    "month",
]


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
    logger = logging.getLogger("train_traffic_driven_model")
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


def safe_divide(numerator: float, denominator: float, fill: float = np.nan) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or abs(float(denominator)) < 1e-9:
        return fill
    return float(numerator) / float(denominator)


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


def load_sales() -> pd.DataFrame:
    sales = pd.read_csv(SALES_PATH, parse_dates=[DATE_COL], low_memory=False)
    sales[DATE_COL] = pd.to_datetime(sales[DATE_COL], errors="coerce").dt.normalize()
    sales = sales.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    sales[TARGET_COL] = pd.to_numeric(sales[TARGET_COL], errors="coerce")
    sales[COGS_COL] = pd.to_numeric(sales[COGS_COL], errors="coerce")
    return sales


def load_web_traffic() -> pd.DataFrame:
    web = pd.read_csv(WEB_TRAFFIC_PATH, parse_dates=["date"], low_memory=False)
    web["date"] = pd.to_datetime(web["date"], errors="coerce").dt.normalize()
    return web.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def load_sample_submission() -> pd.DataFrame:
    sample = pd.read_csv(SAMPLE_SUBMISSION_PATH, parse_dates=[DATE_COL], low_memory=False)
    sample[DATE_COL] = pd.to_datetime(sample[DATE_COL], errors="coerce").dt.normalize()
    return sample.sort_values(DATE_COL).reset_index(drop=True)


def load_promotions() -> pd.DataFrame:
    return promo_known.load_promotions(PROMOTIONS_PATH)


def build_web_daily(web: pd.DataFrame) -> pd.DataFrame:
    df = web.copy()
    numeric_columns = ["sessions", "unique_visitors", "page_views", "bounce_rate", "avg_session_duration_sec"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    df["traffic_source"] = df["traffic_source"].astype(str).str.strip().str.lower()
    df["weighted_bounce"] = df["bounce_rate"] * df["sessions"]
    df["weighted_duration"] = df["avg_session_duration_sec"] * df["sessions"]

    base_daily = df.groupby("date", as_index=False).agg(
        sessions_sum=("sessions", "sum"),
        unique_visitors_sum=("unique_visitors", "sum"),
        page_views_sum=("page_views", "sum"),
        weighted_bounce=("weighted_bounce", "sum"),
        weighted_duration=("weighted_duration", "sum"),
        source_diversity_count=("traffic_source", "nunique"),
    )
    base_daily["avg_bounce_rate"] = np.where(
        base_daily["sessions_sum"] > 1e-9,
        base_daily["weighted_bounce"] / base_daily["sessions_sum"],
        0.0,
    )
    base_daily["avg_session_duration_sec"] = np.where(
        base_daily["sessions_sum"] > 1e-9,
        base_daily["weighted_duration"] / base_daily["sessions_sum"],
        0.0,
    )
    base_daily = base_daily.drop(columns=["weighted_bounce", "weighted_duration"])

    source_pivot = (
        df.pivot_table(index="date", columns="traffic_source", values="sessions", aggfunc="sum", fill_value=0.0)
        .rename(columns={source: f"{source}_sessions" for source in df["traffic_source"].dropna().unique()})
        .reset_index()
    )
    daily = base_daily.merge(source_pivot, on="date", how="left").fillna(0.0)
    daily = daily.rename(columns={"date": DATE_COL}).sort_values(DATE_COL).reset_index(drop=True)
    return daily


def add_traffic_features(raw_traffic: pd.DataFrame) -> pd.DataFrame:
    traffic = raw_traffic.sort_values(DATE_COL).reset_index(drop=True).copy()
    traffic["sessions_lag_1"] = traffic["sessions_sum"].shift(1)
    traffic["sessions_lag_2"] = traffic["sessions_sum"].shift(2)
    traffic["sessions_lag_3"] = traffic["sessions_sum"].shift(3)
    traffic["sessions_lag_7"] = traffic["sessions_sum"].shift(7)
    traffic["sessions_lag_14"] = traffic["sessions_sum"].shift(14)

    for lag in [1, 2, 3, 7]:
        traffic[f"unique_visitors_lag_{lag}"] = traffic["unique_visitors_sum"].shift(lag)
        traffic[f"page_views_lag_{lag}"] = traffic["page_views_sum"].shift(lag)

    traffic["avg_bounce_rate_lag_1"] = traffic["avg_bounce_rate"].shift(1)
    traffic["avg_session_duration_sec_lag_1"] = traffic["avg_session_duration_sec"].shift(1)
    traffic["source_diversity_count_lag_1"] = traffic["source_diversity_count"].shift(1)

    shifted_sessions = traffic["sessions_sum"].shift(1)
    traffic["sessions_roll_mean_3"] = shifted_sessions.rolling(window=3, min_periods=3).mean()
    traffic["sessions_roll_mean_7"] = shifted_sessions.rolling(window=7, min_periods=7).mean()
    traffic["sessions_roll_mean_14"] = shifted_sessions.rolling(window=14, min_periods=14).mean()
    traffic["sessions_roll_mean_30"] = shifted_sessions.rolling(window=30, min_periods=30).mean()
    traffic["sessions_roll_std_7"] = shifted_sessions.rolling(window=7, min_periods=7).std()
    traffic["sessions_roll_std_30"] = shifted_sessions.rolling(window=30, min_periods=30).std()
    traffic["page_views_roll_mean_7"] = traffic["page_views_sum"].shift(1).rolling(window=7, min_periods=7).mean()
    traffic["unique_visitors_roll_mean_7"] = traffic["unique_visitors_sum"].shift(1).rolling(window=7, min_periods=7).mean()

    mean_lag1_3 = pd.concat([traffic["sessions_lag_1"], traffic["sessions_lag_2"], traffic["sessions_lag_3"]], axis=1).mean(axis=1)
    traffic["sessions_growth_1_7"] = traffic["sessions_lag_1"] / traffic["sessions_roll_mean_7"].replace(0, np.nan)
    traffic["sessions_growth_3_14"] = mean_lag1_3 / traffic["sessions_roll_mean_14"].replace(0, np.nan)
    traffic["traffic_spike_125"] = (traffic["sessions_growth_1_7"] > 1.25).astype(float)
    traffic["traffic_spike_150"] = (traffic["sessions_growth_1_7"] > 1.50).astype(float)
    traffic["traffic_acceleration"] = traffic["sessions_lag_1"] - traffic["sessions_lag_3"]
    traffic["pageview_per_session_lag_1"] = traffic["page_views_lag_1"] / traffic["sessions_lag_1"].replace(0, np.nan)
    traffic["visitor_to_session_ratio_lag_1"] = traffic["unique_visitors_lag_1"] / traffic["sessions_lag_1"].replace(0, np.nan)
    traffic["engagement_index"] = (
        traffic["page_views_lag_1"] * traffic["avg_session_duration_sec_lag_1"] / traffic["avg_bounce_rate_lag_1"].replace(0, np.nan)
    )

    for source in TRAFFIC_SOURCES:
        source_col = f"{source}_sessions"
        if source_col not in traffic.columns:
            traffic[source_col] = 0.0
        traffic[f"{source}_sessions_lag_1"] = traffic[source_col].shift(1)
        traffic[f"{source}_sessions_lag_7"] = traffic[source_col].shift(7)
        traffic[f"{source}_sessions_roll_mean_7"] = traffic[source_col].shift(1).rolling(window=7, min_periods=7).mean()

    numeric_cols = [column for column in traffic.columns if column != DATE_COL]
    traffic[numeric_cols] = traffic[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return traffic


def prepare_inventory_snapshots() -> pd.DataFrame:
    inventory = pd.read_csv(INVENTORY_PATH, low_memory=False)
    inventory["snapshot_date"] = pd.to_datetime(inventory["snapshot_date"], errors="coerce").dt.normalize()
    for column in [
        "days_of_supply",
        "fill_rate",
        "stockout_flag",
        "reorder_flag",
        "sell_through_rate",
    ]:
        inventory[column] = pd.to_numeric(inventory.get(column, 0.0), errors="coerce").fillna(0.0)

    snapshots = (
        inventory.dropna(subset=["snapshot_date"])
        .groupby("snapshot_date", as_index=False)
        .agg(
            inv_avg_days_of_supply=("days_of_supply", "mean"),
            inv_min_days_of_supply=("days_of_supply", "min"),
            inv_avg_sell_through_rate=("sell_through_rate", "mean"),
            inv_stockout_rate=("stockout_flag", "mean"),
            inv_reorder_rate=("reorder_flag", "mean"),
            inv_fill_rate_mean=("fill_rate", "mean"),
        )
        .rename(columns={"snapshot_date": DATE_COL})
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )
    snapshots["month"] = snapshots[DATE_COL].dt.month.astype(int)
    return snapshots


def build_inventory_context(
    dates: pd.Series,
    snapshots: pd.DataFrame,
    mode: str,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    calendar = pd.DataFrame({DATE_COL: pd.to_datetime(dates).sort_values().unique()})
    calendar["month"] = calendar[DATE_COL].dt.month.astype(int)
    available = snapshots.loc[snapshots[DATE_COL] <= cutoff].copy()
    if available.empty:
        for feature in INVENTORY_BASE_FEATURES:
            calendar[feature] = 0.0
        return calendar[[DATE_COL] + INVENTORY_BASE_FEATURES]

    if mode == "carry":
        merged = pd.merge_asof(
            calendar[[DATE_COL]].sort_values(DATE_COL),
            available[[DATE_COL] + INVENTORY_BASE_FEATURES].sort_values(DATE_COL),
            on=DATE_COL,
            direction="backward",
        )
        if merged[INVENTORY_BASE_FEATURES].isna().any().any():
            monthly = available.groupby("month", as_index=False)[INVENTORY_BASE_FEATURES].mean()
            merged = merged.merge(calendar[[DATE_COL, "month"]], on=DATE_COL, how="left")
            merged = merged.merge(monthly, on="month", how="left", suffixes=("", "_month"))
            for feature in INVENTORY_BASE_FEATURES:
                merged[feature] = pd.to_numeric(merged[feature], errors="coerce").fillna(
                    pd.to_numeric(merged[f"{feature}_month"], errors="coerce")
                )
            drop_cols = ["month"] + [f"{feature}_month" for feature in INVENTORY_BASE_FEATURES]
            merged = merged.drop(columns=[column for column in drop_cols if column in merged.columns])
        merged[INVENTORY_BASE_FEATURES] = merged[INVENTORY_BASE_FEATURES].fillna(0.0)
        return merged[[DATE_COL] + INVENTORY_BASE_FEATURES]

    monthly = available.groupby("month", as_index=False)[INVENTORY_BASE_FEATURES].mean()
    merged = calendar.merge(monthly, on="month", how="left")
    merged[INVENTORY_BASE_FEATURES] = merged[INVENTORY_BASE_FEATURES].fillna(0.0)
    return merged[[DATE_COL] + INVENTORY_BASE_FEATURES]


def add_inventory_derived_features(frame: pd.DataFrame, low_stock_threshold: float) -> pd.DataFrame:
    output = frame.copy()
    output["low_stock_flag"] = (pd.to_numeric(output["inv_avg_days_of_supply"], errors="coerce") < low_stock_threshold).astype(float)
    output["stock_pressure"] = output["inv_avg_sell_through_rate"] / output["inv_avg_days_of_supply"].replace(0, np.nan)
    output["sessions_growth_x_promo_active"] = output["sessions_growth_1_7"] * output["calendar_any_promo"]
    output["traffic_spike125_x_promo_active"] = output["traffic_spike_125"] * output["calendar_any_promo"]
    output["traffic_spike125_x_discount"] = output["traffic_spike_125"] * output["calendar_avg_discount_value"]
    output["low_stock_x_traffic_spike125"] = output["low_stock_flag"] * output["traffic_spike_125"]
    output["stock_pressure_x_traffic_growth"] = output["stock_pressure"] * output["sessions_growth_1_7"]
    return output


def build_calendar_context(dates: pd.Series, min_date: pd.Timestamp) -> pd.DataFrame:
    calendar = base.build_calendar_features(dates, min_date)
    return calendar[[DATE_COL] + CALENDAR_FEATURES]


def build_historical_static_context(
    sales: pd.DataFrame,
    traffic_features: pd.DataFrame,
    promo_features: pd.DataFrame,
    inventory_carry: pd.DataFrame,
    inventory_month: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    calendar = build_calendar_context(sales[DATE_COL], sales[DATE_COL].min())
    common = (
        calendar.merge(traffic_features, on=DATE_COL, how="left", validate="one_to_one")
        .merge(promo_features, on=DATE_COL, how="left", validate="one_to_one")
    )
    carry = common.merge(inventory_carry, on=DATE_COL, how="left", validate="one_to_one").fillna(0.0)
    month = common.merge(inventory_month, on=DATE_COL, how="left", validate="one_to_one").fillna(0.0)
    return {"carry": carry, "month_avg": month}


def build_training_table(sales: pd.DataFrame, static_context: pd.DataFrame, low_stock_threshold: float) -> pd.DataFrame:
    revenue_history = base.add_historical_revenue_features(sales[[DATE_COL, TARGET_COL]].copy())
    revenue_history = revenue_history.rename(
        columns={
            "revenue_lag_60": "lag_60_unused",
            "revenue_lag_90": "lag_90",
            "revenue_lag_180": "lag_180",
            "revenue_lag_365": "lag_365",
            "revenue_roll_mean_14": "rolling_mean_14_unused",
            "revenue_roll_mean_60": "rolling_mean_60_unused",
            "revenue_roll_mean_90": "rolling_mean_90",
            "revenue_roll_mean_180": "rolling_mean_180_unused",
            "revenue_roll_mean_365": "rolling_mean_365",
        }
    )
    keep_cols = [
        DATE_COL,
        TARGET_COL,
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
    table = revenue_history[keep_cols].merge(static_context, on=DATE_COL, how="left", validate="one_to_one")
    table = add_inventory_derived_features(table, low_stock_threshold)
    return table


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[Any, str]:
    if base.lightgbm_available():
        import lightgbm as lgb

        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.03,
            "max_depth": 5,
            "num_leaves": 16,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "min_data_in_leaf": 25,
            "seed": RANDOM_STATE,
            "verbosity": -1,
            "force_col_wise": True,
        }
        dataset = lgb.Dataset(X_train, label=y_train, feature_name=X_train.columns.tolist(), free_raw_data=False)
        model = lgb.train(params=params, train_set=dataset, num_boost_round=400)
        return model, "lightgbm"

    try:
        from sklearn.ensemble import ExtraTreesRegressor
    except Exception as exc:
        raise ImportError("LightGBM unavailable and sklearn fallback missing") from exc

    model = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model, "extra_trees"


def predict_model(model: Any, model_type: str, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(X), dtype=float)


def prepare_training_matrix(training_table: pd.DataFrame, feature_list: list[str], train_end: pd.Timestamp) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    subset = training_table.loc[training_table[DATE_COL] <= train_end].copy()
    required_lag = "lag_365" if "lag_365" in feature_list else None
    if required_lag is not None:
        subset = subset.dropna(subset=[required_lag]).copy()
    X_train = subset[feature_list].apply(pd.to_numeric, errors="coerce")
    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)
    y_train = pd.to_numeric(subset[TARGET_COL], errors="coerce")
    mask = ~y_train.isna()
    return X_train.loc[mask].reset_index(drop=True), y_train.loc[mask].reset_index(drop=True), medians


def build_prediction_row(history: pd.DataFrame, target_date: pd.Timestamp, static_row: pd.Series) -> dict[str, float]:
    past = history.loc[history[DATE_COL] < target_date, [DATE_COL, TARGET_COL]].copy().sort_values(DATE_COL).reset_index(drop=True)
    revenue_map = past.set_index(DATE_COL)[TARGET_COL]

    def get_lag(days: int) -> float:
        ref_date = target_date - pd.Timedelta(days=days)
        if ref_date in revenue_map.index:
            return float(pd.to_numeric(revenue_map.loc[ref_date], errors="coerce"))
        return np.nan

    def get_recent_mean(window: int) -> float:
        if len(past) < window:
            return np.nan
        values = pd.to_numeric(past[TARGET_COL].tail(window), errors="coerce")
        return float(values.mean()) if len(values) == window else np.nan

    row = {column: float(pd.to_numeric(static_row.get(column, 0.0), errors="coerce")) for column in static_row.index if column != DATE_COL}
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
    if "low_stock_flag" not in row:
        row["low_stock_flag"] = 0.0
    if "stock_pressure" not in row:
        row["stock_pressure"] = safe_divide(row.get("inv_avg_sell_through_rate", np.nan), row.get("inv_avg_days_of_supply", np.nan), fill=0.0)
    row["sessions_growth_x_promo_active"] = row.get("sessions_growth_1_7", np.nan) * row.get("calendar_any_promo", 0.0)
    row["traffic_spike125_x_promo_active"] = row.get("traffic_spike_125", 0.0) * row.get("calendar_any_promo", 0.0)
    row["traffic_spike125_x_discount"] = row.get("traffic_spike_125", 0.0) * row.get("calendar_avg_discount_value", 0.0)
    row["low_stock_x_traffic_spike125"] = row.get("low_stock_flag", 0.0) * row.get("traffic_spike_125", 0.0)
    row["stock_pressure_x_traffic_growth"] = row.get("stock_pressure", 0.0) * row.get("sessions_growth_1_7", np.nan)
    return row


def recursive_predict_revenue(
    model: Any,
    model_type: str,
    medians: pd.Series,
    feature_list: list[str],
    history: pd.DataFrame,
    static_context: pd.DataFrame,
) -> pd.DataFrame:
    history_frame = history[[DATE_COL, TARGET_COL]].copy().sort_values(DATE_COL).reset_index(drop=True)
    static_index = static_context.set_index(DATE_COL).sort_index()
    rows: list[dict[str, Any]] = []

    for target_date in static_context[DATE_COL]:
        static_row = static_index.loc[target_date]
        feature_row = build_prediction_row(history_frame, target_date, static_row)
        X = pd.DataFrame([feature_row]).reindex(columns=feature_list).apply(pd.to_numeric, errors="coerce").fillna(medians)
        prediction = float(np.clip(predict_model(model, model_type, X)[0], 0.0, None))
        rows.append({DATE_COL: target_date, "predicted_Revenue": prediction})
        history_frame = pd.concat(
            [history_frame, pd.DataFrame({DATE_COL: [target_date], TARGET_COL: [prediction]})],
            ignore_index=True,
        )

    return pd.DataFrame(rows)


def build_current_best_validation_2022() -> pd.DataFrame | None:
    if not CURRENT_BEST_VALIDATION_PATH.exists() or not DIRECT_SEASONAL_VALIDATION_PATH.exists():
        return None
    current_base = pd.read_csv(CURRENT_BEST_VALIDATION_PATH, parse_dates=[DATE_COL], low_memory=False)
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


def compute_metrics(actual: pd.Series, predicted: np.ndarray, promo_mask: pd.Series, high_traffic_mask: pd.Series, low_stock_mask: pd.Series) -> dict[str, float]:
    metrics = base.evaluate_predictions(actual, predicted)
    actual_np = actual.to_numpy(dtype=float)
    pred_np = np.asarray(predicted, dtype=float)
    errors = actual_np - pred_np
    top10_threshold = float(np.quantile(actual_np, 0.90))
    top10_mask = actual_np >= top10_threshold
    promo_np = promo_mask.to_numpy(dtype=bool)
    high_np = high_traffic_mask.to_numpy(dtype=bool)
    low_np = low_stock_mask.to_numpy(dtype=bool)

    metrics["top10_RMSE"] = float(np.sqrt(np.mean(errors[top10_mask] ** 2))) if top10_mask.any() else np.nan
    metrics["top10_underprediction"] = int(np.sum(errors[top10_mask] > 0)) if top10_mask.any() else 0
    metrics["high_traffic_spike_RMSE"] = float(np.sqrt(np.mean(errors[high_np] ** 2))) if high_np.any() else np.nan
    metrics["promo_day_RMSE"] = float(np.sqrt(np.mean(errors[promo_np] ** 2))) if promo_np.any() else np.nan
    metrics["non_promo_RMSE"] = float(np.sqrt(np.mean(errors[~promo_np] ** 2))) if (~promo_np).any() else np.nan
    metrics["low_stock_RMSE"] = float(np.sqrt(np.mean(errors[low_np] ** 2))) if low_np.any() else np.nan
    return metrics


def extract_feature_importance(model: Any, model_type: str, feature_list: list[str]) -> pd.DataFrame:
    if model_type == "lightgbm":
        return pd.DataFrame(
            {
                "feature": model.feature_name(),
                "importance_gain": model.feature_importance(importance_type="gain").astype(float),
                "importance_split": model.feature_importance(importance_type="split").astype(float),
            }
        ).sort_values("importance_gain", ascending=False).reset_index(drop=True)

    if hasattr(model, "feature_importances_"):
        return pd.DataFrame(
            {
                "feature": feature_list,
                "importance_gain": np.asarray(model.feature_importances_, dtype=float),
                "importance_split": np.nan,
            }
        ).sort_values("importance_gain", ascending=False).reset_index(drop=True)

    return pd.DataFrame({"feature": feature_list, "importance_gain": np.nan, "importance_split": np.nan})


def build_promo_uplift_factor(web_daily: pd.DataFrame, promo_features: pd.DataFrame) -> float:
    merged = web_daily.merge(promo_features[[DATE_COL, "calendar_any_promo"]], on=DATE_COL, how="left").fillna(0.0)
    merged["year"] = merged[DATE_COL].dt.year.astype(int)
    merged["month"] = merged[DATE_COL].dt.month.astype(int)
    merged["day"] = merged[DATE_COL].dt.day.astype(int)

    values = []
    for row in merged.itertuples(index=False):
        if int(getattr(row, "calendar_any_promo", 0)) == 0:
            continue
        candidates = merged.loc[
            (merged["calendar_any_promo"] == 0)
            & (merged["month"] == row.month)
            & (merged["day"] == row.day)
            & (merged["year"].isin([row.year - 1, row.year - 2, row.year - 3])),
            "sessions_sum",
        ]
        if not candidates.empty and float(candidates.mean()) > 1e-9:
            values.append(float(row.sessions_sum) / float(candidates.mean()))
    if not values:
        return 1.05
    return float(np.clip(np.median(values), 1.0, 1.25))


def safe_same_day_history(series: pd.Series, target_date: pd.Timestamp, years_weights: list[tuple[int, float]]) -> float:
    values = []
    total_weight = 0.0
    for ref_year, weight in years_weights:
        try:
            ref_date = target_date.replace(year=ref_year)
        except ValueError:
            ref_date = pd.Timestamp(year=ref_year, month=2, day=28)
        if ref_date in series.index:
            value = float(pd.to_numeric(series.loc[ref_date], errors="coerce"))
            if np.isfinite(value):
                values.append(weight * value)
                total_weight += weight
    if total_weight <= 0:
        return np.nan
    return float(sum(values) / total_weight)


def build_future_traffic_raw_scenarios(
    web_daily: pd.DataFrame,
    sample_submission: pd.DataFrame,
    future_promo: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    future_dates = sample_submission[DATE_COL].copy()
    years_weights = [(2022, 0.5), (2021, 0.3), (2020, 0.2)]
    web_indexed = web_daily.set_index(DATE_COL).sort_index()
    month_means = web_daily.assign(month=web_daily[DATE_COL].dt.month.astype(int)).groupby("month").mean(numeric_only=True)
    promo_uplift = build_promo_uplift_factor(web_daily, future_promo.rename(columns={"future_calendar_any_promo": "calendar_any_promo"}) if "future_calendar_any_promo" in future_promo.columns else future_promo)

    metric_columns = [
        "sessions_sum",
        "unique_visitors_sum",
        "page_views_sum",
        "avg_bounce_rate",
        "avg_session_duration_sec",
    ] + [f"{source}_sessions" for source in TRAFFIC_SOURCES]

    rows_a: list[dict[str, Any]] = []
    rows_b: list[dict[str, Any]] = []
    for target_date in future_dates:
        row_a: dict[str, Any] = {DATE_COL: target_date}
        row_b: dict[str, Any] = {DATE_COL: target_date}
        month = int(target_date.month)
        for column in metric_columns:
            base_value = safe_same_day_history(web_indexed[column], target_date, years_weights)
            if pd.isna(base_value):
                base_value = float(month_means.loc[month, column]) if month in month_means.index and column in month_means.columns else 0.0
            value_2022 = safe_same_day_history(web_indexed[column], target_date, [(2022, 1.0)])
            if pd.isna(value_2022):
                value_2022 = float(month_means.loc[month, column]) if month in month_means.index and column in month_means.columns else 0.0
            month_avg = float(month_means.loc[month, column]) if month in month_means.index and column in month_means.columns else value_2022
            row_a[column] = base_value
            row_b[column] = 0.7 * value_2022 + 0.3 * month_avg

        promo_row = future_promo.loc[future_promo[DATE_COL] == target_date]
        promo_active = int(float(promo_row["calendar_any_promo"].iloc[0])) if not promo_row.empty and "calendar_any_promo" in promo_row.columns else 0
        if promo_active == 1:
            row_a["sessions_sum"] *= promo_uplift
            row_a["unique_visitors_sum"] *= promo_uplift
            row_a["page_views_sum"] *= promo_uplift
            for source in TRAFFIC_SOURCES:
                row_a[f"{source}_sessions"] *= promo_uplift

        rows_a.append(row_a)
        rows_b.append(row_b)

    seasonal = pd.DataFrame(rows_a)
    conservative = pd.DataFrame(rows_b)
    for frame in [seasonal, conservative]:
        frame["source_diversity_count"] = (frame[[f"{source}_sessions" for source in TRAFFIC_SOURCES]] > 0).sum(axis=1)
    high = seasonal.copy()
    mask = high[DATE_COL].dt.month.isin(HIGH_RISK_MONTHS) | future_promo["calendar_any_promo"].reset_index(drop=True).fillna(0).astype(bool)
    for column in ["sessions_sum", "unique_visitors_sum", "page_views_sum"] + [f"{source}_sessions" for source in TRAFFIC_SOURCES]:
        high.loc[mask, column] = high.loc[mask, column] * 1.05
    high["source_diversity_count"] = (high[[f"{source}_sessions" for source in TRAFFIC_SOURCES]] > 0).sum(axis=1)

    stats = pd.DataFrame(
        [
            {"scenario": "seasonal", "sessions_mean": seasonal["sessions_sum"].mean(), "sessions_min": seasonal["sessions_sum"].min(), "sessions_max": seasonal["sessions_sum"].max()},
            {"scenario": "conservative", "sessions_mean": conservative["sessions_sum"].mean(), "sessions_min": conservative["sessions_sum"].min(), "sessions_max": conservative["sessions_sum"].max()},
            {"scenario": "high_demand", "sessions_mean": high["sessions_sum"].mean(), "sessions_min": high["sessions_sum"].min(), "sessions_max": high["sessions_sum"].max()},
        ]
    )
    return {"seasonal": seasonal, "conservative": conservative, "high_demand": high}, stats


def build_traffic_lead_lag_correlation(sales: pd.DataFrame, traffic_features: pd.DataFrame) -> pd.DataFrame:
    merged = sales[[DATE_COL, TARGET_COL]].merge(
        traffic_features[[DATE_COL, "sessions_growth_1_7", "traffic_spike_125", "traffic_spike_150"]],
        on=DATE_COL,
        how="inner",
        validate="one_to_one",
    )
    revenue_spike_threshold = float(np.quantile(pd.to_numeric(merged[TARGET_COL], errors="coerce"), 0.90))
    merged["revenue_spike_top10"] = (merged[TARGET_COL] >= revenue_spike_threshold).astype(float)
    rows = []
    for horizon in [1, 2, 3, 7]:
        shifted_revenue = merged[TARGET_COL].shift(-horizon)
        shifted_spike = merged["revenue_spike_top10"].shift(-horizon)
        rows.append(
            {
                "horizon_days": horizon,
                "corr_sessions_growth_to_revenue": float(pd.Series(merged["sessions_growth_1_7"]).corr(shifted_revenue)),
                "corr_traffic_spike125_to_revenue_spike": float(pd.Series(merged["traffic_spike_125"]).corr(shifted_spike)),
                "corr_traffic_spike150_to_revenue_spike": float(pd.Series(merged["traffic_spike_150"]).corr(shifted_spike)),
            }
        )
    return pd.DataFrame(rows)


def evaluate_core_model_on_scope(
    model_name: str,
    feature_list: list[str],
    train_end: pd.Timestamp,
    valid_start: pd.Timestamp,
    valid_end: pd.Timestamp,
    training_table: pd.DataFrame,
    validation_static: pd.DataFrame,
    sales: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    X_train, y_train, medians = prepare_training_matrix(training_table, feature_list, train_end)
    model, model_type = train_model(X_train, y_train)
    history = sales.loc[sales[DATE_COL] <= train_end, [DATE_COL, TARGET_COL]].copy()
    validation_static_window = validation_static.loc[
        (validation_static[DATE_COL] >= valid_start) & (validation_static[DATE_COL] <= valid_end)
    ].copy()
    predictions = recursive_predict_revenue(model, model_type, medians, feature_list, history, validation_static_window)
    actual = sales.loc[(sales[DATE_COL] >= valid_start) & (sales[DATE_COL] <= valid_end), [DATE_COL, TARGET_COL]].copy()
    merged = actual.merge(predictions, on=DATE_COL, how="left", validate="one_to_one")
    merged = merged.merge(
        validation_static_window[[DATE_COL, "calendar_any_promo", "traffic_spike_125", "low_stock_flag"]],
        on=DATE_COL,
        how="left",
        validate="one_to_one",
    )
    metrics = compute_metrics(
        merged[TARGET_COL],
        merged["predicted_Revenue"].to_numpy(dtype=float),
        merged["calendar_any_promo"].fillna(0).astype(int),
        merged["traffic_spike_125"].fillna(0).astype(bool),
        merged["low_stock_flag"].fillna(0).astype(bool),
    )
    metrics.update({"model_name": model_name})
    importance = extract_feature_importance(model, model_type, feature_list)
    importance["model_name"] = model_name
    return metrics, merged, importance


def evaluate_correction_model(
    validation_static_carry: pd.DataFrame,
    current_best_validation: pd.DataFrame,
    sales: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame] | None:
    if current_best_validation is None:
        return None

    revenue_history = base.add_historical_revenue_features(sales[[DATE_COL, TARGET_COL]].copy()).rename(
        columns={
            "revenue_lag_90": "lag_90_unused",
            "revenue_lag_180": "lag_180_unused",
            "revenue_lag_365": "lag_365",
            "revenue_roll_mean_90": "rolling_mean_90_unused",
            "revenue_roll_mean_365": "rolling_mean_365_unused",
        }
    )
    revenue_keep = revenue_history[[DATE_COL, "lag_7", "lag_30", "lag_365", "rolling_mean_30"]].copy()

    eval_frame = current_best_validation.merge(
        validation_static_carry,
        on=DATE_COL,
        how="inner",
        validate="one_to_one",
    ).merge(
        revenue_keep,
        on=DATE_COL,
        how="left",
        validate="one_to_one",
    )
    eval_frame["base_pred"] = pd.to_numeric(eval_frame["current_best_pred"], errors="coerce")
    eval_frame["base_pred_log1p"] = np.log1p(np.clip(eval_frame["base_pred"], 0.0, None))
    eval_frame["base_pred_rank_pct"] = eval_frame["base_pred"].rank(pct=True)
    eval_frame["target_residual"] = pd.to_numeric(eval_frame["actual_Revenue"], errors="coerce") - eval_frame["base_pred"]

    split_date = pd.Timestamp("2022-09-30")
    train_subset = eval_frame.loc[eval_frame[DATE_COL] <= split_date].copy()
    test_subset = eval_frame.loc[eval_frame[DATE_COL] > split_date].copy()
    if train_subset.empty or test_subset.empty:
        return None

    X_train = train_subset[CORRECTION_FEATURES].apply(pd.to_numeric, errors="coerce")
    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)
    y_train = pd.to_numeric(train_subset["target_residual"], errors="coerce").fillna(0.0)
    model, model_type = train_model(X_train, y_train)

    X_test = test_subset[CORRECTION_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(medians)
    residual_pred = predict_model(model, model_type, X_test)
    final_pred = np.clip(pd.to_numeric(test_subset["base_pred"], errors="coerce").to_numpy(dtype=float) + residual_pred, 0.0, None)

    merged = test_subset[[DATE_COL, "actual_Revenue", "base_pred", "calendar_any_promo", "traffic_spike_125", "low_stock_flag"]].copy()
    merged["predicted_Revenue"] = final_pred
    metrics = compute_metrics(
        merged["actual_Revenue"],
        final_pred,
        merged["calendar_any_promo"].fillna(0).astype(int),
        merged["traffic_spike_125"].fillna(0).astype(bool),
        merged["low_stock_flag"].fillna(0).astype(bool),
    )
    metrics.update({"model_name": "F_traffic_spike_correction"})
    importance = extract_feature_importance(model, model_type, CORRECTION_FEATURES)
    importance["model_name"] = "F_traffic_spike_correction"
    return metrics, merged, importance


def build_submission(dates: pd.Series, revenue: pd.Series | np.ndarray, ratio: float = 0.8900) -> pd.DataFrame:
    revenue_series = pd.Series(np.asarray(revenue, dtype=float))
    return pd.DataFrame(
        {
            DATE_COL: pd.to_datetime(dates).reset_index(drop=True),
            TARGET_COL: revenue_series.clip(lower=0.0),
            COGS_COL: revenue_series.clip(lower=0.0) * ratio,
        }
    )


def run() -> None:
    logger = setup_logging()
    reporter = Reporter(logger)
    reporter.emit("Traffic-Driven Demand Forecasting")
    reporter.emit("================================")
    reporter.emit("")

    sales = load_sales()
    web = load_web_traffic()
    sample_submission = load_sample_submission()
    promotions = load_promotions()
    inventory_snapshots = prepare_inventory_snapshots()

    reporter.emit("1. Build daily traffic table and lagged traffic features")
    web_daily = build_web_daily(web)
    traffic_features = add_traffic_features(web_daily)
    reporter.emit_frame(
        "Traffic source totals:",
        pd.DataFrame(
            [
                {"source": source, "sessions_sum": float(web_daily.get(f"{source}_sessions", pd.Series(dtype=float)).sum())}
                for source in TRAFFIC_SOURCES
            ]
        ),
    )

    reporter.emit("")
    reporter.emit("2. Build promo and inventory contexts")
    historical_promo = promo_known.build_daily_promo_known_features(sales[DATE_COL], promotions)
    if FUTURE_PROMO_KNOWN_PATH.exists():
        future_promo = pd.read_csv(FUTURE_PROMO_KNOWN_PATH, parse_dates=[DATE_COL], low_memory=False)
        future_promo[DATE_COL] = pd.to_datetime(future_promo[DATE_COL], errors="coerce").dt.normalize()
    else:
        synthetic_promotions, _ = promo_known.promo_builder.build_synthetic_promotions(promotions)
        future_promo = promo_known.build_daily_promo_known_features(sample_submission[DATE_COL], synthetic_promotions)
    future_promo = promo_known.merge_future_reference_features(future_promo, FUTURE_PROMO_KNOWN_PATH, logger) if FUTURE_PROMO_KNOWN_PATH.exists() else future_promo

    inventory_carry_full = build_inventory_context(sales[DATE_COL], inventory_snapshots, mode="carry", cutoff=sales[DATE_COL].max())
    inventory_month_full = build_inventory_context(sales[DATE_COL], inventory_snapshots, mode="month_avg", cutoff=sales[DATE_COL].max())
    historical_static = build_historical_static_context(sales, traffic_features, historical_promo, inventory_carry_full, inventory_month_full)

    correlation_table = build_traffic_lead_lag_correlation(sales, traffic_features)
    correlation_table.to_csv(CORRELATION_PATH, index=False)
    reporter.emit_frame("Lead-lag traffic vs revenue correlation:", correlation_table)

    reporter.emit("")
    reporter.emit("3. Train and validate model variants")
    comparison_rows: list[dict[str, Any]] = []
    validation_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    validation_2022_predictions: dict[str, pd.DataFrame] = {}

    for scope_name, train_end, valid_start, valid_end in ALL_SCOPES:
        reporter.emit(f"{scope_name}: train <= {train_end.date()}, validate {valid_start.date()} -> {valid_end.date()}")
        inventory_carry_scope = build_inventory_context(sales[DATE_COL], inventory_snapshots, mode="carry", cutoff=train_end)
        inventory_month_scope = build_inventory_context(sales[DATE_COL], inventory_snapshots, mode="month_avg", cutoff=train_end)
        static_scope = build_historical_static_context(sales, traffic_features, historical_promo, inventory_carry_scope, inventory_month_scope)

        low_stock_threshold = float(
            np.nanpercentile(
                pd.to_numeric(
                    inventory_carry_scope.loc[inventory_carry_scope[DATE_COL] <= train_end, "inv_avg_days_of_supply"],
                    errors="coerce",
                ).dropna(),
                25,
            )
        ) if not inventory_carry_scope.loc[inventory_carry_scope[DATE_COL] <= train_end, "inv_avg_days_of_supply"].dropna().empty else 0.0

        training_carry = build_training_table(sales, static_scope["carry"], low_stock_threshold)
        training_month = build_training_table(sales, static_scope["month_avg"], low_stock_threshold)

        for model_name, feature_list in MODEL_FEATURE_SETS.items():
            mode = "month_avg" if model_name.startswith("E_") else "carry"
            training_table = training_month if mode == "month_avg" else training_carry
            validation_static = static_scope["month_avg"] if mode == "month_avg" else static_scope["carry"]
            validation_static = add_inventory_derived_features(validation_static, low_stock_threshold)
            metrics, merged, importance = evaluate_core_model_on_scope(
                model_name=model_name,
                feature_list=feature_list,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
                training_table=training_table,
                validation_static=validation_static,
                sales=sales,
            )
            metrics["scope"] = scope_name
            comparison_rows.append(metrics)
            merged["model_name"] = model_name
            merged["scope"] = scope_name
            validation_frames.append(merged)
            importance["scope"] = scope_name
            importance_frames.append(importance)
            if scope_name == "validation_2022":
                validation_2022_predictions[model_name] = merged.copy()

        if scope_name == "validation_2022":
            current_best_validation = build_current_best_validation_2022()
            validation_static_2022 = add_inventory_derived_features(
                static_scope["carry"].loc[(static_scope["carry"][DATE_COL] >= valid_start) & (static_scope["carry"][DATE_COL] <= valid_end)].copy(),
                low_stock_threshold,
            )
            correction_result = evaluate_correction_model(validation_static_2022, current_best_validation, sales)
            if correction_result is not None:
                metrics, merged, importance = correction_result
                metrics["scope"] = scope_name
                comparison_rows.append(metrics)
                merged["model_name"] = "F_traffic_spike_correction"
                merged["scope"] = scope_name
                validation_frames.append(merged)
                importance["scope"] = scope_name
                importance_frames.append(importance)
                validation_2022_predictions["F_traffic_spike_correction"] = merged.copy()

    comparison = pd.DataFrame(comparison_rows)
    validation_predictions = pd.concat(validation_frames, ignore_index=True)
    importance_table = pd.concat(importance_frames, ignore_index=True)

    long_avg = (
        comparison.loc[comparison["scope"].isin([fold[0] for fold in LONG_FOLDS]) & comparison["model_name"].str.startswith(tuple(["A_", "B_", "C_", "D_", "E_"]))]
        .groupby("model_name", as_index=False)
        .agg(
            avg_MAE=("MAE", "mean"),
            avg_RMSE=("RMSE", "mean"),
            avg_R2=("R2", "mean"),
            avg_top10_RMSE=("top10_RMSE", "mean"),
            avg_high_traffic_spike_RMSE=("high_traffic_spike_RMSE", "mean"),
            avg_promo_day_RMSE=("promo_day_RMSE", "mean"),
            avg_non_promo_RMSE=("non_promo_RMSE", "mean"),
            avg_low_stock_RMSE=("low_stock_RMSE", "mean"),
        )
        .sort_values("avg_RMSE")
        .reset_index(drop=True)
    )

    validation_2022_metrics = (
        comparison.loc[comparison["scope"] == "validation_2022"]
        .sort_values("RMSE")
        .reset_index(drop=True)
    )

    comparison.to_csv(MODEL_COMPARISON_PATH, index=False)
    validation_predictions.to_csv(VALIDATION_PREDICTIONS_PATH, index=False, date_format="%Y-%m-%d")
    importance_table.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    reporter.emit_frame("2022 analog metrics:", validation_2022_metrics)
    reporter.emit_frame("Long-horizon average metrics:", long_avg)

    core_best_model = str(long_avg.iloc[0]["model_name"])
    overall_best_2022 = str(validation_2022_metrics.iloc[0]["model_name"])
    reporter.emit(f"Best core traffic-driven model: {core_best_model}")
    reporter.emit(f"Best 2022 analog model: {overall_best_2022}")

    reporter.emit("")
    reporter.emit("4. Future traffic scenarios")
    future_scenarios_raw, scenario_stats = build_future_traffic_raw_scenarios(web_daily, sample_submission, future_promo)
    reporter.emit_frame("Future traffic scenario stats:", scenario_stats)

    future_submissions: dict[str, pd.DataFrame] = {}
    best_core_is_month = core_best_model.startswith("E_")

    # Refit best core model on full history using carry or month-average inventory mode.
    full_inventory_carry = build_inventory_context(sales[DATE_COL], inventory_snapshots, mode="carry", cutoff=sales[DATE_COL].max())
    full_inventory_month = build_inventory_context(sales[DATE_COL], inventory_snapshots, mode="month_avg", cutoff=sales[DATE_COL].max())
    static_full = build_historical_static_context(sales, traffic_features, historical_promo, full_inventory_carry, full_inventory_month)
    full_low_stock_threshold = float(
        np.nanpercentile(
            pd.to_numeric(full_inventory_carry["inv_avg_days_of_supply"], errors="coerce").dropna(),
            25,
        )
    ) if not full_inventory_carry["inv_avg_days_of_supply"].dropna().empty else 0.0

    full_training_table = build_training_table(
        sales,
        static_full["month_avg"] if best_core_is_month else static_full["carry"],
        full_low_stock_threshold,
    )
    feature_list = MODEL_FEATURE_SETS[core_best_model]
    X_full, y_full, medians_full = prepare_training_matrix(full_training_table, feature_list, sales[DATE_COL].max())
    full_model, full_model_type = train_model(X_full, y_full)

    for scenario_name, raw_future_traffic in future_scenarios_raw.items():
        scenario_traffic_features = add_traffic_features(
            pd.concat([web_daily[[DATE_COL] + [column for column in raw_future_traffic.columns if column != DATE_COL]], raw_future_traffic], ignore_index=True)
        )
        future_only_traffic = scenario_traffic_features.loc[
            scenario_traffic_features[DATE_COL].isin(sample_submission[DATE_COL])
        ].copy()

        future_inventory = build_inventory_context(
            sample_submission[DATE_COL],
            inventory_snapshots,
            mode="month_avg" if best_core_is_month else "carry",
            cutoff=sales[DATE_COL].max(),
        )
        future_calendar = build_calendar_context(sample_submission[DATE_COL], sales[DATE_COL].min())
        future_static = (
            future_calendar.merge(future_only_traffic, on=DATE_COL, how="left", validate="one_to_one")
            .merge(future_promo, on=DATE_COL, how="left", validate="one_to_one")
            .merge(future_inventory, on=DATE_COL, how="left", validate="one_to_one")
            .fillna(0.0)
        )
        future_static = add_inventory_derived_features(future_static, full_low_stock_threshold)
        predictions = recursive_predict_revenue(
            full_model,
            full_model_type,
            medians_full,
            feature_list,
            sales[[DATE_COL, TARGET_COL]].copy(),
            future_static,
        )
        submission = build_submission(sample_submission[DATE_COL], predictions["predicted_Revenue"], ratio=0.8900)
        validate_submission_frame(submission, sample_submission)
        future_submissions[scenario_name] = submission

    future_submissions["seasonal"].to_csv(SUBMISSION_SEASONAL_PATH, index=False, date_format="%Y-%m-%d")
    future_submissions["conservative"].to_csv(SUBMISSION_CONSERVATIVE_PATH, index=False, date_format="%Y-%m-%d")
    future_submissions["high_demand"].to_csv(SUBMISSION_HIGH_PATH, index=False, date_format="%Y-%m-%d")

    reporter.emit("")
    reporter.emit("5. Create blends with current best")
    current_best_submission = pd.read_csv(CURRENT_BEST_SUBMISSION_PATH, parse_dates=[DATE_COL], low_memory=False)
    current_best_submission[DATE_COL] = pd.to_datetime(current_best_submission[DATE_COL], errors="coerce").dt.normalize()
    current_best_revenue = pd.to_numeric(current_best_submission[TARGET_COL], errors="coerce")

    created_files = [
        str(SUBMISSION_SEASONAL_PATH),
        str(SUBMISSION_CONSERVATIVE_PATH),
        str(SUBMISSION_HIGH_PATH),
    ]
    for weight, output_path in BLEND_OUTPUTS.items():
        revenue = (1.0 - weight) * current_best_revenue + weight * future_submissions["seasonal"][TARGET_COL]
        output = build_submission(sample_submission[DATE_COL], revenue, ratio=0.8900)
        validate_submission_frame(output, sample_submission)
        output.to_csv(output_path, index=False, date_format="%Y-%m-%d")
        created_files.append(str(output_path))

    for weight, output_path in HIGH_BLEND_OUTPUTS.items():
        revenue = (1.0 - weight) * current_best_revenue + weight * future_submissions["high_demand"][TARGET_COL]
        output = build_submission(sample_submission[DATE_COL], revenue, ratio=0.8900)
        validate_submission_frame(output, sample_submission)
        output.to_csv(output_path, index=False, date_format="%Y-%m-%d")
        created_files.append(str(output_path))

    reporter.emit("")
    reporter.emit("6. Final summary")
    reporter.emit_frame("Top 30 features for best core model:", importance_table.loc[importance_table["model_name"] == core_best_model].head(30))
    reporter.emit(f"Best validation model: {overall_best_2022}")
    reporter.emit(
        "2022 analog best metrics: "
        f"RMSE={validation_2022_metrics.iloc[0]['RMSE']:,.2f}, "
        f"MAE={validation_2022_metrics.iloc[0]['MAE']:,.2f}, "
        f"R2={validation_2022_metrics.iloc[0]['R2']:.6f}"
    )
    reporter.emit(
        "Long-horizon average best metrics: "
        f"{core_best_model} | RMSE={long_avg.iloc[0]['avg_RMSE']:,.2f}, "
        f"MAE={long_avg.iloc[0]['avg_MAE']:,.2f}, "
        f"R2={long_avg.iloc[0]['avg_R2']:.6f}"
    )
    reporter.emit(
        f"High traffic spike-day RMSE (best 2022 model): {validation_2022_metrics.iloc[0]['high_traffic_spike_RMSE']:,.2f}"
    )
    reporter.emit(f"Created submission files: {', '.join(created_files)}")
    reporter.emit(
        "Recommended upload order: "
        "submission_traffic_blend_05.csv, submission_traffic_blend_10.csv, "
        "submission_traffic_high_blend_10.csv, submission_traffic_driven_seasonal.csv, "
        "submission_traffic_blend_15.csv, submission_traffic_driven_high_demand.csv"
    )
    reporter.emit(
        "Leakage safety confirmation: the traffic-driven branch uses only lagged traffic, lagged safe revenue features, "
        "promo-known future calendar, and inventory snapshots available at or before each prediction date. "
        "No same-day traffic, no future actual Revenue/COGS, and no realized same-day demand signals are used."
    )
    reporter.save()


if __name__ == "__main__":
    run()
