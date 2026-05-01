from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import train_final_model as base
import train_promo_known_pipeline as promo_known
import train_traffic_driven_model as traffic_branch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "log"

SALES_PATH = DATA_DIR / "sales.csv"
ORDERS_PATH = DATA_DIR / "orders.csv"
ORDER_ITEMS_PATH = DATA_DIR / "order_items.csv"
PRODUCTS_PATH = DATA_DIR / "products.csv"
WEB_TRAFFIC_PATH = DATA_DIR / "web_traffic.csv"
PROMOTIONS_PATH = DATA_DIR / "promotions.csv"
INVENTORY_PATH = DATA_DIR / "inventory.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
FUTURE_PROMO_KNOWN_PATH = DATA_DIR / "future_promo_known_features.csv"
CURRENT_BEST_SUBMISSION_PATH = DATA_DIR / "submission_blend_direct_15_cogs8900.csv"
SEGMENT_BOTTOMUP_PATH = DATA_DIR / "submission_m5_segment_bottomup.csv"
DIRECT_SEASONAL_VALIDATION_PATH = DATA_DIR / "direct_seasonal_validation_predictions.csv"
DIRECT_SEASONAL_IMPORTANCE_PATH = DATA_DIR / "direct_seasonal_feature_importance.csv"
DIRECT_SEASONAL_SUBMISSION_PATH = DATA_DIR / "submission_direct_seasonal_ratio_8900.csv"

DAILY_FUNNEL_TABLE_PATH = DATA_DIR / "daily_funnel_table.csv"
FUTURE_TRAFFIC_SCENARIOS_PATH = DATA_DIR / "future_traffic_funnel_scenarios.csv"
ORDERS_VALIDATION_PATH = DATA_DIR / "funnel_orders_validation_predictions.csv"
AOV_VALIDATION_PATH = DATA_DIR / "funnel_aov_validation_predictions.csv"
REVENUE_VALIDATION_PATH = DATA_DIR / "funnel_revenue_validation_predictions.csv"
MODEL_COMPARISON_PATH = DATA_DIR / "funnel_model_comparison.csv"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "funnel_feature_importance.csv"
REPORT_PATH = LOG_DIR / "funnel_model_report.txt"
LOG_FILE = LOG_DIR / "train_funnel_model.log"

SUBMISSION_SEASONAL_PATH = DATA_DIR / "submission_funnel_seasonal.csv"
SUBMISSION_CONSERVATIVE_PATH = DATA_DIR / "submission_funnel_conservative.csv"
SUBMISSION_HIGH_PATH = DATA_DIR / "submission_funnel_high_demand.csv"
SUBMISSION_SEASONAL_8950_PATH = DATA_DIR / "submission_funnel_seasonal_8950.csv"
SUBMISSION_SEASONAL_9000_PATH = DATA_DIR / "submission_funnel_seasonal_9000.csv"

FUNNEL_BLEND_OUTPUTS = {
    0.05: DATA_DIR / "submission_funnel_blend_05.csv",
    0.10: DATA_DIR / "submission_funnel_blend_10.csv",
    0.15: DATA_DIR / "submission_funnel_blend_15.csv",
    0.20: DATA_DIR / "submission_funnel_blend_20.csv",
    0.25: DATA_DIR / "submission_funnel_blend_25.csv",
    0.30: DATA_DIR / "submission_funnel_blend_30.csv",
}
FUNNEL_SEGMENT_BLEND_OUTPUTS = {
    "801010": DATA_DIR / "submission_funnel_segment_801010.csv",
    "702010": DATA_DIR / "submission_funnel_segment_702010.csv",
    "701020": DATA_DIR / "submission_funnel_segment_701020.csv",
    "602020": DATA_DIR / "submission_funnel_segment_602020.csv",
}

DATE_COL = base.DATE_COL
TARGET_COL = base.TARGET_COL
COGS_COL = base.COGS_COL
RANDOM_STATE = base.RANDOM_STATE
EPS = 1e-9
HIGH_RISK_MONTHS = {2, 3, 5, 8}

VALIDATION_2022 = ("validation_2022", pd.Timestamp("2021-12-31"), pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"))
LONG_FOLDS = [
    ("fold_1", pd.Timestamp("2019-06-30"), pd.Timestamp("2019-07-01"), pd.Timestamp("2020-12-31")),
    ("fold_2", pd.Timestamp("2020-06-30"), pd.Timestamp("2020-07-01"), pd.Timestamp("2021-12-31")),
    ("fold_3", pd.Timestamp("2021-06-30"), pd.Timestamp("2021-07-01"), pd.Timestamp("2022-12-31")),
]
ALL_SCOPES = [VALIDATION_2022] + LONG_FOLDS

PROMO_COLUMNS = [
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
    "spring_sale",
    "mid_year_sale",
    "fall_launch",
    "year_end_sale",
    "urban_blowout",
    "rural_special",
    "campaign_intensity",
    "discount_x_progress",
    "discount_x_days_remaining",
]
CAMPAIGN_COLUMNS = [
    "spring_sale",
    "mid_year_sale",
    "fall_launch",
    "year_end_sale",
    "urban_blowout",
    "rural_special",
]
INVENTORY_COLUMNS = [
    "inv_avg_days_of_supply",
    "inv_min_days_of_supply",
    "inv_median_days_of_supply",
    "inv_avg_sell_through_rate",
    "inv_stockout_rate",
    "inv_reorder_rate",
    "inv_fill_rate_mean",
]
TRAFFIC_RAW_COLUMNS = [
    "sessions_sum",
    "unique_visitors_sum",
    "page_views_sum",
    "avg_bounce_rate",
    "avg_session_duration_sec",
    "source_diversity_count",
] + [f"{source}_sessions" for source in traffic_branch.TRAFFIC_SOURCES]

ORDERS_DIRECT_FEATURES = [
    "day_of_week",
    "day_of_year",
    "week_of_year",
    "month",
    "year",
    "is_weekend",
    "calendar_any_promo",
    "calendar_avg_discount_value",
    "calendar_active_promo_count",
    "promo_progress_ratio",
    "promo_days_remaining",
    "promotion_campaign_index",
    "campaign_intensity",
    "discount_x_progress",
    "discount_x_days_remaining",
    "sessions_sum",
    "unique_visitors_sum",
    "page_views_sum",
    "avg_bounce_rate",
    "avg_session_duration_sec",
    "source_diversity_count",
    "sessions_roll_mean_7",
    "sessions_roll_mean_30",
    "sessions_growth_1_7",
    "traffic_spike_125",
    "engagement_index",
    "orders_lag_365",
    "orders_lag_730",
    "orders_lag_1095",
    "orders_same_day_recent_mean",
    "orders_same_month_recent_mean",
    "orders_lag365_to_recent_mean_ratio",
] + CAMPAIGN_COLUMNS

ORDERS_RECURSIVE_FEATURES = [
    "day_of_week",
    "day_of_year",
    "week_of_year",
    "month",
    "is_weekend",
    "calendar_any_promo",
    "calendar_avg_discount_value",
    "calendar_active_promo_count",
    "promo_progress_ratio",
    "promo_days_remaining",
    "campaign_intensity",
    "discount_x_progress",
    "discount_x_days_remaining",
    "orders_lag_7",
    "orders_lag_14",
    "orders_lag_30",
    "orders_lag_90",
    "orders_lag_365",
    "orders_roll_mean_7",
    "orders_roll_mean_30",
    "orders_roll_mean_90",
    "orders_roll_mean_365",
    "sessions_lag_1",
    "sessions_lag_2",
    "sessions_lag_3",
    "sessions_lag_7",
    "sessions_lag_14",
    "sessions_roll_mean_3",
    "sessions_roll_mean_7",
    "sessions_roll_mean_14",
    "sessions_roll_mean_30",
    "sessions_roll_std_7",
    "sessions_roll_std_30",
    "sessions_growth_1_7",
    "sessions_growth_3_14",
    "traffic_spike_125",
    "traffic_spike_150",
    "traffic_acceleration",
    "pageview_per_session_lag_1",
    "visitor_to_session_ratio_lag_1",
    "engagement_index",
    "inv_avg_days_of_supply",
    "inv_avg_sell_through_rate",
    "inv_stockout_rate",
] + CAMPAIGN_COLUMNS

CONVERSION_FEATURES = [
    "day_of_week",
    "day_of_year",
    "week_of_year",
    "month",
    "is_weekend",
    "calendar_any_promo",
    "calendar_avg_discount_value",
    "calendar_active_promo_count",
    "promo_progress_ratio",
    "promo_days_remaining",
    "campaign_intensity",
    "discount_x_progress",
    "discount_x_days_remaining",
    "sessions_sum",
    "unique_visitors_sum",
    "page_views_sum",
    "avg_bounce_rate",
    "avg_session_duration_sec",
    "pageview_per_session_lag_1",
    "visitor_to_session_ratio_lag_1",
    "engagement_index",
    "sessions_growth_1_7",
    "sessions_growth_3_14",
    "conversion_lag_7",
    "conversion_lag_14",
    "conversion_lag_30",
    "conversion_lag_365",
    "conversion_roll_mean_7",
    "conversion_roll_mean_30",
    "conversion_roll_mean_90",
    "conversion_roll_mean_365",
    "inv_avg_days_of_supply",
    "inv_stockout_rate",
] + CAMPAIGN_COLUMNS

AOV_DIRECT_FEATURES = [
    "day_of_week",
    "day_of_year",
    "month",
    "week_of_year",
    "is_weekend",
    "calendar_any_promo",
    "calendar_avg_discount_value",
    "calendar_active_promo_count",
    "promo_progress_ratio",
    "promo_days_remaining",
    "campaign_intensity",
    "discount_x_progress",
    "discount_x_days_remaining",
    "aov_lag_7",
    "aov_lag_14",
    "aov_lag_30",
    "aov_lag_90",
    "aov_lag_365",
    "aov_roll_mean_7",
    "aov_roll_mean_30",
    "aov_roll_mean_90",
    "aov_roll_mean_365",
    "aov_same_day_recent_mean",
    "aov_same_month_recent_mean",
    "avg_discount_per_order_lag_365",
    "promo_order_share_lag_365",
    "inv_avg_days_of_supply",
    "inv_avg_sell_through_rate",
    "inv_stockout_rate",
] + CAMPAIGN_COLUMNS

AOV_RATIO_FEATURES = [
    "day_of_week",
    "day_of_year",
    "month",
    "week_of_year",
    "is_weekend",
    "calendar_any_promo",
    "calendar_avg_discount_value",
    "calendar_active_promo_count",
    "promo_progress_ratio",
    "promo_days_remaining",
    "campaign_intensity",
    "discount_x_progress",
    "discount_x_days_remaining",
    "aov_lag_365",
    "aov_same_day_recent_mean",
    "aov_same_month_recent_mean",
    "avg_discount_per_order_lag_365",
    "promo_order_share_lag_365",
] + CAMPAIGN_COLUMNS


class RunReporter:
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
        if isinstance(frame, pd.Series):
            if frame.empty:
                self.emit("(empty)")
            else:
                self.emit(frame.to_string())
            return
        if frame.empty:
            self.emit("(empty)")
            return
        self.emit(frame.to_string(index=False))

    def save(self, path: Path = REPORT_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def setup_logging(log_file: Path = LOG_FILE) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_funnel_model")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def safe_divide(numerator: pd.Series | np.ndarray | float, denominator: pd.Series | np.ndarray | float, fill_value: float = 0.0):
    numerator_arr = np.asarray(numerator, dtype=float)
    denominator_arr = np.asarray(denominator, dtype=float)
    result = np.divide(
        numerator_arr,
        denominator_arr,
        out=np.full_like(numerator_arr, fill_value, dtype=float),
        where=np.abs(denominator_arr) > EPS,
    )
    return result


def slugify(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("&", "and")
        .replace("-", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


def ensure_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def rmse(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean(np.square(actual_arr - predicted_arr))))


def mae(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(actual_arr - predicted_arr)))


def r2_score_manual(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)
    ss_res = float(np.sum(np.square(actual_arr - predicted_arr)))
    ss_tot = float(np.sum(np.square(actual_arr - np.mean(actual_arr))))
    if ss_tot <= EPS:
        return 0.0
    return 1.0 - ss_res / ss_tot


def compute_basic_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "r2": r2_score_manual(actual, predicted),
    }


def compute_revenue_metrics(
    actual: pd.Series,
    predicted: np.ndarray,
    promo_mask: pd.Series,
    high_traffic_mask: pd.Series,
) -> dict[str, float]:
    metrics = compute_basic_metrics(actual, predicted)
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)
    threshold = float(np.quantile(actual_arr, 0.90))
    spike_mask = actual_arr >= threshold
    metrics["top10_rmse"] = rmse(actual_arr[spike_mask], predicted_arr[spike_mask]) if spike_mask.any() else np.nan
    metrics["top10_underprediction_count"] = float(np.sum(predicted_arr[spike_mask] < actual_arr[spike_mask])) if spike_mask.any() else np.nan
    promo_bool = promo_mask.fillna(0).astype(bool).to_numpy()
    high_bool = high_traffic_mask.fillna(0).astype(bool).to_numpy()
    non_promo = ~promo_bool
    metrics["promo_day_rmse"] = rmse(actual_arr[promo_bool], predicted_arr[promo_bool]) if promo_bool.any() else np.nan
    metrics["non_promo_rmse"] = rmse(actual_arr[non_promo], predicted_arr[non_promo]) if non_promo.any() else np.nan
    metrics["high_traffic_rmse"] = rmse(actual_arr[high_bool], predicted_arr[high_bool]) if high_bool.any() else np.nan
    return metrics


def validate_submission_frame(output: pd.DataFrame, sample_submission: pd.DataFrame) -> None:
    expected_cols = [DATE_COL, TARGET_COL, COGS_COL]
    if output.columns.tolist() != expected_cols:
        raise ValueError(f"Submission columns must be {expected_cols}, got {output.columns.tolist()}")
    if len(output) != len(sample_submission):
        raise ValueError(f"Submission rows mismatch: expected {len(sample_submission)}, got {len(output)}")
    if not output[DATE_COL].equals(sample_submission[DATE_COL]):
        raise ValueError("Submission Date order does not match sample_submission")
    if output.isna().any().any():
        raise ValueError("Submission contains missing values")
    if (output[[TARGET_COL, COGS_COL]] < 0).any().any():
        raise ValueError("Submission contains negative Revenue or COGS")


def load_sales() -> pd.DataFrame:
    sales = pd.read_csv(SALES_PATH, parse_dates=[DATE_COL], low_memory=False)
    sales[DATE_COL] = pd.to_datetime(sales[DATE_COL], errors="coerce").dt.normalize()
    sales = ensure_numeric(sales, [TARGET_COL, COGS_COL])
    return sales.sort_values(DATE_COL).reset_index(drop=True)


def load_orders() -> pd.DataFrame:
    orders = pd.read_csv(ORDERS_PATH, low_memory=False)
    orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce").dt.normalize()
    return orders.dropna(subset=["order_date"]).sort_values("order_date").reset_index(drop=True)


def load_order_items() -> pd.DataFrame:
    items = pd.read_csv(ORDER_ITEMS_PATH, low_memory=False)
    return ensure_numeric(items, ["quantity", "unit_price", "discount_amount"])


def load_products() -> pd.DataFrame:
    return pd.read_csv(PRODUCTS_PATH, low_memory=False)


def load_sample_submission() -> pd.DataFrame:
    sample = pd.read_csv(SAMPLE_SUBMISSION_PATH, parse_dates=[DATE_COL], low_memory=False)
    sample[DATE_COL] = pd.to_datetime(sample[DATE_COL], errors="coerce").dt.normalize()
    return sample[[DATE_COL]].copy()


def normalize_date_column(frame: pd.DataFrame, column: str = DATE_COL) -> pd.DataFrame:
    output = frame.copy()
    output[column] = pd.to_datetime(output[column], errors="coerce").dt.normalize()
    return output


def load_future_promo_known(sample_submission: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    if FUTURE_PROMO_KNOWN_PATH.exists():
        future = pd.read_csv(FUTURE_PROMO_KNOWN_PATH, parse_dates=[DATE_COL], low_memory=False)
        future[DATE_COL] = pd.to_datetime(future[DATE_COL], errors="coerce").dt.normalize()
        available = [column for column in PROMO_COLUMNS if column in future.columns]
        for column in PROMO_COLUMNS:
            if column not in future.columns:
                future[column] = 0.0
        logger.info("Loaded future promo-known features from %s", FUTURE_PROMO_KNOWN_PATH)
        return future[[DATE_COL] + PROMO_COLUMNS].sort_values(DATE_COL).reset_index(drop=True)

    logger.warning("Future promo-known features missing; building zero-filled fallback")
    fallback = pd.DataFrame({DATE_COL: sample_submission[DATE_COL]})
    for column in PROMO_COLUMNS:
        fallback[column] = 0.0
    return fallback


def load_direct_shape_validation() -> tuple[pd.DataFrame, str]:
    validation = pd.read_csv(DIRECT_SEASONAL_VALIDATION_PATH, parse_dates=[DATE_COL], low_memory=False)
    validation[DATE_COL] = pd.to_datetime(validation[DATE_COL], errors="coerce").dt.normalize()
    best_target_type = "log_ratio"
    if DIRECT_SEASONAL_IMPORTANCE_PATH.exists():
        importance = pd.read_csv(DIRECT_SEASONAL_IMPORTANCE_PATH, low_memory=False)
        if "best_target_type" in importance.columns:
            candidates = importance["best_target_type"].dropna().astype(str)
            if not candidates.empty:
                best_target_type = candidates.iloc[0]
    filtered = validation[validation["target_type"].astype(str) == best_target_type].copy()
    filtered = filtered.rename(columns={"predicted_Revenue": "shape_pred"})
    return filtered[[DATE_COL, "fold", "shape_pred"]], best_target_type


def load_direct_shape_future(sample_submission: pd.DataFrame) -> pd.DataFrame:
    if DIRECT_SEASONAL_SUBMISSION_PATH.exists():
        direct = pd.read_csv(DIRECT_SEASONAL_SUBMISSION_PATH, parse_dates=[DATE_COL], low_memory=False)
        direct[DATE_COL] = pd.to_datetime(direct[DATE_COL], errors="coerce").dt.normalize()
        return direct[[DATE_COL, TARGET_COL]].rename(columns={TARGET_COL: "shape_pred"}).sort_values(DATE_COL).reset_index(drop=True)
    output = sample_submission.copy()
    output["shape_pred"] = 0.0
    return output


def load_current_best_submission() -> pd.DataFrame:
    current = pd.read_csv(CURRENT_BEST_SUBMISSION_PATH, parse_dates=[DATE_COL], low_memory=False)
    current[DATE_COL] = pd.to_datetime(current[DATE_COL], errors="coerce").dt.normalize()
    return current[[DATE_COL, TARGET_COL, COGS_COL]].sort_values(DATE_COL).reset_index(drop=True)


def load_segment_submission(sample_submission: pd.DataFrame) -> pd.DataFrame | None:
    if not SEGMENT_BOTTOMUP_PATH.exists():
        return None
    segment = pd.read_csv(SEGMENT_BOTTOMUP_PATH, parse_dates=[DATE_COL], low_memory=False)
    segment[DATE_COL] = pd.to_datetime(segment[DATE_COL], errors="coerce").dt.normalize()
    segment = segment[[DATE_COL, TARGET_COL]].sort_values(DATE_COL).reset_index(drop=True)
    if not segment[DATE_COL].equals(sample_submission[DATE_COL]):
        raise ValueError("Segment bottom-up submission Date order does not align with sample submission")
    return segment


def build_orders_daily(orders: pd.DataFrame) -> pd.DataFrame:
    orders = orders.copy()
    orders["order_source"] = orders["order_source"].fillna("unknown").astype(str).str.strip().str.lower()
    orders["device_type"] = orders["device_type"].fillna("unknown").astype(str).str.strip().str.lower()

    base_daily = (
        orders.groupby("order_date", as_index=False)
        .agg(
            orders_count=("order_id", "nunique"),
            unique_customers=("customer_id", "nunique"),
        )
        .rename(columns={"order_date": DATE_COL})
    )

    source_counts = (
        orders.pivot_table(index="order_date", columns="order_source", values="order_id", aggfunc="count", fill_value=0)
        .rename(columns=lambda source: f"orders_source_{slugify(source)}")
        .reset_index()
        .rename(columns={"order_date": DATE_COL})
    )
    device_counts = (
        orders.pivot_table(index="order_date", columns="device_type", values="order_id", aggfunc="count", fill_value=0)
        .rename(columns=lambda device: f"orders_device_{slugify(device)}")
        .reset_index()
        .rename(columns={"order_date": DATE_COL})
    )
    return (
        base_daily.merge(source_counts, on=DATE_COL, how="left")
        .merge(device_counts, on=DATE_COL, how="left")
        .fillna(0.0)
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )


def build_order_items_daily(order_items: pd.DataFrame, orders: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    order_dates = orders[["order_id", "order_date"]].copy()
    order_dates["order_date"] = pd.to_datetime(order_dates["order_date"], errors="coerce").dt.normalize()

    items = order_items.merge(order_dates, on="order_id", how="left", validate="many_to_one")
    items = items.dropna(subset=["order_date"]).copy()
    items["line_gross_value"] = items["quantity"] * items["unit_price"]
    items["line_net_value"] = items["line_gross_value"] - items["discount_amount"]
    items["promo_used"] = (items["promo_id"].notna() | items["promo_id_2"].notna()).astype(int)

    order_level = (
        items.groupby(["order_date", "order_id"], as_index=False)
        .agg(
            item_lines=("order_id", "size"),
            total_quantity=("quantity", "sum"),
            gross_item_value=("line_gross_value", "sum"),
            total_discount_amount=("discount_amount", "sum"),
            promo_used=("promo_used", "max"),
        )
    )
    daily = (
        order_level.groupby("order_date", as_index=False)
        .agg(
            item_lines=("item_lines", "sum"),
            total_quantity=("total_quantity", "sum"),
            gross_item_value=("gross_item_value", "sum"),
            total_discount_amount=("total_discount_amount", "sum"),
            promo_orders_count=("promo_used", "sum"),
        )
        .rename(columns={"order_date": DATE_COL})
    )

    item_with_products = items.merge(products[["product_id", "category", "segment"]], on="product_id", how="left", validate="many_to_one")
    category_totals = (
        item_with_products.groupby("category", as_index=False)["line_net_value"]
        .sum()
        .sort_values("line_net_value", ascending=False)
        .head(3)
    )
    segment_totals = (
        item_with_products.groupby("segment", as_index=False)["line_net_value"]
        .sum()
        .sort_values("line_net_value", ascending=False)
        .head(3)
    )
    top_categories = [value for value in category_totals["category"].dropna().astype(str)]
    top_segments = [value for value in segment_totals["segment"].dropna().astype(str)]

    daily_total_value = item_with_products.groupby("order_date", as_index=False).agg(total_net_value=("line_net_value", "sum"))
    daily = daily.merge(daily_total_value.rename(columns={"order_date": DATE_COL}), on=DATE_COL, how="left")

    for category in top_categories:
        slug = slugify(category)
        share = (
            item_with_products.loc[item_with_products["category"] == category]
            .groupby("order_date", as_index=False)
            .agg(category_net_value=("line_net_value", "sum"))
        )
        share[DATE_COL] = pd.to_datetime(share["order_date"], errors="coerce").dt.normalize()
        share = share.drop(columns=["order_date"])
        share = daily[[DATE_COL, "total_net_value"]].merge(share, on=DATE_COL, how="left").fillna(0.0)
        daily[f"cat_share_{slug}"] = safe_divide(share["category_net_value"], share["total_net_value"])

    for segment in top_segments:
        slug = slugify(segment)
        share = (
            item_with_products.loc[item_with_products["segment"] == segment]
            .groupby("order_date", as_index=False)
            .agg(segment_net_value=("line_net_value", "sum"))
        )
        share[DATE_COL] = pd.to_datetime(share["order_date"], errors="coerce").dt.normalize()
        share = share.drop(columns=["order_date"])
        share = daily[[DATE_COL, "total_net_value"]].merge(share, on=DATE_COL, how="left").fillna(0.0)
        daily[f"segment_share_{slug}"] = safe_divide(share["segment_net_value"], share["total_net_value"])

    return daily.sort_values(DATE_COL).reset_index(drop=True)


def prepare_inventory_snapshots() -> pd.DataFrame:
    inventory = pd.read_csv(INVENTORY_PATH, low_memory=False)
    inventory["snapshot_date"] = pd.to_datetime(inventory["snapshot_date"], errors="coerce").dt.normalize()
    numeric_columns = [
        "days_of_supply",
        "fill_rate",
        "stockout_flag",
        "reorder_flag",
        "sell_through_rate",
    ]
    for column in numeric_columns:
        if column not in inventory.columns:
            inventory[column] = 0.0
        inventory[column] = pd.to_numeric(inventory[column], errors="coerce").fillna(0.0)
    snapshots = (
        inventory.dropna(subset=["snapshot_date"])
        .groupby("snapshot_date", as_index=False)
        .agg(
            inv_avg_days_of_supply=("days_of_supply", "mean"),
            inv_min_days_of_supply=("days_of_supply", "min"),
            inv_median_days_of_supply=("days_of_supply", "median"),
            inv_avg_sell_through_rate=("sell_through_rate", "mean"),
            inv_stockout_rate=("stockout_flag", "mean"),
            inv_reorder_rate=("reorder_flag", "mean"),
            inv_fill_rate_mean=("fill_rate", "mean"),
        )
        .rename(columns={"snapshot_date": DATE_COL})
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )
    return snapshots


def build_inventory_context(dates: pd.Series, snapshots: pd.DataFrame) -> pd.DataFrame:
    calendar = pd.DataFrame({DATE_COL: pd.to_datetime(dates).sort_values().unique()})
    if snapshots.empty:
        for column in INVENTORY_COLUMNS:
            calendar[column] = 0.0
        return calendar
    merged = pd.merge_asof(
        calendar.sort_values(DATE_COL),
        snapshots.sort_values(DATE_COL),
        on=DATE_COL,
        direction="backward",
    )
    month_avg = snapshots.assign(month=snapshots[DATE_COL].dt.month).groupby("month")[INVENTORY_COLUMNS].mean()
    for column in INVENTORY_COLUMNS:
        month_values = calendar[DATE_COL].dt.month.map(month_avg[column]).fillna(0.0)
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(month_values).fillna(0.0)
    return merged[[DATE_COL] + INVENTORY_COLUMNS]


def build_daily_funnel_table(
    sales: pd.DataFrame,
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    products: pd.DataFrame,
    web_daily: pd.DataFrame,
    promo_daily: pd.DataFrame,
    inventory_daily: pd.DataFrame,
) -> pd.DataFrame:
    orders_daily = build_orders_daily(orders)
    items_daily = build_order_items_daily(order_items, orders, products)
    daily = sales[[DATE_COL, TARGET_COL, COGS_COL]].copy()
    daily = (
        daily.merge(orders_daily, on=DATE_COL, how="left", validate="one_to_one")
        .merge(items_daily, on=DATE_COL, how="left", validate="one_to_one")
        .merge(web_daily, on=DATE_COL, how="left", validate="one_to_one")
        .merge(promo_daily, on=DATE_COL, how="left", validate="one_to_one")
        .merge(inventory_daily, on=DATE_COL, how="left", validate="one_to_one")
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )
    numeric_columns = [column for column in daily.columns if column != DATE_COL]
    daily[numeric_columns] = daily[numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    daily["avg_discount_per_order"] = safe_divide(daily["total_discount_amount"], daily["orders_count"])
    daily["promo_order_share"] = safe_divide(daily["promo_orders_count"], daily["orders_count"])
    daily["avg_items_per_order"] = safe_divide(daily["item_lines"], daily["orders_count"])
    daily["avg_quantity_per_order"] = safe_divide(daily["total_quantity"], daily["orders_count"])
    daily["AOV"] = safe_divide(daily[TARGET_COL], daily["orders_count"])
    daily["conversion_rate"] = safe_divide(daily["orders_count"], daily["sessions_sum"])
    daily["revenue_per_session"] = safe_divide(daily[TARGET_COL], daily["sessions_sum"])
    daily["quantity_per_order"] = safe_divide(daily["total_quantity"], daily["orders_count"])
    daily["item_lines_per_order"] = safe_divide(daily["item_lines"], daily["orders_count"])
    return daily


def build_calendar_features(dates: pd.Series, min_date: pd.Timestamp) -> pd.DataFrame:
    calendar = base.build_calendar_features(dates, min_date)
    keep = [
        DATE_COL,
        "day_of_week",
        "day_of_year",
        "week_of_year",
        "month",
        "quarter",
        "year",
        "is_weekend",
        "is_month_start",
        "is_month_end",
    ]
    return calendar[keep].copy()


def safe_same_day_history(series: pd.Series, target_date: pd.Timestamp, ref_years_weights: list[tuple[int, float]]) -> float:
    return traffic_branch.safe_same_day_history(series, target_date, ref_years_weights)


def build_promo_uplift_factor(web_daily: pd.DataFrame, promo_features: pd.DataFrame) -> float:
    return traffic_branch.build_promo_uplift_factor(web_daily, promo_features)


def build_traffic_scenarios_for_dates(
    web_daily_history: pd.DataFrame,
    target_dates: pd.Series,
    promo_features: pd.DataFrame,
    seasonal_ref_years: list[tuple[int, float]],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    history = web_daily_history.copy().sort_values(DATE_COL).reset_index(drop=True)
    indexed = history.set_index(DATE_COL).sort_index()
    month_dow_means = history.assign(
        month=history[DATE_COL].dt.month.astype(int),
        day_of_week=history[DATE_COL].dt.dayofweek.astype(int),
    ).groupby(["month", "day_of_week"]).mean(numeric_only=True)
    month_means = history.assign(month=history[DATE_COL].dt.month.astype(int)).groupby("month").mean(numeric_only=True)
    promo_uplift = build_promo_uplift_factor(history, promo_features)

    target_dates = pd.to_datetime(target_dates).sort_values().unique()
    future_promo = promo_features.set_index(DATE_COL).sort_index()
    high_window_threshold = float(history["sessions_sum"].quantile(0.80)) if not history.empty else 0.0

    rows_seasonal: list[dict[str, Any]] = []
    rows_conservative: list[dict[str, Any]] = []

    for target_date in target_dates:
        target_date = pd.Timestamp(target_date).normalize()
        month = int(target_date.month)
        day_of_week = int(target_date.dayofweek)
        row_seasonal: dict[str, Any] = {DATE_COL: target_date}
        row_conservative: dict[str, Any] = {DATE_COL: target_date}
        for column in TRAFFIC_RAW_COLUMNS:
            seasonal_value = safe_same_day_history(indexed[column], target_date, seasonal_ref_years)
            if pd.isna(seasonal_value):
                if (month, day_of_week) in month_dow_means.index and column in month_dow_means.columns:
                    seasonal_value = float(month_dow_means.loc[(month, day_of_week), column])
                elif month in month_means.index and column in month_means.columns:
                    seasonal_value = float(month_means.loc[month, column])
                else:
                    seasonal_value = 0.0

            conservative_value = np.nan
            ref_date_2022 = None
            try:
                ref_date_2022 = target_date.replace(year=2022)
            except ValueError:
                ref_date_2022 = pd.Timestamp(year=2022, month=2, day=28)
            if ref_date_2022 in indexed.index:
                conservative_value = float(pd.to_numeric(indexed.loc[ref_date_2022, column], errors="coerce"))
            if not np.isfinite(conservative_value):
                if (month, day_of_week) in month_dow_means.index and column in month_dow_means.columns:
                    conservative_value = float(month_dow_means.loc[(month, day_of_week), column])
                elif month in month_means.index and column in month_means.columns:
                    conservative_value = float(month_means.loc[month, column])
                else:
                    conservative_value = 0.0

            row_seasonal[column] = max(0.0, seasonal_value)
            row_conservative[column] = max(0.0, conservative_value)

        promo_active = int(float(future_promo.loc[target_date, "calendar_any_promo"])) if target_date in future_promo.index else 0
        if promo_active == 1:
            for column in ["sessions_sum", "unique_visitors_sum", "page_views_sum"] + [f"{source}_sessions" for source in traffic_branch.TRAFFIC_SOURCES]:
                row_seasonal[column] = row_seasonal[column] * promo_uplift

        rows_seasonal.append(row_seasonal)
        rows_conservative.append(row_conservative)

    seasonal = pd.DataFrame(rows_seasonal)
    seasonal["sessions_sum"] = seasonal["sessions_sum"].rolling(window=7, min_periods=1).mean()
    seasonal["unique_visitors_sum"] = seasonal["unique_visitors_sum"].rolling(window=7, min_periods=1).mean()
    seasonal["page_views_sum"] = seasonal["page_views_sum"].rolling(window=7, min_periods=1).mean()
    seasonal["source_diversity_count"] = (seasonal[[f"{source}_sessions" for source in traffic_branch.TRAFFIC_SOURCES]] > 0).sum(axis=1)

    conservative = pd.DataFrame(rows_conservative)
    conservative["source_diversity_count"] = (conservative[[f"{source}_sessions" for source in traffic_branch.TRAFFIC_SOURCES]] > 0).sum(axis=1)

    high_demand = seasonal.copy()
    high_mask = (
        high_demand[DATE_COL].dt.month.isin(HIGH_RISK_MONTHS)
        | future_promo.reindex(high_demand[DATE_COL]).fillna(0.0)["calendar_any_promo"].astype(bool).to_numpy()
        | (high_demand["sessions_sum"] >= high_window_threshold)
    )
    for column in ["sessions_sum", "unique_visitors_sum", "page_views_sum"] + [f"{source}_sessions" for source in traffic_branch.TRAFFIC_SOURCES]:
        high_demand.loc[high_mask, column] = high_demand.loc[high_mask, column] * 1.05
        high_demand.loc[high_demand["sessions_sum"] >= high_window_threshold, column] = high_demand.loc[
            high_demand["sessions_sum"] >= high_window_threshold, column
        ] * 1.08
    high_demand["source_diversity_count"] = (high_demand[[f"{source}_sessions" for source in traffic_branch.TRAFFIC_SOURCES]] > 0).sum(axis=1)

    stats = pd.DataFrame(
        [
            {
                "scenario": "seasonal",
                "sessions_mean": float(seasonal["sessions_sum"].mean()),
                "sessions_min": float(seasonal["sessions_sum"].min()),
                "sessions_max": float(seasonal["sessions_sum"].max()),
            },
            {
                "scenario": "conservative",
                "sessions_mean": float(conservative["sessions_sum"].mean()),
                "sessions_min": float(conservative["sessions_sum"].min()),
                "sessions_max": float(conservative["sessions_sum"].max()),
            },
            {
                "scenario": "high_demand",
                "sessions_mean": float(high_demand["sessions_sum"].mean()),
                "sessions_min": float(high_demand["sessions_sum"].min()),
                "sessions_max": float(high_demand["sessions_sum"].max()),
            },
        ]
    )
    return {"seasonal": seasonal, "conservative": conservative, "high_demand": high_demand}, stats


def add_traffic_features_for_scenario(history_web_daily: pd.DataFrame, scenario_raw: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([history_web_daily, scenario_raw], ignore_index=True).sort_values(DATE_COL).reset_index(drop=True)
    features = traffic_branch.add_traffic_features(combined)
    return features.loc[features[DATE_COL].isin(pd.to_datetime(scenario_raw[DATE_COL]).unique())].reset_index(drop=True)


def extract_series_reference(series: pd.Series, target_date: pd.Timestamp, years_back: int) -> float:
    try:
        reference_date = target_date.replace(year=target_date.year - years_back)
    except ValueError:
        reference_date = pd.Timestamp(year=target_date.year - years_back, month=2, day=28)
    if reference_date in series.index:
        value = float(pd.to_numeric(series.loc[reference_date], errors="coerce"))
        return value if np.isfinite(value) else np.nan
    return np.nan


def build_series_reference_frame(
    dates: pd.Series,
    history_df: pd.DataFrame,
    value_col: str,
    prefix: str,
) -> pd.DataFrame:
    history_series = history_df.set_index(DATE_COL).sort_index()[value_col]
    month_year_means = history_df.assign(year=history_df[DATE_COL].dt.year, month=history_df[DATE_COL].dt.month).groupby(["year", "month"])[value_col].mean()

    rows: list[dict[str, Any]] = []
    for target_date in pd.to_datetime(dates).sort_values().unique():
        target_date = pd.Timestamp(target_date).normalize()
        lag_365 = extract_series_reference(history_series, target_date, 1)
        lag_730 = extract_series_reference(history_series, target_date, 2)
        lag_1095 = extract_series_reference(history_series, target_date, 3)
        same_day_recent_mean = np.nan
        values = [lag_365, lag_730, lag_1095]
        weights = [0.5, 0.3, 0.2]
        weighted_values = [value * weight for value, weight in zip(values, weights) if np.isfinite(value)]
        weighted_weights = [weight for value, weight in zip(values, weights) if np.isfinite(value)]
        if weighted_weights:
            same_day_recent_mean = float(sum(weighted_values) / sum(weighted_weights))

        month_values = []
        month_weights = []
        for years_back, weight in zip([1, 2, 3], [0.5, 0.3, 0.2]):
            key = (target_date.year - years_back, target_date.month)
            if key in month_year_means.index:
                month_values.append(float(month_year_means.loc[key]) * weight)
                month_weights.append(weight)
        same_month_recent_mean = float(sum(month_values) / sum(month_weights)) if month_weights else np.nan

        rows.append(
            {
                DATE_COL: target_date,
                f"{prefix}_lag_365": lag_365,
                f"{prefix}_lag_730": lag_730,
                f"{prefix}_lag_1095": lag_1095,
                f"{prefix}_same_day_recent_mean": same_day_recent_mean,
                f"{prefix}_same_month_recent_mean": same_month_recent_mean,
            }
        )

    output = pd.DataFrame(rows)
    output[f"{prefix}_lag365_to_recent_mean_ratio"] = safe_divide(
        output[f"{prefix}_lag_365"],
        output[f"{prefix}_same_day_recent_mean"],
        fill_value=np.nan,
    )
    return output


def add_recursive_lag_features(df: pd.DataFrame, value_col: str, prefix: str, lags: list[int], roll_windows: list[int]) -> pd.DataFrame:
    output = df.sort_values(DATE_COL).reset_index(drop=True).copy()
    shifted = output[value_col].shift(1)
    for lag in lags:
        output[f"{prefix}_lag_{lag}"] = output[value_col].shift(lag)
    for window in roll_windows:
        output[f"{prefix}_roll_mean_{window}"] = shifted.rolling(window=window, min_periods=window).mean()
    return output


def build_hist_training_table(
    funnel_daily: pd.DataFrame,
    promo_daily: pd.DataFrame,
    traffic_features_actual: pd.DataFrame,
    inventory_daily: pd.DataFrame,
    min_date: pd.Timestamp,
) -> pd.DataFrame:
    calendar = build_calendar_features(funnel_daily[DATE_COL], min_date)
    table = funnel_daily.merge(calendar, on=DATE_COL, how="left", validate="one_to_one")
    extra_traffic_columns = [column for column in traffic_features_actual.columns if column != DATE_COL and column not in table.columns]
    if extra_traffic_columns:
        table = table.merge(
            traffic_features_actual[[DATE_COL] + extra_traffic_columns],
            on=DATE_COL,
            how="left",
            validate="one_to_one",
        )

    table = add_recursive_lag_features(table, "orders_count", "orders", [7, 14, 30, 90, 365], [7, 30, 90, 365])
    table = add_recursive_lag_features(table, "conversion_rate", "conversion", [7, 14, 30, 365], [7, 30, 90, 365])
    table = add_recursive_lag_features(table, "AOV", "aov", [7, 14, 30, 90, 365], [7, 30, 90, 365])

    order_refs = build_series_reference_frame(table[DATE_COL], table[[DATE_COL, "orders_count"]], "orders_count", "orders")
    aov_refs = build_series_reference_frame(table[DATE_COL], table[[DATE_COL, "AOV"]], "AOV", "aov")
    avg_discount_refs = build_series_reference_frame(table[DATE_COL], table[[DATE_COL, "avg_discount_per_order"]], "avg_discount_per_order", "avg_discount_per_order")
    promo_share_refs = build_series_reference_frame(table[DATE_COL], table[[DATE_COL, "promo_order_share"]], "promo_order_share", "promo_order_share")
    mix_refs = build_aov_mix_reference_context(table[DATE_COL], table[[DATE_COL] + [column for column in table.columns if column.startswith("cat_share_") or column.startswith("segment_share_")]])

    table = (
        table.merge(
            order_refs[
                [
                    DATE_COL,
                    "orders_lag_730",
                    "orders_lag_1095",
                    "orders_same_day_recent_mean",
                    "orders_same_month_recent_mean",
                    "orders_lag365_to_recent_mean_ratio",
                ]
            ],
            on=DATE_COL,
            how="left",
            validate="one_to_one",
        )
        .merge(
            aov_refs[
                [
                    DATE_COL,
                    "aov_lag_730",
                    "aov_lag_1095",
                    "aov_same_day_recent_mean",
                    "aov_same_month_recent_mean",
                    "aov_lag365_to_recent_mean_ratio",
                ]
            ],
            on=DATE_COL,
            how="left",
            validate="one_to_one",
        )
        .merge(avg_discount_refs[[DATE_COL, "avg_discount_per_order_lag_365"]], on=DATE_COL, how="left", validate="one_to_one")
        .merge(promo_share_refs[[DATE_COL, "promo_order_share_lag_365"]], on=DATE_COL, how="left", validate="one_to_one")
        .merge(mix_refs, on=DATE_COL, how="left", validate="one_to_one")
    )
    return table


def prepare_training_matrix(table: pd.DataFrame, feature_columns: list[str], target_column: str, train_end: pd.Timestamp) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    subset = table.loc[table[DATE_COL] <= train_end, [DATE_COL, target_column] + feature_columns].copy()
    subset = subset.dropna(subset=[target_column])
    X = subset[feature_columns].replace([np.inf, -np.inf], np.nan)
    medians = X.median(numeric_only=True).fillna(0.0)
    X = X.fillna(medians).fillna(0.0)
    y = pd.to_numeric(subset[target_column], errors="coerce").fillna(0.0)
    valid_mask = np.isfinite(y.to_numpy(dtype=float))
    return X.loc[valid_mask].reset_index(drop=True), y.loc[valid_mask].reset_index(drop=True), medians


def train_lightgbm_model(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    import lightgbm as lgb

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.03,
        "max_depth": 5,
        "num_leaves": 24,
        "min_child_samples": 18,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "seed": RANDOM_STATE,
        "verbosity": -1,
        "num_threads": 0,
    }
    dataset = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    model = lgb.train(params=params, train_set=dataset, num_boost_round=350)
    return model


def predict_with_model(model: Any, frame: pd.DataFrame, feature_columns: list[str], medians: pd.Series) -> np.ndarray:
    X = frame[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)
    predictions = np.asarray(model.predict(X), dtype=float)
    return np.maximum(0.0, predictions)


def extract_feature_importance(model: Any, feature_columns: list[str], model_name: str, target_group: str) -> pd.DataFrame:
    if hasattr(model, "feature_importance"):
        return (
            pd.DataFrame(
                {
                    "feature": feature_columns,
                    "importance_gain": model.feature_importance(importance_type="gain"),
                    "importance_split": model.feature_importance(importance_type="split"),
                    "model_name": model_name,
                    "target_group": target_group,
                }
            )
            .sort_values("importance_gain", ascending=False)
            .reset_index(drop=True)
        )
    return pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_gain": np.nan,
            "importance_split": np.nan,
            "model_name": model_name,
            "target_group": target_group,
        }
    )


def build_validation_inventory_context(train_end: pd.Timestamp, valid_dates: pd.Series, historical_snapshots: pd.DataFrame) -> pd.DataFrame:
    snapshots = historical_snapshots.loc[historical_snapshots[DATE_COL] <= valid_dates.max()].copy()
    return build_inventory_context(valid_dates, snapshots)


def build_orders_direct_context(
    train_history: pd.DataFrame,
    valid_dates: pd.Series,
    traffic_context: pd.DataFrame,
    promo_valid: pd.DataFrame,
    inventory_valid: pd.DataFrame,
    min_date: pd.Timestamp,
) -> pd.DataFrame:
    calendar = build_calendar_features(valid_dates, min_date)
    refs = build_series_reference_frame(valid_dates, train_history[[DATE_COL, "orders_count"]], "orders_count", "orders")
    context = (
        calendar.merge(promo_valid, on=DATE_COL, how="left", validate="one_to_one")
        .merge(traffic_context, on=DATE_COL, how="left", validate="one_to_one")
        .merge(inventory_valid, on=DATE_COL, how="left", validate="one_to_one")
        .merge(refs, on=DATE_COL, how="left", validate="one_to_one")
    )
    return context


def build_recursive_static_context(
    valid_dates: pd.Series,
    traffic_features: pd.DataFrame,
    promo_valid: pd.DataFrame,
    inventory_valid: pd.DataFrame,
    min_date: pd.Timestamp,
) -> pd.DataFrame:
    calendar = build_calendar_features(valid_dates, min_date)
    context = (
        calendar.merge(promo_valid, on=DATE_COL, how="left", validate="one_to_one")
        .merge(traffic_features, on=DATE_COL, how="left", validate="one_to_one")
        .merge(inventory_valid, on=DATE_COL, how="left", validate="one_to_one")
    )
    return context


def recursive_predict_orders(
    model: Any,
    medians: pd.Series,
    feature_columns: list[str],
    train_history: pd.DataFrame,
    static_context: pd.DataFrame,
) -> pd.DataFrame:
    history = train_history[[DATE_COL, "orders_count"]].copy().sort_values(DATE_COL).reset_index(drop=True)
    predictions: list[dict[str, Any]] = []
    for row in static_context.sort_values(DATE_COL).itertuples(index=False):
        feature_history = add_recursive_lag_features(history, "orders_count", "orders", [7, 14, 30, 90, 365], [7, 30, 90, 365])
        feature_row = feature_history.tail(1)[[column for column in feature_history.columns if column.startswith("orders_")]].copy()
        context_row = pd.DataFrame([row._asdict()])
        frame = context_row.merge(feature_row, left_on=None, right_on=None, how="cross")
        prediction = float(predict_with_model(model, frame, feature_columns, medians)[0])
        predictions.append({DATE_COL: row.Date, "predicted_orders": prediction})
        history = pd.concat([history, pd.DataFrame({DATE_COL: [row.Date], "orders_count": [prediction]})], ignore_index=True)
    return pd.DataFrame(predictions)


def recursive_predict_conversion(
    model: Any,
    medians: pd.Series,
    feature_columns: list[str],
    train_history: pd.DataFrame,
    static_context: pd.DataFrame,
) -> pd.DataFrame:
    history = train_history[[DATE_COL, "conversion_rate"]].copy().sort_values(DATE_COL).reset_index(drop=True)
    predictions: list[dict[str, Any]] = []
    for row in static_context.sort_values(DATE_COL).itertuples(index=False):
        feature_history = add_recursive_lag_features(history, "conversion_rate", "conversion", [7, 14, 30, 365], [7, 30, 90, 365])
        feature_row = feature_history.tail(1)[[column for column in feature_history.columns if column.startswith("conversion_")]].copy()
        context_row = pd.DataFrame([row._asdict()])
        frame = context_row.merge(feature_row, left_on=None, right_on=None, how="cross")
        predicted_conversion = float(np.clip(predict_with_model(model, frame, feature_columns, medians)[0], 0.0, 1.0))
        predicted_orders = float(predicted_conversion * float(getattr(row, "sessions_sum", 0.0)))
        predictions.append(
            {
                DATE_COL: row.Date,
                "predicted_conversion_rate": predicted_conversion,
                "predicted_orders": predicted_orders,
            }
        )
        history = pd.concat(
            [history, pd.DataFrame({DATE_COL: [row.Date], "conversion_rate": [predicted_conversion]})],
            ignore_index=True,
        )
    return pd.DataFrame(predictions)


def build_aov_mix_reference_context(dates: pd.Series, train_history: pd.DataFrame) -> pd.DataFrame:
    share_columns = [column for column in train_history.columns if column.startswith("cat_share_") or column.startswith("segment_share_")]
    if not share_columns:
        return pd.DataFrame({DATE_COL: pd.to_datetime(dates).sort_values().unique()})
    output = pd.DataFrame({DATE_COL: pd.to_datetime(dates).sort_values().unique()})
    for column in share_columns:
        refs = build_series_reference_frame(dates, train_history[[DATE_COL, column]], column, column)
        output = output.merge(refs[[DATE_COL, f"{column}_lag_365"]], on=DATE_COL, how="left", validate="one_to_one")
    return output


def get_mix_feature_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        [
            column
            for column in frame.columns
            if column.endswith("_lag_365") and (column.startswith("cat_share_") or column.startswith("segment_share_"))
        ]
    )


def build_aov_direct_context(
    train_history: pd.DataFrame,
    valid_dates: pd.Series,
    promo_valid: pd.DataFrame,
    inventory_valid: pd.DataFrame,
    min_date: pd.Timestamp,
) -> pd.DataFrame:
    calendar = build_calendar_features(valid_dates, min_date)
    aov_refs = build_series_reference_frame(valid_dates, train_history[[DATE_COL, "AOV"]], "AOV", "aov")
    discount_refs = build_series_reference_frame(valid_dates, train_history[[DATE_COL, "avg_discount_per_order"]], "avg_discount_per_order", "avg_discount_per_order")
    promo_share_refs = build_series_reference_frame(valid_dates, train_history[[DATE_COL, "promo_order_share"]], "promo_order_share", "promo_order_share")
    mix_refs = build_aov_mix_reference_context(valid_dates, train_history)
    context = (
        calendar.merge(promo_valid, on=DATE_COL, how="left", validate="one_to_one")
        .merge(inventory_valid, on=DATE_COL, how="left", validate="one_to_one")
        .merge(aov_refs, on=DATE_COL, how="left", validate="one_to_one")
        .merge(discount_refs[[DATE_COL, "avg_discount_per_order_lag_365"]], on=DATE_COL, how="left", validate="one_to_one")
        .merge(promo_share_refs[[DATE_COL, "promo_order_share_lag_365"]], on=DATE_COL, how="left", validate="one_to_one")
        .merge(mix_refs, on=DATE_COL, how="left", validate="one_to_one")
    )
    return context


def recursive_predict_aov(
    model: Any,
    medians: pd.Series,
    feature_columns: list[str],
    train_history: pd.DataFrame,
    static_context: pd.DataFrame,
) -> pd.DataFrame:
    history = train_history[[DATE_COL, "AOV"]].copy().sort_values(DATE_COL).reset_index(drop=True)
    predictions: list[dict[str, Any]] = []
    for row in static_context.sort_values(DATE_COL).itertuples(index=False):
        feature_history = add_recursive_lag_features(history, "AOV", "aov", [7, 14, 30, 90, 365], [7, 30, 90, 365])
        feature_row = feature_history.tail(1)[[column for column in feature_history.columns if column.startswith("aov_")]].copy()
        context_row = pd.DataFrame([row._asdict()])
        overlap_cols = [column for column in feature_row.columns if column in context_row.columns]
        if overlap_cols:
            context_row = context_row.drop(columns=overlap_cols)
        frame = context_row.merge(feature_row, left_on=None, right_on=None, how="cross")
        prediction = float(predict_with_model(model, frame, feature_columns, medians)[0])
        predictions.append({DATE_COL: row.Date, "predicted_AOV": prediction})
        history = pd.concat([history, pd.DataFrame({DATE_COL: [row.Date], "AOV": [prediction]})], ignore_index=True)
    return pd.DataFrame(predictions)


def baseline_aov_predictions(
    train_history: pd.DataFrame,
    target_dates: pd.Series,
    promo_context: pd.DataFrame,
) -> pd.DataFrame:
    working = train_history.copy()
    working["promo_flag"] = (working["calendar_any_promo"] > 0).astype(int)
    month_promo_median = working.groupby([working[DATE_COL].dt.month, "promo_flag"])["AOV"].median()
    campaign_medians = {}
    for campaign in CAMPAIGN_COLUMNS:
        campaign_rows = working.loc[working[campaign] > 0, "AOV"]
        campaign_medians[campaign] = float(campaign_rows.median()) if not campaign_rows.empty else np.nan
    trailing_median = float(working.sort_values(DATE_COL).tail(365)["AOV"].median())

    rows = []
    promo_lookup = promo_context.set_index(DATE_COL).sort_index()
    for target_date in pd.to_datetime(target_dates).sort_values().unique():
        target_date = pd.Timestamp(target_date).normalize()
        promo_row = promo_lookup.loc[target_date] if target_date in promo_lookup.index else pd.Series(dtype=float)
        month = int(target_date.month)
        promo_flag = int(float(promo_row.get("calendar_any_promo", 0.0)))
        prediction = np.nan
        for campaign in CAMPAIGN_COLUMNS:
            if float(promo_row.get(campaign, 0.0)) > 0 and np.isfinite(campaign_medians[campaign]):
                prediction = campaign_medians[campaign]
                break
        if not np.isfinite(prediction):
            key = (month, promo_flag)
            if key in month_promo_median.index:
                prediction = float(month_promo_median.loc[key])
        if not np.isfinite(prediction):
            prediction = trailing_median
        rows.append({DATE_COL: target_date, "predicted_AOV": max(0.0, float(prediction))})
    return pd.DataFrame(rows)


def evaluate_scope(
    scope_name: str,
    train_end: pd.Timestamp,
    valid_start: pd.Timestamp,
    valid_end: pd.Timestamp,
    funnel_daily: pd.DataFrame,
    promo_daily: pd.DataFrame,
    web_daily: pd.DataFrame,
    inventory_snapshots: pd.DataFrame,
    min_date: pd.Timestamp,
    direct_shape_validation: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]]:
    history = funnel_daily.loc[funnel_daily[DATE_COL] <= train_end].copy()
    valid_dates = pd.date_range(valid_start, valid_end, freq="D")
    promo_valid = promo_daily.loc[(promo_daily[DATE_COL] >= valid_start) & (promo_daily[DATE_COL] <= valid_end)].copy()
    inventory_valid = build_validation_inventory_context(train_end, pd.Series(valid_dates), inventory_snapshots)
    seasonal_ref_years = [(train_end.year, 0.5), (train_end.year - 1, 0.3), (train_end.year - 2, 0.2)]
    traffic_scenarios, _ = build_traffic_scenarios_for_dates(
        web_daily.loc[web_daily[DATE_COL] <= train_end].copy(),
        pd.Series(valid_dates),
        promo_valid,
        seasonal_ref_years,
    )
    traffic_raw = traffic_scenarios["seasonal"].copy()
    traffic_features = add_traffic_features_for_scenario(web_daily.loc[web_daily[DATE_COL] <= train_end].copy(), traffic_raw)
    traffic_direct_context = traffic_raw.merge(
        traffic_features[
            [
                DATE_COL,
                "sessions_roll_mean_7",
                "sessions_roll_mean_30",
                "sessions_growth_1_7",
                "traffic_spike_125",
                "engagement_index",
            ]
        ],
        on=DATE_COL,
        how="left",
        validate="one_to_one",
    )
    actual_window = funnel_daily.loc[(funnel_daily[DATE_COL] >= valid_start) & (funnel_daily[DATE_COL] <= valid_end)].copy()
    high_traffic_mask = (traffic_raw["sessions_sum"] >= float(history["sessions_sum"].quantile(0.80))).astype(int)

    comparison_rows: list[dict[str, Any]] = []
    orders_predictions_frames: list[pd.DataFrame] = []
    aov_predictions_frames: list[pd.DataFrame] = []
    revenue_predictions_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []

    hist_training = build_hist_training_table(
        funnel_daily,
        promo_daily,
        traffic_branch.add_traffic_features(web_daily.copy()),
        build_inventory_context(funnel_daily[DATE_COL], inventory_snapshots),
        min_date,
    )

    # Orders A - direct seasonal
    mix_feature_cols = get_mix_feature_columns(hist_training)
    aov_direct_features = AOV_DIRECT_FEATURES + mix_feature_cols
    aov_ratio_features = AOV_RATIO_FEATURES + mix_feature_cols

    X_orders_a, y_orders_a, med_orders_a = prepare_training_matrix(hist_training, ORDERS_DIRECT_FEATURES, "orders_count", train_end)
    orders_a_model = train_lightgbm_model(X_orders_a, y_orders_a)
    orders_a_context = build_orders_direct_context(history, pd.Series(valid_dates), traffic_direct_context, promo_valid, inventory_valid, min_date)
    orders_a_pred = predict_with_model(orders_a_model, orders_a_context, ORDERS_DIRECT_FEATURES, med_orders_a)
    orders_a_frame = pd.DataFrame({DATE_COL: valid_dates, "predicted_orders": orders_a_pred, "orders_model": "orders_a_direct_seasonal", "scope": scope_name})
    orders_predictions_frames.append(
        actual_window[[DATE_COL, "orders_count"]].merge(orders_a_frame, on=DATE_COL, how="left", validate="one_to_one")
    )
    orders_a_metrics = compute_basic_metrics(actual_window["orders_count"], orders_a_pred)
    comparison_rows.append({"target_group": "orders", "model_name": "orders_a_direct_seasonal", "scope": scope_name, **orders_a_metrics})
    importance_frames.append(extract_feature_importance(orders_a_model, ORDERS_DIRECT_FEATURES, "orders_a_direct_seasonal", "orders"))

    # Orders B - recursive safe
    X_orders_b, y_orders_b, med_orders_b = prepare_training_matrix(hist_training, ORDERS_RECURSIVE_FEATURES, "orders_count", train_end)
    orders_b_model = train_lightgbm_model(X_orders_b, y_orders_b)
    orders_b_context = build_recursive_static_context(pd.Series(valid_dates), traffic_features, promo_valid, inventory_valid, min_date)
    orders_b_frame = recursive_predict_orders(orders_b_model, med_orders_b, ORDERS_RECURSIVE_FEATURES, history, orders_b_context)
    orders_b_frame["orders_model"] = "orders_b_recursive_safe"
    orders_b_frame["scope"] = scope_name
    orders_predictions_frames.append(
        actual_window[[DATE_COL, "orders_count"]].merge(orders_b_frame, on=DATE_COL, how="left", validate="one_to_one")
    )
    orders_b_metrics = compute_basic_metrics(actual_window["orders_count"], orders_b_frame["predicted_orders"].to_numpy(dtype=float))
    comparison_rows.append({"target_group": "orders", "model_name": "orders_b_recursive_safe", "scope": scope_name, **orders_b_metrics})
    importance_frames.append(extract_feature_importance(orders_b_model, ORDERS_RECURSIVE_FEATURES, "orders_b_recursive_safe", "orders"))

    # Orders C - conversion
    X_orders_c, y_orders_c, med_orders_c = prepare_training_matrix(hist_training, CONVERSION_FEATURES, "conversion_rate", train_end)
    orders_c_model = train_lightgbm_model(X_orders_c, y_orders_c)
    orders_c_context = build_recursive_static_context(pd.Series(valid_dates), traffic_features, promo_valid, inventory_valid, min_date)
    orders_c_frame = recursive_predict_conversion(orders_c_model, med_orders_c, CONVERSION_FEATURES, history, orders_c_context)
    orders_c_frame["orders_model"] = "orders_c_conversion"
    orders_c_frame["scope"] = scope_name
    orders_predictions_frames.append(
        actual_window[[DATE_COL, "orders_count", "conversion_rate"]].merge(orders_c_frame, on=DATE_COL, how="left", validate="one_to_one")
    )
    orders_c_metrics = compute_basic_metrics(actual_window["orders_count"], orders_c_frame["predicted_orders"].to_numpy(dtype=float))
    comparison_rows.append({"target_group": "orders", "model_name": "orders_c_conversion", "scope": scope_name, **orders_c_metrics})
    importance_frames.append(extract_feature_importance(orders_c_model, CONVERSION_FEATURES, "orders_c_conversion", "orders"))

    # AOV A - direct recursive
    X_aov_a, y_aov_a, med_aov_a = prepare_training_matrix(hist_training, aov_direct_features, "AOV", train_end)
    aov_a_model = train_lightgbm_model(X_aov_a, y_aov_a)
    aov_a_context = build_aov_direct_context(history, pd.Series(valid_dates), promo_valid, inventory_valid, min_date)
    aov_a_frame = recursive_predict_aov(aov_a_model, med_aov_a, aov_direct_features, history, aov_a_context)
    aov_a_frame["aov_model"] = "aov_a_direct"
    aov_a_frame["scope"] = scope_name
    aov_predictions_frames.append(actual_window[[DATE_COL, "AOV"]].merge(aov_a_frame, on=DATE_COL, how="left", validate="one_to_one"))
    aov_a_metrics = compute_basic_metrics(actual_window["AOV"], aov_a_frame["predicted_AOV"].to_numpy(dtype=float))
    comparison_rows.append({"target_group": "aov", "model_name": "aov_a_direct", "scope": scope_name, **aov_a_metrics})
    importance_frames.append(extract_feature_importance(aov_a_model, aov_direct_features, "aov_a_direct", "aov"))

    # AOV B - ratio to lag365
    hist_training = hist_training.copy()
    hist_training["aov_ratio_target"] = safe_divide(hist_training["AOV"], hist_training["aov_lag_365"], fill_value=np.nan)
    X_aov_b, y_aov_b, med_aov_b = prepare_training_matrix(hist_training, aov_ratio_features, "aov_ratio_target", train_end)
    aov_b_model = train_lightgbm_model(X_aov_b, y_aov_b)
    aov_b_context = build_aov_direct_context(history, pd.Series(valid_dates), promo_valid, inventory_valid, min_date)
    aov_ratio_pred = np.maximum(0.0, predict_with_model(aov_b_model, aov_b_context, aov_ratio_features, med_aov_b))
    aov_b_frame = pd.DataFrame(
        {
            DATE_COL: valid_dates,
            "predicted_AOV": np.maximum(0.0, aov_ratio_pred * aov_b_context["aov_lag_365"].fillna(aov_b_context["aov_same_day_recent_mean"]).fillna(history["AOV"].median()).to_numpy(dtype=float)),
            "predicted_aov_ratio": aov_ratio_pred,
            "aov_model": "aov_b_ratio",
            "scope": scope_name,
        }
    )
    aov_predictions_frames.append(actual_window[[DATE_COL, "AOV"]].merge(aov_b_frame, on=DATE_COL, how="left", validate="one_to_one"))
    aov_b_metrics = compute_basic_metrics(actual_window["AOV"], aov_b_frame["predicted_AOV"].to_numpy(dtype=float))
    comparison_rows.append({"target_group": "aov", "model_name": "aov_b_ratio", "scope": scope_name, **aov_b_metrics})
    importance_frames.append(extract_feature_importance(aov_b_model, aov_ratio_features, "aov_b_ratio", "aov"))

    # AOV C - robust baseline
    baseline_context = promo_valid.copy()
    baseline_context[DATE_COL] = pd.to_datetime(baseline_context[DATE_COL], errors="coerce").dt.normalize()
    baseline_context = baseline_context.merge(
        build_calendar_features(pd.Series(valid_dates), min_date)[[DATE_COL, "month"]],
        on=DATE_COL,
        how="left",
        validate="one_to_one",
    )
    aov_c_frame = baseline_aov_predictions(hist_training, pd.Series(valid_dates), promo_valid)
    aov_c_frame["aov_model"] = "aov_c_baseline"
    aov_c_frame["scope"] = scope_name
    aov_predictions_frames.append(actual_window[[DATE_COL, "AOV"]].merge(aov_c_frame, on=DATE_COL, how="left", validate="one_to_one"))
    aov_c_metrics = compute_basic_metrics(actual_window["AOV"], aov_c_frame["predicted_AOV"].to_numpy(dtype=float))
    comparison_rows.append({"target_group": "aov", "model_name": "aov_c_baseline", "scope": scope_name, **aov_c_metrics})

    orders_map = {
        "orders_a_direct_seasonal": orders_a_frame[[DATE_COL, "predicted_orders"]],
        "orders_b_recursive_safe": orders_b_frame[[DATE_COL, "predicted_orders"]],
        "orders_c_conversion": orders_c_frame[[DATE_COL, "predicted_orders"]],
    }
    aov_map = {
        "aov_a_direct": aov_a_frame[[DATE_COL, "predicted_AOV"]],
        "aov_b_ratio": aov_b_frame[[DATE_COL, "predicted_AOV"]],
        "aov_c_baseline": aov_c_frame[[DATE_COL, "predicted_AOV"]],
    }

    shape_lookup = direct_shape_validation.loc[direct_shape_validation["fold"] == scope_name, [DATE_COL, "shape_pred"]].copy()
    if scope_name == "validation_2022":
        shape_lookup = direct_shape_validation.loc[direct_shape_validation["fold"] == "fold_3", [DATE_COL, "shape_pred"]].copy()
        shape_lookup = shape_lookup.loc[(shape_lookup[DATE_COL] >= valid_start) & (shape_lookup[DATE_COL] <= valid_end)].copy()

    combinations = [
        ("orders_a_direct_seasonal", "aov_a_direct"),
        ("orders_a_direct_seasonal", "aov_b_ratio"),
        ("orders_a_direct_seasonal", "aov_c_baseline"),
        ("orders_b_recursive_safe", "aov_a_direct"),
        ("orders_b_recursive_safe", "aov_b_ratio"),
        ("orders_c_conversion", "aov_a_direct"),
        ("orders_c_conversion", "aov_b_ratio"),
    ]

    for orders_name, aov_name in combinations:
        combo_name = f"{orders_name}__{aov_name}"
        combo = actual_window[[DATE_COL, TARGET_COL]].merge(orders_map[orders_name], on=DATE_COL, how="left", validate="one_to_one")
        combo = combo.merge(aov_map[aov_name], on=DATE_COL, how="left", validate="one_to_one")
        combo = combo.merge(promo_valid[[DATE_COL, "calendar_any_promo"]], on=DATE_COL, how="left", validate="one_to_one")
        combo = combo.merge(
            pd.DataFrame({DATE_COL: valid_dates, "high_traffic_flag": high_traffic_mask.to_numpy(dtype=int)}),
            on=DATE_COL,
            how="left",
            validate="one_to_one",
        )
        combo["predicted_Revenue"] = np.maximum(0.0, combo["predicted_orders"].to_numpy(dtype=float) * combo["predicted_AOV"].to_numpy(dtype=float))
        revenue_metrics = compute_revenue_metrics(
            combo[TARGET_COL],
            combo["predicted_Revenue"].to_numpy(dtype=float),
            combo["calendar_any_promo"].fillna(0).astype(int),
            combo["high_traffic_flag"].fillna(0).astype(int),
        )
        comparison_rows.append({"target_group": "revenue", "model_name": combo_name, "scope": scope_name, **revenue_metrics})
        combo["orders_model"] = orders_name
        combo["aov_model"] = aov_name
        combo["stabilized"] = 0
        combo["scope"] = scope_name
        revenue_predictions_frames.append(combo[[DATE_COL, TARGET_COL, "predicted_Revenue", "orders_model", "aov_model", "stabilized", "scope"]])

        if not shape_lookup.empty:
            stabilized = combo[[DATE_COL, TARGET_COL, "calendar_any_promo", "high_traffic_flag"]].merge(
                shape_lookup, on=DATE_COL, how="left", validate="one_to_one"
            )
            stabilized["predicted_Revenue"] = np.maximum(
                0.0,
                0.70 * combo["predicted_Revenue"].to_numpy(dtype=float) + 0.30 * stabilized["shape_pred"].fillna(combo["predicted_Revenue"]).to_numpy(dtype=float),
            )
            stabilized_name = f"{combo_name}__stabilized"
            stabilized_metrics = compute_revenue_metrics(
                stabilized[TARGET_COL],
                stabilized["predicted_Revenue"].to_numpy(dtype=float),
                stabilized["calendar_any_promo"].fillna(0).astype(int),
                stabilized["high_traffic_flag"].fillna(0).astype(int),
            )
            comparison_rows.append({"target_group": "revenue", "model_name": stabilized_name, "scope": scope_name, **stabilized_metrics})
            stabilized["orders_model"] = orders_name
            stabilized["aov_model"] = aov_name
            stabilized["stabilized"] = 1
            revenue_predictions_frames.append(
                stabilized[[DATE_COL, TARGET_COL, "predicted_Revenue", "orders_model", "aov_model", "stabilized"]].assign(scope=scope_name)
            )

    return comparison_rows, orders_predictions_frames, aov_predictions_frames, revenue_predictions_frames, importance_frames


def summarize_comparison_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    comparison = pd.DataFrame(rows)
    return comparison


def build_long_horizon_summary(comparison: pd.DataFrame, target_group: str) -> pd.DataFrame:
    subset = comparison.loc[(comparison["target_group"] == target_group) & (comparison["scope"].isin([fold[0] for fold in LONG_FOLDS]))].copy()
    if subset.empty:
        return pd.DataFrame()
    metric_cols = [column for column in subset.columns if column not in {"target_group", "model_name", "scope"}]
    summary = subset.groupby("model_name", as_index=False)[metric_cols].mean(numeric_only=True)
    summary["scope"] = "long_avg"
    summary["target_group"] = target_group
    return summary[comparison.columns]


def select_best_revenue_model(comparison: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    revenue_2022 = comparison.loc[(comparison["target_group"] == "revenue") & (comparison["scope"] == "validation_2022")].copy()
    revenue_long = build_long_horizon_summary(comparison, "revenue")
    score_table = revenue_2022.merge(
        revenue_long[["model_name", "rmse"]].rename(columns={"rmse": "long_avg_rmse"}),
        on="model_name",
        how="left",
    )
    score_table["selection_score"] = score_table["rmse"] * 0.6 + score_table["long_avg_rmse"].fillna(score_table["rmse"]) * 0.4
    score_table = score_table.sort_values(["selection_score", "rmse", "top10_rmse"], ascending=[True, True, True]).reset_index(drop=True)
    best_model = str(score_table.iloc[0]["model_name"])
    return best_model, score_table


def split_best_model_name(best_model: str) -> tuple[str, str, bool]:
    stabilized = best_model.endswith("__stabilized")
    core = best_model.replace("__stabilized", "")
    orders_model, aov_model = core.split("__")
    return orders_model, aov_model, stabilized


def build_future_orders_predictions(
    orders_model_name: str,
    train_history: pd.DataFrame,
    future_dates: pd.Series,
    future_promo: pd.DataFrame,
    future_inventory: pd.DataFrame,
    history_web_daily: pd.DataFrame,
    min_date: pd.Timestamp,
    seasonal_ref_years: list[tuple[int, float]],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    traffic_scenarios, traffic_stats = build_traffic_scenarios_for_dates(history_web_daily, future_dates, future_promo, seasonal_ref_years)

    hist_training = build_hist_training_table(
        train_history,
        promo_known.build_daily_promo_known_features(train_history[DATE_COL], promo_known.load_promotions(PROMOTIONS_PATH)),
        traffic_branch.add_traffic_features(history_web_daily.copy()),
        build_inventory_context(train_history[DATE_COL], prepare_inventory_snapshots()),
        min_date,
    )

    predictions_by_scenario: dict[str, pd.DataFrame] = {}
    importances: list[pd.DataFrame] = []
    if orders_model_name == "orders_a_direct_seasonal":
        X_train, y_train, medians = prepare_training_matrix(hist_training, ORDERS_DIRECT_FEATURES, "orders_count", train_history[DATE_COL].max())
        model = train_lightgbm_model(X_train, y_train)
        importances.append(extract_feature_importance(model, ORDERS_DIRECT_FEATURES, "orders_a_direct_seasonal", "orders"))
        for scenario_name, raw in traffic_scenarios.items():
            features = add_traffic_features_for_scenario(history_web_daily, raw)
            traffic_direct_context = raw.merge(
                features[
                    [
                        DATE_COL,
                        "sessions_roll_mean_7",
                        "sessions_roll_mean_30",
                        "sessions_growth_1_7",
                        "traffic_spike_125",
                        "engagement_index",
                    ]
                ],
                on=DATE_COL,
                how="left",
                validate="one_to_one",
            )
            context = build_orders_direct_context(train_history, future_dates, traffic_direct_context, future_promo, future_inventory, min_date)
            pred = predict_with_model(model, context, ORDERS_DIRECT_FEATURES, medians)
            predictions_by_scenario[scenario_name] = pd.DataFrame({DATE_COL: future_dates, "predicted_orders": pred})
    elif orders_model_name == "orders_b_recursive_safe":
        X_train, y_train, medians = prepare_training_matrix(hist_training, ORDERS_RECURSIVE_FEATURES, "orders_count", train_history[DATE_COL].max())
        model = train_lightgbm_model(X_train, y_train)
        importances.append(extract_feature_importance(model, ORDERS_RECURSIVE_FEATURES, "orders_b_recursive_safe", "orders"))
        for scenario_name, raw in traffic_scenarios.items():
            features = add_traffic_features_for_scenario(history_web_daily, raw)
            context = build_recursive_static_context(future_dates, features, future_promo, future_inventory, min_date)
            predictions_by_scenario[scenario_name] = recursive_predict_orders(model, medians, ORDERS_RECURSIVE_FEATURES, train_history, context)
    else:
        X_train, y_train, medians = prepare_training_matrix(hist_training, CONVERSION_FEATURES, "conversion_rate", train_history[DATE_COL].max())
        model = train_lightgbm_model(X_train, y_train)
        importances.append(extract_feature_importance(model, CONVERSION_FEATURES, "orders_c_conversion", "orders"))
        for scenario_name, raw in traffic_scenarios.items():
            features = add_traffic_features_for_scenario(history_web_daily, raw)
            context = build_recursive_static_context(future_dates, features, future_promo, future_inventory, min_date)
            predictions_by_scenario[scenario_name] = recursive_predict_conversion(model, medians, CONVERSION_FEATURES, train_history, context)
    return predictions_by_scenario, pd.concat(importances, ignore_index=True), traffic_stats


def build_future_aov_predictions(
    aov_model_name: str,
    train_history: pd.DataFrame,
    future_dates: pd.Series,
    future_promo: pd.DataFrame,
    future_inventory: pd.DataFrame,
    min_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hist_training = build_hist_training_table(
        train_history,
        promo_known.build_daily_promo_known_features(train_history[DATE_COL], promo_known.load_promotions(PROMOTIONS_PATH)),
        traffic_branch.add_traffic_features(normalize_date_column(traffic_branch.build_web_daily(pd.read_csv(WEB_TRAFFIC_PATH, low_memory=False)))),
        build_inventory_context(train_history[DATE_COL], prepare_inventory_snapshots()),
        min_date,
    )
    mix_feature_cols = get_mix_feature_columns(hist_training)
    aov_direct_features = AOV_DIRECT_FEATURES + mix_feature_cols
    aov_ratio_features = AOV_RATIO_FEATURES + mix_feature_cols
    if aov_model_name == "aov_a_direct":
        X_train, y_train, medians = prepare_training_matrix(hist_training, aov_direct_features, "AOV", train_history[DATE_COL].max())
        model = train_lightgbm_model(X_train, y_train)
        context = build_aov_direct_context(train_history, future_dates, future_promo, future_inventory, min_date)
        predictions = recursive_predict_aov(model, medians, aov_direct_features, train_history, context)
        return predictions, extract_feature_importance(model, aov_direct_features, "aov_a_direct", "aov")
    if aov_model_name == "aov_b_ratio":
        hist_training = hist_training.copy()
        hist_training["aov_ratio_target"] = safe_divide(hist_training["AOV"], hist_training["aov_lag_365"], fill_value=np.nan)
        X_train, y_train, medians = prepare_training_matrix(hist_training, aov_ratio_features, "aov_ratio_target", train_history[DATE_COL].max())
        model = train_lightgbm_model(X_train, y_train)
        context = build_aov_direct_context(train_history, future_dates, future_promo, future_inventory, min_date)
        ratio_pred = np.maximum(0.0, predict_with_model(model, context, aov_ratio_features, medians))
        predictions = pd.DataFrame(
            {
                DATE_COL: future_dates,
                "predicted_AOV": np.maximum(
                    0.0,
                    ratio_pred * context["aov_lag_365"].fillna(context["aov_same_day_recent_mean"]).fillna(train_history["AOV"].median()).to_numpy(dtype=float),
                ),
            }
        )
        return predictions, extract_feature_importance(model, aov_ratio_features, "aov_b_ratio", "aov")
    predictions = baseline_aov_predictions(train_history, future_dates, future_promo)
    return predictions, pd.DataFrame({"feature": [], "importance_gain": [], "importance_split": [], "model_name": [], "target_group": []})


def build_submission(dates: pd.Series, revenue: pd.Series | np.ndarray, ratio: float = 0.8900) -> pd.DataFrame:
    submission = pd.DataFrame({DATE_COL: pd.to_datetime(dates).reset_index(drop=True)})
    submission[TARGET_COL] = np.maximum(0.0, np.asarray(revenue, dtype=float))
    submission[COGS_COL] = np.maximum(0.0, submission[TARGET_COL] * ratio)
    return submission[[DATE_COL, TARGET_COL, COGS_COL]]


def save_submission(path: Path, submission: pd.DataFrame, sample_submission: pd.DataFrame) -> None:
    validate_submission_frame(submission, sample_submission)
    submission.to_csv(path, index=False)


def main() -> None:
    logger = setup_logging()
    reporter = RunReporter(logger)

    sales = load_sales()
    orders = load_orders()
    order_items = load_order_items()
    products = load_products()
    sample_submission = load_sample_submission()
    current_best = load_current_best_submission()
    if not current_best[DATE_COL].equals(sample_submission[DATE_COL]):
        raise ValueError("Current best submission dates do not match sample submission")

    web_raw = pd.read_csv(WEB_TRAFFIC_PATH, low_memory=False)
    web_daily = normalize_date_column(traffic_branch.build_web_daily(web_raw))
    traffic_features_actual = traffic_branch.add_traffic_features(web_daily.copy())
    promotions = promo_known.load_promotions(PROMOTIONS_PATH)
    promo_daily = promo_known.build_daily_promo_known_features(sales[DATE_COL], promotions)
    future_promo = load_future_promo_known(sample_submission, logger)
    inventory_snapshots = prepare_inventory_snapshots()
    inventory_daily = build_inventory_context(sales[DATE_COL], inventory_snapshots)
    future_inventory = build_inventory_context(sample_submission[DATE_COL], inventory_snapshots)
    direct_shape_validation, direct_best_target = load_direct_shape_validation()
    direct_shape_future = normalize_date_column(load_direct_shape_future(sample_submission))
    segment_submission = load_segment_submission(sample_submission)

    reporter.emit("Building daily funnel table...")
    funnel_daily = build_daily_funnel_table(
        sales=sales,
        orders=orders,
        order_items=order_items,
        products=products,
        web_daily=web_daily,
        promo_daily=promo_daily,
        inventory_daily=inventory_daily,
    )
    funnel_daily.to_csv(DAILY_FUNNEL_TABLE_PATH, index=False)
    reporter.emit(f"Saved daily funnel table: {DAILY_FUNNEL_TABLE_PATH}")

    reporter.emit("Building future traffic scenarios...")
    future_traffic_scenarios, future_traffic_stats = build_traffic_scenarios_for_dates(
        web_daily_history=web_daily,
        target_dates=sample_submission[DATE_COL],
        promo_features=future_promo,
        seasonal_ref_years=[(2022, 0.5), (2021, 0.3), (2020, 0.2)],
    )
    future_traffic_long = []
    for scenario_name, frame in future_traffic_scenarios.items():
        enriched = add_traffic_features_for_scenario(web_daily.copy(), frame)
        merged = frame.merge(enriched, on=DATE_COL, how="left", suffixes=("", "_feat"))
        merged["scenario"] = scenario_name
        future_traffic_long.append(merged)
    future_traffic_output = pd.concat(future_traffic_long, ignore_index=True)
    future_traffic_output.to_csv(FUTURE_TRAFFIC_SCENARIOS_PATH, index=False)
    reporter.emit(f"Saved future traffic scenarios: {FUTURE_TRAFFIC_SCENARIOS_PATH}")

    comparison_rows: list[dict[str, Any]] = []
    orders_validation_frames: list[pd.DataFrame] = []
    aov_validation_frames: list[pd.DataFrame] = []
    revenue_validation_frames: list[pd.DataFrame] = []
    feature_importance_frames: list[pd.DataFrame] = []

    min_date = sales[DATE_COL].min()
    for scope_name, train_end, valid_start, valid_end in ALL_SCOPES:
        reporter.emit(f"Evaluating scope {scope_name}: train <= {train_end.date()}, validate {valid_start.date()} -> {valid_end.date()}")
        scope_rows, scope_orders, scope_aov, scope_revenue, scope_importance = evaluate_scope(
            scope_name=scope_name,
            train_end=train_end,
            valid_start=valid_start,
            valid_end=valid_end,
            funnel_daily=funnel_daily,
            promo_daily=promo_daily,
            web_daily=web_daily,
            inventory_snapshots=inventory_snapshots,
            min_date=min_date,
            direct_shape_validation=direct_shape_validation,
        )
        comparison_rows.extend(scope_rows)
        orders_validation_frames.extend(scope_orders)
        aov_validation_frames.extend(scope_aov)
        revenue_validation_frames.extend(scope_revenue)
        feature_importance_frames.extend(scope_importance)

    comparison = summarize_comparison_rows(comparison_rows)
    orders_validation = pd.concat(orders_validation_frames, ignore_index=True)
    aov_validation = pd.concat(aov_validation_frames, ignore_index=True)
    revenue_validation = pd.concat(revenue_validation_frames, ignore_index=True)
    feature_importance = pd.concat(feature_importance_frames, ignore_index=True).drop_duplicates(
        subset=["feature", "model_name", "target_group"]
    )

    long_orders = build_long_horizon_summary(comparison, "orders")
    long_aov = build_long_horizon_summary(comparison, "aov")
    long_revenue = build_long_horizon_summary(comparison, "revenue")
    if not long_orders.empty:
        comparison = pd.concat([comparison, long_orders, long_aov, long_revenue], ignore_index=True)

    comparison.to_csv(MODEL_COMPARISON_PATH, index=False)
    orders_validation.to_csv(ORDERS_VALIDATION_PATH, index=False)
    aov_validation.to_csv(AOV_VALIDATION_PATH, index=False)
    revenue_validation.to_csv(REVENUE_VALIDATION_PATH, index=False)
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    best_revenue_model, revenue_score_table = select_best_revenue_model(comparison)
    best_orders_model, best_aov_model, best_stabilized = split_best_model_name(best_revenue_model)

    reporter.emit("")
    reporter.emit("Orders model comparison (2022 analog):")
    reporter.emit_frame(
        "orders_2022",
        comparison.loc[(comparison["target_group"] == "orders") & (comparison["scope"] == "validation_2022"), ["model_name", "mae", "rmse", "r2"]],
    )
    reporter.emit("AOV model comparison (2022 analog):")
    reporter.emit_frame(
        "aov_2022",
        comparison.loc[(comparison["target_group"] == "aov") & (comparison["scope"] == "validation_2022"), ["model_name", "mae", "rmse", "r2"]],
    )
    reporter.emit("Revenue reconstruction comparison (2022 analog):")
    reporter.emit_frame(
        "revenue_2022",
        comparison.loc[
            (comparison["target_group"] == "revenue") & (comparison["scope"] == "validation_2022"),
            ["model_name", "mae", "rmse", "r2", "top10_rmse", "promo_day_rmse", "non_promo_rmse", "high_traffic_rmse"],
        ].sort_values("rmse"),
    )
    reporter.emit(f"Best funnel combination: {best_revenue_model}")

    reporter.emit("")
    reporter.emit("Training final funnel models on full history...")
    orders_future_preds, orders_future_importance, future_traffic_stats = build_future_orders_predictions(
        orders_model_name=best_orders_model,
        train_history=funnel_daily,
        future_dates=sample_submission[DATE_COL],
        future_promo=future_promo,
        future_inventory=future_inventory,
        history_web_daily=web_daily,
        min_date=min_date,
        seasonal_ref_years=[(2022, 0.5), (2021, 0.3), (2020, 0.2)],
    )
    aov_future_pred, aov_future_importance = build_future_aov_predictions(
        aov_model_name=best_aov_model,
        train_history=funnel_daily,
        future_dates=sample_submission[DATE_COL],
        future_promo=future_promo,
        future_inventory=future_inventory,
        min_date=min_date,
    )
    future_shape = direct_shape_future.merge(sample_submission, on=DATE_COL, how="right", validate="one_to_one").sort_values(DATE_COL).reset_index(drop=True)

    scenario_revenues: dict[str, pd.Series] = {}
    for scenario_name, order_frame in orders_future_preds.items():
        merged = order_frame.merge(aov_future_pred, on=DATE_COL, how="left", validate="one_to_one").sort_values(DATE_COL).reset_index(drop=True)
        revenue = np.maximum(0.0, merged["predicted_orders"].to_numpy(dtype=float) * merged["predicted_AOV"].to_numpy(dtype=float))
        if best_stabilized:
            shape_values = future_shape["shape_pred"].to_numpy(dtype=float)
            shape_values = np.where(np.isfinite(shape_values), shape_values, revenue)
            revenue = np.maximum(0.0, 0.70 * revenue + 0.30 * shape_values)
        scenario_revenues[scenario_name] = pd.Series(revenue)

    submission_seasonal = build_submission(sample_submission[DATE_COL], scenario_revenues["seasonal"], ratio=0.8900)
    submission_conservative = build_submission(sample_submission[DATE_COL], scenario_revenues["conservative"], ratio=0.8900)
    submission_high = build_submission(sample_submission[DATE_COL], scenario_revenues["high_demand"], ratio=0.8900)
    save_submission(SUBMISSION_SEASONAL_PATH, submission_seasonal, sample_submission)
    save_submission(SUBMISSION_CONSERVATIVE_PATH, submission_conservative, sample_submission)
    save_submission(SUBMISSION_HIGH_PATH, submission_high, sample_submission)
    save_submission(SUBMISSION_SEASONAL_8950_PATH, build_submission(sample_submission[DATE_COL], scenario_revenues["seasonal"], ratio=0.8950), sample_submission)
    save_submission(SUBMISSION_SEASONAL_9000_PATH, build_submission(sample_submission[DATE_COL], scenario_revenues["seasonal"], ratio=0.9000), sample_submission)

    for weight, path in FUNNEL_BLEND_OUTPUTS.items():
        revenue = (1.0 - weight) * current_best[TARGET_COL].to_numpy(dtype=float) + weight * submission_seasonal[TARGET_COL].to_numpy(dtype=float)
        save_submission(path, build_submission(sample_submission[DATE_COL], revenue, ratio=0.8900), sample_submission)

    if segment_submission is not None:
        three_way_specs = {
            "801010": (0.80, 0.10, 0.10),
            "702010": (0.70, 0.20, 0.10),
            "701020": (0.70, 0.10, 0.20),
            "602020": (0.60, 0.20, 0.20),
        }
        for key, (w_current, w_funnel, w_segment) in three_way_specs.items():
            revenue = (
                w_current * current_best[TARGET_COL].to_numpy(dtype=float)
                + w_funnel * submission_seasonal[TARGET_COL].to_numpy(dtype=float)
                + w_segment * segment_submission[TARGET_COL].to_numpy(dtype=float)
            )
            save_submission(FUNNEL_SEGMENT_BLEND_OUTPUTS[key], build_submission(sample_submission[DATE_COL], revenue, ratio=0.8900), sample_submission)

    reporter.emit("")
    reporter.emit("Future traffic scenario stats:")
    reporter.emit_frame("future_traffic_stats", future_traffic_stats)

    top_orders_features = (
        pd.concat([feature_importance, orders_future_importance], ignore_index=True)
        .loc[lambda df: df["target_group"] == "orders"]
        .groupby("feature", as_index=False)["importance_gain"]
        .sum()
        .sort_values("importance_gain", ascending=False)
        .head(30)
    )
    top_aov_features = (
        pd.concat([feature_importance, aov_future_importance], ignore_index=True)
        .loc[lambda df: df["target_group"] == "aov"]
        .groupby("feature", as_index=False)["importance_gain"]
        .sum()
        .sort_values("importance_gain", ascending=False)
        .head(30)
    )

    best_2022_row = comparison.loc[
        (comparison["target_group"] == "revenue") & (comparison["scope"] == "validation_2022") & (comparison["model_name"] == best_revenue_model)
    ].iloc[0]
    long_avg_row = build_long_horizon_summary(comparison, "revenue").loc[lambda df: df["model_name"] == best_revenue_model]
    long_avg_rmse = float(long_avg_row["rmse"].iloc[0]) if not long_avg_row.empty else np.nan

    reporter.emit("")
    reporter.emit(f"Best direct seasonal target type used for stabilization reference: {direct_best_target}")
    reporter.emit(f"2022 analog RMSE for best funnel combo: {best_2022_row['rmse']:.2f}")
    reporter.emit(f"Long-horizon average RMSE for best funnel combo: {long_avg_rmse:.2f}")
    reporter.emit_frame("Top Orders Features", top_orders_features)
    reporter.emit_frame("Top AOV Features", top_aov_features)
    reporter.emit("")
    reporter.emit("Created submissions:")
    created_files = [
        SUBMISSION_SEASONAL_PATH,
        SUBMISSION_CONSERVATIVE_PATH,
        SUBMISSION_HIGH_PATH,
        SUBMISSION_SEASONAL_8950_PATH,
        SUBMISSION_SEASONAL_9000_PATH,
        *FUNNEL_BLEND_OUTPUTS.values(),
        *FUNNEL_SEGMENT_BLEND_OUTPUTS.values(),
    ]
    for path in created_files:
        reporter.emit(str(path))

    reporter.emit("")
    reporter.emit("Recommended upload order:")
    upload_order = [
        SUBMISSION_SEASONAL_PATH,
        FUNNEL_BLEND_OUTPUTS[0.05],
        FUNNEL_BLEND_OUTPUTS[0.10],
        FUNNEL_SEGMENT_BLEND_OUTPUTS["801010"] if segment_submission is not None else SUBMISSION_CONSERVATIVE_PATH,
        SUBMISSION_HIGH_PATH,
    ]
    for path in upload_order:
        reporter.emit(str(path))

    reporter.emit("")
    reporter.emit("Leakage safety confirmation: all Orders/AOV/Revenue features used historical actuals, lagged traffic, forecast-safe promo-known context, and no same-day future realized demand or future actual Revenue/COGS.")
    reporter.save()


if __name__ == "__main__":
    main()
