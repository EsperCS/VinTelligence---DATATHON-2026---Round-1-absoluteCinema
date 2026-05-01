from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import train_funnel_model as funnel
import train_feature_union_model as union_model
import train_promo_known_pipeline as promo_known
import train_stock_aware_scaling as stock_scale
import train_traffic_driven_model as traffic_branch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "log"

DATE_COL = "Date"
TARGET_COL = "Revenue"
COGS_COL = "COGS"
RANDOM_STATE = 42
RATIO = 0.8900
EPS = 1e-9

ORDERS_AOV_TABLE_PATH = DATA_DIR / "daily_orders_aov_table.csv"
ORDERS_VALIDATION_PATH = DATA_DIR / "orders_aov_orders_validation_predictions.csv"
AOV_VALIDATION_PATH = DATA_DIR / "orders_aov_aov_validation_predictions.csv"
REVENUE_VALIDATION_PATH = DATA_DIR / "orders_aov_revenue_validation_predictions.csv"
MODEL_COMPARISON_PATH = DATA_DIR / "orders_aov_model_comparison.csv"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "orders_aov_feature_importance.csv"
REPORT_PATH = LOG_DIR / "orders_aov_pipeline_report.txt"
LOG_PATH = LOG_DIR / "train_orders_aov_pipeline.log"

CURRENT_BEST_SUBMISSION_PATH = DATA_DIR / "submission_blend_direct_15_cogs8900.csv"
ANTI_OVERFIT_BEST_PATH = DATA_DIR / "submission_anti_overfit_9505.csv"

SUBMISSION_FUNNEL_PATH = DATA_DIR / "submission_orders_aov_funnel.csv"
SUBMISSION_FUNNEL_CONSERVATIVE_PATH = DATA_DIR / "submission_orders_aov_funnel_conservative.csv"
SUBMISSION_FUNNEL_AGGRESSIVE_PATH = DATA_DIR / "submission_orders_aov_funnel_aggressive.csv"

ORDERS_CORRECTION_OUTPUTS = {
    "9505": DATA_DIR / "submission_orders_ratio_corrected_9505.csv",
    "9307": DATA_DIR / "submission_orders_ratio_corrected_9307.csv",
    "9010": DATA_DIR / "submission_orders_ratio_corrected_9010.csv",
}
AOV_CORRECTION_OUTPUTS = {
    "9505": DATA_DIR / "submission_aov_ratio_corrected_9505.csv",
    "9307": DATA_DIR / "submission_aov_ratio_corrected_9307.csv",
}
COMBINED_CORRECTION_OUTPUTS = {
    "soft": DATA_DIR / "submission_orders_aov_corrected_soft.csv",
    "medium": DATA_DIR / "submission_orders_aov_corrected_medium.csv",
    "strong": DATA_DIR / "submission_orders_aov_corrected_strong.csv",
}
BLEND_OUTPUTS = {
    0.03: DATA_DIR / "submission_orders_aov_blend_03.csv",
    0.05: DATA_DIR / "submission_orders_aov_blend_05.csv",
    0.08: DATA_DIR / "submission_orders_aov_blend_08.csv",
    0.10: DATA_DIR / "submission_orders_aov_blend_10.csv",
    0.15: DATA_DIR / "submission_orders_aov_blend_15.csv",
    0.20: DATA_DIR / "submission_orders_aov_blend_20.csv",
}
STOCK_BLEND_OUTPUTS = {
    0.03: DATA_DIR / "submission_orders_aov_stock_blend_03.csv",
    0.05: DATA_DIR / "submission_orders_aov_stock_blend_05.csv",
    0.08: DATA_DIR / "submission_orders_aov_stock_blend_08.csv",
}

VALIDATION_2022 = ("validation_2022", pd.Timestamp("2021-12-31"), pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"))
LONG_FOLDS = [
    ("fold_1", pd.Timestamp("2019-06-30"), pd.Timestamp("2019-07-01"), pd.Timestamp("2020-12-31")),
    ("fold_2", pd.Timestamp("2020-06-30"), pd.Timestamp("2020-07-01"), pd.Timestamp("2021-12-31")),
    ("fold_3", pd.Timestamp("2021-06-30"), pd.Timestamp("2021-07-01"), pd.Timestamp("2022-12-31")),
]
ALL_SCOPES = [VALIDATION_2022] + LONG_FOLDS
HIGH_RISK_MONTHS = {2, 3, 5, 8}

PROMO_COLUMNS = funnel.PROMO_COLUMNS
CAMPAIGN_COLUMNS = funnel.CAMPAIGN_COLUMNS
INVENTORY_COLUMNS = funnel.INVENTORY_COLUMNS
TRAFFIC_SOURCES = traffic_branch.TRAFFIC_SOURCES
TRAFFIC_FEATURE_COLUMNS = [
    "sessions_sum",
    "unique_visitors_sum",
    "page_views_sum",
    "avg_bounce_rate",
    "avg_session_duration_sec",
    "source_diversity_count",
    "sessions_roll_mean_7",
    "sessions_roll_mean_30",
    "sessions_growth_1_7",
    "sessions_growth_3_14",
    "traffic_spike_125",
    "engagement_index",
] + [f"{source}_sessions" for source in TRAFFIC_SOURCES]

ORDERS_MODEL_A_FEATURES = [
    "orders_lag_365",
    "orders_lag_730",
    "orders_lag_1095",
    "orders_same_day_recent_mean",
    "orders_same_month_recent_mean",
    "orders_same_dow_month_mean",
    "day_of_week",
    "day_of_year",
    "month",
    "week_of_year",
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
    "inv_avg_days_of_supply",
    "inv_avg_sell_through_rate",
    "inv_stockout_rate",
] + CAMPAIGN_COLUMNS

ORDERS_MODEL_B_FEATURES = [
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
    "engagement_index",
] + CAMPAIGN_COLUMNS

AOV_MODEL_A_FEATURES = [
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
    "aov_roll_median_30",
    "avg_discount_per_order_lag_365",
    "promo_item_share_lag_365",
    "item_lines_per_order_lag_365",
    "quantity_per_order_lag_365",
] + CAMPAIGN_COLUMNS

AOV_MODEL_B_FEATURES = [
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
    "aov_lag_730",
    "aov_lag_1095",
    "aov_same_day_recent_mean",
    "aov_same_month_recent_mean",
    "avg_discount_per_order_lag_365",
    "promo_item_share_lag_365",
    "item_lines_per_order_lag_365",
    "quantity_per_order_lag_365",
] + CAMPAIGN_COLUMNS

AOV_MODEL_D_FEATURES = [
    "day_of_week",
    "day_of_year",
    "month",
    "is_weekend",
    "calendar_any_promo",
    "calendar_avg_discount_value",
    "calendar_active_promo_count",
    "calendar_max_discount_value",
    "promo_progress_ratio",
    "promo_days_remaining",
    "campaign_intensity",
    "discount_x_progress",
    "discount_x_days_remaining",
    "promotion_campaign_index",
    "aov_lag_7",
    "aov_lag_30",
    "aov_lag_365",
    "aov_roll_mean_7",
    "aov_roll_mean_30",
    "avg_discount_per_order_lag_365",
    "promo_item_share_lag_365",
    "item_lines_per_order_lag_365",
    "quantity_per_order_lag_365",
] + CAMPAIGN_COLUMNS


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
        if isinstance(frame, pd.Series):
            self.emit(frame.to_string() if not frame.empty else "(empty)")
        else:
            self.emit(frame.to_string(index=False) if not frame.empty else "(empty)")

    def save(self) -> None:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_orders_aov_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger


def safe_divide(num: Any, den: Any, fill_value: float = 0.0):
    numerator = np.asarray(num, dtype=float)
    denominator = np.asarray(den, dtype=float)
    return np.divide(numerator, denominator, out=np.full_like(numerator, fill_value, dtype=float), where=np.abs(denominator) > EPS)


def mae(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float))))


def rmse(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)))))


def r2_score_manual(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    actual_arr = np.asarray(actual, dtype=float)
    pred_arr = np.asarray(predicted, dtype=float)
    denom = float(np.sum(np.square(actual_arr - np.mean(actual_arr))))
    if denom <= EPS:
        return 0.0
    return 1.0 - float(np.sum(np.square(actual_arr - pred_arr))) / denom


def compute_basic_metrics(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> dict[str, float]:
    return {
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "r2": r2_score_manual(actual, predicted),
    }


def compute_revenue_metrics(
    actual_revenue: pd.Series,
    predicted_revenue: np.ndarray,
    promo_mask: pd.Series,
    actual_orders: pd.Series,
    actual_aov: pd.Series,
    predicted_orders: np.ndarray | None = None,
    predicted_aov: np.ndarray | None = None,
) -> dict[str, float]:
    actual = np.asarray(actual_revenue, dtype=float)
    predicted = np.asarray(predicted_revenue, dtype=float)
    promo = promo_mask.fillna(0).astype(bool).to_numpy()
    high_orders = np.asarray(actual_orders, dtype=float) >= float(np.quantile(np.asarray(actual_orders, dtype=float), 0.90))
    high_aov = np.asarray(actual_aov, dtype=float) >= float(np.quantile(np.asarray(actual_aov, dtype=float), 0.90))
    top10 = actual >= float(np.quantile(actual, 0.90))
    metrics = {
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "r2": r2_score_manual(actual, predicted),
        "top10_rmse": rmse(actual[top10], predicted[top10]) if top10.any() else np.nan,
        "promo_day_rmse": rmse(actual[promo], predicted[promo]) if promo.any() else np.nan,
        "non_promo_rmse": rmse(actual[~promo], predicted[~promo]) if (~promo).any() else np.nan,
        "high_orders_day_rmse": rmse(actual[high_orders], predicted[high_orders]) if high_orders.any() else np.nan,
        "high_aov_day_rmse": rmse(actual[high_aov], predicted[high_aov]) if high_aov.any() else np.nan,
    }
    if predicted_orders is not None and predicted_aov is not None:
        orders_contrib = (np.asarray(predicted_orders, dtype=float) - np.asarray(actual_orders, dtype=float)) * np.asarray(actual_aov, dtype=float)
        aov_contrib = np.asarray(predicted_orders, dtype=float) * (np.asarray(predicted_aov, dtype=float) - np.asarray(actual_aov, dtype=float))
        metrics["orders_error_contribution_mae"] = float(np.mean(np.abs(orders_contrib)))
        metrics["aov_error_contribution_mae"] = float(np.mean(np.abs(aov_contrib)))
    return metrics


def validate_submission_frame(output: pd.DataFrame, sample_submission: pd.DataFrame) -> None:
    if list(output.columns) != [DATE_COL, TARGET_COL, COGS_COL]:
        raise ValueError("Submission columns must be exactly Date, Revenue, COGS")
    if len(output) != len(sample_submission):
        raise ValueError("Submission row count mismatch")
    if not output[DATE_COL].equals(sample_submission[DATE_COL]):
        raise ValueError("Submission dates do not match sample_submission order")
    if output.isna().any().any():
        raise ValueError("Submission contains missing values")
    if (output[[TARGET_COL, COGS_COL]] < 0).any().any():
        raise ValueError("Submission contains negative Revenue or COGS")


def build_submission(dates: pd.Series, revenue: np.ndarray, ratio: float = RATIO) -> pd.DataFrame:
    output = pd.DataFrame({DATE_COL: pd.to_datetime(dates).reset_index(drop=True)})
    output[TARGET_COL] = np.maximum(0.0, np.asarray(revenue, dtype=float))
    output[COGS_COL] = np.maximum(0.0, output[TARGET_COL] * ratio)
    return output[[DATE_COL, TARGET_COL, COGS_COL]]


def save_submission_no_overwrite(path: Path, submission: pd.DataFrame, sample_submission: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing submission: {path}")
    validate_submission_frame(submission, sample_submission)
    submission.to_csv(path, index=False)


def load_sample_submission() -> pd.DataFrame:
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv", parse_dates=[DATE_COL], low_memory=False)
    sample[DATE_COL] = pd.to_datetime(sample[DATE_COL], errors="coerce").dt.normalize()
    return sample[[DATE_COL]].copy()


def build_daily_orders_aov_table(logger: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sales = funnel.load_sales()
    orders = funnel.load_orders()
    order_items = funnel.load_order_items()
    products = funnel.load_products()
    web_raw = pd.read_csv(DATA_DIR / "web_traffic.csv", low_memory=False)
    web_daily = funnel.normalize_date_column(traffic_branch.build_web_daily(web_raw))
    promotions = promo_known.load_promotions(DATA_DIR / "promotions.csv")
    promo_daily = promo_known.build_daily_promo_known_features(sales[DATE_COL], promotions)
    inventory_snapshots = funnel.prepare_inventory_snapshots()
    inventory_daily = funnel.build_inventory_context(sales[DATE_COL], inventory_snapshots)

    daily = funnel.build_daily_funnel_table(sales, orders, order_items, products, web_daily, promo_daily, inventory_daily)

    orders_status = orders.copy()
    orders_status["order_status"] = orders_status["order_status"].fillna("unknown").astype(str).str.strip().str.lower()
    status_daily = (
        orders_status.groupby("order_date", as_index=False)
        .agg(
            cancelled_order_count=("order_status", lambda s: int(np.sum(pd.Series(s).isin(["cancelled"])))),
            shipped_or_delivered_order_count=("order_status", lambda s: int(np.sum(pd.Series(s).isin(["shipped", "delivered"])))),
        )
        .rename(columns={"order_date": DATE_COL})
    )

    order_dates = orders[["order_id", "order_date"]].copy()
    order_dates["order_date"] = pd.to_datetime(order_dates["order_date"], errors="coerce").dt.normalize()
    items = order_items.merge(order_dates, on="order_id", how="left", validate="many_to_one")
    items = items.dropna(subset=["order_date"]).copy()
    items["promo_item_flag"] = (items["promo_id"].notna() | items["promo_id_2"].notna()).astype(int)
    item_extra = (
        items.groupby("order_date", as_index=False)
        .agg(
            promo_item_count=("promo_item_flag", "sum"),
        )
        .rename(columns={"order_date": DATE_COL})
    )

    daily = (
        daily.merge(status_daily, on=DATE_COL, how="left", validate="one_to_one")
        .merge(item_extra, on=DATE_COL, how="left", validate="one_to_one")
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )
    numeric_cols = [column for column in daily.columns if column != DATE_COL]
    daily[numeric_cols] = daily[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    daily["avg_item_lines_per_order"] = safe_divide(daily["item_lines"], daily["orders_count"], fill_value=0.0)
    daily["avg_quantity_per_order"] = safe_divide(daily["total_quantity"], daily["orders_count"], fill_value=0.0)
    daily["promo_item_share"] = safe_divide(daily["promo_item_count"], daily["item_lines"], fill_value=0.0)
    daily["avg_unit_price"] = safe_divide(daily["gross_item_value"], daily["total_quantity"], fill_value=0.0)
    daily["net_item_value"] = pd.to_numeric(daily["total_net_value"], errors="coerce").fillna(0.0)
    daily["AOV"] = safe_divide(daily[TARGET_COL], daily["orders_count"], fill_value=0.0)
    daily["COGS_per_order"] = safe_divide(daily[COGS_COL], daily["orders_count"], fill_value=0.0)
    daily["Revenue_per_item_line"] = safe_divide(daily[TARGET_COL], daily["item_lines"], fill_value=0.0)
    daily["Revenue_per_quantity"] = safe_divide(daily[TARGET_COL], daily["total_quantity"], fill_value=0.0)
    daily["conversion_rate"] = safe_divide(daily["orders_count"], daily["sessions_sum"], fill_value=0.0)
    daily["revenue_per_session"] = safe_divide(daily[TARGET_COL], daily["sessions_sum"], fill_value=0.0)
    daily["quantity_per_order"] = safe_divide(daily["total_quantity"], daily["orders_count"], fill_value=0.0)
    daily["item_lines_per_order"] = safe_divide(daily["item_lines"], daily["orders_count"], fill_value=0.0)

    daily.to_csv(ORDERS_AOV_TABLE_PATH, index=False)
    logger.info("Saved daily orders/AOV table to %s", ORDERS_AOV_TABLE_PATH)
    return sales, daily, web_daily, promo_daily, inventory_snapshots


def add_orders_aov_features(daily: pd.DataFrame) -> pd.DataFrame:
    table = daily.sort_values(DATE_COL).reset_index(drop=True).copy()
    min_date = table[DATE_COL].min()
    calendar = funnel.build_calendar_features(table[DATE_COL], min_date)
    table = table.merge(calendar, on=DATE_COL, how="left", validate="one_to_one")

    table = funnel.add_recursive_lag_features(table, "orders_count", "orders", [7, 14, 30, 90, 365], [7, 30, 90, 365])
    table = funnel.add_recursive_lag_features(table, "AOV", "aov", [7, 14, 30, 90, 365], [7, 30, 90, 365])

    table["aov_roll_median_30"] = table["AOV"].shift(1).rolling(window=30, min_periods=30).median()

    order_refs = funnel.build_series_reference_frame(table[DATE_COL], table[[DATE_COL, "orders_count"]], "orders_count", "orders")
    aov_refs = funnel.build_series_reference_frame(table[DATE_COL], table[[DATE_COL, "AOV"]], "AOV", "aov")
    discount_refs = funnel.build_series_reference_frame(table[DATE_COL], table[[DATE_COL, "avg_discount_per_order"]], "avg_discount_per_order", "avg_discount_per_order")
    promo_item_refs = funnel.build_series_reference_frame(table[DATE_COL], table[[DATE_COL, "promo_item_share"]], "promo_item_share", "promo_item_share")
    quantity_refs = funnel.build_series_reference_frame(table[DATE_COL], table[[DATE_COL, "quantity_per_order"]], "quantity_per_order", "quantity_per_order")
    item_lines_refs = funnel.build_series_reference_frame(table[DATE_COL], table[[DATE_COL, "item_lines_per_order"]], "item_lines_per_order", "item_lines_per_order")

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
        .merge(discount_refs[[DATE_COL, "avg_discount_per_order_lag_365"]], on=DATE_COL, how="left", validate="one_to_one")
        .merge(promo_item_refs[[DATE_COL, "promo_item_share_lag_365"]], on=DATE_COL, how="left", validate="one_to_one")
        .merge(quantity_refs[[DATE_COL, "quantity_per_order_lag_365"]], on=DATE_COL, how="left", validate="one_to_one")
        .merge(item_lines_refs[[DATE_COL, "item_lines_per_order_lag_365"]], on=DATE_COL, how="left", validate="one_to_one")
    )
    table["orders_same_dow_month_mean"] = build_same_dow_month_reference_series(table[DATE_COL], table[[DATE_COL, "orders_count"]], "orders_count")
    return table


def build_same_dow_month_reference_series(target_dates: pd.Series, history_df: pd.DataFrame, value_col: str) -> pd.Series:
    frame = history_df[[DATE_COL, value_col]].copy()
    frame["year"] = frame[DATE_COL].dt.year.astype(int)
    frame["month"] = frame[DATE_COL].dt.month.astype(int)
    frame["day_of_week"] = frame[DATE_COL].dt.dayofweek.astype(int)
    grouped = frame.groupby(["year", "month", "day_of_week"], as_index=False)[value_col].mean()
    fallback = frame.groupby(["month", "day_of_week"], as_index=False)[value_col].mean().rename(columns={value_col: "fallback_value"})

    values: list[float] = []
    grouped_lookup = grouped.set_index(["year", "month", "day_of_week"])[value_col]
    fallback_lookup = fallback.set_index(["month", "day_of_week"])["fallback_value"]
    for target_date in pd.to_datetime(target_dates):
        month = int(target_date.month)
        dow = int(target_date.dayofweek)
        weighted_vals: list[float] = []
        weighted_wts: list[float] = []
        for years_back, weight in [(1, 0.5), (2, 0.3), (3, 0.2)]:
            key = (int(target_date.year) - years_back, month, dow)
            if key in grouped_lookup.index:
                value = float(pd.to_numeric(grouped_lookup.loc[key], errors="coerce"))
                if np.isfinite(value):
                    weighted_vals.append(value * weight)
                    weighted_wts.append(weight)
        if weighted_wts:
            values.append(float(sum(weighted_vals) / sum(weighted_wts)))
        elif (month, dow) in fallback_lookup.index:
            values.append(float(pd.to_numeric(fallback_lookup.loc[(month, dow)], errors="coerce")))
        else:
            values.append(np.nan)
    return pd.Series(values)


def prepare_training_matrix(
    table: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    train_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    use_cols = [DATE_COL, target_column] + [column for column in feature_columns if column in table.columns]
    subset = table.loc[table[DATE_COL] <= train_end, use_cols].copy()
    subset = subset.dropna(subset=[target_column]).reset_index(drop=True)
    X = subset[[column for column in feature_columns if column in subset.columns]].replace([np.inf, -np.inf], np.nan)
    medians = X.median(numeric_only=True).fillna(0.0)
    X = X.fillna(medians).fillna(0.0)
    y = pd.to_numeric(subset[target_column], errors="coerce").fillna(0.0)
    return X.reset_index(drop=True), y.reset_index(drop=True), medians, subset.reset_index(drop=True)


def train_lightgbm_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: np.ndarray | None = None,
) -> Any:
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
    dataset = lgb.Dataset(X_train, label=y_train, weight=sample_weight, free_raw_data=False)
    model = lgb.train(params=params, train_set=dataset, num_boost_round=260)
    return model


def predict_lightgbm(model: Any, frame: pd.DataFrame, feature_columns: list[str], medians: pd.Series) -> np.ndarray:
    deduped = frame.loc[:, ~frame.columns.duplicated()].copy()
    X = deduped[[column for column in feature_columns if column in deduped.columns]].copy()
    X = X.reindex(columns=feature_columns).replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)
    pred = np.asarray(model.predict(X), dtype=float)
    return np.maximum(0.0, pred)


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


def build_orders_direct_context(
    train_history: pd.DataFrame,
    target_dates: pd.Series,
    traffic_context: pd.DataFrame,
    promo_context: pd.DataFrame,
    inventory_context: pd.DataFrame,
    min_date: pd.Timestamp,
) -> pd.DataFrame:
    context = funnel.build_orders_direct_context(train_history, target_dates, traffic_context, promo_context, inventory_context, min_date)
    refs = pd.DataFrame({DATE_COL: pd.to_datetime(target_dates).sort_values().unique()})
    refs["orders_same_dow_month_mean"] = build_same_dow_month_reference_series(refs[DATE_COL], train_history[[DATE_COL, "orders_count"]], "orders_count")
    context = context.merge(refs, on=DATE_COL, how="left", validate="one_to_one")
    return context


def build_aov_direct_context(train_history: pd.DataFrame, target_dates: pd.Series, promo_context: pd.DataFrame, inventory_context: pd.DataFrame, min_date: pd.Timestamp) -> pd.DataFrame:
    context = funnel.build_aov_direct_context(train_history, target_dates, promo_context, inventory_context, min_date)
    aov_hist = train_history[[DATE_COL, "AOV"]].sort_values(DATE_COL).reset_index(drop=True).copy()
    aov_hist["aov_roll_median_30"] = aov_hist["AOV"].shift(1).rolling(window=30, min_periods=30).median()
    med_map = aov_hist.set_index(DATE_COL)["aov_roll_median_30"]
    promo_item_refs = funnel.build_series_reference_frame(target_dates, train_history[[DATE_COL, "promo_item_share"]], "promo_item_share", "promo_item_share")
    quantity_refs = funnel.build_series_reference_frame(target_dates, train_history[[DATE_COL, "quantity_per_order"]], "quantity_per_order", "quantity_per_order")
    item_lines_refs = funnel.build_series_reference_frame(target_dates, train_history[[DATE_COL, "item_lines_per_order"]], "item_lines_per_order", "item_lines_per_order")
    refs = []
    for target_date in pd.to_datetime(target_dates).sort_values().unique():
        target_date = pd.Timestamp(target_date).normalize()
        ref_date = target_date - pd.Timedelta(days=1)
        refs.append({DATE_COL: target_date, "aov_roll_median_30": float(pd.to_numeric(med_map.loc[ref_date], errors="coerce")) if ref_date in med_map.index else np.nan})
    return (
        context.merge(pd.DataFrame(refs), on=DATE_COL, how="left", validate="one_to_one")
        .merge(promo_item_refs[[DATE_COL, "promo_item_share_lag_365"]], on=DATE_COL, how="left", validate="one_to_one")
        .merge(quantity_refs[[DATE_COL, "quantity_per_order_lag_365"]], on=DATE_COL, how="left", validate="one_to_one")
        .merge(item_lines_refs[[DATE_COL, "item_lines_per_order_lag_365"]], on=DATE_COL, how="left", validate="one_to_one")
    )


def build_recency_weights(dates: pd.Series) -> np.ndarray:
    years = pd.to_datetime(dates).dt.year.astype(int)
    weights = np.where(years <= 2018, 0.5, np.where(years <= 2020, 0.8, 1.2))
    return weights.astype(float)


def build_orders_spike_weights(values: pd.Series) -> np.ndarray:
    threshold = float(np.quantile(pd.to_numeric(values, errors="coerce").fillna(0.0), 0.90))
    weights = np.where(pd.to_numeric(values, errors="coerce").fillna(0.0) >= threshold, 2.0, 1.0)
    return weights.astype(float)


def build_aov_promo_weights(subset: pd.DataFrame) -> np.ndarray:
    promo_flag = pd.to_numeric(subset.get("calendar_any_promo", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    weights = np.where(promo_flag > 0, 1.6, 1.0)
    return weights.astype(float)


def clip_series(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), low, high)


def select_columns(frame: pd.DataFrame, desired: list[str]) -> list[str]:
    return [column for column in desired if column in frame.columns]


def evaluate_scope(
    scope_name: str,
    train_end: pd.Timestamp,
    valid_start: pd.Timestamp,
    valid_end: pd.Timestamp,
    full_table: pd.DataFrame,
    web_daily: pd.DataFrame,
    promo_daily: pd.DataFrame,
    inventory_snapshots: pd.DataFrame,
    min_date: pd.Timestamp,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]]:
    history = full_table.loc[full_table[DATE_COL] <= train_end].copy()
    actual_window = full_table.loc[(full_table[DATE_COL] >= valid_start) & (full_table[DATE_COL] <= valid_end)].copy()
    valid_dates = pd.Series(pd.date_range(valid_start, valid_end, freq="D"))
    promo_valid = promo_daily.loc[(promo_daily[DATE_COL] >= valid_start) & (promo_daily[DATE_COL] <= valid_end)].copy()
    inventory_valid = funnel.build_inventory_context(valid_dates, inventory_snapshots.loc[inventory_snapshots[DATE_COL] <= valid_end].copy())
    traffic_scenarios, _ = funnel.build_traffic_scenarios_for_dates(
        web_daily.loc[web_daily[DATE_COL] <= train_end].copy(),
        valid_dates,
        promo_valid,
        [(train_end.year, 0.5), (train_end.year - 1, 0.3), (train_end.year - 2, 0.2)],
    )
    traffic_raw = traffic_scenarios["seasonal"].copy()
    traffic_features = funnel.add_traffic_features_for_scenario(web_daily.loc[web_daily[DATE_COL] <= train_end].copy(), traffic_raw)
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

    comparison_rows: list[dict[str, Any]] = []
    orders_frames: list[pd.DataFrame] = []
    aov_frames: list[pd.DataFrame] = []
    revenue_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []

    # Orders A direct seasonal
    orders_a_features = select_columns(full_table, ORDERS_MODEL_A_FEATURES)
    X_a, y_a, med_a, subset_a = prepare_training_matrix(full_table, orders_a_features, "orders_count", train_end)
    model_orders_a = train_lightgbm_regression(X_a, y_a)
    context_orders_a = build_orders_direct_context(history, valid_dates, traffic_direct_context, promo_valid, inventory_valid, min_date)
    pred_orders_a = predict_lightgbm(model_orders_a, context_orders_a, orders_a_features, med_a)
    orders_a_frame = pd.DataFrame({DATE_COL: valid_dates, "actual_orders": actual_window["orders_count"].to_numpy(dtype=float), "predicted_orders": pred_orders_a, "orders_model": "orders_a_direct_seasonal", "scope": scope_name})
    orders_frames.append(orders_a_frame)
    comparison_rows.append({"target_group": "orders", "model_name": "orders_a_direct_seasonal", "scope": scope_name, **compute_basic_metrics(actual_window["orders_count"], pred_orders_a)})
    importance_frames.append(extract_feature_importance(model_orders_a, orders_a_features, "orders_a_direct_seasonal", "orders"))

    # Orders B recursive safe
    orders_b_features = select_columns(full_table, ORDERS_MODEL_B_FEATURES)
    X_b, y_b, med_b, subset_b = prepare_training_matrix(full_table, orders_b_features, "orders_count", train_end)
    model_orders_b = train_lightgbm_regression(X_b, y_b)
    context_orders_b = funnel.build_recursive_static_context(valid_dates, traffic_features, promo_valid, inventory_valid, min_date)
    orders_b_pred_frame = funnel.recursive_predict_orders(model_orders_b, med_b, orders_b_features, history, context_orders_b)
    orders_b_frame = pd.DataFrame({DATE_COL: valid_dates, "actual_orders": actual_window["orders_count"].to_numpy(dtype=float), "predicted_orders": orders_b_pred_frame["predicted_orders"].to_numpy(dtype=float), "orders_model": "orders_b_recursive_safe", "scope": scope_name})
    orders_frames.append(orders_b_frame)
    comparison_rows.append({"target_group": "orders", "model_name": "orders_b_recursive_safe", "scope": scope_name, **compute_basic_metrics(actual_window["orders_count"], orders_b_frame["predicted_orders"])})
    importance_frames.append(extract_feature_importance(model_orders_b, orders_b_features, "orders_b_recursive_safe", "orders"))

    # Orders C weighted recent
    X_c, y_c, med_c, subset_c = prepare_training_matrix(full_table, orders_b_features, "orders_count", train_end)
    weights_recent = build_recency_weights(subset_c[DATE_COL])
    model_orders_c = train_lightgbm_regression(X_c, y_c, sample_weight=weights_recent)
    orders_c_pred_frame = funnel.recursive_predict_orders(model_orders_c, med_c, orders_b_features, history, context_orders_b)
    orders_c_frame = pd.DataFrame({DATE_COL: valid_dates, "actual_orders": actual_window["orders_count"].to_numpy(dtype=float), "predicted_orders": orders_c_pred_frame["predicted_orders"].to_numpy(dtype=float), "orders_model": "orders_c_weighted_recent", "scope": scope_name})
    orders_frames.append(orders_c_frame)
    comparison_rows.append({"target_group": "orders", "model_name": "orders_c_weighted_recent", "scope": scope_name, **compute_basic_metrics(actual_window["orders_count"], orders_c_frame["predicted_orders"])})
    importance_frames.append(extract_feature_importance(model_orders_c, orders_b_features, "orders_c_weighted_recent", "orders"))

    # Orders D spike weighted
    X_d, y_d, med_d, subset_d = prepare_training_matrix(full_table, orders_b_features, "orders_count", train_end)
    spike_weights = build_orders_spike_weights(subset_d["orders_count"])
    model_orders_d = train_lightgbm_regression(X_d, y_d, sample_weight=spike_weights)
    orders_d_pred_frame = funnel.recursive_predict_orders(model_orders_d, med_d, orders_b_features, history, context_orders_b)
    orders_d_frame = pd.DataFrame({DATE_COL: valid_dates, "actual_orders": actual_window["orders_count"].to_numpy(dtype=float), "predicted_orders": orders_d_pred_frame["predicted_orders"].to_numpy(dtype=float), "orders_model": "orders_d_spike_weighted", "scope": scope_name})
    orders_frames.append(orders_d_frame)
    comparison_rows.append({"target_group": "orders", "model_name": "orders_d_spike_weighted", "scope": scope_name, **compute_basic_metrics(actual_window["orders_count"], orders_d_frame["predicted_orders"])})
    importance_frames.append(extract_feature_importance(model_orders_d, orders_b_features, "orders_d_spike_weighted", "orders"))

    # AOV A direct recursive
    aov_a_features = select_columns(full_table, AOV_MODEL_A_FEATURES)
    X_aa, y_aa, med_aa, subset_aa = prepare_training_matrix(full_table, aov_a_features, "AOV", train_end)
    model_aov_a = train_lightgbm_regression(X_aa, y_aa)
    context_aov_a = build_aov_direct_context(history, valid_dates, promo_valid, inventory_valid, min_date)
    aov_a_pred_frame = funnel.recursive_predict_aov(model_aov_a, med_aa, aov_a_features, history, context_aov_a)
    aov_a_frame = pd.DataFrame({DATE_COL: valid_dates, "actual_AOV": actual_window["AOV"].to_numpy(dtype=float), "predicted_AOV": aov_a_pred_frame["predicted_AOV"].to_numpy(dtype=float), "aov_model": "aov_a_direct", "scope": scope_name})
    aov_frames.append(aov_a_frame)
    comparison_rows.append({"target_group": "aov", "model_name": "aov_a_direct", "scope": scope_name, **compute_basic_metrics(actual_window["AOV"], aov_a_frame["predicted_AOV"])})
    importance_frames.append(extract_feature_importance(model_aov_a, aov_a_features, "aov_a_direct", "aov"))

    # AOV B seasonal ratio, two clip variants
    aov_b_features = select_columns(full_table, AOV_MODEL_B_FEATURES)
    ratio_train = full_table.copy()
    ratio_train["aov_ratio_target"] = safe_divide(ratio_train["AOV"], ratio_train["aov_lag_365"].replace(0.0, np.nan), fill_value=np.nan)
    X_ab, y_ab, med_ab, subset_ab = prepare_training_matrix(ratio_train, aov_b_features, "aov_ratio_target", train_end)
    model_aov_b = train_lightgbm_regression(X_ab, y_ab)
    context_aov_b = build_aov_direct_context(history, valid_dates, promo_valid, inventory_valid, min_date)
    ratio_pred = predict_lightgbm(model_aov_b, context_aov_b, aov_b_features, med_ab)
    train_ratio_series = pd.to_numeric(subset_ab["aov_ratio_target"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    clip_variants = {
        "aov_b_ratio_p5p95": (float(train_ratio_series.quantile(0.05)), float(train_ratio_series.quantile(0.95))),
        "aov_b_ratio_p10p90": (float(train_ratio_series.quantile(0.10)), float(train_ratio_series.quantile(0.90))),
    }
    for aov_model_name, (clip_lo, clip_hi) in clip_variants.items():
        clipped_ratio = clip_series(ratio_pred, clip_lo, clip_hi)
        aov_baseline = pd.to_numeric(context_aov_b["aov_lag_365"], errors="coerce").fillna(pd.to_numeric(context_aov_b["aov_same_day_recent_mean"], errors="coerce")).fillna(float(history["AOV"].median())).to_numpy(dtype=float)
        pred_aov = np.maximum(0.0, clipped_ratio * aov_baseline)
        frame = pd.DataFrame({DATE_COL: valid_dates, "actual_AOV": actual_window["AOV"].to_numpy(dtype=float), "predicted_AOV": pred_aov, "aov_model": aov_model_name, "scope": scope_name, "predicted_ratio": clipped_ratio})
        aov_frames.append(frame)
        comparison_rows.append({"target_group": "aov", "model_name": aov_model_name, "scope": scope_name, **compute_basic_metrics(actual_window["AOV"], pred_aov)})
    importance_frames.append(extract_feature_importance(model_aov_b, aov_b_features, "aov_b_ratio", "aov"))

    # AOV C robust median baseline
    aov_c_pred_frame = funnel.baseline_aov_predictions(history, valid_dates, promo_valid)
    aov_c_frame = pd.DataFrame({DATE_COL: valid_dates, "actual_AOV": actual_window["AOV"].to_numpy(dtype=float), "predicted_AOV": aov_c_pred_frame["predicted_AOV"].to_numpy(dtype=float), "aov_model": "aov_c_baseline", "scope": scope_name})
    aov_frames.append(aov_c_frame)
    comparison_rows.append({"target_group": "aov", "model_name": "aov_c_baseline", "scope": scope_name, **compute_basic_metrics(actual_window["AOV"], aov_c_frame["predicted_AOV"])})

    # AOV D promo-adjusted
    aov_d_features = select_columns(full_table, AOV_MODEL_D_FEATURES)
    X_ad, y_ad, med_ad, subset_ad = prepare_training_matrix(full_table, aov_d_features, "AOV", train_end)
    aov_promo_weights = build_aov_promo_weights(subset_ad)
    model_aov_d = train_lightgbm_regression(X_ad, y_ad, sample_weight=aov_promo_weights)
    context_aov_d = build_aov_direct_context(history, valid_dates, promo_valid, inventory_valid, min_date)
    aov_d_pred_frame = funnel.recursive_predict_aov(model_aov_d, med_ad, aov_d_features, history, context_aov_d)
    aov_d_frame = pd.DataFrame({DATE_COL: valid_dates, "actual_AOV": actual_window["AOV"].to_numpy(dtype=float), "predicted_AOV": aov_d_pred_frame["predicted_AOV"].to_numpy(dtype=float), "aov_model": "aov_d_promo_adjusted", "scope": scope_name})
    aov_frames.append(aov_d_frame)
    comparison_rows.append({"target_group": "aov", "model_name": "aov_d_promo_adjusted", "scope": scope_name, **compute_basic_metrics(actual_window["AOV"], aov_d_frame["predicted_AOV"])})
    importance_frames.append(extract_feature_importance(model_aov_d, aov_d_features, "aov_d_promo_adjusted", "aov"))

    orders_map = {
        "orders_a_direct_seasonal": orders_a_frame,
        "orders_b_recursive_safe": orders_b_frame,
        "orders_c_weighted_recent": orders_c_frame,
        "orders_d_spike_weighted": orders_d_frame,
    }
    aov_map = {
        "aov_a_direct": aov_a_frame,
        "aov_b_ratio_p5p95": pd.DataFrame({DATE_COL: valid_dates, "predicted_AOV": aov_frames[-3]["predicted_AOV"].to_numpy(dtype=float)}),
        "aov_b_ratio_p10p90": pd.DataFrame({DATE_COL: valid_dates, "predicted_AOV": aov_frames[-2]["predicted_AOV"].to_numpy(dtype=float)}),
        "aov_c_baseline": aov_c_frame,
        "aov_d_promo_adjusted": aov_d_frame,
    }

    revenue_combos = [
        ("orders_a_direct_seasonal", "aov_a_direct"),
        ("orders_a_direct_seasonal", "aov_b_ratio_p5p95"),
        ("orders_a_direct_seasonal", "aov_c_baseline"),
        ("orders_b_recursive_safe", "aov_a_direct"),
        ("orders_b_recursive_safe", "aov_b_ratio_p10p90"),
        ("orders_c_weighted_recent", "aov_a_direct"),
        ("orders_d_spike_weighted", "aov_d_promo_adjusted"),
    ]

    for orders_name, aov_name in revenue_combos:
        orders_pred = orders_map[orders_name]["predicted_orders"].to_numpy(dtype=float)
        aov_pred = aov_map[aov_name]["predicted_AOV"].to_numpy(dtype=float)
        revenue_pred = np.maximum(0.0, orders_pred * aov_pred)
        combo_name = f"{orders_name}__{aov_name}"
        frame = pd.DataFrame(
            {
                DATE_COL: valid_dates,
                "actual_Revenue": actual_window[TARGET_COL].to_numpy(dtype=float),
                "predicted_Revenue": revenue_pred,
                "orders_model": orders_name,
                "aov_model": aov_name,
                "revenue_model": combo_name,
                "scope": scope_name,
                "actual_orders": actual_window["orders_count"].to_numpy(dtype=float),
                "actual_AOV": actual_window["AOV"].to_numpy(dtype=float),
                "predicted_orders": orders_pred,
                "predicted_AOV": aov_pred,
                "calendar_any_promo": actual_window["calendar_any_promo"].to_numpy(dtype=float),
            }
        )
        revenue_frames.append(frame)
        comparison_rows.append(
            {
                "target_group": "revenue",
                "model_name": combo_name,
                "scope": scope_name,
                **compute_revenue_metrics(
                    actual_window[TARGET_COL],
                    revenue_pred,
                    actual_window["calendar_any_promo"],
                    actual_window["orders_count"],
                    actual_window["AOV"],
                    predicted_orders=orders_pred,
                    predicted_aov=aov_pred,
                ),
            }
        )

    return comparison_rows, orders_frames, aov_frames, revenue_frames, importance_frames


def add_long_average_rows(comparison: pd.DataFrame) -> pd.DataFrame:
    long_scopes = [fold[0] for fold in LONG_FOLDS]
    metric_columns = [column for column in comparison.columns if column not in {"target_group", "model_name", "scope"}]
    long_avg = (
        comparison.loc[comparison["scope"].isin(long_scopes)]
        .groupby(["target_group", "model_name"], as_index=False)[metric_columns]
        .mean(numeric_only=True)
    )
    long_avg["scope"] = "long_avg"
    return pd.concat([comparison, long_avg[comparison.columns]], ignore_index=True)


def choose_best_model(comparison: pd.DataFrame, target_group: str) -> tuple[str, pd.DataFrame]:
    valid_2022 = comparison.loc[(comparison["target_group"] == target_group) & (comparison["scope"] == "validation_2022")].copy()
    long_avg = comparison.loc[(comparison["target_group"] == target_group) & (comparison["scope"] == "long_avg"), ["model_name", "rmse"]].rename(columns={"rmse": "long_avg_rmse"})
    score = valid_2022.merge(long_avg, on="model_name", how="left")
    score["selection_score"] = 0.6 * score["rmse"] + 0.4 * score["long_avg_rmse"].fillna(score["rmse"])
    sort_cols = ["selection_score", "rmse"]
    if "top10_rmse" in score.columns:
        sort_cols.append("top10_rmse")
    score = score.sort_values(sort_cols).reset_index(drop=True)
    return str(score.iloc[0]["model_name"]), score


def build_current_best_validation_2022(sample_submission: pd.DataFrame) -> pd.DataFrame:
    current = stock_scale.build_current_best_validation_2022()
    current[DATE_COL] = pd.to_datetime(current[DATE_COL], errors="coerce").dt.normalize()
    return current.rename(columns={"actual_Revenue": "actual_Revenue", "base_pred": "current_best_pred"})[[DATE_COL, "actual_Revenue", "current_best_pred", "spike_prob"]]


def evaluate_corrections_2022(
    daily_table: pd.DataFrame,
    best_orders_2022: pd.DataFrame,
    best_aov_2022: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    current_best = build_current_best_validation_2022(load_sample_submission())
    actual = daily_table.loc[(daily_table[DATE_COL] >= VALIDATION_2022[2]) & (daily_table[DATE_COL] <= VALIDATION_2022[3]), [DATE_COL, TARGET_COL, "orders_count", "AOV", "calendar_any_promo"]].copy()
    baseline_orders = funnel.build_series_reference_frame(actual[DATE_COL], daily_table[[DATE_COL, "orders_count"]], "orders_count", "orders")
    baseline_aov = funnel.build_series_reference_frame(actual[DATE_COL], daily_table[[DATE_COL, "AOV"]], "AOV", "aov")

    frame = (
        actual.merge(current_best[[DATE_COL, "current_best_pred"]], on=DATE_COL, how="left", validate="one_to_one")
        .merge(best_orders_2022[[DATE_COL, "predicted_orders"]], on=DATE_COL, how="left", validate="one_to_one")
        .merge(best_aov_2022[[DATE_COL, "predicted_AOV"]], on=DATE_COL, how="left", validate="one_to_one")
        .merge(baseline_orders[[DATE_COL, "orders_same_day_recent_mean"]], on=DATE_COL, how="left", validate="one_to_one")
        .merge(baseline_aov[[DATE_COL, "aov_same_day_recent_mean"]], on=DATE_COL, how="left", validate="one_to_one")
    )
    frame["orders_ratio_raw"] = safe_divide(frame["predicted_orders"], frame["orders_same_day_recent_mean"].replace(0.0, np.nan), fill_value=1.0)
    frame["aov_ratio_raw"] = safe_divide(frame["predicted_AOV"], frame["aov_same_day_recent_mean"].replace(0.0, np.nan), fill_value=1.0)

    rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []

    orders_clips = {
        "orders_ratio_corrected_9505": (0.95, 1.05),
        "orders_ratio_corrected_9307": (0.93, 1.07),
        "orders_ratio_corrected_9010": (0.90, 1.10),
    }
    aov_clips = {
        "aov_ratio_corrected_9505": (0.95, 1.05),
        "aov_ratio_corrected_9307": (0.93, 1.07),
    }
    combined_clips = {
        "orders_aov_corrected_soft": ((0.95, 1.05), (0.95, 1.05)),
        "orders_aov_corrected_medium": ((0.93, 1.07), (0.95, 1.05)),
        "orders_aov_corrected_strong": ((0.90, 1.10), (0.93, 1.07)),
    }

    def append_eval(model_name: str, pred: np.ndarray) -> None:
        rows.append(
            {
                "target_group": "correction",
                "model_name": model_name,
                "scope": "validation_2022",
                **compute_revenue_metrics(
                    frame[TARGET_COL],
                    pred,
                    frame["calendar_any_promo"],
                    frame["orders_count"],
                    frame["AOV"],
                ),
            }
        )
        predictions.append(pd.DataFrame({DATE_COL: frame[DATE_COL], "actual_Revenue": frame[TARGET_COL], "predicted_Revenue": pred, "revenue_model": model_name, "scope": "validation_2022"}))

    for model_name, (lo, hi) in orders_clips.items():
        ratio = clip_series(frame["orders_ratio_raw"].fillna(1.0).to_numpy(dtype=float), lo, hi)
        pred = np.maximum(0.0, frame["current_best_pred"].to_numpy(dtype=float) * ratio)
        append_eval(model_name, pred)

    for model_name, (lo, hi) in aov_clips.items():
        ratio = clip_series(frame["aov_ratio_raw"].fillna(1.0).to_numpy(dtype=float), lo, hi)
        pred = np.maximum(0.0, frame["current_best_pred"].to_numpy(dtype=float) * ratio)
        append_eval(model_name, pred)

    for model_name, (orders_clip, aov_clip) in combined_clips.items():
        orders_ratio = clip_series(frame["orders_ratio_raw"].fillna(1.0).to_numpy(dtype=float), orders_clip[0], orders_clip[1])
        aov_ratio = clip_series(frame["aov_ratio_raw"].fillna(1.0).to_numpy(dtype=float), aov_clip[0], aov_clip[1])
        pred = np.maximum(0.0, frame["current_best_pred"].to_numpy(dtype=float) * orders_ratio * aov_ratio)
        append_eval(model_name, pred)

    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


def build_future_baselines(daily_table: pd.DataFrame, future_dates: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    orders_baseline = funnel.build_series_reference_frame(future_dates, daily_table[[DATE_COL, "orders_count"]], "orders_count", "orders")
    aov_baseline = funnel.build_series_reference_frame(future_dates, daily_table[[DATE_COL, "AOV"]], "AOV", "aov")
    return orders_baseline, aov_baseline


def apply_correction(
    current_best_revenue: np.ndarray,
    ratio_raw: np.ndarray,
    clip_lo: float,
    clip_hi: float,
) -> np.ndarray:
    clipped = clip_series(np.where(np.isfinite(ratio_raw), ratio_raw, 1.0), clip_lo, clip_hi)
    return np.maximum(0.0, current_best_revenue * clipped)


def main() -> None:
    logger = setup_logging()
    reporter = Reporter(logger)

    sample_submission = load_sample_submission()
    sales, daily_table, web_daily, promo_daily, inventory_snapshots = build_daily_orders_aov_table(logger)
    full_table = add_orders_aov_features(daily_table)
    min_date = full_table[DATE_COL].min()

    current_best_future = pd.read_csv(CURRENT_BEST_SUBMISSION_PATH, parse_dates=[DATE_COL], low_memory=False)
    current_best_future[DATE_COL] = pd.to_datetime(current_best_future[DATE_COL], errors="coerce").dt.normalize()
    current_best_future = current_best_future[[DATE_COL, TARGET_COL]].sort_values(DATE_COL).reset_index(drop=True)
    if not current_best_future[DATE_COL].equals(sample_submission[DATE_COL]):
        raise ValueError("Current best submission is not aligned with sample_submission")

    comparison_rows: list[dict[str, Any]] = []
    orders_frames: list[pd.DataFrame] = []
    aov_frames: list[pd.DataFrame] = []
    revenue_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []

    for scope_name, train_end, valid_start, valid_end in ALL_SCOPES:
        reporter.emit(f"Evaluating scope {scope_name}: train <= {train_end.date()}, validate {valid_start.date()} -> {valid_end.date()}")
        scope_comparison, scope_orders, scope_aov, scope_revenue, scope_importance = evaluate_scope(
            scope_name,
            train_end,
            valid_start,
            valid_end,
            full_table,
            web_daily,
            promo_daily,
            inventory_snapshots,
            min_date,
        )
        comparison_rows.extend(scope_comparison)
        orders_frames.extend(scope_orders)
        aov_frames.extend(scope_aov)
        revenue_frames.extend(scope_revenue)
        importance_frames.extend(scope_importance)

    comparison = add_long_average_rows(pd.DataFrame(comparison_rows))
    orders_validation = pd.concat(orders_frames, ignore_index=True)
    aov_validation = pd.concat(aov_frames, ignore_index=True)
    revenue_validation = pd.concat(revenue_frames, ignore_index=True)
    feature_importance = pd.concat(importance_frames, ignore_index=True).reset_index(drop=True)

    best_orders_model, orders_score = choose_best_model(comparison, "orders")
    best_aov_model, aov_score = choose_best_model(comparison, "aov")
    revenue_only = comparison.loc[comparison["target_group"] == "revenue"].copy()
    best_revenue_model, revenue_score = choose_best_model(revenue_only, "revenue")

    best_orders_2022 = orders_validation.loc[(orders_validation["scope"] == "validation_2022") & (orders_validation["orders_model"] == best_orders_model), [DATE_COL, "predicted_orders"]].copy()
    best_aov_2022 = aov_validation.loc[(aov_validation["scope"] == "validation_2022") & (aov_validation["aov_model"] == best_aov_model), [DATE_COL, "predicted_AOV"]].copy()
    correction_comparison, correction_predictions = evaluate_corrections_2022(full_table, best_orders_2022, best_aov_2022)

    comparison = pd.concat([comparison, correction_comparison], ignore_index=True)
    revenue_validation = pd.concat([revenue_validation, correction_predictions], ignore_index=True)

    best_correction = (
        correction_comparison.sort_values(["rmse", "top10_rmse", "mae"]).iloc[0]
        if not correction_comparison.empty
        else pd.Series(dtype=float)
    )

    comparison.to_csv(MODEL_COMPARISON_PATH, index=False)
    orders_validation.to_csv(ORDERS_VALIDATION_PATH, index=False)
    aov_validation.to_csv(AOV_VALIDATION_PATH, index=False)
    revenue_validation.to_csv(REVENUE_VALIDATION_PATH, index=False)
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    # Train final selected orders and AOV models.
    future_promo = union_model.load_future_promo_features(sample_submission, logger)
    future_traffic_all = union_model.load_or_build_future_traffic_scenarios(sample_submission, future_promo, logger)
    future_stock = union_model.build_stock_context_future(sample_submission[DATE_COL], stock_scale.load_inventory_snapshot_features())

    traffic_by_scenario: dict[str, pd.DataFrame] = {}
    for scenario_name in ["seasonal", "conservative", "high_demand"]:
        frame = future_traffic_all.loc[future_traffic_all["scenario"].astype(str) == scenario_name].copy()
        renamed = frame.rename(columns={"sessions_sum_feat": "sessions_sum", "unique_visitors_sum_feat": "unique_visitors_sum", "page_views_sum_feat": "page_views_sum"})
        traffic_by_scenario[scenario_name] = renamed

    full_history = full_table.copy()
    future_dates = sample_submission[DATE_COL].copy()

    # Orders future predictions
    orders_future_by_scenario: dict[str, pd.DataFrame] = {}
    orders_importance_future = pd.DataFrame()
    seasonal_ref_years = [(2022, 0.5), (2021, 0.3), (2020, 0.2)]
    if best_orders_model == "orders_a_direct_seasonal":
        features = select_columns(full_table, ORDERS_MODEL_A_FEATURES)
        X, y, med, subset = prepare_training_matrix(full_table, features, "orders_count", full_history[DATE_COL].max())
        model = train_lightgbm_regression(X, y)
        orders_importance_future = extract_feature_importance(model, features, best_orders_model, "orders")
        for scenario_name, raw in traffic_by_scenario.items():
            traffic_direct_context = raw.merge(
                raw[
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
            context = build_orders_direct_context(full_history, future_dates, traffic_direct_context, future_promo, future_stock, min_date)
            pred = predict_lightgbm(model, context, features, med)
            orders_future_by_scenario[scenario_name] = pd.DataFrame({DATE_COL: future_dates, "predicted_orders": pred})
    else:
        features = select_columns(full_table, ORDERS_MODEL_B_FEATURES)
        X, y, med, subset = prepare_training_matrix(full_table, features, "orders_count", full_history[DATE_COL].max())
        sample_weight = None
        if best_orders_model == "orders_c_weighted_recent":
            sample_weight = build_recency_weights(subset[DATE_COL])
        elif best_orders_model == "orders_d_spike_weighted":
            sample_weight = build_orders_spike_weights(subset["orders_count"])
        model = train_lightgbm_regression(X, y, sample_weight=sample_weight)
        orders_importance_future = extract_feature_importance(model, features, best_orders_model, "orders")
        for scenario_name, raw in traffic_by_scenario.items():
            context = funnel.build_recursive_static_context(future_dates, raw, future_promo, future_stock, min_date)
            pred_frame = funnel.recursive_predict_orders(model, med, features, full_history, context)
            orders_future_by_scenario[scenario_name] = pred_frame[[DATE_COL, "predicted_orders"]].copy()

    # AOV future predictions
    if best_aov_model == "aov_a_direct":
        features = select_columns(full_table, AOV_MODEL_A_FEATURES)
        X, y, med, subset = prepare_training_matrix(full_table, features, "AOV", full_history[DATE_COL].max())
        model = train_lightgbm_regression(X, y)
        aov_importance_future = extract_feature_importance(model, features, best_aov_model, "aov")
        context = build_aov_direct_context(full_history, future_dates, future_promo, future_stock, min_date)
        pred_frame = funnel.recursive_predict_aov(model, med, features, full_history, context)
        future_aov = pred_frame[[DATE_COL, "predicted_AOV"]].copy()
    elif best_aov_model in {"aov_b_ratio_p5p95", "aov_b_ratio_p10p90"}:
        features = select_columns(full_table, AOV_MODEL_B_FEATURES)
        ratio_train = full_table.copy()
        ratio_train["aov_ratio_target"] = safe_divide(ratio_train["AOV"], ratio_train["aov_lag_365"].replace(0.0, np.nan), fill_value=np.nan)
        X, y, med, subset = prepare_training_matrix(ratio_train, features, "aov_ratio_target", full_history[DATE_COL].max())
        model = train_lightgbm_regression(X, y)
        aov_importance_future = extract_feature_importance(model, features, "aov_b_ratio", "aov")
        context = build_aov_direct_context(full_history, future_dates, future_promo, future_stock, min_date)
        ratio_pred = predict_lightgbm(model, context, features, med)
        low_q, high_q = (0.05, 0.95) if best_aov_model.endswith("p5p95") else (0.10, 0.90)
        ratio_values = pd.to_numeric(subset["aov_ratio_target"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        ratio_pred = clip_series(ratio_pred, float(ratio_values.quantile(low_q)), float(ratio_values.quantile(high_q)))
        aov_baseline = pd.to_numeric(context["aov_lag_365"], errors="coerce").fillna(pd.to_numeric(context["aov_same_day_recent_mean"], errors="coerce")).fillna(float(full_history["AOV"].median())).to_numpy(dtype=float)
        future_aov = pd.DataFrame({DATE_COL: future_dates, "predicted_AOV": np.maximum(0.0, ratio_pred * aov_baseline)})
    elif best_aov_model == "aov_d_promo_adjusted":
        features = select_columns(full_table, AOV_MODEL_D_FEATURES)
        X, y, med, subset = prepare_training_matrix(full_table, features, "AOV", full_history[DATE_COL].max())
        model = train_lightgbm_regression(X, y, sample_weight=build_aov_promo_weights(subset))
        aov_importance_future = extract_feature_importance(model, features, best_aov_model, "aov")
        context = build_aov_direct_context(full_history, future_dates, future_promo, future_stock, min_date)
        pred_frame = funnel.recursive_predict_aov(model, med, features, full_history, context)
        future_aov = pred_frame[[DATE_COL, "predicted_AOV"]].copy()
    else:
        aov_importance_future = pd.DataFrame()
        future_aov = funnel.baseline_aov_predictions(full_history, future_dates, future_promo)[[DATE_COL, "predicted_AOV"]].copy()

    future_orders_baseline, future_aov_baseline = build_future_baselines(full_table, future_dates)

    # Standalone funnel scenario submissions
    funnel_submissions: dict[str, pd.DataFrame] = {}
    for scenario_name, orders_pred_df in orders_future_by_scenario.items():
        revenue = np.maximum(0.0, orders_pred_df["predicted_orders"].to_numpy(dtype=float) * future_aov["predicted_AOV"].to_numpy(dtype=float))
        funnel_submissions[scenario_name] = build_submission(future_dates, revenue, ratio=RATIO)

    save_submission_no_overwrite(SUBMISSION_FUNNEL_PATH, funnel_submissions["seasonal"], sample_submission)
    save_submission_no_overwrite(SUBMISSION_FUNNEL_CONSERVATIVE_PATH, funnel_submissions["conservative"], sample_submission)
    save_submission_no_overwrite(SUBMISSION_FUNNEL_AGGRESSIVE_PATH, funnel_submissions["high_demand"], sample_submission)

    # Current-best corrections
    future_orders_ratio_raw = safe_divide(
        orders_future_by_scenario["seasonal"]["predicted_orders"].to_numpy(dtype=float),
        pd.to_numeric(future_orders_baseline["orders_same_day_recent_mean"], errors="coerce").replace(0.0, np.nan).fillna(pd.to_numeric(future_orders_baseline["orders_same_month_recent_mean"], errors="coerce")).fillna(float(full_history["orders_count"].median())).to_numpy(dtype=float),
        fill_value=1.0,
    )
    future_aov_ratio_raw = safe_divide(
        future_aov["predicted_AOV"].to_numpy(dtype=float),
        pd.to_numeric(future_aov_baseline["aov_same_day_recent_mean"], errors="coerce").replace(0.0, np.nan).fillna(pd.to_numeric(future_aov_baseline["aov_same_month_recent_mean"], errors="coerce")).fillna(float(full_history["AOV"].median())).to_numpy(dtype=float),
        fill_value=1.0,
    )
    current_best_future_revenue = current_best_future[TARGET_COL].to_numpy(dtype=float)

    orders_clip_defs = {"9505": (0.95, 1.05), "9307": (0.93, 1.07), "9010": (0.90, 1.10)}
    for key, (lo, hi) in orders_clip_defs.items():
        corrected = apply_correction(current_best_future_revenue, future_orders_ratio_raw, lo, hi)
        save_submission_no_overwrite(ORDERS_CORRECTION_OUTPUTS[key], build_submission(future_dates, corrected, ratio=RATIO), sample_submission)

    aov_clip_defs = {"9505": (0.95, 1.05), "9307": (0.93, 1.07)}
    for key, (lo, hi) in aov_clip_defs.items():
        corrected = apply_correction(current_best_future_revenue, future_aov_ratio_raw, lo, hi)
        save_submission_no_overwrite(AOV_CORRECTION_OUTPUTS[key], build_submission(future_dates, corrected, ratio=RATIO), sample_submission)

    combined_defs = {
        "soft": ((0.95, 1.05), (0.95, 1.05)),
        "medium": ((0.93, 1.07), (0.95, 1.05)),
        "strong": ((0.90, 1.10), (0.93, 1.07)),
    }
    for key, (orders_clip, aov_clip) in combined_defs.items():
        corrected = apply_correction(current_best_future_revenue, future_orders_ratio_raw, orders_clip[0], orders_clip[1])
        corrected = apply_correction(corrected, future_aov_ratio_raw, aov_clip[0], aov_clip[1])
        save_submission_no_overwrite(COMBINED_CORRECTION_OUTPUTS[key], build_submission(future_dates, corrected, ratio=RATIO), sample_submission)

    # Current-best + best funnel blends
    best_funnel_revenue = funnel_submissions["seasonal"][TARGET_COL].to_numpy(dtype=float)
    for weight, path in BLEND_OUTPUTS.items():
        revenue = np.maximum(0.0, (1.0 - weight) * current_best_future_revenue + weight * best_funnel_revenue)
        save_submission_no_overwrite(path, build_submission(future_dates, revenue, ratio=RATIO), sample_submission)

    if ANTI_OVERFIT_BEST_PATH.exists():
        anti = pd.read_csv(ANTI_OVERFIT_BEST_PATH, parse_dates=[DATE_COL], low_memory=False)
        anti[DATE_COL] = pd.to_datetime(anti[DATE_COL], errors="coerce").dt.normalize()
        anti = anti[[DATE_COL, TARGET_COL]].sort_values(DATE_COL).reset_index(drop=True)
        if anti[DATE_COL].equals(sample_submission[DATE_COL]):
            anti_revenue = anti[TARGET_COL].to_numpy(dtype=float)
            for weight, path in STOCK_BLEND_OUTPUTS.items():
                revenue = np.maximum(0.0, (1.0 - weight) * anti_revenue + weight * best_funnel_revenue)
                save_submission_no_overwrite(path, build_submission(future_dates, revenue, ratio=RATIO), sample_submission)

    # Final summaries
    top_orders_features = (
        feature_importance.loc[feature_importance["model_name"] == best_orders_model]
        .groupby("feature", as_index=False)["importance_gain"]
        .sum()
        .sort_values("importance_gain", ascending=False)
        .head(20)
    )
    top_aov_features = (
        feature_importance.loc[feature_importance["model_name"].str.startswith(best_aov_model.split("_p")[0], na=False)]
        .groupby("feature", as_index=False)["importance_gain"]
        .sum()
        .sort_values("importance_gain", ascending=False)
        .head(20)
    )

    best_revenue_row = revenue_score.iloc[0]
    current_best_2022 = build_current_best_validation_2022(sample_submission)
    current_best_rmse_2022 = rmse(current_best_2022["actual_Revenue"], current_best_2022["current_best_pred"])
    orders_correction_helped = bool(not correction_comparison.empty and correction_comparison.loc[correction_comparison["model_name"].str.startswith("orders_ratio_corrected"), "rmse"].min() < current_best_rmse_2022)
    aov_correction_helped = bool(not correction_comparison.empty and correction_comparison.loc[correction_comparison["model_name"].str.startswith("aov_ratio_corrected"), "rmse"].min() < current_best_rmse_2022)

    reporter.emit(f"Best Orders model and metrics: {best_orders_model}")
    reporter.emit_frame("orders_score_table", orders_score[["model_name", "rmse", "long_avg_rmse", "selection_score"]].head(10))
    reporter.emit("")
    reporter.emit(f"Best AOV model and metrics: {best_aov_model}")
    reporter.emit_frame("aov_score_table", aov_score[["model_name", "rmse", "long_avg_rmse", "selection_score"]].head(10))
    reporter.emit("")
    reporter.emit(f"Best Revenue reconstruction method: {best_revenue_model}")
    reporter.emit_frame("revenue_score_table", revenue_score[["model_name", "rmse", "long_avg_rmse", "selection_score", "top10_rmse"]].head(10))
    reporter.emit("")
    reporter.emit(f"Best correction method: {best_correction.get('model_name', 'n/a')}")
    if not correction_comparison.empty:
        reporter.emit_frame("correction_comparison", correction_comparison.sort_values(["rmse", "top10_rmse"])[["model_name", "rmse", "mae", "top10_rmse"]])
    reporter.emit("")
    reporter.emit(f"2022 analog RMSE: {float(best_revenue_row['rmse']):.2f}")
    reporter.emit(f"Long-horizon average RMSE: {float(best_revenue_row['long_avg_rmse']):.2f}")
    reporter.emit("")
    reporter.emit_frame("Top Orders features", top_orders_features)
    reporter.emit("")
    reporter.emit_frame("Top AOV features", top_aov_features)
    reporter.emit("")
    reporter.emit(f"Whether Orders correction helped: {orders_correction_helped}")
    reporter.emit(f"Whether AOV correction helped: {aov_correction_helped}")
    reporter.emit("")
    reporter.emit("Created submissions")
    created = [
        SUBMISSION_FUNNEL_PATH,
        SUBMISSION_FUNNEL_CONSERVATIVE_PATH,
        SUBMISSION_FUNNEL_AGGRESSIVE_PATH,
        *ORDERS_CORRECTION_OUTPUTS.values(),
        *AOV_CORRECTION_OUTPUTS.values(),
        *COMBINED_CORRECTION_OUTPUTS.values(),
        *BLEND_OUTPUTS.values(),
        *[path for path in STOCK_BLEND_OUTPUTS.values() if path.exists()],
    ]
    for path in created:
        if path.exists():
            reporter.emit(str(path))
    reporter.emit("")
    reporter.emit("Recommended upload order")
    preferred = [
        STOCK_BLEND_OUTPUTS[0.03],
        ORDERS_CORRECTION_OUTPUTS["9505"],
        BLEND_OUTPUTS[0.03],
        COMBINED_CORRECTION_OUTPUTS["soft"],
    ]
    for path in preferred:
        if path.exists():
            reporter.emit(str(path))
    reporter.emit("")
    reporter.emit("Leakage safety confirmation")
    reporter.emit("This pipeline uses only historical orders/order_items/sales/web_traffic/promotions/inventory/products, future promo-known features, future traffic scenarios, and seasonal references. No future actual Revenue/COGS or same-day future realized demand was used.")
    reporter.save()


if __name__ == "__main__":
    main()
