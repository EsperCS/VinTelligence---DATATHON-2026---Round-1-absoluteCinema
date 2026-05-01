from __future__ import annotations

import logging
import re
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import train_direct_seasonal_residual_model as dsr
import train_final_model as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "log"

ORDERS_PATH = DATA_DIR / "orders.csv"
ORDER_ITEMS_PATH = DATA_DIR / "order_items.csv"
PRODUCTS_PATH = DATA_DIR / "products.csv"
PROMOTIONS_PATH = DATA_DIR / "promotions.csv"
GEOGRAPHY_PATH = DATA_DIR / "geography.csv"
INVENTORY_PATH = DATA_DIR / "inventory.csv"
WEB_TRAFFIC_PATH = DATA_DIR / "web_traffic.csv"
SALES_PATH = DATA_DIR / "sales.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
FUTURE_PROMO_FEATURES_PATH = DATA_DIR / "future_promo_calendar_features.csv"
CURRENT_BEST_SUBMISSION_PATH = DATA_DIR / "submission_cogs_ratio_8900.csv"
CURRENT_BEST_VALIDATION_PATH = DATA_DIR / "final_micro_calibration_validation_predictions.csv"

SUBMISSION_CATEGORY_PATH = DATA_DIR / "submission_m5_category_bottomup.csv"
SUBMISSION_SEGMENT_PATH = DATA_DIR / "submission_m5_segment_bottomup.csv"
SUBMISSION_BLEND_PATH = DATA_DIR / "submission_m5_multilevel_blend.csv"
SUBMISSION_BLEND_8900_PATH = DATA_DIR / "submission_m5_multilevel_blend_cogs8900.csv"
SUBMISSION_BLEND_8950_PATH = DATA_DIR / "submission_m5_multilevel_blend_cogs8950.csv"
SUBMISSION_BLEND_9000_PATH = DATA_DIR / "submission_m5_multilevel_blend_cogs9000.csv"

VALIDATION_PREDICTIONS_PATH = DATA_DIR / "m5_multilevel_validation_predictions.csv"
MODEL_COMPARISON_PATH = DATA_DIR / "m5_multilevel_model_comparison.csv"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "m5_multilevel_feature_importance.csv"
REPORT_PATH = LOG_DIR / "m5_multilevel_report.txt"
LOG_FILE = LOG_DIR / "train_m5_style_multilevel_model.log"

DATE_COL = base.DATE_COL
TARGET_COL = base.TARGET_COL
COGS_COL = base.COGS_COL
RANDOM_STATE = base.RANDOM_STATE

FOLDS = [
    ("fold_1", pd.Timestamp("2019-06-30"), pd.Timestamp("2019-07-01"), pd.Timestamp("2020-12-31")),
    ("fold_2", pd.Timestamp("2020-06-30"), pd.Timestamp("2020-07-01"), pd.Timestamp("2021-12-31")),
    ("fold_3", pd.Timestamp("2021-06-30"), pd.Timestamp("2021-07-01"), pd.Timestamp("2022-12-31")),
]

PROMO_AGG_FEATURES = [
    "calendar_active_promo_count",
    "calendar_any_promo",
    "calendar_avg_discount_value",
    "calendar_max_discount_value",
    "calendar_stackable_promo_count",
    "calendar_has_stackable_promo",
    "calendar_has_category_specific_promo",
    "calendar_percentage_promo_count",
    "calendar_fixed_promo_count",
]
PROMO_PHASE_FEATURES = [
    "promotion_campaign_index",
    "promo_duration",
    "promo_progress_ratio",
    "promo_days_remaining",
] + dsr.CAMPAIGN_FLAG_COLUMNS
INVENTORY_FEATURES = dsr.INVENTORY_FEATURES
WEB_FEATURES = [
    "web_sessions_ref_365",
    "web_sessions_ref_730",
    "web_sessions_ref_1095",
    "web_page_views_ref_365",
    "web_page_views_ref_730",
    "web_page_views_ref_1095",
    "web_sessions_recent_mean",
]
SAFE_CALENDAR_FEATURES = [
    "day_of_week",
    "day_of_year",
    "week_of_year",
    "month",
    "quarter",
    "year",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "is_odd_year",
]
RECURSIVE_NUMERIC_FEATURES = SAFE_CALENDAR_FEATURES + PROMO_AGG_FEATURES + PROMO_PHASE_FEATURES + INVENTORY_FEATURES + WEB_FEATURES + [
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
    "lag365_to_roll365_ratio",
    "lag365_to_recent_mean_ratio",
]
DIRECT_NUMERIC_FEATURES = RECURSIVE_NUMERIC_FEATURES + [
    "baseline_revenue",
    "weighted_recent_same_day_revenue",
    "same_month_recent_mean",
    "same_day_of_year_recent_mean",
    "same_campaign_last_year_revenue",
    "lag_730",
    "lag_1095",
    "lag365_to_lag730_ratio",
]
STATIC_CATEGORICAL_FEATURES = ["series_id", "level", "category", "segment", "region", "campaign_base", "level_id"]


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
    logger = logging.getLogger("train_m5_style_multilevel_model")
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


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    metrics = base.evaluate_predictions(y_true, y_pred)
    actual = y_true.to_numpy(dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    error = actual - predicted
    top10_threshold = float(np.quantile(actual, 0.90))
    top10_mask = actual >= top10_threshold
    non_spike_mask = actual < top10_threshold
    metrics["top10_RMSE"] = float(np.sqrt(np.mean(error[top10_mask] ** 2))) if top10_mask.any() else np.nan
    metrics["top10_underprediction"] = int(np.sum(error[top10_mask] > 0)) if top10_mask.any() else 0
    metrics["non_spike_RMSE"] = float(np.sqrt(np.mean(error[non_spike_mask] ** 2))) if non_spike_mask.any() else np.nan
    return metrics


def compute_monthly_rmse(dates: pd.Series, actual: pd.Series, predicted: np.ndarray) -> pd.DataFrame:
    temp = pd.DataFrame({DATE_COL: pd.to_datetime(dates), "actual": actual, "predicted": predicted})
    temp["year_month"] = temp[DATE_COL].dt.to_period("M").astype(str)
    temp["sq_error"] = (temp["actual"] - temp["predicted"]) ** 2
    monthly = temp.groupby("year_month", as_index=False)["sq_error"].mean()
    monthly["RMSE"] = np.sqrt(monthly["sq_error"])
    return monthly[["year_month", "RMSE"]]


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


def load_inputs() -> dict[str, pd.DataFrame]:
    orders = pd.read_csv(ORDERS_PATH, low_memory=False, parse_dates=["order_date"])
    order_items = pd.read_csv(ORDER_ITEMS_PATH, low_memory=False)
    products = pd.read_csv(PRODUCTS_PATH, low_memory=False)
    promotions = dsr.load_promotions_with_campaign_index(PROMOTIONS_PATH)
    geography = pd.read_csv(GEOGRAPHY_PATH, low_memory=False)
    inventory = dsr.prepare_inventory_snapshots()
    sales = pd.read_csv(SALES_PATH, low_memory=False, parse_dates=[DATE_COL])
    sample = pd.read_csv(SAMPLE_SUBMISSION_PATH, low_memory=False, parse_dates=[DATE_COL])
    future_promo = pd.read_csv(FUTURE_PROMO_FEATURES_PATH, low_memory=False, parse_dates=[DATE_COL])
    web = pd.read_csv(WEB_TRAFFIC_PATH, low_memory=False, parse_dates=["date"])

    orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce").dt.normalize()
    sales[DATE_COL] = pd.to_datetime(sales[DATE_COL], errors="coerce").dt.normalize()
    sample[DATE_COL] = pd.to_datetime(sample[DATE_COL], errors="coerce").dt.normalize()
    future_promo[DATE_COL] = pd.to_datetime(future_promo[DATE_COL], errors="coerce").dt.normalize()
    web["date"] = pd.to_datetime(web["date"], errors="coerce").dt.normalize()

    return {
        "orders": orders,
        "order_items": order_items,
        "products": products,
        "promotions": promotions,
        "geography": geography,
        "inventory_snapshots": inventory,
        "sales": sales.sort_values(DATE_COL).reset_index(drop=True),
        "sample_submission": sample.sort_values(DATE_COL).reset_index(drop=True),
        "future_promo": future_promo.sort_values(DATE_COL).reset_index(drop=True),
        "web": web.sort_values("date").reset_index(drop=True),
    }


def load_current_best_validation() -> pd.DataFrame | None:
    if not CURRENT_BEST_VALIDATION_PATH.exists():
        return None
    frame = pd.read_csv(CURRENT_BEST_VALIDATION_PATH, parse_dates=[DATE_COL], low_memory=False)
    frame[DATE_COL] = pd.to_datetime(frame[DATE_COL], errors="coerce").dt.normalize()
    if "current_base_pred" not in frame.columns or "actual_Revenue" not in frame.columns:
        return None
    return frame[[DATE_COL, "actual_Revenue", "current_base_pred"]].copy()


def build_transaction_fact(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    orders = inputs["orders"].copy()
    order_items = inputs["order_items"].copy()
    products = inputs["products"].copy()
    geography = inputs["geography"].copy()
    promotions = inputs["promotions"].copy()
    sales = inputs["sales"].copy()

    products["cogs"] = pd.to_numeric(products["cogs"], errors="coerce").fillna(0.0)
    order_items["quantity"] = pd.to_numeric(order_items["quantity"], errors="coerce").fillna(0.0)
    order_items["unit_price"] = pd.to_numeric(order_items["unit_price"], errors="coerce").fillna(0.0)
    order_items["discount_amount"] = pd.to_numeric(order_items["discount_amount"], errors="coerce").fillna(0.0)

    promo_map = promotions[["promo_id", "promo_name_base"]].drop_duplicates().rename(columns={"promo_name_base": "campaign_base"})
    fact = (
        order_items.merge(
            orders[["order_id", "order_date", "zip"]],
            on="order_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            products[["product_id", "category", "segment", "cogs"]],
            on="product_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            geography[["zip", "region"]],
            on="zip",
            how="left",
            validate="many_to_one",
        )
        .merge(
            promo_map.rename(columns={"promo_id": "promo_id", "campaign_base": "campaign_base_primary"}),
            on="promo_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            promo_map.rename(columns={"promo_id": "promo_id_2", "campaign_base": "campaign_base_secondary"}),
            on="promo_id_2",
            how="left",
            validate="many_to_one",
        )
    )
    fact[DATE_COL] = pd.to_datetime(fact["order_date"], errors="coerce").dt.normalize()
    fact["campaign_base"] = fact["campaign_base_primary"].combine_first(fact["campaign_base_secondary"]).fillna("NO_PROMO")
    fact["category"] = fact["category"].fillna("UNKNOWN")
    fact["segment"] = fact["segment"].fillna("UNKNOWN")
    fact["region"] = fact["region"].fillna("UNKNOWN")
    fact["raw_revenue"] = (fact["quantity"] * fact["unit_price"] - fact["discount_amount"]).clip(lower=0.0)
    fact["raw_cogs"] = (fact["quantity"] * fact["cogs"]).clip(lower=0.0)
    fact["item_lines"] = 1.0

    daily_raw = fact.groupby(DATE_COL, as_index=False).agg(
        raw_revenue_sum=("raw_revenue", "sum"),
        raw_cogs_sum=("raw_cogs", "sum"),
    )
    daily_scalars = sales.merge(daily_raw, on=DATE_COL, how="left")
    daily_scalars["raw_revenue_sum"] = pd.to_numeric(daily_scalars["raw_revenue_sum"], errors="coerce").fillna(0.0)
    daily_scalars["raw_cogs_sum"] = pd.to_numeric(daily_scalars["raw_cogs_sum"], errors="coerce").fillna(0.0)
    daily_scalars["revenue_scale"] = np.where(
        daily_scalars["raw_revenue_sum"] > 1e-9,
        daily_scalars[TARGET_COL] / daily_scalars["raw_revenue_sum"],
        1.0,
    )
    daily_scalars["cogs_scale"] = np.where(
        daily_scalars["raw_cogs_sum"] > 1e-9,
        daily_scalars[COGS_COL] / daily_scalars["raw_cogs_sum"],
        1.0,
    )
    scale_map = daily_scalars.set_index(DATE_COL)[["revenue_scale", "cogs_scale"]]
    fact = fact.join(scale_map, on=DATE_COL, how="left")
    fact["revenue"] = fact["raw_revenue"] * fact["revenue_scale"].fillna(1.0)
    fact["cogs_value"] = fact["raw_cogs"] * fact["cogs_scale"].fillna(1.0)
    return fact.dropna(subset=[DATE_COL]).copy()


def build_level_aggregates(inputs: dict[str, pd.DataFrame], fact: pd.DataFrame) -> dict[str, pd.DataFrame]:
    sales = inputs["sales"].copy()
    sales["series_id"] = "TOTAL"
    sales["level"] = "TOTAL"
    sales["target_revenue"] = sales[TARGET_COL]
    sales["target_cogs"] = sales[COGS_COL]
    sales["target_quantity"] = 0.0
    sales["target_item_lines"] = 0.0
    sales["category"] = "ALL"
    sales["segment"] = "ALL"
    sales["region"] = "ALL"
    sales["campaign_base"] = "ALL"
    sales["level_id"] = 0
    total_df = sales[
        [DATE_COL, "series_id", "level", "target_revenue", "target_cogs", "target_quantity", "target_item_lines", "category", "segment", "region", "campaign_base", "level_id"]
    ].copy()

    category_df = (
        fact.groupby([DATE_COL, "category"], as_index=False)
        .agg(
            target_revenue=("revenue", "sum"),
            target_cogs=("cogs_value", "sum"),
            target_quantity=("quantity", "sum"),
            target_item_lines=("item_lines", "sum"),
        )
    )
    category_df["series_id"] = "CAT::" + category_df["category"].astype(str)
    category_df["level"] = "CATEGORY"
    category_df["segment"] = "ALL"
    category_df["region"] = "ALL"
    category_df["campaign_base"] = "ALL"
    category_df["level_id"] = 1

    segment_df = (
        fact.groupby([DATE_COL, "segment"], as_index=False)
        .agg(
            target_revenue=("revenue", "sum"),
            target_cogs=("cogs_value", "sum"),
            target_quantity=("quantity", "sum"),
            target_item_lines=("item_lines", "sum"),
        )
    )
    segment_df["series_id"] = "SEG::" + segment_df["segment"].astype(str)
    segment_df["level"] = "SEGMENT"
    segment_df["category"] = "ALL"
    segment_df["region"] = "ALL"
    segment_df["campaign_base"] = "ALL"
    segment_df["level_id"] = 2

    cat_promo_df = (
        fact[fact["campaign_base"].ne("NO_PROMO")]
        .groupby([DATE_COL, "category", "campaign_base"], as_index=False)
        .agg(
            target_revenue=("revenue", "sum"),
            target_cogs=("cogs_value", "sum"),
            target_quantity=("quantity", "sum"),
            target_item_lines=("item_lines", "sum"),
        )
    )
    cat_promo_df["series_id"] = "CATPROMO::" + cat_promo_df["category"].astype(str) + "::" + cat_promo_df["campaign_base"].astype(str)
    cat_promo_df["level"] = "CATEGORY_PROMO"
    cat_promo_df["segment"] = "ALL"
    cat_promo_df["region"] = "ALL"
    cat_promo_df["level_id"] = 3

    region_df = (
        fact.groupby([DATE_COL, "region"], as_index=False)
        .agg(
            target_revenue=("revenue", "sum"),
            target_cogs=("cogs_value", "sum"),
            target_quantity=("quantity", "sum"),
            target_item_lines=("item_lines", "sum"),
        )
    )
    region_df["series_id"] = "REG::" + region_df["region"].astype(str)
    region_df["level"] = "REGION"
    region_df["category"] = "ALL"
    region_df["segment"] = "ALL"
    region_df["campaign_base"] = "ALL"
    region_df["level_id"] = 4

    return {
        "total": total_df,
        "category": category_df[[DATE_COL, "series_id", "level", "target_revenue", "target_cogs", "target_quantity", "target_item_lines", "category", "segment", "region", "campaign_base", "level_id"]],
        "segment": segment_df[[DATE_COL, "series_id", "level", "target_revenue", "target_cogs", "target_quantity", "target_item_lines", "category", "segment", "region", "campaign_base", "level_id"]],
        "category_promo": cat_promo_df[[DATE_COL, "series_id", "level", "target_revenue", "target_cogs", "target_quantity", "target_item_lines", "category", "segment", "region", "campaign_base", "level_id"]],
        "region": region_df[[DATE_COL, "series_id", "level", "target_revenue", "target_cogs", "target_quantity", "target_item_lines", "category", "segment", "region", "campaign_base", "level_id"]],
    }


def densify_level_table(level_df: pd.DataFrame, full_dates: pd.Series) -> pd.DataFrame:
    series_meta = level_df[["series_id", "level", "category", "segment", "region", "campaign_base", "level_id"]].drop_duplicates().reset_index(drop=True)
    dates_df = pd.DataFrame({DATE_COL: pd.to_datetime(full_dates).sort_values().unique()})
    series_meta["key"] = 1
    dates_df["key"] = 1
    grid = series_meta.merge(dates_df, on="key", how="outer").drop(columns="key")
    merged = grid.merge(
        level_df,
        on=[DATE_COL, "series_id", "level", "category", "segment", "region", "campaign_base", "level_id"],
        how="left",
    )
    for column in ["target_revenue", "target_cogs", "target_quantity", "target_item_lines"]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
    return merged.sort_values(["series_id", DATE_COL]).reset_index(drop=True)


def build_web_daily(web: pd.DataFrame) -> pd.DataFrame:
    df = web.copy()
    df["sessions"] = pd.to_numeric(df["sessions"], errors="coerce").fillna(0.0)
    df["unique_visitors"] = pd.to_numeric(df["unique_visitors"], errors="coerce").fillna(0.0)
    df["page_views"] = pd.to_numeric(df["page_views"], errors="coerce").fillna(0.0)
    df["bounce_rate"] = pd.to_numeric(df["bounce_rate"], errors="coerce").fillna(0.0)
    df["avg_session_duration_sec"] = pd.to_numeric(df["avg_session_duration_sec"], errors="coerce").fillna(0.0)
    df["weighted_duration"] = df["avg_session_duration_sec"] * df["sessions"]
    daily = df.groupby("date", as_index=False).agg(
        sessions=("sessions", "sum"),
        unique_visitors=("unique_visitors", "sum"),
        page_views=("page_views", "sum"),
        weighted_duration=("weighted_duration", "sum"),
    )
    daily["avg_session_duration_sec"] = np.where(
        daily["sessions"] > 1e-9,
        daily["weighted_duration"] / daily["sessions"],
        0.0,
    )
    return daily.rename(columns={"date": DATE_COL}).sort_values(DATE_COL).reset_index(drop=True)


def build_historical_date_contexts(inputs: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sales = inputs["sales"]
    promotions = inputs["promotions"]
    inventory_snapshots = inputs["inventory_snapshots"]
    web_daily = build_web_daily(inputs["web"])

    calendar = base.build_calendar_features(sales[DATE_COL], sales[DATE_COL].min())
    calendar["is_odd_year"] = (calendar["year"] % 2 == 1).astype(int)
    promo_agg = base.build_promotion_calendar(sales[DATE_COL], PROMOTIONS_PATH, logging.getLogger("m5_promo_hist"))
    promo_phase = dsr.build_daily_promo_context(sales[DATE_COL], promotions)[
        [DATE_COL, "promotion_campaign_index", "promo_duration", "promo_progress_ratio", "promo_days_remaining"]
        + dsr.CAMPAIGN_FLAG_COLUMNS
    ]
    inventory = dsr.build_inventory_context(sales[DATE_COL], inventory_snapshots, snapshot_cutoff=None)

    static = (
        calendar.merge(promo_agg, on=DATE_COL, how="left", validate="one_to_one")
        .merge(promo_phase, on=DATE_COL, how="left", validate="one_to_one")
        .merge(inventory, on=DATE_COL, how="left", validate="one_to_one")
        .fillna(0.0)
    )
    static = static.loc[:, ~static.columns.duplicated()].copy()

    web_lookup = web_daily.set_index(DATE_COL).sort_index()
    return static, web_lookup


def build_future_date_contexts(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    sample = inputs["sample_submission"]
    sales = inputs["sales"]
    promotions = inputs["promotions"]
    synthetic_promotions = dsr.load_or_build_synthetic_promotions(promotions)
    inventory_snapshots = inputs["inventory_snapshots"]

    calendar = base.build_calendar_features(sample[DATE_COL], sales[DATE_COL].min())
    calendar["is_odd_year"] = (calendar["year"] % 2 == 1).astype(int)

    future_promo = inputs["future_promo"].rename(
        columns={
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
            "future_promo_avg_progress_ratio": "promo_progress_ratio",
            "future_promo_avg_days_remaining": "promo_days_remaining",
            "future_promotion_campaign_index": "promotion_campaign_index",
        }
    )
    keep_cols = [DATE_COL] + PROMO_AGG_FEATURES + ["promotion_campaign_index", "promo_duration", "promo_progress_ratio", "promo_days_remaining"]
    future_promo = future_promo[keep_cols].copy()
    future_campaign = dsr.build_daily_promo_context(sample[DATE_COL], synthetic_promotions)[[DATE_COL] + dsr.CAMPAIGN_FLAG_COLUMNS]
    inventory = dsr.build_inventory_context(sample[DATE_COL], inventory_snapshots, snapshot_cutoff=inputs["sales"][DATE_COL].max())

    static = (
        calendar.merge(future_promo, on=DATE_COL, how="left", validate="one_to_one")
        .merge(future_campaign, on=DATE_COL, how="left", validate="one_to_one")
        .merge(inventory, on=DATE_COL, how="left", validate="one_to_one")
        .fillna(0.0)
    )
    static = static.loc[:, ~static.columns.duplicated()].copy()
    return static


def build_group_revenue_lookup(level_table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    lookup: dict[str, pd.DataFrame] = {}
    for series_id, group in level_table.groupby("series_id"):
        series = group[[DATE_COL, "target_revenue"]].sort_values(DATE_COL).copy()
        series["year"] = series[DATE_COL].dt.year.astype(int)
        series["month"] = series[DATE_COL].dt.month.astype(int)
        series["day_of_year"] = series[DATE_COL].dt.dayofyear.astype(int)
        lookup[str(series_id)] = series
    return lookup


def compute_web_reference_features(
    target_date: pd.Timestamp,
    reference_end: pd.Timestamp,
    web_lookup: pd.DataFrame,
) -> dict[str, float]:
    def get_web_value(column: str, years_back: int) -> float:
        ref_date = dsr.safe_replace_year(target_date, target_date.year - years_back)
        if ref_date > reference_end or ref_date not in web_lookup.index:
            return np.nan
        return float(pd.to_numeric(web_lookup.loc[ref_date, column], errors="coerce"))

    sessions_365 = get_web_value("sessions", 1)
    sessions_730 = get_web_value("sessions", 2)
    sessions_1095 = get_web_value("sessions", 3)
    page_views_365 = get_web_value("page_views", 1)
    page_views_730 = get_web_value("page_views", 2)
    page_views_1095 = get_web_value("page_views", 3)

    recent_vals = []
    for years_back in (1, 2, 3):
        ref_date = dsr.safe_replace_year(target_date, target_date.year - years_back)
        if ref_date <= reference_end and ref_date in web_lookup.index:
            recent_vals.append(float(pd.to_numeric(web_lookup.loc[ref_date, "sessions"], errors="coerce")))
    return {
        "web_sessions_ref_365": sessions_365,
        "web_sessions_ref_730": sessions_730,
        "web_sessions_ref_1095": sessions_1095,
        "web_page_views_ref_365": page_views_365,
        "web_page_views_ref_730": page_views_730,
        "web_page_views_ref_1095": page_views_1095,
        "web_sessions_recent_mean": float(np.mean(recent_vals)) if recent_vals else np.nan,
    }


def add_recursive_lag_features(level_table: pd.DataFrame) -> pd.DataFrame:
    output = level_table.sort_values(["series_id", DATE_COL]).reset_index(drop=True).copy()
    group = output.groupby("series_id", sort=False)["target_revenue"]

    lag_map = {7: "lag_7", 14: "lag_14", 30: "lag_30", 90: "lag_90", 180: "lag_180", 365: "lag_365"}
    for lag, feature in lag_map.items():
        output[feature] = group.shift(lag)

    shifted = group.shift(1)
    for window in [7, 30, 90, 365]:
        output[f"rolling_mean_{window}"] = shifted.rolling(window=window, min_periods=window).mean().reset_index(level=0, drop=True)

    output["lag365_to_roll365_ratio"] = output["lag_365"] / output["rolling_mean_365"].replace(0, np.nan)
    output["lag365_to_recent_mean_ratio"] = output["lag_365"] / output["rolling_mean_90"].replace(0, np.nan)
    return output


def append_safe_web_features(table: pd.DataFrame, web_lookup: pd.DataFrame, use_date_minus_one: bool) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in table.itertuples(index=False):
        target_date = getattr(row, DATE_COL)
        reference_end = target_date - pd.Timedelta(days=1) if use_date_minus_one else getattr(row, "reference_end")
        rows.append(compute_web_reference_features(target_date, reference_end, web_lookup))
    return pd.concat([table.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def prepare_recursive_training_table(level_table: pd.DataFrame, static_context: pd.DataFrame, web_lookup: pd.DataFrame) -> pd.DataFrame:
    table = add_recursive_lag_features(level_table)
    table = table.merge(static_context, on=DATE_COL, how="left", validate="many_to_one")
    table = append_safe_web_features(table, web_lookup, use_date_minus_one=True)
    return table


def build_series_reference_table(
    level_table: pd.DataFrame,
    static_context: pd.DataFrame,
    revenue_lookup: dict[str, pd.DataFrame],
    web_lookup: pd.DataFrame,
    reference_end_map: pd.Series,
) -> pd.DataFrame:
    static_idx = static_context.set_index(DATE_COL).sort_index()
    rows: list[dict[str, Any]] = []

    for row in level_table.itertuples(index=False):
        target_date = getattr(row, DATE_COL)
        reference_end = pd.Timestamp(reference_end_map.loc[target_date] if target_date in reference_end_map.index else reference_end_map.iloc[0])
        series_id = str(getattr(row, "series_id"))
        series_frame = revenue_lookup[series_id]
        static_row = static_idx.loc[target_date].to_dict()

        def get_ref(years_back: int) -> float:
            ref_date = dsr.safe_replace_year(target_date, target_date.year - years_back)
            if ref_date > reference_end:
                return np.nan
            matched = series_frame.loc[series_frame[DATE_COL] == ref_date, "target_revenue"]
            return float(matched.iloc[0]) if not matched.empty else np.nan

        lag_365 = get_ref(1)
        lag_730 = get_ref(2)
        lag_1095 = get_ref(3)
        weighted_recent = dsr.robust_weighted_average([(0.5, lag_365), (0.3, lag_730), (0.2, lag_1095)])

        candidate_years = {target_date.year - 1, target_date.year - 2, target_date.year - 3}
        month_values = series_frame.loc[
            (series_frame[DATE_COL] <= reference_end)
            & (series_frame["month"] == target_date.month)
            & (series_frame["year"].isin(candidate_years)),
            "target_revenue",
        ]
        same_month_recent_mean = float(month_values.mean()) if not month_values.empty else np.nan

        day_diff = np.abs(series_frame["day_of_year"] - target_date.dayofyear)
        wrap_diff = np.minimum(day_diff, 366 - day_diff)
        doy_values = series_frame.loc[
            (series_frame[DATE_COL] <= reference_end)
            & (series_frame["year"].isin(candidate_years))
            & (wrap_diff <= 3),
            "target_revenue",
        ]
        same_day_recent_mean = float(doy_values.mean()) if not doy_values.empty else np.nan

        same_campaign_last_year = np.nan
        if any(pd.to_numeric(static_row.get(flag, 0), errors="coerce") > 0 for flag in dsr.CAMPAIGN_FLAG_COLUMNS):
            ref_date = dsr.safe_replace_year(target_date, target_date.year - 1)
            if ref_date <= reference_end:
                matched = series_frame.loc[series_frame[DATE_COL] == ref_date, "target_revenue"]
                if not matched.empty:
                    same_campaign_last_year = float(matched.iloc[0])

        baseline_revenue = dsr.robust_weighted_average(
            [
                (0.50, weighted_recent),
                (0.20, same_day_recent_mean),
                (0.15, same_month_recent_mean),
                (0.10, lag_365),
                (0.05, same_campaign_last_year),
            ]
        )

        web_features = compute_web_reference_features(target_date, reference_end, web_lookup)

        rows.append(
            {
                DATE_COL: target_date,
                "series_id": series_id,
                "lag_7": np.nan,
                "lag_14": np.nan,
                "lag_30": np.nan,
                "lag_90": np.nan,
                "lag_180": np.nan,
                "lag_365": lag_365,
                "lag_730": lag_730,
                "lag_1095": lag_1095,
                "rolling_mean_7": np.nan,
                "rolling_mean_30": np.nan,
                "rolling_mean_90": np.nan,
                "rolling_mean_365": np.nan,
                "lag365_to_roll365_ratio": safe_divide(lag_365, same_day_recent_mean),
                "lag365_to_recent_mean_ratio": safe_divide(lag_365, same_day_recent_mean),
                "lag365_to_lag730_ratio": safe_divide(lag_365, lag_730),
                "weighted_recent_same_day_revenue": weighted_recent,
                "same_month_recent_mean": same_month_recent_mean,
                "same_day_of_year_recent_mean": same_day_recent_mean,
                "same_campaign_last_year_revenue": same_campaign_last_year,
                "baseline_revenue": baseline_revenue,
                "reference_end": reference_end,
                **web_features,
            }
        )

    direct_features = pd.DataFrame(rows)
    direct_features = direct_features.merge(level_table, on=[DATE_COL, "series_id"], how="left", validate="one_to_one")
    direct_features = direct_features.merge(static_context, on=DATE_COL, how="left", validate="many_to_one")
    return direct_features


def make_reference_end_map(dates: pd.Series, cutoff: pd.Timestamp | None) -> pd.Series:
    index = pd.to_datetime(dates).sort_values().unique()
    if cutoff is None:
        values = pd.to_datetime(index) - pd.Timedelta(days=1)
    else:
        values = np.repeat(pd.Timestamp(cutoff), len(index))
    return pd.Series(values, index=index)


def fit_global_lgbm(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[Any, str]:
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
        from sklearn.ensemble import HistGradientBoostingRegressor
    except Exception as exc:
        raise ImportError("LightGBM unavailable and sklearn fallback missing") from exc

    model = HistGradientBoostingRegressor(
        learning_rate=0.03,
        max_iter=500,
        max_leaf_nodes=31,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model, "hist_gradient_boosting"


def make_safe_feature_names(columns: pd.Index) -> pd.Index:
    seen: dict[str, int] = {}
    safe_names: list[str] = []
    for raw_column in columns:
        safe = re.sub(r"[^0-9A-Za-z_]+", "_", str(raw_column)).strip("_")
        safe = safe or "feature"
        count = seen.get(safe, 0)
        seen[safe] = count + 1
        safe_names.append(safe if count == 0 else f"{safe}_{count}")
    return pd.Index(safe_names)


def encode_features(
    frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    template_columns: pd.Index | None = None,
    medians: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    numeric = frame[numeric_features].apply(pd.to_numeric, errors="coerce")
    if medians is None:
        medians = numeric.median(numeric_only=True)
    numeric = numeric.fillna(medians)

    categoricals = frame[categorical_features].fillna("ALL").astype(str)
    encoded = pd.get_dummies(categoricals, columns=categorical_features, drop_first=False, dtype=float)
    X = pd.concat([numeric.reset_index(drop=True), encoded.reset_index(drop=True)], axis=1)
    X.columns = make_safe_feature_names(X.columns)
    if template_columns is not None:
        X = X.reindex(columns=template_columns, fill_value=0.0)
    return X, medians


def prepare_training_matrix(
    model_table: pd.DataFrame,
    target_column: str,
    numeric_features: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.Index, pd.Series]:
    clean = model_table.dropna(subset=[target_column]).copy()
    X, medians = encode_features(clean, numeric_features, STATIC_CATEGORICAL_FEATURES)
    y = pd.to_numeric(clean[target_column], errors="coerce").reset_index(drop=True)
    mask = (~y.isna()).to_numpy()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    return X, y, X.columns, medians


def predict_model(
    model: Any,
    model_type: str,
    X: pd.DataFrame,
) -> np.ndarray:
    return np.asarray(model.predict(X), dtype=float)


def build_group_ratio_map(level_table: pd.DataFrame, train_end: pd.Timestamp) -> dict[str, float]:
    subset = level_table[level_table[DATE_COL] <= train_end].copy()
    global_ratio = safe_divide(subset["target_cogs"].sum(), subset["target_revenue"].sum())
    ratio_map: dict[str, float] = {}
    grouped = subset.groupby("series_id", as_index=False).agg(revenue=("target_revenue", "sum"), cogs=("target_cogs", "sum"))
    for row in grouped.itertuples(index=False):
        ratio = safe_divide(row.cogs, row.revenue)
        ratio_map[str(row.series_id)] = float(global_ratio if pd.isna(ratio) else ratio)
    return ratio_map


def fit_recursive_family(
    level_name: str,
    level_table: pd.DataFrame,
    static_context: pd.DataFrame,
    web_lookup: pd.DataFrame,
    train_end: pd.Timestamp,
) -> dict[str, Any]:
    training_table = prepare_recursive_training_table(level_table, static_context, web_lookup)
    train_subset = training_table[training_table[DATE_COL] <= train_end].copy()
    X_train, y_train, feature_columns, medians = prepare_training_matrix(
        train_subset,
        target_column="target_revenue",
        numeric_features=RECURSIVE_NUMERIC_FEATURES,
    )
    model, model_type = fit_global_lgbm(X_train, y_train)
    feature_importance = extract_feature_importance(model, model_type, feature_columns)
    ratio_map = build_group_ratio_map(level_table, train_end)
    return {
        "family": f"{level_name}_recursive",
        "model": model,
        "model_type": model_type,
        "feature_columns": feature_columns,
        "medians": medians,
        "feature_importance": feature_importance,
        "ratio_map": ratio_map,
    }


def fit_direct_family(
    level_name: str,
    direct_table: pd.DataFrame,
    train_end: pd.Timestamp,
) -> dict[str, Any]:
    train_subset = direct_table[direct_table[DATE_COL] <= train_end].copy()
    train_subset = train_subset[train_subset["baseline_revenue"] > 0].copy()
    train_subset["target_log_ratio"] = np.log((train_subset["target_revenue"] / train_subset["baseline_revenue"]).clip(lower=1e-6))
    X_train, y_train, feature_columns, medians = prepare_training_matrix(
        train_subset,
        target_column="target_log_ratio",
        numeric_features=DIRECT_NUMERIC_FEATURES,
    )
    model, model_type = fit_global_lgbm(X_train, y_train)
    feature_importance = extract_feature_importance(model, model_type, feature_columns)
    ratio_map = build_group_ratio_map(direct_table, train_end)
    return {
        "family": f"{level_name}_direct",
        "model": model,
        "model_type": model_type,
        "feature_columns": feature_columns,
        "medians": medians,
        "feature_importance": feature_importance,
        "ratio_map": ratio_map,
    }


def extract_feature_importance(model: Any, model_type: str, feature_columns: pd.Index) -> pd.DataFrame:
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
        return pd.DataFrame({"feature": feature_columns, "importance_split": np.nan, "importance_gain": model.feature_importances_})
    return pd.DataFrame({"feature": feature_columns, "importance_split": np.nan, "importance_gain": np.nan})


def build_recursive_prediction_row(
    forecast_date: pd.Timestamp,
    series_id: str,
    static_date_row: pd.Series,
    history: pd.Series,
    series_meta: dict[str, Any],
    web_lookup: pd.DataFrame,
    reference_end: pd.Timestamp,
) -> dict[str, Any]:
    row = {
        DATE_COL: forecast_date,
        "series_id": series_id,
        **series_meta,
    }
    row.update(static_date_row.to_dict())

    for lag, feature in [(7, "lag_7"), (14, "lag_14"), (30, "lag_30"), (90, "lag_90"), (180, "lag_180"), (365, "lag_365")]:
        lag_date = forecast_date - pd.Timedelta(days=lag)
        row[feature] = float(history.get(lag_date, np.nan))

    past = history[history.index < forecast_date].sort_index()
    for window in [7, 30, 90, 365]:
        values = past.tail(window)
        row[f"rolling_mean_{window}"] = float(values.mean()) if len(values) == window else np.nan

    row["lag365_to_roll365_ratio"] = safe_divide(row["lag_365"], row["rolling_mean_365"])
    row["lag365_to_recent_mean_ratio"] = safe_divide(row["lag_365"], row["rolling_mean_90"])
    row.update(compute_web_reference_features(forecast_date, reference_end, web_lookup))
    return row


def recursive_predict_family(
    fit_info: dict[str, Any],
    level_table: pd.DataFrame,
    static_context: pd.DataFrame,
    prediction_dates: pd.Series,
    train_end: pd.Timestamp,
    web_lookup: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    static_by_date = static_context.set_index(DATE_COL).sort_index()
    histories = {
        str(series_id): group.set_index(DATE_COL)["target_revenue"].sort_index()
        for series_id, group in level_table[level_table[DATE_COL] <= train_end].groupby("series_id")
    }
    series_meta_map = {
        str(row.series_id): {
            "level": row.level,
            "category": row.category,
            "segment": row.segment,
            "region": row.region,
            "campaign_base": row.campaign_base,
            "level_id": row.level_id,
        }
        for row in level_table.drop_duplicates("series_id")[["series_id", "level", "category", "segment", "region", "campaign_base", "level_id"]].itertuples(index=False)
    }

    group_rows: list[dict[str, Any]] = []
    total_rows: list[dict[str, Any]] = []
    for forecast_date in pd.to_datetime(prediction_dates):
        rows = []
        for series_id, history in histories.items():
            static_row = static_by_date.loc[forecast_date]
            row = build_recursive_prediction_row(
                forecast_date=forecast_date,
                series_id=series_id,
                static_date_row=static_row,
                history=history,
                series_meta=series_meta_map[series_id],
                web_lookup=web_lookup,
                reference_end=train_end,
            )
            rows.append(row)

        pred_frame = pd.DataFrame(rows)
        X_pred, _ = encode_features(
            pred_frame,
            RECURSIVE_NUMERIC_FEATURES,
            STATIC_CATEGORICAL_FEATURES,
            template_columns=fit_info["feature_columns"],
            medians=fit_info["medians"],
        )
        preds = np.clip(predict_model(fit_info["model"], fit_info["model_type"], X_pred), 0.0, None)
        pred_frame["predicted_revenue"] = preds
        pred_frame["predicted_cogs"] = pred_frame["series_id"].map(fit_info["ratio_map"]).fillna(0.89) * pred_frame["predicted_revenue"]
        group_rows.append(pred_frame[[DATE_COL, "series_id", "predicted_revenue", "predicted_cogs"]].copy())

        totals = pred_frame.groupby(DATE_COL, as_index=False).agg(
            predicted_revenue=("predicted_revenue", "sum"),
            predicted_cogs=("predicted_cogs", "sum"),
        )
        total_rows.append(totals)

        for series_id, pred_value in zip(pred_frame["series_id"], pred_frame["predicted_revenue"]):
            histories[series_id].loc[forecast_date] = float(pred_value)

    return pd.concat(group_rows, ignore_index=True), pd.concat(total_rows, ignore_index=True)


def direct_predict_family(
    fit_info: dict[str, Any],
    prediction_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_pred, _ = encode_features(
        prediction_table,
        DIRECT_NUMERIC_FEATURES,
        STATIC_CATEGORICAL_FEATURES,
        template_columns=fit_info["feature_columns"],
        medians=fit_info["medians"],
    )
    predicted_log_ratio = predict_model(fit_info["model"], fit_info["model_type"], X_pred)
    predicted_ratio = np.clip(np.exp(predicted_log_ratio), 0.20, 5.0)
    group = prediction_table.copy()
    group["predicted_revenue"] = np.clip(group["baseline_revenue"] * predicted_ratio, 0.0, None)
    group["predicted_cogs"] = group["series_id"].map(fit_info["ratio_map"]).fillna(0.89) * group["predicted_revenue"]
    totals = group.groupby(DATE_COL, as_index=False).agg(
        predicted_revenue=("predicted_revenue", "sum"),
        predicted_cogs=("predicted_cogs", "sum"),
    )
    return group[[DATE_COL, "series_id", "predicted_revenue", "predicted_cogs"]], totals


def build_validation_prediction_table(
    fold_name: str,
    dates: pd.Series,
    actual_total: pd.Series,
    model_predictions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    output = pd.DataFrame({DATE_COL: pd.to_datetime(dates), "fold": fold_name, "actual_Revenue": actual_total.to_numpy(dtype=float)})
    for family_name, frame in model_predictions.items():
        renamed = frame[[DATE_COL, "predicted_revenue"]].rename(columns={"predicted_revenue": family_name})
        output = output.merge(renamed, on=DATE_COL, how="left", validate="one_to_one")
    return output


def search_multilevel_blend(validation_table: pd.DataFrame, component_columns: list[str]) -> dict[str, Any]:
    actual = validation_table["actual_Revenue"].to_numpy(dtype=float)
    components = validation_table[component_columns].to_numpy(dtype=float)
    best: dict[str, Any] | None = None

    step_units = 20
    for weights in product(range(step_units + 1), repeat=len(component_columns)):
        if sum(weights) != step_units:
            continue
        weights_arr = np.asarray(weights, dtype=float) / step_units
        pred = components @ weights_arr
        metrics = compute_metrics(validation_table["actual_Revenue"], pred)
        row = {
            "blend_type": "multi_level_only",
            "weights": dict(zip(component_columns, weights_arr)),
            **metrics,
        }
        if best is None or row["RMSE"] < best["RMSE"]:
            best = row

    if best is None:
        raise ValueError("Blend search failed to produce any candidate")
    return best


def search_benchmark_blend(
    validation_table: pd.DataFrame,
    component_columns: list[str],
) -> dict[str, Any] | None:
    if "benchmark_current_best" not in validation_table.columns:
        return None
    subset = validation_table.dropna(subset=["benchmark_current_best"]).copy()
    if subset.empty:
        return None

    actual = subset["actual_Revenue"].to_numpy(dtype=float)
    components = subset[component_columns].to_numpy(dtype=float)
    benchmark = subset["benchmark_current_best"].to_numpy(dtype=float)

    best: dict[str, Any] | None = None
    for bench_units in [0, 1, 2, 3, 4]:
        remaining_units = 20 - bench_units
        for weights in product(range(remaining_units + 1), repeat=len(component_columns)):
            if sum(weights) != remaining_units:
                continue
            weights_arr = np.asarray(weights, dtype=float) / 20.0
            bench_weight = bench_units / 20.0
            pred = components @ weights_arr + benchmark * bench_weight
            metrics = compute_metrics(pd.Series(actual), pred)
            row = {
                "blend_type": "with_small_current_best",
                "benchmark_weight": bench_weight,
                "weights": dict(zip(component_columns, weights_arr)),
                **metrics,
            }
            if best is None or row["RMSE"] < best["RMSE"]:
                best = row
    return best


def save_submission(
    path: Path,
    sample_submission: pd.DataFrame,
    revenue_pred: np.ndarray,
    cogs_pred: np.ndarray,
) -> None:
    output = sample_submission[[DATE_COL]].copy()
    output[TARGET_COL] = np.clip(np.asarray(revenue_pred, dtype=float), 0.0, None)
    output[COGS_COL] = np.clip(np.asarray(cogs_pred, dtype=float), 0.0, None)
    validate_submission_frame(output, sample_submission)
    output.to_csv(path, index=False)


def run() -> None:
    logger = setup_logging()
    reporter = Reporter(logger)

    reporter.emit("M5-Style Multi-Level Forecasting")
    reporter.emit("================================")
    reporter.emit("")

    inputs = load_inputs()
    fact = build_transaction_fact(inputs)
    level_aggs = build_level_aggregates(inputs, fact)
    full_dates = inputs["sales"][DATE_COL]

    dense_levels = {name: densify_level_table(df, full_dates) for name, df in level_aggs.items()}
    historical_static, web_lookup = build_historical_date_contexts(inputs)
    future_static = build_future_date_contexts(inputs)

    reporter.emit(f"Built dense level tables: { {name: len(df) for name, df in dense_levels.items()} }")
    reporter.emit(
        "Series counts: "
        f"category={dense_levels['category']['series_id'].nunique()}, "
        f"segment={dense_levels['segment']['series_id'].nunique()}, "
        f"region={dense_levels['region']['series_id'].nunique()}, "
        f"category_promo={dense_levels['category_promo']['series_id'].nunique()}"
    )

    # Precompute direct training tables.
    reference_lookup = {
        "total": build_group_revenue_lookup(dense_levels["total"]),
        "category": build_group_revenue_lookup(dense_levels["category"]),
        "segment": build_group_revenue_lookup(dense_levels["segment"]),
    }
    direct_training_tables = {}
    for level_name in ["total", "category", "segment"]:
        ref_map = make_reference_end_map(dense_levels[level_name][DATE_COL], cutoff=None)
        direct_training_tables[level_name] = build_series_reference_table(
            level_table=dense_levels[level_name],
            static_context=historical_static,
            revenue_lookup=reference_lookup[level_name],
            web_lookup=web_lookup,
            reference_end_map=ref_map,
        )

    reporter.emit("")
    reporter.emit("1. Long-horizon validation")
    fold_metrics_rows: list[dict[str, Any]] = []
    validation_frames: list[pd.DataFrame] = []
    feature_importance_frames: list[pd.DataFrame] = []
    avg_family_predictions: dict[str, list[np.ndarray]] = {}

    for fold_name, train_end, validation_start, validation_end in FOLDS:
        reporter.emit(f"{fold_name}: train <= {train_end.date()}, validate {validation_start.date()} -> {validation_end.date()}")
        validation_dates = inputs["sales"].loc[
            (inputs["sales"][DATE_COL] >= validation_start) & (inputs["sales"][DATE_COL] <= validation_end),
            DATE_COL,
        ].reset_index(drop=True)
        actual_total = inputs["sales"].loc[
            (inputs["sales"][DATE_COL] >= validation_start) & (inputs["sales"][DATE_COL] <= validation_end),
            TARGET_COL,
        ].reset_index(drop=True)

        fold_model_predictions: dict[str, pd.DataFrame] = {}

        # Total direct
        total_direct_fit = fit_direct_family("total", direct_training_tables["total"], train_end=train_end)
        feature_importance_frames.append(total_direct_fit["feature_importance"].assign(family="total_direct", fold=fold_name))
        total_direct_pred_table = build_series_reference_table(
            level_table=dense_levels["total"][dense_levels["total"][DATE_COL].isin(validation_dates)].copy(),
            static_context=historical_static[historical_static[DATE_COL].isin(validation_dates)].copy(),
            revenue_lookup=reference_lookup["total"],
            web_lookup=web_lookup,
            reference_end_map=make_reference_end_map(validation_dates, cutoff=train_end),
        )
        _, total_direct_totals = direct_predict_family(total_direct_fit, total_direct_pred_table)
        total_direct_totals["predicted_cogs"] = total_direct_totals["predicted_revenue"] * 0.8900
        fold_model_predictions["total_direct"] = total_direct_totals.copy()

        # Category family
        category_recursive_fit = fit_recursive_family("category", dense_levels["category"], historical_static, web_lookup, train_end=train_end)
        feature_importance_frames.append(category_recursive_fit["feature_importance"].assign(family="category_recursive", fold=fold_name))
        _, category_recursive_totals = recursive_predict_family(
            category_recursive_fit,
            dense_levels["category"],
            historical_static[historical_static[DATE_COL].isin(validation_dates)].copy(),
            validation_dates,
            train_end,
            web_lookup,
        )
        fold_model_predictions["category_recursive_sum"] = category_recursive_totals.copy()

        category_direct_fit = fit_direct_family("category", direct_training_tables["category"], train_end=train_end)
        feature_importance_frames.append(category_direct_fit["feature_importance"].assign(family="category_direct", fold=fold_name))
        category_direct_pred_table = build_series_reference_table(
            level_table=dense_levels["category"][dense_levels["category"][DATE_COL].isin(validation_dates)].copy(),
            static_context=historical_static[historical_static[DATE_COL].isin(validation_dates)].copy(),
            revenue_lookup=reference_lookup["category"],
            web_lookup=web_lookup,
            reference_end_map=make_reference_end_map(validation_dates, cutoff=train_end),
        )
        _, category_direct_totals = direct_predict_family(category_direct_fit, category_direct_pred_table)
        fold_model_predictions["category_direct_sum"] = category_direct_totals.copy()

        # Segment family
        segment_recursive_fit = fit_recursive_family("segment", dense_levels["segment"], historical_static, web_lookup, train_end=train_end)
        feature_importance_frames.append(segment_recursive_fit["feature_importance"].assign(family="segment_recursive", fold=fold_name))
        _, segment_recursive_totals = recursive_predict_family(
            segment_recursive_fit,
            dense_levels["segment"],
            historical_static[historical_static[DATE_COL].isin(validation_dates)].copy(),
            validation_dates,
            train_end,
            web_lookup,
        )
        fold_model_predictions["segment_recursive_sum"] = segment_recursive_totals.copy()

        segment_direct_fit = fit_direct_family("segment", direct_training_tables["segment"], train_end=train_end)
        feature_importance_frames.append(segment_direct_fit["feature_importance"].assign(family="segment_direct", fold=fold_name))
        segment_direct_pred_table = build_series_reference_table(
            level_table=dense_levels["segment"][dense_levels["segment"][DATE_COL].isin(validation_dates)].copy(),
            static_context=historical_static[historical_static[DATE_COL].isin(validation_dates)].copy(),
            revenue_lookup=reference_lookup["segment"],
            web_lookup=web_lookup,
            reference_end_map=make_reference_end_map(validation_dates, cutoff=train_end),
        )
        _, segment_direct_totals = direct_predict_family(segment_direct_fit, segment_direct_pred_table)
        fold_model_predictions["segment_direct_sum"] = segment_direct_totals.copy()

        fold_validation = build_validation_prediction_table(fold_name, validation_dates, actual_total, fold_model_predictions)

        # optional benchmark on 2022 only
        benchmark = load_current_best_validation()
        if benchmark is not None:
            bench_slice = benchmark[(benchmark[DATE_COL] >= validation_start) & (benchmark[DATE_COL] <= validation_end)].copy()
            if not bench_slice.empty:
                fold_validation = fold_validation.merge(
                    bench_slice[[DATE_COL, "current_base_pred"]].rename(columns={"current_base_pred": "benchmark_current_best"}),
                    on=DATE_COL,
                    how="left",
                )

        # metrics per family
        for family_name in fold_model_predictions:
            metrics = compute_metrics(fold_validation["actual_Revenue"], fold_validation[family_name].to_numpy(dtype=float))
            monthly = compute_monthly_rmse(fold_validation[DATE_COL], fold_validation["actual_Revenue"], fold_validation[family_name].to_numpy(dtype=float))
            fold_metrics_rows.append(
                {
                    "fold": fold_name,
                    "family": family_name,
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "R2": metrics["R2"],
                    "top10_RMSE": metrics["top10_RMSE"],
                    "top10_underprediction": metrics["top10_underprediction"],
                    "non_spike_RMSE": metrics["non_spike_RMSE"],
                    "monthly_RMSE_mean": float(monthly["RMSE"].mean()),
                }
            )

        validation_frames.append(fold_validation)

    validation_all = pd.concat(validation_frames, ignore_index=True)
    comparison_df = pd.DataFrame(fold_metrics_rows)
    avg_comparison = (
        comparison_df.groupby("family", as_index=False)
        .agg(
            avg_MAE=("MAE", "mean"),
            avg_RMSE=("RMSE", "mean"),
            avg_R2=("R2", "mean"),
            avg_top10_RMSE=("top10_RMSE", "mean"),
            avg_non_spike_RMSE=("non_spike_RMSE", "mean"),
            avg_monthly_RMSE=("monthly_RMSE_mean", "mean"),
        )
        .sort_values("avg_RMSE")
        .reset_index(drop=True)
    )
    reporter.emit_frame("Fold metrics by family:", comparison_df)
    reporter.emit_frame("Average metrics by family:", avg_comparison)

    reporter.emit("")
    reporter.emit("2. Multi-level blend search")
    component_columns = [
        "total_direct",
        "category_recursive_sum",
        "category_direct_sum",
        "segment_recursive_sum",
        "segment_direct_sum",
    ]
    blend_best = search_multilevel_blend(validation_all.dropna(subset=component_columns), component_columns)
    benchmark_blend = search_benchmark_blend(validation_all.dropna(subset=component_columns), component_columns)
    reporter.emit(f"Best multi-level only blend weights: {blend_best['weights']}")
    if benchmark_blend is not None:
        reporter.emit(
            f"Best optional benchmark blend (2022-available subset): benchmark_weight={benchmark_blend['benchmark_weight']:.2f}, "
            f"weights={benchmark_blend['weights']}"
        )

    weights_series = pd.Series(blend_best["weights"])
    validation_all["best_multilevel_blend"] = validation_all[component_columns].to_numpy(dtype=float) @ weights_series.reindex(component_columns).to_numpy(dtype=float)
    blend_metrics = compute_metrics(validation_all["actual_Revenue"], validation_all["best_multilevel_blend"].to_numpy(dtype=float))
    reporter.emit(
        f"Best blend validation RMSE={blend_metrics['RMSE']:,.2f}, MAE={blend_metrics['MAE']:,.2f}, R2={blend_metrics['R2']:.6f}"
    )

    # Save validation outputs
    validation_all.to_csv(VALIDATION_PREDICTIONS_PATH, index=False)
    comparison_save = pd.concat(
        [
            comparison_df.assign(summary_type="fold"),
            avg_comparison.assign(summary_type="average", fold="ALL"),
            pd.DataFrame(
                [
                    {
                        "fold": "ALL",
                        "family": "best_multilevel_blend",
                        "MAE": blend_metrics["MAE"],
                        "RMSE": blend_metrics["RMSE"],
                        "R2": blend_metrics["R2"],
                        "top10_RMSE": blend_metrics["top10_RMSE"],
                        "top10_underprediction": blend_metrics["top10_underprediction"],
                        "non_spike_RMSE": blend_metrics["non_spike_RMSE"],
                        "monthly_RMSE_mean": np.nan,
                        "summary_type": "blend",
                        "weights": str(blend_best["weights"]),
                    }
                ]
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    comparison_save.to_csv(MODEL_COMPARISON_PATH, index=False)
    feature_importance_output = pd.concat(feature_importance_frames, ignore_index=True)
    feature_importance_output.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    reporter.emit("")
    reporter.emit("3. Train final models on full history")
    # Choose best variant within category and segment families.
    best_category_family = avg_comparison[avg_comparison["family"].isin(["category_recursive_sum", "category_direct_sum"])].sort_values("avg_RMSE").iloc[0]["family"]
    best_segment_family = avg_comparison[avg_comparison["family"].isin(["segment_recursive_sum", "segment_direct_sum"])].sort_values("avg_RMSE").iloc[0]["family"]
    reporter.emit(f"Best category family: {best_category_family}")
    reporter.emit(f"Best segment family: {best_segment_family}")

    train_end = inputs["sales"][DATE_COL].max()
    prediction_dates = inputs["sample_submission"][DATE_COL]

    # Fit final total direct
    total_direct_fit_final = fit_direct_family("total", direct_training_tables["total"], train_end=train_end)
    total_direct_future_table = build_series_reference_table(
        level_table=dense_levels["total"].iloc[:0].assign(**{DATE_COL: prediction_dates, "series_id": "TOTAL", "level": "TOTAL", "category": "ALL", "segment": "ALL", "region": "ALL", "campaign_base": "ALL", "level_id": 0}),
        static_context=future_static,
        revenue_lookup=reference_lookup["total"],
        web_lookup=web_lookup,
        reference_end_map=make_reference_end_map(prediction_dates, cutoff=train_end),
    )
    # the assign above can create weird broadcasting; rebuild cleanly:
    total_future_level = pd.DataFrame(
        {
            DATE_COL: prediction_dates,
            "series_id": "TOTAL",
            "level": "TOTAL",
            "category": "ALL",
            "segment": "ALL",
            "region": "ALL",
            "campaign_base": "ALL",
            "level_id": 0,
            "target_revenue": np.nan,
            "target_cogs": np.nan,
            "target_quantity": np.nan,
            "target_item_lines": np.nan,
        }
    )
    total_direct_future_table = build_series_reference_table(
        level_table=total_future_level,
        static_context=future_static,
        revenue_lookup=reference_lookup["total"],
        web_lookup=web_lookup,
        reference_end_map=make_reference_end_map(prediction_dates, cutoff=train_end),
    )
    _, total_direct_future = direct_predict_family(total_direct_fit_final, total_direct_future_table)
    total_direct_future["predicted_cogs"] = total_direct_future["predicted_revenue"] * 0.8900

    # Fit final category models
    category_recursive_fit_final = fit_recursive_family("category", dense_levels["category"], historical_static, web_lookup, train_end=train_end)
    _, category_recursive_future = recursive_predict_family(
        category_recursive_fit_final,
        dense_levels["category"],
        future_static,
        prediction_dates,
        train_end,
        web_lookup,
    )
    category_direct_fit_final = fit_direct_family("category", direct_training_tables["category"], train_end=train_end)
    category_future_level = dense_levels["category"].drop_duplicates("series_id")[
        ["series_id", "level", "category", "segment", "region", "campaign_base", "level_id"]
    ].assign(key=1).merge(
        pd.DataFrame({DATE_COL: prediction_dates, "key": 1}),
        on="key",
        how="outer",
    ).drop(columns="key")
    category_future_level["target_revenue"] = np.nan
    category_future_level["target_cogs"] = np.nan
    category_future_level["target_quantity"] = np.nan
    category_future_level["target_item_lines"] = np.nan
    category_direct_future_table = build_series_reference_table(
        level_table=category_future_level,
        static_context=future_static,
        revenue_lookup=reference_lookup["category"],
        web_lookup=web_lookup,
        reference_end_map=make_reference_end_map(prediction_dates, cutoff=train_end),
    )
    _, category_direct_future = direct_predict_family(category_direct_fit_final, category_direct_future_table)

    # Fit final segment models
    segment_recursive_fit_final = fit_recursive_family("segment", dense_levels["segment"], historical_static, web_lookup, train_end=train_end)
    _, segment_recursive_future = recursive_predict_family(
        segment_recursive_fit_final,
        dense_levels["segment"],
        future_static,
        prediction_dates,
        train_end,
        web_lookup,
    )
    segment_direct_fit_final = fit_direct_family("segment", direct_training_tables["segment"], train_end=train_end)
    segment_future_level = dense_levels["segment"].drop_duplicates("series_id")[
        ["series_id", "level", "category", "segment", "region", "campaign_base", "level_id"]
    ].assign(key=1).merge(
        pd.DataFrame({DATE_COL: prediction_dates, "key": 1}),
        on="key",
        how="outer",
    ).drop(columns="key")
    segment_future_level["target_revenue"] = np.nan
    segment_future_level["target_cogs"] = np.nan
    segment_future_level["target_quantity"] = np.nan
    segment_future_level["target_item_lines"] = np.nan
    segment_direct_future_table = build_series_reference_table(
        level_table=segment_future_level,
        static_context=future_static,
        revenue_lookup=reference_lookup["segment"],
        web_lookup=web_lookup,
        reference_end_map=make_reference_end_map(prediction_dates, cutoff=train_end),
    )
    _, segment_direct_future = direct_predict_family(segment_direct_fit_final, segment_direct_future_table)

    future_component_map = {
        "total_direct": total_direct_future,
        "category_recursive_sum": category_recursive_future,
        "category_direct_sum": category_direct_future,
        "segment_recursive_sum": segment_recursive_future,
        "segment_direct_sum": segment_direct_future,
    }

    def get_total_frame(frame: pd.DataFrame) -> pd.DataFrame:
        if "series_id" in frame.columns:
            return frame.groupby(DATE_COL, as_index=False).agg(
                predicted_revenue=("predicted_revenue", "sum"),
                predicted_cogs=("predicted_cogs", "sum"),
            )
        return frame[[DATE_COL, "predicted_revenue", "predicted_cogs"]].copy()

    future_totals = {name: get_total_frame(frame) for name, frame in future_component_map.items()}

    # Save category and segment bottom-up using best family within each.
    category_best_totals = future_totals[best_category_family]
    segment_best_totals = future_totals[best_segment_family]
    save_submission(
        SUBMISSION_CATEGORY_PATH,
        inputs["sample_submission"],
        category_best_totals["predicted_revenue"].to_numpy(dtype=float),
        category_best_totals["predicted_cogs"].to_numpy(dtype=float),
    )
    save_submission(
        SUBMISSION_SEGMENT_PATH,
        inputs["sample_submission"],
        segment_best_totals["predicted_revenue"].to_numpy(dtype=float),
        segment_best_totals["predicted_cogs"].to_numpy(dtype=float),
    )

    blend_revenue = np.zeros(len(inputs["sample_submission"]), dtype=float)
    blend_cogs = np.zeros(len(inputs["sample_submission"]), dtype=float)
    for component_name, weight in blend_best["weights"].items():
        totals = future_totals[component_name]
        blend_revenue += totals["predicted_revenue"].to_numpy(dtype=float) * weight
        blend_cogs += totals["predicted_cogs"].to_numpy(dtype=float) * weight

    save_submission(SUBMISSION_BLEND_PATH, inputs["sample_submission"], blend_revenue, blend_cogs)
    save_submission(SUBMISSION_BLEND_8900_PATH, inputs["sample_submission"], blend_revenue, blend_revenue * 0.8900)
    save_submission(SUBMISSION_BLEND_8950_PATH, inputs["sample_submission"], blend_revenue, blend_revenue * 0.8950)
    save_submission(SUBMISSION_BLEND_9000_PATH, inputs["sample_submission"], blend_revenue, blend_revenue * 0.9000)

    reporter.emit("")
    reporter.emit("4. Final summary")
    reporter.emit_frame("Top features (all families head):", feature_importance_output.head(30))
    reporter.emit(f"Best multi-level blend weights: {blend_best['weights']}")
    if benchmark_blend is not None:
        reporter.emit(
            f"Optional benchmark blend on 2022-available subset: benchmark_weight={benchmark_blend['benchmark_weight']:.2f}, "
            f"weights={benchmark_blend['weights']}"
        )

    fold3_avg = avg_comparison[avg_comparison["family"].isin(["category_recursive_sum", "category_direct_sum", "segment_recursive_sum", "segment_direct_sum"])]
    reporter.emit_frame("Category/segment families average:", fold3_avg)

    if load_current_best_validation() is not None:
        benchmark_current = load_current_best_validation()
        year2022_mask = validation_all[DATE_COL].dt.year == 2022
        if year2022_mask.any():
            current_2022 = benchmark_current.set_index(DATE_COL)
            compare_rows = []
            for family in ["category_recursive_sum", "category_direct_sum", "segment_recursive_sum", "segment_direct_sum", "best_multilevel_blend"]:
                subset = validation_all.loc[year2022_mask, [DATE_COL, "actual_Revenue", family]].copy()
                subset = subset.merge(
                    current_2022[["current_base_pred"]],
                    left_on=DATE_COL,
                    right_index=True,
                    how="left",
                )
                family_metrics = compute_metrics(subset["actual_Revenue"], subset[family].to_numpy(dtype=float))
                benchmark_metrics = compute_metrics(subset["actual_Revenue"], subset["current_base_pred"].to_numpy(dtype=float))
                compare_rows.append(
                    {
                        "family": family,
                        "family_RMSE_2022": family_metrics["RMSE"],
                        "benchmark_RMSE_2022": benchmark_metrics["RMSE"],
                        "beats_benchmark_2022": bool(family_metrics["RMSE"] < benchmark_metrics["RMSE"]),
                    }
                )
            reporter.emit_frame("2022 benchmark comparison:", pd.DataFrame(compare_rows))

    reporter.emit(
        "Created submission files: submission_m5_category_bottomup.csv, "
        "submission_m5_segment_bottomup.csv, submission_m5_multilevel_blend.csv, "
        "submission_m5_multilevel_blend_cogs8900.csv, submission_m5_multilevel_blend_cogs8950.csv, "
        "submission_m5_multilevel_blend_cogs9000.csv"
    )
    reporter.emit(
        "Recommended upload order: "
        f"{[SUBMISSION_BLEND_8900_PATH.name, SUBMISSION_BLEND_PATH.name, SUBMISSION_CATEGORY_PATH.name, SUBMISSION_SEGMENT_PATH.name, SUBMISSION_BLEND_8950_PATH.name, SUBMISSION_BLEND_9000_PATH.name]}"
    )
    reporter.emit(
        "Leakage safety confirmation: multi-level models only use historical transaction targets, safe calendar/promo/inventory context, and direct/reference or recursive lag structures without any future actual Revenue/COGS or same-day future demand."
    )
    reporter.save(REPORT_PATH)


if __name__ == "__main__":
    run()
