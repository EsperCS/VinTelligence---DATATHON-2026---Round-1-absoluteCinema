from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert numeric columns with logging if non-null values become invalid."""
    for column in columns:
        if column not in df.columns:
            continue

        original_nulls = int(df[column].isna().sum())
        df[column] = pd.to_numeric(df[column], errors="coerce")
        converted_nulls = int(df[column].isna().sum())
        introduced_nulls = converted_nulls - original_nulls

        if introduced_nulls > 0:
            logger.warning(
                "%s values in column %s could not be converted to numeric",
                f"{introduced_nulls:,}",
                column,
            )

    return df


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide while returning 0 for zero or missing denominators."""
    denominator = denominator.astype(float).replace(0, np.nan)
    result = numerator.astype(float).divide(denominator)
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def prepare_sales(sales: pd.DataFrame) -> pd.DataFrame:
    """Prepare the daily base table from sales.csv."""
    df = sales.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df = coerce_numeric(df, ["Revenue", "COGS"])

    if df[["Date", "Revenue", "COGS"]].isna().any().any():
        missing = df[["Date", "Revenue", "COGS"]].isna().sum()
        raise ValueError(f"sales.csv contains missing required values:\n{missing}")

    df = (
        df.groupby("Date", as_index=False)
        .agg(Revenue=("Revenue", "sum"), COGS=("COGS", "sum"))
        .sort_values("Date")
        .reset_index(drop=True)
    )

    expected_dates = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    missing_dates = expected_dates.difference(df["Date"])
    if len(missing_dates) > 0:
        raise ValueError(f"sales.csv has missing daily dates: {len(missing_dates):,}")

    logger.info("Prepared sales base table | rows=%s", f"{len(df):,}")
    return df


def aggregate_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Aggregate order-level features by order_date."""
    df = orders.copy()
    df["Date"] = pd.to_datetime(df["order_date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["Date"])

    daily = (
        df.groupby("Date", as_index=False)
        .agg(
            orders_count=("order_id", "nunique"),
            unique_customers=("customer_id", "nunique"),
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    logger.info("Aggregated orders features | rows=%s", f"{len(daily):,}")
    return daily


def aggregate_order_items(order_items: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Aggregate order item features to the order date via orders.csv."""
    item_df = order_items.copy()
    order_dates = orders[["order_id", "order_date"]].copy()
    order_dates["Date"] = pd.to_datetime(order_dates["order_date"], errors="coerce").dt.normalize()
    order_dates = order_dates.dropna(subset=["Date"]).drop(columns=["order_date"])

    df = item_df.merge(order_dates, on="order_id", how="left", validate="many_to_one")
    missing_dates = int(df["Date"].isna().sum())
    if missing_dates:
        logger.warning(
            "Dropping %s order_items rows without a matching order date",
            f"{missing_dates:,}",
        )
    df = df.dropna(subset=["Date"])

    df = coerce_numeric(df, ["quantity", "unit_price", "discount_amount"])
    df[["quantity", "unit_price", "discount_amount"]] = df[
        ["quantity", "unit_price", "discount_amount"]
    ].fillna(0)

    promo_cols = [column for column in ["promo_id", "promo_id_2"] if column in df.columns]
    if promo_cols:
        df["promo_used"] = df[promo_cols].notna().any(axis=1).astype(int)
    else:
        df["promo_used"] = 0

    df["gross_item_value"] = df["quantity"] * df["unit_price"]
    df["discount_rate_line"] = safe_divide(df["discount_amount"], df["gross_item_value"])

    grouped = (
        df.groupby("Date", as_index=False)
        .agg(
            item_lines_count=("order_id", "size"),
            total_quantity=("quantity", "sum"),
            gross_item_value=("gross_item_value", "sum"),
            total_discount_amount=("discount_amount", "sum"),
            avg_discount_amount=("discount_amount", "mean"),
            avg_discount_rate=("discount_rate_line", "mean"),
            promo_usage_rate=("promo_used", "mean"),
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    grouped["discount_to_gross_rate"] = safe_divide(
        grouped["total_discount_amount"],
        grouped["gross_item_value"],
    )
    grouped = grouped.drop(columns=["gross_item_value"])

    logger.info("Aggregated order item features | rows=%s", f"{len(grouped):,}")
    return grouped


def aggregate_web_traffic(web_traffic: pd.DataFrame) -> pd.DataFrame:
    """Aggregate web traffic features by date."""
    df = web_traffic.copy()
    df["Date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["Date"])
    df = coerce_numeric(
        df,
        ["sessions", "unique_visitors", "page_views", "bounce_rate", "avg_session_duration_sec"],
    )
    numeric_cols = ["sessions", "unique_visitors", "page_views", "bounce_rate", "avg_session_duration_sec"]
    df[numeric_cols] = df[numeric_cols].fillna(0)

    df["bounce_rate_weighted"] = df["bounce_rate"] * df["sessions"]
    df["duration_weighted"] = df["avg_session_duration_sec"] * df["sessions"]

    daily = (
        df.groupby("Date", as_index=False)
        .agg(
            web_sessions=("sessions", "sum"),
            web_unique_visitors=("unique_visitors", "sum"),
            web_page_views=("page_views", "sum"),
            web_bounce_rate_weighted=("bounce_rate_weighted", "sum"),
            web_duration_weighted=("duration_weighted", "sum"),
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    daily["web_bounce_rate"] = safe_divide(daily["web_bounce_rate_weighted"], daily["web_sessions"])
    daily["web_avg_session_duration_sec"] = safe_divide(
        daily["web_duration_weighted"],
        daily["web_sessions"],
    )
    daily["web_traffic_available"] = 1

    daily = daily.drop(columns=["web_bounce_rate_weighted", "web_duration_weighted"])
    logger.info("Aggregated web traffic features | rows=%s", f"{len(daily):,}")
    return daily


def aggregate_inventory_snapshots(inventory: pd.DataFrame) -> pd.DataFrame:
    """Aggregate product inventory snapshots to snapshot_date."""
    df = inventory.copy()
    df["Date"] = pd.to_datetime(df["snapshot_date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["Date"])

    numeric_cols = [
        "stock_on_hand",
        "units_received",
        "units_sold",
        "stockout_days",
        "days_of_supply",
        "fill_rate",
        "stockout_flag",
        "overstock_flag",
        "reorder_flag",
        "sell_through_rate",
    ]
    df = coerce_numeric(df, numeric_cols)
    for column in numeric_cols:
        if column in df.columns:
            df[column] = df[column].fillna(0)

    optional_mean_cols = {
        "days_of_supply": "inventory_days_of_supply",
        "overstock_flag": "inventory_overstock_rate",
        "reorder_flag": "inventory_reorder_rate",
    }

    aggregations = {
        "inventory_product_count": ("product_id", "nunique"),
        "inventory_stock_on_hand": ("stock_on_hand", "sum"),
        "inventory_units_received": ("units_received", "sum"),
        "inventory_units_sold": ("units_sold", "sum"),
        "inventory_stockout_days": ("stockout_days", "sum"),
        "inventory_stockout_rate": ("stockout_flag", "mean"),
        "inventory_fill_rate": ("fill_rate", "mean"),
        "inventory_sell_through_rate": ("sell_through_rate", "mean"),
    }

    for source_col, output_col in optional_mean_cols.items():
        if source_col in df.columns:
            aggregations[output_col] = (source_col, "mean")

    daily = (
        df.groupby("Date", as_index=False)
        .agg(**aggregations)
        .sort_values("Date")
        .reset_index(drop=True)
    )

    logger.info("Aggregated inventory snapshot features | rows=%s", f"{len(daily):,}")
    return daily


def align_inventory_to_daily(
    inventory_daily: pd.DataFrame,
    sales_dates: pd.Series,
) -> pd.DataFrame:
    """
    Align inventory snapshots to the sales calendar.

    Snapshot values are forward-filled only after the snapshot_date, which avoids
    leaking month-end inventory information backward into earlier days.
    """
    calendar = pd.DataFrame({"Date": pd.to_datetime(sales_dates).sort_values().unique()})
    daily = calendar.merge(inventory_daily, on="Date", how="left")

    feature_cols = [column for column in daily.columns if column != "Date"]
    daily["inventory_snapshot_observed"] = daily[feature_cols].notna().any(axis=1).astype(int)
    daily[feature_cols] = daily[feature_cols].ffill()
    daily["inventory_data_available"] = daily[feature_cols].notna().any(axis=1).astype(int)
    daily[feature_cols] = daily[feature_cols].fillna(0)

    logger.info("Aligned inventory snapshots to daily calendar | rows=%s", f"{len(daily):,}")
    return daily


def aggregate_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Aggregate return features by return_date."""
    df = returns.copy()
    df["Date"] = pd.to_datetime(df["return_date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["Date"])
    df = coerce_numeric(df, ["return_quantity", "refund_amount"])
    df[["return_quantity", "refund_amount"]] = df[["return_quantity", "refund_amount"]].fillna(0)

    daily = (
        df.groupby("Date", as_index=False)
        .agg(
            returns_count=("return_id", "nunique"),
            returned_orders_count=("order_id", "nunique"),
            return_quantity=("return_quantity", "sum"),
            refund_amount=("refund_amount", "sum"),
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    logger.info("Aggregated returns features | rows=%s", f"{len(daily):,}")
    return daily


def build_daily_base_features(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build the daily feature table before time-series lag features."""
    sales = prepare_sales(data["sales"])
    orders = aggregate_orders(data["orders"])
    order_items = aggregate_order_items(data["order_items"], data["orders"])
    web_traffic = aggregate_web_traffic(data["web_traffic"])
    inventory = align_inventory_to_daily(
        aggregate_inventory_snapshots(data["inventory"]),
        sales["Date"],
    )
    returns = aggregate_returns(data["returns"])

    feature_frames = [orders, order_items, web_traffic, inventory, returns]
    dataset = sales.copy()

    for feature_frame in feature_frames:
        dataset = dataset.merge(feature_frame, on="Date", how="left", validate="one_to_one")

    dataset["return_rate"] = safe_divide(dataset["return_quantity"], dataset["total_quantity"])

    feature_cols = [column for column in dataset.columns if column not in {"Date", "Revenue", "COGS"}]
    dataset[feature_cols] = dataset[feature_cols].fillna(0)

    dataset = dataset.sort_values("Date").reset_index(drop=True)
    logger.info(
        "Built joined daily base feature table | rows=%s columns=%s",
        f"{len(dataset):,}",
        dataset.shape[1],
    )
    return dataset
