from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "log"
LOG_FILE = LOG_DIR / "data_pipeline.log"
OUTPUT_TABLE_NAME = "daily_feature_table"


DATE_COLUMNS: dict[str, list[str]] = {
    "customers": ["signup_date"],
    "inventory": ["snapshot_date"],
    "orders": ["order_date"],
    "promotions": ["start_date", "end_date"],
    "returns": ["return_date"],
    "reviews": ["review_date"],
    "sales": ["Date"],
    "sample_submission": ["Date"],
    "shipments": ["ship_date", "delivery_date"],
    "web_traffic": ["date"],
}


REQUIRED_COLUMNS: dict[str, list[str]] = {
    "customers": [
        "customer_id",
        "zip",
        "city",
        "signup_date",
        "gender",
        "age_group",
        "acquisition_channel",
    ],
    "geography": ["zip", "city", "region", "district"],
    "inventory": [
        "snapshot_date",
        "product_id",
        "stock_on_hand",
        "units_received",
        "units_sold",
        "stockout_days",
        "fill_rate",
        "stockout_flag",
        "sell_through_rate",
    ],
    "orders": [
        "order_id",
        "order_date",
        "customer_id",
        "zip",
        "order_status",
        "payment_method",
        "device_type",
        "order_source",
    ],
    "order_items": [
        "order_id",
        "product_id",
        "quantity",
        "unit_price",
        "discount_amount",
        "promo_id",
        "promo_id_2",
    ],
    "payments": ["order_id", "payment_method", "payment_value", "installments"],
    "products": ["product_id", "product_name", "category", "segment", "price", "cogs"],
    "promotions": [
        "promo_id",
        "promo_name",
        "promo_type",
        "discount_value",
        "start_date",
        "end_date",
    ],
    "returns": [
        "return_id",
        "order_id",
        "product_id",
        "return_date",
        "return_quantity",
        "refund_amount",
    ],
    "reviews": ["review_id", "order_id", "product_id", "customer_id", "review_date", "rating"],
    "sales": ["Date", "Revenue", "COGS"],
    "sample_submission": ["Date", "Revenue", "COGS"],
    "shipments": ["order_id", "ship_date", "delivery_date", "shipping_fee"],
    "web_traffic": [
        "date",
        "sessions",
        "unique_visitors",
        "page_views",
        "bounce_rate",
        "avg_session_duration_sec",
        "traffic_source",
    ],
}


DTYPE_OVERRIDES: dict[str, dict[str, str]] = {
    "customers": {"customer_id": "string", "zip": "string"},
    "geography": {"zip": "string"},
    "inventory": {"product_id": "string"},
    "orders": {"order_id": "string", "customer_id": "string", "zip": "string"},
    "order_items": {
        "order_id": "string",
        "product_id": "string",
        "promo_id": "string",
        "promo_id_2": "string",
    },
    "payments": {"order_id": "string"},
    "products": {"product_id": "string"},
    "promotions": {"promo_id": "string"},
    "returns": {"return_id": "string", "order_id": "string", "product_id": "string"},
    "reviews": {
        "review_id": "string",
        "order_id": "string",
        "product_id": "string",
        "customer_id": "string",
    },
    "shipments": {"order_id": "string"},
}


CRITICAL_TABLES = {"sales", "orders", "order_items", "web_traffic", "inventory", "returns"}


def setup_logging(log_file: Path = LOG_FILE) -> logging.Logger:
    """Configure file and console logging for the pipeline."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logger = logging.getLogger(__name__)
    logger.info("Logging initialized: %s", log_file)
    return logger


def _coerce_date_columns(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    for column in DATE_COLUMNS.get(table_name, []):
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce").dt.normalize()
    return df


def _missing_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [column for column in columns if column not in df.columns]


def load_table(path: Path) -> pd.DataFrame:
    """Load one CSV and normalize configured date columns."""
    table_name = path.stem
    logger = logging.getLogger(__name__)

    df = pd.read_csv(
        path,
        dtype=DTYPE_OVERRIDES.get(table_name),
        low_memory=False,
    )
    df = _coerce_date_columns(df, table_name)

    logger.info("Loaded %-18s rows=%s cols=%s", path.name, f"{len(df):,}", df.shape[1])
    return df


def load_all_data(data_dir: Path = DATA_DIR) -> Dict[str, pd.DataFrame]:
    """Load every raw CSV in data_dir except generated pipeline outputs."""
    logger = logging.getLogger(__name__)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    data: Dict[str, pd.DataFrame] = {}
    csv_paths = sorted(data_dir.glob("*.csv"))

    for path in csv_paths:
        if path.stem == OUTPUT_TABLE_NAME:
            logger.info("Skipping generated output table: %s", path.name)
            continue
        data[path.stem] = load_table(path)

    missing_critical = sorted(CRITICAL_TABLES - set(data))
    if missing_critical:
        raise FileNotFoundError(f"Missing critical CSV tables: {missing_critical}")

    return data


def validate_raw_data(data: Dict[str, pd.DataFrame]) -> None:
    """Validate table presence, expected columns, date parsing, and duplicate rows."""
    logger = logging.getLogger(__name__)

    for table_name, required_columns in REQUIRED_COLUMNS.items():
        if table_name not in data:
            logger.warning("Expected table not found: %s.csv", table_name)
            continue

        df = data[table_name]
        missing = _missing_columns(df, required_columns)
        if missing:
            raise ValueError(f"{table_name}.csv missing required columns: {missing}")

        duplicate_rows = int(df.duplicated().sum())
        if duplicate_rows:
            logger.warning("%s.csv has %s duplicate rows", table_name, f"{duplicate_rows:,}")

        for column in DATE_COLUMNS.get(table_name, []):
            if column not in df.columns:
                continue

            null_dates = int(df[column].isna().sum())
            if null_dates:
                message = f"{table_name}.{column} has {null_dates:,} unparsable/missing dates"
                if table_name in CRITICAL_TABLES or table_name == "sales":
                    raise ValueError(message)
                logger.warning(message)
                continue

            logger.info(
                "%s.%s date range: %s -> %s",
                table_name,
                column,
                df[column].min().date(),
                df[column].max().date(),
            )

        logger.info(
            "%s.csv validation complete | rows=%s columns=%s",
            table_name,
            f"{len(df):,}",
            df.shape[1],
        )
