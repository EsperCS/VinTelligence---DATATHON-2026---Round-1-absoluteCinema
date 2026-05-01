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

PROMOTIONS_PATH = DATA_DIR / "promotions.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"

SYNTHETIC_PROMOTIONS_PATH = DATA_DIR / "synthetic_promotions_2023_2024.csv"
FUTURE_PROMO_FEATURES_PATH = DATA_DIR / "future_promo_calendar_features.csv"
LOG_FILE = LOG_DIR / "build_promo_2023_features.log"

DATE_COL = "Date"

FUTURE_FEATURE_COLUMNS = [
    "future_calendar_active_promo_count",
    "future_calendar_any_promo",
    "future_calendar_avg_discount_value",
    "future_calendar_max_discount_value",
    "future_calendar_stackable_promo_count",
    "future_calendar_has_stackable_promo",
    "future_calendar_has_category_specific_promo",
    "future_calendar_percentage_promo_count",
    "future_calendar_fixed_promo_count",
    "future_promo_avg_duration_days",
    "future_promo_max_duration_days",
    "future_promo_avg_day_number",
    "future_promo_avg_days_remaining",
    "future_promo_avg_progress_ratio",
    "future_promo_is_first_3_days",
    "future_promo_is_last_3_days",
    "future_promo_is_first_7_days",
    "future_promo_is_last_7_days",
    "future_promotion_campaign_index",
]


class Reporter:
    """Print and log progress."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def emit(self, message: str = "") -> None:
        print(message)
        if message:
            self.logger.info(message)

    def emit_frame(self, title: str, frame: pd.DataFrame | pd.Series) -> None:
        self.emit(title)
        if frame.empty:
            self.emit("(empty)")
            return
        self.emit(frame.to_string(index=False))


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("build_promo_2023_features")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger


def load_promotions(path: Path = PROMOTIONS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"promotions.csv not found: {path}")

    promotions = pd.read_csv(path, low_memory=False)
    promotions["start_date"] = pd.to_datetime(promotions["start_date"], errors="coerce").dt.normalize()
    promotions["end_date"] = pd.to_datetime(promotions["end_date"], errors="coerce").dt.normalize()
    promotions = promotions.dropna(subset=["start_date", "end_date"]).copy()
    promotions["source_year"] = promotions["start_date"].dt.year.astype(int)
    promotions["promo_name_base"] = (
        promotions["promo_name"]
        .astype(str)
        .str.replace(r"\s+\d{4}$", "", regex=True)
        .str.strip()
    )
    promotions["discount_value"] = pd.to_numeric(promotions.get("discount_value", 0), errors="coerce").fillna(0)
    promotions["min_order_value"] = pd.to_numeric(promotions.get("min_order_value", 0), errors="coerce").fillna(0)
    return promotions.sort_values(["start_date", "end_date", "promo_id"]).reset_index(drop=True)


def load_sample_submission(path: Path = SAMPLE_SUBMISSION_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"sample_submission.csv not found: {path}")
    sample = pd.read_csv(path, parse_dates=[DATE_COL], low_memory=False)
    sample[DATE_COL] = pd.to_datetime(sample[DATE_COL], errors="coerce").dt.normalize()
    if sample[DATE_COL].isna().any():
        raise ValueError("sample_submission.csv contains invalid dates")
    return sample


def safe_replace_year(date_value: pd.Timestamp, target_year: int) -> pd.Timestamp:
    try:
        return date_value.replace(year=target_year)
    except ValueError:
        return pd.Timestamp(year=target_year, month=2, day=28)


def determine_even_source_year(promotions: pd.DataFrame) -> int:
    even_counts = promotions[promotions["source_year"] % 2 == 0].groupby("source_year").size()
    expected_even_count = int(even_counts.mode().iloc[0]) if not even_counts.empty else 4
    count_2022 = int(even_counts.get(2022, 0))
    if count_2022 >= expected_even_count and count_2022 > 0:
        return 2022
    count_2020 = int(even_counts.get(2020, 0))
    if count_2020 > 0:
        return 2020
    fallback_year = int(even_counts.sort_index(ascending=False).index[0])
    return fallback_year


def build_synthetic_promotions_for_year(
    promotions: pd.DataFrame,
    source_year: int,
    target_year: int,
) -> pd.DataFrame:
    source = promotions[promotions["source_year"] == source_year].copy()
    if source.empty:
        return pd.DataFrame()

    source = source.sort_values(["start_date", "end_date", "promo_name"]).reset_index(drop=True)
    year_shift = target_year - source_year
    rows: list[dict[str, Any]] = []

    for idx, row in enumerate(source.itertuples(index=False), start=1):
        shifted_start = safe_replace_year(row.start_date, row.start_date.year + year_shift)
        shifted_end = safe_replace_year(row.end_date, row.end_date.year + year_shift)
        duration_days = (shifted_end - shifted_start).days + 1
        rows.append(
            {
                "promo_id": f"SYN_PROMO_{target_year}_{idx:02d}",
                "promo_name": f"{row.promo_name_base} {target_year}",
                "promo_name_base": row.promo_name_base,
                "promo_type": row.promo_type,
                "discount_value": row.discount_value,
                "start_date": shifted_start,
                "end_date": shifted_end,
                "applicable_category": row.applicable_category,
                "promo_channel": row.promo_channel,
                "stackable_flag": row.stackable_flag,
                "min_order_value": row.min_order_value,
                "source_pattern_year": source_year,
                "target_year": target_year,
                "campaign_index": idx,
                "duration_days": duration_days,
            }
        )

    return pd.DataFrame(rows)


def build_synthetic_promotions(promotions: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    synthetic_2023 = build_synthetic_promotions_for_year(promotions, source_year=2021, target_year=2023)
    even_source_year = determine_even_source_year(promotions)
    synthetic_2024 = build_synthetic_promotions_for_year(promotions, source_year=even_source_year, target_year=2024)
    synthetic = pd.concat([synthetic_2023, synthetic_2024], ignore_index=True)
    synthetic = synthetic.sort_values(["start_date", "end_date", "promo_id"]).reset_index(drop=True)
    return synthetic, even_source_year


def stackable_to_int(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    return text.isin(["1", "true", "yes", "y"]).astype(int)


def has_category_specific(series: pd.Series) -> pd.Series:
    text = series.astype("string")
    return (text.notna() & text.str.strip().ne("")).astype(int)


def build_future_promo_calendar_features(
    dates: pd.Series,
    synthetic_promotions: pd.DataFrame,
) -> pd.DataFrame:
    calendar = pd.DataFrame({DATE_COL: pd.to_datetime(dates).sort_values().unique()})
    for feature in FUTURE_FEATURE_COLUMNS:
        calendar[feature] = 0.0

    if synthetic_promotions.empty:
        return calendar

    promos = synthetic_promotions.copy()
    promos["stackable_flag_numeric"] = stackable_to_int(promos["stackable_flag"])
    promos["category_specific"] = has_category_specific(promos["applicable_category"])
    promo_type = promos["promo_type"].astype(str).str.lower()
    promos["percentage_promo"] = promo_type.eq("percentage").astype(int)
    promos["fixed_promo"] = promo_type.eq("fixed").astype(int)

    min_date = calendar[DATE_COL].min()
    max_date = calendar[DATE_COL].max()
    rows: list[dict[str, Any]] = []

    for row in promos.itertuples(index=False):
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
                    "promo_id": row.promo_id,
                    "discount_value": row.discount_value,
                    "stackable_flag_numeric": row.stackable_flag_numeric,
                    "category_specific": row.category_specific,
                    "percentage_promo": row.percentage_promo,
                    "fixed_promo": row.fixed_promo,
                    "duration_days": row.duration_days,
                    "promo_day_number": promo_day_number,
                    "promo_days_remaining": promo_days_remaining,
                    "promo_progress_ratio": promo_day_number / row.duration_days,
                    "promo_is_first_3_days": int(promo_day_number <= 3),
                    "promo_is_last_3_days": int(promo_days_remaining <= 2),
                    "promo_is_first_7_days": int(promo_day_number <= 7),
                    "promo_is_last_7_days": int(promo_days_remaining <= 6),
                    "campaign_index": row.campaign_index,
                }
            )

    if not rows:
        return calendar

    expanded = pd.DataFrame(rows)
    daily = (
        expanded.groupby(DATE_COL, as_index=False)
        .agg(
            future_calendar_active_promo_count=("promo_id", "nunique"),
            future_calendar_avg_discount_value=("discount_value", "mean"),
            future_calendar_max_discount_value=("discount_value", "max"),
            future_calendar_stackable_promo_count=("stackable_flag_numeric", "sum"),
            future_calendar_has_stackable_promo=("stackable_flag_numeric", "max"),
            future_calendar_has_category_specific_promo=("category_specific", "max"),
            future_calendar_percentage_promo_count=("percentage_promo", "sum"),
            future_calendar_fixed_promo_count=("fixed_promo", "sum"),
            future_promo_avg_duration_days=("duration_days", "mean"),
            future_promo_max_duration_days=("duration_days", "max"),
            future_promo_avg_day_number=("promo_day_number", "mean"),
            future_promo_avg_days_remaining=("promo_days_remaining", "mean"),
            future_promo_avg_progress_ratio=("promo_progress_ratio", "mean"),
            future_promo_is_first_3_days=("promo_is_first_3_days", "max"),
            future_promo_is_last_3_days=("promo_is_last_3_days", "max"),
            future_promo_is_first_7_days=("promo_is_first_7_days", "max"),
            future_promo_is_last_7_days=("promo_is_last_7_days", "max"),
            future_promotion_campaign_index=("campaign_index", "max"),
        )
    )
    daily["future_calendar_any_promo"] = (daily["future_calendar_active_promo_count"] > 0).astype(int)

    calendar = calendar.drop(columns=FUTURE_FEATURE_COLUMNS).merge(daily, on=DATE_COL, how="left")
    for feature in FUTURE_FEATURE_COLUMNS:
        calendar[feature] = calendar[feature].fillna(0)
    return calendar[[DATE_COL] + FUTURE_FEATURE_COLUMNS]


def run() -> None:
    logger = setup_logging()
    reporter = Reporter(logger)

    reporter.emit("Build Promo 2023/2024 Features")
    reporter.emit("==============================")
    reporter.emit("")

    promotions = load_promotions(PROMOTIONS_PATH)
    sample_submission = load_sample_submission(SAMPLE_SUBMISSION_PATH)
    reporter.emit(f"Loaded promotions: {len(promotions):,}")
    reporter.emit(
        f"Sample submission date range: {sample_submission[DATE_COL].min().date()} -> "
        f"{sample_submission[DATE_COL].max().date()}"
    )

    synthetic_promotions, even_source_year = build_synthetic_promotions(promotions)
    if synthetic_promotions.empty:
        raise ValueError("Synthetic promotions could not be built from promotions.csv")

    synthetic_promotions.to_csv(SYNTHETIC_PROMOTIONS_PATH, index=False)

    future_features = build_future_promo_calendar_features(
        dates=sample_submission[DATE_COL],
        synthetic_promotions=synthetic_promotions,
    )
    future_features.to_csv(FUTURE_PROMO_FEATURES_PATH, index=False)

    synthetic_2023 = synthetic_promotions[synthetic_promotions["target_year"] == 2023].copy()
    synthetic_2024 = synthetic_promotions[synthetic_promotions["target_year"] == 2024].copy()

    reporter.emit("")
    reporter.emit("Synthetic promotion summary")
    reporter.emit(f"Synthetic promotions for 2023: {len(synthetic_2023):,}")
    reporter.emit(f"Synthetic promotions for 2024: {len(synthetic_2024):,}")
    reporter.emit(f"2024 source pattern year used: {even_source_year}")
    reporter.emit(
        f"2023 synthetic date range: {synthetic_2023['start_date'].min().date()} -> "
        f"{synthetic_2023['end_date'].max().date()}"
    )
    reporter.emit(
        f"2024 synthetic date range: {synthetic_2024['start_date'].min().date()} -> "
        f"{synthetic_2024['end_date'].max().date()}"
    )
    reporter.emit_frame(
        "Synthetic campaigns and dates:",
        synthetic_promotions[
            [
                "promo_id",
                "promo_name",
                "promo_name_base",
                "target_year",
                "discount_value",
                "start_date",
                "end_date",
                "duration_days",
                "source_pattern_year",
            ]
        ],
    )

    promo_days_2023 = int(
        future_features[
            (future_features[DATE_COL].dt.year == 2023)
            & (future_features["future_calendar_any_promo"] == 1)
        ].shape[0]
    )
    promo_days_2024 = int(
        future_features[
            (future_features[DATE_COL].dt.year == 2024)
            & (future_features["future_calendar_any_promo"] == 1)
        ].shape[0]
    )
    missing_values = int(future_features.isna().sum().sum())
    date_match = future_features[DATE_COL].reset_index(drop=True).equals(
        sample_submission[DATE_COL].reset_index(drop=True)
    )

    reporter.emit("")
    reporter.emit("Validation checks")
    reporter.emit(f"Promo days in 2023 forecast period: {promo_days_2023}")
    reporter.emit(f"Promo days in 2024 forecast period: {promo_days_2024}")
    reporter.emit(f"No missing values in daily features: {missing_values == 0}")
    reporter.emit(f"Daily feature dates match sample_submission exactly: {date_match}")
    reporter.emit(f"Saved synthetic promotions: {SYNTHETIC_PROMOTIONS_PATH}")
    reporter.emit(f"Saved future promo features: {FUTURE_PROMO_FEATURES_PATH}")

    if missing_values != 0:
        raise ValueError(f"Future promo feature table contains {missing_values} missing values")
    if not date_match:
        raise ValueError("Future promo feature dates do not match sample_submission.csv exactly")


if __name__ == "__main__":
    run()
