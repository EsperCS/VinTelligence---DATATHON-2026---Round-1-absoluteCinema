from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from feature_engineering import add_all_features
from load_data import DATA_DIR, LOG_FILE, load_all_data, setup_logging, validate_raw_data
from preprocess import build_daily_base_features


OUTPUT_PATH = DATA_DIR / "daily_feature_table.csv"


def validate_final_dataset(df: pd.DataFrame) -> None:
    """Run final checks before writing the feature table."""
    if df["Date"].duplicated().any():
        raise ValueError("Final dataset contains duplicate Date values")

    if not df["Date"].is_monotonic_increasing:
        raise ValueError("Final dataset is not sorted by Date")

    expected_dates = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    missing_dates = expected_dates.difference(df["Date"])
    if len(missing_dates) > 0:
        raise ValueError(f"Final dataset has missing daily dates: {len(missing_dates):,}")

    logger = logging.getLogger(__name__)
    logger.info("Final dataset validation complete")


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact missing-value report for columns with missing values."""
    report = (
        df.isna()
        .sum()
        .rename("missing_count")
        .to_frame()
        .assign(missing_pct=lambda x: x["missing_count"] / len(df))
    )
    return report[report["missing_count"] > 0].sort_values("missing_count", ascending=False)


def print_summary(df: pd.DataFrame) -> None:
    """Print the required run summary."""
    report = missing_value_report(df)
    feature_count = len([column for column in df.columns if column not in {"Date", "Revenue"}])

    print("\nDaily feature table summary")
    print(f"Rows: {len(df):,}")
    print(f"Feature columns excluding Date and Revenue: {feature_count:,}")
    print(f"Date range: {df['Date'].min().date()} -> {df['Date'].max().date()}")

    if report.empty:
        print("Missing value report: no missing values")
        return

    print("Missing value report:")
    printable = report.copy()
    printable["missing_pct"] = (printable["missing_pct"] * 100).round(2)
    print(printable.to_string())


def build_dataset(output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    """Orchestrate the end-to-end daily feature table build."""
    logger = setup_logging(LOG_FILE)
    logger.info("Starting daily feature table build")

    data = load_all_data(DATA_DIR)
    validate_raw_data(data)

    dataset = build_daily_base_features(data)
    dataset = add_all_features(dataset)
    validate_final_dataset(dataset)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    logger.info("Saved final dataset to %s", output_path)

    print_summary(dataset)
    return dataset


def main() -> None:
    build_dataset()


if __name__ == "__main__":
    main()
