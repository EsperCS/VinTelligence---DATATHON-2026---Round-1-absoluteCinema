from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "daily_feature_table.csv"
LOG_DIR = PROJECT_ROOT / "log"
LOG_FILE = LOG_DIR / "dataset_validation.log"
REPORT_PATH = LOG_DIR / "dataset_validation_report.txt"

TARGET_COLUMNS = ["Revenue", "COGS"]
TRAIN_CUTOFF = pd.Timestamp("2022-01-01")
VALIDATION_END = pd.Timestamp("2022-12-31")
LEAKAGE_CORR_THRESHOLD = 0.95


class AuditReporter:
    """Print, log, and persist all audit messages."""

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
        self.emit(frame.to_string())

    def save(self, path: Path = REPORT_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        self.logger.info("Saved validation report to %s", path)


def setup_logging(log_file: Path = LOG_FILE) -> logging.Logger:
    """Configure file logging for dataset validation."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("dataset_validation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Logging initialized: %s", log_file)
    return logger


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load the generated daily feature table."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, parse_dates=["Date"], low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    return df.sort_values("Date").reset_index(drop=True)


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing counts and percentages for columns with missing values."""
    report = (
        df.isna()
        .sum()
        .rename("missing_count")
        .to_frame()
        .assign(missing_pct=lambda x: (x["missing_count"] / len(df) * 100).round(2))
    )
    return report[report["missing_count"] > 0].sort_values("missing_count", ascending=False)


def audit_basic_structure(df: pd.DataFrame, reporter: AuditReporter) -> dict[str, int]:
    """Audit shape, date continuity, missing values, and dtypes."""
    reporter.emit("Dataset Validation & Leakage Audit")
    reporter.emit("=" * 41)
    reporter.emit("")
    reporter.emit("1. Basic validation")
    reporter.emit(f"Rows: {len(df):,}")
    reporter.emit(f"Columns: {df.shape[1]:,}")

    min_date = df["Date"].min()
    max_date = df["Date"].max()
    reporter.emit(f"Date range: {min_date.date()} -> {max_date.date()}")

    duplicated_dates = int(df["Date"].duplicated().sum())
    expected_dates = pd.date_range(min_date, max_date, freq="D")
    observed_dates = pd.DatetimeIndex(df["Date"].dropna().drop_duplicates().sort_values())
    missing_dates = expected_dates.difference(observed_dates)

    reporter.emit(f"Duplicated dates count: {duplicated_dates:,}")
    reporter.emit(f"Missing dates count in daily sequence: {len(missing_dates):,}")
    if len(missing_dates) > 0:
        reporter.emit(f"First missing dates: {missing_dates[:10].strftime('%Y-%m-%d').tolist()}")

    missing_report = missing_value_report(df)
    if missing_report.empty:
        reporter.emit("Missing value report: no missing values")
    else:
        reporter.emit_frame("Missing value report:", missing_report)

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_columns = [column for column in df.columns if column not in numeric_columns]
    reporter.emit(f"Numeric column count: {len(numeric_columns):,}")
    reporter.emit(f"Non-numeric column count: {len(non_numeric_columns):,}")
    reporter.emit(f"Non-numeric columns: {non_numeric_columns}")

    return {
        "duplicated_dates": duplicated_dates,
        "missing_dates": len(missing_dates),
        "missing_values": int(df.isna().sum().sum()),
    }


def iqr_outlier_count(series: pd.Series) -> tuple[int, float, float]:
    """Count outliers using the 1.5 * IQR rule."""
    clean = series.dropna()
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    count = int(((clean < lower_bound) | (clean > upper_bound)).sum())
    return count, float(lower_bound), float(upper_bound)


def audit_targets(df: pd.DataFrame, reporter: AuditReporter) -> None:
    """Audit target-like columns for invalid values and extreme outliers."""
    reporter.emit("")
    reporter.emit("2. Target validation")

    for column in TARGET_COLUMNS:
        reporter.emit("")
        reporter.emit(f"Target column: {column}")
        summary = df[column].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        reporter.emit_frame("Summary statistics:", summary)

        negative_count = int((df[column] < 0).sum())
        zero_count = int((df[column] == 0).sum())
        outlier_count, lower_bound, upper_bound = iqr_outlier_count(df[column])

        reporter.emit(f"Negative values: {negative_count:,}")
        reporter.emit(f"Zero values: {zero_count:,}")
        reporter.emit(
            f"IQR outliers: {outlier_count:,} "
            f"(lower={lower_bound:,.2f}, upper={upper_bound:,.2f})"
        )


def audit_missing_handling(df: pd.DataFrame, reporter: AuditReporter) -> pd.DataFrame:
    """Create an in-memory copy with missing lag/rolling rows removed."""
    reporter.emit("")
    reporter.emit("3. Missing value handling recommendation")

    lag_rolling_columns = [
        column
        for column in df.columns
        if column.startswith("lag_") or column.startswith("rolling_")
    ]

    cleaned = df.dropna(subset=lag_rolling_columns).copy()
    reporter.emit(f"Lag/rolling columns checked: {lag_rolling_columns}")
    reporter.emit(
        "Recommended training copy: drop rows with missing lag/rolling features "
        f"-> shape={cleaned.shape}"
    )
    reporter.emit("Original CSV was not modified and no cleaned copy was saved.")

    return cleaned


def audit_correlations(df: pd.DataFrame, reporter: AuditReporter) -> pd.DataFrame:
    """Compute feature correlations with Revenue and flag leakage candidates."""
    reporter.emit("")
    reporter.emit("4. Correlation audit")

    numeric_df = df.select_dtypes(include=[np.number])
    corr = (
        numeric_df.corr(numeric_only=True)["Revenue"]
        .drop(labels=["Revenue"], errors="ignore")
        .dropna()
        .sort_values(ascending=False)
    )

    reporter.emit_frame("Top 20 positive correlations with Revenue:", corr.head(20).to_frame("corr"))
    reporter.emit_frame(
        "Top 20 negative correlations with Revenue:",
        corr.sort_values(ascending=True).head(20).to_frame("corr"),
    )

    high_corr = corr[corr.abs() >= LEAKAGE_CORR_THRESHOLD].sort_values(
        key=lambda values: values.abs(),
        ascending=False,
    )
    if high_corr.empty:
        reporter.emit(
            f"Possible leakage candidates with abs(correlation) >= {LEAKAGE_CORR_THRESHOLD}: none"
        )
    else:
        reporter.emit_frame(
            f"Possible leakage candidates with abs(correlation) >= {LEAKAGE_CORR_THRESHOLD}:",
            high_corr.to_frame("corr"),
        )

    return high_corr


def audit_time_split(df: pd.DataFrame, reporter: AuditReporter) -> None:
    """Audit a fixed 2022 holdout split."""
    reporter.emit("")
    reporter.emit("5. Time-based split audit")

    train = df[df["Date"] < TRAIN_CUTOFF]
    validation = df[(df["Date"] >= TRAIN_CUTOFF) & (df["Date"] <= VALIDATION_END)]

    reporter.emit(f"Train rows: {len(train):,}")
    reporter.emit(f"Validation rows: {len(validation):,}")
    reporter.emit(
        f"Train date range: {train['Date'].min().date()} -> {train['Date'].max().date()}"
    )
    reporter.emit(
        "Validation date range: "
        f"{validation['Date'].min().date()} -> {validation['Date'].max().date()}"
    )
    reporter.emit(f"Mean Revenue - train: {train['Revenue'].mean():,.2f}")
    reporter.emit(f"Mean Revenue - validation: {validation['Revenue'].mean():,.2f}")
    reporter.emit(f"Median Revenue - train: {train['Revenue'].median():,.2f}")
    reporter.emit(f"Median Revenue - validation: {validation['Revenue'].median():,.2f}")


def period_revenue_summary(df: pd.DataFrame, start_year: int, end_year: int) -> pd.Series:
    """Return revenue statistics for a closed year range."""
    mask = (df["Date"].dt.year >= start_year) & (df["Date"].dt.year <= end_year)
    values = df.loc[mask, "Revenue"]
    return pd.Series(
        {
            "count": int(values.count()),
            "mean_revenue": values.mean(),
            "median_revenue": values.median(),
            "total_revenue": values.sum(),
        },
        name=f"{start_year}-{end_year}",
    )


def audit_regime_shift(df: pd.DataFrame, reporter: AuditReporter) -> pd.DataFrame:
    """Compare yearly and pre/post-2019 revenue regimes."""
    reporter.emit("")
    reporter.emit("6. Regime-shift audit")

    yearly = (
        df.assign(year=df["Date"].dt.year)
        .groupby("year")
        .agg(
            count=("Revenue", "count"),
            mean_revenue=("Revenue", "mean"),
            median_revenue=("Revenue", "median"),
            std_revenue=("Revenue", "std"),
            total_revenue=("Revenue", "sum"),
        )
    )
    reporter.emit_frame("Yearly Revenue statistics:", yearly.round(2))

    pre_2019 = period_revenue_summary(df, 2012, 2018)
    post_2019 = period_revenue_summary(df, 2019, 2022)
    comparison = pd.concat([pre_2019, post_2019], axis=1).T

    ratios = pd.Series(
        {
            "mean_revenue_ratio_2019_2022_vs_2012_2018": (
                post_2019["mean_revenue"] / pre_2019["mean_revenue"]
            ),
            "median_revenue_ratio_2019_2022_vs_2012_2018": (
                post_2019["median_revenue"] / pre_2019["median_revenue"]
            ),
            "total_revenue_ratio_2019_2022_vs_2012_2018": (
                post_2019["total_revenue"] / pre_2019["total_revenue"]
            ),
        }
    )

    reporter.emit_frame("Period Revenue comparison:", comparison.round(2))
    reporter.emit_frame("Period ratios:", ratios.round(4))
    return comparison


def final_assessment(
    basic_issues: dict[str, int],
    high_corr: pd.Series,
    regime_comparison: pd.DataFrame,
    reporter: AuditReporter,
) -> None:
    """Write a concise final audit conclusion."""
    reporter.emit("")
    reporter.emit("7. Final assessment")

    structural_ok = basic_issues["duplicated_dates"] == 0 and basic_issues["missing_dates"] == 0
    leakage_candidates = high_corr.index.tolist()

    pre = regime_comparison.loc["2012-2018"]
    post = regime_comparison.loc["2019-2022"]
    mean_ratio = post["mean_revenue"] / pre["mean_revenue"]
    median_ratio = post["median_revenue"] / pre["median_revenue"]
    regime_shift = abs(mean_ratio - 1.0) >= 0.15 or abs(median_ratio - 1.0) >= 0.15

    if structural_ok:
        reporter.emit("Dataset structure: PASS - no duplicate dates and no missing daily dates.")
    else:
        reporter.emit("Dataset structure: REVIEW - duplicate dates or missing dates were detected.")

    if leakage_candidates:
        reporter.emit(f"Possible leakage candidates: {leakage_candidates}")
    else:
        reporter.emit("Possible leakage candidates: none by correlation threshold.")

    if regime_shift:
        reporter.emit(
            "Revenue regime shift: YES - 2019-2022 differs materially from 2012-2018 "
            f"(mean ratio={mean_ratio:.3f}, median ratio={median_ratio:.3f})."
        )
    else:
        reporter.emit(
            "Revenue regime shift: not material by 15% mean/median rule "
            f"(mean ratio={mean_ratio:.3f}, median ratio={median_ratio:.3f})."
        )

    if structural_ok and not leakage_candidates:
        reporter.emit("Step 3 readiness: OK to proceed with baseline model training.")
    elif structural_ok:
        reporter.emit(
            "Step 3 readiness: proceed only after reviewing/removing possible leakage candidates."
        )
    else:
        reporter.emit("Step 3 readiness: fix structural data issues before model training.")


def run_audit() -> None:
    logger = setup_logging()
    reporter = AuditReporter(logger)

    df = load_dataset(DATASET_PATH)
    logger.info("Loaded dataset %s | shape=%s", DATASET_PATH, df.shape)

    basic_issues = audit_basic_structure(df, reporter)
    audit_targets(df, reporter)
    audit_missing_handling(df, reporter)
    high_corr = audit_correlations(df, reporter)
    audit_time_split(df, reporter)
    regime_comparison = audit_regime_shift(df, reporter)
    final_assessment(basic_issues, high_corr, regime_comparison, reporter)
    reporter.save(REPORT_PATH)


if __name__ == "__main__":
    run_audit()
