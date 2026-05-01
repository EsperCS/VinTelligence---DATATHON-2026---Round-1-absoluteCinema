from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "log"

FEATURE_TABLE_PATH = DATA_DIR / "daily_feature_table.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
PRUNED_SUBMISSION_PATH = DATA_DIR / "submission_pruned_ensemble.csv"
SPIKE_SUBMISSION_PATH = DATA_DIR / "submission_spike_aware.csv"

ADAPTIVE_PATH = DATA_DIR / "submission_adaptive.csv"
CONSERVATIVE_PATH = DATA_DIR / "submission_adaptive_conservative.csv"
AGGRESSIVE_PATH = DATA_DIR / "submission_adaptive_aggressive.csv"
LOG_FILE = LOG_DIR / "adaptive_ensemble.log"

DATE_COL = "Date"
TARGET_COL = "Revenue"
COGS_COL = "COGS"
SPIKE_HEAVY_THRESHOLD = 0.70
EPSILON = 1e-6


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("adaptive_ensemble")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def load_submission(path: Path, sample_dates: pd.Series) -> pd.DataFrame:
    submission = pd.read_csv(path, parse_dates=[DATE_COL])
    submission[DATE_COL] = pd.to_datetime(submission[DATE_COL], errors="coerce").dt.normalize()
    aligned = pd.DataFrame({DATE_COL: sample_dates.copy()}).merge(
        submission[[DATE_COL, TARGET_COL, COGS_COL]],
        on=DATE_COL,
        how="left",
        validate="one_to_one",
    )
    if aligned[[TARGET_COL, COGS_COL]].isna().any().any():
        raise ValueError(f"Submission has missing dates after alignment: {path}")
    aligned[TARGET_COL] = pd.to_numeric(aligned[TARGET_COL], errors="coerce")
    aligned[COGS_COL] = pd.to_numeric(aligned[COGS_COL], errors="coerce")
    return aligned


def load_feature_table(path: Path = FEATURE_TABLE_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[DATE_COL], low_memory=False)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce").dt.normalize()
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    if df[DATE_COL].isna().any():
        raise ValueError("Feature table contains invalid dates.")
    return df


def safe_divide(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator):
        return np.nan
    return float(numerator) / (abs(float(denominator)) + EPSILON)


def compute_volatility_30_from_history(history: pd.Series) -> pd.Series:
    ordered = pd.to_numeric(history, errors="coerce").sort_index()
    rolling_mean_30 = ordered.shift(1).rolling(window=30, min_periods=30).mean()
    rolling_std_30 = ordered.shift(1).rolling(window=30, min_periods=30).std()
    return rolling_std_30 / (rolling_mean_30.abs() + EPSILON)


def compute_submission_features(history: pd.Series, forecast_date: pd.Timestamp) -> dict[str, float]:
    lag7 = float(history.get(forecast_date - pd.Timedelta(days=7), np.nan))
    lag365 = float(history.get(forecast_date - pd.Timedelta(days=365), np.nan))
    past_history = history[history.index < forecast_date].sort_index()
    tail30 = past_history.tail(30)

    if len(tail30) == 30:
        rolling_mean_30 = float(tail30.mean())
        rolling_std_30 = float(tail30.std(ddof=1))
        volatility_30 = safe_divide(rolling_std_30, rolling_mean_30)
    else:
        rolling_mean_30 = np.nan
        volatility_30 = np.nan

    return {
        "lag7_to_roll30_ratio": safe_divide(lag7, rolling_mean_30),
        "volatility_30": volatility_30,
        "lag365_to_roll30_ratio": safe_divide(lag365, rolling_mean_30),
    }


def compute_spike_weight(
    lag7_to_roll30_ratio: float,
    volatility_30: float,
    lag365_to_roll30_ratio: float,
    median_volatility: float,
    base_weight: float,
) -> float:
    spike_weight = base_weight

    if pd.notna(lag7_to_roll30_ratio) and lag7_to_roll30_ratio > 1.2:
        spike_weight += 0.2
    if pd.notna(volatility_30) and volatility_30 > median_volatility:
        spike_weight += 0.1
    if pd.notna(lag365_to_roll30_ratio) and lag365_to_roll30_ratio > 1.3:
        spike_weight += 0.1

    spike_weight = min(spike_weight, 0.85)
    spike_weight = max(spike_weight, 0.15)
    return float(spike_weight)


def build_adaptive_submission(
    sample_submission: pd.DataFrame,
    pruned_submission: pd.DataFrame,
    spike_submission: pd.DataFrame,
    historical_revenue: pd.Series,
    median_volatility: float,
    base_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    history = historical_revenue.copy().sort_index()
    rows: list[dict[str, float | pd.Timestamp]] = []
    diagnostics: list[dict[str, float | pd.Timestamp]] = []

    pruned_by_date = pruned_submission.set_index(DATE_COL)
    spike_by_date = spike_submission.set_index(DATE_COL)

    for forecast_date in pd.to_datetime(sample_submission[DATE_COL]):
        features = compute_submission_features(history, forecast_date)
        spike_weight = compute_spike_weight(
            lag7_to_roll30_ratio=features["lag7_to_roll30_ratio"],
            volatility_30=features["volatility_30"],
            lag365_to_roll30_ratio=features["lag365_to_roll30_ratio"],
            median_volatility=median_volatility,
            base_weight=base_weight,
        )

        pruned_revenue = float(pruned_by_date.at[forecast_date, TARGET_COL])
        pruned_cogs = float(pruned_by_date.at[forecast_date, COGS_COL])
        spike_revenue = float(spike_by_date.at[forecast_date, TARGET_COL])
        spike_cogs = float(spike_by_date.at[forecast_date, COGS_COL])

        final_revenue = max(0.0, spike_weight * spike_revenue + (1.0 - spike_weight) * pruned_revenue)
        final_cogs = max(0.0, spike_weight * spike_cogs + (1.0 - spike_weight) * pruned_cogs)

        rows.append(
            {
                DATE_COL: forecast_date,
                TARGET_COL: final_revenue,
                COGS_COL: final_cogs,
            }
        )
        diagnostics.append(
            {
                DATE_COL: forecast_date,
                "spike_weight": spike_weight,
                **features,
            }
        )

        history.loc[forecast_date] = final_revenue

    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def validate_submission(submission: pd.DataFrame, sample_submission: pd.DataFrame) -> dict[str, int | bool]:
    exact_columns = list(submission.columns) == [DATE_COL, TARGET_COL, COGS_COL]
    row_count_matches = len(submission) == len(sample_submission) == 548
    date_order_matches = submission[DATE_COL].reset_index(drop=True).equals(
        sample_submission[DATE_COL].reset_index(drop=True)
    )
    missing_values = int(submission.isna().sum().sum())
    negative_values = int(
        (
            (pd.to_numeric(submission[TARGET_COL], errors="coerce") < 0)
            | (pd.to_numeric(submission[COGS_COL], errors="coerce") < 0)
        ).sum()
    )
    return {
        "rows": len(submission),
        "exact_columns": exact_columns,
        "row_count_matches": row_count_matches,
        "date_order_matches": date_order_matches,
        "missing_values": missing_values,
        "negative_values": negative_values,
    }


def print_summary(name: str, diagnostics: pd.DataFrame, validation: dict[str, int | bool]) -> None:
    avg_weight = float(diagnostics["spike_weight"].mean())
    spike_heavy_pct = float((diagnostics["spike_weight"] >= SPIKE_HEAVY_THRESHOLD).mean() * 100.0)
    max_weight = float(diagnostics["spike_weight"].max())
    min_weight = float(diagnostics["spike_weight"].min())

    print(name)
    print(f"avg spike weight: {avg_weight:.4f}")
    print(f"% days spike-heavy (weight >= {SPIKE_HEAVY_THRESHOLD:.2f}): {spike_heavy_pct:.2f}%")
    print(f"max weight: {max_weight:.4f}")
    print(f"min weight: {min_weight:.4f}")
    print(
        "validation: "
        f"rows={validation['rows']}, "
        f"row_count_matches={validation['row_count_matches']}, "
        f"date_order_matches={validation['date_order_matches']}, "
        f"missing_values={validation['missing_values']}, "
        f"negative_values={validation['negative_values']}"
    )
    print("")


def main() -> None:
    logger = setup_logging()

    feature_table = load_feature_table()
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH, parse_dates=[DATE_COL])
    sample_submission[DATE_COL] = pd.to_datetime(sample_submission[DATE_COL], errors="coerce").dt.normalize()

    pruned_submission = load_submission(PRUNED_SUBMISSION_PATH, sample_submission[DATE_COL])
    spike_submission = load_submission(SPIKE_SUBMISSION_PATH, sample_submission[DATE_COL])

    historical_revenue = (
        feature_table[[DATE_COL, TARGET_COL]]
        .dropna(subset=[TARGET_COL])
        .assign(**{TARGET_COL: lambda df: pd.to_numeric(df[TARGET_COL], errors="coerce")})
        .dropna(subset=[TARGET_COL])
        .set_index(DATE_COL)[TARGET_COL]
        .sort_index()
    )

    historical_volatility = compute_volatility_30_from_history(historical_revenue).dropna()
    median_volatility = float(historical_volatility.median()) if not historical_volatility.empty else 0.0
    logger.info("Median historical volatility_30: %.6f", median_volatility)

    variants = [
        ("submission_adaptive.csv", 0.50, ADAPTIVE_PATH),
        ("submission_adaptive_conservative.csv", 0.40, CONSERVATIVE_PATH),
        ("submission_adaptive_aggressive.csv", 0.60, AGGRESSIVE_PATH),
    ]

    for name, base_weight, output_path in variants:
        submission, diagnostics = build_adaptive_submission(
            sample_submission=sample_submission,
            pruned_submission=pruned_submission,
            spike_submission=spike_submission,
            historical_revenue=historical_revenue,
            median_volatility=median_volatility,
            base_weight=base_weight,
        )
        submission.to_csv(output_path, index=False)
        validation = validate_submission(submission, sample_submission)
        logger.info("%s | base_weight=%.2f | validation=%s", name, base_weight, validation)
        print_summary(name, diagnostics, validation)


if __name__ == "__main__":
    main()
