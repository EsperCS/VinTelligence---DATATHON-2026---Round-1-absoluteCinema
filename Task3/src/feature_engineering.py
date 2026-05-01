from __future__ import annotations

import logging

import pandas as pd


logger = logging.getLogger(__name__)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic calendar features available for future dates."""
    output = df.copy()
    output["Date"] = pd.to_datetime(output["Date"], errors="coerce").dt.normalize()

    output["day_of_week"] = output["Date"].dt.dayofweek
    output["month"] = output["Date"].dt.month
    output["quarter"] = output["Date"].dt.quarter
    output["year"] = output["Date"].dt.year
    output["is_weekend"] = output["day_of_week"].isin([5, 6]).astype(int)

    logger.info("Added calendar time features")
    return output


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add trend and regime indicators."""
    output = df.sort_values("Date").reset_index(drop=True).copy()
    output["time_index"] = range(len(output))
    output["post_2019_flag"] = (output["Date"] >= pd.Timestamp("2019-01-01")).astype(int)

    logger.info("Added trend and regime features")
    return output


def add_lag_features(df: pd.DataFrame, target_col: str = "Revenue") -> pd.DataFrame:
    """Add lag and rolling features using only prior target values."""
    output = df.sort_values("Date").reset_index(drop=True).copy()

    for lag in [7, 14, 30]:
        output[f"lag_{lag}"] = output[target_col].shift(lag)

    shifted_target = output[target_col].shift(1)
    output["rolling_mean_7"] = shifted_target.rolling(window=7, min_periods=7).mean()
    output["rolling_mean_30"] = shifted_target.rolling(window=30, min_periods=30).mean()

    logger.info("Added leakage-safe lag and rolling target features")
    return output


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps."""
    output = add_time_features(df)
    output = add_trend_features(output)
    output = add_lag_features(output, target_col="Revenue")

    first_columns = ["Date", "Revenue", "COGS"]
    other_columns = [column for column in output.columns if column not in first_columns]
    output = output[first_columns + other_columns]

    logger.info("Feature engineering complete | columns=%s", output.shape[1])
    return output
