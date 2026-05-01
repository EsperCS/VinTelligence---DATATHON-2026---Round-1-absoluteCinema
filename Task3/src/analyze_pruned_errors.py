from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import train_final_model as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "log"

PREDICTIONS_PATH = DATA_DIR / "pruned_ensemble_validation_predictions.csv"
BY_DATE_PATH = DATA_DIR / "pruned_error_analysis_by_date.csv"
MONTHLY_PATH = DATA_DIR / "pruned_error_analysis_monthly.csv"
DOW_PATH = DATA_DIR / "pruned_error_analysis_dow.csv"
REPORT_PATH = LOG_DIR / "pruned_error_analysis_report.txt"


class Reporter:
    """Print and save report lines."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def emit(self, message: str = "") -> None:
        print(message)
        self.lines.append(message)

    def emit_frame(self, title: str, frame: pd.DataFrame) -> None:
        self.emit(title)
        if frame.empty:
            self.emit("(empty)")
            return
        self.emit(frame.to_string(index=False))

    def save(self, path: Path = REPORT_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def metrics(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> dict[str, float]:
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def load_predictions(path: Path = PREDICTIONS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run python src/final_feature_prune_and_retrain.py first."
        )
    df = pd.read_csv(path, parse_dates=["Date"], low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    required = ["Date", "actual_Revenue", "predicted_Revenue"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    df = df.dropna(subset=required).sort_values("Date").reset_index(drop=True)
    df["error"] = df["actual_Revenue"] - df["predicted_Revenue"]
    df["abs_error"] = df["error"].abs()
    df["pct_error"] = np.where(df["actual_Revenue"] != 0, df["error"] / df["actual_Revenue"], np.nan)
    return df


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach calendar, promo, and high-revenue flags for error slicing."""
    out = df.copy()
    out["month"] = out["Date"].dt.month
    out["year_month"] = out["Date"].dt.to_period("M").astype(str)
    out["day_of_week"] = out["Date"].dt.dayofweek
    out["dow_name"] = out["Date"].dt.day_name()

    promo = base.build_promotion_calendar(out["Date"], base.PROMOTIONS_PATH, logger=_NullLogger())
    out = out.merge(promo, on="Date", how="left", validate="one_to_one")
    out[base.PROMOTION_FEATURES] = out[base.PROMOTION_FEATURES].fillna(0)
    out["is_promo_day"] = (out["calendar_active_promo_count"] > 0).astype(int)

    p90 = out["actual_Revenue"].quantile(0.90)
    p95 = out["actual_Revenue"].quantile(0.95)
    out["is_high_revenue_day_p90"] = (out["actual_Revenue"] >= p90).astype(int)
    out["is_extreme_revenue_day_p95"] = (out["actual_Revenue"] >= p95).astype(int)
    out["prediction_direction"] = np.where(out["error"] > 0, "underprediction", "overprediction")
    out.loc[out["error"].abs() < 1e-9, "prediction_direction"] = "exact"
    return out


class _NullLogger:
    def warning(self, *_args, **_kwargs) -> None:
        return None


def group_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, part in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: value for column, value in zip(group_cols, keys)}
        row["count"] = len(part)
        row.update(metrics(part["actual_Revenue"], part["predicted_Revenue"]))
        row["mean_actual_Revenue"] = part["actual_Revenue"].mean()
        row["mean_predicted_Revenue"] = part["predicted_Revenue"].mean()
        row["bias_mean_error"] = part["error"].mean()
        row["underprediction_count"] = int((part["error"] > 0).sum())
        row["overprediction_count"] = int((part["error"] < 0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def analyze() -> None:
    reporter = Reporter()
    reporter.emit("Pruned Ensemble Error Analysis")
    reporter.emit("==============================")
    reporter.emit("")

    df = add_context_features(load_predictions(PREDICTIONS_PATH))
    overall = metrics(df["actual_Revenue"], df["predicted_Revenue"])
    reporter.emit(
        f"Overall validation metrics: MAE={overall['MAE']:,.2f}, "
        f"RMSE={overall['RMSE']:,.2f}, R2={overall['R2']:.6f}"
    )

    by_date = df.sort_values("abs_error", ascending=False).reset_index(drop=True)
    by_date.to_csv(BY_DATE_PATH, index=False)
    worst30 = by_date.head(30)

    monthly = group_metrics(df, ["year_month"]).sort_values("RMSE", ascending=False).reset_index(drop=True)
    monthly.to_csv(MONTHLY_PATH, index=False)

    dow = group_metrics(df, ["day_of_week", "dow_name"]).sort_values("RMSE", ascending=False).reset_index(drop=True)
    dow.to_csv(DOW_PATH, index=False)

    promo = group_metrics(df, ["is_promo_day"]).sort_values("is_promo_day").reset_index(drop=True)
    high_rev = group_metrics(df, ["is_high_revenue_day_p90"]).sort_values("is_high_revenue_day_p90").reset_index(drop=True)
    extreme_rev = group_metrics(df, ["is_extreme_revenue_day_p95"]).sort_values("is_extreme_revenue_day_p95").reset_index(drop=True)
    direction_counts = df["prediction_direction"].value_counts().rename_axis("direction").reset_index(name="count")

    reporter.emit("")
    reporter.emit_frame("Top 30 worst dates by absolute error:", worst30[
        [
            "Date",
            "actual_Revenue",
            "predicted_Revenue",
            "error",
            "abs_error",
            "pct_error",
            "year_month",
            "dow_name",
            "is_promo_day",
            "is_high_revenue_day_p90",
        ]
    ])
    reporter.emit("")
    reporter.emit_frame("Monthly error summary:", monthly)
    reporter.emit("")
    reporter.emit_frame("Day-of-week error summary:", dow)
    reporter.emit("")
    reporter.emit_frame("Promo-day vs non-promo-day error:", promo)
    reporter.emit("")
    reporter.emit_frame("High-revenue day error, top 10% threshold:", high_rev)
    reporter.emit("")
    reporter.emit_frame("Extreme-revenue day error, top 5% threshold:", extreme_rev)
    reporter.emit("")
    reporter.emit_frame("Underprediction vs overprediction count:", direction_counts)

    worst_month = monthly.iloc[0]
    worst_dow = dow.iloc[0]
    promo_rmse = promo.loc[promo["is_promo_day"] == 1, "RMSE"]
    nonpromo_rmse = promo.loc[promo["is_promo_day"] == 0, "RMSE"]
    high_rmse = high_rev.loc[high_rev["is_high_revenue_day_p90"] == 1, "RMSE"].iloc[0]
    normal_rmse = high_rev.loc[high_rev["is_high_revenue_day_p90"] == 0, "RMSE"].iloc[0]
    under_count = int((df["error"] > 0).sum())
    over_count = int((df["error"] < 0).sum())

    reporter.emit("")
    reporter.emit("Concentration diagnostics:")
    reporter.emit(f"Worst month by RMSE: {worst_month['year_month']} | RMSE={worst_month['RMSE']:,.2f}")
    reporter.emit(f"Worst day of week by RMSE: {worst_dow['dow_name']} | RMSE={worst_dow['RMSE']:,.2f}")
    if not promo_rmse.empty and not nonpromo_rmse.empty:
        reporter.emit(
            f"Promo-day RMSE={promo_rmse.iloc[0]:,.2f}; "
            f"non-promo RMSE={nonpromo_rmse.iloc[0]:,.2f}"
        )
    reporter.emit(f"High-revenue top-10% RMSE={high_rmse:,.2f}; normal-day RMSE={normal_rmse:,.2f}")
    reporter.emit(f"Underpredictions={under_count}; overpredictions={over_count}")

    reporter.emit("")
    reporter.emit("Concrete next feature recommendations:")
    reporter.emit("1. Add spike-specific features: previous-year same-week max/quantile and rolling max over 7/14/30 days.")
    reporter.emit("2. Add holiday/event proxy from calendar only: pre/post month-end, payday-like day-of-month buckets, and year-end sale windows.")
    reporter.emit("3. Add asymmetric objective or target transform experiment, because high-revenue top-10% days dominate RMSE.")
    reporter.emit("4. Add promo intensity phase again, but only if regularized/pruned; compare promo interactions specifically on worst promo dates.")
    reporter.emit("5. Add residual correction model for months/days with systematic bias, using only calendar and lagged residual features.")

    reporter.emit("")
    reporter.emit(f"Saved by-date analysis: {BY_DATE_PATH}")
    reporter.emit(f"Saved monthly analysis: {MONTHLY_PATH}")
    reporter.emit(f"Saved day-of-week analysis: {DOW_PATH}")
    reporter.save(REPORT_PATH)


if __name__ == "__main__":
    analyze()
