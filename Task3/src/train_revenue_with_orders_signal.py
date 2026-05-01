from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import train_final_model as base
import train_orders_model as orders_mod
import train_revenue_with_predicted_orders as pred_orders_mod
import train_spike_aware_model as spike1


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "log"

TRAIN_DATA_PATH = DATA_DIR / "daily_feature_table.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"

VALIDATION_PREDICTIONS_PATH = DATA_DIR / "revenue_orders_signal_validation_predictions.csv"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "revenue_orders_signal_feature_importance.csv"
METRICS_PATH = LOG_DIR / "revenue_orders_signal_metrics.txt"
LOG_FILE = LOG_DIR / "train_revenue_with_orders_signal.log"

SUBMISSION_PATH = DATA_DIR / "submission_revenue_orders_signal.csv"

CURRENT_REFERENCE_RMSE = 921_000.0
CURRENT_STRONGER_SPIKE_RMSE = 842_278.60
VALIDATION_START = pd.Timestamp("2022-01-01")
VALIDATION_END = pd.Timestamp("2022-12-31")

ORDERS_SIGNAL_FEATURES = [
    "orders_spike_flag",
    "orders_momentum_flag",
    "orders_high_flag",
]


class Reporter:
    """Print, log, and persist a compact run report."""

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
        self.emit(frame.to_string(index=False))

    def save(self, path: Path = METRICS_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        self.logger.info("Saved report to %s", path)


def setup_logging(log_file: Path = LOG_FILE) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train_revenue_with_orders_signal")
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


def build_orders_predictions(
    revenue_df: pd.DataFrame,
    reporter: Reporter,
    logger: logging.Logger,
    sample_submission: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    orders_df = orders_mod.load_daily_feature_table(TRAIN_DATA_PATH)
    all_dates = pd.Series(pd.date_range(revenue_df["Date"].min(), sample_submission["Date"].max(), freq="D"))
    orders_static_all = pred_orders_mod.build_orders_static_features(all_dates, logger)
    orders_model_table = orders_mod.build_model_table(orders_df, logger)

    reporter.emit("Generate OOF predicted orders for training history")
    predicted_orders_oof = pred_orders_mod.generate_historical_oof_orders_predictions(
        train_df=orders_df,
        orders_model_table=orders_model_table,
        orders_static_features=orders_static_all,
        reporter=reporter,
    )

    reporter.emit("Generate predicted orders for validation 2022")
    validation_dates = orders_df[
        (orders_df["Date"] >= VALIDATION_START) & (orders_df["Date"] <= VALIDATION_END)
    ]["Date"]
    predicted_orders_2022 = pred_orders_mod.generate_orders_predictions_for_period(
        train_df=orders_df,
        orders_model_table=orders_model_table,
        orders_static_features=orders_static_all,
        train_end_exclusive=VALIDATION_START,
        prediction_dates=validation_dates,
        reporter=reporter,
        label="VALIDATION_2022",
    )

    reporter.emit("Generate predicted orders for future 2023-2024")
    predicted_orders_future = pred_orders_mod.generate_orders_predictions_for_period(
        train_df=orders_df,
        orders_model_table=orders_model_table,
        orders_static_features=orders_static_all,
        train_end_exclusive=sample_submission["Date"].min(),
        prediction_dates=sample_submission["Date"],
        reporter=reporter,
        label="FUTURE_2023_2024",
    )

    historical_pred_orders = (
        pd.concat([predicted_orders_oof, predicted_orders_2022], ignore_index=True)
        .sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )
    return historical_pred_orders, predicted_orders_future


def build_orders_signal_frame(
    dates: pd.Series,
    predicted_orders: pd.DataFrame,
    high_threshold: float,
) -> pd.DataFrame:
    frame = pd.DataFrame({"Date": pd.to_datetime(dates).sort_values().unique()})
    frame = frame.merge(predicted_orders, on="Date", how="left")
    frame["predicted_orders_count"] = pd.to_numeric(frame["predicted_orders_count"], errors="coerce")
    frame = frame.sort_values("Date").reset_index(drop=True)

    frame["orders_lag_1"] = frame["predicted_orders_count"].shift(1)
    frame["orders_lag_7"] = frame["predicted_orders_count"].shift(7)
    frame["orders_roll_mean_30"] = frame["predicted_orders_count"].shift(1).rolling(window=30, min_periods=30).mean()

    frame["orders_spike_flag"] = np.where(
        frame[["orders_lag_7", "orders_roll_mean_30"]].notna().all(axis=1),
        (frame["orders_lag_7"] > 1.2 * frame["orders_roll_mean_30"]).astype(int),
        np.nan,
    )
    frame["orders_momentum_flag"] = np.where(
        frame[["orders_lag_1", "orders_lag_7"]].notna().all(axis=1),
        (frame["orders_lag_1"] > frame["orders_lag_7"]).astype(int),
        np.nan,
    )
    frame["orders_high_flag"] = np.where(
        frame["orders_lag_7"].notna(),
        (frame["orders_lag_7"] > high_threshold).astype(int),
        np.nan,
    )
    return frame[["Date"] + ORDERS_SIGNAL_FEATURES]


def compute_high_threshold(predicted_orders: pd.DataFrame, train_end_exclusive: pd.Timestamp) -> float:
    train_values = predicted_orders[predicted_orders["Date"] < train_end_exclusive]["predicted_orders_count"]
    train_values = pd.to_numeric(train_values, errors="coerce").dropna()
    if train_values.empty:
        return 0.0
    return float(train_values.quantile(0.80))


def save_validation_predictions(
    dates: pd.Series,
    actual: pd.Series,
    predicted: np.ndarray,
    selected_model: str,
    path: Path = VALIDATION_PREDICTIONS_PATH,
) -> pd.DataFrame:
    actual_values = actual.to_numpy(dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    error = actual_values - predicted_values
    output = pd.DataFrame(
        {
            "Date": pd.to_datetime(dates).reset_index(drop=True),
            "actual_Revenue": actual_values,
            "predicted_Revenue": predicted_values,
            "selected_model": selected_model,
            "error": error,
            "abs_error": np.abs(error),
            "pct_error": np.where(actual_values != 0, error / actual_values, np.nan),
        }
    )
    output.to_csv(path, index=False)
    return output


def build_feature_importance_frame(
    pruned_trained: dict[str, Any],
    spike_trained: dict[str, Any],
) -> pd.DataFrame:
    pruned_importance = base.get_feature_importance(
        model=pruned_trained["model_object"],
        model_type=pruned_trained["model_type"],
        feature_columns=pruned_trained["feature_columns"],
        X_ref=pruned_trained["X_train"],
        y_ref=pruned_trained["y_train"],
        baseline_rmse=pruned_trained["metrics"]["RMSE"],
    ).copy()
    pruned_importance.insert(0, "model", "PRUNED_WITH_ORDERS_SIGNAL")

    spike_importance = base.get_feature_importance(
        model=spike_trained["model_object"],
        model_type=spike_trained["model_type"],
        feature_columns=spike_trained["feature_columns"],
        X_ref=spike_trained["X_train"],
        y_ref=spike_trained["y_train"],
        baseline_rmse=spike_trained["metrics"]["RMSE"],
    ).copy()
    spike_importance.insert(0, "model", "SPIKE_WITH_ORDERS_SIGNAL")

    return pd.concat([pruned_importance, spike_importance], ignore_index=True)


def run() -> None:
    logger = setup_logging()
    reporter = Reporter(logger)

    reporter.emit("Revenue Model With Orders-Derived Regime Signals")
    reporter.emit("==============================================")
    reporter.emit("")

    reporter.emit("1. Load base revenue data and sample submission")
    revenue_df = base.load_train_data(TRAIN_DATA_PATH)
    sample_submission = base.load_sample_submission(SAMPLE_SUBMISSION_PATH)
    all_dates = pd.Series(pd.date_range(revenue_df["Date"].min(), sample_submission["Date"].max(), freq="D"))
    reporter.emit(f"Revenue dataset shape: {revenue_df.shape}")
    reporter.emit(f"Sample submission rows: {len(sample_submission):,}")

    reporter.emit("")
    reporter.emit("2. Build predicted-orders timeline and derive regime flags")
    historical_pred_orders, predicted_orders_future = build_orders_predictions(
        revenue_df=revenue_df,
        reporter=reporter,
        logger=logger,
        sample_submission=sample_submission,
    )
    validation_threshold = compute_high_threshold(historical_pred_orders, VALIDATION_START)
    full_threshold = compute_high_threshold(historical_pred_orders, sample_submission["Date"].min())
    reporter.emit(f"Validation orders_high_flag p80 threshold: {validation_threshold:,.4f}")
    reporter.emit(f"Full-train orders_high_flag p80 threshold: {full_threshold:,.4f}")

    predicted_orders_validation_all = historical_pred_orders.copy()
    predicted_orders_full_all = (
        pd.concat([historical_pred_orders, predicted_orders_future], ignore_index=True)
        .sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )

    orders_signal_validation = build_orders_signal_frame(
        dates=all_dates,
        predicted_orders=predicted_orders_validation_all,
        high_threshold=validation_threshold,
    )
    orders_signal_full = build_orders_signal_frame(
        dates=all_dates,
        predicted_orders=predicted_orders_full_all,
        high_threshold=full_threshold,
    )

    reporter.emit_frame(
        "Orders signal availability (validation view):",
        orders_signal_validation[ORDERS_SIGNAL_FEATURES].isna().sum().rename("missing_count"),
    )

    reporter.emit("")
    reporter.emit("3. Build revenue feature tables with orders-derived flags")
    static_base = base.build_static_features(all_dates, revenue_df["Date"].min(), logger)
    revenue_static_validation = (
        static_base.merge(orders_signal_validation, on="Date", how="left", validate="one_to_one")
        .sort_values("Date")
        .reset_index(drop=True)
    )
    revenue_static_full = (
        static_base.merge(orders_signal_full, on="Date", how="left", validate="one_to_one")
        .sort_values("Date")
        .reset_index(drop=True)
    )

    pruned_model_table = base.build_historical_model_table(
        revenue_df,
        revenue_static_validation,
        include_business_lag365=False,
    )
    spike_model_table = spike1.build_spike_model_table(revenue_df, revenue_static_validation)

    pruned_feature_columns = spike1.deduplicate_preserve_order(
        [feature for feature in spike1.load_top_full_features(limit=50) if feature in pruned_model_table.columns]
        + [feature for feature in ORDERS_SIGNAL_FEATURES if feature in pruned_model_table.columns]
    )
    spike_feature_columns = spike1.deduplicate_preserve_order(
        [feature for feature in spike1.load_top_full_features(limit=50) if feature in spike_model_table.columns]
        + [feature for feature in spike1.SPIKE_FEATURES if feature in spike_model_table.columns]
        + [feature for feature in ORDERS_SIGNAL_FEATURES if feature in spike_model_table.columns]
    )

    reporter.emit(f"Pruned+orders-signal feature count: {len(pruned_feature_columns)}")
    reporter.emit(f"Spike+orders-signal feature count: {len(spike_feature_columns)}")

    reporter.emit("")
    reporter.emit("4. Recursive validation on 2022")
    pruned_result = spike1.validate_variant(
        variant_name="PRUNED_WITH_ORDERS_SIGNAL",
        model_table=pruned_model_table,
        static_features=revenue_static_validation,
        train_df=revenue_df,
        feature_columns=pruned_feature_columns,
        reporter=reporter,
        objective="regression",
    )
    spike_result = spike1.validate_variant(
        variant_name="SPIKE_WITH_ORDERS_SIGNAL",
        model_table=spike_model_table,
        static_features=revenue_static_validation,
        train_df=revenue_df,
        feature_columns=spike_feature_columns,
        reporter=reporter,
        objective="quantile",
        alpha=0.70,
    )

    comparison_df = pd.DataFrame(
        [
            {"model": "PRUNED_WITH_ORDERS_SIGNAL", **pruned_result["metrics"]},
            {"model": "SPIKE_WITH_ORDERS_SIGNAL", **spike_result["metrics"]},
        ]
    ).sort_values(["RMSE", "MAE"]).reset_index(drop=True)
    best_model_name = str(comparison_df.iloc[0]["model"])
    best_result = spike_result if best_model_name == "SPIKE_WITH_ORDERS_SIGNAL" else pruned_result

    reporter.emit_frame("Variant comparison:", comparison_df)
    reporter.emit(
        f"Best validation RMSE vs current best ~921k: {best_result['metrics']['RMSE']:,.2f} "
        f"(delta={best_result['metrics']['RMSE'] - CURRENT_REFERENCE_RMSE:,.2f})"
    )
    reporter.emit(
        f"Best validation RMSE vs stronger spike benchmark 842,278.60: "
        f"{best_result['metrics']['RMSE'] - CURRENT_STRONGER_SPIKE_RMSE:,.2f}"
    )

    save_validation_predictions(
        dates=best_result["validation_dates"],
        actual=best_result["actual"],
        predicted=best_result["predictions"],
        selected_model=best_model_name,
        path=VALIDATION_PREDICTIONS_PATH,
    )

    importance_df = build_feature_importance_frame(pruned_result, spike_result)
    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    reporter.emit_frame(
        f"Top 20 feature importances for {best_model_name}:",
        importance_df[importance_df["model"] == best_model_name]
        .sort_values(["importance_gain", "importance_split"], ascending=False)
        .head(20),
    )

    reporter.emit("")
    reporter.emit("5. Generate submission only if improved over current best (~921k)")
    if best_result["metrics"]["RMSE"] < CURRENT_REFERENCE_RMSE:
        pruned_model_table_full = base.build_historical_model_table(
            revenue_df,
            revenue_static_full,
            include_business_lag365=False,
        )
        spike_model_table_full = spike1.build_spike_model_table(revenue_df, revenue_static_full)

        final_pruned = spike1.train_full_variant(
            variant_name="PRUNED_WITH_ORDERS_SIGNAL",
            model_table=pruned_model_table_full,
            feature_columns=pruned_feature_columns,
            reporter=reporter,
            objective="regression",
        )
        final_spike = spike1.train_full_variant(
            variant_name="SPIKE_WITH_ORDERS_SIGNAL",
            model_table=spike_model_table_full,
            feature_columns=spike_feature_columns,
            reporter=reporter,
            objective="quantile",
            alpha=0.70,
        )

        final_best = final_spike if best_model_name == "SPIKE_WITH_ORDERS_SIGNAL" else final_pruned
        submission = spike1.forecast_variant_submission(
            trained=final_best,
            static_features=revenue_static_full,
            train_df=revenue_df,
            sample_submission=sample_submission,
            path=SUBMISSION_PATH,
        )
        checks = pred_orders_mod.validate_submission_frame(submission, sample_submission)
        reporter.emit(f"Submission created: {SUBMISSION_PATH} | checks={checks}")
    else:
        reporter.emit("No submission generated because validation RMSE did not beat ~921k reference.")

    reporter.emit("")
    reporter.emit("6. Final summary")
    reporter.emit_frame("Validation summary:", comparison_df)
    reporter.emit(f"Best model: {best_model_name}")
    reporter.emit(
        "Orders-derived regime signals used: orders_spike_flag, orders_momentum_flag, orders_high_flag"
    )
    reporter.emit(
        "Leakage confirmation: revenue model never uses true same-day orders_count. "
        "Signals come from lagged predicted orders generated by the orders model."
    )
    reporter.emit(f"Saved validation predictions: {VALIDATION_PREDICTIONS_PATH}")
    reporter.emit(f"Saved feature importance: {FEATURE_IMPORTANCE_PATH}")
    reporter.save(METRICS_PATH)


if __name__ == "__main__":
    run()
