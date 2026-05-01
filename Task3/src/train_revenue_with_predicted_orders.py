from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import train_final_model as base
import train_orders_model as orders_mod
import train_spike_aware_model as spike1


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "log"

TRAIN_DATA_PATH = DATA_DIR / "daily_feature_table.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"

PREDICTED_ORDERS_2022_PATH = DATA_DIR / "predicted_orders_2022.csv"
PREDICTED_ORDERS_FUTURE_PATH = DATA_DIR / "predicted_orders_future.csv"

VALIDATION_PREDICTIONS_PATH = DATA_DIR / "revenue_pred_orders_validation_predictions.csv"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "revenue_pred_orders_feature_importance.csv"
MODEL_COMPARISON_PATH = DATA_DIR / "revenue_pred_orders_model_comparison.csv"

SUBMISSION_PRUNED_PRED_ORDERS_PATH = DATA_DIR / "submission_revenue_pred_orders_pruned.csv"
SUBMISSION_SPIKE_PRED_ORDERS_PATH = DATA_DIR / "submission_revenue_pred_orders_spike.csv"
SUBMISSION_ENSEMBLE_PATH = DATA_DIR / "submission_revenue_pred_orders_ensemble.csv"

REPORT_PATH = LOG_DIR / "revenue_pred_orders_report.txt"
LOG_FILE = LOG_DIR / "train_revenue_with_predicted_orders.log"

CURRENT_PRUNED_METRICS = {
    "MAE": 669_832.08,
    "RMSE": 943_731.57,
    "R2": 0.682104,
}

CURRENT_SPIKE_METRICS = {
    "MAE": 623_974.93,
    "RMSE": 842_278.60,
    "R2": 0.746779,
}

PREDICTED_ORDER_FEATURES = [
    "predicted_orders_count",
    "predicted_orders_log1p",
    "predicted_orders_x_calendar_avg_discount_value",
    "predicted_orders_x_calendar_any_promo",
    "predicted_orders_x_day_of_week",
]

VALIDATION_START = pd.Timestamp("2022-01-01")
VALIDATION_END = pd.Timestamp("2022-12-31")


class Reporter:
    """Print, log, and persist run messages."""

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

    def save(self, path: Path = REPORT_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        self.logger.info("Saved revenue+predicted-orders report to %s", path)


def setup_logging(log_file: Path = LOG_FILE) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train_revenue_with_predicted_orders")
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


def build_orders_static_features(dates: pd.Series, logger: logging.Logger) -> pd.DataFrame:
    calendar = orders_mod.build_calendar_features(dates)
    promotions = orders_mod.build_promotion_features(dates, orders_mod.PROMOTIONS_PATH, logger)
    return (
        calendar.merge(promotions, on=orders_mod.DATE_COL, how="left", validate="one_to_one")
        .fillna(0)
        .sort_values(orders_mod.DATE_COL)
        .reset_index(drop=True)
    )


def generate_orders_predictions_for_period(
    train_df: pd.DataFrame,
    orders_model_table: pd.DataFrame,
    orders_static_features: pd.DataFrame,
    train_end_exclusive: pd.Timestamp,
    prediction_dates: pd.Series,
    reporter: Reporter,
    label: str,
) -> pd.DataFrame:
    X_train, y_train, _, feature_medians = orders_mod.make_training_matrix(
        orders_model_table,
        orders_mod.FEATURE_COLUMNS,
        train_end_exclusive,
    )
    reporter.emit(
        f"Orders model {label}: train rows={len(X_train):,}, "
        f"train end={(train_end_exclusive - pd.Timedelta(days=1)).date()}, "
        f"predict rows={len(prediction_dates):,}"
    )
    model, _ = orders_mod.train_model(X_train, y_train, reporter)
    initial_history = train_df[train_df[orders_mod.DATE_COL] < train_end_exclusive].set_index(orders_mod.DATE_COL)[
        orders_mod.TARGET_COL
    ]
    predictions = orders_mod.recursive_predict_orders(
        model=model,
        feature_columns=orders_mod.FEATURE_COLUMNS,
        static_features=orders_static_features,
        initial_history=initial_history,
        prediction_dates=prediction_dates,
        feature_medians=feature_medians,
    )
    output = pd.DataFrame(
        {
            "Date": pd.to_datetime(prediction_dates).reset_index(drop=True),
            "predicted_orders_count": predictions,
        }
    )
    return output


def generate_historical_oof_orders_predictions(
    train_df: pd.DataFrame,
    orders_model_table: pd.DataFrame,
    orders_static_features: pd.DataFrame,
    reporter: Reporter,
) -> pd.DataFrame:
    """Generate time-safe predicted orders for revenue-model training rows."""
    rows: list[pd.DataFrame] = []
    for year in range(2014, 2022):
        start = pd.Timestamp(f"{year}-01-01")
        end = pd.Timestamp(f"{year}-12-31")
        prediction_dates = train_df[
            (train_df[orders_mod.DATE_COL] >= start) & (train_df[orders_mod.DATE_COL] <= end)
        ][orders_mod.DATE_COL]
        if prediction_dates.empty:
            continue
        block = generate_orders_predictions_for_period(
            train_df=train_df,
            orders_model_table=orders_model_table,
            orders_static_features=orders_static_features,
            train_end_exclusive=start,
            prediction_dates=prediction_dates,
            reporter=reporter,
            label=f"OOF_{year}",
        )
        rows.append(block)
    if not rows:
        return pd.DataFrame(columns=["Date", "predicted_orders_count"])
    return pd.concat(rows, ignore_index=True).sort_values("Date").reset_index(drop=True)


def build_predicted_orders_feature_frame(
    static_features: pd.DataFrame,
    predicted_orders: pd.DataFrame,
) -> pd.DataFrame:
    output = static_features.merge(predicted_orders, on="Date", how="left")
    output["predicted_orders_count"] = pd.to_numeric(output["predicted_orders_count"], errors="coerce")
    output["predicted_orders_log1p"] = np.log1p(output["predicted_orders_count"].clip(lower=0))
    output["predicted_orders_x_calendar_avg_discount_value"] = (
        output["predicted_orders_count"] * output["calendar_avg_discount_value"]
    )
    output["predicted_orders_x_calendar_any_promo"] = (
        output["predicted_orders_count"] * output["calendar_any_promo"]
    )
    output["predicted_orders_x_day_of_week"] = output["predicted_orders_count"] * output["day_of_week"]
    return output


def load_existing_validation_predictions(
    path: Path,
    dates: pd.Series,
) -> tuple[np.ndarray, str] | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["Date"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    if "predicted_Revenue" not in df.columns:
        return None
    aligned = pd.DataFrame({"Date": pd.to_datetime(dates)}).merge(
        df[["Date", "predicted_Revenue"]],
        on="Date",
        how="left",
        validate="one_to_one",
    )
    if aligned["predicted_Revenue"].isna().any():
        return None
    return aligned["predicted_Revenue"].to_numpy(dtype=float), path.name


def build_feature_importance_frame(
    pruned_trained: dict[str, Any],
    pruned_validation_rmse: float,
    spike_trained: dict[str, Any],
    spike_validation_rmse: float,
) -> pd.DataFrame:
    pruned_importance = base.get_feature_importance(
        model=pruned_trained["model_object"],
        model_type=pruned_trained["model_type"],
        feature_columns=pruned_trained["feature_columns"],
        X_ref=pruned_trained["X_train"],
        y_ref=pruned_trained["y_train"],
        baseline_rmse=pruned_validation_rmse,
    ).copy()
    pruned_importance.insert(0, "model", "PRUNED_WITH_PRED_ORDERS")

    spike_importance = base.get_feature_importance(
        model=spike_trained["model_object"],
        model_type=spike_trained["model_type"],
        feature_columns=spike_trained["feature_columns"],
        X_ref=spike_trained["X_train"],
        y_ref=spike_trained["y_train"],
        baseline_rmse=spike_validation_rmse,
    ).copy()
    spike_importance.insert(0, "model", "SPIKE_WITH_PRED_ORDERS")

    return pd.concat([pruned_importance, spike_importance], ignore_index=True)


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


def blend_submissions(
    sample_submission: pd.DataFrame,
    submissions: dict[str, pd.DataFrame],
    weights: dict[str, float],
) -> pd.DataFrame:
    output = sample_submission[["Date"]].copy()
    output["Revenue"] = 0.0
    output["COGS"] = 0.0

    for model_name, weight in weights.items():
        if weight == 0:
            continue
        submission = submissions[model_name]
        output["Revenue"] = output["Revenue"] + weight * submission["Revenue"]
        output["COGS"] = output["COGS"] + weight * submission["COGS"]

    output["Revenue"] = output["Revenue"].clip(lower=0)
    output["COGS"] = output["COGS"].clip(lower=0)
    return output[["Date", "Revenue", "COGS"]]


def validate_submission_frame(submission: pd.DataFrame, sample_submission: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(submission)),
        "exact_columns": list(submission.columns) == ["Date", "Revenue", "COGS"],
        "date_order_matches": submission["Date"].reset_index(drop=True).equals(
            sample_submission["Date"].reset_index(drop=True)
        ),
        "missing_values": int(submission.isna().sum().sum()),
        "negative_values": int(
            ((pd.to_numeric(submission["Revenue"], errors="coerce") < 0) |
             (pd.to_numeric(submission["COGS"], errors="coerce") < 0)).sum()
        ),
    }


def run() -> None:
    logger = setup_logging()
    reporter = Reporter(logger)

    reporter.emit("Revenue Forecasting With Predicted Orders")
    reporter.emit("========================================")
    reporter.emit("")

    reporter.emit("1. Load base data")
    revenue_df = base.load_train_data(TRAIN_DATA_PATH)
    sample_submission = base.load_sample_submission(SAMPLE_SUBMISSION_PATH)
    orders_df = orders_mod.load_daily_feature_table(TRAIN_DATA_PATH)
    all_dates = pd.Series(
        pd.date_range(revenue_df["Date"].min(), sample_submission["Date"].max(), freq="D")
    )
    reporter.emit(f"Revenue dataset shape: {revenue_df.shape}")
    reporter.emit(f"Sample submission rows: {len(sample_submission):,}")

    reporter.emit("")
    reporter.emit("2. Orders model: build predicted orders feature")
    orders_static_all = build_orders_static_features(all_dates, logger)
    orders_model_table = orders_mod.build_model_table(orders_df, logger)

    historical_oof_orders = generate_historical_oof_orders_predictions(
        train_df=orders_df,
        orders_model_table=orders_model_table,
        orders_static_features=orders_static_all,
        reporter=reporter,
    )

    orders_2022_dates = orders_df[
        (orders_df["Date"] >= VALIDATION_START) & (orders_df["Date"] <= VALIDATION_END)
    ]["Date"]
    predicted_orders_2022 = generate_orders_predictions_for_period(
        train_df=orders_df,
        orders_model_table=orders_model_table,
        orders_static_features=orders_static_all,
        train_end_exclusive=VALIDATION_START,
        prediction_dates=orders_2022_dates,
        reporter=reporter,
        label="VALIDATION_2022",
    )
    predicted_orders_2022["actual_orders_count"] = (
        orders_df.set_index("Date").loc[predicted_orders_2022["Date"], "orders_count"].to_numpy(dtype=float)
    )
    predicted_orders_2022.to_csv(PREDICTED_ORDERS_2022_PATH, index=False)
    orders_validation_metrics = base.evaluate_predictions(
        pd.Series(predicted_orders_2022["actual_orders_count"]),
        predicted_orders_2022["predicted_orders_count"].to_numpy(dtype=float),
    )
    reporter.emit(
        f"Orders validation metrics: MAE={orders_validation_metrics['MAE']:,.4f} | "
        f"RMSE={orders_validation_metrics['RMSE']:,.4f} | R2={orders_validation_metrics['R2']:.6f}"
    )

    future_orders_dates = sample_submission["Date"]
    predicted_orders_future = generate_orders_predictions_for_period(
        train_df=orders_df,
        orders_model_table=orders_model_table,
        orders_static_features=orders_static_all,
        train_end_exclusive=pd.Timestamp("2023-01-01"),
        prediction_dates=future_orders_dates,
        reporter=reporter,
        label="FUTURE_2023_2024",
    )
    predicted_orders_future.to_csv(PREDICTED_ORDERS_FUTURE_PATH, index=False)
    reporter.emit(f"Saved predicted orders 2022: {PREDICTED_ORDERS_2022_PATH}")
    reporter.emit(f"Saved predicted orders future: {PREDICTED_ORDERS_FUTURE_PATH}")

    all_predicted_orders = pd.concat(
        [
            historical_oof_orders,
            predicted_orders_2022[["Date", "predicted_orders_count"]],
            predicted_orders_future,
        ],
        ignore_index=True,
    ).sort_values("Date")
    all_predicted_orders = all_predicted_orders.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

    reporter.emit("")
    reporter.emit("3. Build Revenue model tables with predicted orders")
    revenue_static_base = base.build_static_features(all_dates, revenue_df["Date"].min(), logger)
    revenue_static_with_pred_orders = build_predicted_orders_feature_frame(revenue_static_base, all_predicted_orders)
    pruned_model_table = base.build_historical_model_table(
        revenue_df,
        revenue_static_with_pred_orders,
        include_business_lag365=False,
    )
    spike_model_table = spike1.build_spike_model_table(revenue_df, revenue_static_with_pred_orders)

    pruned_feature_columns = spike1.deduplicate_preserve_order(
        [feature for feature in spike1.load_top_full_features(limit=50) if feature in pruned_model_table.columns]
        + [feature for feature in PREDICTED_ORDER_FEATURES if feature in pruned_model_table.columns]
    )
    spike_feature_columns = spike1.deduplicate_preserve_order(
        [feature for feature in spike1.load_top_full_features(limit=50) if feature in spike_model_table.columns]
        + [feature for feature in spike1.SPIKE_FEATURES if feature in spike_model_table.columns]
        + [feature for feature in PREDICTED_ORDER_FEATURES if feature in spike_model_table.columns]
    )
    reporter.emit(f"Pruned+predicted-orders feature count: {len(pruned_feature_columns)}")
    reporter.emit(f"Spike+predicted-orders feature count: {len(spike_feature_columns)}")

    reporter.emit("")
    reporter.emit("4. Recursive Revenue validation on 2022")
    pruned_result = spike1.validate_variant(
        variant_name="PRUNED_WITH_PRED_ORDERS",
        model_table=pruned_model_table,
        static_features=revenue_static_with_pred_orders,
        train_df=revenue_df,
        feature_columns=pruned_feature_columns,
        reporter=reporter,
        objective="regression",
    )
    spike_result = spike1.validate_variant(
        variant_name="SPIKE_WITH_PRED_ORDERS",
        model_table=spike_model_table,
        static_features=revenue_static_with_pred_orders,
        train_df=revenue_df,
        feature_columns=spike_feature_columns,
        reporter=reporter,
        objective="quantile",
        alpha=0.70,
    )
    reporter.emit(
        f"Revenue Variant A metrics: MAE={pruned_result['metrics']['MAE']:,.2f} | "
        f"RMSE={pruned_result['metrics']['RMSE']:,.2f} | R2={pruned_result['metrics']['R2']:.6f}"
    )
    reporter.emit(
        f"Revenue Variant B metrics: MAE={spike_result['metrics']['MAE']:,.2f} | "
        f"RMSE={spike_result['metrics']['RMSE']:,.2f} | R2={spike_result['metrics']['R2']:.6f}"
    )

    actual_validation = pruned_result["actual"]
    validation_dates = pruned_result["validation_dates"]
    candidate_predictions = {
        "PRUNED_WITH_PRED_ORDERS": pruned_result["predictions"],
        "SPIKE_WITH_PRED_ORDERS": spike_result["predictions"],
    }

    existing_pruned = load_existing_validation_predictions(
        DATA_DIR / "pruned_ensemble_validation_predictions.csv",
        validation_dates,
    )
    if existing_pruned is not None:
        candidate_predictions["CURRENT_PRUNED_ENSEMBLE"] = existing_pruned[0]

    existing_spike = load_existing_validation_predictions(
        DATA_DIR / "spike_model_validation_predictions.csv",
        validation_dates,
    )
    if existing_spike is not None:
        candidate_predictions["CURRENT_SPIKE_MODEL"] = existing_spike[0]

    ensemble_search = spike1.evaluate_ensemble_candidates(actual_validation, candidate_predictions)
    best_ensemble_row = ensemble_search.iloc[0].to_dict()
    best_weights = {
        model_name: float(best_ensemble_row.get(f"weight_{model_name}", 0.0))
        for model_name in candidate_predictions
        if float(best_ensemble_row.get(f"weight_{model_name}", 0.0)) > 0
    }
    ensemble_predictions = np.zeros(len(actual_validation), dtype=float)
    for model_name, weight in best_weights.items():
        ensemble_predictions = ensemble_predictions + weight * candidate_predictions[model_name]
    ensemble_metrics = spike1.evaluate_candidate(
        "REVENUE_PRED_ORDERS_ENSEMBLE",
        actual_validation,
        ensemble_predictions,
    )

    comparison_rows = [
        {
            "model": "CURRENT_PRUNED_ENSEMBLE",
            **CURRENT_PRUNED_METRICS,
            "top10_RMSE": np.nan,
            "top10_mean_error": np.nan,
            "top10_underprediction": np.nan,
            "top10_count": np.nan,
            "top5_RMSE": np.nan,
            "top5_underprediction": np.nan,
            "top5_count": np.nan,
        },
        {
            "model": "CURRENT_SPIKE_MODEL",
            **CURRENT_SPIKE_METRICS,
            "top10_RMSE": np.nan,
            "top10_mean_error": np.nan,
            "top10_underprediction": np.nan,
            "top10_count": np.nan,
            "top5_RMSE": np.nan,
            "top5_underprediction": np.nan,
            "top5_count": np.nan,
        },
        pruned_result["metrics"],
        spike_result["metrics"],
        ensemble_metrics,
    ]
    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        ["RMSE", "MAE"],
        ascending=[True, True],
    )
    comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)

    reporter.emit(
        "Best ensemble weights: "
        + ", ".join(f"{name}={weight:.2f}" for name, weight in best_weights.items())
    )
    reporter.emit(
        f"Best validation metrics: MAE={ensemble_metrics['MAE']:,.2f} | "
        f"RMSE={ensemble_metrics['RMSE']:,.2f} | R2={ensemble_metrics['R2']:.6f}"
    )
    reporter.emit(
        f"Spike-day metrics: top10_RMSE={ensemble_metrics['top10_RMSE']:,.2f} | "
        f"top10_underprediction={ensemble_metrics['top10_underprediction']}/{ensemble_metrics['top10_count']}"
    )

    save_validation_predictions(
        dates=validation_dates,
        actual=actual_validation,
        predicted=ensemble_predictions,
        selected_model="REVENUE_PRED_ORDERS_ENSEMBLE",
        path=VALIDATION_PREDICTIONS_PATH,
    )

    reporter.emit("")
    reporter.emit("5. Retrain Revenue variants on all available rows and generate submissions")
    final_pruned = spike1.train_full_variant(
        variant_name="PRUNED_WITH_PRED_ORDERS",
        model_table=pruned_model_table,
        feature_columns=pruned_feature_columns,
        reporter=reporter,
        objective="regression",
    )
    final_spike = spike1.train_full_variant(
        variant_name="SPIKE_WITH_PRED_ORDERS",
        model_table=spike_model_table,
        feature_columns=spike_feature_columns,
        reporter=reporter,
        objective="quantile",
        alpha=0.70,
    )

    submission_pruned = spike1.forecast_variant_submission(
        trained=final_pruned,
        static_features=revenue_static_with_pred_orders,
        train_df=revenue_df,
        sample_submission=sample_submission,
        path=SUBMISSION_PRUNED_PRED_ORDERS_PATH,
    )
    submission_spike = spike1.forecast_variant_submission(
        trained=final_spike,
        static_features=revenue_static_with_pred_orders,
        train_df=revenue_df,
        sample_submission=sample_submission,
        path=SUBMISSION_SPIKE_PRED_ORDERS_PATH,
    )

    ensemble_submissions = {
        "PRUNED_WITH_PRED_ORDERS": submission_pruned,
        "SPIKE_WITH_PRED_ORDERS": submission_spike,
    }
    if (DATA_DIR / "submission_pruned_ensemble.csv").exists():
        ensemble_submissions["CURRENT_PRUNED_ENSEMBLE"] = spike1.load_submission(
            DATA_DIR / "submission_pruned_ensemble.csv",
            sample_submission,
        )
    if (DATA_DIR / "submission_spike_aware.csv").exists():
        ensemble_submissions["CURRENT_SPIKE_MODEL"] = spike1.load_submission(
            DATA_DIR / "submission_spike_aware.csv",
            sample_submission,
        )

    submission_ensemble = blend_submissions(
        sample_submission=sample_submission,
        submissions=ensemble_submissions,
        weights=best_weights,
    )
    submission_ensemble.to_csv(SUBMISSION_ENSEMBLE_PATH, index=False)

    importance_df = build_feature_importance_frame(
        pruned_trained=final_pruned,
        pruned_validation_rmse=pruned_result["metrics"]["RMSE"],
        spike_trained=final_spike,
        spike_validation_rmse=spike_result["metrics"]["RMSE"],
    )
    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    reporter.emit("")
    reporter.emit("6. Final summary")
    reporter.emit_frame("Model comparison:", comparison_df)
    top30_source = "SPIKE_WITH_PRED_ORDERS" if spike_result["metrics"]["RMSE"] <= pruned_result["metrics"]["RMSE"] else "PRUNED_WITH_PRED_ORDERS"
    reporter.emit_frame(
        f"Top 30 feature importances for {top30_source}:",
        importance_df[importance_df["model"] == top30_source].head(30),
    )
    reporter.emit(
        "Predicted orders improved Revenue forecasting: "
        f"{ensemble_metrics['RMSE'] < CURRENT_SPIKE_METRICS['RMSE'] or ensemble_metrics['RMSE'] < CURRENT_PRUNED_METRICS['RMSE']}"
    )
    reporter.emit(
        "Submission checks: "
        f"pruned={validate_submission_frame(submission_pruned, sample_submission)}, "
        f"spike={validate_submission_frame(submission_spike, sample_submission)}, "
        f"ensemble={validate_submission_frame(submission_ensemble, sample_submission)}"
    )
    reporter.emit(f"Saved validation predictions: {VALIDATION_PREDICTIONS_PATH}")
    reporter.emit(f"Saved feature importance: {FEATURE_IMPORTANCE_PATH}")
    reporter.emit(f"Saved model comparison: {MODEL_COMPARISON_PATH}")
    reporter.emit(
        f"Created submission files: {SUBMISSION_PRUNED_PRED_ORDERS_PATH.name}, "
        f"{SUBMISSION_SPIKE_PRED_ORDERS_PATH.name}, {SUBMISSION_ENSEMBLE_PATH.name}"
    )
    reporter.emit(
        "Leakage confirmation: no true same-day orders_count was used in the Revenue model. "
        "Revenue training/validation/future runs use only time-safe predicted orders."
    )

    reporter.save(REPORT_PATH)


if __name__ == "__main__":
    run()
