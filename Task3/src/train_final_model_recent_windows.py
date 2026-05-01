from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import train_final_model as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "log"

VALIDATION_RESULTS_PATH = DATA_DIR / "recent_window_validation_results.csv"
VALIDATION_PREDICTION_PATHS = {
    "FULL": DATA_DIR / "recent_full_validation_predictions.csv",
    "RECENT_2015": DATA_DIR / "recent_2015_validation_predictions.csv",
    "RECENT_2019": DATA_DIR / "recent_2019_validation_predictions.csv",
}
FEATURE_IMPORTANCE_PATHS = {
    "FULL": DATA_DIR / "recent_window_feature_importance_full.csv",
    "RECENT_2015": DATA_DIR / "recent_window_feature_importance_2015.csv",
    "RECENT_2019": DATA_DIR / "recent_window_feature_importance_2019.csv",
}
SUBMISSION_PATHS = {
    "FULL": DATA_DIR / "submission_full_window.csv",
    "RECENT_2015": DATA_DIR / "submission_2015_window.csv",
    "RECENT_2019": DATA_DIR / "submission_2019_window.csv",
}
BEST_SUBMISSION_PATH = DATA_DIR / "submission_recent_window_best.csv"
METRICS_PATH = LOG_DIR / "recent_window_metrics.txt"
LOG_FILE = LOG_DIR / "train_final_model_recent_windows.log"

PREVIOUS_FINAL_MODEL_A = {
    "MAE": 695_337.54,
    "RMSE": 985_315.32,
    "R2": 0.653472,
}

WINDOW_CONFIGS = [
    {
        "name": "FULL",
        "train_start": pd.Timestamp("2012-07-04"),
        "submission_path": SUBMISSION_PATHS["FULL"],
        "feature_importance_path": FEATURE_IMPORTANCE_PATHS["FULL"],
    },
    {
        "name": "RECENT_2015",
        "train_start": pd.Timestamp("2015-01-01"),
        "submission_path": SUBMISSION_PATHS["RECENT_2015"],
        "feature_importance_path": FEATURE_IMPORTANCE_PATHS["RECENT_2015"],
    },
    {
        "name": "RECENT_2019",
        "train_start": pd.Timestamp("2019-01-01"),
        "submission_path": SUBMISSION_PATHS["RECENT_2019"],
        "feature_importance_path": FEATURE_IMPORTANCE_PATHS["RECENT_2019"],
    },
]

FEATURE_COLUMNS = base.MODEL_A_FEATURES.copy()


class RunReporter:
    """Print, log, and save all experiment messages."""

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

    def save_metrics(self, path: Path = METRICS_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        self.logger.info("Saved recent-window metrics report to %s", path)


def setup_logging(log_file: Path = LOG_FILE) -> logging.Logger:
    """Configure simple file logging."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train_final_model_recent_windows")
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


def make_window_training_matrix(
    model_table: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end_exclusive: pd.Timestamp | None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Create a clean training matrix for a specific target-row window."""
    table = model_table[model_table[base.DATE_COL] >= train_start].copy()
    if train_end_exclusive is not None:
        table = table[table[base.DATE_COL] < train_end_exclusive].copy()

    missing_features = [column for column in FEATURE_COLUMNS if column not in table.columns]
    if missing_features:
        raise ValueError(f"Missing expected feature columns: {missing_features}")

    clean = table.dropna(subset=FEATURE_COLUMNS + [base.TARGET_COL]).reset_index(drop=True)
    X = clean[FEATURE_COLUMNS].copy()
    y = clean[base.TARGET_COL].copy()
    feature_medians = X.median(numeric_only=True)
    return X, y, clean, feature_medians


def train_window_model(
    window_name: str,
    train_start: pd.Timestamp,
    model_table: pd.DataFrame,
    train_end_exclusive: pd.Timestamp | None,
    reporter: RunReporter,
) -> dict[str, Any]:
    """Train one recent-window model."""
    X_train, y_train, clean, feature_medians = make_window_training_matrix(
        model_table,
        train_start=train_start,
        train_end_exclusive=train_end_exclusive,
    )
    date_end = (
        (train_end_exclusive - pd.Timedelta(days=1)).date()
        if train_end_exclusive is not None
        else clean[base.DATE_COL].max().date()
    )
    reporter.emit(
        f"Training {window_name}: rows={len(X_train):,}, features={len(FEATURE_COLUMNS)}, "
        f"target rows {train_start.date()} -> {date_end}"
    )
    model, model_type = base.train_model(X_train, y_train, reporter)
    return {
        "window_name": window_name,
        "train_start": train_start,
        "model": model,
        "model_type": model_type,
        "feature_columns": FEATURE_COLUMNS,
        "feature_medians": feature_medians,
        "X_train": X_train,
        "y_train": y_train,
        "clean": clean,
    }


def validate_window_model(
    trained: dict[str, Any],
    static_features: pd.DataFrame,
    train_df: pd.DataFrame,
) -> dict[str, Any]:
    """Recursively validate one trained window model on 2022."""
    validation_dates = train_df[
        (train_df[base.DATE_COL] >= base.TRAIN_CUTOFF)
        & (train_df[base.DATE_COL] <= base.VALIDATION_END)
    ][base.DATE_COL]
    actual = train_df.set_index(base.DATE_COL).loc[validation_dates, base.TARGET_COL]
    initial_history = train_df[train_df[base.DATE_COL] < base.TRAIN_CUTOFF].set_index(base.DATE_COL)[
        base.TARGET_COL
    ]

    predictions = base.recursive_predict(
        model=trained["model"],
        model_type=trained["model_type"],
        prediction_dates=validation_dates,
        feature_columns=trained["feature_columns"],
        static_features=static_features,
        initial_revenue_history=initial_history,
        business_maps={},
        feature_medians=trained["feature_medians"],
        include_business_lag365=False,
    )
    metrics = base.evaluate_predictions(actual, predictions)

    return {
        "window_name": trained["window_name"],
        "train_start": trained["train_start"],
        "validation_dates": validation_dates.reset_index(drop=True),
        "actual": actual.reset_index(drop=True),
        "predictions": predictions,
        "metrics": metrics,
    }


def save_feature_importance(
    trained: dict[str, Any],
    validation_rmse: float,
    path: Path,
) -> pd.DataFrame:
    """Save feature importance for one model."""
    importance = base.get_feature_importance(
        trained["model"],
        trained["model_type"],
        trained["feature_columns"],
        trained["X_train"],
        trained["y_train"],
        validation_rmse,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(path, index=False)
    return importance


def estimate_window_cogs_ratio(train_df: pd.DataFrame, train_start: pd.Timestamp) -> float:
    """Estimate COGS/Revenue ratio from the latest 365 days inside the window."""
    window = train_df[train_df[base.DATE_COL] >= train_start].sort_values(base.DATE_COL).copy()
    recent = window.tail(365)
    ratio = recent[base.COGS_COL].sum() / recent[base.TARGET_COL].sum()
    if not np.isfinite(ratio) or ratio <= 0:
        ratio = window[base.COGS_COL].sum() / window[base.TARGET_COL].sum()
    return float(np.clip(ratio, 0.0, 2.0))


def build_submission(
    sample_submission: pd.DataFrame,
    revenue_predictions: np.ndarray,
    cogs_ratio: float,
    path: Path,
) -> pd.DataFrame:
    """Create a window-specific submission without touching data/submission.csv."""
    submission = sample_submission[[base.DATE_COL]].copy()
    submission[base.TARGET_COL] = np.maximum(0.0, revenue_predictions)
    submission[base.COGS_COL] = np.maximum(0.0, submission[base.TARGET_COL] * cogs_ratio)
    submission = submission[[base.DATE_COL, base.TARGET_COL, base.COGS_COL]]
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    return submission


def forecast_submission_for_window(
    trained: dict[str, Any],
    static_features: pd.DataFrame,
    train_df: pd.DataFrame,
    sample_submission: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Generate recursive 2023-2024 forecast for one trained final window model."""
    initial_history = train_df.set_index(base.DATE_COL)[base.TARGET_COL].sort_index()
    predictions = base.recursive_predict(
        model=trained["model"],
        model_type=trained["model_type"],
        prediction_dates=sample_submission[base.DATE_COL],
        feature_columns=trained["feature_columns"],
        static_features=static_features,
        initial_revenue_history=initial_history,
        business_maps={},
        feature_medians=trained["feature_medians"],
        include_business_lag365=False,
    )
    cogs_ratio = estimate_window_cogs_ratio(train_df, trained["train_start"])
    return build_submission(sample_submission, predictions, cogs_ratio, output_path)


def validation_results_frame(validation_results: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a concise comparison table."""
    rows = []
    for result in validation_results:
        metrics = result["metrics"]
        rows.append(
            {
                "model": result["window_name"],
                "train_start": result["train_start"].date().isoformat(),
                "validation_start": base.TRAIN_CUTOFF.date().isoformat(),
                "validation_end": base.VALIDATION_END.date().isoformat(),
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
            }
        )
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def save_window_validation_predictions(result: dict[str, Any], path: Path) -> pd.DataFrame:
    """Save per-date recursive validation predictions for one window model."""
    output = pd.DataFrame(
        {
            "Date": result["validation_dates"],
            "actual_Revenue": result["actual"],
            "predicted_Revenue": result["predictions"],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    return output


def metric_diff_vs_previous(best_metrics: dict[str, float]) -> dict[str, float]:
    """Return differences from the previous final Model A."""
    return {
        "MAE_change": best_metrics["MAE"] - PREVIOUS_FINAL_MODEL_A["MAE"],
        "RMSE_change": best_metrics["RMSE"] - PREVIOUS_FINAL_MODEL_A["RMSE"],
        "R2_change": best_metrics["R2"] - PREVIOUS_FINAL_MODEL_A["R2"],
    }


def run_training() -> None:
    logger = setup_logging()
    reporter = RunReporter(logger)

    reporter.emit("Final Model Recent-Window Experiment")
    reporter.emit("====================================")
    reporter.emit("")
    reporter.emit("1. Load data")

    train_df = base.load_train_data(base.TRAIN_DATA_PATH)
    sample_submission = base.load_sample_submission(base.SAMPLE_SUBMISSION_PATH)
    all_dates = pd.Series(
        pd.date_range(train_df[base.DATE_COL].min(), sample_submission[base.DATE_COL].max(), freq="D")
    )
    static_features = base.build_static_features(all_dates, train_df[base.DATE_COL].min(), logger)
    model_table = base.build_historical_model_table(
        train_df,
        static_features,
        include_business_lag365=False,
    )

    reporter.emit(f"Train data shape: {train_df.shape}")
    reporter.emit(
        f"Train date range: {train_df[base.DATE_COL].min().date()} -> "
        f"{train_df[base.DATE_COL].max().date()}"
    )
    reporter.emit(
        f"Forecast date range: {sample_submission[base.DATE_COL].min().date()} -> "
        f"{sample_submission[base.DATE_COL].max().date()}"
    )
    reporter.emit(f"Static feature table shape: {static_features.shape}")
    reporter.emit(f"Forecast-safe Model A feature count: {len(FEATURE_COLUMNS)}")
    reporter.emit(
        "Blocked same-day realized features: "
        + str([column for column in base.UNSAFE_SAME_DAY_COLUMNS if column in train_df.columns])
    )

    reporter.emit("")
    reporter.emit("2. Recursive validation on 2022")
    validation_results: list[dict[str, Any]] = []
    validation_trained_models: dict[str, dict[str, Any]] = {}
    validation_importances: dict[str, pd.DataFrame] = {}

    for config in WINDOW_CONFIGS:
        trained = train_window_model(
            window_name=config["name"],
            train_start=config["train_start"],
            model_table=model_table,
            train_end_exclusive=base.TRAIN_CUTOFF,
            reporter=reporter,
        )
        result = validate_window_model(trained, static_features, train_df)
        validation_results.append(result)
        prediction_path = VALIDATION_PREDICTION_PATHS[config["name"]]
        save_window_validation_predictions(result, prediction_path)
        validation_trained_models[config["name"]] = trained
        importance = save_feature_importance(
            trained,
            result["metrics"]["RMSE"],
            config["feature_importance_path"],
        )
        validation_importances[config["name"]] = importance
        reporter.emit(
            f"{config['name']} metrics: MAE={result['metrics']['MAE']:,.2f}, "
            f"RMSE={result['metrics']['RMSE']:,.2f}, R2={result['metrics']['R2']:.6f}"
        )
        reporter.emit(f"Saved {config['name']} validation predictions: {prediction_path}")

    results_df = validation_results_frame(validation_results)
    VALIDATION_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(VALIDATION_RESULTS_PATH, index=False)

    reporter.emit("")
    reporter.emit_frame("Validation result table:", results_df)
    best_name = str(results_df.iloc[0]["model"])
    best_result = next(result for result in validation_results if result["window_name"] == best_name)
    diff = metric_diff_vs_previous(best_result["metrics"])
    reporter.emit(f"Best model selected by RMSE: {best_name}")
    reporter.emit(
        "Metric difference vs previous final Model A: "
        f"MAE={diff['MAE_change']:,.2f}, RMSE={diff['RMSE_change']:,.2f}, "
        f"R2={diff['R2_change']:.6f}"
    )

    reporter.emit("")
    reporter.emit("3. Retrain full-window variants through 2022 and create submissions")
    final_trained_models: dict[str, dict[str, Any]] = {}
    submissions: dict[str, pd.DataFrame] = {}
    final_importances: dict[str, pd.DataFrame] = {}

    for config in WINDOW_CONFIGS:
        trained = train_window_model(
            window_name=config["name"],
            train_start=config["train_start"],
            model_table=model_table,
            train_end_exclusive=None,
            reporter=reporter,
        )
        final_trained_models[config["name"]] = trained
        submissions[config["name"]] = forecast_submission_for_window(
            trained,
            static_features,
            train_df,
            sample_submission,
            config["submission_path"],
        )
        final_importance = save_feature_importance(
            trained,
            next(result for result in validation_results if result["window_name"] == config["name"])["metrics"][
                "RMSE"
            ],
            config["feature_importance_path"],
        )
        final_importances[config["name"]] = final_importance
        reporter.emit(f"Saved {config['name']} submission: {config['submission_path']}")
        reporter.emit(f"Saved {config['name']} feature importance: {config['feature_importance_path']}")

    best_submission_source = submissions[best_name]
    best_submission_source.to_csv(BEST_SUBMISSION_PATH, index=False)
    reporter.emit(f"Saved selected best submission: {BEST_SUBMISSION_PATH}")

    reporter.emit("")
    reporter.emit("4. Top feature importances")
    for name in ["FULL", "RECENT_2015", "RECENT_2019"]:
        reporter.emit_frame(f"Top 20 feature importances - {name}:", final_importances[name].head(20))

    hypothesis_supported = best_name != "FULL"
    reporter.emit("")
    reporter.emit("5. Final summary")
    reporter.emit_frame("Validation result table:", results_df)
    reporter.emit(f"Best model selected by RMSE: {best_name}")
    reporter.emit(
        "Metric difference vs previous final Model A: "
        f"MAE={diff['MAE_change']:,.2f}, RMSE={diff['RMSE_change']:,.2f}, "
        f"R2={diff['R2_change']:.6f}"
    )
    reporter.emit(f"Submission file - FULL: {SUBMISSION_PATHS['FULL']}")
    reporter.emit(f"Submission file - RECENT_2015: {SUBMISSION_PATHS['RECENT_2015']}")
    reporter.emit(f"Submission file - RECENT_2019: {SUBMISSION_PATHS['RECENT_2019']}")
    reporter.emit(f"Selected best submission: {BEST_SUBMISSION_PATH}")
    reporter.emit(
        "Leakage confirmation: no same-day realized demand, future Revenue, or future COGS was used. "
        "Validation and submission forecasts are recursive; COGS uses only historical COGS/Revenue ratios."
    )
    reporter.emit(
        "Recommendation: "
        + (
            f"hypothesis supported by validation; use {best_name} recent-window submission."
            if hypothesis_supported
            else "hypothesis rejected by validation; FULL history remains best by RMSE."
        )
    )

    reporter.save_metrics(METRICS_PATH)


if __name__ == "__main__":
    run_training()
