from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import train_final_model as base
import train_final_model_promo_duration as promo_duration


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "log"

FULL_IMPORTANCE_PATH = DATA_DIR / "final_feature_importance.csv"
PROMO_IMPORTANCE_PATH = DATA_DIR / "final_promo_duration_feature_importance.csv"
PRUNED_FEATURE_SETS_PATH = DATA_DIR / "pruned_feature_sets.csv"
METRICS_PATH = LOG_DIR / "final_prune_metrics.txt"
LOG_FILE = LOG_DIR / "final_feature_prune_and_retrain.log"

SUBMISSION_PRUNED_FULL_PATH = DATA_DIR / "submission_pruned_full.csv"
SUBMISSION_PRUNED_PROMO_PATH = DATA_DIR / "submission_pruned_promo.csv"
SUBMISSION_PRUNED_ENSEMBLE_PATH = DATA_DIR / "submission_pruned_ensemble.csv"
PRUNED_ENSEMBLE_VALIDATION_PATH = DATA_DIR / "pruned_ensemble_validation_predictions.csv"

BASELINE_FULL_RMSE = 985_315.32
BASELINE_ENSEMBLE_RMSE = 967_539.78
FEATURE_SET_SIZES = [30, 40, 50]
WEIGHT_STEP = 0.05


class Reporter:
    """Print, log, and persist experiment messages."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self.lines: list[str] = []

    def emit(self, message: str = "") -> None:
        print(message)
        self.lines.append(message)
        if message:
            self.logger.info(message)

    def emit_frame(self, title: str, frame: pd.DataFrame) -> None:
        self.emit(title)
        if frame.empty:
            self.emit("(empty)")
            return
        self.emit(frame.to_string(index=False))

    def save(self, path: Path = METRICS_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        self.logger.info("Saved pruning metrics to %s", path)


def setup_logging(log_file: Path = LOG_FILE) -> logging.Logger:
    """Configure simple file logging."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("final_feature_prune_and_retrain")
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


def load_ranked_features(path: Path) -> list[str]:
    """Load feature importance sorted by gain."""
    if not path.exists():
        raise FileNotFoundError(f"Feature importance file not found: {path}")

    importance = pd.read_csv(path)
    required = {"feature", "importance_gain"}
    if not required.issubset(importance.columns):
        raise ValueError(f"{path} must contain columns: {required}")

    importance["importance_gain"] = pd.to_numeric(importance["importance_gain"], errors="coerce").fillna(0)
    return (
        importance.sort_values("importance_gain", ascending=False)["feature"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )


def filter_existing_features(features: list[str], table: pd.DataFrame) -> list[str]:
    """Keep only model-safe columns present in the model table."""
    blocked = {base.DATE_COL, base.TARGET_COL, base.COGS_COL}
    blocked.update(base.UNSAFE_SAME_DAY_COLUMNS)
    return [feature for feature in features if feature in table.columns and feature not in blocked]


def make_training_matrix(
    model_table: pd.DataFrame,
    feature_columns: list[str],
    train_end_exclusive: pd.Timestamp | None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Create clean matrix for selected features."""
    table = model_table.copy()
    if train_end_exclusive is not None:
        table = table[table[base.DATE_COL] < train_end_exclusive].copy()

    missing = [column for column in feature_columns if column not in table.columns]
    if missing:
        raise ValueError(f"Missing selected feature columns: {missing}")

    clean = table.dropna(subset=feature_columns + [base.TARGET_COL]).reset_index(drop=True)
    X = clean[feature_columns].copy()
    y = clean[base.TARGET_COL].copy()
    medians = X.median(numeric_only=True)
    return X, y, clean, medians


def train_selected_features(
    model_name: str,
    model_table: pd.DataFrame,
    feature_columns: list[str],
    train_end_exclusive: pd.Timestamp | None,
    reporter: Reporter,
) -> dict[str, Any]:
    """Train one pruned model."""
    X_train, y_train, clean, medians = make_training_matrix(
        model_table,
        feature_columns,
        train_end_exclusive,
    )
    period_end = (
        (train_end_exclusive - pd.Timedelta(days=1)).date()
        if train_end_exclusive is not None
        else clean[base.DATE_COL].max().date()
    )
    reporter.emit(
        f"Training {model_name}: rows={len(X_train):,}, features={len(feature_columns)}, "
        f"date range={clean[base.DATE_COL].min().date()} -> {period_end}"
    )
    model, model_type = base.train_model(X_train, y_train, reporter)
    return {
        "model_name": model_name,
        "model": model,
        "model_type": model_type,
        "feature_columns": feature_columns,
        "feature_medians": medians,
        "X_train": X_train,
        "y_train": y_train,
        "clean": clean,
    }


def recursive_validate(
    trained: dict[str, Any],
    static_features: pd.DataFrame,
    train_df: pd.DataFrame,
) -> dict[str, Any]:
    """Validate recursively on 2022."""
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
        "dates": validation_dates.reset_index(drop=True),
        "actual": actual.reset_index(drop=True),
        "predictions": predictions,
        "metrics": metrics,
    }


def grid_search_two_model_ensemble(
    actual: pd.Series,
    full_pred: np.ndarray,
    promo_pred: np.ndarray,
) -> pd.DataFrame:
    """Optimize two-model ensemble weights by validation RMSE."""
    rows: list[dict[str, float]] = []
    units = int(round(1.0 / WEIGHT_STEP))
    for unit in range(units + 1):
        full_weight = unit * WEIGHT_STEP
        promo_weight = 1.0 - full_weight
        blended = full_weight * full_pred + promo_weight * promo_pred
        metrics = base.evaluate_predictions(actual, blended)
        rows.append(
            {
                "weight_FULL_pruned": full_weight,
                "weight_PROMO_pruned": promo_weight,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
            }
        )
    return pd.DataFrame(rows).sort_values(["RMSE", "MAE"]).reset_index(drop=True)


def build_feature_set_report(
    full_sets: dict[str, list[str]],
    promo_sets: dict[str, list[str]],
) -> pd.DataFrame:
    """Save feature sets in long format."""
    rows: list[dict[str, str | int]] = []
    for set_name, features in full_sets.items():
        for rank, feature in enumerate(features, start=1):
            rows.append({"feature_set": set_name, "model": "FULL", "rank": rank, "feature": feature})
    for set_name, features in promo_sets.items():
        for rank, feature in enumerate(features, start=1):
            rows.append({"feature_set": set_name, "model": "PROMO_DURATION", "rank": rank, "feature": feature})
    return pd.DataFrame(rows)


def build_submission(
    sample_submission: pd.DataFrame,
    predictions: np.ndarray,
    cogs_ratio: float,
    path: Path,
) -> pd.DataFrame:
    """Create a submission with exact sample schema."""
    output = sample_submission[[base.DATE_COL]].copy()
    output[base.TARGET_COL] = np.maximum(0.0, predictions)
    output[base.COGS_COL] = np.maximum(0.0, output[base.TARGET_COL] * cogs_ratio)
    output = output[[base.DATE_COL, base.TARGET_COL, base.COGS_COL]]
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    return output


def forecast_submission(
    trained: dict[str, Any],
    static_features: pd.DataFrame,
    train_df: pd.DataFrame,
    sample_submission: pd.DataFrame,
    path: Path,
) -> pd.DataFrame:
    """Generate recursive future predictions and save submission."""
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
    cogs_ratio = base.estimate_cogs_ratio(train_df)
    return build_submission(sample_submission, predictions, cogs_ratio, path)


def blend_submissions(
    full_submission: pd.DataFrame,
    promo_submission: pd.DataFrame,
    full_weight: float,
    path: Path,
) -> pd.DataFrame:
    """Blend FULL and PROMO pruned submissions."""
    promo_weight = 1.0 - full_weight
    output = full_submission[[base.DATE_COL]].copy()
    output[base.TARGET_COL] = (
        full_weight * full_submission[base.TARGET_COL]
        + promo_weight * promo_submission[base.TARGET_COL]
    )
    output[base.COGS_COL] = (
        full_weight * full_submission[base.COGS_COL]
        + promo_weight * promo_submission[base.COGS_COL]
    )
    output[base.TARGET_COL] = output[base.TARGET_COL].clip(lower=0)
    output[base.COGS_COL] = output[base.COGS_COL].clip(lower=0)
    output.to_csv(path, index=False)
    return output


def save_pruned_ensemble_validation_predictions(
    full_validation: dict[str, Any],
    promo_validation: dict[str, Any],
    full_weight: float,
    path: Path = PRUNED_ENSEMBLE_VALIDATION_PATH,
) -> pd.DataFrame:
    """Save per-date validation predictions for the selected pruned ensemble."""
    promo_weight = 1.0 - full_weight
    predicted = (
        full_weight * np.asarray(full_validation["predictions"], dtype=float)
        + promo_weight * np.asarray(promo_validation["predictions"], dtype=float)
    )
    actual = np.asarray(full_validation["actual"], dtype=float)
    error = actual - predicted
    output = pd.DataFrame(
        {
            "Date": full_validation["dates"],
            "actual_Revenue": actual,
            "predicted_Revenue": predicted,
            "error": error,
            "abs_error": np.abs(error),
            "pct_error": np.where(actual != 0, error / actual, np.nan),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    return output


def get_top_importance(
    trained: dict[str, Any],
    validation_rmse: float,
) -> pd.DataFrame:
    """Extract feature importance for a pruned model."""
    return base.get_feature_importance(
        trained["model"],
        trained["model_type"],
        trained["feature_columns"],
        trained["X_train"],
        trained["y_train"],
        validation_rmse,
    )


def run_experiment() -> None:
    logger = setup_logging()
    reporter = Reporter(logger)

    reporter.emit("Final Feature Pruning + Retrain + Re-Ensemble")
    reporter.emit("==============================================")
    reporter.emit("")

    train_df = base.load_train_data(base.TRAIN_DATA_PATH)
    sample_submission = base.load_sample_submission(base.SAMPLE_SUBMISSION_PATH)
    all_dates = pd.Series(
        pd.date_range(train_df[base.DATE_COL].min(), sample_submission[base.DATE_COL].max(), freq="D")
    )
    static_full = base.build_static_features(all_dates, train_df[base.DATE_COL].min(), logger)
    static_promo = promo_duration.build_static_features_with_promo_duration(
        all_dates,
        train_df[base.DATE_COL].min(),
        logger,
    )
    table_full = base.build_historical_model_table(train_df, static_full, include_business_lag365=False)
    table_promo = base.build_historical_model_table(train_df, static_promo, include_business_lag365=False)

    reporter.emit("1. Load feature importance")
    full_ranked = load_ranked_features(FULL_IMPORTANCE_PATH)
    if PROMO_IMPORTANCE_PATH.exists():
        promo_ranked = load_ranked_features(PROMO_IMPORTANCE_PATH)
        reporter.emit(f"Loaded PROMO importance: {PROMO_IMPORTANCE_PATH}")
    else:
        promo_ranked = full_ranked.copy()
        reporter.emit("PROMO importance missing; using FULL ranked features for PROMO branch")

    full_feature_sets: dict[str, list[str]] = {}
    promo_feature_sets: dict[str, list[str]] = {}
    for size in FEATURE_SET_SIZES:
        set_name = f"Top {size}"
        full_feature_sets[set_name] = filter_existing_features(full_ranked[:size], table_full)
        promo_feature_sets[set_name] = filter_existing_features(promo_ranked[:size], table_promo)
        reporter.emit(
            f"{set_name}: FULL features={len(full_feature_sets[set_name])}, "
            f"PROMO features={len(promo_feature_sets[set_name])}"
        )

    feature_set_report = build_feature_set_report(full_feature_sets, promo_feature_sets)
    feature_set_report.to_csv(PRUNED_FEATURE_SETS_PATH, index=False)
    reporter.emit(f"Saved pruned feature sets: {PRUNED_FEATURE_SETS_PATH}")

    reporter.emit("")
    reporter.emit("2. Retrain and recursive-validate pruned models")
    experiment_rows: list[dict[str, float | str]] = []
    experiment_state: dict[str, dict[str, Any]] = {}

    for set_name in [f"Top {size}" for size in FEATURE_SET_SIZES]:
        reporter.emit("")
        reporter.emit(f"Feature set: {set_name}")
        full_trained = train_selected_features(
            f"FULL {set_name}",
            table_full,
            full_feature_sets[set_name],
            base.TRAIN_CUTOFF,
            reporter,
        )
        promo_trained = train_selected_features(
            f"PROMO_DURATION {set_name}",
            table_promo,
            promo_feature_sets[set_name],
            base.TRAIN_CUTOFF,
            reporter,
        )

        full_validation = recursive_validate(full_trained, static_full, train_df)
        promo_validation = recursive_validate(promo_trained, static_promo, train_df)
        ensemble_search = grid_search_two_model_ensemble(
            full_validation["actual"],
            full_validation["predictions"],
            promo_validation["predictions"],
        )
        best_ensemble = ensemble_search.iloc[0].to_dict()

        experiment_rows.append(
            {
                "feature_set": set_name,
                "model": "FULL_pruned",
                **full_validation["metrics"],
            }
        )
        experiment_rows.append(
            {
                "feature_set": set_name,
                "model": "PROMO_pruned",
                **promo_validation["metrics"],
            }
        )
        experiment_rows.append(
            {
                "feature_set": set_name,
                "model": "ENSEMBLE_pruned",
                "MAE": best_ensemble["MAE"],
                "RMSE": best_ensemble["RMSE"],
                "R2": best_ensemble["R2"],
                "weight_FULL": best_ensemble["weight_FULL_pruned"],
                "weight_PROMO": best_ensemble["weight_PROMO_pruned"],
            }
        )

        experiment_state[set_name] = {
            "full_features": full_feature_sets[set_name],
            "promo_features": promo_feature_sets[set_name],
            "full_validation": full_validation,
            "promo_validation": promo_validation,
            "ensemble_search": ensemble_search,
            "best_ensemble": best_ensemble,
        }
        reporter.emit(
            f"{set_name} FULL RMSE={full_validation['metrics']['RMSE']:,.2f}; "
            f"PROMO RMSE={promo_validation['metrics']['RMSE']:,.2f}; "
            f"ENSEMBLE RMSE={best_ensemble['RMSE']:,.2f} "
            f"(w_full={best_ensemble['weight_FULL_pruned']:.2f}, "
            f"w_promo={best_ensemble['weight_PROMO_pruned']:.2f})"
        )

    metrics_df = pd.DataFrame(experiment_rows).sort_values(["RMSE", "MAE"]).reset_index(drop=True)
    reporter.emit("")
    reporter.emit_frame("Validation results:", metrics_df)

    ensemble_rows = metrics_df[metrics_df["model"] == "ENSEMBLE_pruned"].copy()
    best_row = ensemble_rows.sort_values(["RMSE", "MAE"]).iloc[0]
    best_set = str(best_row["feature_set"])
    best_state = experiment_state[best_set]
    reporter.emit(f"Best feature set by ensemble RMSE: {best_set}")
    validation_predictions = save_pruned_ensemble_validation_predictions(
        best_state["full_validation"],
        best_state["promo_validation"],
        float(best_row["weight_FULL"]),
        PRUNED_ENSEMBLE_VALIDATION_PATH,
    )
    reporter.emit(f"Saved pruned ensemble validation predictions: {PRUNED_ENSEMBLE_VALIDATION_PATH}")
    reporter.emit(f"Pruned ensemble validation prediction shape: {validation_predictions.shape}")

    reporter.emit("")
    reporter.emit("3. Final retrain on all 2012-2022 usable rows")
    final_full = train_selected_features(
        f"FULL final {best_set}",
        table_full,
        best_state["full_features"],
        train_end_exclusive=None,
        reporter=reporter,
    )
    final_promo = train_selected_features(
        f"PROMO final {best_set}",
        table_promo,
        best_state["promo_features"],
        train_end_exclusive=None,
        reporter=reporter,
    )

    full_submission = forecast_submission(
        final_full,
        static_full,
        train_df,
        sample_submission,
        SUBMISSION_PRUNED_FULL_PATH,
    )
    promo_submission = forecast_submission(
        final_promo,
        static_promo,
        train_df,
        sample_submission,
        SUBMISSION_PRUNED_PROMO_PATH,
    )
    ensemble_submission = blend_submissions(
        full_submission,
        promo_submission,
        float(best_row["weight_FULL"]),
        SUBMISSION_PRUNED_ENSEMBLE_PATH,
    )
    del ensemble_submission

    full_importance = get_top_importance(final_full, best_state["full_validation"]["metrics"]["RMSE"])
    promo_importance = get_top_importance(final_promo, best_state["promo_validation"]["metrics"]["RMSE"])

    reporter.emit("")
    reporter.emit("4. Final summary")
    reporter.emit(f"Best feature set: {best_set}")
    reporter.emit(
        f"Best ensemble weights: FULL={float(best_row['weight_FULL']):.2f}, "
        f"PROMO={float(best_row['weight_PROMO']):.2f}"
    )
    reporter.emit(
        f"Best ensemble metrics: MAE={float(best_row['MAE']):,.2f}, "
        f"RMSE={float(best_row['RMSE']):,.2f}, R2={float(best_row['R2']):.6f}"
    )
    reporter.emit(
        f"RMSE improvement vs original FULL: {BASELINE_FULL_RMSE - float(best_row['RMSE']):,.2f}"
    )
    reporter.emit(
        f"RMSE improvement vs previous ensemble: {BASELINE_ENSEMBLE_RMSE - float(best_row['RMSE']):,.2f}"
    )
    reporter.emit_frame("Top 20 FULL pruned features:", full_importance.head(20))
    reporter.emit_frame("Top 20 PROMO pruned features:", promo_importance.head(20))
    reporter.emit(f"Saved pruned FULL submission: {SUBMISSION_PRUNED_FULL_PATH}")
    reporter.emit(f"Saved pruned PROMO submission: {SUBMISSION_PRUNED_PROMO_PATH}")
    reporter.emit(f"Saved pruned ENSEMBLE submission: {SUBMISSION_PRUNED_ENSEMBLE_PATH}")
    reporter.emit(f"Final recommended submission file: {SUBMISSION_PRUNED_ENSEMBLE_PATH}")
    reporter.emit(
        "Leakage confirmation: pruning reused forecast-safe features only; recursive validation/forecasting "
        "and historical COGS ratio logic are unchanged."
    )

    reporter.save(METRICS_PATH)


if __name__ == "__main__":
    run_experiment()
