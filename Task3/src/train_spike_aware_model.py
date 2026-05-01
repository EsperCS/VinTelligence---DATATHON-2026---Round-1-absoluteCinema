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

PRUNED_FEATURE_SETS_PATH = DATA_DIR / "pruned_feature_sets.csv"
FINAL_IMPORTANCE_PATH = DATA_DIR / "final_feature_importance.csv"

BASELINE_VALIDATION_PATH = DATA_DIR / "pruned_ensemble_validation_predictions.csv"
BASELINE_SUBMISSION_PATH = DATA_DIR / "submission_pruned_ensemble.csv"

SUBMISSION_SPIKE_AWARE_PATH = DATA_DIR / "submission_spike_aware.csv"
SUBMISSION_VARIANT_A_PATH = DATA_DIR / "submission_spike_variant_a.csv"
SUBMISSION_VARIANT_B_PATH = DATA_DIR / "submission_spike_quantile.csv"
SUBMISSION_VARIANT_C_PATH = DATA_DIR / "submission_spike_weighted.csv"

SPIKE_VALIDATION_PATH = DATA_DIR / "spike_model_validation_predictions.csv"
SPIKE_FEATURE_IMPORTANCE_PATH = DATA_DIR / "spike_model_feature_importance.csv"
SPIKE_COMPARISON_PATH = DATA_DIR / "spike_model_comparison.csv"

REPORT_PATH = LOG_DIR / "spike_aware_model_report.txt"
LOG_FILE = LOG_DIR / "train_spike_aware_model.log"

TOP_FEATURE_SET_NAME = "Top 50"
WEIGHT_STEP = 0.05
EPSILON = 1e-6
HIGH_ERROR_MONTHS = {2, 3, 5, 8}

CURRENT_BEST = {
    "MAE": 669_832.08,
    "RMSE": 943_731.57,
    "R2": 0.682104,
    "top10_RMSE": 1_829_369.87,
    "top10_underprediction": 31,
}

SPIKE_FEATURES = [
    "lag7_to_roll30_ratio",
    "lag14_to_roll30_ratio",
    "lag30_to_roll90_ratio",
    "lag365_to_roll365_ratio",
    "lag365_to_roll30_ratio",
    "lag7_spike_flag",
    "lag30_spike_flag",
    "lag365_spike_flag",
    "recent_volatility_high",
    "volatility_30",
    "volatility_90",
    "volatility_365",
    "lag365_x_month",
    "lag365_x_day_of_week",
    "lag365_x_day_of_year",
    "lag7_x_day_of_week",
    "is_high_error_month",
    "lag365_x_high_error_month",
    "rolling7_x_high_error_month",
]


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
        self.logger.info("Saved spike-aware report to %s", path)


def setup_logging(log_file: Path = LOG_FILE) -> logging.Logger:
    """Configure simple file logging."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train_spike_aware_model")
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


def deduplicate_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def safe_divide(numerator: Any, denominator: Any, epsilon: float = EPSILON) -> Any:
    if isinstance(numerator, pd.Series) or isinstance(denominator, pd.Series):
        numerator_series = pd.Series(numerator, copy=False)
        denominator_series = pd.Series(denominator, copy=False)
        valid = numerator_series.notna() & denominator_series.notna()
        result = pd.Series(np.nan, index=numerator_series.index, dtype=float)
        result.loc[valid] = numerator_series.loc[valid] / (denominator_series.loc[valid].abs() + epsilon)
        return result

    if pd.isna(numerator) or pd.isna(denominator):
        return np.nan
    return float(numerator) / (abs(float(denominator)) + epsilon)


def load_top_full_features(limit: int = 50) -> list[str]:
    """Load the exact FULL pruned feature set when available."""
    if PRUNED_FEATURE_SETS_PATH.exists():
        feature_sets = pd.read_csv(PRUNED_FEATURE_SETS_PATH)
        mask = (feature_sets["feature_set"] == TOP_FEATURE_SET_NAME) & (feature_sets["model"] == "FULL")
        selected = feature_sets.loc[mask].sort_values("rank")["feature"].dropna().astype(str).tolist()
        if selected:
            return selected[:limit]

    importance = pd.read_csv(FINAL_IMPORTANCE_PATH)
    importance["importance_gain"] = pd.to_numeric(importance["importance_gain"], errors="coerce").fillna(0)
    return (
        importance.sort_values("importance_gain", ascending=False)["feature"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .head(limit)
        .tolist()
    )


def add_historical_spike_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create spike-aware features using only already-safe historical inputs."""
    output = df.sort_values(base.DATE_COL).reset_index(drop=True).copy()

    output["volatility_30"] = safe_divide(output["revenue_roll_std_30"], output["rolling_mean_30"])
    output["volatility_90"] = safe_divide(output["revenue_roll_std_90"], output["revenue_roll_mean_90"])
    output["volatility_365"] = safe_divide(output["revenue_roll_std_365"], output["revenue_roll_mean_365"])

    output["lag7_to_roll30_ratio"] = safe_divide(output["lag_7"], output["rolling_mean_30"])
    output["lag14_to_roll30_ratio"] = safe_divide(output["lag_14"], output["rolling_mean_30"])
    output["lag30_to_roll90_ratio"] = safe_divide(output["lag_30"], output["revenue_roll_mean_90"])
    output["lag365_to_roll365_ratio"] = safe_divide(output["revenue_lag_365"], output["revenue_roll_mean_365"])
    output["lag365_to_roll30_ratio"] = safe_divide(output["revenue_lag_365"], output["rolling_mean_30"])

    lag7_valid = output[["lag_7", "rolling_mean_30"]].notna().all(axis=1)
    lag30_valid = output[["lag_30", "revenue_roll_mean_90"]].notna().all(axis=1)
    lag365_valid = output[["revenue_lag_365", "revenue_roll_mean_365"]].notna().all(axis=1)
    output["lag7_spike_flag"] = np.where(
        lag7_valid,
        (output["lag_7"] > output["rolling_mean_30"] * 1.25).astype(int),
        np.nan,
    )
    output["lag30_spike_flag"] = np.where(
        lag30_valid,
        (output["lag_30"] > output["revenue_roll_mean_90"] * 1.25).astype(int),
        np.nan,
    )
    output["lag365_spike_flag"] = np.where(
        lag365_valid,
        (output["revenue_lag_365"] > output["revenue_roll_mean_365"] * 1.25).astype(int),
        np.nan,
    )

    volatility_history_median = output["volatility_30"].shift(1).expanding(min_periods=1).median()
    volatility_valid = output["volatility_30"].notna() & volatility_history_median.notna()
    output["recent_volatility_high"] = np.where(
        volatility_valid,
        (output["volatility_30"] > volatility_history_median).astype(int),
        np.nan,
    )

    output["lag365_x_month"] = output["revenue_lag_365"] * output["month"]
    output["lag365_x_day_of_week"] = output["revenue_lag_365"] * output["day_of_week"]
    output["lag365_x_day_of_year"] = output["revenue_lag_365"] * output["day_of_year"]
    output["lag7_x_day_of_week"] = output["lag_7"] * output["day_of_week"]

    output["is_high_error_month"] = output["month"].isin(HIGH_ERROR_MONTHS).astype(int)
    output["lag365_x_high_error_month"] = output["revenue_lag_365"] * output["is_high_error_month"]
    output["rolling7_x_high_error_month"] = output["rolling_mean_7"] * output["is_high_error_month"]
    return output


def build_spike_model_table(train_df: pd.DataFrame, static_features: pd.DataFrame) -> pd.DataFrame:
    """Build the forecast-safe FULL model table and append spike-aware features."""
    table = base.build_historical_model_table(train_df, static_features, include_business_lag365=False)
    return add_historical_spike_features(table)


def compute_fixed_volatility_threshold(history: pd.Series) -> float:
    """Median 30-day relative volatility from observed history only."""
    ordered = pd.to_numeric(history, errors="coerce").sort_index()
    shifted = ordered.shift(1)
    roll_mean = shifted.rolling(window=30, min_periods=30).mean()
    roll_std = shifted.rolling(window=30, min_periods=30).std()
    volatility = safe_divide(roll_std, roll_mean)
    values = pd.Series(volatility).dropna()
    if values.empty:
        return 0.0
    return float(values.median())


def compute_spike_features_from_row(row: dict[str, float], volatility_threshold: float) -> dict[str, float]:
    """Compute one spike-aware feature row from already-safe row inputs."""
    volatility_30 = safe_divide(row.get("revenue_roll_std_30"), row.get("rolling_mean_30"))
    volatility_90 = safe_divide(row.get("revenue_roll_std_90"), row.get("revenue_roll_mean_90"))
    volatility_365 = safe_divide(row.get("revenue_roll_std_365"), row.get("revenue_roll_mean_365"))

    is_high_error_month = int(int(row.get("month", 0)) in HIGH_ERROR_MONTHS)

    return {
        "lag7_to_roll30_ratio": safe_divide(row.get("lag_7"), row.get("rolling_mean_30")),
        "lag14_to_roll30_ratio": safe_divide(row.get("lag_14"), row.get("rolling_mean_30")),
        "lag30_to_roll90_ratio": safe_divide(row.get("lag_30"), row.get("revenue_roll_mean_90")),
        "lag365_to_roll365_ratio": safe_divide(row.get("revenue_lag_365"), row.get("revenue_roll_mean_365")),
        "lag365_to_roll30_ratio": safe_divide(row.get("revenue_lag_365"), row.get("rolling_mean_30")),
        "lag7_spike_flag": float(
            int(
                pd.notna(row.get("lag_7"))
                and pd.notna(row.get("rolling_mean_30"))
                and row.get("lag_7", np.nan) > row.get("rolling_mean_30", np.nan) * 1.25
            )
        )
        if pd.notna(row.get("lag_7")) and pd.notna(row.get("rolling_mean_30"))
        else np.nan,
        "lag30_spike_flag": float(
            int(
                pd.notna(row.get("lag_30"))
                and pd.notna(row.get("revenue_roll_mean_90"))
                and row.get("lag_30", np.nan) > row.get("revenue_roll_mean_90", np.nan) * 1.25
            )
        )
        if pd.notna(row.get("lag_30")) and pd.notna(row.get("revenue_roll_mean_90"))
        else np.nan,
        "lag365_spike_flag": float(
            int(
                pd.notna(row.get("revenue_lag_365"))
                and pd.notna(row.get("revenue_roll_mean_365"))
                and row.get("revenue_lag_365", np.nan) > row.get("revenue_roll_mean_365", np.nan) * 1.25
            )
        )
        if pd.notna(row.get("revenue_lag_365")) and pd.notna(row.get("revenue_roll_mean_365"))
        else np.nan,
        "recent_volatility_high": float(int(pd.notna(volatility_30) and volatility_30 > volatility_threshold))
        if pd.notna(volatility_30)
        else np.nan,
        "volatility_30": volatility_30,
        "volatility_90": volatility_90,
        "volatility_365": volatility_365,
        "lag365_x_month": row.get("revenue_lag_365", np.nan) * row.get("month", np.nan),
        "lag365_x_day_of_week": row.get("revenue_lag_365", np.nan) * row.get("day_of_week", np.nan),
        "lag365_x_day_of_year": row.get("revenue_lag_365", np.nan) * row.get("day_of_year", np.nan),
        "lag7_x_day_of_week": row.get("lag_7", np.nan) * row.get("day_of_week", np.nan),
        "is_high_error_month": float(is_high_error_month),
        "lag365_x_high_error_month": row.get("revenue_lag_365", np.nan) * is_high_error_month,
        "rolling7_x_high_error_month": row.get("rolling_mean_7", np.nan) * is_high_error_month,
    }


def make_training_matrix(
    model_table: pd.DataFrame,
    feature_columns: list[str],
    train_end_exclusive: pd.Timestamp | None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    table = model_table.copy()
    if train_end_exclusive is not None:
        table = table[table[base.DATE_COL] < train_end_exclusive].copy()

    missing = [column for column in feature_columns if column not in table.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    clean = table.dropna(subset=feature_columns + [base.TARGET_COL]).reset_index(drop=True)
    X = clean[feature_columns].copy()
    y = clean[base.TARGET_COL].copy()
    feature_medians = X.median(numeric_only=True)
    return X, y, clean, feature_medians


def train_lightgbm_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    reporter: Reporter,
    objective: str = "regression",
    alpha: float | None = None,
    sample_weight: np.ndarray | None = None,
) -> Any:
    import lightgbm as lgb

    params = {
        "objective": objective,
        "metric": "rmse",
        "learning_rate": 0.025,
        "max_depth": 6,
        "num_leaves": 31,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "feature_fraction": 0.9,
        "seed": base.RANDOM_STATE,
        "verbosity": -1,
        "force_col_wise": True,
    }
    if objective == "quantile":
        params["metric"] = "quantile"
        params["alpha"] = alpha if alpha is not None else 0.70

    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        weight=sample_weight,
        feature_name=X_train.columns.tolist(),
        free_raw_data=False,
    )
    model = lgb.train(params=params, train_set=train_data, num_boost_round=2000)
    reporter.logger.info(
        "Trained LightGBM objective=%s rows=%s features=%s",
        objective,
        len(X_train),
        X_train.shape[1],
    )
    return model


def train_hist_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    objective: str,
    alpha: float | None,
    sample_weight: np.ndarray | None,
) -> Any:
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError as exc:
        raise ImportError("scikit-learn fallback is not installed.") from exc

    kwargs: dict[str, Any] = {
        "learning_rate": 0.025,
        "max_iter": 2000,
        "max_leaf_nodes": 31,
        "random_state": base.RANDOM_STATE,
    }
    if objective == "quantile":
        kwargs["loss"] = "quantile"
        kwargs["quantile"] = alpha if alpha is not None else 0.70

    model = HistGradientBoostingRegressor(**kwargs)
    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
    return model


def train_variant_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    reporter: Reporter,
    objective: str = "regression",
    alpha: float | None = None,
    sample_weight: np.ndarray | None = None,
) -> tuple[Any, str]:
    if base.lightgbm_available():
        return train_lightgbm_model(
            X_train,
            y_train,
            reporter,
            objective=objective,
            alpha=alpha,
            sample_weight=sample_weight,
        ), "lightgbm"
    return train_hist_model(
        X_train,
        y_train,
        objective=objective,
        alpha=alpha,
        sample_weight=sample_weight,
    ), "hist_gradient_boosting"


def build_weighted_sample_weights(y_train: pd.Series) -> np.ndarray:
    q80 = float(y_train.quantile(0.80))
    q90 = float(y_train.quantile(0.90))
    weights = np.ones(len(y_train), dtype=float)
    values = y_train.to_numpy(dtype=float)
    weights[values >= q80] = 2.0
    weights[values >= q90] = 3.0
    return weights


def recursive_predict_spike(
    model: Any,
    model_type: str,
    prediction_dates: pd.Series,
    feature_columns: list[str],
    static_features: pd.DataFrame,
    initial_revenue_history: pd.Series,
    feature_medians: pd.Series,
    volatility_threshold: float,
) -> np.ndarray:
    del model_type
    static_by_date = static_features.set_index(base.DATE_COL).sort_index()
    history = pd.to_numeric(initial_revenue_history, errors="coerce").sort_index().copy()
    predictions: list[float] = []

    for forecast_date in pd.to_datetime(prediction_dates):
        if forecast_date not in static_by_date.index:
            raise ValueError(f"Missing static features for forecast date {forecast_date.date()}")

        row: dict[str, float] = static_by_date.loc[forecast_date].to_dict()
        row.update(base.compute_revenue_features_from_history(history, forecast_date))
        row.update(compute_spike_features_from_row(row, volatility_threshold))

        X_row = pd.DataFrame([row], columns=feature_columns)
        X_row = X_row.apply(pd.to_numeric, errors="coerce").fillna(feature_medians).fillna(0)

        prediction = float(model.predict(X_row)[0])
        prediction = max(0.0, prediction)
        predictions.append(prediction)
        history.loc[forecast_date] = prediction

    return np.asarray(predictions, dtype=float)


def evaluate_candidate(name: str, actual: pd.Series, predictions: np.ndarray) -> dict[str, Any]:
    actual_values = actual.to_numpy(dtype=float)
    predicted_values = np.asarray(predictions, dtype=float)
    error = actual_values - predicted_values
    overall = base.evaluate_predictions(actual, predicted_values)

    top10_threshold = float(np.quantile(actual_values, 0.90))
    top5_threshold = float(np.quantile(actual_values, 0.95))
    top10_mask = actual_values >= top10_threshold
    top5_mask = actual_values >= top5_threshold

    def masked_rmse(mask: np.ndarray) -> float:
        return float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan

    return {
        "model": name,
        "MAE": overall["MAE"],
        "RMSE": overall["RMSE"],
        "R2": overall["R2"],
        "top10_RMSE": masked_rmse(top10_mask),
        "top10_mean_error": float(np.mean(error[top10_mask])) if top10_mask.any() else np.nan,
        "top10_underprediction": int(np.sum(error[top10_mask] > 0)) if top10_mask.any() else 0,
        "top10_count": int(np.sum(top10_mask)),
        "top5_RMSE": masked_rmse(top5_mask),
        "top5_underprediction": int(np.sum(error[top5_mask] > 0)) if top5_mask.any() else 0,
        "top5_count": int(np.sum(top5_mask)),
    }


def validate_variant(
    variant_name: str,
    model_table: pd.DataFrame,
    static_features: pd.DataFrame,
    train_df: pd.DataFrame,
    feature_columns: list[str],
    reporter: Reporter,
    objective: str = "regression",
    alpha: float | None = None,
    weighted: bool = False,
) -> dict[str, Any]:
    X_train, y_train, train_clean, feature_medians = make_training_matrix(
        model_table,
        feature_columns,
        base.TRAIN_CUTOFF,
    )
    sample_weight = build_weighted_sample_weights(y_train) if weighted else None
    reporter.emit(
        f"Training {variant_name}: rows={len(X_train):,}, features={len(feature_columns)}, "
        f"objective={objective}"
    )
    model, model_type = train_variant_model(
        X_train,
        y_train,
        reporter,
        objective=objective,
        alpha=alpha,
        sample_weight=sample_weight,
    )

    validation_dates = train_df[
        (train_df[base.DATE_COL] >= base.TRAIN_CUTOFF) & (train_df[base.DATE_COL] <= base.VALIDATION_END)
    ][base.DATE_COL]
    actual = train_df.set_index(base.DATE_COL).loc[validation_dates, base.TARGET_COL].reset_index(drop=True)
    initial_history = train_df[train_df[base.DATE_COL] < base.TRAIN_CUTOFF].set_index(base.DATE_COL)[base.TARGET_COL]
    volatility_threshold = compute_fixed_volatility_threshold(initial_history)
    predictions = recursive_predict_spike(
        model=model,
        model_type=model_type,
        prediction_dates=validation_dates,
        feature_columns=feature_columns,
        static_features=static_features,
        initial_revenue_history=initial_history,
        feature_medians=feature_medians,
        volatility_threshold=volatility_threshold,
    )
    metrics = evaluate_candidate(variant_name, actual, predictions)
    return {
        "model": variant_name,
        "model_object": model,
        "model_type": model_type,
        "feature_columns": feature_columns,
        "feature_medians": feature_medians,
        "X_train": X_train,
        "y_train": y_train,
        "train_clean": train_clean,
        "validation_dates": validation_dates.reset_index(drop=True),
        "actual": actual,
        "predictions": predictions,
        "metrics": metrics,
        "volatility_threshold": volatility_threshold,
    }


def train_full_variant(
    variant_name: str,
    model_table: pd.DataFrame,
    feature_columns: list[str],
    reporter: Reporter,
    objective: str = "regression",
    alpha: float | None = None,
    weighted: bool = False,
) -> dict[str, Any]:
    X_train, y_train, train_clean, feature_medians = make_training_matrix(
        model_table,
        feature_columns,
        train_end_exclusive=None,
    )
    sample_weight = build_weighted_sample_weights(y_train) if weighted else None
    reporter.emit(
        f"Retraining {variant_name} on all rows: rows={len(X_train):,}, features={len(feature_columns)}"
    )
    model, model_type = train_variant_model(
        X_train,
        y_train,
        reporter,
        objective=objective,
        alpha=alpha,
        sample_weight=sample_weight,
    )
    return {
        "model": variant_name,
        "model_object": model,
        "model_type": model_type,
        "feature_columns": feature_columns,
        "feature_medians": feature_medians,
        "X_train": X_train,
        "y_train": y_train,
        "train_clean": train_clean,
    }


def forecast_variant_submission(
    trained: dict[str, Any],
    static_features: pd.DataFrame,
    train_df: pd.DataFrame,
    sample_submission: pd.DataFrame,
    path: Path,
) -> pd.DataFrame:
    initial_history = train_df.set_index(base.DATE_COL)[base.TARGET_COL].sort_index()
    volatility_threshold = compute_fixed_volatility_threshold(initial_history)
    predictions = recursive_predict_spike(
        model=trained["model_object"],
        model_type=trained["model_type"],
        prediction_dates=sample_submission[base.DATE_COL],
        feature_columns=trained["feature_columns"],
        static_features=static_features,
        initial_revenue_history=initial_history,
        feature_medians=trained["feature_medians"],
        volatility_threshold=volatility_threshold,
    )
    cogs_ratio = base.estimate_cogs_ratio(train_df)
    return base.build_submission(sample_submission, predictions, cogs_ratio, path)


def load_baseline_validation() -> tuple[pd.Series, np.ndarray]:
    baseline = pd.read_csv(BASELINE_VALIDATION_PATH, parse_dates=["Date"])
    if not {"actual_Revenue", "predicted_Revenue"}.issubset(baseline.columns):
        raise ValueError("Baseline validation file is missing required columns.")
    actual = pd.to_numeric(baseline["actual_Revenue"], errors="coerce")
    predictions = pd.to_numeric(baseline["predicted_Revenue"], errors="coerce").to_numpy(dtype=float)
    return actual, predictions


def load_submission(path: Path, sample_submission: pd.DataFrame) -> pd.DataFrame:
    submission = pd.read_csv(path, parse_dates=[base.DATE_COL])
    submission[base.DATE_COL] = pd.to_datetime(submission[base.DATE_COL], errors="coerce").dt.normalize()
    aligned = sample_submission[[base.DATE_COL]].merge(
        submission[[base.DATE_COL, base.TARGET_COL, base.COGS_COL]],
        on=base.DATE_COL,
        how="left",
        validate="one_to_one",
    )
    if aligned[[base.TARGET_COL, base.COGS_COL]].isna().any().any():
        raise ValueError(f"Submission file has missing rows after alignment: {path}")
    aligned[base.TARGET_COL] = pd.to_numeric(aligned[base.TARGET_COL], errors="coerce").clip(lower=0)
    aligned[base.COGS_COL] = pd.to_numeric(aligned[base.COGS_COL], errors="coerce").clip(lower=0)
    return aligned


def build_weight_combinations(model_names: list[str], step: float = WEIGHT_STEP) -> list[dict[str, float]]:
    units = int(round(1.0 / step))
    combinations: list[dict[str, float]] = []

    def recurse(index: int, remaining: int, current: list[int]) -> None:
        if index == len(model_names) - 1:
            weights = current + [remaining]
            if sum(weights) == units:
                combinations.append(
                    {
                        model_name: weight / units
                        for model_name, weight in zip(model_names, weights)
                    }
                )
            return

        for value in range(remaining + 1):
            recurse(index + 1, remaining - value, current + [value])

    recurse(0, units, [])
    return combinations


def evaluate_ensemble_candidates(
    actual: pd.Series,
    candidate_predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    model_names = list(candidate_predictions.keys())
    rows: list[dict[str, Any]] = []

    for weights in build_weight_combinations(model_names, WEIGHT_STEP):
        if all(weight == 0 for weight in weights.values()):
            continue

        blended = np.zeros(len(actual), dtype=float)
        for model_name, weight in weights.items():
            blended = blended + weight * candidate_predictions[model_name]

        metrics = evaluate_candidate("ENSEMBLE", actual, blended)
        rmse_improved = metrics["RMSE"] < CURRENT_BEST["RMSE"]
        spike_ok = (
            metrics["top10_RMSE"] <= CURRENT_BEST["top10_RMSE"]
            and metrics["top10_underprediction"] <= CURRENT_BEST["top10_underprediction"]
        )
        accepted = (not rmse_improved) or spike_ok

        row: dict[str, Any] = {
            "candidate": "ensemble",
            **metrics,
            "accepted": int(accepted),
            "rmse_improved_vs_best": int(rmse_improved),
            "spike_guard_pass": int(spike_ok),
        }
        for model_name, weight in weights.items():
            row[f"weight_{model_name}"] = weight
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["accepted", "RMSE", "top10_RMSE", "top10_underprediction", "MAE"],
        ascending=[False, True, True, True, True],
    ).reset_index(drop=True)


def build_importance_frame(
    trained_models: list[dict[str, Any]],
    validation_lookup: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for trained in trained_models:
        model_name = trained["model"]
        baseline_rmse = validation_lookup[model_name]["metrics"]["RMSE"]
        importance = base.get_feature_importance(
            trained["model_object"],
            trained["model_type"],
            trained["feature_columns"],
            trained["X_train"],
            trained["y_train"],
            baseline_rmse,
        ).copy()
        importance.insert(0, "model", model_name)
        importance["validation_rmse"] = baseline_rmse
        rows.append(importance)
    if not rows:
        return pd.DataFrame(columns=["model", "feature", "importance_split", "importance_gain", "validation_rmse"])
    return pd.concat(rows, ignore_index=True)


def blend_submissions(
    sample_submission: pd.DataFrame,
    submissions: dict[str, pd.DataFrame],
    weights: dict[str, float],
) -> pd.DataFrame:
    output = sample_submission[[base.DATE_COL]].copy()
    output[base.TARGET_COL] = 0.0
    output[base.COGS_COL] = 0.0

    for model_name, weight in weights.items():
        if weight == 0:
            continue
        aligned = submissions[model_name]
        output[base.TARGET_COL] = output[base.TARGET_COL] + weight * aligned[base.TARGET_COL]
        output[base.COGS_COL] = output[base.COGS_COL] + weight * aligned[base.COGS_COL]

    output[base.TARGET_COL] = output[base.TARGET_COL].clip(lower=0)
    output[base.COGS_COL] = output[base.COGS_COL].clip(lower=0)
    return output[[base.DATE_COL, base.TARGET_COL, base.COGS_COL]]


def save_validation_predictions(
    dates: pd.Series,
    actual: pd.Series,
    predicted: np.ndarray,
    selected_name: str,
    path: Path = SPIKE_VALIDATION_PATH,
) -> pd.DataFrame:
    error = actual.to_numpy(dtype=float) - np.asarray(predicted, dtype=float)
    output = pd.DataFrame(
        {
            base.DATE_COL: dates,
            "actual_Revenue": actual.to_numpy(dtype=float),
            "predicted_Revenue": np.asarray(predicted, dtype=float),
            "selected_model": selected_name,
            "error": error,
            "abs_error": np.abs(error),
            "pct_error": np.where(actual.to_numpy(dtype=float) != 0, error / actual.to_numpy(dtype=float), np.nan),
        }
    )
    output.to_csv(path, index=False)
    return output


def emit_metrics(reporter: Reporter, title: str, metrics: dict[str, Any]) -> None:
    reporter.emit(title)
    reporter.emit(
        f"MAE={metrics['MAE']:,.2f} | RMSE={metrics['RMSE']:,.2f} | R2={metrics['R2']:.6f}"
    )
    reporter.emit(
        f"Top10 RMSE={metrics['top10_RMSE']:,.2f} | Top10 mean error={metrics['top10_mean_error']:,.2f} | "
        f"Top10 underprediction={metrics['top10_underprediction']}/{metrics['top10_count']}"
    )
    reporter.emit(
        f"Top5 RMSE={metrics['top5_RMSE']:,.2f} | Top5 underprediction={metrics['top5_underprediction']}/{metrics['top5_count']}"
    )


def run_experiment() -> None:
    logger = setup_logging()
    reporter = Reporter(logger)

    reporter.emit("Spike-aware Forecasting Model")
    reporter.emit("=============================")
    reporter.emit("")

    reporter.emit("1. Load data and rebuild safe feature table")
    train_df = base.load_train_data(base.TRAIN_DATA_PATH)
    sample_submission = base.load_sample_submission(base.SAMPLE_SUBMISSION_PATH)
    all_dates = pd.Series(
        pd.date_range(train_df[base.DATE_COL].min(), sample_submission[base.DATE_COL].max(), freq="D")
    )
    static_features = base.build_static_features(all_dates, train_df[base.DATE_COL].min(), logger)
    spike_table = build_spike_model_table(train_df, static_features)
    top_features = load_top_full_features(limit=50)
    feature_columns = deduplicate_preserve_order(
        [feature for feature in top_features if feature in spike_table.columns] + SPIKE_FEATURES
    )
    reporter.emit(f"Static feature table shape: {static_features.shape}")
    reporter.emit(f"Spike model table shape: {spike_table.shape}")
    reporter.emit(f"Base pruned feature count: {len(top_features)}")
    reporter.emit(f"Spike-aware feature count: {len(SPIKE_FEATURES)}")
    reporter.emit(f"Variant feature count after merge/dedupe: {len(feature_columns)}")

    reporter.emit("")
    reporter.emit("2. Load current best validation baseline")
    baseline_actual, baseline_predictions = load_baseline_validation()
    baseline_metrics = evaluate_candidate("PRUNED_ENSEMBLE_BASELINE", baseline_actual, baseline_predictions)
    emit_metrics(reporter, "Current pruned ensemble baseline:", baseline_metrics)

    reporter.emit("")
    reporter.emit("3. Train spike-aware variants on recursive validation 2022")
    validation_results: dict[str, dict[str, Any]] = {}

    variant_a = validate_variant(
        "SPIKE_VARIANT_A",
        spike_table,
        static_features,
        train_df,
        feature_columns,
        reporter,
        objective="regression",
    )
    validation_results[variant_a["model"]] = variant_a
    emit_metrics(reporter, "Variant A metrics:", variant_a["metrics"])

    try:
        variant_b = validate_variant(
            "SPIKE_VARIANT_B_QUANTILE",
            spike_table,
            static_features,
            train_df,
            feature_columns,
            reporter,
            objective="quantile",
            alpha=0.70,
        )
        validation_results[variant_b["model"]] = variant_b
        emit_metrics(reporter, "Variant B metrics:", variant_b["metrics"])
    except Exception as exc:  # pragma: no cover - graceful fallback path
        reporter.emit(f"Variant B skipped: quantile training unavailable ({exc})")
        variant_b = None

    variant_c = validate_variant(
        "SPIKE_VARIANT_C_WEIGHTED",
        spike_table,
        static_features,
        train_df,
        feature_columns,
        reporter,
        objective="regression",
        weighted=True,
    )
    validation_results[variant_c["model"]] = variant_c
    emit_metrics(reporter, "Variant C metrics:", variant_c["metrics"])

    comparison_rows: list[dict[str, Any]] = [baseline_metrics]
    comparison_rows.extend(result["metrics"] for result in validation_results.values())

    reporter.emit("")
    reporter.emit("4. Ensemble search with spike guard")
    candidate_predictions = {"PRUNED_ENSEMBLE_BASELINE": baseline_predictions}
    for result in validation_results.values():
        candidate_predictions[result["model"]] = result["predictions"]

    ensemble_search = evaluate_ensemble_candidates(baseline_actual, candidate_predictions)
    best_ensemble_row = ensemble_search.iloc[0].to_dict()
    best_ensemble_weights = {
        name: float(best_ensemble_row.get(f"weight_{name}", 0.0))
        for name in candidate_predictions.keys()
        if float(best_ensemble_row.get(f"weight_{name}", 0.0)) > 0
    }
    ensemble_predictions = np.zeros(len(baseline_actual), dtype=float)
    for model_name, weight in best_ensemble_weights.items():
        ensemble_predictions = ensemble_predictions + weight * candidate_predictions[model_name]
    ensemble_metrics = evaluate_candidate("SPIKE_ENSEMBLE", baseline_actual, ensemble_predictions)
    comparison_rows.append(ensemble_metrics)

    reporter.emit(
        "Best ensemble weights: "
        + ", ".join(f"{name}={weight:.2f}" for name, weight in best_ensemble_weights.items())
    )
    emit_metrics(reporter, "Best ensemble metrics:", ensemble_metrics)

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        ["RMSE", "top10_RMSE", "top10_underprediction", "MAE"],
        ascending=[True, True, True, True],
    )
    comparison_df.to_csv(SPIKE_COMPARISON_PATH, index=False)
    ensemble_search.to_csv(DATA_DIR / "spike_ensemble_weight_search.csv", index=False)

    reporter.emit("")
    reporter.emit("5. Final retraining for submission candidates")
    trained_models: list[dict[str, Any]] = []
    full_variant_a = train_full_variant(
        "SPIKE_VARIANT_A",
        spike_table,
        feature_columns,
        reporter,
        objective="regression",
    )
    trained_models.append(full_variant_a)
    submission_a = forecast_variant_submission(
        full_variant_a,
        static_features,
        train_df,
        sample_submission,
        SUBMISSION_VARIANT_A_PATH,
    )

    submission_b: pd.DataFrame | None = None
    if variant_b is not None:
        full_variant_b = train_full_variant(
            "SPIKE_VARIANT_B_QUANTILE",
            spike_table,
            feature_columns,
            reporter,
            objective="quantile",
            alpha=0.70,
        )
        trained_models.append(full_variant_b)
        submission_b = forecast_variant_submission(
            full_variant_b,
            static_features,
            train_df,
            sample_submission,
            SUBMISSION_VARIANT_B_PATH,
        )

    full_variant_c = train_full_variant(
        "SPIKE_VARIANT_C_WEIGHTED",
        spike_table,
        feature_columns,
        reporter,
        objective="regression",
        weighted=True,
    )
    trained_models.append(full_variant_c)
    submission_c = forecast_variant_submission(
        full_variant_c,
        static_features,
        train_df,
        sample_submission,
        SUBMISSION_VARIANT_C_PATH,
    )

    importance_df = build_importance_frame(trained_models, validation_results)
    importance_df.to_csv(SPIKE_FEATURE_IMPORTANCE_PATH, index=False)

    submission_lookup = {
        "PRUNED_ENSEMBLE_BASELINE": load_submission(BASELINE_SUBMISSION_PATH, sample_submission),
        "SPIKE_VARIANT_A": submission_a,
        "SPIKE_VARIANT_C_WEIGHTED": submission_c,
    }
    if submission_b is not None:
        submission_lookup["SPIKE_VARIANT_B_QUANTILE"] = submission_b

    best_ensemble_submission = blend_submissions(sample_submission, submission_lookup, best_ensemble_weights)
    best_ensemble_submission.to_csv(SUBMISSION_SPIKE_AWARE_PATH, index=False)

    selected_name = "SPIKE_ENSEMBLE"
    selected_metrics = ensemble_metrics
    selected_predictions = ensemble_predictions
    selected_component_name = max(best_ensemble_weights, key=best_ensemble_weights.get)

    if (
        ensemble_metrics["RMSE"] > baseline_metrics["RMSE"]
        and ensemble_metrics["top10_underprediction"] > baseline_metrics["top10_underprediction"]
    ):
        selected_name = "PRUNED_ENSEMBLE_BASELINE"
        selected_metrics = baseline_metrics
        selected_predictions = baseline_predictions
        selected_component_name = "PRUNED_ENSEMBLE_BASELINE"
        baseline_submission = submission_lookup["PRUNED_ENSEMBLE_BASELINE"].copy()
        baseline_submission.to_csv(SUBMISSION_SPIKE_AWARE_PATH, index=False)

    validation_output = save_validation_predictions(
        dates=variant_a["validation_dates"],
        actual=baseline_actual,
        predicted=selected_predictions,
        selected_name=selected_name,
        path=SPIKE_VALIDATION_PATH,
    )

    reporter.emit("")
    reporter.emit("6. Final comparison summary")
    reporter.emit_frame("Model comparison:", comparison_df)
    reporter.emit(
        f"Top10 spike RMSE improved vs current best: {selected_metrics['top10_RMSE'] < CURRENT_BEST['top10_RMSE']}"
    )
    reporter.emit(
        "Top10 underprediction reduced vs current best: "
        f"{selected_metrics['top10_underprediction'] < CURRENT_BEST['top10_underprediction']}"
    )
    reporter.emit(f"Overall RMSE improved vs current best: {selected_metrics['RMSE'] < CURRENT_BEST['RMSE']}")
    selected_importance = importance_df[importance_df["model"] == selected_component_name].copy()
    if selected_importance.empty:
        selected_importance = importance_df.copy()
    reporter.emit_frame(
        f"Top 30 feature importance for {selected_component_name}:",
        selected_importance.sort_values(["importance_gain", "importance_split"], ascending=False).head(30),
    )
    reporter.emit(f"Saved validation predictions: {SPIKE_VALIDATION_PATH}")
    reporter.emit(f"Saved feature importance: {SPIKE_FEATURE_IMPORTANCE_PATH}")
    reporter.emit(f"Saved comparison report: {SPIKE_COMPARISON_PATH}")
    reporter.emit(f"Saved candidate submission A: {SUBMISSION_VARIANT_A_PATH}")
    if submission_b is not None:
        reporter.emit(f"Saved candidate submission B: {SUBMISSION_VARIANT_B_PATH}")
    reporter.emit(f"Saved candidate submission C: {SUBMISSION_VARIANT_C_PATH}")
    reporter.emit(f"Final recommended submission file: {SUBMISSION_SPIKE_AWARE_PATH}")
    reporter.emit(
        "Leakage confirmation: spike-aware features use only lagged Revenue, calendar, promotion schedule, "
        "and inventory as-of inputs; validation and submission remain fully recursive without future Revenue/COGS."
    )
    reporter.emit(
        f"Selected output: {selected_name} | rows={len(validation_output):,} validation predictions, "
        f"{len(sample_submission):,} submission rows"
    )
    reporter.save(REPORT_PATH)


if __name__ == "__main__":
    run_experiment()
