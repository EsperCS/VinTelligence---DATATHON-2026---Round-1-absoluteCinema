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

TRAIN_DATA_PATH = DATA_DIR / "daily_feature_table.csv"
COMPARISON_PATH = DATA_DIR / "conversion_regime_no_promo_model_comparison.csv"
VALIDATION_PREDICTIONS_PATH = DATA_DIR / "conversion_regime_no_promo_validation_predictions.csv"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "conversion_regime_no_promo_feature_importance.csv"
REPORT_PATH = LOG_DIR / "conversion_regime_no_promo_model_report.txt"
LOG_FILE = LOG_DIR / "train_conversion_regime_no_promo.log"

DATE_COL = "Date"
ORDERS_COL = "orders_count"
SESSIONS_COL = "web_sessions"
TARGET_COL = "conversion_rate"
TRAIN_CUTOFF = pd.Timestamp("2022-01-01")
VALIDATION_END = pd.Timestamp("2022-12-31")
RECENT_2019_START = pd.Timestamp("2019-01-01")
RECENT_2020_START = pd.Timestamp("2020-01-01")
RANDOM_STATE = 42
EPSILON = 1e-6
DIRECT_ORDERS_BASELINE_RMSE = 26.2598

CALENDAR_FEATURES = [
    "month",
    "day_of_year",
    "day_of_week",
    "is_weekend",
]

CONVERSION_LAG_FEATURES = [
    "conversion_lag_7",
    "conversion_lag_14",
    "conversion_lag_30",
    "conversion_lag_365",
    "conversion_roll_mean_7",
    "conversion_roll_mean_30",
]

TRAFFIC_FEATURES = [
    "web_sessions_lag_7",
    "web_sessions_roll_mean_30",
    "web_sessions_growth",
]

FEATURE_COLUMNS = CALENDAR_FEATURES + CONVERSION_LAG_FEATURES + TRAFFIC_FEATURES


class Reporter:
    """Print, log, and persist a run summary."""

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
        self.logger.info("Saved report to %s", path)


def setup_logging(log_file: Path = LOG_FILE) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train_conversion_regime_no_promo")
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


def load_daily_feature_table(path: Path = TRAIN_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Training dataset not found: {path}")

    df = pd.read_csv(path, parse_dates=[DATE_COL], low_memory=False)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce").dt.normalize()
    df[ORDERS_COL] = pd.to_numeric(df[ORDERS_COL], errors="coerce")
    df[SESSIONS_COL] = pd.to_numeric(df.get(SESSIONS_COL, 0), errors="coerce").fillna(0.0)
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    if df[DATE_COL].isna().any():
        raise ValueError("Date column contains invalid timestamps")
    if df[ORDERS_COL].isna().any():
        raise ValueError("orders_count contains invalid values")

    df[TARGET_COL] = np.where(df[SESSIONS_COL] > 0, df[ORDERS_COL] / df[SESSIONS_COL], 0.0)
    return df


def build_calendar_features(dates: pd.Series) -> pd.DataFrame:
    output = pd.DataFrame({DATE_COL: pd.to_datetime(dates).sort_values().unique()})
    output["month"] = output[DATE_COL].dt.month.astype(int)
    output["day_of_year"] = output[DATE_COL].dt.dayofyear.astype(int)
    output["day_of_week"] = output[DATE_COL].dt.dayofweek.astype(int)
    output["is_weekend"] = output["day_of_week"].isin([5, 6]).astype(int)
    return output[[DATE_COL] + CALENDAR_FEATURES]


def build_traffic_features(df: pd.DataFrame) -> pd.DataFrame:
    output = df[[DATE_COL, SESSIONS_COL]].sort_values(DATE_COL).reset_index(drop=True)
    output["web_sessions_lag_7"] = output[SESSIONS_COL].shift(7)
    output["web_sessions_roll_mean_30"] = output[SESSIONS_COL].shift(1).rolling(window=30, min_periods=30).mean()
    output["web_sessions_growth"] = safe_divide(
        output[SESSIONS_COL].shift(1),
        output[SESSIONS_COL].shift(7),
    ) - 1.0
    return output[[DATE_COL] + TRAFFIC_FEATURES]


def build_static_features(df: pd.DataFrame) -> pd.DataFrame:
    calendar = build_calendar_features(df[DATE_COL])
    traffic = build_traffic_features(df)
    return (
        df[[DATE_COL, SESSIONS_COL]]
        .merge(calendar, on=DATE_COL, how="left", validate="one_to_one")
        .merge(traffic, on=DATE_COL, how="left", validate="one_to_one")
    )


def add_historical_conversion_features(df: pd.DataFrame) -> pd.DataFrame:
    output = df.sort_values(DATE_COL).reset_index(drop=True).copy()
    shifted_conversion = output[TARGET_COL].shift(1)
    output["conversion_lag_7"] = output[TARGET_COL].shift(7)
    output["conversion_lag_14"] = output[TARGET_COL].shift(14)
    output["conversion_lag_30"] = output[TARGET_COL].shift(30)
    output["conversion_lag_365"] = output[TARGET_COL].shift(365)
    output["conversion_roll_mean_7"] = shifted_conversion.rolling(window=7, min_periods=7).mean()
    output["conversion_roll_mean_30"] = shifted_conversion.rolling(window=30, min_periods=30).mean()
    return output


def build_model_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    static_features = build_static_features(df)
    table = (
        df[[DATE_COL, ORDERS_COL, SESSIONS_COL, TARGET_COL]]
        .merge(static_features, on=[DATE_COL, SESSIONS_COL], how="left", validate="one_to_one")
    )
    table = add_historical_conversion_features(table)
    return table, static_features


def make_training_matrix(
    model_table: pd.DataFrame,
    feature_columns: list[str],
    train_start_inclusive: pd.Timestamp | None,
    train_end_exclusive: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    table = model_table.copy()
    if train_start_inclusive is not None:
        table = table[table[DATE_COL] >= train_start_inclusive].copy()
    table = table[table[DATE_COL] < train_end_exclusive].copy()

    clean = table.dropna(subset=feature_columns + [TARGET_COL]).reset_index(drop=True)
    X = clean[feature_columns].copy()
    y = clean[TARGET_COL].copy()
    medians = X.median(numeric_only=True)
    return X, y, clean, medians


def build_weighted_sample_weights(clean: pd.DataFrame) -> np.ndarray:
    dates = pd.to_datetime(clean[DATE_COL])
    return np.where(dates < RECENT_2019_START, 0.3, 1.0).astype(float)


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    reporter: Reporter,
    sample_weight: np.ndarray | None = None,
) -> Any:
    import lightgbm as lgb

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.03,
        "max_depth": 6,
        "num_leaves": 31,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "feature_fraction": 0.9,
        "seed": RANDOM_STATE,
        "verbosity": -1,
        "force_col_wise": True,
    }
    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        weight=sample_weight,
        feature_name=X_train.columns.tolist(),
        free_raw_data=False,
    )
    model = lgb.train(params=params, train_set=train_data, num_boost_round=1500)
    reporter.logger.info(
        "Trained LightGBM rows=%s features=%s weighted=%s",
        len(X_train),
        X_train.shape[1],
        sample_weight is not None,
    )
    return model


def train_hist_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: np.ndarray | None = None,
) -> Any:
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError as exc:
        raise ImportError("scikit-learn fallback is not installed.") from exc

    model = HistGradientBoostingRegressor(
        learning_rate=0.03,
        max_iter=1500,
        max_leaf_nodes=31,
        random_state=RANDOM_STATE,
    )
    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
    return model


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    reporter: Reporter,
    sample_weight: np.ndarray | None = None,
) -> tuple[Any, str]:
    if base.lightgbm_available():
        return train_lightgbm(X_train, y_train, reporter, sample_weight=sample_weight), "lightgbm"
    reporter.emit("LightGBM unavailable; using HistGradientBoostingRegressor fallback")
    return train_hist_gradient_boosting(X_train, y_train, sample_weight=sample_weight), "hist_gradient_boosting"


def compute_conversion_features_from_history(history: pd.Series, forecast_date: pd.Timestamp) -> dict[str, float]:
    past_history = history[history.index < forecast_date].sort_index()

    def lag_value(days: int) -> float:
        return float(history.get(forecast_date - pd.Timedelta(days=days), np.nan))

    def roll_mean(window: int) -> float:
        values = past_history.tail(window)
        return float(values.mean()) if len(values) == window else np.nan

    return {
        "conversion_lag_7": lag_value(7),
        "conversion_lag_14": lag_value(14),
        "conversion_lag_30": lag_value(30),
        "conversion_lag_365": lag_value(365),
        "conversion_roll_mean_7": roll_mean(7),
        "conversion_roll_mean_30": roll_mean(30),
    }


def recursive_predict_conversion(
    model: Any,
    feature_columns: list[str],
    static_features: pd.DataFrame,
    initial_conversion_history: pd.Series,
    prediction_dates: pd.Series,
    feature_medians: pd.Series,
) -> np.ndarray:
    static_by_date = static_features.set_index(DATE_COL).sort_index()
    history = pd.to_numeric(initial_conversion_history, errors="coerce").sort_index().copy()
    predictions: list[float] = []

    for forecast_date in pd.to_datetime(prediction_dates):
        if forecast_date not in static_by_date.index:
            raise ValueError(f"Missing static features for forecast date {forecast_date.date()}")

        row = static_by_date.loc[forecast_date].to_dict()
        row.update(compute_conversion_features_from_history(history, forecast_date))
        X_row = pd.DataFrame([row], columns=feature_columns)
        X_row = X_row.apply(pd.to_numeric, errors="coerce").fillna(feature_medians).fillna(0.0)

        prediction = float(model.predict(X_row)[0])
        prediction = float(np.clip(prediction, 0.0, 1.0))
        predictions.append(prediction)
        history.loc[forecast_date] = prediction

    return np.asarray(predictions, dtype=float)


def evaluate_one_model(
    model_name: str,
    model_table: pd.DataFrame,
    static_features: pd.DataFrame,
    validation_dates: pd.Series,
    actual_conversion: pd.Series,
    actual_orders: pd.Series,
    actual_sessions: pd.Series,
    initial_conversion_history: pd.Series,
    train_start_inclusive: pd.Timestamp | None,
    reporter: Reporter,
    weighted: bool = False,
) -> dict[str, Any]:
    X_train, y_train, clean, medians = make_training_matrix(
        model_table=model_table,
        feature_columns=FEATURE_COLUMNS,
        train_start_inclusive=train_start_inclusive,
        train_end_exclusive=TRAIN_CUTOFF,
    )
    sample_weight = build_weighted_sample_weights(clean) if weighted else None
    reporter.emit(
        f"Training {model_name}: rows={len(X_train):,}, features={len(FEATURE_COLUMNS)}, "
        f"train range={clean[DATE_COL].min().date()} -> {clean[DATE_COL].max().date()}"
    )
    model, model_type = train_model(X_train, y_train, reporter, sample_weight=sample_weight)

    predicted_conversion = recursive_predict_conversion(
        model=model,
        feature_columns=FEATURE_COLUMNS,
        static_features=static_features,
        initial_conversion_history=initial_conversion_history,
        prediction_dates=validation_dates,
        feature_medians=medians,
    )
    predicted_orders = np.maximum(0.0, predicted_conversion * actual_sessions.to_numpy(dtype=float))

    conversion_metrics = base.evaluate_predictions(actual_conversion, predicted_conversion)
    orders_metrics = base.evaluate_predictions(actual_orders, predicted_orders)
    return {
        "model": model_name,
        "model_object": model,
        "model_type": model_type,
        "X_train": X_train,
        "y_train": y_train,
        "predicted_conversion": predicted_conversion,
        "predicted_orders": predicted_orders,
        "conversion_metrics": conversion_metrics,
        "orders_metrics": orders_metrics,
        "train_rows": len(X_train),
    }


def build_comparison_frame(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.append(
            {
                "model": result["model"],
                "train_rows": result["train_rows"],
                "conversion_MAE": result["conversion_metrics"]["MAE"],
                "conversion_RMSE": result["conversion_metrics"]["RMSE"],
                "conversion_R2": result["conversion_metrics"]["R2"],
                "orders_MAE": result["orders_metrics"]["MAE"],
                "orders_RMSE": result["orders_metrics"]["RMSE"],
                "orders_R2": result["orders_metrics"]["R2"],
                "orders_rmse_vs_direct_baseline": result["orders_metrics"]["RMSE"] - DIRECT_ORDERS_BASELINE_RMSE,
            }
        )
    return pd.DataFrame(rows).sort_values(["orders_RMSE", "conversion_RMSE"]).reset_index(drop=True)


def save_validation_predictions(
    validation_dates: pd.Series,
    actual_conversion: pd.Series,
    actual_orders: pd.Series,
    actual_sessions: pd.Series,
    results: list[dict[str, Any]],
    best_model_name: str,
    path: Path = VALIDATION_PREDICTIONS_PATH,
) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            DATE_COL: pd.to_datetime(validation_dates).reset_index(drop=True),
            "web_sessions": actual_sessions.to_numpy(dtype=float),
            "actual_conversion_rate": actual_conversion.to_numpy(dtype=float),
            "actual_orders_count": actual_orders.to_numpy(dtype=float),
            "best_model": best_model_name,
        }
    )
    for result in results:
        suffix = result["model"].lower()
        output[f"predicted_conversion_{suffix}"] = result["predicted_conversion"]
        output[f"predicted_orders_{suffix}"] = result["predicted_orders"]
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    return output


def build_feature_importance_frame(results: list[dict[str, Any]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for result in results:
        importance = base.get_feature_importance(
            model=result["model_object"],
            model_type=result["model_type"],
            feature_columns=FEATURE_COLUMNS,
            X_ref=result["X_train"],
            y_ref=result["y_train"],
            baseline_rmse=result["conversion_metrics"]["RMSE"],
        ).copy()
        importance.insert(0, "model", result["model"])
        importance["orders_validation_rmse"] = result["orders_metrics"]["RMSE"]
        frames.append(importance)
    return pd.concat(frames, ignore_index=True)


def run() -> None:
    logger = setup_logging()
    reporter = Reporter(logger)

    reporter.emit("Conversion-Rate Regime Window Experiment (No Promo)")
    reporter.emit("================================================")
    reporter.emit("")

    reporter.emit("1. Load data and build no-promo conversion model table")
    df = load_daily_feature_table(TRAIN_DATA_PATH)
    model_table, static_features = build_model_table(df)
    reporter.emit(f"Loaded dataset: {TRAIN_DATA_PATH} | shape={df.shape}")
    reporter.emit(f"Date range: {df[DATE_COL].min().date()} -> {df[DATE_COL].max().date()}")
    reporter.emit(f"Model table shape: {model_table.shape}")
    reporter.emit(f"Feature count: {len(FEATURE_COLUMNS)}")

    validation_dates = df[(df[DATE_COL] >= TRAIN_CUTOFF) & (df[DATE_COL] <= VALIDATION_END)][DATE_COL]
    actual_conversion = df.set_index(DATE_COL).loc[validation_dates, TARGET_COL].reset_index(drop=True)
    actual_orders = df.set_index(DATE_COL).loc[validation_dates, ORDERS_COL].reset_index(drop=True)
    actual_sessions = df.set_index(DATE_COL).loc[validation_dates, SESSIONS_COL].reset_index(drop=True)
    initial_conversion_history = df[df[DATE_COL] < TRAIN_CUTOFF].set_index(DATE_COL)[TARGET_COL]

    reporter.emit("")
    reporter.emit("2. Train FULL vs RECENT_2019 vs RECENT_2020 vs WEIGHTED")
    specs = [
        ("FULL", None, False),
        ("RECENT_2019", RECENT_2019_START, False),
        ("RECENT_2020", RECENT_2020_START, False),
        ("WEIGHTED", None, True),
    ]
    results: list[dict[str, Any]] = []
    for model_name, start_date, weighted in specs:
        results.append(
            evaluate_one_model(
                model_name=model_name,
                model_table=model_table,
                static_features=static_features,
                validation_dates=validation_dates,
                actual_conversion=actual_conversion,
                actual_orders=actual_orders,
                actual_sessions=actual_sessions,
                initial_conversion_history=initial_conversion_history,
                train_start_inclusive=start_date,
                reporter=reporter,
                weighted=weighted,
            )
        )

    comparison_df = build_comparison_frame(results)
    comparison_df.to_csv(COMPARISON_PATH, index=False)
    best_model_name = str(comparison_df.iloc[0]["model"])
    best_result = next(result for result in results if result["model"] == best_model_name)
    reporter.emit_frame("Model comparison:", comparison_df)

    reporter.emit("")
    reporter.emit("3. Save validation predictions and feature importance")
    validation_output = save_validation_predictions(
        validation_dates=validation_dates,
        actual_conversion=actual_conversion,
        actual_orders=actual_orders,
        actual_sessions=actual_sessions,
        results=results,
        best_model_name=best_model_name,
        path=VALIDATION_PREDICTIONS_PATH,
    )
    reporter.emit(f"Saved validation predictions: {VALIDATION_PREDICTIONS_PATH} | shape={validation_output.shape}")

    importance_df = build_feature_importance_frame(results)
    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    reporter.emit(f"Saved feature importance: {FEATURE_IMPORTANCE_PATH}")
    reporter.emit_frame(
        f"Top 20 features for {best_model_name}:",
        importance_df[importance_df["model"] == best_model_name].head(20),
    )

    reporter.emit("")
    reporter.emit("4. Final summary")
    reporter.emit(
        f"Best training window by reconstructed orders RMSE: {best_model_name} "
        f"(RMSE={best_result['orders_metrics']['RMSE']:,.4f})"
    )
    full_conversion_rmse = next(result for result in results if result["model"] == "FULL")["conversion_metrics"]["RMSE"]
    recent2019_conversion_rmse = next(result for result in results if result["model"] == "RECENT_2019")["conversion_metrics"]["RMSE"]
    recent2020_conversion_rmse = next(result for result in results if result["model"] == "RECENT_2020")["conversion_metrics"]["RMSE"]
    reporter.emit(
        "Does post-2019 training improve conversion prediction vs FULL? "
        f"RECENT_2019={recent2019_conversion_rmse < full_conversion_rmse}, "
        f"RECENT_2020={recent2020_conversion_rmse < full_conversion_rmse}"
    )
    reporter.emit(
        "Does reconstructed orders beat direct orders model (RMSE 26.2598)? "
        f"{best_result['orders_metrics']['RMSE'] < DIRECT_ORDERS_BASELINE_RMSE}"
    )
    reporter.emit(
        f"Best conversion metrics: MAE={best_result['conversion_metrics']['MAE']:.6f}, "
        f"RMSE={best_result['conversion_metrics']['RMSE']:.6f}, "
        f"R2={best_result['conversion_metrics']['R2']:.6f}"
    )
    reporter.emit(
        f"Best reconstructed orders metrics: MAE={best_result['orders_metrics']['MAE']:,.4f}, "
        f"RMSE={best_result['orders_metrics']['RMSE']:,.4f}, "
        f"R2={best_result['orders_metrics']['R2']:.6f}"
    )
    reporter.emit(
        "This no-promo variant isolates pure calendar + conversion-lag + traffic effects to test whether promo "
        "features were helping or hurting the previous regime-window conversion experiment."
    )
    reporter.emit(
        "Evaluation assumption: reconstructed orders use known 2022 web_sessions to isolate conversion prediction "
        "quality, without introducing a future sessions model."
    )
    reporter.save(REPORT_PATH)


if __name__ == "__main__":
    run()
