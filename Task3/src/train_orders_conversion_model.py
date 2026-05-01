from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import train_final_model as base
import train_orders_model as orders_base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "log"

TRAIN_DATA_PATH = DATA_DIR / "daily_feature_table.csv"
VALIDATION_PREDICTIONS_PATH = DATA_DIR / "orders_conversion_validation_predictions.csv"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "orders_conversion_feature_importance.csv"
METRICS_PATH = LOG_DIR / "orders_conversion_metrics.txt"
LOG_FILE = LOG_DIR / "train_orders_conversion_model.log"

DATE_COL = "Date"
ORDERS_COL = "orders_count"
SESSIONS_COL = "web_sessions"
DURATION_COL = "web_avg_session_duration_sec"
TARGET_COL = "conversion_rate"
TRAIN_CUTOFF = pd.Timestamp("2022-01-01")
VALIDATION_END = pd.Timestamp("2022-12-31")
RANDOM_STATE = 42
EPSILON = 1e-6
BASELINE_ORDERS_RMSE = 26.2598

TRAFFIC_FEATURES = [
    "web_sessions_lag_1",
    "web_sessions_lag_7",
    "web_sessions_roll_mean_7",
    "web_sessions_roll_mean_30",
    "web_sessions_to_roll_ratio",
    "web_engagement",
    "web_sessions_growth",
]

PROMOTION_FEATURES = [
    "calendar_any_promo",
    "calendar_avg_discount_value",
    "promotion_campaign_index",
]

CONVERSION_FEATURES = [
    "conversion_lag_7",
    "conversion_lag_14",
    "conversion_roll_mean_7",
    "conversion_roll_mean_30",
]

FEATURE_COLUMNS = TRAFFIC_FEATURES + PROMOTION_FEATURES + CONVERSION_FEATURES


class Reporter:
    """Print, log, and persist a compact report."""

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
        self.logger.info("Saved metrics report to %s", path)


def setup_logging(log_file: Path = LOG_FILE) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train_orders_conversion_model")
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
    df[DURATION_COL] = pd.to_numeric(df.get(DURATION_COL, 0), errors="coerce").fillna(0.0)
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    if df[DATE_COL].isna().any():
        raise ValueError("Date column contains invalid timestamps")
    if df[ORDERS_COL].isna().any():
        raise ValueError("orders_count contains invalid values")

    df[TARGET_COL] = np.where(df[SESSIONS_COL] > 0, df[ORDERS_COL] / df[SESSIONS_COL], 0.0)
    return df


def build_traffic_features(df: pd.DataFrame) -> pd.DataFrame:
    output = df[[DATE_COL, SESSIONS_COL, DURATION_COL]].sort_values(DATE_COL).reset_index(drop=True)
    shifted_sessions = output[SESSIONS_COL].shift(1)
    shifted_duration = output[DURATION_COL].shift(1)

    output["web_sessions_lag_1"] = shifted_sessions
    output["web_sessions_lag_7"] = output[SESSIONS_COL].shift(7)
    output["web_sessions_roll_mean_7"] = shifted_sessions.rolling(window=7, min_periods=7).mean()
    output["web_sessions_roll_mean_30"] = shifted_sessions.rolling(window=30, min_periods=30).mean()
    output["web_sessions_to_roll_ratio"] = safe_divide(
        output["web_sessions_lag_1"],
        output["web_sessions_roll_mean_30"],
    )
    output["web_engagement"] = shifted_sessions * shifted_duration
    output["web_sessions_growth"] = safe_divide(
        output["web_sessions_lag_1"],
        output["web_sessions_lag_7"],
    ) - 1.0

    return output[[DATE_COL] + TRAFFIC_FEATURES]


def build_static_features(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    traffic = build_traffic_features(df)
    promo = orders_base.build_promotion_features(df[DATE_COL], orders_base.PROMOTIONS_PATH, logger)
    keep_promo = promo[[DATE_COL] + PROMOTION_FEATURES].copy()
    return (
        df[[DATE_COL, SESSIONS_COL]]
        .merge(traffic, on=DATE_COL, how="left", validate="one_to_one")
        .merge(keep_promo, on=DATE_COL, how="left", validate="one_to_one")
        .fillna({SESSIONS_COL: 0.0, "calendar_any_promo": 0.0, "calendar_avg_discount_value": 0.0, "promotion_campaign_index": 0.0})
    )


def add_historical_conversion_features(df: pd.DataFrame) -> pd.DataFrame:
    output = df.sort_values(DATE_COL).reset_index(drop=True).copy()
    shifted_conversion = output[TARGET_COL].shift(1)
    output["conversion_lag_7"] = output[TARGET_COL].shift(7)
    output["conversion_lag_14"] = output[TARGET_COL].shift(14)
    output["conversion_roll_mean_7"] = shifted_conversion.rolling(window=7, min_periods=7).mean()
    output["conversion_roll_mean_30"] = shifted_conversion.rolling(window=30, min_periods=30).mean()
    return output


def build_model_table(df: pd.DataFrame, logger: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    static_features = build_static_features(df, logger)
    table = (
        df[[DATE_COL, ORDERS_COL, SESSIONS_COL, TARGET_COL]]
        .merge(static_features, on=[DATE_COL, SESSIONS_COL], how="left", validate="one_to_one")
    )
    table = add_historical_conversion_features(table)
    return table, static_features


def make_training_matrix(
    model_table: pd.DataFrame,
    feature_columns: list[str],
    train_end_exclusive: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    table = model_table[model_table[DATE_COL] < train_end_exclusive].copy()
    clean = table.dropna(subset=feature_columns + [TARGET_COL]).reset_index(drop=True)
    X = clean[feature_columns].copy()
    y = clean[TARGET_COL].copy()
    medians = X.median(numeric_only=True)
    return X, y, clean, medians


def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, reporter: Reporter) -> Any:
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
        feature_name=X_train.columns.tolist(),
        free_raw_data=False,
    )
    model = lgb.train(params=params, train_set=train_data, num_boost_round=1500)
    reporter.logger.info("Trained LightGBM rows=%s features=%s", len(X_train), X_train.shape[1])
    return model


def train_hist_gradient_boosting(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
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
    model.fit(X_train, y_train)
    return model


def train_model(X_train: pd.DataFrame, y_train: pd.Series, reporter: Reporter) -> tuple[Any, str]:
    if base.lightgbm_available():
        return train_lightgbm(X_train, y_train, reporter), "lightgbm"
    reporter.emit("LightGBM unavailable; using HistGradientBoostingRegressor fallback")
    return train_hist_gradient_boosting(X_train, y_train), "hist_gradient_boosting"


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
        prediction = max(0.0, prediction)
        predictions.append(prediction)
        history.loc[forecast_date] = prediction

    return np.asarray(predictions, dtype=float)


def save_validation_predictions(
    dates: pd.Series,
    actual_conversion: pd.Series,
    predicted_conversion: np.ndarray,
    actual_orders: pd.Series,
    predicted_orders: np.ndarray,
    actual_sessions: pd.Series,
    path: Path = VALIDATION_PREDICTIONS_PATH,
) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            DATE_COL: pd.to_datetime(dates).reset_index(drop=True),
            "web_sessions": actual_sessions.to_numpy(dtype=float),
            "actual_conversion_rate": actual_conversion.to_numpy(dtype=float),
            "predicted_conversion_rate": np.asarray(predicted_conversion, dtype=float),
            "actual_orders_count": actual_orders.to_numpy(dtype=float),
            "predicted_orders_count": np.asarray(predicted_orders, dtype=float),
        }
    )
    output["orders_error"] = output["actual_orders_count"] - output["predicted_orders_count"]
    output["orders_abs_error"] = output["orders_error"].abs()
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    return output


def run() -> None:
    logger = setup_logging()
    reporter = Reporter(logger)

    reporter.emit("Orders Conversion-Rate Model")
    reporter.emit("============================")
    reporter.emit("")

    reporter.emit("1. Load data and prepare conversion target")
    df = load_daily_feature_table(TRAIN_DATA_PATH)
    model_table, static_features = build_model_table(df, logger)
    reporter.emit(f"Loaded dataset: {TRAIN_DATA_PATH} | shape={df.shape}")
    reporter.emit(f"Date range: {df[DATE_COL].min().date()} -> {df[DATE_COL].max().date()}")
    reporter.emit(f"Model table shape: {model_table.shape}")
    reporter.emit(f"Feature count: {len(FEATURE_COLUMNS)}")
    reporter.emit(
        f"Conversion rate summary: mean={df[TARGET_COL].mean():.6f}, median={df[TARGET_COL].median():.6f}, "
        f"max={df[TARGET_COL].max():.6f}"
    )

    reporter.emit("")
    reporter.emit("2. Train conversion model on pre-2022 and recursively validate on 2022")
    X_train, y_train, train_clean, feature_medians = make_training_matrix(
        model_table,
        FEATURE_COLUMNS,
        TRAIN_CUTOFF,
    )
    reporter.emit(
        f"Train rows after lag cleanup: {len(X_train):,} | "
        f"train range: {train_clean[DATE_COL].min().date()} -> {train_clean[DATE_COL].max().date()}"
    )
    model, model_type = train_model(X_train, y_train, reporter)

    validation_dates = df[(df[DATE_COL] >= TRAIN_CUTOFF) & (df[DATE_COL] <= VALIDATION_END)][DATE_COL]
    actual_conversion = df.set_index(DATE_COL).loc[validation_dates, TARGET_COL].reset_index(drop=True)
    actual_orders = df.set_index(DATE_COL).loc[validation_dates, ORDERS_COL].reset_index(drop=True)
    actual_sessions = df.set_index(DATE_COL).loc[validation_dates, SESSIONS_COL].reset_index(drop=True)
    initial_conversion_history = df[df[DATE_COL] < TRAIN_CUTOFF].set_index(DATE_COL)[TARGET_COL]

    predicted_conversion = recursive_predict_conversion(
        model=model,
        feature_columns=FEATURE_COLUMNS,
        static_features=static_features,
        initial_conversion_history=initial_conversion_history,
        prediction_dates=validation_dates,
        feature_medians=feature_medians,
    )
    predicted_orders = np.maximum(0.0, predicted_conversion * actual_sessions.to_numpy(dtype=float))

    conversion_metrics = base.evaluate_predictions(actual_conversion, predicted_conversion)
    orders_metrics = base.evaluate_predictions(actual_orders, predicted_orders)

    reporter.emit(
        f"Conversion model metrics: MAE={conversion_metrics['MAE']:.6f} | "
        f"RMSE={conversion_metrics['RMSE']:.6f} | R2={conversion_metrics['R2']:.6f}"
    )
    reporter.emit(
        f"Reconstructed orders metrics: MAE={orders_metrics['MAE']:,.4f} | "
        f"RMSE={orders_metrics['RMSE']:,.4f} | R2={orders_metrics['R2']:.6f}"
    )
    reporter.emit(
        f"Baseline orders RMSE reference: {BASELINE_ORDERS_RMSE:,.4f} | "
        f"Delta={orders_metrics['RMSE'] - BASELINE_ORDERS_RMSE:,.4f}"
    )

    validation_output = save_validation_predictions(
        dates=validation_dates,
        actual_conversion=actual_conversion,
        predicted_conversion=predicted_conversion,
        actual_orders=actual_orders,
        predicted_orders=predicted_orders,
        actual_sessions=actual_sessions,
        path=VALIDATION_PREDICTIONS_PATH,
    )
    reporter.emit(f"Saved validation predictions: {VALIDATION_PREDICTIONS_PATH} | shape={validation_output.shape}")

    importance = base.get_feature_importance(
        model=model,
        model_type=model_type,
        feature_columns=FEATURE_COLUMNS,
        X_ref=X_train,
        y_ref=y_train,
        baseline_rmse=orders_metrics["RMSE"],
    )
    importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    reporter.emit(f"Saved feature importance: {FEATURE_IMPORTANCE_PATH}")
    reporter.emit_frame("Top 20 features:", importance.head(20))

    reporter.emit("")
    reporter.emit("3. Final summary")
    reporter.emit(
        f"Conversion target performance: MAE={conversion_metrics['MAE']:.6f}, "
        f"RMSE={conversion_metrics['RMSE']:.6f}, R2={conversion_metrics['R2']:.6f}"
    )
    reporter.emit(
        f"Orders reconstructed performance: MAE={orders_metrics['MAE']:,.4f}, "
        f"RMSE={orders_metrics['RMSE']:,.4f}, R2={orders_metrics['R2']:.6f}"
    )
    reporter.emit(
        "Comparison vs baseline direct-orders model: "
        + (
            f"improved by {- (orders_metrics['RMSE'] - BASELINE_ORDERS_RMSE):,.4f} RMSE"
            if orders_metrics["RMSE"] < BASELINE_ORDERS_RMSE
            else f"worse by {orders_metrics['RMSE'] - BASELINE_ORDERS_RMSE:,.4f} RMSE"
        )
    )
    reporter.emit(
        "Model note: conversion is predicted recursively from lagged conversion, lagged/rolling traffic, "
        "and promotion features; reconstructed orders then multiply predicted conversion by observed web_sessions."
    )
    reporter.save(METRICS_PATH)


if __name__ == "__main__":
    run()
