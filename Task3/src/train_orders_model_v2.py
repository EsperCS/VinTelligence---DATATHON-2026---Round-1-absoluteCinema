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
VALIDATION_PREDICTIONS_PATH = DATA_DIR / "orders_v2_validation_predictions.csv"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "orders_v2_feature_importance.csv"
METRICS_PATH = LOG_DIR / "orders_v2_metrics.txt"
LOG_FILE = LOG_DIR / "train_orders_model_v2.log"

DATE_COL = "Date"
TARGET_COL = "orders_count"
TRAIN_CUTOFF = pd.Timestamp("2022-01-01")
VALIDATION_END = pd.Timestamp("2022-12-31")
RECENT_START = pd.Timestamp("2019-01-01")
RANDOM_STATE = 42
EPSILON = 1e-6

CALENDAR_FEATURES = [
    "day_of_week",
    "day_of_year",
    "week_of_year",
    "month",
    "is_weekend",
    "is_month_end",
    "time_index",
    "post_2019_flag",
    "years_since_2019",
]

PROMOTION_FEATURES = [
    "calendar_any_promo",
    "calendar_active_promo_count",
    "calendar_avg_discount_value",
    "calendar_max_discount_value",
]

PROMOTION_PHASE_FEATURES = [
    "promo_day_number",
    "promo_days_remaining",
    "promo_duration",
    "promo_progress_ratio",
]

CAMPAIGN_FEATURES = [
    "promotion_campaign_index",
]

TRAFFIC_FEATURES = [
    "web_sessions_lag_1",
    "web_sessions_lag_7",
    "web_sessions_roll_mean_7",
    "web_sessions_roll_mean_30",
    "web_sessions_to_roll_ratio",
    "web_sessions_growth",
    "web_engagement",
]

PROMO_INTERACTION_FEATURES = [
    "promo_x_traffic",
    "discount_x_traffic",
]

ORDER_LAG_FEATURES = [
    "orders_lag_7",
    "orders_lag_14",
    "orders_lag_30",
    "orders_lag_365",
]

ORDER_ROLL_FEATURES = [
    "orders_roll_mean_7",
    "orders_roll_mean_30",
]

ORDER_SPIKE_FEATURES = [
    "orders_lag7_to_roll30",
    "orders_volatility_30",
    "orders_momentum",
]

STATIC_FEATURE_COLUMNS = (
    CALENDAR_FEATURES
    + PROMOTION_FEATURES
    + PROMOTION_PHASE_FEATURES
    + CAMPAIGN_FEATURES
    + TRAFFIC_FEATURES
    + PROMO_INTERACTION_FEATURES
)

FEATURE_COLUMNS = STATIC_FEATURE_COLUMNS + ORDER_LAG_FEATURES + ORDER_ROLL_FEATURES + ORDER_SPIKE_FEATURES


class Reporter:
    """Print, log, and save experiment notes."""

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

    logger = logging.getLogger("train_orders_model_v2")
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
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    for column in ["web_sessions", "web_avg_session_duration_sec"]:
        if column not in df.columns:
            df[column] = 0.0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    df = df.sort_values(DATE_COL).reset_index(drop=True)
    if df[DATE_COL].isna().any():
        raise ValueError("Date column contains invalid timestamps")
    if df[TARGET_COL].isna().any():
        raise ValueError("orders_count contains missing or invalid values")
    return df


def build_calendar_features(dates: pd.Series, min_date: pd.Timestamp) -> pd.DataFrame:
    output = pd.DataFrame({DATE_COL: pd.to_datetime(dates).sort_values().unique()})
    output["day_of_week"] = output[DATE_COL].dt.dayofweek.astype(int)
    output["day_of_year"] = output[DATE_COL].dt.dayofyear.astype(int)
    output["week_of_year"] = output[DATE_COL].dt.isocalendar().week.astype(int)
    output["month"] = output[DATE_COL].dt.month.astype(int)
    output["is_weekend"] = output["day_of_week"].isin([5, 6]).astype(int)
    output["is_month_end"] = output[DATE_COL].dt.is_month_end.astype(int)
    output["time_index"] = (output[DATE_COL] - pd.Timestamp(min_date)).dt.days.astype(int)
    output["post_2019_flag"] = (output[DATE_COL] >= RECENT_START).astype(int)
    output["years_since_2019"] = np.maximum(0, output[DATE_COL].dt.year.astype(int) - 2019)
    return output[[DATE_COL] + CALENDAR_FEATURES]


def build_traffic_features(df: pd.DataFrame) -> pd.DataFrame:
    output = df[[DATE_COL, "web_sessions", "web_avg_session_duration_sec"]].sort_values(DATE_COL).reset_index(drop=True)

    output["web_sessions_lag_1"] = output["web_sessions"].shift(1)
    output["web_sessions_lag_7"] = output["web_sessions"].shift(7)
    shifted_sessions = output["web_sessions"].shift(1)
    shifted_duration = output["web_avg_session_duration_sec"].shift(1)

    output["web_sessions_roll_mean_7"] = shifted_sessions.rolling(window=7, min_periods=7).mean()
    output["web_sessions_roll_mean_30"] = shifted_sessions.rolling(window=30, min_periods=30).mean()
    output["web_sessions_to_roll_ratio"] = safe_divide(
        output["web_sessions_lag_1"],
        output["web_sessions_roll_mean_30"],
    )
    output["web_sessions_growth"] = safe_divide(output["web_sessions_lag_1"], output["web_sessions_lag_7"]) - 1.0
    output["web_engagement"] = output["web_sessions_lag_1"] * shifted_duration

    return output[[DATE_COL] + TRAFFIC_FEATURES]


def add_promo_interactions(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["promo_x_traffic"] = output["calendar_any_promo"] * output["web_sessions_roll_mean_7"]
    output["discount_x_traffic"] = output["calendar_avg_discount_value"] * output["web_sessions_roll_mean_7"]
    return output


def add_orders_history_features(df: pd.DataFrame) -> pd.DataFrame:
    output = df.sort_values(DATE_COL).reset_index(drop=True).copy()

    output["orders_lag_7"] = output[TARGET_COL].shift(7)
    output["orders_lag_14"] = output[TARGET_COL].shift(14)
    output["orders_lag_30"] = output[TARGET_COL].shift(30)
    output["orders_lag_365"] = output[TARGET_COL].shift(365)

    shifted_orders = output[TARGET_COL].shift(1)
    output["orders_roll_mean_7"] = shifted_orders.rolling(window=7, min_periods=7).mean()
    output["orders_roll_mean_30"] = shifted_orders.rolling(window=30, min_periods=30).mean()
    orders_roll_std_30 = shifted_orders.rolling(window=30, min_periods=30).std()

    output["orders_lag7_to_roll30"] = safe_divide(output["orders_lag_7"], output["orders_roll_mean_30"])
    output["orders_volatility_30"] = safe_divide(orders_roll_std_30, output["orders_roll_mean_30"])
    output["orders_momentum"] = output["orders_lag_7"] - output["orders_lag_30"]
    return output


def build_model_table(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    calendar = build_calendar_features(df[DATE_COL], df[DATE_COL].min())
    promotions = orders_base.build_promotion_features(df[DATE_COL], orders_base.PROMOTIONS_PATH, logger)
    traffic = build_traffic_features(df)

    table = (
        df[[DATE_COL, TARGET_COL]]
        .merge(calendar, on=DATE_COL, how="left", validate="one_to_one")
        .merge(promotions, on=DATE_COL, how="left", validate="one_to_one")
        .merge(traffic, on=DATE_COL, how="left", validate="one_to_one")
    )
    fill_zero_columns = CALENDAR_FEATURES + PROMOTION_FEATURES + PROMOTION_PHASE_FEATURES + CAMPAIGN_FEATURES
    table[fill_zero_columns] = table[fill_zero_columns].fillna(0.0)
    table = add_promo_interactions(table)
    table = add_orders_history_features(table)
    return table


def make_training_matrix(
    model_table: pd.DataFrame,
    feature_columns: list[str],
    train_start_inclusive: pd.Timestamp | None,
    train_end_exclusive: pd.Timestamp | None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    table = model_table.copy()
    if train_start_inclusive is not None:
        table = table[table[DATE_COL] >= train_start_inclusive].copy()
    if train_end_exclusive is not None:
        table = table[table[DATE_COL] < train_end_exclusive].copy()

    clean = table.dropna(subset=feature_columns + [TARGET_COL]).reset_index(drop=True)
    X = clean[feature_columns].copy()
    y = clean[TARGET_COL].copy()
    feature_medians = X.median(numeric_only=True)
    return X, y, clean, feature_medians


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


def compute_orders_features_from_history(history: pd.Series, forecast_date: pd.Timestamp) -> dict[str, float]:
    past_history = history[history.index < forecast_date].sort_index()

    def lag_value(days: int) -> float:
        return float(history.get(forecast_date - pd.Timedelta(days=days), np.nan))

    def roll_mean(window: int) -> float:
        values = past_history.tail(window)
        return float(values.mean()) if len(values) == window else np.nan

    def roll_std(window: int) -> float:
        values = past_history.tail(window)
        return float(values.std(ddof=1)) if len(values) == window else np.nan

    orders_lag_7 = lag_value(7)
    orders_lag_30 = lag_value(30)
    orders_roll_mean_30 = roll_mean(30)
    orders_roll_std_30 = roll_std(30)

    return {
        "orders_lag_7": orders_lag_7,
        "orders_lag_14": lag_value(14),
        "orders_lag_30": orders_lag_30,
        "orders_lag_365": lag_value(365),
        "orders_roll_mean_7": roll_mean(7),
        "orders_roll_mean_30": orders_roll_mean_30,
        "orders_lag7_to_roll30": safe_divide(orders_lag_7, orders_roll_mean_30),
        "orders_volatility_30": safe_divide(orders_roll_std_30, orders_roll_mean_30),
        "orders_momentum": orders_lag_7 - orders_lag_30 if pd.notna(orders_lag_7) and pd.notna(orders_lag_30) else np.nan,
    }


def recursive_predict_orders(
    model: Any,
    feature_columns: list[str],
    static_features: pd.DataFrame,
    initial_history: pd.Series,
    prediction_dates: pd.Series,
    feature_medians: pd.Series,
) -> np.ndarray:
    static_by_date = static_features.set_index(DATE_COL).sort_index()
    history = pd.to_numeric(initial_history, errors="coerce").sort_index().copy()
    predictions: list[float] = []

    for forecast_date in pd.to_datetime(prediction_dates):
        if forecast_date not in static_by_date.index:
            raise ValueError(f"Missing static features for forecast date {forecast_date.date()}")

        row = static_by_date.loc[forecast_date].to_dict()
        row.update(compute_orders_features_from_history(history, forecast_date))
        X_row = pd.DataFrame([row], columns=feature_columns)
        X_row = X_row.apply(pd.to_numeric, errors="coerce").fillna(feature_medians).fillna(0.0)

        prediction = float(model.predict(X_row)[0])
        prediction = max(0.0, prediction)
        predictions.append(prediction)
        history.loc[forecast_date] = prediction

    return np.asarray(predictions, dtype=float)


def build_sample_weights(clean: pd.DataFrame) -> np.ndarray:
    return np.where(clean[DATE_COL] >= RECENT_START, 2.0, 1.0).astype(float)


def save_validation_predictions(
    dates: pd.Series,
    actual: pd.Series,
    prediction_map: dict[str, np.ndarray],
    best_model_name: str,
    path: Path = VALIDATION_PREDICTIONS_PATH,
) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            DATE_COL: pd.to_datetime(dates).reset_index(drop=True),
            "actual_orders_count": actual.to_numpy(dtype=float),
        }
    )
    for model_name, prediction in prediction_map.items():
        output[f"predicted_{model_name.lower()}"] = np.asarray(prediction, dtype=float)
    output["best_model"] = best_model_name
    output["predicted_best_model"] = np.asarray(prediction_map[best_model_name], dtype=float)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    return output


def summarize_group_importance(importance: pd.DataFrame, prefixes: tuple[str, ...]) -> list[str]:
    top20 = importance.head(20)["feature"].astype(str).tolist()
    return [feature for feature in top20 if feature.startswith(prefixes)]


def run() -> None:
    logger = setup_logging()
    reporter = Reporter(logger)

    reporter.emit("Orders Forecasting Model V2")
    reporter.emit("==========================")
    reporter.emit("")

    reporter.emit("1. Load dataset and build richer feature table")
    df = load_daily_feature_table(TRAIN_DATA_PATH)
    model_table = build_model_table(df, logger)
    static_features = model_table[[DATE_COL] + STATIC_FEATURE_COLUMNS].copy()
    reporter.emit(f"Loaded dataset: {TRAIN_DATA_PATH} | shape={df.shape}")
    reporter.emit(f"Date range: {df[DATE_COL].min().date()} -> {df[DATE_COL].max().date()}")
    reporter.emit(f"Model table shape: {model_table.shape}")
    reporter.emit(f"Feature count: {len(FEATURE_COLUMNS)}")

    validation_dates = df[(df[DATE_COL] >= TRAIN_CUTOFF) & (df[DATE_COL] <= VALIDATION_END)][DATE_COL]
    actual = df.set_index(DATE_COL).loc[validation_dates, TARGET_COL].reset_index(drop=True)
    initial_history = df[df[DATE_COL] < TRAIN_CUTOFF].set_index(DATE_COL)[TARGET_COL]

    model_specs = [
        ("FULL", None, None),
        ("RECENT", RECENT_START, None),
        ("WEIGHTED", None, "recent_weighted"),
    ]

    comparison_rows: list[dict[str, Any]] = []
    prediction_map: dict[str, np.ndarray] = {}
    trained_results: dict[str, dict[str, Any]] = {}

    reporter.emit("")
    reporter.emit("2. Train FULL vs RECENT vs WEIGHTED with recursive validation 2022")
    for model_name, start_date, weighting_mode in model_specs:
        X_train, y_train, clean, feature_medians = make_training_matrix(
            model_table=model_table,
            feature_columns=FEATURE_COLUMNS,
            train_start_inclusive=start_date,
            train_end_exclusive=TRAIN_CUTOFF,
        )
        sample_weight = build_sample_weights(clean) if weighting_mode == "recent_weighted" else None
        reporter.emit(
            f"Training {model_name}: rows={len(X_train):,}, features={len(FEATURE_COLUMNS)}, "
            f"train range={clean[DATE_COL].min().date()} -> {clean[DATE_COL].max().date()}"
        )

        model, model_type = train_model(X_train, y_train, reporter, sample_weight=sample_weight)
        predictions = recursive_predict_orders(
            model=model,
            feature_columns=FEATURE_COLUMNS,
            static_features=static_features,
            initial_history=initial_history,
            prediction_dates=validation_dates,
            feature_medians=feature_medians,
        )
        metrics = base.evaluate_predictions(actual, predictions)
        comparison_rows.append({"model": model_name, **metrics, "train_rows": len(X_train)})
        prediction_map[model_name] = predictions
        trained_results[model_name] = {
            "model_object": model,
            "model_type": model_type,
            "X_train": X_train,
            "y_train": y_train,
            "metrics": metrics,
        }

    comparison_df = pd.DataFrame(comparison_rows).sort_values(["RMSE", "MAE"]).reset_index(drop=True)
    best_model_name = str(comparison_df.iloc[0]["model"])
    reporter.emit_frame("Model comparison:", comparison_df)
    reporter.emit(f"Best model by RMSE: {best_model_name}")

    reporter.emit("")
    reporter.emit("3. Save validation predictions")
    validation_output = save_validation_predictions(
        dates=validation_dates,
        actual=actual,
        prediction_map=prediction_map,
        best_model_name=best_model_name,
        path=VALIDATION_PREDICTIONS_PATH,
    )
    reporter.emit(f"Saved validation predictions: {VALIDATION_PREDICTIONS_PATH} | shape={validation_output.shape}")

    reporter.emit("")
    reporter.emit("4. Save feature importance by model")
    importance_frames: list[pd.DataFrame] = []
    for row in comparison_df.itertuples(index=False):
        model_name = row.model
        trained = trained_results[model_name]
        importance = base.get_feature_importance(
            model=trained["model_object"],
            model_type=trained["model_type"],
            feature_columns=FEATURE_COLUMNS,
            X_ref=trained["X_train"],
            y_ref=trained["y_train"],
            baseline_rmse=trained["metrics"]["RMSE"],
        ).copy()
        importance.insert(0, "model", model_name)
        importance["validation_rmse"] = trained["metrics"]["RMSE"]
        importance_frames.append(importance)

    importance_df = pd.concat(importance_frames, ignore_index=True)
    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    reporter.emit(f"Saved feature importance: {FEATURE_IMPORTANCE_PATH}")

    best_importance = (
        importance_df[importance_df["model"] == best_model_name]
        .sort_values(["importance_gain", "importance_split"], ascending=False)
        .reset_index(drop=True)
    )
    reporter.emit_frame(f"Top 20 features for {best_model_name}:", best_importance.head(20))

    traffic_top = summarize_group_importance(best_importance, ("web_",))
    promo_top = summarize_group_importance(best_importance, ("calendar_", "promo_", "promotion_campaign_index"))
    spike_top = summarize_group_importance(best_importance, ("orders_lag7_to_roll30", "orders_volatility_30", "orders_momentum", "orders_lag_", "orders_roll_"))

    reporter.emit("")
    reporter.emit("5. Final summary")
    reporter.emit(
        "FULL vs RECENT vs WEIGHTED: "
        + " | ".join(
            f"{row.model}: MAE={row.MAE:,.4f}, RMSE={row.RMSE:,.4f}, R2={row.R2:.6f}"
            for row in comparison_df.itertuples(index=False)
        )
    )
    reporter.emit(f"Best model: {best_model_name}")
    reporter.emit(f"Top traffic-related features in best model Top 20: {traffic_top}")
    reporter.emit(f"Top promotion-related features in best model Top 20: {promo_top}")
    reporter.emit(f"Top orders spike/history features in best model Top 20: {spike_top}")
    reporter.emit(
        "Leakage note: model uses only lagged orders history, lagged/rolling web traffic, and known promotion calendar features."
    )
    reporter.save(METRICS_PATH)


if __name__ == "__main__":
    run()
