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
PROMOTIONS_PATH = DATA_DIR / "promotions.csv"
VALIDATION_PREDICTIONS_PATH = DATA_DIR / "orders_validation_predictions.csv"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "orders_feature_importance.csv"
METRICS_PATH = LOG_DIR / "orders_model_metrics.txt"
LOG_FILE = LOG_DIR / "train_orders_model.log"

DATE_COL = "Date"
TARGET_COL = "orders_count"
TRAIN_CUTOFF = pd.Timestamp("2022-01-01")
VALIDATION_END = pd.Timestamp("2022-12-31")
RANDOM_STATE = 42

CALENDAR_FEATURES = [
    "day_of_week",
    "day_of_year",
    "week_of_year",
    "month",
    "is_weekend",
    "is_month_end",
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

FEATURE_COLUMNS = (
    CALENDAR_FEATURES
    + PROMOTION_FEATURES
    + PROMOTION_PHASE_FEATURES
    + CAMPAIGN_FEATURES
    + ORDER_LAG_FEATURES
    + ORDER_ROLL_FEATURES
)


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

    def save(self, path: Path = METRICS_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        self.logger.info("Saved metrics report to %s", path)


def setup_logging(log_file: Path = LOG_FILE) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train_orders_model")
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


def load_daily_feature_table(path: Path = TRAIN_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Training dataset not found: {path}")

    df = pd.read_csv(path, parse_dates=[DATE_COL], low_memory=False)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce").dt.normalize()
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    if df[DATE_COL].isna().any():
        raise ValueError("Date column contains invalid timestamps")
    if df[TARGET_COL].isna().any():
        raise ValueError("orders_count contains missing or invalid values")

    return df


def build_calendar_features(dates: pd.Series) -> pd.DataFrame:
    output = pd.DataFrame({DATE_COL: pd.to_datetime(dates).sort_values().unique()})
    output["day_of_week"] = output[DATE_COL].dt.dayofweek.astype(int)
    output["day_of_year"] = output[DATE_COL].dt.dayofyear.astype(int)
    output["week_of_year"] = output[DATE_COL].dt.isocalendar().week.astype(int)
    output["month"] = output[DATE_COL].dt.month.astype(int)
    output["is_weekend"] = output["day_of_week"].isin([5, 6]).astype(int)
    output["is_month_end"] = output[DATE_COL].dt.is_month_end.astype(int)
    return output[[DATE_COL] + CALENDAR_FEATURES]


def build_promotion_features(
    dates: pd.Series,
    promotions_path: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    calendar = pd.DataFrame({DATE_COL: pd.to_datetime(dates).sort_values().unique()})
    for feature in PROMOTION_FEATURES + PROMOTION_PHASE_FEATURES + CAMPAIGN_FEATURES:
        calendar[feature] = 0.0

    if not promotions_path.exists():
        logger.warning("promotions.csv not found at %s; promotion features default to zero", promotions_path)
        return calendar

    promotions = pd.read_csv(promotions_path, low_memory=False)
    required = {"promo_id", "start_date", "end_date"}
    if not required.issubset(promotions.columns):
        logger.warning("promotions.csv missing required columns; promotion features default to zero")
        return calendar

    promotions["start_date"] = pd.to_datetime(promotions["start_date"], errors="coerce").dt.normalize()
    promotions["end_date"] = pd.to_datetime(promotions["end_date"], errors="coerce").dt.normalize()
    promotions["discount_value"] = pd.to_numeric(promotions.get("discount_value", 0), errors="coerce").fillna(0)
    promotions = promotions.dropna(subset=["start_date", "end_date"]).copy()
    promotions = promotions[promotions["end_date"] >= promotions["start_date"]].copy()
    promotions = promotions.sort_values(["start_date", "end_date", "promo_id"]).reset_index(drop=True)
    promotions["promo_duration"] = (promotions["end_date"] - promotions["start_date"]).dt.days + 1
    promotions["promotion_campaign_index"] = np.arange(1, len(promotions) + 1, dtype=float)

    if promotions.empty:
        return calendar

    min_date = calendar[DATE_COL].min()
    max_date = calendar[DATE_COL].max()
    rows: list[dict[str, Any]] = []

    for row in promotions.itertuples(index=False):
        active_start = max(row.start_date, min_date)
        active_end = min(row.end_date, max_date)
        if active_start > active_end:
            continue

        for active_date in pd.date_range(active_start, active_end, freq="D"):
            promo_day_number = (active_date - row.start_date).days + 1
            promo_days_remaining = (row.end_date - active_date).days
            rows.append(
                {
                    DATE_COL: active_date,
                    "promo_id": row.promo_id,
                    "discount_value": row.discount_value,
                    "promo_day_number": promo_day_number,
                    "promo_days_remaining": promo_days_remaining,
                    "promo_duration": row.promo_duration,
                    "promo_progress_ratio": promo_day_number / row.promo_duration,
                }
            )

    if rows:
        expanded = pd.DataFrame(rows)
        daily = (
            expanded.groupby(DATE_COL, as_index=False)
            .agg(
                calendar_active_promo_count=("promo_id", "nunique"),
                calendar_avg_discount_value=("discount_value", "mean"),
                calendar_max_discount_value=("discount_value", "max"),
                promo_day_number=("promo_day_number", "mean"),
                promo_days_remaining=("promo_days_remaining", "mean"),
                promo_duration=("promo_duration", "mean"),
                promo_progress_ratio=("promo_progress_ratio", "mean"),
            )
        )
        daily["calendar_any_promo"] = (daily["calendar_active_promo_count"] > 0).astype(int)
        calendar = calendar.drop(columns=PROMOTION_FEATURES + PROMOTION_PHASE_FEATURES).merge(
            daily,
            on=DATE_COL,
            how="left",
        )
        for feature in PROMOTION_FEATURES + PROMOTION_PHASE_FEATURES:
            calendar[feature] = calendar[feature].fillna(0)

    daily_starts = (
        promotions.groupby("start_date")
        .size()
        .rename("campaign_start_count")
        .sort_index()
        .cumsum()
        .reset_index()
        .rename(columns={"start_date": DATE_COL})
    )
    calendar = pd.merge_asof(
        calendar.sort_values(DATE_COL),
        daily_starts.sort_values(DATE_COL),
        on=DATE_COL,
        direction="backward",
    )
    calendar["promotion_campaign_index"] = calendar["campaign_start_count"].fillna(0)
    if "campaign_start_count" in calendar.columns:
        calendar = calendar.drop(columns=["campaign_start_count"])

    return calendar[[DATE_COL] + PROMOTION_FEATURES + PROMOTION_PHASE_FEATURES + CAMPAIGN_FEATURES]


def add_orders_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    output = df.sort_values(DATE_COL).reset_index(drop=True).copy()
    output["orders_lag_7"] = output[TARGET_COL].shift(7)
    output["orders_lag_14"] = output[TARGET_COL].shift(14)
    output["orders_lag_30"] = output[TARGET_COL].shift(30)
    output["orders_lag_365"] = output[TARGET_COL].shift(365)

    shifted_orders = output[TARGET_COL].shift(1)
    output["orders_roll_mean_7"] = shifted_orders.rolling(window=7, min_periods=7).mean()
    output["orders_roll_mean_30"] = shifted_orders.rolling(window=30, min_periods=30).mean()
    return output


def build_model_table(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    calendar = build_calendar_features(df[DATE_COL])
    promotions = build_promotion_features(df[DATE_COL], PROMOTIONS_PATH, logger)
    table = (
        df[[DATE_COL, TARGET_COL]]
        .merge(calendar, on=DATE_COL, how="left", validate="one_to_one")
        .merge(promotions, on=DATE_COL, how="left", validate="one_to_one")
    )
    table = table.fillna(0)
    table = add_orders_lag_features(table)
    return table


def make_training_matrix(
    model_table: pd.DataFrame,
    feature_columns: list[str],
    train_end_exclusive: pd.Timestamp | None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    table = model_table.copy()
    if train_end_exclusive is not None:
        table = table[table[DATE_COL] < train_end_exclusive].copy()

    clean = table.dropna(subset=feature_columns + [TARGET_COL]).reset_index(drop=True)
    X = clean[feature_columns].copy()
    y = clean[TARGET_COL].copy()
    feature_medians = X.median(numeric_only=True)
    return X, y, clean, feature_medians


def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, reporter: Reporter) -> Any:
    import lightgbm as lgb

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.03,
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
    reporter.logger.info("Trained LightGBM on %s rows and %s features", len(X_train), X_train.shape[1])
    return model


def train_hist_gradient_boosting(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError as exc:
        raise ImportError("scikit-learn fallback is not installed.") from exc

    model = HistGradientBoostingRegressor(
        learning_rate=0.03,
        max_iter=1500,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def train_model(X_train: pd.DataFrame, y_train: pd.Series, reporter: Reporter) -> tuple[Any, str]:
    if base.lightgbm_available():
        return train_lightgbm(X_train, y_train, reporter), "lightgbm"
    reporter.emit("LightGBM unavailable; using HistGradientBoostingRegressor fallback")
    return train_hist_gradient_boosting(X_train, y_train), "hist_gradient_boosting"


def compute_orders_features_from_history(history: pd.Series, forecast_date: pd.Timestamp) -> dict[str, float]:
    past_history = history[history.index < forecast_date].sort_index()

    def lag_value(days: int) -> float:
        return float(history.get(forecast_date - pd.Timedelta(days=days), np.nan))

    def roll_mean(window: int) -> float:
        values = past_history.tail(window)
        return float(values.mean()) if len(values) == window else np.nan

    return {
        "orders_lag_7": lag_value(7),
        "orders_lag_14": lag_value(14),
        "orders_lag_30": lag_value(30),
        "orders_lag_365": lag_value(365),
        "orders_roll_mean_7": roll_mean(7),
        "orders_roll_mean_30": roll_mean(30),
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
    history = initial_history.copy().sort_index()
    predictions: list[float] = []

    for forecast_date in pd.to_datetime(prediction_dates):
        if forecast_date not in static_by_date.index:
            raise ValueError(f"Missing static features for forecast date {forecast_date.date()}")

        row = static_by_date.loc[forecast_date].to_dict()
        row.update(compute_orders_features_from_history(history, forecast_date))
        X_row = pd.DataFrame([row], columns=feature_columns)
        X_row = X_row.apply(pd.to_numeric, errors="coerce").fillna(feature_medians).fillna(0)

        prediction = float(model.predict(X_row)[0])
        prediction = max(0.0, prediction)
        predictions.append(prediction)
        history.loc[forecast_date] = prediction

    return np.asarray(predictions, dtype=float)


def save_validation_predictions(
    dates: pd.Series,
    actual: pd.Series,
    predicted: np.ndarray,
    path: Path = VALIDATION_PREDICTIONS_PATH,
) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            DATE_COL: dates,
            "actual_orders_count": actual.to_numpy(dtype=float),
            "predicted_orders_count": np.asarray(predicted, dtype=float),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    return output


def promotion_importance_summary(importance: pd.DataFrame) -> tuple[bool, list[str]]:
    promo_prefixes = ("calendar_", "promo_", "promotion_campaign_index")
    top20 = importance.head(20)["feature"].astype(str).tolist()
    important = [feature for feature in top20 if feature.startswith(promo_prefixes)]
    return len(important) > 0, important


def run() -> None:
    logger = setup_logging()
    reporter = Reporter(logger)

    reporter.emit("Orders Count Forecasting Mini Pipeline")
    reporter.emit("====================================")
    reporter.emit("")

    reporter.emit("1. Load daily feature table")
    df = load_daily_feature_table(TRAIN_DATA_PATH)
    reporter.emit(f"Loaded dataset: {TRAIN_DATA_PATH} | shape={df.shape}")
    reporter.emit(f"Date range: {df[DATE_COL].min().date()} -> {df[DATE_COL].max().date()}")

    reporter.emit("")
    reporter.emit("2. Build static calendar + promotion features")
    model_table = build_model_table(df, logger)
    static_features = model_table[[DATE_COL] + CALENDAR_FEATURES + PROMOTION_FEATURES + PROMOTION_PHASE_FEATURES + CAMPAIGN_FEATURES].copy()
    reporter.emit(f"Model table shape: {model_table.shape}")
    reporter.emit(f"Feature count: {len(FEATURE_COLUMNS)}")

    reporter.emit("")
    reporter.emit("3. Train on 2012-2021 and recursively validate on 2022")
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
    validation_dates = df[
        (df[DATE_COL] >= TRAIN_CUTOFF) & (df[DATE_COL] <= VALIDATION_END)
    ][DATE_COL]
    actual = df.set_index(DATE_COL).loc[validation_dates, TARGET_COL].reset_index(drop=True)
    initial_history = df[df[DATE_COL] < TRAIN_CUTOFF].set_index(DATE_COL)[TARGET_COL]

    predictions = recursive_predict_orders(
        model=model,
        feature_columns=FEATURE_COLUMNS,
        static_features=static_features,
        initial_history=initial_history,
        prediction_dates=validation_dates,
        feature_medians=feature_medians,
    )
    metrics = base.evaluate_predictions(actual, predictions)

    reporter.emit(f"MAE: {metrics['MAE']:,.4f}")
    reporter.emit(f"RMSE: {metrics['RMSE']:,.4f}")
    reporter.emit(f"R2: {metrics['R2']:.6f}")

    validation_output = save_validation_predictions(validation_dates.reset_index(drop=True), actual, predictions)
    reporter.emit(f"Saved validation predictions: {VALIDATION_PREDICTIONS_PATH} | shape={validation_output.shape}")

    importance = base.get_feature_importance(
        model=model,
        model_type=model_type,
        feature_columns=FEATURE_COLUMNS,
        X_ref=X_train,
        y_ref=y_train,
        baseline_rmse=metrics["RMSE"],
    )
    importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    reporter.emit(f"Saved feature importance: {FEATURE_IMPORTANCE_PATH}")
    reporter.emit_frame("Top 20 features:", importance.head(20))

    promo_important, promo_features_in_top20 = promotion_importance_summary(importance)
    reporter.emit("")
    reporter.emit(f"Promotion features important: {promo_important}")
    reporter.emit(f"Promotion-related features in Top 20: {promo_features_in_top20}")

    reporter.save(METRICS_PATH)


if __name__ == "__main__":
    run()
