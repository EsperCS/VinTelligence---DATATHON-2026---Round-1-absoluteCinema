from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "log"

TRAIN_DATA_PATH = DATA_DIR / "daily_feature_table.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
PROMOTIONS_PATH = DATA_DIR / "promotions.csv"
INVENTORY_PATH = DATA_DIR / "inventory.csv"

SUBMISSION_PATH = DATA_DIR / "submission.csv"
VALIDATION_PREDICTIONS_PATH = DATA_DIR / "final_validation_predictions.csv"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "final_feature_importance.csv"
METRICS_PATH = LOG_DIR / "final_model_metrics.txt"
LOG_FILE = LOG_DIR / "train_final_model.log"

DATE_COL = "Date"
TARGET_COL = "Revenue"
COGS_COL = "COGS"
TRAIN_CUTOFF = pd.Timestamp("2022-01-01")
VALIDATION_END = pd.Timestamp("2022-12-31")
RANDOM_STATE = 42

REVENUE_LAGS = [7, 14, 30, 60, 90, 180, 365]
REVENUE_ROLL_MEAN_WINDOWS = [7, 30, 14, 60, 90, 180, 365]
REVENUE_ROLL_STD_WINDOWS = [30, 90, 365]
BUSINESS_LAG365_SOURCES = [
    "orders_count",
    "unique_customers",
    "total_quantity",
    "item_lines_count",
]

CALENDAR_FEATURES = [
    "day_of_week",
    "day_of_year",
    "week_of_year",
    "month",
    "quarter",
    "year",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "is_quarter_start",
    "is_quarter_end",
    "time_index",
    "post_2019_flag",
    "years_since_start",
    "years_since_2019",
    "post_2019_time_index",
]
REVENUE_FEATURES = [
    "lag_7",
    "lag_14",
    "lag_30",
    "revenue_lag_60",
    "revenue_lag_90",
    "revenue_lag_180",
    "revenue_lag_365",
    "rolling_mean_7",
    "rolling_mean_30",
    "revenue_roll_mean_14",
    "revenue_roll_mean_60",
    "revenue_roll_mean_90",
    "revenue_roll_mean_180",
    "revenue_roll_mean_365",
    "revenue_roll_std_30",
    "revenue_roll_std_90",
    "revenue_roll_std_365",
]
PROMOTION_FEATURES = [
    "calendar_active_promo_count",
    "calendar_any_promo",
    "calendar_avg_discount_value",
    "calendar_max_discount_value",
    "calendar_stackable_promo_count",
    "calendar_has_stackable_promo",
    "calendar_has_category_specific_promo",
    "calendar_percentage_promo_count",
    "calendar_fixed_promo_count",
]
INVENTORY_FEATURES = [
    "inv_stockout_rate",
    "inv_avg_fill_rate",
    "inv_avg_days_of_supply",
    "inv_avg_sell_through_rate",
    "inv_reorder_rate",
    "inv_overstock_rate",
]
BUSINESS_LAG365_FEATURES = [f"{column}_lag_365" for column in BUSINESS_LAG365_SOURCES]

MODEL_A_FEATURES = CALENDAR_FEATURES + REVENUE_FEATURES + PROMOTION_FEATURES + INVENTORY_FEATURES
MODEL_B_FEATURES = MODEL_A_FEATURES + BUSINESS_LAG365_FEATURES

UNSAFE_SAME_DAY_COLUMNS = [
    COGS_COL,
    "orders_count",
    "unique_customers",
    "total_quantity",
    "item_lines_count",
    "avg_order_value",
    "total_discount_amount",
    "avg_discount_amount",
    "avg_discount_rate",
    "promo_usage_rate",
    "discount_to_gross_rate",
    "payments_count",
    "payment_count",
    "payment_value",
    "payment_value_sum",
    "avg_payment_value",
    "returns_count",
    "returned_orders_count",
    "return_quantity",
    "refund_amount",
    "return_rate",
    "review_count",
    "reviews_count",
    "avg_rating",
    "rating",
    "web_sessions",
    "web_unique_visitors",
    "web_page_views",
    "web_bounce_rate",
    "web_avg_session_duration_sec",
    "web_traffic_available",
]


class RunReporter:
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

    def save_metrics(self, path: Path = METRICS_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        self.logger.info("Saved final metrics report to %s", path)


def setup_logging(log_file: Path = LOG_FILE) -> logging.Logger:
    """Configure simple file logging."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train_final_model")
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


def load_train_data(path: Path = TRAIN_DATA_PATH) -> pd.DataFrame:
    """Load historical daily feature table."""
    if not path.exists():
        raise FileNotFoundError(f"Train dataset not found: {path}")

    df = pd.read_csv(path, parse_dates=[DATE_COL], low_memory=False)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce").dt.normalize()
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    if df[DATE_COL].isna().any():
        raise ValueError("Train Date column contains invalid timestamps")

    return df


def load_sample_submission(path: Path = SAMPLE_SUBMISSION_PATH) -> pd.DataFrame:
    """Load sample submission but only use its Date/order."""
    if not path.exists():
        raise FileNotFoundError(f"Sample submission not found: {path}")

    sample = pd.read_csv(path, parse_dates=[DATE_COL], low_memory=False)
    sample[DATE_COL] = pd.to_datetime(sample[DATE_COL], errors="coerce").dt.normalize()
    if sample[DATE_COL].isna().any():
        raise ValueError("Sample submission Date column contains invalid timestamps")
    return sample


def build_calendar_features(dates: pd.Series, min_date: pd.Timestamp) -> pd.DataFrame:
    """Build deterministic date features for historical and future dates."""
    output = pd.DataFrame({DATE_COL: pd.to_datetime(dates).sort_values().unique()})
    output["day_of_week"] = output[DATE_COL].dt.dayofweek.astype(int)
    output["day_of_year"] = output[DATE_COL].dt.dayofyear.astype(int)
    output["week_of_year"] = output[DATE_COL].dt.isocalendar().week.astype(int)
    output["month"] = output[DATE_COL].dt.month.astype(int)
    output["quarter"] = output[DATE_COL].dt.quarter.astype(int)
    output["year"] = output[DATE_COL].dt.year.astype(int)
    output["is_weekend"] = output["day_of_week"].isin([5, 6]).astype(int)
    output["is_month_start"] = output[DATE_COL].dt.is_month_start.astype(int)
    output["is_month_end"] = output[DATE_COL].dt.is_month_end.astype(int)
    output["is_quarter_start"] = output[DATE_COL].dt.is_quarter_start.astype(int)
    output["is_quarter_end"] = output[DATE_COL].dt.is_quarter_end.astype(int)
    output["time_index"] = (output[DATE_COL] - min_date).dt.days.astype(int)
    output["post_2019_flag"] = (output[DATE_COL] >= pd.Timestamp("2019-01-01")).astype(int)
    output["years_since_start"] = output["year"] - int(min_date.year)
    output["years_since_2019"] = np.maximum(0, output["year"] - 2019)
    output["post_2019_time_index"] = output["time_index"] * output["post_2019_flag"]
    return output[[DATE_COL] + CALENDAR_FEATURES]


def _stackable_to_int(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    return text.isin(["1", "true", "yes", "y"]).astype(int)


def build_promotion_calendar(
    dates: pd.Series,
    promotions_path: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Build safe promotion schedule features known before the forecast date."""
    calendar = pd.DataFrame({DATE_COL: pd.to_datetime(dates).sort_values().unique()})
    for feature in PROMOTION_FEATURES:
        calendar[feature] = 0.0

    if not promotions_path.exists():
        logger.warning("promotions.csv not found at %s; promotion features are zero", promotions_path)
        return calendar

    promotions = pd.read_csv(promotions_path, low_memory=False)
    required = {"promo_id", "start_date", "end_date"}
    if not required.issubset(promotions.columns):
        logger.warning("promotions.csv missing required columns; promotion features are zero")
        return calendar

    promotions["start_date"] = pd.to_datetime(promotions["start_date"], errors="coerce").dt.normalize()
    promotions["end_date"] = pd.to_datetime(promotions["end_date"], errors="coerce").dt.normalize()
    promotions["discount_value"] = pd.to_numeric(
        promotions.get("discount_value", 0),
        errors="coerce",
    ).fillna(0)
    promotions["stackable_flag_numeric"] = (
        _stackable_to_int(promotions["stackable_flag"])
        if "stackable_flag" in promotions.columns
        else 0
    )
    promotions["category_specific"] = (
        promotions["applicable_category"].notna().astype(int)
        if "applicable_category" in promotions.columns
        else 0
    )
    promo_type = promotions["promo_type"].astype(str).str.lower() if "promo_type" in promotions else ""
    promotions["percentage_promo"] = promo_type.eq("percentage").astype(int)
    promotions["fixed_promo"] = promo_type.eq("fixed").astype(int)

    rows: list[dict[str, Any]] = []
    min_date = calendar[DATE_COL].min()
    max_date = calendar[DATE_COL].max()
    for row in promotions.dropna(subset=["start_date", "end_date"]).itertuples(index=False):
        start_date = max(row.start_date, min_date)
        end_date = min(row.end_date, max_date)
        if start_date > end_date:
            continue

        for active_date in pd.date_range(start_date, end_date, freq="D"):
            rows.append(
                {
                    DATE_COL: active_date,
                    "promo_id": row.promo_id,
                    "discount_value": row.discount_value,
                    "stackable_flag_numeric": row.stackable_flag_numeric,
                    "category_specific": row.category_specific,
                    "percentage_promo": row.percentage_promo,
                    "fixed_promo": row.fixed_promo,
                }
            )

    if not rows:
        return calendar

    expanded = pd.DataFrame(rows)
    daily = (
        expanded.groupby(DATE_COL, as_index=False)
        .agg(
            calendar_active_promo_count=("promo_id", "nunique"),
            calendar_avg_discount_value=("discount_value", "mean"),
            calendar_max_discount_value=("discount_value", "max"),
            calendar_stackable_promo_count=("stackable_flag_numeric", "sum"),
            calendar_has_stackable_promo=("stackable_flag_numeric", "max"),
            calendar_has_category_specific_promo=("category_specific", "max"),
            calendar_percentage_promo_count=("percentage_promo", "sum"),
            calendar_fixed_promo_count=("fixed_promo", "sum"),
        )
    )
    daily["calendar_any_promo"] = (daily["calendar_active_promo_count"] > 0).astype(int)

    calendar = calendar.drop(columns=PROMOTION_FEATURES).merge(daily, on=DATE_COL, how="left")
    for feature in PROMOTION_FEATURES:
        calendar[feature] = calendar[feature].fillna(0)

    return calendar[[DATE_COL] + PROMOTION_FEATURES]


def build_inventory_asof_features(
    dates: pd.Series,
    inventory_path: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Build inventory snapshot features using only the latest known past snapshot."""
    calendar = pd.DataFrame({DATE_COL: pd.to_datetime(dates).sort_values().unique()})
    for feature in INVENTORY_FEATURES:
        calendar[feature] = 0.0

    if not inventory_path.exists():
        logger.warning("inventory.csv not found at %s; inventory features are zero", inventory_path)
        return calendar

    inventory = pd.read_csv(inventory_path, low_memory=False)
    if "snapshot_date" not in inventory.columns:
        logger.warning("inventory.csv missing snapshot_date; inventory features are zero")
        return calendar

    inventory["snapshot_date"] = pd.to_datetime(
        inventory["snapshot_date"],
        errors="coerce",
    ).dt.normalize()

    numeric_columns = [
        "stockout_flag",
        "fill_rate",
        "days_of_supply",
        "sell_through_rate",
        "reorder_flag",
        "overstock_flag",
    ]
    for column in numeric_columns:
        if column not in inventory.columns:
            inventory[column] = 0
        inventory[column] = pd.to_numeric(inventory[column], errors="coerce").fillna(0)

    snapshots = (
        inventory.dropna(subset=["snapshot_date"])
        .groupby("snapshot_date", as_index=False)
        .agg(
            inv_stockout_rate=("stockout_flag", "mean"),
            inv_avg_fill_rate=("fill_rate", "mean"),
            inv_avg_days_of_supply=("days_of_supply", "mean"),
            inv_avg_sell_through_rate=("sell_through_rate", "mean"),
            inv_reorder_rate=("reorder_flag", "mean"),
            inv_overstock_rate=("overstock_flag", "mean"),
        )
        .rename(columns={"snapshot_date": DATE_COL})
        .sort_values(DATE_COL)
    )

    if snapshots.empty:
        return calendar

    merged = pd.merge_asof(
        calendar[[DATE_COL]].sort_values(DATE_COL),
        snapshots,
        on=DATE_COL,
        direction="backward",
    )
    merged[INVENTORY_FEATURES] = merged[INVENTORY_FEATURES].fillna(0)
    return merged[[DATE_COL] + INVENTORY_FEATURES]


def build_static_features(
    dates: pd.Series,
    min_date: pd.Timestamp,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Build all non-autoregressive forecast-known features."""
    static = build_calendar_features(dates, min_date)
    promo = build_promotion_calendar(dates, PROMOTIONS_PATH, logger)
    inventory = build_inventory_asof_features(dates, INVENTORY_PATH, logger)
    return (
        static.merge(promo, on=DATE_COL, how="left", validate="one_to_one")
        .merge(inventory, on=DATE_COL, how="left", validate="one_to_one")
        .fillna(0)
    )


def add_historical_revenue_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create revenue lag/rolling features from actual historical Revenue."""
    output = df.sort_values(DATE_COL).reset_index(drop=True).copy()
    new_features: dict[str, pd.Series] = {}

    for lag in REVENUE_LAGS:
        feature = f"lag_{lag}" if lag in {7, 14, 30} else f"revenue_lag_{lag}"
        new_features[feature] = output[TARGET_COL].shift(lag)

    shifted_revenue = output[TARGET_COL].shift(1)
    for window in REVENUE_ROLL_MEAN_WINDOWS:
        feature = f"rolling_mean_{window}" if window in {7, 30} else f"revenue_roll_mean_{window}"
        new_features[feature] = shifted_revenue.rolling(window=window, min_periods=window).mean()

    for window in REVENUE_ROLL_STD_WINDOWS:
        feature = f"revenue_roll_std_{window}"
        new_features[feature] = shifted_revenue.rolling(window=window, min_periods=window).std()

    return pd.concat([output, pd.DataFrame(new_features)], axis=1).copy()


def add_historical_business_lag365(df: pd.DataFrame) -> pd.DataFrame:
    """Create previous-year realized business lags for historical rows."""
    output = df.sort_values(DATE_COL).reset_index(drop=True).copy()
    new_features: dict[str, pd.Series] = {}
    for column in BUSINESS_LAG365_SOURCES:
        if column in output.columns:
            new_features[f"{column}_lag_365"] = output[column].shift(365)
    if new_features:
        output = pd.concat([output, pd.DataFrame(new_features)], axis=1).copy()
    return output


def build_historical_model_table(
    train_df: pd.DataFrame,
    static_features: pd.DataFrame,
    include_business_lag365: bool,
) -> pd.DataFrame:
    """Build model table for historical training/backtesting."""
    base_columns = [DATE_COL, TARGET_COL] + [
        column for column in BUSINESS_LAG365_SOURCES if column in train_df.columns
    ]
    table = train_df[base_columns].merge(static_features, on=DATE_COL, how="left", validate="one_to_one")
    table = add_historical_revenue_features(table)
    if include_business_lag365:
        table = add_historical_business_lag365(table)

    dropped = [column for column in UNSAFE_SAME_DAY_COLUMNS if column in table.columns]
    table = table.drop(columns=dropped)
    return table


def get_feature_columns(model_variant: str) -> list[str]:
    """Return exact feature set for Model A or Model B."""
    if model_variant == "A":
        return MODEL_A_FEATURES.copy()
    if model_variant == "B":
        return MODEL_B_FEATURES.copy()
    raise ValueError(f"Unknown model variant: {model_variant}")


def make_training_matrix(
    model_table: pd.DataFrame,
    feature_columns: list[str],
    train_end_exclusive: pd.Timestamp | None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Create clean train matrix and medians for safe future imputation."""
    table = model_table.copy()
    if train_end_exclusive is not None:
        table = table[table[DATE_COL] < train_end_exclusive].copy()

    missing_features = [column for column in feature_columns if column not in table.columns]
    if missing_features:
        raise ValueError(f"Missing expected feature columns: {missing_features}")

    clean = table.dropna(subset=feature_columns + [TARGET_COL]).reset_index(drop=True)
    X = clean[feature_columns].copy()
    y = clean[TARGET_COL].copy()
    feature_medians = X.median(numeric_only=True)
    return X, y, clean, feature_medians


def lightgbm_available() -> bool:
    return importlib.util.find_spec("lightgbm") is not None


def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, reporter: RunReporter) -> Any:
    """Train native LightGBM using final hyperparameters."""
    import lightgbm as lgb

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.025,
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
    model = lgb.train(params=params, train_set=train_data, num_boost_round=2000)
    reporter.logger.info("Trained LightGBM on %s rows and %s features", len(X_train), X_train.shape[1])
    return model


def train_hist_gradient_boosting(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """Fallback model if LightGBM is unavailable."""
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError as exc:
        raise ImportError(
            "LightGBM is unavailable and scikit-learn fallback is not installed."
        ) from exc

    model = HistGradientBoostingRegressor(
        learning_rate=0.025,
        max_iter=2000,
        max_leaf_nodes=31,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def train_model(X_train: pd.DataFrame, y_train: pd.Series, reporter: RunReporter) -> tuple[Any, str]:
    """Train LightGBM if available, otherwise fallback."""
    if lightgbm_available():
        return train_lightgbm(X_train, y_train, reporter), "lightgbm"
    return train_hist_gradient_boosting(X_train, y_train), "hist_gradient_boosting"


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, and R2."""
    actual = y_true.to_numpy(dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    errors = actual - predicted

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    ss_res = float(np.sum(errors**2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def compute_revenue_features_from_history(
    history: pd.Series,
    forecast_date: pd.Timestamp,
) -> dict[str, float]:
    """Compute one row of revenue lag/rolling features from past actual/predicted history."""
    features: dict[str, float] = {}

    for lag in REVENUE_LAGS:
        feature = f"lag_{lag}" if lag in {7, 14, 30} else f"revenue_lag_{lag}"
        lag_date = forecast_date - pd.Timedelta(days=lag)
        features[feature] = float(history.get(lag_date, np.nan))

    past_history = history[history.index < forecast_date].sort_index()
    for window in REVENUE_ROLL_MEAN_WINDOWS:
        feature = f"rolling_mean_{window}" if window in {7, 30} else f"revenue_roll_mean_{window}"
        values = past_history.tail(window)
        features[feature] = float(values.mean()) if len(values) == window else np.nan

    for window in REVENUE_ROLL_STD_WINDOWS:
        feature = f"revenue_roll_std_{window}"
        values = past_history.tail(window)
        features[feature] = float(values.std(ddof=1)) if len(values) == window else np.nan

    return features


def safe_replace_year(date_value: pd.Timestamp, year: int) -> pd.Timestamp:
    """Return same month/day in target year; use Feb 28 for leap-day edge cases."""
    try:
        return date_value.replace(year=year)
    except ValueError:
        return pd.Timestamp(year=year, month=2, day=28)


def build_business_source_maps(train_df: pd.DataFrame) -> dict[str, pd.Series]:
    """Index historical business source columns by Date."""
    maps: dict[str, pd.Series] = {}
    indexed = train_df.set_index(DATE_COL).sort_index()
    for column in BUSINESS_LAG365_SOURCES:
        if column in indexed.columns:
            maps[column] = pd.to_numeric(indexed[column], errors="coerce")
    return maps


def get_business_lag365_value(
    source: pd.Series,
    forecast_date: pd.Timestamp,
    feature_median: float,
) -> float:
    """Fetch lag-365 business value or safely impute from 2022 same-date/recent history."""
    lag_date = forecast_date - pd.Timedelta(days=365)
    if lag_date in source.index and pd.notna(source.loc[lag_date]):
        return float(source.loc[lag_date])

    same_date_2022 = safe_replace_year(forecast_date, 2022)
    if same_date_2022 in source.index and pd.notna(source.loc[same_date_2022]):
        return float(source.loc[same_date_2022])

    day_of_year_matches = source[source.index.dayofyear == forecast_date.dayofyear].dropna()
    if not day_of_year_matches.empty:
        return float(day_of_year_matches.tail(3).mean())

    recent = source.dropna().tail(365)
    if not recent.empty:
        return float(recent.mean())

    return float(feature_median)


def compute_business_lag365_features(
    business_maps: dict[str, pd.Series],
    forecast_date: pd.Timestamp,
    feature_medians: pd.Series,
) -> dict[str, float]:
    """Compute previous-year business lag features for Model B."""
    features: dict[str, float] = {}
    for source_column in BUSINESS_LAG365_SOURCES:
        feature = f"{source_column}_lag_365"
        source = business_maps.get(source_column)
        if source is None:
            features[feature] = float(feature_medians.get(feature, 0.0))
            continue
        features[feature] = get_business_lag365_value(
            source,
            forecast_date,
            float(feature_medians.get(feature, 0.0)),
        )
    return features


def recursive_predict(
    model: Any,
    model_type: str,
    prediction_dates: pd.Series,
    feature_columns: list[str],
    static_features: pd.DataFrame,
    initial_revenue_history: pd.Series,
    business_maps: dict[str, pd.Series],
    feature_medians: pd.Series,
    include_business_lag365: bool,
) -> np.ndarray:
    """Predict dates one by one and append each predicted Revenue to the lag history."""
    del model_type
    static_by_date = static_features.set_index(DATE_COL).sort_index()
    history = initial_revenue_history.copy().sort_index()
    predictions: list[float] = []

    for forecast_date in pd.to_datetime(prediction_dates):
        row: dict[str, float] = {}
        if forecast_date in static_by_date.index:
            row.update(static_by_date.loc[forecast_date].to_dict())
        else:
            raise ValueError(f"Missing static features for forecast date {forecast_date.date()}")

        row.update(compute_revenue_features_from_history(history, forecast_date))
        if include_business_lag365:
            row.update(compute_business_lag365_features(business_maps, forecast_date, feature_medians))

        X_row = pd.DataFrame([row], columns=feature_columns)
        X_row = X_row.apply(pd.to_numeric, errors="coerce")
        X_row = X_row.fillna(feature_medians).fillna(0)

        prediction = float(model.predict(X_row)[0])
        prediction = max(0.0, prediction)
        predictions.append(prediction)
        history.loc[forecast_date] = prediction

    return np.asarray(predictions, dtype=float)


def validate_model_variant(
    variant: str,
    model_table: pd.DataFrame,
    static_features: pd.DataFrame,
    train_df: pd.DataFrame,
    reporter: RunReporter,
) -> dict[str, Any]:
    """Train on pre-2022 data and recursively validate on 2022."""
    feature_columns = get_feature_columns(variant)
    X_train, y_train, train_clean, feature_medians = make_training_matrix(
        model_table,
        feature_columns,
        TRAIN_CUTOFF,
    )

    reporter.emit(f"Training validation Model {variant}: rows={len(X_train):,}, features={len(feature_columns)}")
    model, model_type = train_model(X_train, y_train, reporter)

    validation_dates = train_df[
        (train_df[DATE_COL] >= TRAIN_CUTOFF) & (train_df[DATE_COL] <= VALIDATION_END)
    ][DATE_COL]
    actual = train_df.set_index(DATE_COL).loc[validation_dates, TARGET_COL]
    initial_history = train_df[train_df[DATE_COL] < TRAIN_CUTOFF].set_index(DATE_COL)[TARGET_COL]
    business_maps = build_business_source_maps(train_df[train_df[DATE_COL] < TRAIN_CUTOFF])

    predictions = recursive_predict(
        model=model,
        model_type=model_type,
        prediction_dates=validation_dates,
        feature_columns=feature_columns,
        static_features=static_features,
        initial_revenue_history=initial_history,
        business_maps=business_maps,
        feature_medians=feature_medians,
        include_business_lag365=(variant == "B"),
    )
    metrics = evaluate_predictions(actual, predictions)

    return {
        "variant": variant,
        "model": model,
        "model_type": model_type,
        "feature_columns": feature_columns,
        "feature_medians": feature_medians,
        "train_clean": train_clean,
        "validation_dates": validation_dates.reset_index(drop=True),
        "actual": actual.reset_index(drop=True),
        "predictions": predictions,
        "metrics": metrics,
    }


def choose_model_variant(result_a: dict[str, Any], result_b: dict[str, Any]) -> dict[str, Any]:
    """Choose the better/safe candidate; B must beat A on MAE and not hurt RMSE/R2."""
    metrics_a = result_a["metrics"]
    metrics_b = result_b["metrics"]

    b_mae_improvement = (metrics_a["MAE"] - metrics_b["MAE"]) / metrics_a["MAE"]
    b_is_better = (
        b_mae_improvement >= 0.005
        and metrics_b["RMSE"] <= metrics_a["RMSE"]
        and metrics_b["R2"] >= metrics_a["R2"]
    )
    return result_b if b_is_better else result_a


def save_validation_predictions(selected: dict[str, Any], path: Path = VALIDATION_PREDICTIONS_PATH) -> pd.DataFrame:
    """Save selected model's recursive 2022 validation predictions."""
    output = pd.DataFrame(
        {
            DATE_COL: selected["validation_dates"],
            "actual_Revenue": selected["actual"],
            "predicted_Revenue": selected["predictions"],
            "selected_model": selected["variant"],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    return output


def lightgbm_feature_importance(model: Any, feature_columns: list[str]) -> pd.DataFrame:
    """Extract LightGBM split/gain importance."""
    return (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "importance_split": model.feature_importance(importance_type="split"),
                "importance_gain": model.feature_importance(importance_type="gain"),
            }
        )
        .sort_values(["importance_gain", "importance_split"], ascending=False)
        .reset_index(drop=True)
    )


def permutation_feature_importance(
    model: Any,
    X_ref: pd.DataFrame,
    y_ref: pd.Series,
    baseline_rmse: float,
) -> pd.DataFrame:
    """Manual permutation importance fallback."""
    rng = np.random.default_rng(RANDOM_STATE)
    rows: list[dict[str, float | str]] = []
    for column in X_ref.columns:
        permuted = X_ref.copy()
        permuted[column] = rng.permutation(permuted[column].to_numpy())
        rmse = evaluate_predictions(y_ref, model.predict(permuted))["RMSE"]
        rows.append(
            {
                "feature": column,
                "importance_split": 0,
                "importance_gain": rmse - baseline_rmse,
            }
        )
    return pd.DataFrame(rows).sort_values("importance_gain", ascending=False).reset_index(drop=True)


def get_feature_importance(
    model: Any,
    model_type: str,
    feature_columns: list[str],
    X_ref: pd.DataFrame,
    y_ref: pd.Series,
    baseline_rmse: float,
) -> pd.DataFrame:
    """Return feature importance for either model backend."""
    if model_type == "lightgbm":
        return lightgbm_feature_importance(model, feature_columns)
    return permutation_feature_importance(model, X_ref, y_ref, baseline_rmse)


def train_final_selected_model(
    selected_variant: str,
    model_table: pd.DataFrame,
    reporter: RunReporter,
) -> dict[str, Any]:
    """Train selected candidate on all 2012-2022 usable rows."""
    feature_columns = get_feature_columns(selected_variant)
    X_all, y_all, train_clean, feature_medians = make_training_matrix(
        model_table,
        feature_columns,
        train_end_exclusive=None,
    )
    reporter.emit("")
    reporter.emit("5. Final training")
    reporter.emit(
        f"Training selected Model {selected_variant} on all usable rows: "
        f"rows={len(X_all):,}, features={len(feature_columns)}"
    )
    model, model_type = train_model(X_all, y_all, reporter)
    return {
        "variant": selected_variant,
        "model": model,
        "model_type": model_type,
        "feature_columns": feature_columns,
        "feature_medians": feature_medians,
        "X_all": X_all,
        "y_all": y_all,
        "train_clean": train_clean,
    }


def estimate_cogs_ratio(train_df: pd.DataFrame) -> float:
    """Estimate future COGS from a recent historical COGS/Revenue ratio."""
    recent = train_df.sort_values(DATE_COL).tail(365).copy()
    ratio = recent[COGS_COL].sum() / recent[TARGET_COL].sum()
    if not np.isfinite(ratio) or ratio <= 0:
        ratio = train_df[COGS_COL].sum() / train_df[TARGET_COL].sum()
    return float(np.clip(ratio, 0.0, 2.0))


def build_submission(
    sample_submission: pd.DataFrame,
    revenue_predictions: np.ndarray,
    cogs_ratio: float,
    path: Path = SUBMISSION_PATH,
) -> pd.DataFrame:
    """Create final submission with exact sample columns and row order."""
    submission = sample_submission[[DATE_COL]].copy()
    submission[TARGET_COL] = np.maximum(0.0, revenue_predictions)
    submission[COGS_COL] = np.maximum(0.0, submission[TARGET_COL] * cogs_ratio)
    submission = submission[[DATE_COL, TARGET_COL, COGS_COL]]
    submission.to_csv(path, index=False)
    return submission


def emit_metrics(title: str, metrics: dict[str, float], reporter: RunReporter) -> None:
    reporter.emit(title)
    reporter.emit(f"MAE: {metrics['MAE']:,.2f}")
    reporter.emit(f"RMSE: {metrics['RMSE']:,.2f}")
    reporter.emit(f"R2: {metrics['R2']:.6f}")


def run_training() -> None:
    logger = setup_logging()
    reporter = RunReporter(logger)

    reporter.emit("Final Forecasting Model Training")
    reporter.emit("=================================")
    reporter.emit("")
    reporter.emit("1. Load data")

    train_df = load_train_data(TRAIN_DATA_PATH)
    sample_submission = load_sample_submission(SAMPLE_SUBMISSION_PATH)
    min_date = train_df[DATE_COL].min()
    all_dates = pd.Series(
        pd.date_range(train_df[DATE_COL].min(), sample_submission[DATE_COL].max(), freq="D")
    )

    reporter.emit(f"Loaded train data: {TRAIN_DATA_PATH} | shape={train_df.shape}")
    reporter.emit(
        f"Train date range: {train_df[DATE_COL].min().date()} -> {train_df[DATE_COL].max().date()}"
    )
    reporter.emit(
        "Forecast date range: "
        f"{sample_submission[DATE_COL].min().date()} -> {sample_submission[DATE_COL].max().date()}"
    )
    reporter.emit(f"Sample submission rows: {len(sample_submission):,}")

    reporter.emit("")
    reporter.emit("2. Build forecast-known static features")
    static_features = build_static_features(all_dates, min_date, logger)
    reporter.emit(f"Static feature table shape: {static_features.shape}")

    reporter.emit("")
    reporter.emit("3. Build candidate model tables")
    table_a = build_historical_model_table(train_df, static_features, include_business_lag365=False)
    table_b = build_historical_model_table(train_df, static_features, include_business_lag365=True)
    dropped_unsafe = [column for column in UNSAFE_SAME_DAY_COLUMNS if column in train_df.columns]
    reporter.emit(f"Dropped/blocked unsafe same-day features: {dropped_unsafe}")
    reporter.emit(f"Model A features: {len(MODEL_A_FEATURES)}")
    reporter.emit(f"Model B features: {len(MODEL_B_FEATURES)}")

    reporter.emit("")
    reporter.emit("4. Validation backtest - recursive 2022 forecast")
    result_a = validate_model_variant("A", table_a, static_features, train_df, reporter)
    result_b = validate_model_variant("B", table_b, static_features, train_df, reporter)
    emit_metrics("Model A validation metrics:", result_a["metrics"], reporter)
    emit_metrics("Model B validation metrics:", result_b["metrics"], reporter)

    selected = choose_model_variant(result_a, result_b)
    selected_variant = selected["variant"]
    reporter.emit(f"Selected model: {selected_variant}")
    validation_output = save_validation_predictions(selected, VALIDATION_PREDICTIONS_PATH)
    reporter.emit(f"Saved validation predictions: {VALIDATION_PREDICTIONS_PATH}")
    reporter.emit(f"Validation prediction shape: {validation_output.shape}")

    selected_table = table_b if selected_variant == "B" else table_a
    final_model = train_final_selected_model(selected_variant, selected_table, reporter)

    reporter.emit("")
    reporter.emit("6. Recursive future forecast")
    initial_history = train_df.set_index(DATE_COL)[TARGET_COL].sort_index()
    business_maps = build_business_source_maps(train_df)
    revenue_predictions = recursive_predict(
        model=final_model["model"],
        model_type=final_model["model_type"],
        prediction_dates=sample_submission[DATE_COL],
        feature_columns=final_model["feature_columns"],
        static_features=static_features,
        initial_revenue_history=initial_history,
        business_maps=business_maps,
        feature_medians=final_model["feature_medians"],
        include_business_lag365=(selected_variant == "B"),
    )
    cogs_ratio = estimate_cogs_ratio(train_df)
    submission = build_submission(sample_submission, revenue_predictions, cogs_ratio, SUBMISSION_PATH)
    reporter.emit(f"Estimated COGS/Revenue ratio from latest 365 train days: {cogs_ratio:.6f}")
    reporter.emit(f"Saved submission: {SUBMISSION_PATH}")

    reporter.emit("")
    reporter.emit("7. Feature importance")
    importance = get_feature_importance(
        final_model["model"],
        final_model["model_type"],
        final_model["feature_columns"],
        final_model["X_all"],
        final_model["y_all"],
        selected["metrics"]["RMSE"],
    )
    importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    top30 = importance.head(30)
    reporter.emit_frame("Top 30 feature importances:", top30)
    reporter.emit(f"Saved feature importance: {FEATURE_IMPORTANCE_PATH}")

    reporter.emit("")
    reporter.emit("8. Final summary")
    emit_metrics("Selected validation metrics:", selected["metrics"], reporter)
    reporter.emit(f"Selected model: {selected_variant}")
    reporter.emit(f"Submission rows: {len(submission):,}")
    reporter.emit(
        f"Submission date range: {submission[DATE_COL].min().date()} -> "
        f"{submission[DATE_COL].max().date()}"
    )
    reporter.emit(
        "Leakage confirmation: no same-day realized demand, future Revenue, or future COGS was used. "
        "Revenue lags are generated recursively from actual history plus prior predictions; "
        "COGS is estimated from historical COGS/Revenue ratio."
    )
    reporter.emit(f"Final submission path: {SUBMISSION_PATH}")

    reporter.save_metrics(METRICS_PATH)


if __name__ == "__main__":
    run_training()
