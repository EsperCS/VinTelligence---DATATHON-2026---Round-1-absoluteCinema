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

DATASET_PATH = DATA_DIR / "daily_feature_table.csv"
PROMOTIONS_PATH = DATA_DIR / "promotions.csv"
INVENTORY_PATH = DATA_DIR / "inventory.csv"
PREDICTIONS_PATH = DATA_DIR / "forecast_safe_v2_validation_predictions.csv"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "forecast_safe_v2_feature_importance.csv"
METRICS_PATH = LOG_DIR / "forecast_safe_v2_metrics.txt"
LOG_FILE = LOG_DIR / "train_forecast_safe_v2.log"

DATE_COL = "Date"
TARGET_COL = "Revenue"
TRAIN_CUTOFF = pd.Timestamp("2022-01-01")
VALIDATION_END = pd.Timestamp("2022-12-31")

PREVIOUS_MODEL_METRICS = {
    "MAE": 755_477.45,
    "RMSE": 1_048_173.70,
    "R2": 0.607848,
}

REVENUE_LAGS = [60, 90, 180, 365]
REVENUE_ROLL_MEAN_WINDOWS = [14, 60, 90, 180, 365]
REVENUE_ROLL_STD_WINDOWS = [30, 90, 365]
BUSINESS_LAG_PERIODS = [7, 14, 30, 60, 90, 180, 365]
BUSINESS_ROLL_WINDOWS = [7, 30, 90]

BUSINESS_SOURCE_COLUMNS = [
    "orders_count",
    "unique_customers",
    "total_quantity",
    "item_lines_count",
    "promo_usage_rate",
    "avg_discount_rate",
    "web_sessions",
    "web_unique_visitors",
    "web_page_views",
    "web_bounce_rate",
    "web_avg_session_duration_sec",
]

UNSAFE_SAME_DAY_COLUMNS = [
    "COGS",
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
    """Print progress, log it, and save the same content as a metrics report."""

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
        self.logger.info("Saved metrics report to %s", path)


def setup_logging(log_file: Path = LOG_FILE) -> logging.Logger:
    """Configure simple file logging."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train_forecast_safe_v2")
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


def load_feature_table(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load the daily feature table without modifying the original CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, parse_dates=[DATE_COL], low_memory=False)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce").dt.normalize()
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    if df[DATE_COL].isna().any():
        raise ValueError("Date column contains invalid timestamps")

    return df


def add_revenue_lag_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add long revenue lags and past-only rolling statistics."""
    output = df.sort_values(DATE_COL).reset_index(drop=True).copy()
    new_features: dict[str, pd.Series] = {}
    created: list[str] = []

    for lag in REVENUE_LAGS:
        feature = f"revenue_lag_{lag}"
        new_features[feature] = output[TARGET_COL].shift(lag)
        created.append(feature)

    shifted_revenue = output[TARGET_COL].shift(1)
    for window in REVENUE_ROLL_MEAN_WINDOWS:
        feature = f"revenue_roll_mean_{window}"
        new_features[feature] = shifted_revenue.rolling(window=window, min_periods=window).mean()
        created.append(feature)

    for window in REVENUE_ROLL_STD_WINDOWS:
        feature = f"revenue_roll_std_{window}"
        new_features[feature] = shifted_revenue.rolling(window=window, min_periods=window).std()
        created.append(feature)

    if new_features:
        output = pd.concat([output, pd.DataFrame(new_features)], axis=1).copy()

    return output, created


def add_calendar_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add deterministic date features available for any forecast horizon."""
    output = df.sort_values(DATE_COL).reset_index(drop=True).copy()
    created: list[str] = []

    new_features = {
        "day_of_year": output[DATE_COL].dt.dayofyear.astype(int),
        "week_of_year": output[DATE_COL].dt.isocalendar().week.astype(int),
        "is_month_start": output[DATE_COL].dt.is_month_start.astype(int),
        "is_month_end": output[DATE_COL].dt.is_month_end.astype(int),
        "is_quarter_start": output[DATE_COL].dt.is_quarter_start.astype(int),
        "is_quarter_end": output[DATE_COL].dt.is_quarter_end.astype(int),
    }

    for feature, values in new_features.items():
        output[feature] = values
        created.append(feature)

    if "day_of_week" not in output.columns:
        output["day_of_week"] = output[DATE_COL].dt.dayofweek.astype(int)
        created.append("day_of_week")
    if "month" not in output.columns:
        output["month"] = output[DATE_COL].dt.month.astype(int)
        created.append("month")
    if "quarter" not in output.columns:
        output["quarter"] = output[DATE_COL].dt.quarter.astype(int)
        created.append("quarter")
    if "year" not in output.columns:
        output["year"] = output[DATE_COL].dt.year.astype(int)
        created.append("year")
    if "is_weekend" not in output.columns:
        output["is_weekend"] = output["day_of_week"].isin([5, 6]).astype(int)
        created.append("is_weekend")
    if "time_index" not in output.columns:
        output["time_index"] = np.arange(len(output), dtype=int)
        created.append("time_index")
    if "post_2019_flag" not in output.columns:
        output["post_2019_flag"] = (output[DATE_COL] >= pd.Timestamp("2019-01-01")).astype(int)
        created.append("post_2019_flag")

    return output, created


def add_regime_shift_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add features for the post-2019 revenue regime shift."""
    output = df.copy()

    if "year" not in output.columns:
        output["year"] = output[DATE_COL].dt.year.astype(int)
    if "time_index" not in output.columns:
        output["time_index"] = np.arange(len(output), dtype=int)
    if "post_2019_flag" not in output.columns:
        output["post_2019_flag"] = (output[DATE_COL] >= pd.Timestamp("2019-01-01")).astype(int)

    min_year = int(output["year"].min())
    output["years_since_start"] = output["year"] - min_year
    output["years_since_2019"] = np.maximum(0, output["year"] - 2019)
    output["post_2019_time_index"] = output["time_index"] * output["post_2019_flag"]

    return output, ["years_since_start", "years_since_2019", "post_2019_time_index"]


def add_business_lag_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Lag realized business/web signals before dropping same-day originals."""
    output = df.sort_values(DATE_COL).reset_index(drop=True).copy()
    new_features: dict[str, pd.Series] = {}
    created: list[str] = []

    for column in BUSINESS_SOURCE_COLUMNS:
        if column not in output.columns:
            continue

        for lag in BUSINESS_LAG_PERIODS:
            feature = f"{column}_lag_{lag}"
            new_features[feature] = output[column].shift(lag)
            created.append(feature)

        shifted = output[column].shift(1)
        for window in BUSINESS_ROLL_WINDOWS:
            feature = f"{column}_roll_mean_{window}"
            new_features[feature] = shifted.rolling(window=window, min_periods=window).mean()
            created.append(feature)

    if new_features:
        output = pd.concat([output, pd.DataFrame(new_features)], axis=1).copy()

    return output, created


def _stackable_to_int(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    return text.isin(["1", "true", "yes", "y"]).astype(int)


def build_promotion_calendar(
    calendar_dates: pd.Series,
    promotions_path: Path = PROMOTIONS_PATH,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Create daily promotion features from known promotion schedules."""
    calendar = pd.DataFrame({DATE_COL: pd.to_datetime(calendar_dates).sort_values().unique()})
    feature_columns = [
        "calendar_active_promo_count",
        "calendar_any_promo",
        "calendar_avg_discount_value",
        "calendar_max_discount_value",
        "calendar_stackable_promo_count",
        "calendar_has_category_specific_promo",
    ]
    for column in feature_columns:
        calendar[column] = 0.0

    if not promotions_path.exists():
        if logger:
            logger.warning("promotions.csv not found at %s; skipping promotion calendar", promotions_path)
        return calendar

    promotions = pd.read_csv(promotions_path, low_memory=False)
    required_columns = {"promo_id", "start_date", "end_date"}
    if not required_columns.issubset(promotions.columns):
        if logger:
            logger.warning("promotions.csv missing required columns; skipping promotion calendar")
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
    if "applicable_category" in promotions.columns:
        promotions["category_specific"] = promotions["applicable_category"].notna().astype(int)
    else:
        promotions["category_specific"] = 0

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
                }
            )

    if not rows:
        return calendar

    expanded = pd.DataFrame(rows)
    daily_promos = (
        expanded.groupby(DATE_COL, as_index=False)
        .agg(
            calendar_active_promo_count=("promo_id", "nunique"),
            calendar_avg_discount_value=("discount_value", "mean"),
            calendar_max_discount_value=("discount_value", "max"),
            calendar_stackable_promo_count=("stackable_flag_numeric", "sum"),
            calendar_has_category_specific_promo=("category_specific", "max"),
        )
    )
    daily_promos["calendar_any_promo"] = (
        daily_promos["calendar_active_promo_count"] > 0
    ).astype(int)

    calendar = calendar.drop(columns=feature_columns).merge(daily_promos, on=DATE_COL, how="left")
    for column in feature_columns:
        calendar[column] = calendar[column].fillna(0)

    return calendar[[DATE_COL] + feature_columns]


def add_promotion_calendar_features(
    df: pd.DataFrame,
    reporter: RunReporter,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, list[str]]:
    """Join promotion-calendar features to the daily table."""
    promo_calendar = build_promotion_calendar(df[DATE_COL], PROMOTIONS_PATH, logger)
    feature_columns = [column for column in promo_calendar.columns if column != DATE_COL]
    output = df.merge(promo_calendar, on=DATE_COL, how="left", validate="one_to_one")
    output[feature_columns] = output[feature_columns].fillna(0)

    reporter.emit(f"Created promotion calendar features: {feature_columns}")
    return output, feature_columns


def build_inventory_asof_features(
    calendar_dates: pd.Series,
    inventory_path: Path = INVENTORY_PATH,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Aggregate inventory snapshots and merge only past snapshots to each date."""
    calendar = pd.DataFrame({DATE_COL: pd.to_datetime(calendar_dates).sort_values().unique()})
    feature_columns = [
        "inv_stockout_rate",
        "inv_avg_fill_rate",
        "inv_avg_days_of_supply",
        "inv_avg_sell_through_rate",
        "inv_reorder_rate",
        "inv_overstock_rate",
    ]
    for column in feature_columns:
        calendar[column] = 0.0

    if not inventory_path.exists():
        if logger:
            logger.warning("inventory.csv not found at %s; skipping inventory as-of features", inventory_path)
        return calendar

    inventory = pd.read_csv(inventory_path, low_memory=False)
    if "snapshot_date" not in inventory.columns:
        if logger:
            logger.warning("inventory.csv missing snapshot_date; skipping inventory as-of features")
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

    calendar_dates_sorted = calendar[[DATE_COL]].sort_values(DATE_COL)
    merged = pd.merge_asof(
        calendar_dates_sorted,
        snapshots,
        on=DATE_COL,
        direction="backward",
    )
    merged[feature_columns] = merged[feature_columns].fillna(0)
    return merged[[DATE_COL] + feature_columns]


def add_inventory_asof_features(
    df: pd.DataFrame,
    reporter: RunReporter,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, list[str]]:
    """Join monthly inventory snapshot features with no backward leakage."""
    inventory_features = build_inventory_asof_features(df[DATE_COL], INVENTORY_PATH, logger)
    feature_columns = [column for column in inventory_features.columns if column != DATE_COL]
    output = df.merge(inventory_features, on=DATE_COL, how="left", validate="one_to_one")
    output[feature_columns] = output[feature_columns].fillna(0)

    reporter.emit(f"Created inventory as-of features: {feature_columns}")
    return output, feature_columns


def drop_unsafe_same_day_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop realized same-day features after lagged versions have been created."""
    dropped_columns = [column for column in UNSAFE_SAME_DAY_COLUMNS if column in df.columns]
    return df.drop(columns=dropped_columns), dropped_columns


def prepare_forecast_safe_v2_table(
    df: pd.DataFrame,
    reporter: RunReporter,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, list[str]], list[str]]:
    """Apply all forecast-safe feature expansion steps."""
    output = df.sort_values(DATE_COL).reset_index(drop=True).copy()
    feature_groups: dict[str, list[str]] = {}

    reporter.emit("")
    reporter.emit("2. Forecast-safe v2 feature expansion")

    output, feature_groups["revenue_lag_roll"] = add_revenue_lag_features(output)
    output, feature_groups["calendar"] = add_calendar_features(output)
    output, feature_groups["regime_shift"] = add_regime_shift_features(output)
    output, feature_groups["business_lag_roll"] = add_business_lag_features(output)
    output, feature_groups["promotion_calendar"] = add_promotion_calendar_features(
        output,
        reporter,
        logger,
    )
    output, feature_groups["inventory_asof"] = add_inventory_asof_features(output, reporter, logger)

    output, dropped_columns = drop_unsafe_same_day_features(output)

    missing_before = int(output.isna().sum().sum())
    before_rows = len(output)
    output = output.dropna().reset_index(drop=True)
    dropped_rows = before_rows - len(output)

    for group_name, features in feature_groups.items():
        reporter.emit(f"Created {group_name} features ({len(features)}): {features}")
    reporter.emit(f"Dropped unsafe same-day features ({len(dropped_columns)}): {dropped_columns}")
    reporter.emit(f"Total missing values before dropna: {missing_before:,}")
    reporter.emit(f"Rows dropped after lag/rolling creation: {dropped_rows:,}")
    reporter.emit(f"Forecast-safe v2 table shape: {output.shape}")

    return output, feature_groups, dropped_columns


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create numeric feature matrix and target."""
    X = df.drop(columns=[DATE_COL, TARGET_COL])
    y = df[TARGET_COL]

    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns found: {non_numeric}")

    return X, y


def time_based_split(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    reporter: RunReporter,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split train/validation by fixed calendar dates."""
    train_mask = df[DATE_COL] < TRAIN_CUTOFF
    valid_mask = (df[DATE_COL] >= TRAIN_CUTOFF) & (df[DATE_COL] <= VALIDATION_END)

    X_train = X.loc[train_mask].copy()
    X_valid = X.loc[valid_mask].copy()
    y_train = y.loc[train_mask].copy()
    y_valid = y.loc[valid_mask].copy()
    valid_dates = df.loc[valid_mask, DATE_COL].copy()

    if X_train.empty or X_valid.empty:
        raise ValueError("Train or validation split is empty")

    reporter.emit("")
    reporter.emit("3. Time-based split")
    reporter.emit(f"Train rows: {len(X_train):,}")
    reporter.emit(f"Validation rows: {len(X_valid):,}")
    reporter.emit(
        "Train date range: "
        f"{df.loc[train_mask, DATE_COL].min().date()} -> {df.loc[train_mask, DATE_COL].max().date()}"
    )
    reporter.emit(
        "Validation date range: "
        f"{df.loc[valid_mask, DATE_COL].min().date()} -> "
        f"{df.loc[valid_mask, DATE_COL].max().date()}"
    )

    return X_train, X_valid, y_train, y_valid, valid_dates


def lightgbm_available() -> bool:
    return importlib.util.find_spec("lightgbm") is not None


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    reporter: RunReporter,
) -> Any:
    """Train native LightGBM without requiring scikit-learn wrappers."""
    import lightgbm as lgb

    reporter.emit("")
    reporter.emit("4. Train model")
    reporter.emit("Using LightGBM")
    reporter.emit(
        "Parameters: n_estimators=1500, learning_rate=0.03, max_depth=6, random_state=42"
    )

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.03,
        "max_depth": 6,
        "seed": 42,
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
    reporter.emit("Training complete.")
    return model


def train_hist_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    reporter: RunReporter,
) -> Any:
    """Fallback model if LightGBM is not installed."""
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError as exc:
        raise ImportError(
            "LightGBM is unavailable and scikit-learn fallback is not installed."
        ) from exc

    reporter.emit("")
    reporter.emit("4. Train model")
    reporter.emit("Using sklearn HistGradientBoostingRegressor fallback")
    model = HistGradientBoostingRegressor(
        learning_rate=0.03,
        max_iter=1500,
        max_leaf_nodes=31,
        random_state=42,
    )
    model.fit(X_train, y_train)
    reporter.emit("Training complete.")
    return model


def train_model(X_train: pd.DataFrame, y_train: pd.Series, reporter: RunReporter) -> tuple[Any, str]:
    """Train LightGBM when available, otherwise use the sklearn fallback."""
    if lightgbm_available():
        return train_lightgbm(X_train, y_train, reporter), "lightgbm"

    return train_hist_gradient_boosting(X_train, y_train, reporter), "hist_gradient_boosting"


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


def save_predictions(
    dates: pd.Series,
    y_valid: pd.Series,
    y_pred: np.ndarray,
    path: Path = PREDICTIONS_PATH,
) -> pd.DataFrame:
    """Save validation predictions."""
    predictions = pd.DataFrame(
        {
            DATE_COL: dates.to_numpy(),
            "actual_Revenue": y_valid.to_numpy(),
            "predicted_Revenue": y_pred,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, index=False)
    return predictions


def lightgbm_feature_importance(model: Any, feature_names: list[str]) -> pd.DataFrame:
    """Return LightGBM split and gain importance."""
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_split": model.feature_importance(importance_type="split"),
            "importance_gain": model.feature_importance(importance_type="gain"),
        }
    )
    return importance.sort_values(
        ["importance_gain", "importance_split"],
        ascending=False,
    ).reset_index(drop=True)


def permutation_feature_importance(
    model: Any,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    baseline_rmse: float,
    random_state: int = 42,
) -> pd.DataFrame:
    """Manual permutation importance for fallback models without native importance."""
    rng = np.random.default_rng(random_state)
    importance_rows: list[dict[str, float | str]] = []

    for column in X_valid.columns:
        permuted = X_valid.copy()
        permuted[column] = rng.permutation(permuted[column].to_numpy())
        permuted_pred = model.predict(permuted)
        permuted_rmse = evaluate_predictions(y_valid, permuted_pred)["RMSE"]
        importance_rows.append(
            {
                "feature": column,
                "importance_split": 0,
                "importance_gain": permuted_rmse - baseline_rmse,
            }
        )

    return pd.DataFrame(importance_rows).sort_values(
        "importance_gain",
        ascending=False,
    ).reset_index(drop=True)


def save_feature_importance(
    model: Any,
    model_type: str,
    feature_names: list[str],
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    metrics: dict[str, float],
    path: Path = FEATURE_IMPORTANCE_PATH,
) -> pd.DataFrame:
    """Save top features from native or permutation importance."""
    if model_type == "lightgbm":
        importance = lightgbm_feature_importance(model, feature_names)
    else:
        importance = permutation_feature_importance(model, X_valid, y_valid, metrics["RMSE"])

    path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(path, index=False)
    return importance


def compare_with_previous(metrics: dict[str, float], reporter: RunReporter) -> dict[str, float | bool]:
    """Compare v2 metrics against the previous forecast-safe baseline."""
    mae_change = metrics["MAE"] - PREVIOUS_MODEL_METRICS["MAE"]
    rmse_change = metrics["RMSE"] - PREVIOUS_MODEL_METRICS["RMSE"]
    r2_change = metrics["R2"] - PREVIOUS_MODEL_METRICS["R2"]
    mae_pct = mae_change / PREVIOUS_MODEL_METRICS["MAE"] * 100
    rmse_pct = rmse_change / PREVIOUS_MODEL_METRICS["RMSE"] * 100

    improved = (
        metrics["MAE"] < PREVIOUS_MODEL_METRICS["MAE"]
        and metrics["RMSE"] < PREVIOUS_MODEL_METRICS["RMSE"]
        and metrics["R2"] > PREVIOUS_MODEL_METRICS["R2"]
    )

    reporter.emit("")
    reporter.emit("6. Comparison with previous forecast-safe Model 2")
    reporter.emit(
        f"Previous Model 2 - MAE={PREVIOUS_MODEL_METRICS['MAE']:,.2f}, "
        f"RMSE={PREVIOUS_MODEL_METRICS['RMSE']:,.2f}, R2={PREVIOUS_MODEL_METRICS['R2']:.6f}"
    )
    reporter.emit(
        f"Model v2 - MAE={metrics['MAE']:,.2f}, "
        f"RMSE={metrics['RMSE']:,.2f}, R2={metrics['R2']:.6f}"
    )
    reporter.emit(f"MAE change: {mae_change:,.2f} ({mae_pct:.2f}%)")
    reporter.emit(f"RMSE change: {rmse_change:,.2f} ({rmse_pct:.2f}%)")
    reporter.emit(f"R2 change: {r2_change:.6f}")
    reporter.emit("Performance improved: " + ("yes" if improved else "no"))

    return {
        "mae_change": mae_change,
        "rmse_change": rmse_change,
        "r2_change": r2_change,
        "mae_pct": mae_pct,
        "rmse_pct": rmse_pct,
        "improved": improved,
    }


def print_final_summary(
    metrics: dict[str, float],
    comparison: dict[str, float | bool],
    top30: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    dropped_columns: list[str],
    reporter: RunReporter,
) -> None:
    """Print required final summary."""
    reporter.emit("")
    reporter.emit("8. Final summary")
    reporter.emit(
        f"Model v2 metrics: MAE={metrics['MAE']:,.2f}, "
        f"RMSE={metrics['RMSE']:,.2f}, R2={metrics['R2']:.6f}"
    )
    reporter.emit(
        "Improvement vs previous forecast-safe Model 2: "
        + ("yes" if comparison["improved"] else "no")
    )
    reporter.emit_frame("Top 30 feature importances:", top30)
    reporter.emit(
        "Created feature groups: "
        + str({group: len(features) for group, features in feature_groups.items()})
    )
    reporter.emit(f"Dropped unsafe same-day features: {dropped_columns}")
    reporter.emit(
        "Forecast-safe status: YES - same-day realized demand, payment, return, review, "
        "raw discount, raw web, and COGS features are removed after creating past-only lags/rollings."
    )
    reporter.emit(
        "Next recommended step: run a known-in-advance-only variant for blind 2023-2024 submission, "
        "then compare it with this rolling-forecast-safe v2 model."
    )


def run_training() -> None:
    logger = setup_logging()
    reporter = RunReporter(logger)

    reporter.emit("Forecast-safe LightGBM Training v2")
    reporter.emit("===================================")
    reporter.emit("")
    reporter.emit("1. Load data")

    raw_df = load_feature_table(DATASET_PATH)
    reporter.emit(f"Loaded dataset: {DATASET_PATH}")
    reporter.emit(f"Raw shape: {raw_df.shape}")
    reporter.emit(f"Date range: {raw_df[DATE_COL].min().date()} -> {raw_df[DATE_COL].max().date()}")

    safe_df, feature_groups, dropped_columns = prepare_forecast_safe_v2_table(
        raw_df,
        reporter,
        logger,
    )
    X, y = split_features_target(safe_df)
    reporter.emit(f"Feature matrix shape: {X.shape}")
    reporter.emit(f"Target vector length: {len(y):,}")

    X_train, X_valid, y_train, y_valid, valid_dates = time_based_split(safe_df, X, y, reporter)
    model, model_type = train_model(X_train, y_train, reporter)

    reporter.emit("")
    reporter.emit("5. Evaluation")
    y_pred = model.predict(X_valid)
    metrics = evaluate_predictions(y_valid, y_pred)
    for metric_name, metric_value in metrics.items():
        if metric_name == "R2":
            reporter.emit(f"{metric_name}: {metric_value:.6f}")
        else:
            reporter.emit(f"{metric_name}: {metric_value:,.2f}")

    predictions = save_predictions(valid_dates, y_valid, y_pred, PREDICTIONS_PATH)
    reporter.emit(f"Saved validation predictions: {PREDICTIONS_PATH}")
    reporter.emit(f"Prediction output shape: {predictions.shape}")

    comparison = compare_with_previous(metrics, reporter)

    reporter.emit("")
    reporter.emit("7. Feature importance")
    importance = save_feature_importance(
        model,
        model_type,
        X.columns.tolist(),
        X_valid,
        y_valid,
        metrics,
        FEATURE_IMPORTANCE_PATH,
    )
    top30 = importance.head(30)
    reporter.emit_frame("Top 30 features by importance:", top30)
    reporter.emit(f"Saved feature importance: {FEATURE_IMPORTANCE_PATH}")

    print_final_summary(metrics, comparison, top30, feature_groups, dropped_columns, reporter)
    reporter.save_metrics(METRICS_PATH)


if __name__ == "__main__":
    run_training()
