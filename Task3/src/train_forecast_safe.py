from __future__ import annotations

import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "log"

DATASET_PATH = DATA_DIR / "daily_feature_table.csv"
PROMOTIONS_PATH = DATA_DIR / "promotions.csv"
PREDICTIONS_PATH = DATA_DIR / "forecast_safe_validation_predictions.csv"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "forecast_safe_feature_importance.csv"
METRICS_PATH = LOG_DIR / "forecast_safe_metrics.txt"
LOG_FILE = LOG_DIR / "train_forecast_safe.log"

DATE_COL = "Date"
TARGET_COL = "Revenue"
TRAIN_CUTOFF = pd.Timestamp("2022-01-01")
VALIDATION_END = pd.Timestamp("2022-12-31")

MODEL_1_METRICS = {
    "MAE": 232_155.73,
    "RMSE": 318_715.19,
    "R2": 0.963743,
}

BUSINESS_LAG_SOURCE_COLUMNS = [
    "orders_count",
    "unique_customers",
    "total_quantity",
    "promo_usage_rate",
]
WEB_LAG_SOURCE_COLUMNS = [
    "web_sessions",
    "web_unique_visitors",
    "web_page_views",
    "web_bounce_rate",
    "web_avg_session_duration_sec",
    "web_traffic_available",
]
LAG_PERIODS = [7, 14, 30]

UNAVAILABLE_SAME_DAY_COLUMNS = [
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
    "returns_count",
    "returned_orders_count",
    "return_quantity",
    "refund_amount",
    "return_rate",
    "review_count",
    "avg_rating",
    "payment_value",
    "payments_count",
    "web_sessions",
    "web_unique_visitors",
    "web_page_views",
    "web_bounce_rate",
    "web_avg_session_duration_sec",
    "web_traffic_available",
]


class RunReporter:
    """Collect console output and persist the same content to a metrics file."""

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
    """Configure simple file logging for the forecast-safe training run."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train_forecast_safe")
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
    """Load the daily feature table without modifying the source CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, parse_dates=[DATE_COL], low_memory=False)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce").dt.normalize()
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    if df[DATE_COL].isna().any():
        raise ValueError("Date column contains invalid timestamps")

    return df


def add_lagged_features(
    df: pd.DataFrame,
    source_columns: list[str],
    lags: list[int],
) -> tuple[pd.DataFrame, list[str]]:
    """Create leakage-safe lagged versions of selected same-day signals."""
    output = df.sort_values(DATE_COL).reset_index(drop=True).copy()
    created_features: list[str] = []

    for column in source_columns:
        if column not in output.columns:
            continue

        for lag in lags:
            lagged_name = f"{column}_lag_{lag}"
            output[lagged_name] = output[column].shift(lag)
            created_features.append(lagged_name)

    return output, created_features


def build_promotion_calendar(
    calendar_dates: pd.Series,
    promotions_path: Path = PROMOTIONS_PATH,
) -> pd.DataFrame:
    """Build date-level promotion features from known promotion schedules."""
    calendar = pd.DataFrame({DATE_COL: pd.to_datetime(calendar_dates).sort_values().unique()})

    default_columns = [
        "promo_calendar_active_count",
        "promo_calendar_avg_discount_value",
        "promo_calendar_max_discount_value",
        "promo_calendar_stackable_count",
        "promo_calendar_min_order_value_mean",
    ]
    for column in default_columns:
        calendar[column] = 0.0

    if not promotions_path.exists():
        return calendar

    promotions = pd.read_csv(promotions_path, low_memory=False)
    required_columns = {"start_date", "end_date", "promo_id"}
    if not required_columns.issubset(promotions.columns):
        return calendar

    promotions["start_date"] = pd.to_datetime(promotions["start_date"], errors="coerce").dt.normalize()
    promotions["end_date"] = pd.to_datetime(promotions["end_date"], errors="coerce").dt.normalize()
    promotions["discount_value"] = pd.to_numeric(
        promotions.get("discount_value", 0),
        errors="coerce",
    ).fillna(0)
    promotions["min_order_value"] = pd.to_numeric(
        promotions.get("min_order_value", 0),
        errors="coerce",
    ).fillna(0)

    if "stackable_flag" in promotions.columns:
        stackable_text = promotions["stackable_flag"].astype(str).str.strip().str.lower()
        promotions["stackable_flag_numeric"] = stackable_text.isin(["1", "true", "yes", "y"]).astype(int)
    else:
        promotions["stackable_flag_numeric"] = 0

    rows: list[dict[str, object]] = []
    min_calendar_date = calendar[DATE_COL].min()
    max_calendar_date = calendar[DATE_COL].max()

    for row in promotions.dropna(subset=["start_date", "end_date"]).itertuples(index=False):
        start_date = max(row.start_date, min_calendar_date)
        end_date = min(row.end_date, max_calendar_date)
        if start_date > end_date:
            continue

        for active_date in pd.date_range(start_date, end_date, freq="D"):
            rows.append(
                {
                    DATE_COL: active_date,
                    "promo_id": row.promo_id,
                    "discount_value": row.discount_value,
                    "stackable_flag_numeric": row.stackable_flag_numeric,
                    "min_order_value": row.min_order_value,
                }
            )

    if not rows:
        return calendar

    expanded = pd.DataFrame(rows)
    daily_promos = (
        expanded.groupby(DATE_COL, as_index=False)
        .agg(
            promo_calendar_active_count=("promo_id", "nunique"),
            promo_calendar_avg_discount_value=("discount_value", "mean"),
            promo_calendar_max_discount_value=("discount_value", "max"),
            promo_calendar_stackable_count=("stackable_flag_numeric", "sum"),
            promo_calendar_min_order_value_mean=("min_order_value", "mean"),
        )
    )

    calendar = calendar.drop(columns=default_columns).merge(daily_promos, on=DATE_COL, how="left")
    for column in default_columns:
        calendar[column] = calendar[column].fillna(0)

    return calendar


def add_promotion_features(df: pd.DataFrame, reporter: RunReporter) -> tuple[pd.DataFrame, list[str]]:
    """Join forecast-known promotion calendar features to the modeling table."""
    promo_features = build_promotion_calendar(df[DATE_COL], PROMOTIONS_PATH)
    added_columns = [column for column in promo_features.columns if column != DATE_COL]
    output = df.merge(promo_features, on=DATE_COL, how="left", validate="one_to_one")
    output[added_columns] = output[added_columns].fillna(0)

    reporter.emit(f"Added promotion calendar features: {added_columns}")
    return output, added_columns


def prepare_forecast_safe_table(
    df: pd.DataFrame,
    reporter: RunReporter,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """Create lagged signals, add safe calendar features, and drop unavailable columns."""
    output = df.sort_values(DATE_COL).reset_index(drop=True).copy()

    reporter.emit("")
    reporter.emit("2. Forecast-safe feature preparation")

    output, business_lag_features = add_lagged_features(
        output,
        BUSINESS_LAG_SOURCE_COLUMNS,
        LAG_PERIODS,
    )
    output, web_lag_features = add_lagged_features(
        output,
        WEB_LAG_SOURCE_COLUMNS,
        LAG_PERIODS,
    )
    output, promo_calendar_features = add_promotion_features(output, reporter)

    dropped_columns = [column for column in UNAVAILABLE_SAME_DAY_COLUMNS if column in output.columns]
    output = output.drop(columns=dropped_columns)

    missing_before = int(output.isna().sum().sum())
    before_rows = len(output)
    output = output.dropna().reset_index(drop=True)
    dropped_rows = before_rows - len(output)

    created_lag_features = business_lag_features + web_lag_features
    reporter.emit(f"Created lagged business features: {business_lag_features}")
    reporter.emit(f"Created lagged web features: {web_lag_features}")
    reporter.emit(f"Dropped unavailable same-day features: {dropped_columns}")
    reporter.emit(f"Total missing values before dropna: {missing_before:,}")
    reporter.emit(f"Rows dropped after lag creation/dropna: {dropped_rows:,}")
    reporter.emit(f"Forecast-safe table shape: {output.shape}")

    return output, dropped_columns, created_lag_features, promo_calendar_features


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create model matrix after forecast-safe filtering."""
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
    """Use the fixed 2022 validation period."""
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


def train_model(X_train: pd.DataFrame, y_train: pd.Series, reporter: RunReporter) -> lgb.Booster:
    """Train the forecast-safe LightGBM baseline."""
    reporter.emit("")
    reporter.emit("4. Train LightGBM forecast-safe baseline")
    reporter.emit(
        "Parameters: n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42"
    )

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
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

    model = lgb.train(
        params=params,
        train_set=train_data,
        num_boost_round=1000,
    )
    reporter.emit("Training complete.")
    return model


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, and R2 using numpy."""
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


def save_feature_importance(
    model: lgb.Booster,
    feature_names: list[str],
    path: Path = FEATURE_IMPORTANCE_PATH,
) -> pd.DataFrame:
    """Save LightGBM feature importance by split and gain."""
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_split": model.feature_importance(importance_type="split"),
            "importance_gain": model.feature_importance(importance_type="gain"),
        }
    )
    importance = importance.sort_values(
        ["importance_gain", "importance_split"],
        ascending=False,
    ).reset_index(drop=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(path, index=False)
    return importance


def compare_with_model_1(metrics: dict[str, float], reporter: RunReporter) -> None:
    """Compare forecast-safe model metrics against the same-day baseline."""
    mae_change = metrics["MAE"] - MODEL_1_METRICS["MAE"]
    rmse_change = metrics["RMSE"] - MODEL_1_METRICS["RMSE"]
    r2_change = metrics["R2"] - MODEL_1_METRICS["R2"]
    mae_pct = mae_change / MODEL_1_METRICS["MAE"] * 100
    rmse_pct = rmse_change / MODEL_1_METRICS["RMSE"] * 100

    significant_drop = mae_pct >= 20 or rmse_pct >= 20 or r2_change <= -0.05

    reporter.emit("")
    reporter.emit("6. Comparison with Model 1")
    reporter.emit(
        f"Model 1 - MAE={MODEL_1_METRICS['MAE']:,.2f}, "
        f"RMSE={MODEL_1_METRICS['RMSE']:,.2f}, R2={MODEL_1_METRICS['R2']:.6f}"
    )
    reporter.emit(
        f"Model 2 - MAE={metrics['MAE']:,.2f}, "
        f"RMSE={metrics['RMSE']:,.2f}, R2={metrics['R2']:.6f}"
    )
    reporter.emit(f"MAE change vs Model 1: {mae_change:,.2f} ({mae_pct:.2f}%)")
    reporter.emit(f"RMSE change vs Model 1: {rmse_change:,.2f} ({rmse_pct:.2f}%)")
    reporter.emit(f"R2 change vs Model 1: {r2_change:.6f}")

    if significant_drop:
        reporter.emit(
            "Performance drop: significant, and expected because Model 2 removes same-day realized demand."
        )
    else:
        reporter.emit(
            "Performance drop: not significant by the configured rule; the model remains forecast-safe."
        )


def summarize_feature_groups(top_features: pd.DataFrame) -> dict[str, bool]:
    """Summarize whether lag and business signals appear in the top 20."""
    features = top_features["feature"].tolist()
    lag_features_present = any("_lag_" in feature or feature.startswith("lag_") for feature in features)
    revenue_lags_present = any(
        feature.startswith("lag_") or feature.startswith("rolling_mean_")
        for feature in features
    )
    business_lags_present = any(
        feature.startswith(
            (
                "orders_count_lag_",
                "unique_customers_lag_",
                "total_quantity_lag_",
                "promo_usage_rate_lag_",
            )
        )
        for feature in features
    )
    web_lags_present = any(feature.startswith("web_") and "_lag_" in feature for feature in features)
    inventory_present = any(feature.startswith("inventory_") for feature in features)
    promo_calendar_present = any(feature.startswith("promo_calendar_") for feature in features)

    return {
        "lag_features_present": lag_features_present,
        "revenue_lags_present": revenue_lags_present,
        "business_lags_present": business_lags_present,
        "web_lags_present": web_lags_present,
        "inventory_present": inventory_present,
        "promo_calendar_present": promo_calendar_present,
    }


def print_final_summary(
    metrics: dict[str, float],
    top20: pd.DataFrame,
    dropped_columns: list[str],
    created_lag_features: list[str],
    reporter: RunReporter,
) -> None:
    """Print the required final forecast-safe summary."""
    groups = summarize_feature_groups(top20)

    reporter.emit("")
    reporter.emit("8. Final summary")
    reporter.emit(
        f"Forecast-safe metrics: MAE={metrics['MAE']:,.2f}, "
        f"RMSE={metrics['RMSE']:,.2f}, R2={metrics['R2']:.6f}"
    )
    reporter.emit(f"Dropped unavailable same-day features: {dropped_columns}")
    reporter.emit(f"Newly created lagged business/web features: {created_lag_features}")
    reporter.emit(
        "Lag features in top 20: "
        + ("yes" if groups["lag_features_present"] else "no")
    )
    reporter.emit(
        "Business lag features in top 20: "
        + ("yes" if groups["business_lags_present"] else "no")
    )
    reporter.emit(
        "Web/inventory/promotion features in top 20: "
        + (
            "yes"
            if groups["web_lags_present"] or groups["inventory_present"] or groups["promo_calendar_present"]
            else "no"
        )
    )
    reporter.emit(
        "True future forecasting validity: YES for forecast-safe backtesting and rolling forecasts - "
        "same-day realized demand, returns, discounts, COGS, and raw web traffic were removed. "
        "For a blind multi-month 2023-2024 horizon, lagged business/web exogenous signals must be "
        "observed, separately forecasted, or removed."
    )


def run_training() -> None:
    logger = setup_logging()
    reporter = RunReporter(logger)

    reporter.emit("Forecast-safe LightGBM Training")
    reporter.emit("================================")
    reporter.emit("")
    reporter.emit("1. Load data")

    df = load_feature_table(DATASET_PATH)
    reporter.emit(f"Loaded dataset: {DATASET_PATH}")
    reporter.emit(f"Raw shape: {df.shape}")
    reporter.emit(f"Date range: {df[DATE_COL].min().date()} -> {df[DATE_COL].max().date()}")

    safe_df, dropped_columns, created_lag_features, promo_calendar_features = prepare_forecast_safe_table(
        df,
        reporter,
    )
    reporter.emit(f"Promotion calendar features kept: {promo_calendar_features}")

    X, y = split_features_target(safe_df)
    reporter.emit(f"Feature matrix shape: {X.shape}")
    reporter.emit(f"Target vector length: {len(y):,}")

    X_train, X_valid, y_train, y_valid, valid_dates = time_based_split(safe_df, X, y, reporter)
    model = train_model(X_train, y_train, reporter)

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

    compare_with_model_1(metrics, reporter)

    reporter.emit("")
    reporter.emit("7. Feature importance")
    importance = save_feature_importance(model, X.columns.tolist(), FEATURE_IMPORTANCE_PATH)
    top20 = importance.head(20)
    reporter.emit_frame("Top 20 features by LightGBM gain importance:", top20)
    reporter.emit(f"Saved feature importance: {FEATURE_IMPORTANCE_PATH}")

    print_final_summary(metrics, top20, dropped_columns, created_lag_features, reporter)
    reporter.save_metrics(METRICS_PATH)


if __name__ == "__main__":
    run_training()
