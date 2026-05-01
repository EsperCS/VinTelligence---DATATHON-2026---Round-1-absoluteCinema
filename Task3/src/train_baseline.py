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
PREDICTIONS_PATH = DATA_DIR / "validation_predictions.csv"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "feature_importance.csv"
METRICS_PATH = LOG_DIR / "baseline_metrics.txt"
LOG_FILE = LOG_DIR / "train_baseline.log"

TARGET_COL = "Revenue"
DATE_COL = "Date"
LEAKAGE_COLUMNS = ["COGS"]
TRAIN_CUTOFF = pd.Timestamp("2022-01-01")
VALIDATION_END = pd.Timestamp("2022-12-31")


class RunReporter:
    """Collect messages, print them, and mirror them to a log file."""

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

    logger = logging.getLogger("train_baseline")
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


def load_data(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load the daily feature table and sort by Date."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, parse_dates=[DATE_COL], low_memory=False)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce").dt.normalize()
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    if df[DATE_COL].isna().any():
        raise ValueError("Date column contains missing or invalid timestamps")

    return df


def clean_data(df: pd.DataFrame, reporter: RunReporter) -> pd.DataFrame:
    """Drop lag/rolling missing rows and remove known leakage columns."""
    before_rows = len(df)
    missing_before = int(df.isna().sum().sum())
    cleaned = df.dropna().copy()
    dropped_rows = before_rows - len(cleaned)

    reporter.emit("2. Data cleaning")
    reporter.emit(f"Total missing values before dropna: {missing_before:,}")
    reporter.emit(f"Rows dropped due to missing lag/rolling features: {dropped_rows:,}")

    missing_leakage_cols = [col for col in LEAKAGE_COLUMNS if col not in cleaned.columns]
    if missing_leakage_cols:
        raise ValueError(f"Expected leakage columns not found: {missing_leakage_cols}")

    cleaned = cleaned.drop(columns=LEAKAGE_COLUMNS)
    reporter.emit(f"Dropped leakage columns: {LEAKAGE_COLUMNS}")
    reporter.emit(f"Cleaned shape: {cleaned.shape}")

    return cleaned


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create X and y exactly from the cleaned modeling table."""
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
    """Split train and validation by calendar time."""
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
    """Train a simple LightGBM baseline on the training period only."""
    reporter.emit("")
    reporter.emit("4. Train LightGBM baseline")
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
    """Compute regression metrics without an sklearn dependency."""
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
    """Save LightGBM split and gain importance sorted by gain."""
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


def summarize_feature_groups(top_features: pd.DataFrame) -> dict[str, bool]:
    """Check whether expected feature groups appear in the top importance list."""
    feature_names = top_features["feature"].tolist()

    lag_dominant = any(
        feature.startswith("lag_") or feature.startswith("rolling_")
        for feature in feature_names[:5]
    )
    business_features_present = any(
        feature.startswith(("promo", "web_", "inventory_", "return"))
        or feature
        in {
            "orders_count",
            "unique_customers",
            "item_lines_count",
            "total_quantity",
            "total_discount_amount",
            "avg_discount_amount",
            "avg_discount_rate",
            "discount_to_gross_rate",
        }
        for feature in feature_names
    )

    return {
        "lag_dominant": lag_dominant,
        "business_features_present": business_features_present,
    }


def print_final_summary(
    metrics: dict[str, float],
    top_features: pd.DataFrame,
    reporter: RunReporter,
) -> None:
    """Print a short practical interpretation of the baseline run."""
    group_summary = summarize_feature_groups(top_features)

    reporter.emit("")
    reporter.emit("7. Short summary")
    reporter.emit(
        "Model performance: baseline is usable if errors are acceptable for business planning; "
        f"validation R2={metrics['R2']:.4f}, MAE={metrics['MAE']:,.2f}."
    )
    reporter.emit(
        "Lag features dominant: "
        + ("yes" if group_summary["lag_dominant"] else "no, other features dominate top importance.")
    )
    reporter.emit(
        "Business features in top 20: "
        + ("yes" if group_summary["business_features_present"] else "no.")
    )


def run_training() -> None:
    logger = setup_logging()
    reporter = RunReporter(logger)

    reporter.emit("Baseline LightGBM Training")
    reporter.emit("==========================")
    reporter.emit("")
    reporter.emit("1. Load data")

    df = load_data(DATASET_PATH)
    reporter.emit(f"Loaded dataset: {DATASET_PATH}")
    reporter.emit(f"Raw shape: {df.shape}")
    reporter.emit(f"Date range: {df[DATE_COL].min().date()} -> {df[DATE_COL].max().date()}")

    df = clean_data(df, reporter)
    X, y = split_features_target(df)
    reporter.emit(f"Feature matrix shape: {X.shape}")
    reporter.emit(f"Target vector length: {len(y):,}")

    X_train, X_valid, y_train, y_valid, valid_dates = time_based_split(df, X, y, reporter)
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

    reporter.emit("")
    reporter.emit("6. Feature importance")
    importance = save_feature_importance(model, X.columns.tolist(), FEATURE_IMPORTANCE_PATH)
    top20 = importance.head(20)
    reporter.emit_frame("Top 20 features by LightGBM gain importance:", top20)
    reporter.emit(f"Saved feature importance: {FEATURE_IMPORTANCE_PATH}")

    print_final_summary(metrics, top20, reporter)
    reporter.save_metrics(METRICS_PATH)


if __name__ == "__main__":
    run_training()
