from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import build_promo_2023_features as promo_builder
import train_final_model as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "log"

SALES_PATH = DATA_DIR / "sales.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
PROMOTIONS_PATH = DATA_DIR / "promotions.csv"
FUTURE_PROMO_FEATURES_PATH = DATA_DIR / "future_promo_calendar_features.csv"
INVENTORY_PATH = DATA_DIR / "inventory.csv"
SYNTHETIC_PROMOTIONS_PATH = DATA_DIR / "synthetic_promotions_2023_2024.csv"
CURRENT_BEST_VALIDATION_PATH = DATA_DIR / "final_micro_calibration_validation_predictions.csv"

SUBMISSION_8900_PATH = DATA_DIR / "submission_direct_seasonal_ratio_8900.csv"
SUBMISSION_8950_PATH = DATA_DIR / "submission_direct_seasonal_ratio_8950.csv"
SUBMISSION_9000_PATH = DATA_DIR / "submission_direct_seasonal_ratio_9000.csv"
VALIDATION_PREDICTIONS_PATH = DATA_DIR / "direct_seasonal_validation_predictions.csv"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "direct_seasonal_feature_importance.csv"
REPORT_PATH = LOG_DIR / "direct_seasonal_residual_report.txt"
LOG_FILE = LOG_DIR / "train_direct_seasonal_residual_model.log"

DATE_COL = base.DATE_COL
TARGET_COL = base.TARGET_COL
COGS_COL = base.COGS_COL
RANDOM_STATE = base.RANDOM_STATE

FOLDS = [
    ("fold_1", pd.Timestamp("2019-06-30"), pd.Timestamp("2019-07-01"), pd.Timestamp("2020-12-31")),
    ("fold_2", pd.Timestamp("2020-06-30"), pd.Timestamp("2020-07-01"), pd.Timestamp("2021-12-31")),
    ("fold_3", pd.Timestamp("2021-06-30"), pd.Timestamp("2021-07-01"), pd.Timestamp("2022-12-31")),
]

CAMPAIGN_NAME_MAP = {
    "spring sale": "is_spring_sale",
    "mid-year sale": "is_midyear_sale",
    "fall launch": "is_fall_launch",
    "year-end sale": "is_year_end_sale",
    "urban blowout": "is_urban_blowout",
    "rural special": "is_rural_special",
}
CAMPAIGN_FLAG_COLUMNS = list(CAMPAIGN_NAME_MAP.values())
INVENTORY_FEATURES = [
    "inv_stockout_rate",
    "inv_avg_fill_rate",
    "inv_avg_days_of_supply",
    "inv_avg_sell_through_rate",
    "inv_reorder_rate",
    "inv_overstock_rate",
]

MODEL_FEATURES = [
    "day_of_week",
    "day_of_year",
    "week_of_year",
    "month",
    "quarter",
    "is_month_start",
    "is_month_end",
    "is_weekend",
    "year",
    "is_odd_year",
    "calendar_any_promo",
    "calendar_active_promo_count",
    "calendar_avg_discount_value",
    "calendar_max_discount_value",
    "promotion_campaign_index",
    "promo_duration",
    "promo_progress_ratio",
    "promo_days_remaining",
    "baseline_revenue",
    "lag_365",
    "lag_730",
    "lag_1095",
    "weighted_recent_same_day_revenue",
    "same_month_recent_mean",
    "same_day_of_year_recent_mean",
    "same_campaign_last_year_revenue",
    "lag365_to_lag730_ratio",
    "lag365_to_recent_same_day_mean_ratio",
] + CAMPAIGN_FLAG_COLUMNS + INVENTORY_FEATURES


class Reporter:
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
        if getattr(frame, "empty", False):
            self.emit("(empty)")
            return
        self.emit(frame.to_string(index=False))

    def save(self, path: Path = REPORT_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        self.logger.info("Saved report to %s", path)


def setup_logging(log_file: Path = LOG_FILE) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_direct_seasonal_residual_model")
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


def safe_divide(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or abs(float(denominator)) < 1e-9:
        return np.nan
    return float(numerator) / float(denominator)


def safe_replace_year(date_value: pd.Timestamp, target_year: int) -> pd.Timestamp:
    try:
        return date_value.replace(year=target_year)
    except ValueError:
        return pd.Timestamp(year=target_year, month=2, day=28)


def normalize_campaign_name(value: Any) -> str:
    return str(value).strip().lower()


def validate_submission_frame(output: pd.DataFrame, sample_submission: pd.DataFrame) -> None:
    if list(output.columns) != [DATE_COL, TARGET_COL, COGS_COL]:
        raise ValueError("Submission columns must be exactly Date, Revenue, COGS")
    if len(output) != len(sample_submission):
        raise ValueError("Submission row count does not match sample submission")
    if not output[DATE_COL].equals(sample_submission[DATE_COL]):
        raise ValueError("Submission Date order does not match sample submission")
    if output[[TARGET_COL, COGS_COL]].isna().any().any():
        raise ValueError("Submission contains missing values")
    if (output[[TARGET_COL, COGS_COL]] < 0).any().any():
        raise ValueError("Submission contains negative Revenue or COGS")


def load_sales(path: Path = SALES_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"sales.csv not found: {path}")
    sales = pd.read_csv(path, parse_dates=[DATE_COL], low_memory=False)
    sales[DATE_COL] = pd.to_datetime(sales[DATE_COL], errors="coerce").dt.normalize()
    sales = sales.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    sales[TARGET_COL] = pd.to_numeric(sales[TARGET_COL], errors="coerce")
    sales[COGS_COL] = pd.to_numeric(sales[COGS_COL], errors="coerce")
    sales["year"] = sales[DATE_COL].dt.year.astype(int)
    sales["month"] = sales[DATE_COL].dt.month.astype(int)
    sales["day_of_year"] = sales[DATE_COL].dt.dayofyear.astype(int)
    return sales


def load_sample_submission(path: Path = SAMPLE_SUBMISSION_PATH) -> pd.DataFrame:
    sample = pd.read_csv(path, parse_dates=[DATE_COL], low_memory=False)
    sample[DATE_COL] = pd.to_datetime(sample[DATE_COL], errors="coerce").dt.normalize()
    return sample


def load_promotions_with_campaign_index(path: Path = PROMOTIONS_PATH) -> pd.DataFrame:
    promotions = promo_builder.load_promotions(path)
    promotions["duration_days"] = (promotions["end_date"] - promotions["start_date"]).dt.days + 1
    promotions["campaign_index"] = (
        promotions.sort_values(["source_year", "start_date", "end_date", "promo_id"])
        .groupby("source_year")
        .cumcount()
        .add(1)
    )
    return promotions


def load_or_build_synthetic_promotions(promotions: pd.DataFrame) -> pd.DataFrame:
    if SYNTHETIC_PROMOTIONS_PATH.exists():
        synthetic = pd.read_csv(SYNTHETIC_PROMOTIONS_PATH, low_memory=False)
        synthetic["start_date"] = pd.to_datetime(synthetic["start_date"], errors="coerce").dt.normalize()
        synthetic["end_date"] = pd.to_datetime(synthetic["end_date"], errors="coerce").dt.normalize()
        synthetic["duration_days"] = pd.to_numeric(synthetic["duration_days"], errors="coerce").fillna(
            (synthetic["end_date"] - synthetic["start_date"]).dt.days + 1
        )
        return synthetic
    synthetic, _ = promo_builder.build_synthetic_promotions(promotions)
    return synthetic


def build_daily_promo_context(dates: pd.Series, promotions: pd.DataFrame) -> pd.DataFrame:
    calendar = pd.DataFrame({DATE_COL: pd.to_datetime(dates).sort_values().unique()})
    context_columns = [
        "calendar_any_promo",
        "calendar_active_promo_count",
        "calendar_avg_discount_value",
        "calendar_max_discount_value",
        "promotion_campaign_index",
        "promo_duration",
        "promo_progress_ratio",
        "promo_days_remaining",
    ] + CAMPAIGN_FLAG_COLUMNS
    for column in context_columns:
        calendar[column] = 0.0

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

        campaign_flag_values = {feature: 0 for feature in CAMPAIGN_FLAG_COLUMNS}
        normalized_name = normalize_campaign_name(getattr(row, "promo_name_base", ""))
        flag_column = CAMPAIGN_NAME_MAP.get(normalized_name)
        if flag_column is not None:
            campaign_flag_values[flag_column] = 1

        duration_days = int(getattr(row, "duration_days", (row.end_date - row.start_date).days + 1))
        campaign_index = int(getattr(row, "campaign_index", 0))
        discount_value = float(getattr(row, "discount_value", 0.0))

        for active_date in pd.date_range(active_start, active_end, freq="D"):
            promo_day_number = (active_date - row.start_date).days + 1
            promo_days_remaining = (row.end_date - active_date).days
            rows.append(
                {
                    DATE_COL: active_date,
                    "promo_id": row.promo_id,
                    "calendar_avg_discount_value": discount_value,
                    "calendar_max_discount_value": discount_value,
                    "promotion_campaign_index": campaign_index,
                    "promo_duration": duration_days,
                    "promo_progress_ratio": promo_day_number / max(duration_days, 1),
                    "promo_days_remaining": promo_days_remaining,
                    **campaign_flag_values,
                }
            )

    if not rows:
        return calendar

    expanded = pd.DataFrame(rows)
    aggregations: dict[str, Any] = {
        "calendar_active_promo_count": ("promo_id", "nunique"),
        "calendar_avg_discount_value": ("calendar_avg_discount_value", "mean"),
        "calendar_max_discount_value": ("calendar_max_discount_value", "max"),
        "promotion_campaign_index": ("promotion_campaign_index", "max"),
        "promo_duration": ("promo_duration", "mean"),
        "promo_progress_ratio": ("promo_progress_ratio", "mean"),
        "promo_days_remaining": ("promo_days_remaining", "mean"),
    }
    for flag_column in CAMPAIGN_FLAG_COLUMNS:
        aggregations[flag_column] = (flag_column, "max")

    daily = expanded.groupby(DATE_COL, as_index=False).agg(**aggregations)
    daily["calendar_any_promo"] = (daily["calendar_active_promo_count"] > 0).astype(int)

    merged = calendar.drop(columns=context_columns).merge(daily, on=DATE_COL, how="left")
    for column in context_columns:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
    return merged[[DATE_COL] + context_columns]


def prepare_inventory_snapshots() -> pd.DataFrame:
    inventory = pd.read_csv(INVENTORY_PATH, low_memory=False)
    inventory["snapshot_date"] = pd.to_datetime(inventory["snapshot_date"], errors="coerce").dt.normalize()
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
            inventory[column] = 0.0
        inventory[column] = pd.to_numeric(inventory[column], errors="coerce").fillna(0.0)

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
        .reset_index(drop=True)
    )
    return snapshots


def build_inventory_context(
    dates: pd.Series,
    inventory_snapshots: pd.DataFrame,
    snapshot_cutoff: pd.Timestamp | None,
) -> pd.DataFrame:
    calendar = pd.DataFrame({DATE_COL: pd.to_datetime(dates).sort_values().unique()})

    snapshots = inventory_snapshots.copy()
    if snapshot_cutoff is not None:
        snapshots = snapshots[snapshots[DATE_COL] <= snapshot_cutoff].copy()

    if snapshots.empty:
        for feature in INVENTORY_FEATURES:
            calendar[feature] = 0.0
        return calendar

    merged = pd.merge_asof(
        calendar.sort_values(DATE_COL),
        snapshots.sort_values(DATE_COL),
        on=DATE_COL,
        direction="backward",
    )
    merged[INVENTORY_FEATURES] = merged[INVENTORY_FEATURES].fillna(0.0)
    return merged[[DATE_COL] + INVENTORY_FEATURES]


def build_static_context(
    dates: pd.Series,
    min_date: pd.Timestamp,
    promo_context: pd.DataFrame,
    inventory_context: pd.DataFrame,
) -> pd.DataFrame:
    calendar = base.build_calendar_features(dates, min_date)
    calendar["is_odd_year"] = (calendar["year"] % 2 == 1).astype(int)
    return (
        calendar.merge(promo_context, on=DATE_COL, how="left", validate="one_to_one")
        .merge(inventory_context, on=DATE_COL, how="left", validate="one_to_one")
        .fillna(0.0)
    )


def build_lookup_tables(sales: pd.DataFrame, historical_promo_context: pd.DataFrame) -> dict[str, Any]:
    sales_idx = sales.set_index(DATE_COL).sort_index()
    promo_idx = historical_promo_context.set_index(DATE_COL).sort_index()
    return {
        "revenue_map": pd.to_numeric(sales_idx[TARGET_COL], errors="coerce"),
        "sales_df": sales.copy(),
        "promo_idx": promo_idx,
    }


def get_exact_reference_revenue(
    revenue_map: pd.Series,
    target_date: pd.Timestamp,
    years_back: int,
    reference_end: pd.Timestamp,
) -> float:
    ref_date = safe_replace_year(target_date, target_date.year - years_back)
    if ref_date > reference_end:
        return np.nan
    if ref_date in revenue_map.index:
        return float(revenue_map.loc[ref_date])
    return np.nan


def compute_same_month_recent_mean(
    sales: pd.DataFrame,
    target_date: pd.Timestamp,
    reference_end: pd.Timestamp,
) -> float:
    candidate_years = {target_date.year - 1, target_date.year - 2, target_date.year - 3}
    mask = (
        (sales[DATE_COL] <= reference_end)
        & (sales["month"] == target_date.month)
        & (sales["year"].isin(candidate_years))
    )
    values = sales.loc[mask, TARGET_COL]
    return float(values.mean()) if not values.empty else np.nan


def compute_same_day_recent_mean(
    sales: pd.DataFrame,
    target_date: pd.Timestamp,
    reference_end: pd.Timestamp,
    tolerance_days: int = 3,
) -> float:
    candidate_years = {target_date.year - 1, target_date.year - 2, target_date.year - 3}
    target_doy = int(target_date.dayofyear)
    day_diff = np.abs(sales["day_of_year"] - target_doy)
    wrap_diff = np.minimum(day_diff, 366 - day_diff)
    mask = (
        (sales[DATE_COL] <= reference_end)
        & (sales["year"].isin(candidate_years))
        & (wrap_diff <= tolerance_days)
    )
    values = sales.loc[mask, TARGET_COL]
    return float(values.mean()) if not values.empty else np.nan


def compute_same_campaign_last_year_revenue(
    revenue_map: pd.Series,
    promo_idx: pd.DataFrame,
    target_date: pd.Timestamp,
    reference_end: pd.Timestamp,
    target_campaign_flags: dict[str, float],
) -> float:
    if not any(target_campaign_flags.get(flag, 0) > 0 for flag in CAMPAIGN_FLAG_COLUMNS):
        return np.nan
    ref_date = safe_replace_year(target_date, target_date.year - 1)
    if ref_date > reference_end or ref_date not in revenue_map.index or ref_date not in promo_idx.index:
        return np.nan
    ref_row = promo_idx.loc[ref_date]
    if isinstance(ref_row, pd.DataFrame):
        ref_row = ref_row.iloc[0]
    overlap = any(
        target_campaign_flags.get(flag, 0) > 0 and pd.to_numeric(ref_row.get(flag, 0), errors="coerce") > 0
        for flag in CAMPAIGN_FLAG_COLUMNS
    )
    if overlap:
        return float(revenue_map.loc[ref_date])
    return np.nan


def robust_weighted_average(values: list[tuple[float, float]]) -> float:
    clean = [(weight, value) for weight, value in values if pd.notna(value) and np.isfinite(value)]
    if not clean:
        return np.nan
    total_weight = sum(weight for weight, _ in clean)
    if total_weight <= 0:
        return np.nan
    return float(sum(weight * value for weight, value in clean) / total_weight)


def build_reference_features(
    target_dates: pd.Series,
    reference_ends: pd.Series,
    static_context: pd.DataFrame,
    lookup_tables: dict[str, Any],
) -> pd.DataFrame:
    sales = lookup_tables["sales_df"]
    revenue_map = lookup_tables["revenue_map"]
    promo_idx = lookup_tables["promo_idx"]
    static_idx = static_context.set_index(DATE_COL).sort_index()

    rows: list[dict[str, Any]] = []
    for target_date, reference_end in zip(pd.to_datetime(target_dates), pd.to_datetime(reference_ends)):
        static_row = static_idx.loc[target_date].to_dict()
        campaign_flags = {flag: static_row.get(flag, 0.0) for flag in CAMPAIGN_FLAG_COLUMNS}

        lag_365 = get_exact_reference_revenue(revenue_map, target_date, 1, reference_end)
        lag_730 = get_exact_reference_revenue(revenue_map, target_date, 2, reference_end)
        lag_1095 = get_exact_reference_revenue(revenue_map, target_date, 3, reference_end)
        weighted_recent = robust_weighted_average([(0.5, lag_365), (0.3, lag_730), (0.2, lag_1095)])
        same_month_mean = compute_same_month_recent_mean(sales, target_date, reference_end)
        same_day_mean = compute_same_day_recent_mean(sales, target_date, reference_end)
        same_campaign_last_year = compute_same_campaign_last_year_revenue(
            revenue_map,
            promo_idx,
            target_date,
            reference_end,
            campaign_flags,
        )

        baseline_revenue = robust_weighted_average(
            [
                (0.50, weighted_recent),
                (0.20, same_day_mean),
                (0.15, same_month_mean),
                (0.10, lag_365),
                (0.05, same_campaign_last_year),
            ]
        )

        rows.append(
            {
                DATE_COL: target_date,
                "revenue_same_day_last_year": lag_365,
                "revenue_same_day_2y_ago": lag_730,
                "revenue_same_day_3y_ago": lag_1095,
                "lag_365": lag_365,
                "lag_730": lag_730,
                "lag_1095": lag_1095,
                "weighted_recent_same_day_revenue": weighted_recent,
                "same_month_recent_mean": same_month_mean,
                "same_day_of_year_recent_mean": same_day_mean,
                "same_campaign_last_year_revenue": same_campaign_last_year,
                "baseline_revenue": baseline_revenue,
                "lag365_to_lag730_ratio": safe_divide(lag_365, lag_730),
                "lag365_to_recent_same_day_mean_ratio": safe_divide(lag_365, same_day_mean),
            }
        )

    return pd.DataFrame(rows)


def build_direct_model_table(
    dates: pd.Series,
    reference_ends: pd.Series,
    static_context: pd.DataFrame,
    lookup_tables: dict[str, Any],
    sales: pd.DataFrame,
) -> pd.DataFrame:
    references = build_reference_features(dates, reference_ends, static_context, lookup_tables)
    actuals = sales[[DATE_COL, TARGET_COL]].copy()
    table = (
        static_context.merge(references, on=DATE_COL, how="left", validate="one_to_one")
        .merge(actuals, on=DATE_COL, how="left")
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )
    table["target_ratio"] = safe_divide_series(table[TARGET_COL], table["baseline_revenue"])
    clipped_ratio = table["target_ratio"].clip(lower=1e-6)
    table["target_log_ratio"] = np.log(clipped_ratio)
    return table


def safe_divide_series(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    output = num / den.replace(0, np.nan)
    return output.replace([np.inf, -np.inf], np.nan)


def make_training_matrix(
    model_table: pd.DataFrame,
    target_column: str,
    train_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    train = model_table[model_table[DATE_COL] <= train_end].copy()
    train = train.dropna(subset=["baseline_revenue", target_column]).copy()
    train = train[train["baseline_revenue"] > 0].copy()
    X_train = train[MODEL_FEATURES].apply(pd.to_numeric, errors="coerce")
    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)
    y_train = pd.to_numeric(train[target_column], errors="coerce")
    return X_train, y_train, medians


def make_prediction_matrix(
    model_table: pd.DataFrame,
    medians: pd.Series,
) -> pd.DataFrame:
    X = model_table[MODEL_FEATURES].apply(pd.to_numeric, errors="coerce")
    return X.fillna(medians)


def lightgbm_available() -> bool:
    return base.lightgbm_available()


def train_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    reporter: Reporter,
) -> tuple[Any, str]:
    if lightgbm_available():
        import lightgbm as lgb

        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.03,
            "max_depth": 6,
            "num_leaves": 24,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "min_data_in_leaf": 30,
            "seed": RANDOM_STATE,
            "verbosity": -1,
            "force_col_wise": True,
        }
        dataset = lgb.Dataset(X_train, label=y_train, feature_name=X_train.columns.tolist(), free_raw_data=False)
        model = lgb.train(params=params, train_set=dataset, num_boost_round=600)
        reporter.logger.info("Trained LightGBM direct model on %s rows", len(X_train))
        return model, "lightgbm"

    try:
        from sklearn.ensemble import GradientBoostingRegressor
    except Exception as exc:
        raise ImportError("LightGBM unavailable and sklearn fallback not installed") from exc

    model = GradientBoostingRegressor(
        learning_rate=0.03,
        n_estimators=400,
        max_depth=3,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model, "gradient_boosting"


def predict_regressor(model: Any, model_type: str, X: pd.DataFrame) -> np.ndarray:
    if model_type == "lightgbm":
        return np.asarray(model.predict(X), dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def extract_feature_importance(model: Any, model_type: str, feature_columns: list[str]) -> pd.DataFrame:
    if model_type == "lightgbm":
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
    if hasattr(model, "feature_importances_"):
        return (
            pd.DataFrame(
                {
                    "feature": feature_columns,
                    "importance_split": np.nan,
                    "importance_gain": model.feature_importances_,
                }
            )
            .sort_values("importance_gain", ascending=False)
            .reset_index(drop=True)
        )
    return pd.DataFrame({"feature": feature_columns, "importance_split": np.nan, "importance_gain": np.nan})


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    metrics = base.evaluate_predictions(y_true, y_pred)
    actual = y_true.to_numpy(dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    error = actual - predicted
    top10_threshold = float(np.quantile(actual, 0.90))
    top10_mask = actual >= top10_threshold
    non_spike_mask = actual < top10_threshold
    metrics["top10_RMSE"] = float(np.sqrt(np.mean(error[top10_mask] ** 2))) if top10_mask.any() else np.nan
    metrics["top10_underprediction"] = int(np.sum(error[top10_mask] > 0)) if top10_mask.any() else 0
    metrics["non_spike_RMSE"] = float(np.sqrt(np.mean(error[non_spike_mask] ** 2))) if non_spike_mask.any() else np.nan
    return metrics


def compute_monthly_rmse(frame: pd.DataFrame) -> pd.DataFrame:
    temp = frame.copy()
    temp["year_month"] = temp[DATE_COL].dt.to_period("M").astype(str)
    temp["sq_error"] = (temp["actual_Revenue"] - temp["predicted_Revenue"]) ** 2
    monthly = temp.groupby("year_month", as_index=False)["sq_error"].mean()
    monthly["RMSE"] = np.sqrt(monthly["sq_error"])
    return monthly[["year_month", "RMSE"]]


def clip_ratio_predictions(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.40, 2.50)


def prepare_historical_contexts(
    sales: pd.DataFrame,
    promotions: pd.DataFrame,
    inventory_snapshots: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    historical_promo_context = build_daily_promo_context(sales[DATE_COL], promotions)
    historical_inventory_context = build_inventory_context(sales[DATE_COL], inventory_snapshots, snapshot_cutoff=None)
    static_context = build_static_context(
        dates=sales[DATE_COL],
        min_date=sales[DATE_COL].min(),
        promo_context=historical_promo_context,
        inventory_context=historical_inventory_context,
    )
    lookup_tables = build_lookup_tables(sales, historical_promo_context)
    return historical_promo_context, static_context, lookup_tables


def prepare_future_contexts(
    sample_submission: pd.DataFrame,
    promotions: pd.DataFrame,
    synthetic_promotions: pd.DataFrame,
    inventory_snapshots: pd.DataFrame,
    train_end: pd.Timestamp,
) -> pd.DataFrame:
    if FUTURE_PROMO_FEATURES_PATH.exists():
        future_known = pd.read_csv(FUTURE_PROMO_FEATURES_PATH, parse_dates=[DATE_COL], low_memory=False)
        future_known[DATE_COL] = pd.to_datetime(future_known[DATE_COL], errors="coerce").dt.normalize()
        rename_map = {
            "future_calendar_any_promo": "calendar_any_promo",
            "future_calendar_active_promo_count": "calendar_active_promo_count",
            "future_calendar_avg_discount_value": "calendar_avg_discount_value",
            "future_calendar_max_discount_value": "calendar_max_discount_value",
            "future_promo_avg_duration_days": "promo_duration",
            "future_promo_avg_progress_ratio": "promo_progress_ratio",
            "future_promo_avg_days_remaining": "promo_days_remaining",
            "future_promotion_campaign_index": "promotion_campaign_index",
        }
        future_known = future_known.rename(columns=rename_map)
        keep_columns = [DATE_COL] + list(rename_map.values())
        future_known = future_known[keep_columns]
    else:
        future_known = build_daily_promo_context(sample_submission[DATE_COL], synthetic_promotions)[
            [
                DATE_COL,
                "calendar_any_promo",
                "calendar_active_promo_count",
                "calendar_avg_discount_value",
                "calendar_max_discount_value",
                "promotion_campaign_index",
                "promo_duration",
                "promo_progress_ratio",
                "promo_days_remaining",
            ]
        ]

    future_campaign_context = build_daily_promo_context(sample_submission[DATE_COL], synthetic_promotions)[
        [DATE_COL] + CAMPAIGN_FLAG_COLUMNS
    ]
    inventory_context = build_inventory_context(sample_submission[DATE_COL], inventory_snapshots, snapshot_cutoff=train_end)
    calendar = base.build_calendar_features(sample_submission[DATE_COL], base.load_train_data(base.TRAIN_DATA_PATH)[DATE_COL].min())
    calendar["is_odd_year"] = (calendar["year"] % 2 == 1).astype(int)
    future_static = (
        calendar.merge(future_known, on=DATE_COL, how="left", validate="one_to_one")
        .merge(future_campaign_context, on=DATE_COL, how="left", validate="one_to_one")
        .merge(inventory_context, on=DATE_COL, how="left", validate="one_to_one")
        .fillna(0.0)
    )
    return future_static


def fit_and_predict_fold(
    fold_name: str,
    train_end: pd.Timestamp,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    sales: pd.DataFrame,
    historical_promo_context: pd.DataFrame,
    lookup_tables: dict[str, Any],
    inventory_snapshots: pd.DataFrame,
    reporter: Reporter,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation_dates = sales.loc[
        (sales[DATE_COL] >= validation_start) & (sales[DATE_COL] <= validation_end),
        DATE_COL,
    ]
    if validation_dates.empty:
        return pd.DataFrame(), pd.DataFrame()

    historical_static_context = build_static_context(
        dates=sales[DATE_COL],
        min_date=sales[DATE_COL].min(),
        promo_context=historical_promo_context,
        inventory_context=build_inventory_context(sales[DATE_COL], inventory_snapshots, snapshot_cutoff=None),
    )

    training_reference_ends = sales[DATE_COL] - pd.Timedelta(days=1)
    full_training_table = build_direct_model_table(
        dates=sales[DATE_COL],
        reference_ends=training_reference_ends,
        static_context=historical_static_context,
        lookup_tables=lookup_tables,
        sales=sales,
    )

    validation_promo_context = historical_promo_context[
        historical_promo_context[DATE_COL].isin(validation_dates)
    ].copy()
    validation_inventory_context = build_inventory_context(validation_dates, inventory_snapshots, snapshot_cutoff=train_end)
    validation_static = build_static_context(
        dates=validation_dates,
        min_date=sales[DATE_COL].min(),
        promo_context=validation_promo_context,
        inventory_context=validation_inventory_context,
    )
    validation_table = build_direct_model_table(
        dates=validation_dates,
        reference_ends=pd.Series(np.repeat(train_end, len(validation_dates))),
        static_context=validation_static,
        lookup_tables=lookup_tables,
        sales=sales,
    )
    validation_table = validation_table.merge(
        sales[[DATE_COL, TARGET_COL]],
        on=DATE_COL,
        how="left",
        suffixes=("", "_actual"),
    )
    if f"{TARGET_COL}_actual" in validation_table.columns:
        validation_table[TARGET_COL] = pd.to_numeric(
            validation_table[f"{TARGET_COL}_actual"],
            errors="coerce",
        ).fillna(pd.to_numeric(validation_table[TARGET_COL], errors="coerce"))
        validation_table = validation_table.drop(columns=[f"{TARGET_COL}_actual"])

    outputs: list[pd.DataFrame] = []
    fold_metrics_rows: list[dict[str, Any]] = []

    for target_type, target_column in [("ratio", "target_ratio"), ("log_ratio", "target_log_ratio")]:
        X_train, y_train, medians = make_training_matrix(full_training_table, target_column, train_end=train_end)
        if X_train.empty:
            continue
        model, model_type = train_regressor(X_train, y_train, reporter)
        X_valid = make_prediction_matrix(validation_table, medians)
        raw_pred = predict_regressor(model, model_type, X_valid)

        if target_type == "log_ratio":
            predicted_ratio = np.exp(raw_pred)
        else:
            predicted_ratio = raw_pred
        predicted_ratio = clip_ratio_predictions(predicted_ratio)
        predicted_revenue = validation_table["baseline_revenue"].to_numpy(dtype=float) * predicted_ratio
        predicted_revenue = np.clip(predicted_revenue, 0.0, None)

        metrics = compute_metrics(validation_table[TARGET_COL], predicted_revenue)
        metrics_row = {
            "fold": fold_name,
            "target_type": target_type,
            "train_end": train_end.date(),
            "validation_start": validation_start.date(),
            "validation_end": validation_end.date(),
            **metrics,
        }
        fold_metrics_rows.append(metrics_row)

        output = pd.DataFrame(
            {
                "fold": fold_name,
                "target_type": target_type,
                DATE_COL: validation_table[DATE_COL],
                "actual_Revenue": validation_table[TARGET_COL],
                "baseline_revenue": validation_table["baseline_revenue"],
                "predicted_ratio": predicted_ratio,
                "predicted_Revenue": predicted_revenue,
            }
        )
        outputs.append(output)

    return pd.concat(outputs, ignore_index=True), pd.DataFrame(fold_metrics_rows)


def compare_with_current_best(validation_predictions: pd.DataFrame) -> dict[str, float] | None:
    if not CURRENT_BEST_VALIDATION_PATH.exists():
        return None
    current_best = pd.read_csv(CURRENT_BEST_VALIDATION_PATH, parse_dates=[DATE_COL], low_memory=False)
    current_best[DATE_COL] = pd.to_datetime(current_best[DATE_COL], errors="coerce").dt.normalize()
    if "current_base_pred" not in current_best.columns:
        return None
    current_best = current_best[[DATE_COL, "actual_Revenue", "current_base_pred"]].copy()
    return compute_metrics(current_best["actual_Revenue"], current_best["current_base_pred"].to_numpy(dtype=float))


def train_final_model_table(
    sales: pd.DataFrame,
    historical_promo_context: pd.DataFrame,
    lookup_tables: dict[str, Any],
    inventory_snapshots: pd.DataFrame,
) -> pd.DataFrame:
    training_reference_ends = sales[DATE_COL] - pd.Timedelta(days=1)
    static_context = build_static_context(
        dates=sales[DATE_COL],
        min_date=sales[DATE_COL].min(),
        promo_context=historical_promo_context,
        inventory_context=build_inventory_context(sales[DATE_COL], inventory_snapshots, snapshot_cutoff=None),
    )
    return build_direct_model_table(
        dates=sales[DATE_COL],
        reference_ends=training_reference_ends,
        static_context=static_context,
        lookup_tables=lookup_tables,
        sales=sales,
    )


def save_submission(
    path: Path,
    sample_submission: pd.DataFrame,
    revenue_pred: np.ndarray,
    cogs_ratio: float,
) -> None:
    output = sample_submission[[DATE_COL]].copy()
    output[TARGET_COL] = np.clip(np.asarray(revenue_pred, dtype=float), 0.0, None)
    output[COGS_COL] = output[TARGET_COL] * cogs_ratio
    validate_submission_frame(output, sample_submission)
    output.to_csv(path, index=False)


def run() -> None:
    logger = setup_logging()
    reporter = Reporter(logger)

    reporter.emit("Direct Seasonal Residual Forecasting")
    reporter.emit("===================================")
    reporter.emit("")

    sales = load_sales(SALES_PATH)
    sample_submission = load_sample_submission(SAMPLE_SUBMISSION_PATH)
    promotions = load_promotions_with_campaign_index(PROMOTIONS_PATH)
    synthetic_promotions = load_or_build_synthetic_promotions(promotions)
    inventory_snapshots = prepare_inventory_snapshots()

    reporter.emit(
        f"Historical sales range: {sales[DATE_COL].min().date()} -> {sales[DATE_COL].max().date()} "
        f"({len(sales):,} rows)"
    )
    reporter.emit(
        f"Forecast range: {sample_submission[DATE_COL].min().date()} -> {sample_submission[DATE_COL].max().date()} "
        f"({len(sample_submission):,} rows)"
    )

    historical_promo_context, _, lookup_tables = prepare_historical_contexts(
        sales=sales,
        promotions=promotions,
        inventory_snapshots=inventory_snapshots,
    )

    reporter.emit("")
    reporter.emit("1. Run long-horizon direct folds")
    fold_prediction_frames: list[pd.DataFrame] = []
    fold_metrics_frames: list[pd.DataFrame] = []
    for fold_name, train_end, validation_start, validation_end in FOLDS:
        reporter.emit(
            f"{fold_name}: train <= {train_end.date()}, validate {validation_start.date()} -> {validation_end.date()}"
        )
        fold_predictions, fold_metrics = fit_and_predict_fold(
            fold_name=fold_name,
            train_end=train_end,
            validation_start=validation_start,
            validation_end=validation_end,
            sales=sales,
            historical_promo_context=historical_promo_context,
            lookup_tables=lookup_tables,
            inventory_snapshots=inventory_snapshots,
            reporter=reporter,
        )
        fold_prediction_frames.append(fold_predictions)
        fold_metrics_frames.append(fold_metrics)

    validation_predictions = pd.concat(fold_prediction_frames, ignore_index=True)
    fold_metrics = pd.concat(fold_metrics_frames, ignore_index=True)
    validation_predictions.to_csv(VALIDATION_PREDICTIONS_PATH, index=False)
    reporter.emit_frame("Fold metrics:", fold_metrics)

    avg_metrics = (
        fold_metrics.groupby("target_type", as_index=False)
        .agg(
            avg_MAE=("MAE", "mean"),
            avg_RMSE=("RMSE", "mean"),
            avg_R2=("R2", "mean"),
            avg_top10_RMSE=("top10_RMSE", "mean"),
            avg_non_spike_RMSE=("non_spike_RMSE", "mean"),
        )
        .sort_values("avg_RMSE")
        .reset_index(drop=True)
    )
    reporter.emit_frame("Average fold metrics by target type:", avg_metrics)
    best_target_type = str(avg_metrics.iloc[0]["target_type"])
    best_target_column = "target_log_ratio" if best_target_type == "log_ratio" else "target_ratio"

    best_validation_subset = validation_predictions[validation_predictions["target_type"] == best_target_type].copy()
    monthly_rmse = compute_monthly_rmse(best_validation_subset)
    reporter.emit_frame("Monthly RMSE for best target type:", monthly_rmse.head(24))

    reference_metrics = compare_with_current_best(best_validation_subset)
    if reference_metrics is not None:
        reporter.emit("")
        reporter.emit(
            "2022 current-best validation analog: "
            f"MAE={reference_metrics['MAE']:,.2f}, RMSE={reference_metrics['RMSE']:,.2f}, "
            f"R2={reference_metrics['R2']:.6f}"
        )

    reporter.emit("")
    reporter.emit("2. Train final direct seasonal model on full history")
    final_training_table = train_final_model_table(
        sales=sales,
        historical_promo_context=historical_promo_context,
        lookup_tables=lookup_tables,
        inventory_snapshots=inventory_snapshots,
    )
    X_final, y_final, final_medians = make_training_matrix(
        final_training_table,
        target_column=best_target_column,
        train_end=sales[DATE_COL].max(),
    )
    final_model, final_model_type = train_regressor(X_final, y_final, reporter)

    future_static = prepare_future_contexts(
        sample_submission=sample_submission,
        promotions=promotions,
        synthetic_promotions=synthetic_promotions,
        inventory_snapshots=inventory_snapshots,
        train_end=sales[DATE_COL].max(),
    )
    future_table = build_direct_model_table(
        dates=sample_submission[DATE_COL],
        reference_ends=pd.Series(np.repeat(sales[DATE_COL].max(), len(sample_submission))),
        static_context=future_static,
        lookup_tables=lookup_tables,
        sales=sales,
    )
    X_future = make_prediction_matrix(future_table, final_medians)
    raw_future = predict_regressor(final_model, final_model_type, X_future)
    if best_target_type == "log_ratio":
        future_ratio = np.exp(raw_future)
    else:
        future_ratio = raw_future
    future_ratio = clip_ratio_predictions(future_ratio)
    future_revenue = np.clip(future_table["baseline_revenue"].to_numpy(dtype=float) * future_ratio, 0.0, None)

    importance = extract_feature_importance(final_model, final_model_type, MODEL_FEATURES)
    importance["best_target_type"] = best_target_type
    importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    reporter.emit_frame(
        "Top 30 features:",
        importance.head(30)[["feature", "importance_gain", "importance_split"]],
    )

    reporter.emit("")
    reporter.emit("3. Save direct seasonal submissions")
    save_submission(SUBMISSION_8900_PATH, sample_submission, future_revenue, cogs_ratio=0.8900)
    save_submission(SUBMISSION_8950_PATH, sample_submission, future_revenue, cogs_ratio=0.8950)
    save_submission(SUBMISSION_9000_PATH, sample_submission, future_revenue, cogs_ratio=0.9000)

    reporter.emit("")
    reporter.emit(f"Best target type: {best_target_type}")
    reporter.emit(
        f"Average RMSE for best target type: {avg_metrics.iloc[0]['avg_RMSE']:,.2f}"
    )
    reporter.emit(
        "Created submission files: "
        "submission_direct_seasonal_ratio_8900.csv, "
        "submission_direct_seasonal_ratio_8950.csv, "
        "submission_direct_seasonal_ratio_9000.csv"
    )
    reporter.save(REPORT_PATH)


if __name__ == "__main__":
    run()
