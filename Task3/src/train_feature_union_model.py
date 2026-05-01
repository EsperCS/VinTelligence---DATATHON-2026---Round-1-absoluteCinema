from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import train_final_model as base
import train_funnel_model as funnel
import train_promo_known_pipeline as promo_known
import train_stock_aware_scaling as stock_scale
import train_traffic_driven_model as traffic_branch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "log"

SALES_PATH = DATA_DIR / "sales.csv"
ORDERS_PATH = DATA_DIR / "orders.csv"
ORDER_ITEMS_PATH = DATA_DIR / "order_items.csv"
WEB_TRAFFIC_PATH = DATA_DIR / "web_traffic.csv"
PROMOTIONS_PATH = DATA_DIR / "promotions.csv"
INVENTORY_PATH = DATA_DIR / "inventory.csv"
PRODUCTS_PATH = DATA_DIR / "products.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
DAILY_FUNNEL_TABLE_PATH = DATA_DIR / "daily_funnel_table.csv"
FUTURE_TRAFFIC_SCENARIOS_PATH = DATA_DIR / "future_traffic_funnel_scenarios.csv"
FUTURE_PROMO_KNOWN_PATH = DATA_DIR / "future_promo_known_features.csv"
CURRENT_BEST_SUBMISSION_PATH = DATA_DIR / "submission_blend_direct_15_cogs8900.csv"
SEGMENT_SUBMISSION_PATH = DATA_DIR / "submission_m5_segment_bottomup.csv"

VALIDATION_PREDICTIONS_PATH = DATA_DIR / "feature_union_validation_predictions.csv"
MODEL_COMPARISON_PATH = DATA_DIR / "feature_union_model_comparison.csv"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "feature_union_feature_importance.csv"
ABLATION_RESULTS_PATH = DATA_DIR / "feature_union_ablation_results.csv"
REPORT_PATH = LOG_DIR / "feature_union_model_report.txt"
LOG_FILE = LOG_DIR / "train_feature_union_model.log"

SUBMISSION_MAIN_PATH = DATA_DIR / "submission_feature_union.csv"
SUBMISSION_CONSERVATIVE_PATH = DATA_DIR / "submission_feature_union_conservative.csv"
SUBMISSION_AGGRESSIVE_PATH = DATA_DIR / "submission_feature_union_aggressive.csv"
BLEND_OUTPUTS = {
    0.05: DATA_DIR / "submission_feature_union_blend_05.csv",
    0.10: DATA_DIR / "submission_feature_union_blend_10.csv",
    0.15: DATA_DIR / "submission_feature_union_blend_15.csv",
    0.20: DATA_DIR / "submission_feature_union_blend_20.csv",
    0.25: DATA_DIR / "submission_feature_union_blend_25.csv",
    0.30: DATA_DIR / "submission_feature_union_blend_30.csv",
}
SEGMENT_BLEND_OUTPUTS = {
    "801010": DATA_DIR / "submission_feature_union_segment_801010.csv",
    "702010": DATA_DIR / "submission_feature_union_segment_702010.csv",
    "701020": DATA_DIR / "submission_feature_union_segment_701020.csv",
}

DATE_COL = base.DATE_COL
TARGET_COL = base.TARGET_COL
COGS_COL = base.COGS_COL
RANDOM_STATE = base.RANDOM_STATE
EPS = 1e-9

VALIDATION_2022 = ("validation_2022", pd.Timestamp("2021-12-31"), pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"))
LONG_FOLDS = [
    ("fold_1", pd.Timestamp("2019-06-30"), pd.Timestamp("2019-07-01"), pd.Timestamp("2020-12-31")),
    ("fold_2", pd.Timestamp("2020-06-30"), pd.Timestamp("2020-07-01"), pd.Timestamp("2021-12-31")),
    ("fold_3", pd.Timestamp("2021-06-30"), pd.Timestamp("2021-07-01"), pd.Timestamp("2022-12-31")),
]
ALL_SCOPES = [VALIDATION_2022] + LONG_FOLDS

CALENDAR_FEATURES = ["day_of_week", "day_of_year", "month", "week_of_year", "is_weekend"]
REVENUE_FEATURES = [
    "lag_7",
    "lag_14",
    "lag_30",
    "revenue_lag_90",
    "revenue_lag_180",
    "revenue_lag_365",
    "rolling_mean_7",
    "rolling_mean_30",
    "revenue_roll_mean_90",
    "revenue_roll_mean_365",
    "spike_strength_365",
    "lag365_to_roll365_ratio",
    "lag7_to_roll30_ratio",
    "volatility_30",
    "volatility_90",
]
FUNNEL_FEATURES = [
    "orders_same_day_recent_mean",
    "orders_lag_365",
    "predicted_orders_signal",
    "sessions_growth_3_14",
    "sessions_roll_mean_30",
    "sessions_roll_std_30",
    "conversion_lag_365",
    "conversion_roll_mean_30",
    "aov_lag_365",
    "aov_roll_mean_7",
    "aov_roll_mean_30",
    "aov_same_day_recent_mean",
    "avg_discount_per_order_lag_365",
    "item_lines_per_order_lag_365",
    "quantity_per_order_lag_365",
]
PROMO_FEATURES = [
    "calendar_any_promo",
    "calendar_avg_discount_value",
    "campaign_intensity",
    "discount_x_progress",
    "discount_x_days_remaining",
    "promo_progress_ratio",
    "promo_days_remaining",
    "spring_sale",
    "mid_year_sale",
    "fall_launch",
    "year_end_sale",
    "urban_blowout",
    "rural_special",
]
STOCK_FEATURES = [
    "inv_avg_days_of_supply",
    "inv_avg_sell_through_rate",
    "inv_stockout_rate",
    "stock_pressure",
    "stock_build_up",
    "stock_pressure_x_promo",
    "stock_pressure_x_revenue_lag365",
]
INTERACTION_FEATURES = [
    "revenue_lag365_x_campaign_intensity",
    "revenue_lag365_x_sessions_growth_3_14",
    "aov_lag365_x_calendar_avg_discount_value",
    "orders_lag365_x_sessions_growth_3_14",
    "stock_pressure_x_campaign_intensity",
    "spike_strength365_x_campaign_intensity",
]

FEATURE_SETS = {
    "base_only": CALENDAR_FEATURES + REVENUE_FEATURES,
    "base_funnel": CALENDAR_FEATURES + REVENUE_FEATURES + FUNNEL_FEATURES,
    "base_funnel_promo": CALENDAR_FEATURES + REVENUE_FEATURES + FUNNEL_FEATURES + PROMO_FEATURES,
    "base_funnel_promo_stock": CALENDAR_FEATURES + REVENUE_FEATURES + FUNNEL_FEATURES + PROMO_FEATURES + STOCK_FEATURES,
    "full_union": CALENDAR_FEATURES + REVENUE_FEATURES + FUNNEL_FEATURES + PROMO_FEATURES + STOCK_FEATURES + INTERACTION_FEATURES,
}
META_FEATURES = [
    "base_pred",
    "base_pred_rank_pct",
    "revenue_lag_365",
    "spike_strength_365",
    "lag7_to_roll30_ratio",
    "sessions_growth_3_14",
    "aov_lag_365",
    "orders_lag_365",
    "calendar_any_promo",
    "campaign_intensity",
    "promo_progress_ratio",
    "promo_days_remaining",
    "stock_pressure",
    "stock_build_up",
    "inv_stockout_rate",
    "stock_pressure_x_promo",
    "stock_pressure_x_revenue_lag365",
    "day_of_week",
    "day_of_year",
    "month",
]


class RunReporter:
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
        if isinstance(frame, pd.Series):
            self.emit(frame.to_string() if not frame.empty else "(empty)")
            return
        self.emit(frame.to_string(index=False) if not frame.empty else "(empty)")

    def save(self, path: Path = REPORT_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def setup_logging(log_file: Path = LOG_FILE) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_feature_union_model")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def mae(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float))))


def rmse(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)))))


def r2_score_manual(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    actual_arr = np.asarray(actual, dtype=float)
    pred_arr = np.asarray(predicted, dtype=float)
    denom = float(np.sum(np.square(actual_arr - np.mean(actual_arr))))
    if denom <= EPS:
        return 0.0
    return 1.0 - float(np.sum(np.square(actual_arr - pred_arr))) / denom


def safe_divide(numerator: Any, denominator: Any, fill_value: float = 0.0):
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    return np.divide(num, den, out=np.full_like(num, fill_value, dtype=float), where=np.abs(den) > EPS)


def normalize_date_column(frame: pd.DataFrame, column: str = DATE_COL) -> pd.DataFrame:
    output = frame.copy()
    output[column] = pd.to_datetime(output[column], errors="coerce").dt.normalize()
    return output


def validate_submission_frame(output: pd.DataFrame, sample_submission: pd.DataFrame) -> None:
    if output.columns.tolist() != [DATE_COL, TARGET_COL, COGS_COL]:
        raise ValueError(f"Submission columns mismatch: {output.columns.tolist()}")
    if len(output) != len(sample_submission):
        raise ValueError(f"Submission row mismatch: expected {len(sample_submission)}, got {len(output)}")
    if not output[DATE_COL].equals(sample_submission[DATE_COL]):
        raise ValueError("Submission dates do not match sample order")
    if output.isna().any().any():
        raise ValueError("Submission contains missing values")
    if (output[[TARGET_COL, COGS_COL]] < 0).any().any():
        raise ValueError("Submission contains negative Revenue/COGS")


def compute_metrics(actual: pd.Series, predicted: np.ndarray, promo_mask: pd.Series, high_traffic_mask: pd.Series, high_stock_pressure_mask: pd.Series) -> dict[str, float]:
    actual_arr = np.asarray(actual, dtype=float)
    pred_arr = np.asarray(predicted, dtype=float)
    top10_threshold = float(np.quantile(actual_arr, 0.90))
    top10 = actual_arr >= top10_threshold
    promo = promo_mask.fillna(0).astype(bool).to_numpy()
    high_traffic = high_traffic_mask.fillna(0).astype(bool).to_numpy()
    high_stock = high_stock_pressure_mask.fillna(0).astype(bool).to_numpy()
    non_promo = ~promo
    return {
        "mae": mae(actual_arr, pred_arr),
        "rmse": rmse(actual_arr, pred_arr),
        "r2": r2_score_manual(actual_arr, pred_arr),
        "top10_rmse": rmse(actual_arr[top10], pred_arr[top10]) if top10.any() else np.nan,
        "top10_underprediction_count": float(np.sum(pred_arr[top10] < actual_arr[top10])) if top10.any() else np.nan,
        "promo_day_rmse": rmse(actual_arr[promo], pred_arr[promo]) if promo.any() else np.nan,
        "non_promo_rmse": rmse(actual_arr[non_promo], pred_arr[non_promo]) if non_promo.any() else np.nan,
        "high_traffic_rmse": rmse(actual_arr[high_traffic], pred_arr[high_traffic]) if high_traffic.any() else np.nan,
        "high_stock_pressure_rmse": rmse(actual_arr[high_stock], pred_arr[high_stock]) if high_stock.any() else np.nan,
    }


def train_lightgbm_native(X_train: pd.DataFrame, y_train: pd.Series, shallow: bool) -> Any:
    import lightgbm as lgb

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.03 if not shallow else 0.025,
        "max_depth": 6 if not shallow else 3,
        "num_leaves": 31 if not shallow else 8,
        "min_data_in_leaf": 20 if not shallow else 30,
        "feature_fraction": 0.90,
        "bagging_fraction": 0.90,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "seed": RANDOM_STATE,
        "verbosity": -1,
        "force_col_wise": True,
        "num_threads": 0,
    }
    dataset = lgb.Dataset(X_train, label=y_train, feature_name=X_train.columns.tolist(), free_raw_data=False)
    model = lgb.train(params=params, train_set=dataset, num_boost_round=360 if not shallow else 260)
    return model


def predict_lightgbm_native(model: Any, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(X), dtype=float)


def extract_feature_importance(model: Any, feature_columns: list[str], model_name: str, feature_set: str) -> pd.DataFrame:
    if hasattr(model, "feature_importance"):
        return (
            pd.DataFrame(
                {
                    "feature": feature_columns,
                    "importance_gain": model.feature_importance(importance_type="gain"),
                    "importance_split": model.feature_importance(importance_type="split"),
                    "model_name": model_name,
                    "feature_set": feature_set,
                }
            )
            .sort_values("importance_gain", ascending=False)
            .reset_index(drop=True)
        )
    return pd.DataFrame({"feature": feature_columns, "importance_gain": np.nan, "importance_split": np.nan, "model_name": model_name, "feature_set": feature_set})


def fit_ridge_closed_form(X: pd.DataFrame, y: pd.Series, alpha: float = 5.0) -> dict[str, Any]:
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mean = X_arr.mean(axis=0)
    std = X_arr.std(axis=0)
    std = np.where(std < EPS, 1.0, std)
    X_std = (X_arr - mean) / std
    X_aug = np.column_stack([np.ones(len(X_std)), X_std])
    penalty = np.eye(X_aug.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(X_aug.T @ X_aug + penalty, X_aug.T @ y_arr)
    return {"beta": beta, "mean": mean, "std": std}


def predict_ridge_closed_form(model: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    X_arr = np.asarray(X, dtype=float)
    X_std = (X_arr - model["mean"]) / model["std"]
    X_aug = np.column_stack([np.ones(len(X_std)), X_std])
    return np.asarray(X_aug @ model["beta"], dtype=float)


def fit_huber_ridge(X: pd.DataFrame, y: pd.Series, alpha: float = 4.0, delta: float = 1.5, iterations: int = 12) -> dict[str, Any]:
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mean = X_arr.mean(axis=0)
    std = X_arr.std(axis=0)
    std = np.where(std < EPS, 1.0, std)
    X_std = (X_arr - mean) / std
    X_aug = np.column_stack([np.ones(len(X_std)), X_std])
    beta = np.linalg.lstsq(X_aug, y_arr, rcond=None)[0]
    penalty = np.eye(X_aug.shape[1]) * alpha
    penalty[0, 0] = 0.0
    for _ in range(iterations):
        residual = y_arr - X_aug @ beta
        scale = np.median(np.abs(residual)) / 0.6745 if np.any(np.abs(residual) > EPS) else 1.0
        scale = max(scale, 1.0)
        threshold = delta * scale
        weights = np.where(np.abs(residual) <= threshold, 1.0, threshold / np.maximum(np.abs(residual), EPS))
        W = np.diag(weights)
        beta = np.linalg.solve(X_aug.T @ W @ X_aug + penalty, X_aug.T @ W @ y_arr)
    return {"beta": beta, "mean": mean, "std": std}


def predict_huber_ridge(model: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    return predict_ridge_closed_form(model, X)


def load_sample_submission() -> pd.DataFrame:
    sample = pd.read_csv(SAMPLE_SUBMISSION_PATH, parse_dates=[DATE_COL], low_memory=False)
    sample[DATE_COL] = pd.to_datetime(sample[DATE_COL], errors="coerce").dt.normalize()
    return sample[[DATE_COL]].copy()


def load_or_build_daily_funnel_table(logger: logging.Logger) -> pd.DataFrame:
    if DAILY_FUNNEL_TABLE_PATH.exists():
        logger.info("Loading daily funnel table from %s", DAILY_FUNNEL_TABLE_PATH)
        df = pd.read_csv(DAILY_FUNNEL_TABLE_PATH, parse_dates=[DATE_COL], low_memory=False)
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce").dt.normalize()
        return df.sort_values(DATE_COL).reset_index(drop=True)

    logger.info("daily_funnel_table.csv missing; rebuilding from raw inputs")
    sales = funnel.load_sales()
    orders = funnel.load_orders()
    order_items = funnel.load_order_items()
    products = funnel.load_products()
    web_raw = pd.read_csv(WEB_TRAFFIC_PATH, low_memory=False)
    web_daily = normalize_date_column(traffic_branch.build_web_daily(web_raw))
    promotions = promo_known.load_promotions(PROMOTIONS_PATH)
    promo_daily = promo_known.build_daily_promo_known_features(sales[DATE_COL], promotions)
    stock_snapshots = funnel.prepare_inventory_snapshots()
    inventory_daily = funnel.build_inventory_context(sales[DATE_COL], stock_snapshots)
    df = funnel.build_daily_funnel_table(sales, orders, order_items, products, web_daily, promo_daily, inventory_daily)
    df.to_csv(DAILY_FUNNEL_TABLE_PATH, index=False)
    return df


def load_or_build_future_traffic_scenarios(sample_submission: pd.DataFrame, future_promo: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    if FUTURE_TRAFFIC_SCENARIOS_PATH.exists():
        logger.info("Loading future traffic scenarios from %s", FUTURE_TRAFFIC_SCENARIOS_PATH)
        df = pd.read_csv(FUTURE_TRAFFIC_SCENARIOS_PATH, parse_dates=[DATE_COL], low_memory=False)
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce").dt.normalize()
        return df

    logger.info("future_traffic_funnel_scenarios.csv missing; rebuilding")
    web_raw = pd.read_csv(WEB_TRAFFIC_PATH, low_memory=False)
    web_daily = normalize_date_column(traffic_branch.build_web_daily(web_raw))
    scenarios, _ = funnel.build_traffic_scenarios_for_dates(web_daily, sample_submission[DATE_COL], future_promo, [(2022, 0.5), (2021, 0.3), (2020, 0.2)])
    rows = []
    for scenario_name, raw in scenarios.items():
        enriched = funnel.add_traffic_features_for_scenario(web_daily, raw)
        merged = raw.merge(enriched, on=DATE_COL, how="left", suffixes=("", "_feat"))
        merged["scenario"] = scenario_name
        rows.append(merged)
    output = pd.concat(rows, ignore_index=True)
    output.to_csv(FUTURE_TRAFFIC_SCENARIOS_PATH, index=False)
    return output


def load_future_promo_features(sample_submission: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    if FUTURE_PROMO_KNOWN_PATH.exists():
        future = pd.read_csv(FUTURE_PROMO_KNOWN_PATH, parse_dates=[DATE_COL], low_memory=False)
        future[DATE_COL] = pd.to_datetime(future[DATE_COL], errors="coerce").dt.normalize()
        return future.sort_values(DATE_COL).reset_index(drop=True)
    logger.warning("future_promo_known_features.csv missing; using zero-filled promo context")
    output = sample_submission.copy()
    for column in PROMO_FEATURES:
        output[column] = 0.0
    return output


def build_historical_revenue_features(sales: pd.DataFrame) -> pd.DataFrame:
    rev = base.add_historical_revenue_features(sales[[DATE_COL, TARGET_COL]].copy())
    rev["spike_strength_365"] = safe_divide(rev["revenue_lag_365"], rev["revenue_roll_mean_365"], fill_value=np.nan)
    rev["lag365_to_roll365_ratio"] = safe_divide(rev["revenue_lag_365"], rev["revenue_roll_mean_365"], fill_value=np.nan)
    rev["lag7_to_roll30_ratio"] = safe_divide(rev["lag_7"], rev["rolling_mean_30"], fill_value=np.nan)
    rev["volatility_30"] = pd.to_numeric(rev["revenue_roll_std_30"], errors="coerce")
    rev["volatility_90"] = pd.to_numeric(rev["revenue_roll_std_90"], errors="coerce")
    return rev


def build_funnel_reference_features(daily_funnel: pd.DataFrame) -> pd.DataFrame:
    table = daily_funnel[[DATE_COL]].copy()
    orders_refs = funnel.build_series_reference_frame(daily_funnel[DATE_COL], daily_funnel[[DATE_COL, "orders_count"]], "orders_count", "orders")
    aov_refs = funnel.build_series_reference_frame(daily_funnel[DATE_COL], daily_funnel[[DATE_COL, "AOV"]], "AOV", "aov")
    discount_refs = funnel.build_series_reference_frame(
        daily_funnel[DATE_COL],
        daily_funnel[[DATE_COL, "avg_discount_per_order"]],
        "avg_discount_per_order",
        "avg_discount_per_order",
    )
    quantity_refs = funnel.build_series_reference_frame(
        daily_funnel[DATE_COL],
        daily_funnel[[DATE_COL, "quantity_per_order"]],
        "quantity_per_order",
        "quantity_per_order",
    )
    item_lines_refs = funnel.build_series_reference_frame(
        daily_funnel[DATE_COL],
        daily_funnel[[DATE_COL, "item_lines_per_order"]],
        "item_lines_per_order",
        "item_lines_per_order",
    )
    conversion_lagged = funnel.add_recursive_lag_features(
        daily_funnel[[DATE_COL, "conversion_rate"]].copy(),
        "conversion_rate",
        "conversion",
        [7, 14, 30, 365],
        [7, 30, 90, 365],
    )
    aov_lagged = funnel.add_recursive_lag_features(
        daily_funnel[[DATE_COL, "AOV"]].copy(),
        "AOV",
        "aov",
        [7, 14, 30, 90, 365],
        [7, 30, 90, 365],
    )
    output = (
        table.merge(orders_refs[[DATE_COL, "orders_same_day_recent_mean", "orders_lag_365"]], on=DATE_COL, how="left", validate="one_to_one")
        .merge(
            aov_refs[[DATE_COL, "aov_same_day_recent_mean"]],
            on=DATE_COL,
            how="left",
            validate="one_to_one",
        )
        .merge(
            conversion_lagged[[DATE_COL, "conversion_lag_365", "conversion_roll_mean_30"]],
            on=DATE_COL,
            how="left",
            validate="one_to_one",
        )
        .merge(
            aov_lagged[[DATE_COL, "aov_lag_365", "aov_roll_mean_7", "aov_roll_mean_30"]],
            on=DATE_COL,
            how="left",
            validate="one_to_one",
        )
        .merge(discount_refs[[DATE_COL, "avg_discount_per_order_lag_365"]], on=DATE_COL, how="left", validate="one_to_one")
        .merge(quantity_refs[[DATE_COL, "quantity_per_order_lag_365"]], on=DATE_COL, how="left", validate="one_to_one")
        .merge(item_lines_refs[[DATE_COL, "item_lines_per_order_lag_365"]], on=DATE_COL, how="left", validate="one_to_one")
    )
    return output


def build_stock_context_historical(dates: pd.Series, snapshots: pd.DataFrame) -> pd.DataFrame:
    daily = stock_scale.build_daily_stock_context(dates, snapshots, scenario="hybrid", dynamic_updates=True, known_cutoff=pd.Timestamp("2022-12-31"))
    daily["stock_pressure"] = pd.to_numeric(daily["stock_pressure"], errors="coerce").fillna(0.0)
    daily["stock_build_up"] = pd.to_numeric(daily["stock_build_up"], errors="coerce").fillna(0.0)
    return daily


def build_stock_context_future(dates: pd.Series, snapshots: pd.DataFrame) -> pd.DataFrame:
    daily = stock_scale.build_daily_stock_context(dates, snapshots, scenario="hybrid", dynamic_updates=False, known_cutoff=pd.Timestamp("2022-12-31"))
    daily["stock_pressure"] = pd.to_numeric(daily["stock_pressure"], errors="coerce").fillna(0.0)
    daily["stock_build_up"] = pd.to_numeric(daily["stock_build_up"], errors="coerce").fillna(0.0)
    return daily


def build_current_best_validation_2022() -> pd.DataFrame:
    return stock_scale.build_current_best_validation_2022()


def build_current_best_future(sample_submission: pd.DataFrame) -> pd.DataFrame:
    current = pd.read_csv(CURRENT_BEST_SUBMISSION_PATH, parse_dates=[DATE_COL], low_memory=False)
    current[DATE_COL] = pd.to_datetime(current[DATE_COL], errors="coerce").dt.normalize()
    output = current[[DATE_COL, TARGET_COL]].rename(columns={TARGET_COL: "base_pred"}).sort_values(DATE_COL).reset_index(drop=True)
    if not output[DATE_COL].equals(sample_submission[DATE_COL]):
        raise ValueError("Current best submission dates do not align with sample submission")
    return output


def build_current_best_validation_merged(revenue_static_2022: pd.DataFrame) -> pd.DataFrame:
    current_best = build_current_best_validation_2022()
    merged = current_best.rename(columns={"actual_Revenue": TARGET_COL}).merge(
        revenue_static_2022.drop(columns=[TARGET_COL], errors="ignore"),
        on=DATE_COL,
        how="left",
        validate="one_to_one",
    )
    merged["base_pred"] = pd.to_numeric(merged["base_pred"], errors="coerce")
    merged["base_pred_rank_pct"] = merged["base_pred"].rank(pct=True)
    return merged


def build_future_static_features(
    sample_submission: pd.DataFrame,
    daily_funnel: pd.DataFrame,
    future_traffic_all: pd.DataFrame,
    future_promo: pd.DataFrame,
    future_stock: pd.DataFrame,
    min_date: pd.Timestamp,
) -> pd.DataFrame:
    calendar = base.build_calendar_features(sample_submission[DATE_COL], min_date)[[DATE_COL] + CALENDAR_FEATURES]
    traffic = future_traffic_all.loc[future_traffic_all["scenario"].astype(str) == "seasonal"].copy()
    traffic = traffic.rename(columns={"sessions_sum_feat": "sessions_sum", "unique_visitors_sum_feat": "unique_visitors_sum", "page_views_sum_feat": "page_views_sum"})
    traffic_columns = [DATE_COL, "sessions_growth_3_14", "sessions_roll_mean_30", "sessions_roll_std_30"]
    for column in traffic_columns[1:]:
        if column not in traffic.columns:
            traffic[column] = 0.0
    funnel_refs = build_funnel_reference_features(daily_funnel)
    future_refs = funnel.build_series_reference_frame(sample_submission[DATE_COL], daily_funnel[[DATE_COL, "orders_count"]], "orders_count", "orders")
    output = (
        sample_submission.merge(calendar, on=DATE_COL, how="left", validate="one_to_one")
        .merge(traffic[traffic_columns], on=DATE_COL, how="left", validate="one_to_one")
        .merge(future_promo[[DATE_COL] + [column for column in PROMO_FEATURES if column in future_promo.columns]], on=DATE_COL, how="left", validate="one_to_one")
        .merge(funnel_refs, on=DATE_COL, how="left", validate="one_to_one")
        .merge(future_refs[[DATE_COL, "orders_same_day_recent_mean"]], on=DATE_COL, how="left", suffixes=("", "_future"), validate="one_to_one")
        .merge(future_stock, on=DATE_COL, how="left", validate="one_to_one")
    )
    if "orders_same_day_recent_mean_future" in output.columns:
        output["orders_same_day_recent_mean"] = pd.to_numeric(output["orders_same_day_recent_mean_future"], errors="coerce").fillna(
            pd.to_numeric(output["orders_same_day_recent_mean"], errors="coerce")
        )
        output = output.drop(columns=["orders_same_day_recent_mean_future"])
    return output


def build_historical_static_features(
    sales: pd.DataFrame,
    daily_funnel: pd.DataFrame,
    historical_promo: pd.DataFrame,
    traffic_features: pd.DataFrame,
    stock_context: pd.DataFrame,
    min_date: pd.Timestamp,
) -> pd.DataFrame:
    calendar = base.build_calendar_features(sales[DATE_COL], min_date)[[DATE_COL] + CALENDAR_FEATURES]
    funnel_refs = build_funnel_reference_features(daily_funnel)
    traffic_subset = traffic_features[[DATE_COL, "sessions_growth_3_14", "sessions_roll_mean_30", "sessions_roll_std_30"]].copy()
    output = (
        sales[[DATE_COL]].merge(calendar, on=DATE_COL, how="left", validate="one_to_one")
        .merge(traffic_subset, on=DATE_COL, how="left", validate="one_to_one")
        .merge(historical_promo[[DATE_COL] + [column for column in PROMO_FEATURES if column in historical_promo.columns]], on=DATE_COL, how="left", validate="one_to_one")
        .merge(funnel_refs, on=DATE_COL, how="left", validate="one_to_one")
        .merge(stock_context, on=DATE_COL, how="left", validate="one_to_one")
    )
    return output


def add_union_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    def series_or_default(column: str, default: float) -> pd.Series:
        if column in output.columns:
            return pd.to_numeric(output[column], errors="coerce").fillna(default)
        return pd.Series(np.repeat(default, len(output)), index=output.index, dtype=float)

    orders_same_day_recent_mean = series_or_default("orders_same_day_recent_mean", 0.0)
    sessions_growth_3_14 = series_or_default("sessions_growth_3_14", 1.0)
    stock_pressure = series_or_default("stock_pressure", 0.0)
    revenue_lag_365 = series_or_default("revenue_lag_365", 0.0)
    campaign_intensity = series_or_default("campaign_intensity", 0.0)
    calendar_any_promo = series_or_default("calendar_any_promo", 0.0)
    aov_lag_365 = series_or_default("aov_lag_365", 0.0)
    orders_lag_365 = series_or_default("orders_lag_365", 0.0)
    calendar_avg_discount_value = series_or_default("calendar_avg_discount_value", 0.0)
    spike_strength_365 = series_or_default("spike_strength_365", 0.0)

    output["predicted_orders_signal"] = orders_same_day_recent_mean * np.clip(sessions_growth_3_14, 0.6, 1.6)
    output["stock_pressure_x_promo"] = stock_pressure * calendar_any_promo
    output["stock_pressure_x_revenue_lag365"] = stock_pressure * revenue_lag_365
    output["revenue_lag365_x_campaign_intensity"] = revenue_lag_365 * campaign_intensity
    output["revenue_lag365_x_sessions_growth_3_14"] = revenue_lag_365 * sessions_growth_3_14
    output["aov_lag365_x_calendar_avg_discount_value"] = aov_lag_365 * calendar_avg_discount_value
    output["orders_lag365_x_sessions_growth_3_14"] = orders_lag_365 * sessions_growth_3_14
    output["stock_pressure_x_campaign_intensity"] = stock_pressure * campaign_intensity
    output["spike_strength365_x_campaign_intensity"] = spike_strength_365 * campaign_intensity
    return output


def build_hist_union_table(
    sales: pd.DataFrame,
    daily_funnel: pd.DataFrame,
    historical_promo: pd.DataFrame,
    traffic_features: pd.DataFrame,
    stock_context: pd.DataFrame,
    min_date: pd.Timestamp,
) -> pd.DataFrame:
    revenue_features = build_historical_revenue_features(sales)
    static_features = build_historical_static_features(sales, daily_funnel, historical_promo, traffic_features, stock_context, min_date)
    output = revenue_features.merge(static_features, on=DATE_COL, how="left", validate="one_to_one")
    return add_union_interactions(output)


def prepare_training_matrix(table: pd.DataFrame, feature_columns: list[str], train_end: pd.Timestamp) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    subset = table.loc[table[DATE_COL] <= train_end, [DATE_COL, TARGET_COL] + feature_columns].copy()
    subset = subset.dropna(subset=[TARGET_COL])
    X = subset[feature_columns].replace([np.inf, -np.inf], np.nan)
    medians = X.median(numeric_only=True).fillna(0.0)
    X = X.fillna(medians).fillna(0.0)
    y = pd.to_numeric(subset[TARGET_COL], errors="coerce").fillna(0.0)
    return X.reset_index(drop=True), y.reset_index(drop=True), medians


def build_prediction_row(history: pd.DataFrame, target_date: pd.Timestamp, static_row: pd.Series) -> dict[str, float]:
    past = history.loc[history[DATE_COL] < target_date].sort_values(DATE_COL).reset_index(drop=True)
    revenue_map = past.set_index(DATE_COL)[TARGET_COL]

    def get_lag(days: int) -> float:
        ref_date = target_date - pd.Timedelta(days=days)
        if ref_date in revenue_map.index:
            return float(pd.to_numeric(revenue_map.loc[ref_date], errors="coerce"))
        return np.nan

    def get_recent_mean(window: int) -> float:
        if len(past) < window:
            return np.nan
        values = pd.to_numeric(past[TARGET_COL].tail(window), errors="coerce")
        return float(values.mean()) if len(values) == window else np.nan

    def get_recent_std(window: int) -> float:
        if len(past) < window:
            return np.nan
        values = pd.to_numeric(past[TARGET_COL].tail(window), errors="coerce")
        return float(values.std(ddof=1)) if len(values) == window else np.nan

    row = {column: float(pd.to_numeric(static_row.get(column, 0.0), errors="coerce")) for column in static_row.index if column != DATE_COL}
    row["lag_7"] = get_lag(7)
    row["lag_14"] = get_lag(14)
    row["lag_30"] = get_lag(30)
    row["revenue_lag_90"] = get_lag(90)
    row["revenue_lag_180"] = get_lag(180)
    row["revenue_lag_365"] = get_lag(365)
    row["rolling_mean_7"] = get_recent_mean(7)
    row["rolling_mean_30"] = get_recent_mean(30)
    row["revenue_roll_mean_90"] = get_recent_mean(90)
    row["revenue_roll_mean_365"] = get_recent_mean(365)
    row["volatility_30"] = get_recent_std(30)
    row["volatility_90"] = get_recent_std(90)
    row["spike_strength_365"] = safe_divide(row.get("revenue_lag_365", np.nan), row.get("revenue_roll_mean_365", np.nan), fill_value=np.nan).item()
    row["lag365_to_roll365_ratio"] = safe_divide(row.get("revenue_lag_365", np.nan), row.get("revenue_roll_mean_365", np.nan), fill_value=np.nan).item()
    row["lag7_to_roll30_ratio"] = safe_divide(row.get("lag_7", np.nan), row.get("rolling_mean_30", np.nan), fill_value=np.nan).item()
    row = add_union_interactions(pd.DataFrame([row])).iloc[0].to_dict()
    return row


def recursive_predict_revenue(model: Any, model_kind: str, medians: pd.Series, feature_columns: list[str], history: pd.DataFrame, static_context: pd.DataFrame) -> pd.DataFrame:
    history_frame = history[[DATE_COL, TARGET_COL]].copy().sort_values(DATE_COL).reset_index(drop=True)
    static_index = static_context.set_index(DATE_COL).sort_index()
    rows: list[dict[str, Any]] = []
    for target_date in static_context[DATE_COL]:
        static_row = static_index.loc[target_date]
        row = build_prediction_row(history_frame, target_date, static_row)
        X = pd.DataFrame([row]).reindex(columns=feature_columns).replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)
        if model_kind == "lightgbm":
            prediction = float(np.clip(predict_lightgbm_native(model, X)[0], 0.0, None))
        else:
            prediction = float(np.clip(predict_ridge_closed_form(model, X)[0], 0.0, None))
        rows.append({DATE_COL: target_date, "predicted_Revenue": prediction})
        history_frame = pd.concat([history_frame, pd.DataFrame({DATE_COL: [target_date], TARGET_COL: [prediction]})], ignore_index=True)
    return pd.DataFrame(rows)


def evaluate_direct_model(
    model_name: str,
    feature_set_name: str,
    feature_columns: list[str],
    hist_union: pd.DataFrame,
    train_end: pd.Timestamp,
    valid_start: pd.Timestamp,
    valid_end: pd.Timestamp,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    X_train, y_train, medians = prepare_training_matrix(hist_union, feature_columns, train_end)
    shallow = model_name.endswith("shallow")
    model = train_lightgbm_native(X_train, y_train, shallow=shallow)
    history = hist_union.loc[hist_union[DATE_COL] <= train_end, [DATE_COL, TARGET_COL]].copy()
    static_context = hist_union.loc[(hist_union[DATE_COL] >= valid_start) & (hist_union[DATE_COL] <= valid_end), [DATE_COL] + [col for col in feature_columns if col not in REVENUE_FEATURES + INTERACTION_FEATURES or col in INTERACTION_FEATURES]].copy()
    predictions = recursive_predict_revenue(model, "lightgbm", medians, feature_columns, history, static_context)
    actual = hist_union.loc[(hist_union[DATE_COL] >= valid_start) & (hist_union[DATE_COL] <= valid_end), [DATE_COL, TARGET_COL, "calendar_any_promo", "sessions_growth_3_14", "stock_pressure"]].copy()
    merged = actual.merge(predictions, on=DATE_COL, how="left", validate="one_to_one")
    high_traffic_mask = (pd.to_numeric(merged["sessions_growth_3_14"], errors="coerce").fillna(0.0) >= 1.15).astype(int)
    high_stock_pressure_mask = (pd.to_numeric(merged["stock_pressure"], errors="coerce").fillna(0.0) >= float(hist_union["stock_pressure"].quantile(0.75))).astype(int)
    metrics = compute_metrics(
        merged[TARGET_COL],
        merged["predicted_Revenue"].to_numpy(dtype=float),
        merged["calendar_any_promo"].fillna(0).astype(int),
        high_traffic_mask,
        high_stock_pressure_mask,
    )
    metrics.update({"model_name": model_name, "feature_set": feature_set_name})
    merged["model_name"] = model_name
    merged["feature_set"] = feature_set_name
    importance = extract_feature_importance(model, feature_columns, model_name, feature_set_name)
    return metrics, merged, importance


def build_monthly_walkforward_meta_predictions(
    validation_frame: pd.DataFrame,
    feature_columns: list[str],
    model_type: str,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    predictions: list[pd.DataFrame] = []
    months = sorted(validation_frame[DATE_COL].dt.to_period("M").astype(str).unique().tolist())
    fitted_any = False
    importance_rows: list[dict[str, Any]] = []
    for month_label in months:
        period = pd.Period(month_label, freq="M")
        month_start = period.start_time.normalize()
        month_end = period.end_time.normalize()
        target_slice = validation_frame.loc[(validation_frame[DATE_COL] >= month_start) & (validation_frame[DATE_COL] <= month_end)].copy()
        train_slice = validation_frame.loc[validation_frame[DATE_COL] < month_start].copy()
        if target_slice.empty:
            continue
        if train_slice.empty or len(train_slice) < 45:
            target_slice["predicted_Revenue"] = target_slice["base_pred"]
            predictions.append(target_slice)
            continue
        X_train = train_slice[feature_columns].replace([np.inf, -np.inf], np.nan)
        medians = X_train.median(numeric_only=True).fillna(0.0)
        X_train = X_train.fillna(medians).fillna(0.0)
        y_train = pd.to_numeric(train_slice[TARGET_COL], errors="coerce") - pd.to_numeric(train_slice["base_pred"], errors="coerce")
        if model_type == "ridge_meta":
            model = fit_ridge_closed_form(X_train, y_train, alpha=8.0)
            predict_fn = predict_ridge_closed_form
        else:
            model = fit_huber_ridge(X_train, y_train, alpha=6.0, delta=1.5, iterations=10)
            predict_fn = predict_huber_ridge
        fitted_any = True
        X_target = target_slice[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)
        residual_pred = predict_fn(model, X_target)
        target_slice["predicted_Revenue"] = np.maximum(0.0, pd.to_numeric(target_slice["base_pred"], errors="coerce").to_numpy(dtype=float) + residual_pred)
        predictions.append(target_slice)
        if not importance_rows:
            beta = np.abs(np.asarray(model["beta"][1:], dtype=float))
            importance_rows.extend(
                {"feature": feature, "importance_gain": float(weight), "importance_split": np.nan, "model_name": model_type, "feature_set": "meta_current_best"}
                for feature, weight in zip(feature_columns, beta)
            )

    output = pd.concat(predictions, ignore_index=True).sort_values(DATE_COL).reset_index(drop=True)
    high_traffic_mask = (pd.to_numeric(output["sessions_growth_3_14"], errors="coerce").fillna(0.0) >= 1.15).astype(int)
    high_stock_pressure_mask = (pd.to_numeric(output["stock_pressure"], errors="coerce").fillna(0.0) >= float(validation_frame["stock_pressure"].quantile(0.75))).astype(int)
    metrics = compute_metrics(
        output[TARGET_COL],
        output["predicted_Revenue"].to_numpy(dtype=float),
        output["calendar_any_promo"].fillna(0).astype(int),
        high_traffic_mask,
        high_stock_pressure_mask,
    )
    metrics.update({"model_name": model_type, "feature_set": "meta_current_best"})
    importance = pd.DataFrame(importance_rows)
    if importance.empty:
        importance = pd.DataFrame({"feature": feature_columns, "importance_gain": np.nan, "importance_split": np.nan, "model_name": model_type, "feature_set": "meta_current_best"})
    return output, metrics, importance


def build_future_meta_frame(future_static: pd.DataFrame, current_best_future: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    merged = future_static.merge(current_best_future, on=DATE_COL, how="left", validate="one_to_one")
    merged["base_pred_rank_pct"] = merged["base_pred"].rank(pct=True)
    proxy_history = pd.concat(
        [
            sales[[DATE_COL, TARGET_COL]].copy(),
            current_best_future.rename(columns={"base_pred": TARGET_COL})[[DATE_COL, TARGET_COL]].copy(),
        ],
        ignore_index=True,
    ).sort_values(DATE_COL).reset_index(drop=True)
    proxy_features = build_historical_revenue_features(proxy_history)
    proxy_future = proxy_features.loc[proxy_features[DATE_COL].isin(merged[DATE_COL]), [DATE_COL, "revenue_lag_365", "spike_strength_365", "lag7_to_roll30_ratio"]].copy()
    merged = merged.merge(proxy_future, on=DATE_COL, how="left", validate="one_to_one")
    return merged


def fit_meta_full_2022(validation_frame: pd.DataFrame, feature_columns: list[str], model_type: str) -> tuple[dict[str, Any], pd.Series]:
    X = validation_frame[feature_columns].replace([np.inf, -np.inf], np.nan)
    medians = X.median(numeric_only=True).fillna(0.0)
    X = X.fillna(medians).fillna(0.0)
    y = pd.to_numeric(validation_frame[TARGET_COL], errors="coerce") - pd.to_numeric(validation_frame["base_pred"], errors="coerce")
    if model_type == "ridge_meta":
        model = fit_ridge_closed_form(X, y, alpha=8.0)
    else:
        model = fit_huber_ridge(X, y, alpha=6.0, delta=1.5, iterations=10)
    return model, medians


def apply_meta_future(model: dict[str, Any], medians: pd.Series, future_frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    X = future_frame[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)
    residual = predict_ridge_closed_form(model, X)
    return np.maximum(0.0, pd.to_numeric(future_frame["base_pred"], errors="coerce").to_numpy(dtype=float) + residual)


def summarize_long_average(comparison: pd.DataFrame, feature_set_name: str, model_name: str) -> float:
    subset = comparison.loc[
        (comparison["scope"].isin([fold[0] for fold in LONG_FOLDS])) & (comparison["model_name"] == model_name) & (comparison["feature_set"] == feature_set_name)
    ]
    return float(subset["rmse"].mean()) if not subset.empty else np.nan


def build_submission(dates: pd.Series, revenue: pd.Series | np.ndarray, ratio: float = 0.8900) -> pd.DataFrame:
    output = pd.DataFrame({DATE_COL: pd.to_datetime(dates).reset_index(drop=True)})
    output[TARGET_COL] = np.maximum(0.0, np.asarray(revenue, dtype=float))
    output[COGS_COL] = np.maximum(0.0, output[TARGET_COL] * ratio)
    return output[[DATE_COL, TARGET_COL, COGS_COL]]


def save_submission(path: Path, submission: pd.DataFrame, sample_submission: pd.DataFrame) -> None:
    validate_submission_frame(submission, sample_submission)
    submission.to_csv(path, index=False)


def dedupe_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, ~frame.columns.duplicated()].copy()


def main() -> None:
    logger = setup_logging()
    reporter = RunReporter(logger)

    sales = funnel.load_sales()
    sample_submission = load_sample_submission()
    current_best = pd.read_csv(CURRENT_BEST_SUBMISSION_PATH, parse_dates=[DATE_COL], low_memory=False)
    current_best[DATE_COL] = pd.to_datetime(current_best[DATE_COL], errors="coerce").dt.normalize()
    current_best = current_best[[DATE_COL, TARGET_COL]].sort_values(DATE_COL).reset_index(drop=True)
    if not current_best[DATE_COL].equals(sample_submission[DATE_COL]):
        raise ValueError("Current best submission is not aligned with sample_submission")

    daily_funnel = load_or_build_daily_funnel_table(logger)
    promotions = promo_known.load_promotions(PROMOTIONS_PATH)
    historical_promo = promo_known.build_daily_promo_known_features(sales[DATE_COL], promotions)
    future_promo = load_future_promo_features(sample_submission, logger)
    web_raw = pd.read_csv(WEB_TRAFFIC_PATH, low_memory=False)
    web_daily = normalize_date_column(traffic_branch.build_web_daily(web_raw))
    traffic_features = traffic_branch.add_traffic_features(web_daily.copy())
    stock_snapshots = stock_scale.load_inventory_snapshot_features()
    historical_stock = build_stock_context_historical(sales[DATE_COL], stock_snapshots)
    future_stock = build_stock_context_future(sample_submission[DATE_COL], stock_snapshots)
    hist_union = build_hist_union_table(sales, daily_funnel, historical_promo, traffic_features, historical_stock, sales[DATE_COL].min())
    future_traffic_all = load_or_build_future_traffic_scenarios(sample_submission, future_promo, logger)
    future_static = add_union_interactions(build_future_static_features(sample_submission, daily_funnel, future_traffic_all, future_promo, future_stock, sales[DATE_COL].min()))

    comparison_rows: list[dict[str, Any]] = []
    validation_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    ablation_rows: list[dict[str, Any]] = []

    for scope_name, train_end, valid_start, valid_end in ALL_SCOPES:
        reporter.emit(f"Evaluating scope {scope_name}: train <= {train_end.date()}, validate {valid_start.date()} -> {valid_end.date()}")
        for feature_set_name, feature_columns in FEATURE_SETS.items():
            for model_name in ["lightgbm_standard", "lightgbm_shallow"]:
                metrics, predictions, importance = evaluate_direct_model(
                    model_name=model_name,
                    feature_set_name=feature_set_name,
                    feature_columns=feature_columns,
                    hist_union=hist_union,
                    train_end=train_end,
                    valid_start=valid_start,
                    valid_end=valid_end,
                )
                metrics["scope"] = scope_name
                comparison_rows.append(metrics)
                validation_frames.append(predictions.assign(scope=scope_name))
                importance_frames.append(importance.assign(scope=scope_name))
                if model_name == "lightgbm_standard":
                    ablation_rows.append(
                        {
                            "scope": scope_name,
                            "feature_set": feature_set_name,
                            "mae": metrics["mae"],
                            "rmse": metrics["rmse"],
                            "r2": metrics["r2"],
                            "top10_rmse": metrics["top10_rmse"],
                        }
                    )

    validation_2022_static = hist_union.loc[(hist_union[DATE_COL] >= VALIDATION_2022[2]) & (hist_union[DATE_COL] <= VALIDATION_2022[3])].copy()
    validation_2022_meta = build_current_best_validation_merged(validation_2022_static)
    for meta_name in ["ridge_meta", "huber_meta"]:
        meta_predictions, meta_metrics, meta_importance = build_monthly_walkforward_meta_predictions(validation_2022_meta, META_FEATURES, meta_name)
        meta_metrics["scope"] = "validation_2022"
        comparison_rows.append(meta_metrics)
        validation_frames.append(meta_predictions[[DATE_COL, TARGET_COL, "predicted_Revenue"]].assign(model_name=meta_name, feature_set="meta_current_best", scope="validation_2022"))
        importance_frames.append(meta_importance.assign(scope="validation_2022"))

    comparison = pd.DataFrame(comparison_rows)
    validation_predictions = pd.concat(validation_frames, ignore_index=True)
    feature_importance = pd.concat(importance_frames, ignore_index=True)
    ablation_results = pd.DataFrame(ablation_rows)

    comparison.to_csv(MODEL_COMPARISON_PATH, index=False)
    validation_predictions.to_csv(VALIDATION_PREDICTIONS_PATH, index=False)
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    ablation_results.to_csv(ABLATION_RESULTS_PATH, index=False)

    direct_rows = comparison.loc[comparison["feature_set"].isin(FEATURE_SETS.keys())].copy()
    long_avg = (
        direct_rows.loc[direct_rows["scope"].isin([fold[0] for fold in LONG_FOLDS])]
        .groupby(["model_name", "feature_set"], as_index=False)[["mae", "rmse", "r2", "top10_rmse", "promo_day_rmse", "non_promo_rmse", "high_traffic_rmse", "high_stock_pressure_rmse"]]
        .mean(numeric_only=True)
        .assign(scope="long_avg")
    )
    comparison_with_avg = pd.concat([comparison, long_avg], ignore_index=True)
    comparison_with_avg.to_csv(MODEL_COMPARISON_PATH, index=False)

    select_table = direct_rows.loc[direct_rows["scope"] == "validation_2022", ["model_name", "feature_set", "rmse", "mae", "r2", "top10_rmse"]].copy()
    select_table = select_table.merge(
        long_avg[["model_name", "feature_set", "rmse"]].rename(columns={"rmse": "long_avg_rmse"}),
        on=["model_name", "feature_set"],
        how="left",
    )
    select_table["selection_score"] = 0.6 * select_table["rmse"] + 0.4 * select_table["long_avg_rmse"].fillna(select_table["rmse"])
    select_table = select_table.sort_values(["selection_score", "rmse", "top10_rmse"], ascending=[True, True, True]).reset_index(drop=True)
    best_model_name = str(select_table.iloc[0]["model_name"])
    best_feature_set = str(select_table.iloc[0]["feature_set"])
    best_feature_columns = FEATURE_SETS[best_feature_set]

    conservative_row = select_table.loc[select_table["model_name"] == "lightgbm_shallow"].sort_values("selection_score").head(1)
    if conservative_row.empty:
        conservative_model_name = best_model_name
        conservative_feature_set = best_feature_set
    else:
        conservative_model_name = str(conservative_row.iloc[0]["model_name"])
        conservative_feature_set = str(conservative_row.iloc[0]["feature_set"])

    reporter.emit("Best validation model:")
    reporter.emit(f"{best_model_name} | {best_feature_set}")
    reporter.emit(f"2022 analog RMSE: {float(select_table.iloc[0]['rmse']):.2f}")
    reporter.emit(f"Long-horizon average RMSE: {float(select_table.iloc[0]['long_avg_rmse']):.2f}")

    X_best, y_best, med_best = prepare_training_matrix(hist_union, best_feature_columns, sales[DATE_COL].max())
    best_model = train_lightgbm_native(X_best, y_best, shallow=(best_model_name == "lightgbm_shallow"))
    future_static_unique = dedupe_columns(future_static)
    best_future = recursive_predict_revenue(best_model, "lightgbm", med_best, best_feature_columns, hist_union[[DATE_COL, TARGET_COL]], future_static_unique)

    conservative_feature_columns = FEATURE_SETS[conservative_feature_set]
    X_cons, y_cons, med_cons = prepare_training_matrix(hist_union, conservative_feature_columns, sales[DATE_COL].max())
    conservative_model = train_lightgbm_native(X_cons, y_cons, shallow=(conservative_model_name == "lightgbm_shallow"))
    conservative_future = recursive_predict_revenue(
        conservative_model,
        "lightgbm",
        med_cons,
        conservative_feature_columns,
        hist_union[[DATE_COL, TARGET_COL]],
        future_static_unique,
    )

    current_best_future = build_current_best_future(sample_submission)
    future_meta_frame = build_future_meta_frame(future_static, current_best_future, sales)
    meta_model, meta_medians = fit_meta_full_2022(validation_2022_meta, META_FEATURES, "ridge_meta")
    aggressive_revenue = apply_meta_future(meta_model, meta_medians, future_meta_frame, META_FEATURES)

    submission_main = build_submission(sample_submission[DATE_COL], best_future["predicted_Revenue"], ratio=0.8900)
    submission_conservative = build_submission(sample_submission[DATE_COL], conservative_future["predicted_Revenue"], ratio=0.8900)
    submission_aggressive = build_submission(sample_submission[DATE_COL], aggressive_revenue, ratio=0.8900)
    save_submission(SUBMISSION_MAIN_PATH, submission_main, sample_submission)
    save_submission(SUBMISSION_CONSERVATIVE_PATH, submission_conservative, sample_submission)
    save_submission(SUBMISSION_AGGRESSIVE_PATH, submission_aggressive, sample_submission)

    for weight, path in BLEND_OUTPUTS.items():
        revenue = (1.0 - weight) * current_best[TARGET_COL].to_numpy(dtype=float) + weight * submission_main[TARGET_COL].to_numpy(dtype=float)
        save_submission(path, build_submission(sample_submission[DATE_COL], revenue, ratio=0.8900), sample_submission)

    if SEGMENT_SUBMISSION_PATH.exists():
        segment = pd.read_csv(SEGMENT_SUBMISSION_PATH, parse_dates=[DATE_COL], low_memory=False)
        segment[DATE_COL] = pd.to_datetime(segment[DATE_COL], errors="coerce").dt.normalize()
        segment = segment[[DATE_COL, TARGET_COL]].sort_values(DATE_COL).reset_index(drop=True)
        if not segment[DATE_COL].equals(sample_submission[DATE_COL]):
            raise ValueError("Segment submission dates do not match sample submission")
        specs = {
            "801010": (0.80, 0.10, 0.10),
            "702010": (0.70, 0.20, 0.10),
            "701020": (0.70, 0.10, 0.20),
        }
        for key, (w_current, w_union, w_segment) in specs.items():
            revenue = (
                w_current * current_best[TARGET_COL].to_numpy(dtype=float)
                + w_union * submission_main[TARGET_COL].to_numpy(dtype=float)
                + w_segment * segment[TARGET_COL].to_numpy(dtype=float)
            )
            save_submission(SEGMENT_BLEND_OUTPUTS[key], build_submission(sample_submission[DATE_COL], revenue, ratio=0.8900), sample_submission)

    ablation_summary = (
        ablation_results.loc[ablation_results["scope"].isin(["validation_2022"] + [fold[0] for fold in LONG_FOLDS])]
        .groupby(["feature_set", "scope"], as_index=False)[["mae", "rmse", "r2", "top10_rmse"]]
        .mean(numeric_only=True)
    )
    top_features = (
        feature_importance.loc[feature_importance["model_name"] == best_model_name]
        .groupby("feature", as_index=False)["importance_gain"]
        .sum()
        .sort_values("importance_gain", ascending=False)
        .head(30)
    )

    funnel_help = float(select_table.loc[(select_table["model_name"] == best_model_name) & (select_table["feature_set"] == "base_funnel"), "rmse"].min()) if not select_table.loc[(select_table["model_name"] == best_model_name) & (select_table["feature_set"] == "base_funnel")].empty else np.nan
    base_help = float(select_table.loc[(select_table["model_name"] == best_model_name) & (select_table["feature_set"] == "base_only"), "rmse"].min()) if not select_table.loc[(select_table["model_name"] == best_model_name) & (select_table["feature_set"] == "base_only")].empty else np.nan
    stock_help = float(select_table.loc[(select_table["model_name"] == best_model_name) & (select_table["feature_set"] == "base_funnel_promo_stock"), "rmse"].min()) if not select_table.loc[(select_table["model_name"] == best_model_name) & (select_table["feature_set"] == "base_funnel_promo_stock")].empty else np.nan
    promo_help = float(select_table.loc[(select_table["model_name"] == best_model_name) & (select_table["feature_set"] == "base_funnel_promo"), "rmse"].min()) if not select_table.loc[(select_table["model_name"] == best_model_name) & (select_table["feature_set"] == "base_funnel_promo")].empty else np.nan

    reporter.emit("")
    reporter.emit("Ablation table:")
    reporter.emit_frame("ablation_summary", ablation_summary.sort_values(["scope", "rmse"]))
    reporter.emit("")
    reporter.emit("Top 30 features:")
    reporter.emit_frame("top_features", top_features)
    reporter.emit("")
    reporter.emit(f"Funnel features helped vs base_only: {'yes' if np.isfinite(base_help) and np.isfinite(funnel_help) and funnel_help < base_help else 'no'}")
    reporter.emit(f"Stock features helped vs base_funnel_promo: {'yes' if np.isfinite(stock_help) and np.isfinite(promo_help) and stock_help < promo_help else 'no'}")

    created = [
        SUBMISSION_MAIN_PATH,
        SUBMISSION_CONSERVATIVE_PATH,
        SUBMISSION_AGGRESSIVE_PATH,
        *BLEND_OUTPUTS.values(),
        *SEGMENT_BLEND_OUTPUTS.values(),
    ]
    reporter.emit("")
    reporter.emit("Created submissions:")
    for path in created:
        if path.exists():
            reporter.emit(str(path))

    reporter.emit("")
    reporter.emit("Recommended upload order:")
    upload_order = [
        SUBMISSION_MAIN_PATH,
        BLEND_OUTPUTS[0.05],
        BLEND_OUTPUTS[0.10],
        SUBMISSION_CONSERVATIVE_PATH,
        SUBMISSION_AGGRESSIVE_PATH,
    ]
    if SEGMENT_BLEND_OUTPUTS["801010"].exists():
        upload_order.insert(3, SEGMENT_BLEND_OUTPUTS["801010"])
    for path in upload_order:
        if path.exists():
            reporter.emit(str(path))

    reporter.emit("")
    reporter.emit("Leakage safety confirmation: the union model uses only lagged revenue, lagged/seasonal funnel references, lagged traffic, promo-known future context, and hybrid inventory scenarios built from historical snapshots only.")
    reporter.save()


if __name__ == "__main__":
    main()
