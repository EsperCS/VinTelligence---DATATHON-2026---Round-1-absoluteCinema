from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import train_feature_union_model as union_model
import train_funnel_model as funnel
import train_promo_known_pipeline as promo_known
import train_stock_aware_scaling as stock_scale
import train_traffic_driven_model as traffic_branch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "log"

SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
DAILY_FUNNEL_TABLE_PATH = DATA_DIR / "daily_funnel_table.csv"
FUTURE_TRAFFIC_SCENARIOS_PATH = DATA_DIR / "future_traffic_funnel_scenarios.csv"
FUTURE_PROMO_KNOWN_PATH = DATA_DIR / "future_promo_known_features.csv"
CURRENT_BEST_SUBMISSION_PATH = DATA_DIR / "submission_blend_direct_15_cogs8900.csv"
SEGMENT_SUBMISSION_PATH = DATA_DIR / "submission_m5_segment_bottomup.csv"
FEATURE_UNION_SUBMISSION_PATH = DATA_DIR / "submission_feature_union.csv"
FEATURE_UNION_VALIDATION_PATH = DATA_DIR / "feature_union_validation_predictions.csv"
M5_VALIDATION_PATH = DATA_DIR / "m5_multilevel_validation_predictions.csv"

VALIDATION_PREDICTIONS_PATH = DATA_DIR / "feature_subset_validation_predictions.csv"
MODEL_COMPARISON_PATH = DATA_DIR / "feature_subset_model_comparison.csv"
ENSEMBLE_SEARCH_PATH = DATA_DIR / "feature_subset_ensemble_search.csv"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "feature_subset_feature_importance.csv"
REPORT_PATH = LOG_DIR / "feature_subset_ensemble_report.txt"
LOG_FILE = LOG_DIR / "train_feature_subset_ensemble.log"

SUBMISSION_SEASONAL_LAG_PATH = DATA_DIR / "submission_subset_seasonal_lag.csv"
SUBMISSION_FUNNEL_DEMAND_PATH = DATA_DIR / "submission_subset_funnel_demand.csv"
SUBMISSION_PROMO_SPIKE_PATH = DATA_DIR / "submission_subset_promo_spike.csv"
SUBMISSION_STOCK_CONSTRAINT_PATH = DATA_DIR / "submission_subset_stock_constraint.csv"
SUBMISSION_MINIMAL_HYBRID_PATH = DATA_DIR / "submission_subset_minimal_hybrid.csv"
SUBMISSION_ENSEMBLE_SPECIALISTS_PATH = DATA_DIR / "submission_subset_ensemble_specialists.csv"
SUBMISSION_ENSEMBLE_CURRENT_PATH = DATA_DIR / "submission_subset_ensemble_current.csv"
SUBMISSION_ENSEMBLE_CURRENT_SEGMENT_PATH = DATA_DIR / "submission_subset_ensemble_current_segment.csv"
BLEND_OUTPUTS = {
    0.05: DATA_DIR / "submission_subset_blend_05.csv",
    0.10: DATA_DIR / "submission_subset_blend_10.csv",
    0.15: DATA_DIR / "submission_subset_blend_15.csv",
}

DATE_COL = union_model.DATE_COL
TARGET_COL = union_model.TARGET_COL
COGS_COL = union_model.COGS_COL
RANDOM_STATE = union_model.RANDOM_STATE
EPS = 1e-9

VALIDATION_2022 = union_model.VALIDATION_2022
LONG_FOLDS = union_model.LONG_FOLDS
ALL_SCOPES = union_model.ALL_SCOPES

SPECIALIST_FEATURES = {
    "seasonal_lag": [
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
        "lag365_to_roll365_ratio",
        "lag7_to_roll30_ratio",
        "lag30_to_roll90_ratio",
        "spike_strength_365",
        "volatility_30",
        "volatility_90",
        "calendar_any_promo",
        "day_of_week",
        "day_of_year",
        "month",
    ],
    "funnel_demand": [
        "orders_lag_365",
        "orders_same_day_recent_mean",
        "predicted_orders_signal",
        "orders_roll_mean_7",
        "orders_roll_mean_30",
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
        "day_of_week",
        "day_of_year",
        "month",
    ],
    "promo_spike": [
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
        "spike_strength_365",
        "revenue_lag_365",
        "revenue_lag365_x_campaign_intensity",
        "spike_strength365_x_campaign_intensity",
        "day_of_year",
        "month",
    ],
    "stock_constraint": [
        "inv_avg_days_of_supply",
        "inv_avg_sell_through_rate",
        "inv_stockout_rate",
        "stock_pressure",
        "stock_build_up",
        "stockout_pressure",
        "restock_signal",
        "stock_pressure_x_campaign_intensity",
        "stock_pressure_x_revenue_lag365",
        "revenue_lag_365",
        "day_of_year",
        "month",
    ],
    "minimal_hybrid": [
        "revenue_lag_365",
        "rolling_mean_7",
        "rolling_mean_30",
        "spike_strength_365",
        "orders_lag_365",
        "orders_same_day_recent_mean",
        "sessions_growth_3_14",
        "aov_roll_mean_7",
        "conversion_roll_mean_30",
        "campaign_intensity",
        "discount_x_progress",
        "stock_pressure",
        "day_of_week",
        "day_of_year",
        "month",
    ],
}

SPECIALIST_OUTPUTS = {
    "seasonal_lag": SUBMISSION_SEASONAL_LAG_PATH,
    "funnel_demand": SUBMISSION_FUNNEL_DEMAND_PATH,
    "promo_spike": SUBMISSION_PROMO_SPIKE_PATH,
    "stock_constraint": SUBMISSION_STOCK_CONSTRAINT_PATH,
    "minimal_hybrid": SUBMISSION_MINIMAL_HYBRID_PATH,
}


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
    logger = logging.getLogger("train_feature_subset_ensemble")
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


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def r2_score_manual(actual: np.ndarray, predicted: np.ndarray) -> float:
    denom = float(np.sum(np.square(actual - np.mean(actual))))
    if denom <= EPS:
        return 0.0
    return 1.0 - float(np.sum(np.square(actual - predicted))) / denom


def compute_metrics(actual: pd.Series, predicted: np.ndarray, promo_mask: pd.Series, high_traffic_mask: pd.Series, high_stock_pressure_mask: pd.Series) -> dict[str, float]:
    actual_arr = np.asarray(actual, dtype=float)
    pred_arr = np.asarray(predicted, dtype=float)
    top10_threshold = float(np.quantile(actual_arr, 0.90))
    top10 = actual_arr >= top10_threshold
    promo = promo_mask.fillna(0).astype(bool).to_numpy()
    high_traffic = high_traffic_mask.fillna(0).astype(bool).to_numpy()
    high_stock = high_stock_pressure_mask.fillna(0).astype(bool).to_numpy()
    return {
        "mae": mae(actual_arr, pred_arr),
        "rmse": rmse(actual_arr, pred_arr),
        "r2": r2_score_manual(actual_arr, pred_arr),
        "top10_rmse": rmse(actual_arr[top10], pred_arr[top10]) if top10.any() else np.nan,
        "top10_underprediction_count": float(np.sum(pred_arr[top10] < actual_arr[top10])) if top10.any() else np.nan,
        "promo_day_rmse": rmse(actual_arr[promo], pred_arr[promo]) if promo.any() else np.nan,
        "non_promo_rmse": rmse(actual_arr[~promo], pred_arr[~promo]) if (~promo).any() else np.nan,
        "high_traffic_rmse": rmse(actual_arr[high_traffic], pred_arr[high_traffic]) if high_traffic.any() else np.nan,
        "high_stock_pressure_rmse": rmse(actual_arr[high_stock], pred_arr[high_stock]) if high_stock.any() else np.nan,
    }


def load_sample_submission() -> pd.DataFrame:
    sample = pd.read_csv(SAMPLE_SUBMISSION_PATH, parse_dates=[DATE_COL], low_memory=False)
    sample[DATE_COL] = pd.to_datetime(sample[DATE_COL], errors="coerce").dt.normalize()
    return sample[[DATE_COL]].copy()


def load_feature_union_hist_and_future(logger: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sales = funnel.load_sales()
    sample_submission = load_sample_submission()
    daily_funnel = union_model.load_or_build_daily_funnel_table(logger)
    promotions = promo_known.load_promotions(union_model.PROMOTIONS_PATH)
    historical_promo = promo_known.build_daily_promo_known_features(sales[DATE_COL], promotions)
    future_promo = union_model.load_future_promo_features(sample_submission, logger)
    web_raw = pd.read_csv(union_model.WEB_TRAFFIC_PATH, low_memory=False)
    web_daily = union_model.normalize_date_column(traffic_branch.build_web_daily(web_raw))
    traffic_features = traffic_branch.add_traffic_features(web_daily.copy())
    stock_snapshots = stock_scale.load_inventory_snapshot_features()
    historical_stock = union_model.build_stock_context_historical(sales[DATE_COL], stock_snapshots)
    hist_union = union_model.build_hist_union_table(sales, daily_funnel, historical_promo, traffic_features, historical_stock, sales[DATE_COL].min())

    # Add specialist-only extras to historical table.
    orders_lagged = funnel.add_recursive_lag_features(
        daily_funnel[[DATE_COL, "orders_count"]].copy(),
        "orders_count",
        "orders",
        [7, 14, 30, 90, 365],
        [7, 30, 90, 365],
    )[[DATE_COL, "orders_roll_mean_7", "orders_roll_mean_30"]]
    hist_union = hist_union.merge(orders_lagged, on=DATE_COL, how="left", validate="one_to_one")
    hist_union["lag30_to_roll90_ratio"] = union_model.safe_divide(hist_union["lag_30"], hist_union["revenue_roll_mean_90"], fill_value=np.nan)
    hist_union["revenue_lag365_x_campaign_intensity"] = hist_union["revenue_lag365_x_campaign_intensity"]

    future_traffic_all = union_model.load_or_build_future_traffic_scenarios(sample_submission, future_promo, logger)
    future_stock = union_model.build_stock_context_future(sample_submission[DATE_COL], stock_snapshots)

    # Better future static references for specialist families.
    future_static = union_model.build_future_static_features(sample_submission, daily_funnel, future_traffic_all, future_promo, future_stock, sales[DATE_COL].min())
    future_orders_refs = funnel.build_series_reference_frame(sample_submission[DATE_COL], daily_funnel[[DATE_COL, "orders_count"]], "orders_count", "orders")
    future_conversion_refs = funnel.build_series_reference_frame(sample_submission[DATE_COL], daily_funnel[[DATE_COL, "conversion_rate"]], "conversion_rate", "conversion")
    future_aov_refs = funnel.build_series_reference_frame(sample_submission[DATE_COL], daily_funnel[[DATE_COL, "AOV"]], "AOV", "aov")
    future_discount_refs = funnel.build_series_reference_frame(sample_submission[DATE_COL], daily_funnel[[DATE_COL, "avg_discount_per_order"]], "avg_discount_per_order", "avg_discount_per_order")
    future_qty_refs = funnel.build_series_reference_frame(sample_submission[DATE_COL], daily_funnel[[DATE_COL, "quantity_per_order"]], "quantity_per_order", "quantity_per_order")
    future_lines_refs = funnel.build_series_reference_frame(sample_submission[DATE_COL], daily_funnel[[DATE_COL, "item_lines_per_order"]], "item_lines_per_order", "item_lines_per_order")

    def merge_missing_columns(base_frame: pd.DataFrame, addition_frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        wanted = [column for column in columns if column not in base_frame.columns]
        if not wanted:
            return base_frame
        return base_frame.merge(addition_frame[[DATE_COL] + wanted], on=DATE_COL, how="left", validate="one_to_one")

    future_static = merge_missing_columns(
        future_static,
        future_orders_refs,
        ["orders_lag_365", "orders_same_day_recent_mean", "orders_same_month_recent_mean"],
    )
    future_static = merge_missing_columns(
        future_static,
        future_conversion_refs,
        ["conversion_lag_365", "conversion_same_month_recent_mean"],
    )
    future_static = merge_missing_columns(
        future_static,
        future_aov_refs,
        ["aov_lag_365", "aov_same_day_recent_mean", "aov_same_month_recent_mean"],
    )
    future_static = merge_missing_columns(future_static, future_discount_refs, ["avg_discount_per_order_lag_365"])
    future_static = merge_missing_columns(future_static, future_qty_refs, ["quantity_per_order_lag_365"])
    future_static = merge_missing_columns(future_static, future_lines_refs, ["item_lines_per_order_lag_365"])
    future_static["orders_roll_mean_7"] = pd.to_numeric(future_static["orders_same_day_recent_mean"], errors="coerce")
    future_static["orders_roll_mean_30"] = pd.to_numeric(future_static["orders_same_month_recent_mean"], errors="coerce")
    future_static["conversion_roll_mean_30"] = pd.to_numeric(future_static["conversion_same_month_recent_mean"], errors="coerce")
    future_static["aov_roll_mean_7"] = pd.to_numeric(future_static["aov_same_day_recent_mean"], errors="coerce")
    future_static["aov_roll_mean_30"] = pd.to_numeric(future_static["aov_same_month_recent_mean"], errors="coerce")
    future_static["lag30_to_roll90_ratio"] = np.nan
    future_static = union_model.add_union_interactions(future_static)
    future_static["stock_pressure_x_campaign_intensity"] = pd.to_numeric(future_static["stock_pressure"], errors="coerce").fillna(0.0) * pd.to_numeric(
        future_static["campaign_intensity"], errors="coerce"
    ).fillna(0.0)

    return sales, sample_submission, hist_union, future_static, daily_funnel


def build_specialist_prediction_row(history: pd.DataFrame, target_date: pd.Timestamp, static_row: pd.Series) -> dict[str, float]:
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
    row["spike_strength_365"] = union_model.safe_divide(row.get("revenue_lag_365", np.nan), row.get("revenue_roll_mean_365", np.nan), fill_value=np.nan).item()
    row["lag365_to_roll365_ratio"] = union_model.safe_divide(row.get("revenue_lag_365", np.nan), row.get("revenue_roll_mean_365", np.nan), fill_value=np.nan).item()
    row["lag7_to_roll30_ratio"] = union_model.safe_divide(row.get("lag_7", np.nan), row.get("rolling_mean_30", np.nan), fill_value=np.nan).item()
    row["lag30_to_roll90_ratio"] = union_model.safe_divide(row.get("lag_30", np.nan), row.get("revenue_roll_mean_90", np.nan), fill_value=np.nan).item()
    row = union_model.add_union_interactions(pd.DataFrame([row])).iloc[0].to_dict()
    return row


def recursive_predict_specialist_revenue(
    model: Any,
    model_kind: str,
    medians: pd.Series,
    feature_columns: list[str],
    history: pd.DataFrame,
    static_context: pd.DataFrame,
) -> pd.DataFrame:
    history_frame = history[[DATE_COL, TARGET_COL]].copy().sort_values(DATE_COL).reset_index(drop=True)
    static_index = static_context.set_index(DATE_COL).sort_index()
    rows: list[dict[str, Any]] = []
    for target_date in static_context[DATE_COL]:
        static_row = static_index.loc[target_date]
        row = build_specialist_prediction_row(history_frame, target_date, static_row)
        X = (
            pd.DataFrame([row])
            .reindex(columns=feature_columns)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(medians)
            .fillna(0.0)
        )
        if model_kind == "lightgbm":
            prediction = float(np.clip(union_model.predict_lightgbm_native(model, X)[0], 0.0, None))
        else:
            raise RuntimeError(f"Unsupported model kind: {model_kind}")
        rows.append({DATE_COL: target_date, "predicted_Revenue": prediction})
        history_frame = pd.concat([history_frame, pd.DataFrame({DATE_COL: [target_date], TARGET_COL: [prediction]})], ignore_index=True)
    return pd.DataFrame(rows)


def save_submission_no_overwrite(path: Path, submission: pd.DataFrame, sample_submission: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing submission: {path}")
    union_model.save_submission(path, submission, sample_submission)


def evaluate_specialist(
    specialist_name: str,
    model_variant: str,
    feature_columns: list[str],
    hist_union: pd.DataFrame,
    train_end: pd.Timestamp,
    valid_start: pd.Timestamp,
    valid_end: pd.Timestamp,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    X_train, y_train, medians = union_model.prepare_training_matrix(hist_union, feature_columns, train_end)
    if model_variant in {"lightgbm_standard", "lightgbm_shallow"}:
        model = union_model.train_lightgbm_native(X_train, y_train, shallow=(model_variant == "lightgbm_shallow"))
        model_kind = "lightgbm"
    else:
        raise RuntimeError("ExtraTrees fallback unavailable: scikit-learn is not installed in this environment")

    history = hist_union.loc[hist_union[DATE_COL] <= train_end, [DATE_COL, TARGET_COL]].copy()
    static_context = hist_union.loc[(hist_union[DATE_COL] >= valid_start) & (hist_union[DATE_COL] <= valid_end)].copy()
    predictions = recursive_predict_specialist_revenue(model, model_kind, medians, feature_columns, history, static_context)
    actual = hist_union.loc[(hist_union[DATE_COL] >= valid_start) & (hist_union[DATE_COL] <= valid_end), [DATE_COL, TARGET_COL, "calendar_any_promo", "sessions_growth_3_14", "stock_pressure"]].copy()
    merged = actual.merge(predictions, on=DATE_COL, how="left", validate="one_to_one")
    high_traffic_mask = (pd.to_numeric(merged["sessions_growth_3_14"], errors="coerce").fillna(0.0) >= 1.15).astype(int)
    high_stock_mask = (pd.to_numeric(merged["stock_pressure"], errors="coerce").fillna(0.0) >= float(hist_union["stock_pressure"].quantile(0.75))).astype(int)
    metrics = compute_metrics(
        merged[TARGET_COL],
        merged["predicted_Revenue"].to_numpy(dtype=float),
        merged["calendar_any_promo"].fillna(0).astype(int),
        high_traffic_mask,
        high_stock_mask,
    )
    metrics.update({"specialist_name": specialist_name, "model_variant": model_variant})
    importance = union_model.extract_feature_importance(model, feature_columns, f"{specialist_name}__{model_variant}", specialist_name)
    merged["specialist_name"] = specialist_name
    merged["model_variant"] = model_variant
    return metrics, merged, importance


def summarize_specialist_selection(comparison: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    long_avg = (
        comparison.loc[comparison["scope"].isin([fold[0] for fold in LONG_FOLDS])]
        .groupby(["specialist_name", "model_variant"], as_index=False)[["rmse", "mae", "r2", "top10_rmse"]]
        .mean(numeric_only=True)
        .rename(columns={"rmse": "long_avg_rmse", "mae": "long_avg_mae", "r2": "long_avg_r2", "top10_rmse": "long_avg_top10_rmse"})
    )
    select = comparison.loc[comparison["scope"] == "validation_2022"].merge(long_avg, on=["specialist_name", "model_variant"], how="left")
    select["selection_score"] = 0.6 * select["rmse"] + 0.4 * select["long_avg_rmse"].fillna(select["rmse"])
    select = select.sort_values(["specialist_name", "selection_score", "rmse", "top10_rmse"]).reset_index(drop=True)
    chosen: dict[str, str] = {}
    rows = []
    for specialist_name in SPECIALIST_FEATURES:
        specialist_rows = select.loc[select["specialist_name"] == specialist_name].copy()
        best_row = specialist_rows.iloc[0]
        chosen[specialist_name] = str(best_row["model_variant"])
        rows.append(best_row)
    return pd.DataFrame(rows), chosen


def build_specialist_validation_wide(
    selected_predictions: pd.DataFrame,
    hist_union: pd.DataFrame,
) -> pd.DataFrame:
    base = hist_union.loc[(hist_union[DATE_COL] >= VALIDATION_2022[2]) & (hist_union[DATE_COL] <= VALIDATION_2022[3]), [DATE_COL, TARGET_COL, "calendar_any_promo", "sessions_growth_3_14", "stock_pressure"]].copy()
    for specialist_name in SPECIALIST_FEATURES:
        specialist_pred = selected_predictions.loc[
            (selected_predictions["scope"] == "validation_2022") & (selected_predictions["specialist_name"] == specialist_name),
            [DATE_COL, "predicted_Revenue"],
        ].rename(columns={"predicted_Revenue": specialist_name})
        base = base.merge(specialist_pred, on=DATE_COL, how="left", validate="one_to_one")
    return base


def load_optional_validation_components(validation_wide: pd.DataFrame) -> pd.DataFrame:
    output = validation_wide.copy()
    current_best = union_model.build_current_best_validation_2022()
    output = output.merge(current_best[[DATE_COL, "base_pred"]].rename(columns={"base_pred": "current_best"}), on=DATE_COL, how="left", validate="one_to_one")

    if M5_VALIDATION_PATH.exists():
        m5 = pd.read_csv(M5_VALIDATION_PATH, parse_dates=[DATE_COL], low_memory=False)
        m5[DATE_COL] = pd.to_datetime(m5[DATE_COL], errors="coerce").dt.normalize()
        segment = m5.loc[(m5["fold"] == "fold_3") & (m5[DATE_COL] >= VALIDATION_2022[2]) & (m5[DATE_COL] <= VALIDATION_2022[3]), [DATE_COL, "segment_recursive_sum"]].rename(columns={"segment_recursive_sum": "segment_bottomup"})
        output = output.merge(segment, on=DATE_COL, how="left", validate="one_to_one")

    if FEATURE_UNION_VALIDATION_PATH.exists():
        fu = pd.read_csv(FEATURE_UNION_VALIDATION_PATH, parse_dates=[DATE_COL], low_memory=False)
        fu[DATE_COL] = pd.to_datetime(fu[DATE_COL], errors="coerce").dt.normalize()
        fu_best = fu.loc[
            (fu["scope"] == "validation_2022") & (fu["model_name"] == "lightgbm_standard") & (fu["feature_set"] == "full_union"),
            [DATE_COL, "predicted_Revenue"],
        ].rename(columns={"predicted_Revenue": "feature_union"})
        output = output.merge(fu_best, on=DATE_COL, how="left", validate="one_to_one")
    return output


def generate_weight_tuples(num_components: int, total_units: int, max_units: int | None = None) -> list[tuple[int, ...]]:
    results: list[tuple[int, ...]] = []

    def rec(position: int, remaining: int, current: list[int]) -> None:
        if position == num_components - 1:
            if max_units is None or remaining <= max_units:
                results.append(tuple(current + [remaining]))
            return
        upper = remaining if max_units is None else min(remaining, max_units)
        for value in range(upper + 1):
            rec(position + 1, remaining - value, current + [value])

    rec(0, total_units, [])
    return results


def evaluate_ensemble_predictions(actual: np.ndarray, pred: np.ndarray, promo: np.ndarray, high_traffic: np.ndarray, high_stock: np.ndarray) -> dict[str, float]:
    top10_threshold = float(np.quantile(actual, 0.90))
    top10 = actual >= top10_threshold
    return {
        "mae": mae(actual, pred),
        "rmse": rmse(actual, pred),
        "r2": r2_score_manual(actual, pred),
        "top10_rmse": rmse(actual[top10], pred[top10]) if top10.any() else np.nan,
    }


def search_ensembles(validation_wide: pd.DataFrame) -> pd.DataFrame:
    actual = validation_wide[TARGET_COL].to_numpy(dtype=float)
    promo = validation_wide["calendar_any_promo"].fillna(0).astype(bool).to_numpy()
    high_traffic = (pd.to_numeric(validation_wide["sessions_growth_3_14"], errors="coerce").fillna(0.0) >= 1.15).to_numpy()
    high_stock = (pd.to_numeric(validation_wide["stock_pressure"], errors="coerce").fillna(0.0) >= float(pd.to_numeric(validation_wide["stock_pressure"], errors="coerce").quantile(0.75))).to_numpy()
    specialist_names = list(SPECIALIST_FEATURES.keys())
    pred_arrays = {name: validation_wide[name].to_numpy(dtype=float) for name in specialist_names}
    rows: list[dict[str, Any]] = []

    # Mode 1: specialists only.
    mode1_weights = generate_weight_tuples(len(specialist_names), total_units=20, max_units=10)
    for units in mode1_weights:
        weights = np.asarray(units, dtype=float) / 20.0
        pred = np.zeros(len(validation_wide), dtype=float)
        for idx, name in enumerate(specialist_names):
            pred += weights[idx] * pred_arrays[name]
        metrics = evaluate_ensemble_predictions(actual, pred, promo, high_traffic, high_stock)
        rows.append(
            {
                "mode": "specialists_only",
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
                "top10_rmse": metrics["top10_rmse"],
                "current_best_weight": 0.0,
                "segment_weight": 0.0,
                **{f"weight_{name}": weights[idx] for idx, name in enumerate(specialist_names)},
            }
        )

    # Mode 2: specialists + current best.
    if "current_best" in validation_wide.columns:
        current_pred = validation_wide["current_best"].to_numpy(dtype=float)
        for current_units in range(6, 19):  # 0.30 -> 0.90
            remaining = 20 - current_units
            for units in generate_weight_tuples(len(specialist_names), total_units=remaining, max_units=10):
                weights = np.asarray(units, dtype=float) / 20.0
                pred = current_units / 20.0 * current_pred
                for idx, name in enumerate(specialist_names):
                    pred += weights[idx] * pred_arrays[name]
                metrics = evaluate_ensemble_predictions(actual, pred, promo, high_traffic, high_stock)
                rows.append(
                    {
                        "mode": "specialists_current",
                        "rmse": metrics["rmse"],
                        "mae": metrics["mae"],
                        "r2": metrics["r2"],
                        "top10_rmse": metrics["top10_rmse"],
                        "current_best_weight": current_units / 20.0,
                        "segment_weight": 0.0,
                        **{f"weight_{name}": weights[idx] for idx, name in enumerate(specialist_names)},
                    }
                )

    # Mode 3: specialists + current best + segment.
    if {"current_best", "segment_bottomup"}.issubset(validation_wide.columns):
        current_pred = validation_wide["current_best"].to_numpy(dtype=float)
        segment_pred = validation_wide["segment_bottomup"].to_numpy(dtype=float)
        for current_units in range(6, 19):
            for segment_units in range(0, 21 - current_units):
                remaining = 20 - current_units - segment_units
                if remaining < 0:
                    continue
                for units in generate_weight_tuples(len(specialist_names), total_units=remaining, max_units=10):
                    weights = np.asarray(units, dtype=float) / 20.0
                    pred = current_units / 20.0 * current_pred + segment_units / 20.0 * segment_pred
                    for idx, name in enumerate(specialist_names):
                        pred += weights[idx] * pred_arrays[name]
                    metrics = evaluate_ensemble_predictions(actual, pred, promo, high_traffic, high_stock)
                    rows.append(
                        {
                            "mode": "specialists_current_segment",
                            "rmse": metrics["rmse"],
                            "mae": metrics["mae"],
                            "r2": metrics["r2"],
                            "top10_rmse": metrics["top10_rmse"],
                            "current_best_weight": current_units / 20.0,
                            "segment_weight": segment_units / 20.0,
                            **{f"weight_{name}": weights[idx] for idx, name in enumerate(specialist_names)},
                        }
                    )

    output = pd.DataFrame(rows).sort_values(["rmse", "top10_rmse", "mae"]).reset_index(drop=True)
    return output


def apply_weights_to_future(weights_row: pd.Series, specialist_future: dict[str, pd.Series], current_best: pd.Series | None, segment: pd.Series | None) -> np.ndarray:
    pred = np.zeros(len(next(iter(specialist_future.values()))), dtype=float)
    for name, series in specialist_future.items():
        pred += float(weights_row.get(f"weight_{name}", 0.0)) * np.asarray(series, dtype=float)
    if current_best is not None:
        pred += float(weights_row.get("current_best_weight", 0.0)) * np.asarray(current_best, dtype=float)
    if segment is not None:
        pred += float(weights_row.get("segment_weight", 0.0)) * np.asarray(segment, dtype=float)
    return pred


def main() -> None:
    logger = setup_logging()
    reporter = RunReporter(logger)
    sales, sample_submission, hist_union, future_static, _daily_funnel = load_feature_union_hist_and_future(logger)

    comparison_rows: list[dict[str, Any]] = []
    validation_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []

    extratrees_available = False
    try:
        import sklearn  # type: ignore  # noqa: F401
        extratrees_available = True
    except Exception:
        extratrees_available = False
        logger.info("scikit-learn unavailable; skipping ExtraTrees fallback")

    model_variants = ["lightgbm_standard", "lightgbm_shallow"]
    if extratrees_available:
        model_variants.append("extratrees_fallback")

    for scope_name, train_end, valid_start, valid_end in ALL_SCOPES:
        reporter.emit(f"Evaluating scope {scope_name}: train <= {train_end.date()}, validate {valid_start.date()} -> {valid_end.date()}")
        for specialist_name, feature_columns in SPECIALIST_FEATURES.items():
            for model_variant in model_variants:
                if model_variant == "extratrees_fallback":
                    continue
                metrics, preds, importance = evaluate_specialist(
                    specialist_name=specialist_name,
                    model_variant=model_variant,
                    feature_columns=feature_columns,
                    hist_union=hist_union,
                    train_end=train_end,
                    valid_start=valid_start,
                    valid_end=valid_end,
                )
                metrics["scope"] = scope_name
                comparison_rows.append(metrics)
                validation_frames.append(preds.assign(scope=scope_name))
                importance_frames.append(importance.assign(scope=scope_name))

    comparison = pd.DataFrame(comparison_rows)
    validation_predictions = pd.concat(validation_frames, ignore_index=True)
    feature_importance = pd.concat(importance_frames, ignore_index=True)
    comparison.to_csv(MODEL_COMPARISON_PATH, index=False)
    validation_predictions.to_csv(VALIDATION_PREDICTIONS_PATH, index=False)
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    selection_table, chosen_variants = summarize_specialist_selection(comparison)
    reporter.emit("Selected specialist variants:")
    reporter.emit_frame("selection_table", selection_table[["specialist_name", "model_variant", "rmse", "long_avg_rmse", "selection_score", "top10_rmse"]])

    selected_validation_predictions = validation_predictions.loc[
        validation_predictions.apply(lambda row: chosen_variants.get(row["specialist_name"]) == row["model_variant"], axis=1)
    ].copy()
    validation_wide = build_specialist_validation_wide(selected_validation_predictions, hist_union)
    validation_wide = load_optional_validation_components(validation_wide)

    ensemble_search = search_ensembles(validation_wide)
    ensemble_search.to_csv(ENSEMBLE_SEARCH_PATH, index=False)

    best_specialists_only = ensemble_search.loc[ensemble_search["mode"] == "specialists_only"].iloc[0]
    best_current = ensemble_search.loc[ensemble_search["mode"] == "specialists_current"].iloc[0] if (ensemble_search["mode"] == "specialists_current").any() else None
    best_current_segment = ensemble_search.loc[ensemble_search["mode"] == "specialists_current_segment"].iloc[0] if (ensemble_search["mode"] == "specialists_current_segment").any() else None
    best_overall = ensemble_search.iloc[0]

    # Train final selected specialist models.
    specialist_future_predictions: dict[str, pd.Series] = {}
    for specialist_name, feature_columns in SPECIALIST_FEATURES.items():
        variant = chosen_variants[specialist_name]
        X_train, y_train, medians = union_model.prepare_training_matrix(hist_union, feature_columns, sales[DATE_COL].max())
        model = union_model.train_lightgbm_native(X_train, y_train, shallow=(variant == "lightgbm_shallow"))
        future_pred = recursive_predict_specialist_revenue(
            model,
            "lightgbm",
            medians,
            feature_columns,
            hist_union[[DATE_COL, TARGET_COL]],
            future_static,
        )
        specialist_future_predictions[specialist_name] = future_pred["predicted_Revenue"]
        submission = union_model.build_submission(sample_submission[DATE_COL], future_pred["predicted_Revenue"], ratio=0.8900)
        save_submission_no_overwrite(SPECIALIST_OUTPUTS[specialist_name], submission, sample_submission)

    current_best_future = None
    if CURRENT_BEST_SUBMISSION_PATH.exists():
        current_best_df = pd.read_csv(CURRENT_BEST_SUBMISSION_PATH, parse_dates=[DATE_COL], low_memory=False)
        current_best_df[DATE_COL] = pd.to_datetime(current_best_df[DATE_COL], errors="coerce").dt.normalize()
        current_best_df = current_best_df[[DATE_COL, TARGET_COL]].sort_values(DATE_COL).reset_index(drop=True)
        current_best_future = current_best_df[TARGET_COL]

    segment_future = None
    if SEGMENT_SUBMISSION_PATH.exists():
        seg = pd.read_csv(SEGMENT_SUBMISSION_PATH, parse_dates=[DATE_COL], low_memory=False)
        seg[DATE_COL] = pd.to_datetime(seg[DATE_COL], errors="coerce").dt.normalize()
        seg = seg[[DATE_COL, TARGET_COL]].sort_values(DATE_COL).reset_index(drop=True)
        if seg[DATE_COL].equals(sample_submission[DATE_COL]):
            segment_future = seg[TARGET_COL]

    specialists_future_pred = apply_weights_to_future(best_specialists_only, specialist_future_predictions, None, None)
    save_submission_no_overwrite(
        SUBMISSION_ENSEMBLE_SPECIALISTS_PATH,
        union_model.build_submission(sample_submission[DATE_COL], specialists_future_pred, ratio=0.8900),
        sample_submission,
    )

    if best_current is not None and current_best_future is not None:
        current_future_pred = apply_weights_to_future(best_current, specialist_future_predictions, current_best_future, None)
        save_submission_no_overwrite(
            SUBMISSION_ENSEMBLE_CURRENT_PATH,
            union_model.build_submission(sample_submission[DATE_COL], current_future_pred, ratio=0.8900),
            sample_submission,
        )
    else:
        current_future_pred = specialists_future_pred

    if best_current_segment is not None and current_best_future is not None and segment_future is not None:
        current_segment_pred = apply_weights_to_future(best_current_segment, specialist_future_predictions, current_best_future, segment_future)
        save_submission_no_overwrite(
            SUBMISSION_ENSEMBLE_CURRENT_SEGMENT_PATH,
            union_model.build_submission(sample_submission[DATE_COL], current_segment_pred, ratio=0.8900),
            sample_submission,
        )
    else:
        current_segment_pred = current_future_pred

    # Conservative blends with current best using specialists-only ensemble for diversity.
    if current_best_future is not None:
        for weight, path in BLEND_OUTPUTS.items():
            revenue = (1.0 - weight) * np.asarray(current_best_future, dtype=float) + weight * specialists_future_pred
            save_submission_no_overwrite(path, union_model.build_submission(sample_submission[DATE_COL], revenue, ratio=0.8900), sample_submission)

    best_individual = selection_table.sort_values(["rmse", "top10_rmse"]).iloc[0]
    long_avg_best_specialist = float(
        comparison.loc[
            (comparison["specialist_name"] == best_individual["specialist_name"])
            & (comparison["model_variant"] == best_individual["model_variant"])
            & (comparison["scope"].isin([fold[0] for fold in LONG_FOLDS])),
            "rmse",
        ].mean()
    )

    top_features = (
        feature_importance.loc[
            feature_importance["model_name"] == f"{best_individual['specialist_name']}__{best_individual['model_variant']}"
        ]
        .groupby("feature", as_index=False)["importance_gain"]
        .sum()
        .sort_values("importance_gain", ascending=False)
        .head(30)
    )

    feature_family_help = selection_table.sort_values("selection_score").iloc[0]["specialist_name"]

    reporter.emit("")
    reporter.emit(f"Best individual specialist model: {best_individual['specialist_name']} | {best_individual['model_variant']}")
    reporter.emit_frame(
        "Specialist model comparison table",
        selection_table[["specialist_name", "model_variant", "rmse", "long_avg_rmse", "r2", "top10_rmse", "selection_score"]].sort_values("selection_score"),
    )
    reporter.emit("")
    reporter.emit("Best ensemble rows:")
    ensemble_summary = pd.DataFrame(
        [
            {"mode": "specialists_only", **best_specialists_only.to_dict()},
            {"mode": "specialists_current", **(best_current.to_dict() if best_current is not None else {})},
            {"mode": "specialists_current_segment", **(best_current_segment.to_dict() if best_current_segment is not None else {})},
        ]
    )
    reporter.emit_frame(
        "ensemble_summary",
        ensemble_summary[[column for column in ["mode", "rmse", "mae", "r2", "top10_rmse", "current_best_weight", "segment_weight"] + [f"weight_{name}" for name in SPECIALIST_FEATURES] if column in ensemble_summary.columns]],
    )
    reporter.emit("")
    current_best_rmse = float(rmse(validation_wide[TARGET_COL].to_numpy(dtype=float), validation_wide["current_best"].to_numpy(dtype=float))) if "current_best" in validation_wide.columns else np.nan
    reporter.emit(f"2022 analog RMSE before/after: {current_best_rmse:.2f} -> {float(best_overall['rmse']):.2f}")
    reporter.emit(f"Long-horizon average RMSE (best individual specialist): {long_avg_best_specialist:.2f}")
    if "current_best" in validation_wide.columns:
        top10_threshold = float(np.quantile(validation_wide[TARGET_COL].to_numpy(dtype=float), 0.90))
        top10_mask = validation_wide[TARGET_COL].to_numpy(dtype=float) >= top10_threshold
        before_top10 = rmse(validation_wide[TARGET_COL].to_numpy(dtype=float)[top10_mask], validation_wide["current_best"].to_numpy(dtype=float)[top10_mask])
        reporter.emit(f"Top 10% RMSE before/after: {before_top10:.2f} -> {float(best_overall['top10_rmse']):.2f}")
    reporter.emit(f"Which feature family helped most: {feature_family_help}")
    reporter.emit_frame("Top 30 features", top_features)
    reporter.emit("")
    reporter.emit("Created submissions:")
    created_paths = [
        *SPECIALIST_OUTPUTS.values(),
        SUBMISSION_ENSEMBLE_SPECIALISTS_PATH,
        SUBMISSION_ENSEMBLE_CURRENT_PATH,
        SUBMISSION_ENSEMBLE_CURRENT_SEGMENT_PATH,
        *BLEND_OUTPUTS.values(),
    ]
    for path in created_paths:
        if path.exists():
            reporter.emit(str(path))
    reporter.emit("")
    reporter.emit("Recommended upload order:")
    recommended = []
    if best_current_segment is not None and SUBMISSION_ENSEMBLE_CURRENT_SEGMENT_PATH.exists():
        recommended.append(SUBMISSION_ENSEMBLE_CURRENT_SEGMENT_PATH)
    if best_current is not None and SUBMISSION_ENSEMBLE_CURRENT_PATH.exists():
        recommended.append(SUBMISSION_ENSEMBLE_CURRENT_PATH)
    recommended.extend(
        [
            SUBMISSION_ENSEMBLE_SPECIALISTS_PATH,
            BLEND_OUTPUTS[0.05],
            BLEND_OUTPUTS[0.10],
            SUBMISSION_MINIMAL_HYBRID_PATH,
        ]
    )
    for path in recommended:
        if path.exists():
            reporter.emit(str(path))
    reporter.emit("")
    reporter.emit("Leakage safety confirmation: specialist models use lagged revenue, seasonal funnel references, seasonal traffic scenario, promo-known future context, and hybrid stock scenario built only from historical snapshots. No future actual Revenue/COGS or same-day future realized demand was used.")
    reporter.save()


if __name__ == "__main__":
    main()
