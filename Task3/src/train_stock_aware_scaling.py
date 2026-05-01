from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import train_direct_seasonal_residual_model as direct_seasonal
import train_final_model as base
import train_promo_known_pipeline as promo_known


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "log"

INVENTORY_PATH = DATA_DIR / "inventory.csv"
PRODUCTS_PATH = DATA_DIR / "products.csv"
SALES_PATH = DATA_DIR / "sales.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
CURRENT_BEST_SUBMISSION_PATH = DATA_DIR / "submission_blend_direct_15_cogs8900.csv"
FINAL_MICRO_VALIDATION_PATH = DATA_DIR / "final_micro_calibration_validation_predictions.csv"
DIRECT_SEASONAL_VALIDATION_PATH = DATA_DIR / "direct_seasonal_validation_predictions.csv"
DIRECT_SEASONAL_IMPORTANCE_PATH = DATA_DIR / "direct_seasonal_feature_importance.csv"
PROMO_KNOWN_VALIDATION_PATH = DATA_DIR / "promo_known_validation_predictions.csv"
FUTURE_PROMO_KNOWN_PATH = DATA_DIR / "future_promo_known_features.csv"
PROMOTIONS_PATH = DATA_DIR / "promotions.csv"

RESULTS_PATH = DATA_DIR / "stock_aware_scaling_results.csv"
VALIDATION_OUTPUT_PATH = DATA_DIR / "stock_aware_validation_predictions.csv"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "stock_aware_feature_importance.csv"
REPORT_PATH = LOG_DIR / "stock_aware_scaling_report.txt"
LOG_FILE = LOG_DIR / "train_stock_aware_scaling.log"

SUBMISSION_CARRY_PATH = DATA_DIR / "submission_stock_scale_carry.csv"
SUBMISSION_MONTHAVG_PATH = DATA_DIR / "submission_stock_scale_monthavg.csv"
SUBMISSION_HYBRID_PATH = DATA_DIR / "submission_stock_scale_hybrid.csv"
SUBMISSION_CONSERVATIVE_PATH = DATA_DIR / "submission_stock_scale_conservative.csv"
SUBMISSION_AGGRESSIVE_PATH = DATA_DIR / "submission_stock_scale_aggressive.csv"

BLEND_OUTPUTS = {
    0.05: DATA_DIR / "submission_stock_blend_05.csv",
    0.10: DATA_DIR / "submission_stock_blend_10.csv",
    0.15: DATA_DIR / "submission_stock_blend_15.csv",
    0.20: DATA_DIR / "submission_stock_blend_20.csv",
}

DATE_COL = base.DATE_COL
TARGET_COL = base.TARGET_COL
COGS_COL = base.COGS_COL
RANDOM_STATE = base.RANDOM_STATE

VALID_START = pd.Timestamp("2022-01-01")
VALID_END = pd.Timestamp("2022-12-31")
TRAIN_END = pd.Timestamp("2021-12-31")
EPS = 1e-9

STOCK_BASE_FEATURES = [
    "inv_avg_days_of_supply",
    "inv_min_days_of_supply",
    "inv_median_days_of_supply",
    "inv_avg_sell_through_rate",
    "inv_stockout_rate",
    "inv_reorder_rate",
    "inv_fill_rate_mean",
    "inv_overstock_rate",
    "restock_signal",
    "stock_build_up",
]
STOCK_DERIVED_FEATURES = [
    "stock_pressure",
    "low_stock_flag",
    "high_stock_flag",
    "stockout_pressure",
]
META_FEATURES = [
    "base_pred",
    "base_pred_log1p",
    "base_pred_rank_pct",
    "stock_pressure",
    "low_stock_flag",
    "high_stock_flag",
    "stockout_pressure",
    "restock_signal",
    "stock_build_up",
    "promo_active",
    "discount",
    "month",
    "day_of_week",
    "high_stock_x_spike_prob",
    "low_stock_x_base_pred",
    "stock_pressure_x_base_pred",
    "stockout_pressure_x_base_pred",
    "stock_pressure_x_promo_active",
    "low_stock_x_promo_active",
    "restock_signal_x_promo_active",
    "high_stock_x_discount",
    "stock_build_up_x_campaign_active",
]
TARGET_CLIPS = [(0.80, 1.25), (0.85, 1.20), (0.90, 1.15)]
PREDICTION_CLIPS = [(0.95, 1.08), (0.93, 1.12), (0.90, 1.15)]


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
    logger = logging.getLogger("train_stock_aware_scaling")
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


def safe_divide(numerator: float, denominator: float, fill: float = np.nan) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or abs(float(denominator)) < EPS:
        return fill
    return float(numerator) / float(denominator)


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


def make_safe_feature_names(columns: pd.Index) -> pd.Index:
    seen: dict[str, int] = {}
    safe_names: list[str] = []
    for raw_column in columns:
        safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(raw_column)).strip("_")
        safe = safe or "feature"
        count = seen.get(safe, 0)
        seen[safe] = count + 1
        safe_names.append(safe if count == 0 else f"{safe}_{count}")
    return pd.Index(safe_names)


def load_sales() -> pd.DataFrame:
    sales = pd.read_csv(SALES_PATH, parse_dates=[DATE_COL], low_memory=False)
    sales[DATE_COL] = pd.to_datetime(sales[DATE_COL], errors="coerce").dt.normalize()
    sales = sales.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    sales[TARGET_COL] = pd.to_numeric(sales[TARGET_COL], errors="coerce")
    sales[COGS_COL] = pd.to_numeric(sales[COGS_COL], errors="coerce")
    return sales


def load_sample_submission() -> pd.DataFrame:
    sample = pd.read_csv(SAMPLE_SUBMISSION_PATH, parse_dates=[DATE_COL], low_memory=False)
    sample[DATE_COL] = pd.to_datetime(sample[DATE_COL], errors="coerce").dt.normalize()
    return sample.sort_values(DATE_COL).reset_index(drop=True)


def load_current_best_submission() -> pd.DataFrame:
    frame = pd.read_csv(CURRENT_BEST_SUBMISSION_PATH, parse_dates=[DATE_COL], low_memory=False)
    frame[DATE_COL] = pd.to_datetime(frame[DATE_COL], errors="coerce").dt.normalize()
    return frame.sort_values(DATE_COL).reset_index(drop=True)


def load_inventory_snapshot_features() -> pd.DataFrame:
    inventory = pd.read_csv(INVENTORY_PATH, low_memory=False)
    inventory["snapshot_date"] = pd.to_datetime(inventory["snapshot_date"], errors="coerce").dt.normalize()
    inventory = inventory.dropna(subset=["snapshot_date"]).copy()
    for column in [
        "stock_on_hand",
        "units_received",
        "days_of_supply",
        "fill_rate",
        "stockout_flag",
        "reorder_flag",
        "sell_through_rate",
        "overstock_flag",
    ]:
        inventory[column] = pd.to_numeric(inventory.get(column, 0.0), errors="coerce").fillna(0.0)

    snapshots = (
        inventory.groupby("snapshot_date", as_index=False)
        .agg(
            inv_avg_days_of_supply=("days_of_supply", "mean"),
            inv_min_days_of_supply=("days_of_supply", "min"),
            inv_median_days_of_supply=("days_of_supply", "median"),
            inv_avg_sell_through_rate=("sell_through_rate", "mean"),
            inv_stockout_rate=("stockout_flag", "mean"),
            inv_reorder_rate=("reorder_flag", "mean"),
            inv_fill_rate_mean=("fill_rate", "mean"),
            inv_overstock_rate=("overstock_flag", "mean"),
            stock_on_hand_sum=("stock_on_hand", "sum"),
            units_received_sum=("units_received", "sum"),
        )
        .rename(columns={"snapshot_date": DATE_COL})
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )
    snapshots["restock_signal"] = snapshots["units_received_sum"] / snapshots["stock_on_hand_sum"].replace(0, np.nan)
    snapshots["rolling_mean_stock_on_hand_3_month"] = (
        snapshots["stock_on_hand_sum"].shift(1).rolling(window=3, min_periods=1).mean()
    )
    snapshots["stock_build_up"] = snapshots["stock_on_hand_sum"] / snapshots["rolling_mean_stock_on_hand_3_month"].replace(0, np.nan)
    snapshots["stock_pressure"] = snapshots["inv_avg_sell_through_rate"] / snapshots["inv_avg_days_of_supply"].replace(0, np.nan)
    snapshots["stockout_pressure"] = snapshots["inv_stockout_rate"] * snapshots["inv_avg_sell_through_rate"]
    snapshots["month"] = snapshots[DATE_COL].dt.month.astype(int)
    return snapshots


def build_month_average_lookup(snapshots: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    available = snapshots.loc[snapshots[DATE_COL] <= cutoff].copy()
    if available.empty:
        return pd.DataFrame(columns=["month"] + STOCK_BASE_FEATURES + ["stock_pressure", "stockout_pressure"])
    month_avg = (
        available.groupby("month", as_index=False)[STOCK_BASE_FEATURES + ["stock_pressure", "stockout_pressure"]]
        .mean(numeric_only=True)
        .reset_index(drop=True)
    )
    return month_avg


def build_daily_stock_context(
    dates: pd.Series,
    snapshots: pd.DataFrame,
    scenario: str,
    dynamic_updates: bool,
    known_cutoff: pd.Timestamp,
) -> pd.DataFrame:
    calendar = pd.DataFrame({DATE_COL: pd.to_datetime(dates).sort_values().unique()})
    calendar["month"] = calendar[DATE_COL].dt.month.astype(int)

    if dynamic_updates:
        available = snapshots.copy()
    else:
        available = snapshots.loc[snapshots[DATE_COL] <= known_cutoff].copy()

    if available.empty:
        for column in STOCK_BASE_FEATURES + ["stock_pressure", "stockout_pressure"]:
            calendar[column] = 0.0
        return calendar[[DATE_COL] + STOCK_BASE_FEATURES + ["stock_pressure", "stockout_pressure"]]

    carry = pd.merge_asof(
        calendar[[DATE_COL]].sort_values(DATE_COL),
        available[[DATE_COL] + STOCK_BASE_FEATURES + ["stock_pressure", "stockout_pressure"]].sort_values(DATE_COL),
        on=DATE_COL,
        direction="backward",
    )

    month_avg_rows: list[dict[str, Any]] = []
    if dynamic_updates:
        for target_date in calendar[DATE_COL]:
            candidates = available.loc[available[DATE_COL] <= target_date].copy()
            month = int(target_date.month)
            same_month = candidates.loc[candidates["month"] == month]
            use_frame = same_month if not same_month.empty else candidates
            row = {DATE_COL: target_date}
            for column in STOCK_BASE_FEATURES + ["stock_pressure", "stockout_pressure"]:
                row[column] = float(pd.to_numeric(use_frame[column], errors="coerce").mean()) if not use_frame.empty else 0.0
            month_avg_rows.append(row)
        month_avg = pd.DataFrame(month_avg_rows)
    else:
        month_lookup = build_month_average_lookup(snapshots, known_cutoff)
        month_avg = calendar.merge(month_lookup, on="month", how="left").drop(columns="month")

    carry = carry.fillna(0.0)
    month_avg = month_avg.fillna(0.0)
    if scenario == "carry":
        output = carry
    elif scenario == "monthavg":
        output = month_avg
    else:
        output = carry.copy()
        for column in STOCK_BASE_FEATURES + ["stock_pressure", "stockout_pressure"]:
            output[column] = 0.7 * pd.to_numeric(carry[column], errors="coerce").fillna(0.0) + 0.3 * pd.to_numeric(
                month_avg[column], errors="coerce"
            ).fillna(0.0)
    return output[[DATE_COL] + STOCK_BASE_FEATURES + ["stock_pressure", "stockout_pressure"]]


def add_stock_flags_and_interactions(
    frame: pd.DataFrame,
    low_threshold: float,
    high_threshold: float,
) -> pd.DataFrame:
    output = frame.copy()
    output["low_stock_flag"] = (pd.to_numeric(output["inv_avg_days_of_supply"], errors="coerce") < low_threshold).astype(float)
    output["high_stock_flag"] = (pd.to_numeric(output["inv_avg_days_of_supply"], errors="coerce") > high_threshold).astype(float)
    output["stockout_pressure"] = pd.to_numeric(output["stockout_pressure"], errors="coerce").fillna(0.0)
    output["promo_active"] = pd.to_numeric(output.get("calendar_any_promo", 0.0), errors="coerce").fillna(0.0)
    output["discount"] = pd.to_numeric(output.get("calendar_avg_discount_value", 0.0), errors="coerce").fillna(0.0)
    output["spike_prob_proxy"] = pd.to_numeric(output.get("spike_prob", np.nan), errors="coerce")
    if output["spike_prob_proxy"].isna().all():
        output["spike_prob_proxy"] = pd.to_numeric(output["base_pred_rank_pct"], errors="coerce").fillna(0.0)
    else:
        output["spike_prob_proxy"] = output["spike_prob_proxy"].fillna(pd.to_numeric(output["base_pred_rank_pct"], errors="coerce").fillna(0.0))
    output["campaign_active"] = output.get(promo_known.CAMPAIGN_FEATURES, pd.DataFrame(index=output.index)).sum(axis=1) if all(
        col in output.columns for col in promo_known.CAMPAIGN_FEATURES
    ) else 0.0
    output["low_stock_x_base_pred"] = output["low_stock_flag"] * output["base_pred"]
    output["stock_pressure_x_base_pred"] = output["stock_pressure"] * output["base_pred"]
    output["stockout_pressure_x_base_pred"] = output["stockout_pressure"] * output["base_pred"]
    output["high_stock_x_spike_prob"] = output["high_stock_flag"] * output["spike_prob_proxy"]
    output["stock_pressure_x_promo_active"] = output["stock_pressure"] * output["promo_active"]
    output["low_stock_x_promo_active"] = output["low_stock_flag"] * output["promo_active"]
    output["restock_signal_x_promo_active"] = output["restock_signal"] * output["promo_active"]
    output["high_stock_x_discount"] = output["high_stock_flag"] * output["discount"]
    output["stock_build_up_x_campaign_active"] = output["stock_build_up"] * output["campaign_active"]
    return output


def build_current_best_validation_2022() -> pd.DataFrame:
    current_base = pd.read_csv(FINAL_MICRO_VALIDATION_PATH, parse_dates=[DATE_COL], low_memory=False)
    current_base[DATE_COL] = pd.to_datetime(current_base[DATE_COL], errors="coerce").dt.normalize()
    direct_validation = pd.read_csv(DIRECT_SEASONAL_VALIDATION_PATH, parse_dates=[DATE_COL], low_memory=False)
    direct_validation[DATE_COL] = pd.to_datetime(direct_validation[DATE_COL], errors="coerce").dt.normalize()

    target_type = "log_ratio"
    if DIRECT_SEASONAL_IMPORTANCE_PATH.exists():
        importance = pd.read_csv(DIRECT_SEASONAL_IMPORTANCE_PATH, low_memory=False)
        if "best_target_type" in importance.columns:
            non_null = importance["best_target_type"].dropna()
            if not non_null.empty:
                target_type = str(non_null.iloc[0])

    direct_subset = direct_validation.loc[
        (direct_validation["fold"] == "fold_3") & (direct_validation["target_type"] == target_type),
        [DATE_COL, "predicted_Revenue"],
    ].rename(columns={"predicted_Revenue": "direct_pred"})

    merged = current_base.merge(direct_subset, on=DATE_COL, how="inner", validate="one_to_one")
    merged["base_pred"] = 0.85 * pd.to_numeric(merged["current_base_pred"], errors="coerce") + 0.15 * pd.to_numeric(
        merged["direct_pred"], errors="coerce"
    )
    merged["actual_Revenue"] = pd.to_numeric(merged["actual_Revenue"], errors="coerce")
    merged["spike_prob"] = pd.to_numeric(merged.get("spike_prob", np.nan), errors="coerce")
    return merged[[DATE_COL, "actual_Revenue", "base_pred", "spike_prob"]].copy()


def load_reference_validation_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    current_best = build_current_best_validation_2022()
    base_metrics = compute_masked_metrics(current_best)
    rows.append({"candidate_name": "reference_current_best_analog", **base_metrics})

    current_base = pd.read_csv(FINAL_MICRO_VALIDATION_PATH, parse_dates=[DATE_COL], low_memory=False)
    current_base[DATE_COL] = pd.to_datetime(current_base[DATE_COL], errors="coerce").dt.normalize()
    current_base["base_pred"] = pd.to_numeric(current_base["current_base_pred"], errors="coerce")
    current_base["actual_Revenue"] = pd.to_numeric(current_base["actual_Revenue"], errors="coerce")
    rows.append({"candidate_name": "reference_meta_base", **compute_masked_metrics(current_base)})

    if PROMO_KNOWN_VALIDATION_PATH.exists():
        promo_frame = pd.read_csv(PROMO_KNOWN_VALIDATION_PATH, parse_dates=[DATE_COL], low_memory=False)
        promo_frame[DATE_COL] = pd.to_datetime(promo_frame[DATE_COL], errors="coerce").dt.normalize()
        promo_frame = promo_frame.loc[(promo_frame["fold"] == "fold_3") & (promo_frame[DATE_COL] >= VALID_START)].copy()
        promo_frame["base_pred"] = pd.to_numeric(promo_frame["predicted_Revenue"], errors="coerce")
        promo_frame["actual_Revenue"] = pd.to_numeric(promo_frame[TARGET_COL], errors="coerce")
        rows.append({"candidate_name": "reference_promo_known", **compute_masked_metrics(promo_frame)})

    return pd.DataFrame(rows)


def build_validation_base_frame(
    sales: pd.DataFrame,
    current_best_validation: pd.DataFrame,
    promo_features: pd.DataFrame,
    stock_features: pd.DataFrame,
) -> pd.DataFrame:
    calendar = base.build_calendar_features(current_best_validation[DATE_COL], sales[DATE_COL].min())[[DATE_COL, "month", "day_of_week"]]
    frame = current_best_validation.merge(calendar, on=DATE_COL, how="left", validate="one_to_one")
    frame = frame.merge(
        promo_features[
            [DATE_COL, "calendar_any_promo", "calendar_avg_discount_value", "campaign_intensity"] + promo_known.CAMPAIGN_FEATURES
        ],
        on=DATE_COL,
        how="left",
        validate="one_to_one",
    )
    frame = frame.merge(stock_features, on=DATE_COL, how="left", validate="one_to_one")
    frame["base_pred_log1p"] = np.log1p(np.clip(pd.to_numeric(frame["base_pred"], errors="coerce"), 0.0, None))
    frame["base_pred_rank_pct"] = pd.to_numeric(frame["base_pred"], errors="coerce").rank(pct=True)
    frame["scale_target_raw"] = pd.to_numeric(frame["actual_Revenue"], errors="coerce") / pd.to_numeric(frame["base_pred"], errors="coerce").replace(0, np.nan)
    return frame


def compute_masked_metrics(frame: pd.DataFrame) -> dict[str, float]:
    actual = pd.to_numeric(frame["actual_Revenue"], errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(frame["base_pred"], errors="coerce").to_numpy(dtype=float)
    errors = actual - pred

    def get_series(column_names: list[str], default: float = 0.0) -> pd.Series:
        for column in column_names:
            if column in frame.columns:
                return pd.to_numeric(frame[column], errors="coerce").fillna(default)
        return pd.Series(np.repeat(default, len(frame)), index=frame.index, dtype=float)

    promo = get_series(["promo_active", "calendar_any_promo"]).to_numpy(dtype=bool)
    low_stock = get_series(["low_stock_flag"]).to_numpy(dtype=bool)
    high_stock = get_series(["high_stock_flag"]).to_numpy(dtype=bool)
    top10_threshold = float(np.quantile(actual, 0.90))
    top10 = actual >= top10_threshold

    def rmse(mask: np.ndarray) -> float:
        return float(np.sqrt(np.mean(errors[mask] ** 2))) if mask.any() else np.nan

    metrics = base.evaluate_predictions(pd.Series(actual), pred)
    metrics["top10_RMSE"] = rmse(top10)
    metrics["low_stock_RMSE"] = rmse(low_stock)
    metrics["promo_low_stock_RMSE"] = rmse(promo & low_stock)
    metrics["promo_high_stock_RMSE"] = rmse(promo & high_stock)
    metrics["non_promo_RMSE"] = rmse(~promo)
    return metrics


def fit_meta_model(model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> tuple[Any, str] | None:
    if model_name == "ridge":
        try:
            from sklearn.linear_model import Ridge
        except Exception:
            return None
        model = Ridge(alpha=3.0, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        return model, "ridge"

    if model_name == "huber":
        try:
            from sklearn.linear_model import HuberRegressor
        except Exception:
            return None
        model = HuberRegressor(alpha=0.0005, epsilon=1.35, max_iter=500)
        model.fit(X_train, y_train)
        return model, "huber"

    if model_name == "lightgbm" and base.lightgbm_available():
        import lightgbm as lgb

        dataset = lgb.Dataset(X_train, label=y_train, feature_name=X_train.columns.tolist(), free_raw_data=False)
        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.03,
            "max_depth": 3,
            "num_leaves": 8,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "min_data_in_leaf": 20,
            "seed": RANDOM_STATE,
            "verbosity": -1,
            "force_col_wise": True,
        }
        model = lgb.train(params=params, train_set=dataset, num_boost_round=300)
        return model, "lightgbm"

    return None


def predict_meta_model(model: Any, model_type: str, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(X), dtype=float)


def evaluate_rule_candidate(
    validation_frame: pd.DataFrame,
    alpha: float,
    beta: float,
    pred_clip: tuple[float, float],
    candidate_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    scale = np.ones(len(validation_frame), dtype=float)
    low = validation_frame["low_stock_flag"].to_numpy(dtype=float)
    high = validation_frame["high_stock_flag"].to_numpy(dtype=float)
    stock_pressure = pd.to_numeric(validation_frame["stock_pressure"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    promo_active = pd.to_numeric(validation_frame["promo_active"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    scale = scale - alpha * low * stock_pressure
    scale = scale + beta * ((high > 0.5) & (promo_active > 0.5))
    scale = np.clip(scale, pred_clip[0], pred_clip[1])
    adjusted = np.clip(validation_frame["base_pred"].to_numpy(dtype=float) * scale, 0.0, None)

    output = validation_frame[[DATE_COL, "actual_Revenue", "base_pred", "promo_active", "low_stock_flag", "high_stock_flag"]].copy()
    output["predicted_scale"] = scale
    output["adjusted_pred"] = adjusted
    metrics = compute_candidate_metrics(output)
    metrics.update(
        {
            "candidate_name": candidate_name,
            "model_kind": "rule",
            "target_clip_min": np.nan,
            "target_clip_max": np.nan,
            "pred_clip_min": pred_clip[0],
            "pred_clip_max": pred_clip[1],
            "alpha": alpha,
            "beta": beta,
        }
    )
    return output, metrics


def evaluate_ml_candidate(
    validation_frame: pd.DataFrame,
    model_name: str,
    target_clip: tuple[float, float],
    pred_clip: tuple[float, float],
    candidate_name: str,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame] | None:
    months = sorted(validation_frame[DATE_COL].dt.to_period("M").astype(str).unique().tolist())
    predictions: list[pd.DataFrame] = []
    feature_frames: list[pd.DataFrame] = []
    fitted_any = False

    for month_label in months:
        period = pd.Period(month_label, freq="M")
        month_start = period.start_time.normalize()
        month_end = period.end_time.normalize()
        target_slice = validation_frame.loc[
            (validation_frame[DATE_COL] >= month_start) & (validation_frame[DATE_COL] <= month_end)
        ].copy()
        if target_slice.empty:
            continue

        train_slice = validation_frame.loc[validation_frame[DATE_COL] < month_start].copy()
        if train_slice.empty or len(train_slice) < 45:
            target_slice["predicted_scale"] = 1.0
            target_slice["adjusted_pred"] = target_slice["base_pred"]
            predictions.append(target_slice)
            continue

        X_train = train_slice[META_FEATURES].apply(pd.to_numeric, errors="coerce")
        X_train.columns = make_safe_feature_names(X_train.columns)
        medians = X_train.median(numeric_only=True)
        X_train = X_train.fillna(medians)
        y_train = pd.to_numeric(train_slice["scale_target_raw"], errors="coerce").clip(target_clip[0], target_clip[1]).fillna(1.0)

        fitted = fit_meta_model(model_name, X_train, y_train)
        if fitted is None:
            return None
        model, model_type = fitted
        fitted_any = True

        X_test = target_slice[META_FEATURES].apply(pd.to_numeric, errors="coerce")
        X_test.columns = X_train.columns
        X_test = X_test.fillna(medians)
        scale_pred = np.clip(predict_meta_model(model, model_type, X_test), pred_clip[0], pred_clip[1])
        target_slice["predicted_scale"] = scale_pred
        target_slice["adjusted_pred"] = np.clip(target_slice["base_pred"].to_numpy(dtype=float) * scale_pred, 0.0, None)
        predictions.append(target_slice)

        if model_type == "lightgbm":
            imp = pd.DataFrame(
                {
                    "feature": model.feature_name(),
                    "importance_gain": model.feature_importance(importance_type="gain").astype(float),
                    "importance_split": model.feature_importance(importance_type="split").astype(float),
                    "model_kind": model_name,
                    "target_clip_min": target_clip[0],
                    "target_clip_max": target_clip[1],
                    "pred_clip_min": pred_clip[0],
                    "pred_clip_max": pred_clip[1],
                    "month": month_label,
                }
            )
            feature_frames.append(imp)
        elif hasattr(model, "coef_"):
            imp = pd.DataFrame(
                {
                    "feature": X_train.columns,
                    "importance_gain": np.abs(np.asarray(model.coef_, dtype=float)),
                    "importance_split": np.nan,
                    "model_kind": model_name,
                    "target_clip_min": target_clip[0],
                    "target_clip_max": target_clip[1],
                    "pred_clip_min": pred_clip[0],
                    "pred_clip_max": pred_clip[1],
                    "month": month_label,
                }
            )
            feature_frames.append(imp)

    if not predictions:
        return None

    output = pd.concat(predictions, ignore_index=True).sort_values(DATE_COL).reset_index(drop=True)
    metrics = compute_candidate_metrics(output)
    metrics.update(
        {
            "candidate_name": candidate_name,
            "model_kind": model_name,
            "target_clip_min": target_clip[0],
            "target_clip_max": target_clip[1],
            "pred_clip_min": pred_clip[0],
            "pred_clip_max": pred_clip[1],
            "alpha": np.nan,
            "beta": np.nan,
        }
    )
    feature_output = pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame()
    if not fitted_any:
        feature_output = pd.DataFrame()
    return output, metrics, feature_output


def compute_candidate_metrics(candidate_frame: pd.DataFrame) -> dict[str, float]:
    actual = pd.to_numeric(candidate_frame["actual_Revenue"], errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(candidate_frame["adjusted_pred"], errors="coerce").to_numpy(dtype=float)
    errors = actual - pred
    promo = pd.to_numeric(candidate_frame["promo_active"], errors="coerce").fillna(0.0).to_numpy(dtype=bool)
    low_stock = pd.to_numeric(candidate_frame["low_stock_flag"], errors="coerce").fillna(0.0).to_numpy(dtype=bool)
    high_stock = pd.to_numeric(candidate_frame["high_stock_flag"], errors="coerce").fillna(0.0).to_numpy(dtype=bool)
    top10_threshold = float(np.quantile(actual, 0.90))
    top10 = actual >= top10_threshold

    def rmse(mask: np.ndarray) -> float:
        return float(np.sqrt(np.mean(errors[mask] ** 2))) if mask.any() else np.nan

    metrics = base.evaluate_predictions(pd.Series(actual), pred)
    metrics["top10_RMSE"] = rmse(top10)
    metrics["low_stock_RMSE"] = rmse(low_stock)
    metrics["promo_low_stock_RMSE"] = rmse(promo & low_stock)
    metrics["promo_high_stock_RMSE"] = rmse(promo & high_stock)
    metrics["non_promo_RMSE"] = rmse(~promo)
    return metrics


def choose_best_candidate(results: pd.DataFrame, base_metrics: dict[str, float]) -> pd.Series:
    accepted = results.copy()
    accepted["accepted"] = (
        (accepted["RMSE"] <= base_metrics["RMSE"] * 1.01)
        & (
            accepted["non_promo_RMSE"].isna()
            | pd.isna(base_metrics["non_promo_RMSE"])
            | (accepted["non_promo_RMSE"] <= base_metrics["non_promo_RMSE"] * 1.03)
        )
        & (
            accepted["low_stock_RMSE"].isna()
            | pd.isna(base_metrics["low_stock_RMSE"])
            | (accepted["low_stock_RMSE"] <= base_metrics["low_stock_RMSE"] * 1.05)
        )
    )
    valid = accepted.loc[accepted["accepted"]].sort_values(["RMSE", "top10_RMSE", "non_promo_RMSE"]).reset_index(drop=True)
    if not valid.empty:
        return valid.iloc[0]
    return accepted.sort_values(["RMSE", "non_promo_RMSE", "top10_RMSE"]).reset_index(drop=True).iloc[0]


def fit_full_candidate_model(
    validation_frame: pd.DataFrame,
    best_row: pd.Series,
) -> tuple[Any, str, pd.Series] | None:
    model_kind = str(best_row["model_kind"])
    if model_kind == "rule":
        return None

    X_train = validation_frame[META_FEATURES].apply(pd.to_numeric, errors="coerce")
    X_train.columns = make_safe_feature_names(X_train.columns)
    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)
    y_train = pd.to_numeric(validation_frame["scale_target_raw"], errors="coerce").clip(
        float(best_row["target_clip_min"]), float(best_row["target_clip_max"])
    ).fillna(1.0)

    fitted = fit_meta_model(model_kind, X_train, y_train)
    if fitted is None:
        return None
    model, model_type = fitted
    return model, model_type, medians


def apply_candidate_to_future(
    future_frame: pd.DataFrame,
    best_row: pd.Series,
    fitted: tuple[Any, str, pd.Series] | None,
    clip_override: tuple[float, float] | None = None,
) -> pd.DataFrame:
    pred_clip = clip_override or (float(best_row["pred_clip_min"]), float(best_row["pred_clip_max"]))
    output = future_frame.copy()

    if str(best_row["model_kind"]) == "rule":
        alpha = float(best_row["alpha"])
        beta = float(best_row["beta"])
        scale = np.ones(len(output), dtype=float)
        scale = scale - alpha * output["low_stock_flag"].to_numpy(dtype=float) * output["stock_pressure"].to_numpy(dtype=float)
        scale = scale + beta * (
            (output["high_stock_flag"].to_numpy(dtype=float) > 0.5) & (output["promo_active"].to_numpy(dtype=float) > 0.5)
        )
        scale = np.clip(scale, pred_clip[0], pred_clip[1])
    else:
        assert fitted is not None
        model, model_type, medians = fitted
        X_future = output[META_FEATURES].apply(pd.to_numeric, errors="coerce")
        X_future.columns = make_safe_feature_names(X_future.columns)
        X_future = X_future.fillna(medians)
        scale = np.clip(predict_meta_model(model, model_type, X_future), pred_clip[0], pred_clip[1])

    output["predicted_scale"] = scale
    output["adjusted_pred"] = np.clip(output["base_pred"].to_numpy(dtype=float) * scale, 0.0, None)
    return output


def build_submission(dates: pd.Series, revenue: pd.Series | np.ndarray, ratio: float = 0.8900) -> pd.DataFrame:
    revenue_series = pd.Series(np.asarray(revenue, dtype=float))
    return pd.DataFrame(
        {
            DATE_COL: pd.to_datetime(dates).reset_index(drop=True),
            TARGET_COL: revenue_series.clip(lower=0.0),
            COGS_COL: revenue_series.clip(lower=0.0) * ratio,
        }
    )


def run() -> None:
    logger = setup_logging()
    reporter = Reporter(logger)
    reporter.emit("Stock-Aware Scaling Layer")
    reporter.emit("=========================")
    reporter.emit("")

    sales = load_sales()
    sample_submission = load_sample_submission()
    current_best_submission = load_current_best_submission()
    snapshots = load_inventory_snapshot_features()

    reporter.emit("1. Build stock features and validation base")
    historical_promo_features = promo_known.build_daily_promo_known_features(sales[DATE_COL], promo_known.load_promotions(PROMOTIONS_PATH))
    current_best_validation = build_current_best_validation_2022()

    low_threshold = float(
        np.nanpercentile(
            pd.to_numeric(snapshots.loc[snapshots[DATE_COL] <= TRAIN_END, "inv_avg_days_of_supply"], errors="coerce").dropna(),
            25,
        )
    )
    high_threshold = float(
        np.nanpercentile(
            pd.to_numeric(snapshots.loc[snapshots[DATE_COL] <= TRAIN_END, "inv_avg_days_of_supply"], errors="coerce").dropna(),
            75,
        )
    )

    validation_scenarios: dict[str, pd.DataFrame] = {}
    for scenario in ["carry", "monthavg", "hybrid"]:
        stock_context = build_daily_stock_context(
            current_best_validation[DATE_COL],
            snapshots,
            scenario=scenario,
            dynamic_updates=True,
            known_cutoff=TRAIN_END,
        )
        frame = build_validation_base_frame(sales, current_best_validation, historical_promo_features, stock_context)
        frame = add_stock_flags_and_interactions(frame, low_threshold, high_threshold)
        validation_scenarios[scenario] = frame

    base_metrics_frame = validation_scenarios["carry"][
        [DATE_COL, "actual_Revenue", "base_pred", "promo_active", "low_stock_flag", "high_stock_flag"]
    ].copy()
    base_metrics = compute_masked_metrics(base_metrics_frame)
    reporter.emit_frame("Reference validation rows:", load_reference_validation_rows())

    reporter.emit("")
    reporter.emit("2. Search stock-aware scaling candidates")
    results_rows: list[dict[str, Any]] = []
    candidate_predictions: dict[str, pd.DataFrame] = {}
    feature_importance_frames: list[pd.DataFrame] = []

    for scenario_name, validation_frame in validation_scenarios.items():
        # Rule-based candidates
        for alpha in [0.01, 0.02, 0.03, 0.05, 0.08]:
            for beta in [0.00, 0.01, 0.02, 0.03]:
                for pred_clip in PREDICTION_CLIPS:
                    candidate_name = f"rule_{scenario_name}_a{alpha:.2f}_b{beta:.2f}_clip_{pred_clip[0]:.2f}_{pred_clip[1]:.2f}"
                    output, metrics = evaluate_rule_candidate(validation_frame, alpha, beta, pred_clip, candidate_name)
                    metrics["scenario"] = scenario_name
                    results_rows.append(metrics)
                    candidate_predictions[candidate_name] = output

        # ML candidates
        for model_name in ["ridge", "huber", "lightgbm"]:
            for target_clip in TARGET_CLIPS:
                for pred_clip in PREDICTION_CLIPS:
                    candidate_name = (
                        f"{model_name}_{scenario_name}_target_{target_clip[0]:.2f}_{target_clip[1]:.2f}"
                        f"_pred_{pred_clip[0]:.2f}_{pred_clip[1]:.2f}"
                    )
                    result = evaluate_ml_candidate(validation_frame, model_name, target_clip, pred_clip, candidate_name)
                    if result is None:
                        continue
                    output, metrics, importance = result
                    metrics["scenario"] = scenario_name
                    results_rows.append(metrics)
                    candidate_predictions[candidate_name] = output
                    if not importance.empty:
                        importance["candidate_name"] = candidate_name
                        importance["scenario"] = scenario_name
                        feature_importance_frames.append(importance)

    results = pd.DataFrame(results_rows)
    best_row = choose_best_candidate(results, base_metrics)
    results["is_best"] = results["candidate_name"] == best_row["candidate_name"]
    results.to_csv(RESULTS_PATH, index=False)

    best_predictions = candidate_predictions[str(best_row["candidate_name"])].copy()
    best_predictions["candidate_name"] = str(best_row["candidate_name"])
    best_predictions.to_csv(VALIDATION_OUTPUT_PATH, index=False, date_format="%Y-%m-%d")

    if feature_importance_frames:
        feature_importance = pd.concat(feature_importance_frames, ignore_index=True)
        feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    else:
        feature_importance = pd.DataFrame(columns=["feature", "importance_gain", "importance_split", "candidate_name", "scenario"])
        feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    reporter.emit_frame(
        "Best stock-aware candidates:",
        results.sort_values("RMSE").head(20)[
            [
                "candidate_name",
                "scenario",
                "model_kind",
                "RMSE",
                "MAE",
                "R2",
                "top10_RMSE",
                "low_stock_RMSE",
                "promo_low_stock_RMSE",
                "non_promo_RMSE",
            ]
        ],
    )

    reporter.emit("")
    reporter.emit("3. Apply best stock-aware model to future stock scenarios")
    future_promo_features: pd.DataFrame
    if FUTURE_PROMO_KNOWN_PATH.exists():
        future_promo_features = pd.read_csv(FUTURE_PROMO_KNOWN_PATH, parse_dates=[DATE_COL], low_memory=False)
        future_promo_features[DATE_COL] = pd.to_datetime(future_promo_features[DATE_COL], errors="coerce").dt.normalize()
    else:
        synthetic, _ = promo_known.promo_builder.build_synthetic_promotions(promo_known.load_promotions(PROMOTIONS_PATH))
        future_promo_features = promo_known.build_daily_promo_known_features(sample_submission[DATE_COL], synthetic)

    promo_keep = [DATE_COL, "calendar_any_promo", "calendar_avg_discount_value", "campaign_intensity"] + promo_known.CAMPAIGN_FEATURES
    future_base = current_best_submission[[DATE_COL, TARGET_COL]].rename(columns={TARGET_COL: "base_pred"}).copy()
    future_base["base_pred_log1p"] = np.log1p(np.clip(pd.to_numeric(future_base["base_pred"], errors="coerce"), 0.0, None))
    future_base["base_pred_rank_pct"] = pd.to_numeric(future_base["base_pred"], errors="coerce").rank(pct=True)
    future_base["month"] = future_base[DATE_COL].dt.month.astype(int)
    future_base["day_of_week"] = future_base[DATE_COL].dt.dayofweek.astype(int)

    future_outputs: dict[str, pd.DataFrame] = {}
    best_fitted = fit_full_candidate_model(validation_scenarios[str(best_row["scenario"])], best_row)
    for scenario in ["carry", "monthavg", "hybrid"]:
        stock_context = build_daily_stock_context(
            sample_submission[DATE_COL],
            snapshots,
            scenario=scenario,
            dynamic_updates=False,
            known_cutoff=sales[DATE_COL].max(),
        )
        future_frame = future_base.merge(
            future_promo_features[promo_keep],
            on=DATE_COL,
            how="left",
            validate="one_to_one",
        ).merge(
            stock_context,
            on=DATE_COL,
            how="left",
            validate="one_to_one",
        ).fillna(0.0)
        future_frame = add_stock_flags_and_interactions(future_frame, low_threshold, high_threshold)
        adjusted = apply_candidate_to_future(future_frame, best_row, best_fitted)
        future_outputs[scenario] = adjusted

    submission_carry = build_submission(sample_submission[DATE_COL], future_outputs["carry"]["adjusted_pred"], ratio=0.8900)
    submission_monthavg = build_submission(sample_submission[DATE_COL], future_outputs["monthavg"]["adjusted_pred"], ratio=0.8900)
    submission_hybrid = build_submission(sample_submission[DATE_COL], future_outputs["hybrid"]["adjusted_pred"], ratio=0.8900)
    validate_submission_frame(submission_carry, sample_submission)
    validate_submission_frame(submission_monthavg, sample_submission)
    validate_submission_frame(submission_hybrid, sample_submission)
    submission_carry.to_csv(SUBMISSION_CARRY_PATH, index=False, date_format="%Y-%m-%d")
    submission_monthavg.to_csv(SUBMISSION_MONTHAVG_PATH, index=False, date_format="%Y-%m-%d")
    submission_hybrid.to_csv(SUBMISSION_HYBRID_PATH, index=False, date_format="%Y-%m-%d")

    conservative_clip = (0.97, 1.05)
    aggressive_clip = (0.90, 1.15)
    best_scenario_name = str(best_row["scenario"])
    conservative_output = apply_candidate_to_future(future_outputs[best_scenario_name].copy(), best_row, best_fitted, clip_override=conservative_clip)
    aggressive_output = apply_candidate_to_future(future_outputs[best_scenario_name].copy(), best_row, best_fitted, clip_override=aggressive_clip)
    submission_conservative = build_submission(sample_submission[DATE_COL], conservative_output["adjusted_pred"], ratio=0.8900)
    submission_aggressive = build_submission(sample_submission[DATE_COL], aggressive_output["adjusted_pred"], ratio=0.8900)
    validate_submission_frame(submission_conservative, sample_submission)
    validate_submission_frame(submission_aggressive, sample_submission)
    submission_conservative.to_csv(SUBMISSION_CONSERVATIVE_PATH, index=False, date_format="%Y-%m-%d")
    submission_aggressive.to_csv(SUBMISSION_AGGRESSIVE_PATH, index=False, date_format="%Y-%m-%d")

    reporter.emit("")
    reporter.emit("4. Create blends with current best")
    best_future_submission = {
        "carry": submission_carry,
        "monthavg": submission_monthavg,
        "hybrid": submission_hybrid,
    }[best_scenario_name]
    created_files = [
        str(SUBMISSION_CARRY_PATH),
        str(SUBMISSION_MONTHAVG_PATH),
        str(SUBMISSION_HYBRID_PATH),
        str(SUBMISSION_CONSERVATIVE_PATH),
        str(SUBMISSION_AGGRESSIVE_PATH),
    ]
    for weight, output_path in BLEND_OUTPUTS.items():
        revenue = (1.0 - weight) * pd.to_numeric(current_best_submission[TARGET_COL], errors="coerce") + weight * pd.to_numeric(
            best_future_submission[TARGET_COL], errors="coerce"
        )
        submission = build_submission(sample_submission[DATE_COL], revenue, ratio=0.8900)
        validate_submission_frame(submission, sample_submission)
        submission.to_csv(output_path, index=False, date_format="%Y-%m-%d")
        created_files.append(str(output_path))

    reporter.emit("")
    reporter.emit("5. Final summary")
    reporter.emit(
        f"Best stock-aware model: {best_row['model_kind']} | candidate={best_row['candidate_name']}"
    )
    reporter.emit(f"Best stock scenario: {best_scenario_name}")
    reporter.emit(
        f"Validation RMSE before/after: {base_metrics['RMSE']:,.2f} -> {best_row['RMSE']:,.2f}"
    )
    reporter.emit(
        f"Low-stock RMSE before/after: {base_metrics['low_stock_RMSE']:,.2f} -> {best_row['low_stock_RMSE']:,.2f}"
    )
    reporter.emit(
        f"Promo + low-stock RMSE before/after: {base_metrics['promo_low_stock_RMSE']:,.2f} -> {best_row['promo_low_stock_RMSE']:,.2f}"
    )

    if not feature_importance.empty:
        feature_summary = (
            feature_importance.loc[feature_importance["candidate_name"] == best_row["candidate_name"]]
            .groupby("feature", as_index=False)
            .agg(importance_gain=("importance_gain", "mean"))
            .sort_values("importance_gain", ascending=False)
            .head(20)
        )
    else:
        feature_summary = pd.DataFrame(
            {
                "feature": [
                    "stock_pressure",
                    "low_stock_flag",
                    "stockout_pressure",
                    "restock_signal",
                    "stock_build_up",
                    "promo_active",
                ],
                "importance_gain": np.nan,
            }
        )
    reporter.emit_frame("Top 20 stock features:", feature_summary)

    future_best = future_outputs[best_scenario_name]
    reporter.emit(
        "Future scale mean/min/max: "
        f"{future_best['predicted_scale'].mean():.6f} / {future_best['predicted_scale'].min():.6f} / {future_best['predicted_scale'].max():.6f}"
    )
    reporter.emit(f"Created submission files: {', '.join(created_files)}")
    reporter.emit(
        "Recommended upload order: "
        "submission_stock_blend_05.csv, submission_stock_scale_hybrid.csv, "
        "submission_stock_blend_10.csv, submission_stock_scale_carry.csv, "
        "submission_stock_scale_conservative.csv, submission_stock_scale_aggressive.csv"
    )
    reporter.emit(
        "Leakage safety confirmation: the stock-aware layer never retrains the base forecaster and only uses "
        "inventory snapshots known at or before each prediction date for validation, plus future stock scenarios "
        "derived from historical inventory snapshots. No future actual Revenue/COGS, no same-day realized demand, "
        "and no external data are used."
    )
    reporter.save()


if __name__ == "__main__":
    run()
