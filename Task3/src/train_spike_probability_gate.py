from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import train_final_model as base
import train_orders_model as orders_base
import train_revenue_with_predicted_orders as pred_orders_mod
import train_spike_aware_model as spike1
import train_spike_v2_model as spike2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "log"

TRAIN_DATA_PATH = DATA_DIR / "daily_feature_table.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
BASE_SUBMISSION_PATH = DATA_DIR / "submission_regime_ultra_15.csv"
FUTURE_PROMO_FEATURES_PATH = DATA_DIR / "future_promo_calendar_features.csv"
PROMOTIONS_PATH = DATA_DIR / "promotions.csv"
SYNTHETIC_PROMOTIONS_PATH = DATA_DIR / "synthetic_promotions_2023_2024.csv"

PRUNED_VALIDATION_PATH = DATA_DIR / "pruned_ensemble_validation_predictions.csv"
SPIKE_VALIDATION_PATH = DATA_DIR / "spike_model_validation_predictions.csv"
REGIME_VALIDATION_PATH = DATA_DIR / "promo_regime_validation_predictions.csv"

VALIDATION_PREDICTIONS_PATH = DATA_DIR / "spike_gate_validation_predictions.csv"
SEARCH_RESULTS_PATH = DATA_DIR / "spike_gate_search_results.csv"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "spike_gate_feature_importance.csv"

SUBMISSION_BEST_PATH = DATA_DIR / "submission_spike_gate_best.csv"
SUBMISSION_SOFT_PATH = DATA_DIR / "submission_spike_gate_soft.csv"
SUBMISSION_HARD_PATH = DATA_DIR / "submission_spike_gate_hard.csv"
SUBMISSION_CONSERVATIVE_PATH = DATA_DIR / "submission_spike_gate_conservative.csv"
SUBMISSION_AGGRESSIVE_PATH = DATA_DIR / "submission_spike_gate_aggressive.csv"

REPORT_PATH = LOG_DIR / "spike_gate_report.txt"
LOG_FILE = LOG_DIR / "train_spike_probability_gate.log"

DATE_COL = "Date"
TARGET_COL = "Revenue"
COGS_COL = "COGS"
TRAIN_CUTOFF = pd.Timestamp("2022-01-01")
VALIDATION_END = pd.Timestamp("2022-12-31")
RANDOM_STATE = 42

LABEL_SPECS = {
    "top10": 0.90,
    "top15": 0.85,
}
GATING_THRESHOLDS = [0.40, 0.50, 0.60, 0.70]
UPLIFTS = [0.03, 0.05, 0.07, 0.10, 0.12, 0.15]
GATING_MODES = ["hard", "soft"]

CAMPAIGN_FLAG_MAP = {
    "spring sale": "is_spring_sale",
    "mid-year sale": "is_midyear_sale",
    "fall launch": "is_fall_launch",
    "year-end sale": "is_year_end_sale",
    "urban blowout": "is_urban_blowout",
    "rural special": "is_rural_special",
}
CAMPAIGN_FLAG_COLUMNS = list(CAMPAIGN_FLAG_MAP.values())

CALENDAR_FEATURES = [
    "day_of_year",
    "week_of_year",
    "month",
    "day_of_week",
    "is_weekend",
    "is_month_start",
    "is_month_end",
]

PROMO_FEATURES = [
    "calendar_any_promo",
    "calendar_active_promo_count",
    "calendar_avg_discount_value",
    "calendar_max_discount_value",
    "calendar_stackable_promo_count",
    "promotion_campaign_index",
    "promo_progress_ratio",
    "promo_days_remaining",
    "promo_duration",
] + CAMPAIGN_FLAG_COLUMNS

MEMORY_FEATURES = [
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
]

SPIKE_PRECURSOR_FEATURES = [
    "lag7_to_roll30_ratio",
    "lag30_to_roll90_ratio",
    "lag365_to_roll365_ratio",
    "volatility_30",
    "volatility_90",
    "volatility_365",
    "spike_strength_365",
    "lag365_above_p90",
    "lag7_above_p90",
]

INTERACTION_FEATURES = [
    "promo_x_revenue_lag_365",
    "discount_x_revenue_lag_365",
    "promo_x_lag7_to_roll30_ratio",
    "campaign_index_x_day_of_year",
    "lag365_above_p90_x_calendar_any_promo",
]

CLASSIFIER_FEATURES = (
    CALENDAR_FEATURES
    + PROMO_FEATURES
    + MEMORY_FEATURES
    + SPIKE_PRECURSOR_FEATURES
    + INTERACTION_FEATURES
    + base.INVENTORY_FEATURES
)


class Reporter:
    """Print, log, and persist the run report."""

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

    def save(self, path: Path = REPORT_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        self.logger.info("Saved report to %s", path)


def setup_logging(log_file: Path = LOG_FILE) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train_spike_probability_gate")
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


def normalize_campaign_name(name: Any) -> str:
    text = str(name).strip()
    text = re.sub(r"\s+\d{4}$", "", text)
    return text.lower().strip()


def load_promotions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["start_date", "end_date", "promo_name_base"])

    promotions = pd.read_csv(path, low_memory=False)
    promotions["start_date"] = pd.to_datetime(promotions["start_date"], errors="coerce").dt.normalize()
    promotions["end_date"] = pd.to_datetime(promotions["end_date"], errors="coerce").dt.normalize()
    promotions["promo_name_base"] = promotions.get("promo_name", "").map(normalize_campaign_name)
    promotions = promotions.dropna(subset=["start_date", "end_date"]).copy()
    return promotions


def load_synthetic_promotions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["start_date", "end_date", "promo_name_base"])

    promotions = pd.read_csv(path, low_memory=False)
    promotions["start_date"] = pd.to_datetime(promotions["start_date"], errors="coerce").dt.normalize()
    promotions["end_date"] = pd.to_datetime(promotions["end_date"], errors="coerce").dt.normalize()
    if "promo_name_base" not in promotions.columns:
        promotions["promo_name_base"] = promotions.get("promo_name", "").map(normalize_campaign_name)
    else:
        promotions["promo_name_base"] = promotions["promo_name_base"].map(normalize_campaign_name)
    promotions = promotions.dropna(subset=["start_date", "end_date"]).copy()
    return promotions


def build_campaign_type_daily_features(dates: pd.Series, promotions: pd.DataFrame) -> pd.DataFrame:
    calendar = pd.DataFrame({DATE_COL: pd.to_datetime(dates).sort_values().unique()})
    for column in CAMPAIGN_FLAG_COLUMNS:
        calendar[column] = 0.0

    if promotions.empty:
        return calendar

    min_date = calendar[DATE_COL].min()
    max_date = calendar[DATE_COL].max()
    rows: list[dict[str, Any]] = []

    for row in promotions.itertuples(index=False):
        campaign_flag = CAMPAIGN_FLAG_MAP.get(getattr(row, "promo_name_base", ""))
        if campaign_flag is None:
            continue

        active_start = max(row.start_date, min_date)
        active_end = min(row.end_date, max_date)
        if active_start > active_end:
            continue

        for active_date in pd.date_range(active_start, active_end, freq="D"):
            rows.append({DATE_COL: active_date, campaign_flag: 1.0})

    if not rows:
        return calendar

    expanded = pd.DataFrame(rows)
    aggregated = expanded.groupby(DATE_COL, as_index=False).max()
    return calendar.merge(aggregated, on=DATE_COL, how="left", suffixes=("", "_agg")).fillna(0.0)[
        [DATE_COL] + CAMPAIGN_FLAG_COLUMNS
    ]


def build_historical_promo_context(dates: pd.Series, logger: logging.Logger) -> pd.DataFrame:
    promo_calendar = base.build_promotion_calendar(dates, PROMOTIONS_PATH, logger)
    promo_phase = orders_base.build_promotion_features(dates, PROMOTIONS_PATH, logger)[
        [DATE_COL, "promo_days_remaining", "promo_duration", "promo_progress_ratio", "promotion_campaign_index"]
    ]
    campaign_flags = build_campaign_type_daily_features(dates, load_promotions(PROMOTIONS_PATH))
    return (
        promo_calendar.merge(promo_phase, on=DATE_COL, how="left", validate="one_to_one")
        .merge(campaign_flags, on=DATE_COL, how="left", validate="one_to_one")
        .fillna(0.0)
    )


def build_future_promo_context(dates: pd.Series) -> pd.DataFrame:
    calendar = pd.DataFrame({DATE_COL: pd.to_datetime(dates).sort_values().unique()})
    for column in [
        "calendar_any_promo",
        "calendar_active_promo_count",
        "calendar_avg_discount_value",
        "calendar_max_discount_value",
        "calendar_stackable_promo_count",
        "promotion_campaign_index",
        "promo_progress_ratio",
        "promo_days_remaining",
        "promo_duration",
    ] + CAMPAIGN_FLAG_COLUMNS:
        calendar[column] = 0.0

    if FUTURE_PROMO_FEATURES_PATH.exists():
        future = pd.read_csv(FUTURE_PROMO_FEATURES_PATH, parse_dates=[DATE_COL], low_memory=False)
        future[DATE_COL] = pd.to_datetime(future[DATE_COL], errors="coerce").dt.normalize()
        rename_map = {
            "future_calendar_any_promo": "calendar_any_promo",
            "future_calendar_active_promo_count": "calendar_active_promo_count",
            "future_calendar_avg_discount_value": "calendar_avg_discount_value",
            "future_calendar_max_discount_value": "calendar_max_discount_value",
            "future_calendar_stackable_promo_count": "calendar_stackable_promo_count",
            "future_promotion_campaign_index": "promotion_campaign_index",
            "future_promo_avg_progress_ratio": "promo_progress_ratio",
            "future_promo_avg_days_remaining": "promo_days_remaining",
            "future_promo_avg_duration_days": "promo_duration",
        }
        future = future.rename(columns=rename_map)
        keep = [DATE_COL] + list(rename_map.values())
        future = future[keep].copy()
        calendar = calendar.drop(columns=keep[1:]).merge(future, on=DATE_COL, how="left", validate="one_to_one")
        calendar[keep[1:]] = calendar[keep[1:]].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    campaign_flags = build_campaign_type_daily_features(dates, load_synthetic_promotions(SYNTHETIC_PROMOTIONS_PATH))
    calendar = calendar.drop(columns=CAMPAIGN_FLAG_COLUMNS).merge(
        campaign_flags,
        on=DATE_COL,
        how="left",
        validate="one_to_one",
    )
    calendar[CAMPAIGN_FLAG_COLUMNS] = calendar[CAMPAIGN_FLAG_COLUMNS].fillna(0.0)
    return calendar


def build_static_features_historical(dates: pd.Series, min_date: pd.Timestamp, logger: logging.Logger) -> pd.DataFrame:
    calendar = base.build_calendar_features(dates, min_date)
    promo = build_historical_promo_context(dates, logger)
    inventory = base.build_inventory_asof_features(dates, base.INVENTORY_PATH, logger)
    keep_calendar = [DATE_COL] + CALENDAR_FEATURES
    return (
        calendar[keep_calendar]
        .merge(promo, on=DATE_COL, how="left", validate="one_to_one")
        .merge(inventory, on=DATE_COL, how="left", validate="one_to_one")
        .fillna(0.0)
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )


def build_static_features_future(dates: pd.Series, min_date: pd.Timestamp, logger: logging.Logger) -> pd.DataFrame:
    calendar = base.build_calendar_features(dates, min_date)
    promo = build_future_promo_context(dates)
    inventory = base.build_inventory_asof_features(dates, base.INVENTORY_PATH, logger)
    keep_calendar = [DATE_COL] + CALENDAR_FEATURES
    return (
        calendar[keep_calendar]
        .merge(promo, on=DATE_COL, how="left", validate="one_to_one")
        .merge(inventory, on=DATE_COL, how="left", validate="one_to_one")
        .fillna(0.0)
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )


def add_classifier_interactions_df(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["promo_x_revenue_lag_365"] = output["calendar_any_promo"] * output["revenue_lag_365"]
    output["discount_x_revenue_lag_365"] = output["calendar_avg_discount_value"] * output["revenue_lag_365"]
    output["promo_x_lag7_to_roll30_ratio"] = output["calendar_any_promo"] * output["lag7_to_roll30_ratio"]
    output["campaign_index_x_day_of_year"] = output["promotion_campaign_index"] * output["day_of_year"]
    output["lag365_above_p90_x_calendar_any_promo"] = output["lag365_above_p90"] * output["calendar_any_promo"]
    return output


def add_classifier_interactions_row(row: dict[str, float]) -> dict[str, float]:
    return {
        "promo_x_revenue_lag_365": row.get("calendar_any_promo", 0.0) * row.get("revenue_lag_365", np.nan),
        "discount_x_revenue_lag_365": row.get("calendar_avg_discount_value", 0.0) * row.get("revenue_lag_365", np.nan),
        "promo_x_lag7_to_roll30_ratio": row.get("calendar_any_promo", 0.0) * row.get("lag7_to_roll30_ratio", np.nan),
        "campaign_index_x_day_of_year": row.get("promotion_campaign_index", 0.0) * row.get("day_of_year", np.nan),
        "lag365_above_p90_x_calendar_any_promo": row.get("lag365_above_p90", 0.0) * row.get("calendar_any_promo", 0.0),
    }


def build_classifier_table(train_df: pd.DataFrame, static_features: pd.DataFrame) -> pd.DataFrame:
    table = base.build_historical_model_table(train_df, static_features, include_business_lag365=False)
    table = spike2.add_historical_spike_v2_features(table)
    table = add_classifier_interactions_df(table)
    return table


def load_validation_predictions(path: Path, expected_dates: pd.Series) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Validation prediction file missing: {path}")

    df = pd.read_csv(path, parse_dates=[DATE_COL], low_memory=False)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce").dt.normalize()
    if "predicted_Revenue" not in df.columns or "actual_Revenue" not in df.columns:
        raise ValueError(f"Validation file missing required columns: {path}")

    aligned = pd.DataFrame({DATE_COL: pd.to_datetime(expected_dates).reset_index(drop=True)}).merge(
        df[[DATE_COL, "actual_Revenue", "predicted_Revenue"]],
        on=DATE_COL,
        how="left",
        validate="one_to_one",
    )
    if aligned[["actual_Revenue", "predicted_Revenue"]].isna().any().any():
        raise ValueError(f"Missing aligned rows in validation file: {path}")
    return aligned


def reconstruct_base_validation(validation_dates: pd.Series) -> pd.DataFrame:
    pruned = load_validation_predictions(PRUNED_VALIDATION_PATH, validation_dates)
    spike = load_validation_predictions(SPIKE_VALIDATION_PATH, validation_dates)
    regime = load_validation_predictions(REGIME_VALIDATION_PATH, validation_dates)

    actual = pruned["actual_Revenue"].to_numpy(dtype=float)
    if not np.allclose(actual, spike["actual_Revenue"].to_numpy(dtype=float)) or not np.allclose(
        actual,
        regime["actual_Revenue"].to_numpy(dtype=float),
    ):
        raise ValueError("Validation actuals do not match across component files.")

    base_pred = (
        0.425 * pruned["predicted_Revenue"].to_numpy(dtype=float)
        + 0.425 * spike["predicted_Revenue"].to_numpy(dtype=float)
        + 0.15 * regime["predicted_Revenue"].to_numpy(dtype=float)
    )
    return pd.DataFrame(
        {
            DATE_COL: pd.to_datetime(validation_dates).reset_index(drop=True),
            "actual_Revenue": actual,
            "base_pred": base_pred,
        }
    )


def train_classifier_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    reporter: Reporter,
) -> Any:
    import lightgbm as lgb

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "feature_fraction": 0.9,
        "is_unbalance": True,
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
    model = lgb.train(params=params, train_set=train_data, num_boost_round=800)
    reporter.logger.info("Trained LightGBM classifier rows=%s features=%s", len(X_train), X_train.shape[1])
    return model


def train_classifier_fallback(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[Any, str]:
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier

        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=6,
            max_iter=500,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train)
        return model, "hist_gradient_boosting_classifier"
    except Exception:
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        return model, "random_forest_classifier"


def train_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    reporter: Reporter,
) -> tuple[Any, str]:
    if base.lightgbm_available():
        return train_classifier_lightgbm(X_train, y_train, reporter), "lightgbm_classifier"
    reporter.emit("LightGBM unavailable; using classifier fallback")
    return train_classifier_fallback(X_train, y_train)


def predict_classifier_proba(model: Any, model_type: str, X: pd.DataFrame) -> np.ndarray:
    if model_type == "lightgbm_classifier":
        return np.asarray(model.predict(X), dtype=float)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def get_classifier_feature_importance(
    model: Any,
    model_type: str,
    feature_columns: list[str],
    X_ref: pd.DataFrame,
    y_ref: pd.Series,
) -> pd.DataFrame:
    if model_type == "lightgbm_classifier":
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
                    "importance_gain": np.asarray(model.feature_importances_, dtype=float),
                }
            )
            .sort_values("importance_gain", ascending=False)
            .reset_index(drop=True)
        )

    try:
        from sklearn.inspection import permutation_importance

        scorer = "roc_auc" if y_ref.nunique() > 1 else None
        perm = permutation_importance(
            model,
            X_ref,
            y_ref,
            n_repeats=5,
            random_state=RANDOM_STATE,
            scoring=scorer,
        )
        return (
            pd.DataFrame(
                {
                    "feature": feature_columns,
                    "importance_split": np.nan,
                    "importance_gain": perm.importances_mean,
                }
            )
            .sort_values("importance_gain", ascending=False)
            .reset_index(drop=True)
        )
    except Exception:
        return pd.DataFrame(
            {
                "feature": feature_columns,
                "importance_split": np.nan,
                "importance_gain": np.nan,
            }
        )


def compute_binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    actual = np.asarray(y_true, dtype=int)
    predicted_scores = np.asarray(scores, dtype=float)

    positive_mask = actual == 1
    negative_mask = actual == 0
    positive_count = int(np.sum(positive_mask))
    negative_count = int(np.sum(negative_mask))
    if positive_count == 0 or negative_count == 0:
        return np.nan

    order = np.argsort(predicted_scores)
    ranks = np.empty_like(order, dtype=float)
    sorted_scores = predicted_scores[order]

    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    positive_rank_sum = float(np.sum(ranks[positive_mask]))
    auc = (positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / (positive_count * negative_count)
    return float(auc)


def evaluate_classifier_metrics(y_true: pd.Series, probs: np.ndarray, threshold: float = 0.50) -> dict[str, Any]:
    actual = pd.Series(y_true).astype(int).to_numpy()
    predicted = (np.asarray(probs, dtype=float) >= threshold).astype(int)

    tn = int(np.sum((actual == 0) & (predicted == 0)))
    fp = int(np.sum((actual == 0) & (predicted == 1)))
    fn = int(np.sum((actual == 1) & (predicted == 0)))
    tp = int(np.sum((actual == 1) & (predicted == 1)))

    auc = compute_binary_auc(actual, np.asarray(probs, dtype=float))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "AUC": auc,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def build_train_classifier_matrix(
    classifier_table: pd.DataFrame,
    feature_columns: list[str],
    label_threshold: float,
    train_end_exclusive: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    table = classifier_table[classifier_table[DATE_COL] < train_end_exclusive].copy()
    table["label"] = (pd.to_numeric(table[TARGET_COL], errors="coerce") >= label_threshold).astype(int)
    clean = table.dropna(subset=feature_columns + ["label"]).reset_index(drop=True)
    X_train = clean[feature_columns].copy()
    y_train = clean["label"].copy()
    feature_medians = X_train.median(numeric_only=True)
    return X_train, y_train, clean, feature_medians


def build_recursive_classifier_features(
    prediction_dates: pd.Series,
    static_features: pd.DataFrame,
    initial_revenue_history: pd.Series,
    recursive_revenue_source: pd.Series,
    thresholds_bundle: dict[str, float],
    feature_columns: list[str],
) -> pd.DataFrame:
    static_by_date = static_features.set_index(DATE_COL).sort_index()
    history = pd.to_numeric(initial_revenue_history, errors="coerce").sort_index().copy()
    revenue_source = pd.to_numeric(recursive_revenue_source, errors="coerce").sort_index()
    rows: list[dict[str, Any]] = []

    for forecast_date in pd.to_datetime(prediction_dates):
        if forecast_date not in static_by_date.index:
            raise ValueError(f"Missing static features for {forecast_date.date()}")
        if forecast_date not in revenue_source.index:
            raise ValueError(f"Missing recursive revenue source for {forecast_date.date()}")

        row: dict[str, float] = static_by_date.loc[forecast_date].to_dict()
        row.update(base.compute_revenue_features_from_history(history, forecast_date))
        row.update(spike2.compute_spike_v2_features_from_row(row, thresholds_bundle))
        row.update(add_classifier_interactions_row(row))

        rows.append({DATE_COL: forecast_date, **{feature: row.get(feature, np.nan) for feature in feature_columns}})
        history.loc[forecast_date] = float(revenue_source.loc[forecast_date])

    return pd.DataFrame(rows)


def compute_spike_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    actual_values = actual.to_numpy(dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    error = actual_values - predicted_values

    top10_threshold = float(np.quantile(actual_values, 0.90))
    top5_threshold = float(np.quantile(actual_values, 0.95))
    top10_mask = actual_values >= top10_threshold
    top5_mask = actual_values >= top5_threshold
    non_spike_mask = actual_values < top10_threshold

    def masked_rmse(mask: np.ndarray) -> float:
        return float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan

    return {
        "top10_RMSE": masked_rmse(top10_mask),
        "top10_underprediction": int(np.sum(error[top10_mask] > 0)) if top10_mask.any() else 0,
        "top10_count": int(np.sum(top10_mask)),
        "top5_RMSE": masked_rmse(top5_mask),
        "top5_underprediction": int(np.sum(error[top5_mask] > 0)) if top5_mask.any() else 0,
        "top5_count": int(np.sum(top5_mask)),
        "non_spike_RMSE": masked_rmse(non_spike_mask),
    }


def gate_values(probabilities: np.ndarray, mode: str, threshold: float) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    if mode == "hard":
        return (probs >= threshold).astype(float)
    return np.where(probs >= threshold, probs, 0.0)


def evaluate_gating_config(
    actual: pd.Series,
    base_pred: np.ndarray,
    probs: np.ndarray,
    label_name: str,
    mode: str,
    threshold: float,
    uplift: float,
    base_metrics: dict[str, float],
) -> dict[str, Any]:
    gate = gate_values(probs, mode, threshold)
    factor = 1.0 + uplift * gate
    adjusted_pred = np.asarray(base_pred, dtype=float) * factor

    overall = base.evaluate_predictions(actual, adjusted_pred)
    spike_metrics = compute_spike_metrics(actual, adjusted_pred)

    return {
        "label_name": label_name,
        "gating_mode": mode,
        "threshold": threshold,
        "uplift": uplift,
        "avg_gate": float(np.mean(gate)),
        "avg_factor": float(np.mean(factor)),
        "MAE": overall["MAE"],
        "RMSE": overall["RMSE"],
        "R2": overall["R2"],
        **spike_metrics,
        "improves_overall_rmse": int(overall["RMSE"] < base_metrics["RMSE"]),
        "improves_top10_rmse": int(spike_metrics["top10_RMSE"] < base_metrics["top10_RMSE"]),
        "accepted": int(
            overall["RMSE"] <= base_metrics["RMSE"] and spike_metrics["top10_RMSE"] <= base_metrics["top10_RMSE"]
        ),
    }


def choose_config(search_df: pd.DataFrame, mode: str | None = None) -> dict[str, Any]:
    subset = search_df.copy()
    if mode is not None:
        subset = subset[subset["gating_mode"] == mode].copy()
    if subset.empty:
        raise ValueError("No search rows available for config selection.")

    accepted = subset[subset["accepted"] == 1].copy()
    target = accepted if not accepted.empty else subset
    target = target.sort_values(
        ["RMSE", "top10_RMSE", "top10_underprediction", "non_spike_RMSE", "MAE"],
        ascending=[True, True, True, True, True],
    )
    return target.iloc[0].to_dict()


def choose_neighbor_config(search_df: pd.DataFrame, best_config: dict[str, Any], direction: str) -> dict[str, Any]:
    subset = search_df[
        (search_df["label_name"] == best_config["label_name"])
        & (search_df["gating_mode"] == best_config["gating_mode"])
        & (search_df["threshold"] == best_config["threshold"])
    ].copy()
    if subset.empty:
        return best_config

    uplift = float(best_config["uplift"])
    if direction == "lower":
        candidate = subset[subset["uplift"] < uplift].sort_values("uplift", ascending=False)
    else:
        candidate = subset[subset["uplift"] > uplift].sort_values("uplift", ascending=True)
    if candidate.empty:
        return best_config
    return candidate.iloc[0].to_dict()


def apply_gate_to_submission(
    base_submission: pd.DataFrame,
    probabilities: np.ndarray,
    config: dict[str, Any],
) -> pd.DataFrame:
    gate = gate_values(probabilities, str(config["gating_mode"]), float(config["threshold"]))
    factor = 1.0 + float(config["uplift"]) * gate
    output = base_submission[[DATE_COL]].copy()
    output[TARGET_COL] = np.maximum(0.0, pd.to_numeric(base_submission[TARGET_COL], errors="coerce") * factor)
    output[COGS_COL] = np.maximum(0.0, pd.to_numeric(base_submission[COGS_COL], errors="coerce") * factor)
    return output[[DATE_COL, TARGET_COL, COGS_COL]]


def validate_submission_frame(submission: pd.DataFrame, sample_submission: pd.DataFrame, label: str) -> None:
    if len(submission) != len(sample_submission):
        raise ValueError(f"{label}: row count mismatch")
    if submission.columns.tolist() != [DATE_COL, TARGET_COL, COGS_COL]:
        raise ValueError(f"{label}: invalid columns {submission.columns.tolist()}")
    if not submission[DATE_COL].reset_index(drop=True).equals(sample_submission[DATE_COL].reset_index(drop=True)):
        raise ValueError(f"{label}: Date order mismatch")
    if submission[[TARGET_COL, COGS_COL]].isna().any().any():
        raise ValueError(f"{label}: missing values found")
    if (submission[[TARGET_COL, COGS_COL]] < 0).any().any():
        raise ValueError(f"{label}: negative values found")


def save_submission_if_new(path: Path, submission: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing submission: {path}")
    submission.to_csv(path, index=False)


def build_feature_importance_frame(models: list[dict[str, Any]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for item in models:
        importance = get_classifier_feature_importance(
            model=item["model_object"],
            model_type=item["model_type"],
            feature_columns=item["feature_columns"],
            X_ref=item["X_train"],
            y_ref=item["y_train"],
        ).copy()
        importance.insert(0, "stage", item["stage"])
        importance.insert(1, "label_name", item["label_name"])
        frames.append(importance)
    return pd.concat(frames, ignore_index=True)


def run() -> None:
    logger = setup_logging()
    reporter = Reporter(logger)

    reporter.emit("Spike Probability Gate")
    reporter.emit("======================")
    reporter.emit("")

    reporter.emit("1. Load base data and rebuild safe feature tables")
    revenue_df = base.load_train_data(TRAIN_DATA_PATH)
    sample_submission = base.load_sample_submission(SAMPLE_SUBMISSION_PATH)
    validation_dates = revenue_df[(revenue_df[DATE_COL] >= TRAIN_CUTOFF) & (revenue_df[DATE_COL] <= VALIDATION_END)][
        DATE_COL
    ]
    historical_static = build_static_features_historical(revenue_df[DATE_COL], revenue_df[DATE_COL].min(), logger)
    future_static = build_static_features_future(sample_submission[DATE_COL], revenue_df[DATE_COL].min(), logger)
    classifier_table = build_classifier_table(revenue_df, historical_static)
    reporter.emit(f"Revenue dataset shape: {revenue_df.shape}")
    reporter.emit(f"Classifier table shape: {classifier_table.shape}")
    reporter.emit(f"Classifier feature count: {len(CLASSIFIER_FEATURES)}")

    reporter.emit("")
    reporter.emit("2. Reconstruct current best validation base blend")
    base_validation = reconstruct_base_validation(validation_dates)
    actual_validation = pd.Series(base_validation["actual_Revenue"].to_numpy(dtype=float))
    base_pred_validation = base_validation["base_pred"].to_numpy(dtype=float)
    base_validation_metrics = base.evaluate_predictions(actual_validation, base_pred_validation)
    base_validation_spike = compute_spike_metrics(actual_validation, base_pred_validation)
    reporter.emit(
        f"Base blend metrics: MAE={base_validation_metrics['MAE']:,.2f} | "
        f"RMSE={base_validation_metrics['RMSE']:,.2f} | R2={base_validation_metrics['R2']:.6f}"
    )
    reporter.emit(
        f"Base top10 spike RMSE={base_validation_spike['top10_RMSE']:,.2f} | "
        f"underprediction={base_validation_spike['top10_underprediction']}/{base_validation_spike['top10_count']}"
    )

    reporter.emit("")
    reporter.emit("3. Train spike classifiers on pre-2022 history")
    train_revenue = revenue_df[revenue_df[DATE_COL] < TRAIN_CUTOFF][TARGET_COL]
    validation_thresholds_bundle = spike2.compute_threshold_bundle(
        revenue_df[revenue_df[DATE_COL] < TRAIN_CUTOFF].set_index(DATE_COL)[TARGET_COL]
    )

    validation_recursive_features = build_recursive_classifier_features(
        prediction_dates=validation_dates,
        static_features=historical_static,
        initial_revenue_history=revenue_df[revenue_df[DATE_COL] < TRAIN_CUTOFF].set_index(DATE_COL)[TARGET_COL],
        recursive_revenue_source=pd.Series(base_pred_validation, index=pd.to_datetime(validation_dates)),
        thresholds_bundle=validation_thresholds_bundle,
        feature_columns=CLASSIFIER_FEATURES,
    )
    X_validation = (
        validation_recursive_features[CLASSIFIER_FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .reset_index(drop=True)
    )

    validation_models: list[dict[str, Any]] = []
    label_results: dict[str, dict[str, Any]] = {}

    for label_name, quantile in LABEL_SPECS.items():
        label_threshold = float(train_revenue.quantile(quantile))
        X_train, y_train, clean, feature_medians = build_train_classifier_matrix(
            classifier_table=classifier_table,
            feature_columns=CLASSIFIER_FEATURES,
            label_threshold=label_threshold,
            train_end_exclusive=TRAIN_CUTOFF,
        )
        reporter.emit(
            f"Training classifier {label_name}: rows={len(X_train):,}, positives={int(y_train.sum()):,}, "
            f"threshold={label_threshold:,.2f}"
        )
        model, model_type = train_classifier(X_train, y_train, reporter)

        X_val_filled = X_validation.fillna(feature_medians).fillna(0.0)
        probs = predict_classifier_proba(model, model_type, X_val_filled)
        y_val = (actual_validation >= label_threshold).astype(int)
        metrics = evaluate_classifier_metrics(y_val, probs, threshold=0.50)

        label_results[label_name] = {
            "label_name": label_name,
            "label_threshold": label_threshold,
            "model_object": model,
            "model_type": model_type,
            "X_train": X_train,
            "y_train": y_train,
            "feature_medians": feature_medians,
            "validation_probs": probs,
            "validation_metrics": metrics,
            "stage": "validation",
            "feature_columns": CLASSIFIER_FEATURES,
        }
        validation_models.append(label_results[label_name])
        reporter.emit(
            f"Classifier {label_name}: AUC={metrics['AUC']:.6f} | precision={metrics['precision']:.4f} | "
            f"recall={metrics['recall']:.4f} | F1={metrics['F1']:.4f} | "
            f"confusion=[[{metrics['tn']},{metrics['fp']}],[{metrics['fn']},{metrics['tp']}]]"
        )

    reporter.emit("")
    reporter.emit("4. Search gating rules on validation 2022")
    search_rows: list[dict[str, Any]] = []
    for label_name, result in label_results.items():
        for mode in GATING_MODES:
            for threshold in GATING_THRESHOLDS:
                for uplift in UPLIFTS:
                    row = evaluate_gating_config(
                        actual=actual_validation,
                        base_pred=base_pred_validation,
                        probs=result["validation_probs"],
                        label_name=label_name,
                        mode=mode,
                        threshold=threshold,
                        uplift=uplift,
                        base_metrics={**base_validation_metrics, **base_validation_spike},
                    )
                    row.update(
                        {
                            "classifier_AUC": result["validation_metrics"]["AUC"],
                            "classifier_precision": result["validation_metrics"]["precision"],
                            "classifier_recall": result["validation_metrics"]["recall"],
                            "classifier_F1": result["validation_metrics"]["F1"],
                        }
                    )
                    search_rows.append(row)

    search_df = pd.DataFrame(search_rows).sort_values(
        ["accepted", "RMSE", "top10_RMSE", "top10_underprediction", "non_spike_RMSE", "MAE"],
        ascending=[False, True, True, True, True, True],
    ).reset_index(drop=True)
    search_df.to_csv(SEARCH_RESULTS_PATH, index=False)

    best_config = choose_config(search_df)
    best_soft_config = choose_config(search_df, mode="soft")
    best_hard_config = choose_config(search_df, mode="hard")
    conservative_config = choose_neighbor_config(search_df, best_config, direction="lower")
    aggressive_config = choose_neighbor_config(search_df, best_config, direction="higher")

    reporter.emit_frame("Top gating configs:", search_df.head(10))

    reporter.emit("")
    reporter.emit("5. Retrain top10/top15 classifiers on full 2012-2022 history")
    full_revenue = revenue_df[TARGET_COL]
    full_thresholds_bundle = spike2.compute_threshold_bundle(revenue_df.set_index(DATE_COL)[TARGET_COL])
    full_models: dict[str, dict[str, Any]] = {}
    deploy_models_for_importance: list[dict[str, Any]] = []

    for label_name, quantile in LABEL_SPECS.items():
        label_threshold = float(full_revenue.quantile(quantile))
        full_table = classifier_table.copy()
        full_table["label"] = (pd.to_numeric(full_table[TARGET_COL], errors="coerce") >= label_threshold).astype(int)
        clean = full_table.dropna(subset=CLASSIFIER_FEATURES + ["label"]).reset_index(drop=True)
        X_train = clean[CLASSIFIER_FEATURES].copy()
        y_train = clean["label"].copy()
        feature_medians = X_train.median(numeric_only=True)

        reporter.emit(
            f"Retraining full classifier {label_name}: rows={len(X_train):,}, positives={int(y_train.sum()):,}, "
            f"threshold={label_threshold:,.2f}"
        )
        model, model_type = train_classifier(X_train, y_train, reporter)
        full_models[label_name] = {
            "label_name": label_name,
            "label_threshold": label_threshold,
            "model_object": model,
            "model_type": model_type,
            "feature_medians": feature_medians,
            "feature_columns": CLASSIFIER_FEATURES,
            "stage": "full",
            "X_train": X_train,
            "y_train": y_train,
        }
        deploy_models_for_importance.append(full_models[label_name])

    reporter.emit("")
    reporter.emit("6. Compute future spike probabilities from recursive base-submission history")
    base_future_submission = spike1.load_submission(BASE_SUBMISSION_PATH, sample_submission)
    future_recursive_features = build_recursive_classifier_features(
        prediction_dates=sample_submission[DATE_COL],
        static_features=future_static,
        initial_revenue_history=revenue_df.set_index(DATE_COL)[TARGET_COL],
        recursive_revenue_source=base_future_submission.set_index(DATE_COL)[TARGET_COL],
        thresholds_bundle=full_thresholds_bundle,
        feature_columns=CLASSIFIER_FEATURES,
    )

    future_probabilities: dict[str, np.ndarray] = {}
    for label_name, trained in full_models.items():
        X_future = future_recursive_features[CLASSIFIER_FEATURES].fillna(trained["feature_medians"]).fillna(0.0)
        future_probabilities[label_name] = predict_classifier_proba(
            trained["model_object"],
            trained["model_type"],
            X_future,
        )

    reporter.emit("")
    reporter.emit("7. Save validation outputs and feature importance")
    best_label_name = str(best_config["label_name"])
    best_validation_factor = 1.0 + float(best_config["uplift"]) * gate_values(
        label_results[best_label_name]["validation_probs"],
        str(best_config["gating_mode"]),
        float(best_config["threshold"]),
    )
    best_validation_pred = base_pred_validation * best_validation_factor
    validation_output = pd.DataFrame(
        {
            DATE_COL: pd.to_datetime(validation_dates).reset_index(drop=True),
            "actual_Revenue": actual_validation.to_numpy(dtype=float),
            "base_pred": base_pred_validation,
            "prob_top10": label_results["top10"]["validation_probs"],
            "prob_top15": label_results["top15"]["validation_probs"],
            "best_adjusted_pred": best_validation_pred,
            "best_label": best_label_name,
            "best_mode": best_config["gating_mode"],
            "best_threshold": float(best_config["threshold"]),
            "best_uplift": float(best_config["uplift"]),
        }
    )
    validation_output.to_csv(VALIDATION_PREDICTIONS_PATH, index=False)

    importance_df = build_feature_importance_frame(validation_models + deploy_models_for_importance)
    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    reporter.emit("")
    reporter.emit("8. Create gated future submissions")
    submission_configs = {
        SUBMISSION_BEST_PATH: best_config,
        SUBMISSION_SOFT_PATH: best_soft_config,
        SUBMISSION_HARD_PATH: best_hard_config,
        SUBMISSION_CONSERVATIVE_PATH: conservative_config,
        SUBMISSION_AGGRESSIVE_PATH: aggressive_config,
    }
    submission_summary_rows: list[dict[str, Any]] = []
    for path, config in submission_configs.items():
        label_name = str(config["label_name"])
        submission = apply_gate_to_submission(
            base_submission=base_future_submission,
            probabilities=future_probabilities[label_name],
            config=config,
        )
        validate_submission_frame(submission, sample_submission, path.name)
        save_submission_if_new(path, submission)
        submission_summary_rows.append(
            {
                "file": path.name,
                "label_name": config["label_name"],
                "gating_mode": config["gating_mode"],
                "threshold": config["threshold"],
                "uplift": config["uplift"],
            }
        )

    reporter.emit_frame("Created submission configs:", pd.DataFrame(submission_summary_rows))

    reporter.emit("")
    reporter.emit("9. Final summary")
    best_classifier_metrics = label_results[best_label_name]["validation_metrics"]
    before_metrics = {**base_validation_metrics, **base_validation_spike}
    after_metrics = search_df.iloc[0].to_dict()

    reporter.emit(f"Best spike label: {best_label_name}")
    reporter.emit(
        f"Best classifier metrics: AUC={best_classifier_metrics['AUC']:.6f}, "
        f"precision={best_classifier_metrics['precision']:.4f}, "
        f"recall={best_classifier_metrics['recall']:.4f}, "
        f"F1={best_classifier_metrics['F1']:.4f}"
    )
    reporter.emit(
        f"Best gating mode={best_config['gating_mode']} | threshold={best_config['threshold']:.2f} | "
        f"uplift={best_config['uplift']:.2f}"
    )
    reporter.emit(
        f"Validation before gating: MAE={before_metrics['MAE']:,.2f} | RMSE={before_metrics['RMSE']:,.2f} | "
        f"R2={before_metrics['R2']:.6f}"
    )
    reporter.emit(
        f"Validation after gating: MAE={after_metrics['MAE']:,.2f} | RMSE={after_metrics['RMSE']:,.2f} | "
        f"R2={after_metrics['R2']:.6f}"
    )
    reporter.emit(
        f"Top10 spike RMSE before/after: {before_metrics['top10_RMSE']:,.2f} -> {after_metrics['top10_RMSE']:,.2f}"
    )
    reporter.emit(
        f"Non-spike RMSE before/after: {before_metrics['non_spike_RMSE']:,.2f} -> {after_metrics['non_spike_RMSE']:,.2f}"
    )
    reporter.emit_frame(
        "Top 30 feature importances for best label (full train):",
        importance_df[
            (importance_df["stage"] == "full") & (importance_df["label_name"] == best_label_name)
        ].head(30),
    )
    reporter.emit(f"Saved validation predictions: {VALIDATION_PREDICTIONS_PATH}")
    reporter.emit(f"Saved search results: {SEARCH_RESULTS_PATH}")
    reporter.emit(f"Saved feature importance: {FEATURE_IMPORTANCE_PATH}")
    reporter.emit(
        "Created submission files: "
        + ", ".join(path.name for path in submission_configs.keys())
    )
    upload_order = [
        SUBMISSION_BEST_PATH.name,
        SUBMISSION_CONSERVATIVE_PATH.name,
        SUBMISSION_SOFT_PATH.name,
        SUBMISSION_HARD_PATH.name,
        SUBMISSION_AGGRESSIVE_PATH.name,
    ]
    reporter.emit(f"Recommended upload order: {upload_order}")
    reporter.emit(
        "Leakage confirmation: the gate uses only calendar/promo schedule, inventory as-of, and lagged revenue "
        "features computed from historical actual Revenue plus recursive base-model predictions. No future actual "
        "Revenue, actual COGS, or same-day realized demand features are used."
    )
    reporter.save(REPORT_PATH)


if __name__ == "__main__":
    run()
