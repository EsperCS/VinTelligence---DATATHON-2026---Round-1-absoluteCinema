from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import adaptive_scaling_layer as scale_mod
import train_final_model as base
import train_meta_scaling as meta_mod
import train_spike_probability_gate as gate_mod


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "log"

CURRENT_BEST_SUBMISSION_PATH = DATA_DIR / "submission_meta_scale_conservative.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
META_VALIDATION_PATH = DATA_DIR / "meta_scaling_validation_predictions.csv"
SPIKE_GATE_VALIDATION_PATH = DATA_DIR / "spike_gate_validation_predictions.csv"
FUTURE_PROMO_FEATURES_PATH = DATA_DIR / "future_promo_calendar_features.csv"

RESULTS_PATH = DATA_DIR / "final_micro_calibration_results.csv"
VALIDATION_OUTPUT_PATH = DATA_DIR / "final_micro_calibration_validation_predictions.csv"

SUBMISSION_ULTRA_SAFE_PATH = DATA_DIR / "submission_final_ultra_safe.csv"
SUBMISSION_BALANCED_PATH = DATA_DIR / "submission_final_balanced.csv"
SUBMISSION_SPIKE_PUSH_PATH = DATA_DIR / "submission_final_spike_push.csv"
SUBMISSION_PROMO_PUSH_PATH = DATA_DIR / "submission_final_promo_push.csv"
SUBMISSION_PRIVATE_SAFE_PATH = DATA_DIR / "submission_final_private_safe.csv"

REPORT_PATH = LOG_DIR / "final_micro_calibration_report.txt"
LOG_FILE = LOG_DIR / "train_final_micro_calibration.log"

DATE_COL = base.DATE_COL
TARGET_COL = base.TARGET_COL
COGS_COL = base.COGS_COL
HIGH_RISK_MONTHS = {2, 3, 5, 8}
CURRENT_BASE_CLIP = meta_mod.CONSERVATIVE_CLIP


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

    logger = logging.getLogger("final_micro_calibration")
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


def infer_first_existing(columns: list[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


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


def load_submission(path: Path, sample_submission: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=[DATE_COL], low_memory=False)
    frame[DATE_COL] = pd.to_datetime(frame[DATE_COL], errors="coerce").dt.normalize()
    validate_submission_frame(frame[[DATE_COL, TARGET_COL, COGS_COL]], sample_submission)
    return frame[[DATE_COL, TARGET_COL, COGS_COL]].copy()


def compute_extended_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    actual_values = pd.to_numeric(actual, errors="coerce").to_numpy(dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    error = actual_values - predicted_values

    overall = base.evaluate_predictions(actual, predicted_values)
    top10_threshold = float(np.quantile(actual_values, 0.90))
    top5_threshold = float(np.quantile(actual_values, 0.95))
    top10_mask = actual_values >= top10_threshold
    top5_mask = actual_values >= top5_threshold
    non_spike_mask = actual_values < top10_threshold

    def masked_rmse(mask: np.ndarray) -> float:
        return float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan

    return {
        **overall,
        "top10_RMSE": masked_rmse(top10_mask),
        "top10_underprediction": int(np.sum(error[top10_mask] > 0)) if top10_mask.any() else 0,
        "top10_count": int(np.sum(top10_mask)),
        "top5_RMSE": masked_rmse(top5_mask),
        "top5_underprediction": int(np.sum(error[top5_mask] > 0)) if top5_mask.any() else 0,
        "top5_count": int(np.sum(top5_mask)),
        "non_spike_RMSE": masked_rmse(non_spike_mask),
    }


def safe_divide_array(numerator: np.ndarray, denominator: np.ndarray, default: float = 1.0) -> np.ndarray:
    numerator_arr = np.asarray(numerator, dtype=float)
    denominator_arr = np.asarray(denominator, dtype=float)
    output = np.full(numerator_arr.shape, default, dtype=float)
    valid = np.isfinite(numerator_arr) & np.isfinite(denominator_arr) & (np.abs(denominator_arr) > 1e-9)
    output[valid] = numerator_arr[valid] / denominator_arr[valid]
    return output


def infer_validation_column_mapping(meta_validation: pd.DataFrame, spike_gate_validation: pd.DataFrame) -> dict[str, str | None]:
    meta_columns = meta_validation.columns.tolist()
    spike_columns = spike_gate_validation.columns.tolist()
    return {
        "date": infer_first_existing(meta_columns, [DATE_COL]),
        "actual": infer_first_existing(meta_columns, ["actual_Revenue", TARGET_COL]),
        "base_or_final": infer_first_existing(
            meta_columns,
            ["best_meta_pred", "adaptive_plus_base_pred", "spike_gate_aggressive_base_pred"],
        ),
        "predicted_scale": infer_first_existing(meta_columns, ["best_meta_scale"]),
        "spike_prob": infer_first_existing(spike_columns, ["prob_top10", "spike_prob", "prob_top15"]),
    }


def compute_training_promo_intensity_p75(logger: logging.Logger) -> float:
    train_df = base.load_train_data(base.TRAIN_DATA_PATH)
    hist_static = gate_mod.build_static_features_historical(train_df[DATE_COL], train_df[DATE_COL].min(), logger)
    cutoff_mask = hist_static[DATE_COL] < base.TRAIN_CUTOFF
    promo_intensity = (
        pd.to_numeric(hist_static["calendar_avg_discount_value"], errors="coerce").fillna(0.0)
        * pd.to_numeric(hist_static["calendar_active_promo_count"], errors="coerce").fillna(0.0)
    )
    positive = promo_intensity[cutoff_mask & (promo_intensity > 0)]
    if positive.empty:
        return 0.0
    return float(positive.quantile(0.75))


def add_common_context_features(frame: pd.DataFrame, promo_intensity_p75: float) -> pd.DataFrame:
    output = frame.copy()
    output["spike_prob"] = pd.to_numeric(output.get("prob_top10", output.get("spike_prob", 0.0)), errors="coerce").fillna(0.0)
    output["promo_intensity"] = (
        pd.to_numeric(output.get("calendar_avg_discount_value", 0.0), errors="coerce").fillna(0.0)
        * pd.to_numeric(output.get("calendar_active_promo_count", 0.0), errors="coerce").fillna(0.0)
    )
    output["month"] = pd.to_numeric(output["month"], errors="coerce").fillna(0).astype(int)
    output["is_promo"] = (pd.to_numeric(output.get("calendar_any_promo", 0.0), errors="coerce").fillna(0.0) > 0).astype(int)
    output["is_high_risk_month"] = output["month"].isin(HIGH_RISK_MONTHS).astype(int)
    output["is_high_intensity_promo"] = (
        (output["is_promo"] == 1) & (output["promo_intensity"] >= promo_intensity_p75)
    ).astype(int)
    output["promo_progress_ratio"] = pd.to_numeric(output.get("promo_progress_ratio", 0.0), errors="coerce").fillna(0.0)
    output["promo_days_remaining"] = pd.to_numeric(output.get("promo_days_remaining", 0.0), errors="coerce").fillna(0.0)
    output["promo_is_first_7_days"] = pd.to_numeric(output.get("promo_is_first_7_days", 0.0), errors="coerce").fillna(0.0)
    output["promo_is_last_7_days"] = pd.to_numeric(output.get("promo_is_last_7_days", 0.0), errors="coerce").fillna(0.0)
    return output


def reconstruct_validation_base(
    reporter: Reporter,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    meta_validation = pd.read_csv(META_VALIDATION_PATH, parse_dates=[DATE_COL], low_memory=False)
    spike_gate_validation = pd.read_csv(SPIKE_GATE_VALIDATION_PATH, parse_dates=[DATE_COL], low_memory=False)
    meta_validation[DATE_COL] = pd.to_datetime(meta_validation[DATE_COL], errors="coerce").dt.normalize()
    spike_gate_validation[DATE_COL] = pd.to_datetime(spike_gate_validation[DATE_COL], errors="coerce").dt.normalize()

    mapping = infer_validation_column_mapping(meta_validation, spike_gate_validation)
    reporter.emit(
        "Validation column mapping: "
        f"Date={mapping['date']}, actual={mapping['actual']}, "
        f"base/final={mapping['base_or_final']}, predicted_scale={mapping['predicted_scale']}, "
        f"spike_prob={mapping['spike_prob']}"
    )

    chosen_model_name = None
    if "best_model_name" in meta_validation.columns:
        chosen_model_name = meta_validation["best_model_name"].dropna().astype(str).iloc[0]

    aggressive_context, plus_context = meta_mod.build_validation_contexts(reporter, logger)
    aggressive_context = meta_mod.add_base_prediction_features(aggressive_context)
    plus_context = meta_mod.add_base_prediction_features(plus_context)

    plus_target = meta_mod.clip_scale_target(
        plus_context["actual_Revenue"] / plus_context["base_pred"].replace(0, np.nan)
    ).fillna(1.0)

    model_names = meta_mod.discover_meta_model_names()
    plus_raw_predictions, plus_full_models = meta_mod.generate_time_safe_meta_predictions(
        plus_context,
        plus_target,
        model_names,
        reporter,
        context_name="adaptive_plus",
    )

    if chosen_model_name is None or chosen_model_name not in plus_raw_predictions:
        chosen_model_name = sorted(plus_raw_predictions)[0]
    reporter.emit(f"Chosen meta model for conservative base reconstruction: {chosen_model_name}")

    raw_meta_scale = plus_raw_predictions[chosen_model_name]
    conservative_scale = np.clip(raw_meta_scale, CURRENT_BASE_CLIP[0], CURRENT_BASE_CLIP[1])
    current_base_pred = plus_context["base_pred"].to_numpy(dtype=float) * conservative_scale

    merge_cols = [DATE_COL]
    if mapping["spike_prob"] is not None:
        merge_cols.append(mapping["spike_prob"])
    if "prob_top15" in spike_gate_validation.columns:
        merge_cols.append("prob_top15")
    merge_cols = list(dict.fromkeys(merge_cols))

    current_context = plus_context.merge(
        spike_gate_validation[merge_cols],
        on=DATE_COL,
        how="left",
        suffixes=("", "_gate"),
    )
    if mapping["spike_prob"] and mapping["spike_prob"] != "spike_prob":
        current_context["spike_prob"] = pd.to_numeric(current_context[mapping["spike_prob"]], errors="coerce").fillna(
            pd.to_numeric(current_context.get("spike_prob", 0.0), errors="coerce").fillna(0.0)
        )

    promo_intensity_p75 = compute_training_promo_intensity_p75(logger)
    current_context = add_common_context_features(current_context, promo_intensity_p75)
    current_context["raw_meta_scale"] = raw_meta_scale
    current_context["current_base_scale"] = conservative_scale
    current_context["current_base_pred"] = current_base_pred
    current_context["micro_reference_factor"] = 1.0

    metadata = {
        "chosen_model_name": chosen_model_name,
        "promo_intensity_p75": promo_intensity_p75,
        "base_clip": CURRENT_BASE_CLIP,
    }
    return current_context, metadata, plus_raw_predictions, plus_full_models


def build_future_context(
    reporter: Reporter,
    logger: logging.Logger,
    chosen_model_name: str,
    promo_intensity_p75: float,
    full_models: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_submission = base.load_sample_submission(SAMPLE_SUBMISSION_PATH)
    current_best_submission = load_submission(CURRENT_BEST_SUBMISSION_PATH, sample_submission)

    future_context = meta_mod.build_future_meta_context(meta_mod.ADAPTIVE_PLUS_SUBMISSION_PATH, reporter, logger)
    future_context = meta_mod.add_base_prediction_features(future_context)
    future_context = add_common_context_features(future_context, promo_intensity_p75)

    X_future, _ = meta_mod.build_meta_feature_matrix(future_context)
    raw_meta_scale = meta_mod.predict_meta_model(full_models[chosen_model_name], X_future)
    conservative_scale = np.clip(raw_meta_scale, CURRENT_BASE_CLIP[0], CURRENT_BASE_CLIP[1])

    future_context["raw_meta_scale"] = raw_meta_scale
    future_context["current_base_scale"] = conservative_scale
    future_context["current_base_pred"] = current_best_submission[TARGET_COL].to_numpy(dtype=float)
    return future_context, current_best_submission


def apply_protection_variant(scale: np.ndarray, spike_prob: np.ndarray, variant: str | None) -> np.ndarray:
    output = np.asarray(scale, dtype=float).copy()
    prob = np.asarray(spike_prob, dtype=float)
    if variant is None:
        return output

    if variant == "light":
        output = np.where(prob < 0.15, output * 0.990, output)
        output = np.where((prob >= 0.15) & (prob < 0.25), output * 0.995, output)
        return output
    if variant == "medium":
        output = np.where(prob < 0.15, output * 0.985, output)
        output = np.where((prob >= 0.15) & (prob < 0.25), output * 0.990, output)
        return output
    if variant == "strong":
        output = np.where(prob < 0.15, output * 0.975, output)
        output = np.where((prob >= 0.15) & (prob < 0.25), output * 0.985, output)
        return output
    raise ValueError(f"Unknown protection variant: {variant}")


def apply_high_risk_boost(
    scale: np.ndarray,
    is_high_risk_month: np.ndarray,
    spike_prob: np.ndarray,
    is_promo: np.ndarray,
    boost: float,
) -> np.ndarray:
    output = np.asarray(scale, dtype=float).copy()
    gate = (np.asarray(is_high_risk_month, dtype=int) == 1) & (
        (np.asarray(spike_prob, dtype=float) >= 0.30) | (np.asarray(is_promo, dtype=int) == 1)
    )
    output[gate] *= 1.0 + boost
    return output


def apply_promo_boost(
    scale: np.ndarray,
    is_promo: np.ndarray,
    is_high_intensity_promo: np.ndarray,
    active_boost: float = 0.0,
    high_intensity_boost: float = 0.0,
) -> np.ndarray:
    output = np.asarray(scale, dtype=float).copy()
    if active_boost:
        output[np.asarray(is_promo, dtype=int) == 1] *= 1.0 + active_boost
    if high_intensity_boost:
        output[np.asarray(is_high_intensity_promo, dtype=int) == 1] *= 1.0 + high_intensity_boost
    return output


def build_candidate_scale(context: pd.DataFrame, spec: dict[str, Any]) -> np.ndarray:
    raw_scale = context["raw_meta_scale"].to_numpy(dtype=float)
    conservative_scale = context["current_base_scale"].to_numpy(dtype=float)

    if spec.get("clip") is None:
        total_scale = conservative_scale.copy()
    else:
        clip_min, clip_max = spec["clip"]
        total_scale = np.clip(raw_scale, clip_min, clip_max)

    protection = spec.get("protection")
    if protection:
        total_scale = apply_protection_variant(total_scale, context["spike_prob"].to_numpy(dtype=float), protection)

    high_risk_boost = float(spec.get("high_risk_boost", 0.0))
    if high_risk_boost:
        total_scale = apply_high_risk_boost(
            total_scale,
            context["is_high_risk_month"].to_numpy(dtype=int),
            context["spike_prob"].to_numpy(dtype=float),
            context["is_promo"].to_numpy(dtype=int),
            high_risk_boost,
        )

    active_promo_boost = float(spec.get("active_promo_boost", 0.0))
    high_intensity_boost = float(spec.get("high_intensity_promo_boost", 0.0))
    if active_promo_boost or high_intensity_boost:
        total_scale = apply_promo_boost(
            total_scale,
            context["is_promo"].to_numpy(dtype=int),
            context["is_high_intensity_promo"].to_numpy(dtype=int),
            active_boost=active_promo_boost,
            high_intensity_boost=high_intensity_boost,
        )

    return total_scale


def evaluate_candidate(
    candidate_name: str,
    spec: dict[str, Any],
    context: pd.DataFrame,
) -> tuple[dict[str, Any], np.ndarray]:
    total_scale = build_candidate_scale(context, spec)
    micro_factor = safe_divide_array(total_scale, context["current_base_scale"].to_numpy(dtype=float))
    final_pred = context["current_base_pred"].to_numpy(dtype=float) * micro_factor

    metrics = compute_extended_metrics(context["actual_Revenue"], final_pred)
    row = {
        "candidate_name": candidate_name,
        "clip_min": spec["clip"][0] if spec.get("clip") else CURRENT_BASE_CLIP[0],
        "clip_max": spec["clip"][1] if spec.get("clip") else CURRENT_BASE_CLIP[1],
        "protection": spec.get("protection", ""),
        "high_risk_boost": float(spec.get("high_risk_boost", 0.0)),
        "active_promo_boost": float(spec.get("active_promo_boost", 0.0)),
        "high_intensity_promo_boost": float(spec.get("high_intensity_promo_boost", 0.0)),
        "mean_scale": float(np.mean(total_scale)),
        "min_scale": float(np.min(total_scale)),
        "max_scale": float(np.max(total_scale)),
        **metrics,
    }
    return row, final_pred


def select_best_candidate(results: pd.DataFrame, base_non_spike_rmse: float) -> pd.Series:
    candidates = results[results["candidate_name"] != "current_base"].copy()
    accepted = candidates[candidates["non_spike_RMSE"] <= base_non_spike_rmse * 1.01]
    pool = accepted if not accepted.empty else candidates
    return pool.sort_values(["RMSE", "top10_RMSE", "MAE"]).iloc[0]


def create_submission(
    path: Path,
    candidate_name: str,
    spec: dict[str, Any],
    future_context: pd.DataFrame,
    current_best_submission: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> dict[str, float]:
    total_scale = build_candidate_scale(future_context, spec)
    micro_factor = safe_divide_array(total_scale, future_context["current_base_scale"].to_numpy(dtype=float))

    output = sample_submission[[DATE_COL]].copy()
    output[TARGET_COL] = current_best_submission[TARGET_COL].to_numpy(dtype=float) * micro_factor
    output[COGS_COL] = current_best_submission[COGS_COL].to_numpy(dtype=float) * micro_factor
    output[TARGET_COL] = output[TARGET_COL].clip(lower=0.0)
    output[COGS_COL] = output[COGS_COL].clip(lower=0.0)
    validate_submission_frame(output, sample_submission)
    output.to_csv(path, index=False)
    return {
        "candidate_name": candidate_name,
        "mean_scale": float(np.mean(total_scale)),
        "min_scale": float(np.min(total_scale)),
        "max_scale": float(np.max(total_scale)),
    }


def run() -> None:
    logger = setup_logging()
    reporter = Reporter(logger)

    reporter.emit("Final Micro Calibration")
    reporter.emit("=======================")
    reporter.emit("")

    reporter.emit("1. Load current best validation predictions")
    validation_context, metadata, _, full_models = reconstruct_validation_base(reporter, logger)
    current_base_metrics = compute_extended_metrics(
        validation_context["actual_Revenue"],
        validation_context["current_base_pred"].to_numpy(dtype=float),
    )
    reporter.emit(
        f"Current best validation analog uses meta model {metadata['chosen_model_name']} "
        f"with conservative clip [{CURRENT_BASE_CLIP[0]:.2f}, {CURRENT_BASE_CLIP[1]:.2f}]"
    )
    reporter.emit(
        f"Training promo intensity p75 for high-intensity flag: {metadata['promo_intensity_p75']:.6f}"
    )

    candidate_specs: dict[str, dict[str, Any]] = {
        "current_base": {"clip": None},
        "clip_096_110": {"clip": (0.96, 1.10)},
        "clip_097_108": {"clip": (0.97, 1.08)},
        "clip_098_106": {"clip": (0.98, 1.06)},
        "clip_095_112": {"clip": (0.95, 1.12)},
        "protection_light": {"clip": None, "protection": "light"},
        "protection_medium": {"clip": None, "protection": "medium"},
        "protection_strong": {"clip": None, "protection": "strong"},
        "high_risk_005": {"clip": None, "high_risk_boost": 0.005},
        "high_risk_010": {"clip": None, "high_risk_boost": 0.010},
        "high_risk_015": {"clip": None, "high_risk_boost": 0.015},
        "promo_active_005": {"clip": None, "active_promo_boost": 0.005},
        "promo_active_010": {"clip": None, "active_promo_boost": 0.010},
        "promo_high_intensity_015": {"clip": None, "high_intensity_promo_boost": 0.015},
        "candidate_a_ultra_safe": {"clip": (0.97, 1.08), "protection": "light"},
        "candidate_b_balanced": {"clip": (0.96, 1.10), "protection": "light", "high_risk_boost": 0.005},
        "candidate_c_spike_push": {"clip": (0.95, 1.12), "high_risk_boost": 0.010, "active_promo_boost": 0.005},
        "candidate_d_promo_push": {
            "clip": (0.96, 1.10),
            "active_promo_boost": 0.010,
            "high_intensity_promo_boost": 0.015,
        },
        "candidate_e_private_safe": {"clip": (0.98, 1.06), "protection": "medium"},
    }

    reporter.emit("")
    reporter.emit("2-7. Run micro-calibration candidates on validation")
    result_rows: list[dict[str, Any]] = []
    validation_predictions: dict[str, np.ndarray] = {}
    for candidate_name, spec in candidate_specs.items():
        row, pred = evaluate_candidate(candidate_name, spec, validation_context)
        result_rows.append(row)
        validation_predictions[candidate_name] = pred

    results_df = pd.DataFrame(result_rows).sort_values(["RMSE", "top10_RMSE", "MAE"]).reset_index(drop=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    reporter.emit_frame("Candidate validation table:", results_df.head(20))

    best_candidate = select_best_candidate(results_df, current_base_metrics["non_spike_RMSE"])
    conservative_pool = results_df[results_df["candidate_name"].isin(["candidate_a_ultra_safe", "candidate_e_private_safe"])]
    best_conservative = conservative_pool.sort_values(["RMSE", "top10_RMSE", "MAE"]).iloc[0]
    saved_candidate_names = [
        "candidate_a_ultra_safe",
        "candidate_b_balanced",
        "candidate_c_spike_push",
        "candidate_d_promo_push",
        "candidate_e_private_safe",
    ]
    saved_pool = results_df[results_df["candidate_name"].isin(saved_candidate_names)].copy()
    best_saved_candidate = saved_pool.sort_values(["RMSE", "top10_RMSE", "MAE"]).iloc[0]

    validation_output = pd.DataFrame(
        {
            DATE_COL: validation_context[DATE_COL],
            "actual_Revenue": validation_context["actual_Revenue"],
            "current_base_pred": validation_context["current_base_pred"],
            "current_base_scale": validation_context["current_base_scale"],
            "spike_prob": validation_context["spike_prob"],
            "candidate_b_balanced_pred": validation_predictions["candidate_b_balanced"],
            "candidate_e_private_safe_pred": validation_predictions["candidate_e_private_safe"],
            "best_candidate_name": best_candidate["candidate_name"],
            "best_candidate_pred": validation_predictions[str(best_candidate["candidate_name"])],
        }
    )
    validation_output.to_csv(VALIDATION_OUTPUT_PATH, index=False)

    reporter.emit("")
    reporter.emit("8. Apply selected candidates to future submission")
    future_context, current_best_submission = build_future_context(
        reporter,
        logger,
        chosen_model_name=str(metadata["chosen_model_name"]),
        promo_intensity_p75=float(metadata["promo_intensity_p75"]),
        full_models=full_models,
    )
    sample_submission = base.load_sample_submission(SAMPLE_SUBMISSION_PATH)

    submission_stats = []
    for candidate_name, path in [
        ("candidate_a_ultra_safe", SUBMISSION_ULTRA_SAFE_PATH),
        ("candidate_b_balanced", SUBMISSION_BALANCED_PATH),
        ("candidate_c_spike_push", SUBMISSION_SPIKE_PUSH_PATH),
        ("candidate_d_promo_push", SUBMISSION_PROMO_PUSH_PATH),
        ("candidate_e_private_safe", SUBMISSION_PRIVATE_SAFE_PATH),
    ]:
        stats = create_submission(
            path=path,
            candidate_name=candidate_name,
            spec=candidate_specs[candidate_name],
            future_context=future_context,
            current_best_submission=current_best_submission,
            sample_submission=sample_submission,
        )
        submission_stats.append(stats)

    submission_stats_df = pd.DataFrame(submission_stats)

    reporter.emit("")
    reporter.emit("9. Final summary")
    reporter.emit(
        f"Current best validation metrics: MAE={current_base_metrics['MAE']:,.2f}, "
        f"RMSE={current_base_metrics['RMSE']:,.2f}, R2={current_base_metrics['R2']:.6f}, "
        f"Top10 RMSE={current_base_metrics['top10_RMSE']:,.2f}, "
        f"Non-spike RMSE={current_base_metrics['non_spike_RMSE']:,.2f}"
    )
    reporter.emit(
        f"Best candidate by RMSE: {best_candidate['candidate_name']} | "
        f"RMSE={best_candidate['RMSE']:,.2f}, Top10 RMSE={best_candidate['top10_RMSE']:,.2f}, "
        f"Non-spike RMSE={best_candidate['non_spike_RMSE']:,.2f}"
    )
    reporter.emit(
        f"Best saved candidate: {best_saved_candidate['candidate_name']} | "
        f"RMSE={best_saved_candidate['RMSE']:,.2f}"
    )
    reporter.emit(
        f"Best conservative/private-safe candidate: {best_conservative['candidate_name']} | "
        f"RMSE={best_conservative['RMSE']:,.2f}"
    )
    reporter.emit_frame(
        "Future mean/min/max scale by submission:",
        submission_stats_df[["candidate_name", "mean_scale", "min_scale", "max_scale"]],
    )
    reporter.emit(
        "Created submission files: submission_final_ultra_safe.csv, submission_final_balanced.csv, "
        "submission_final_spike_push.csv, submission_final_promo_push.csv, submission_final_private_safe.csv"
    )

    candidate_to_file = {
        "candidate_a_ultra_safe": "submission_final_ultra_safe.csv",
        "candidate_b_balanced": "submission_final_balanced.csv",
        "candidate_c_spike_push": "submission_final_spike_push.csv",
        "candidate_d_promo_push": "submission_final_promo_push.csv",
        "candidate_e_private_safe": "submission_final_private_safe.csv",
    }
    upload_order = [
        candidate_to_file[str(best_saved_candidate["candidate_name"])],
        candidate_to_file[str(best_conservative["candidate_name"])],
        candidate_to_file["candidate_b_balanced"],
        candidate_to_file["candidate_c_spike_push"],
        candidate_to_file["candidate_d_promo_push"],
        candidate_to_file["candidate_e_private_safe"],
    ]
    upload_order = list(dict.fromkeys(upload_order))
    reporter.emit(f"Recommended upload order: {upload_order}")
    reporter.save(REPORT_PATH)


if __name__ == "__main__":
    run()
