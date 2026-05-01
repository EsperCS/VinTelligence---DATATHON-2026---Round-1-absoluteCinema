from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "log"

DATE_COL = "Date"
TARGET_COL = "Revenue"
COGS_COL = "COGS"
RATIO = 0.8900
EPS = 1e-9

SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
CURRENT_BEST_PATH = DATA_DIR / "submission_blend_direct_15_cogs8900.csv"

OUTPUT_DIAGNOSTICS_PATH = DATA_DIR / "anti_overfit_candidate_diagnostics.csv"
OUTPUT_RISK_PATH = DATA_DIR / "anti_overfit_blend_risk_table.csv"
OUTPUT_METADATA_PATH = DATA_DIR / "anti_overfit_submission_metadata.csv"
REPORT_PATH = LOG_DIR / "anti_overfit_ensemble_report.txt"
LOG_PATH = LOG_DIR / "train_anti_overfit_ensemble.log"

OUTPUT_9505_PATH = DATA_DIR / "submission_anti_overfit_9505.csv"
OUTPUT_9010_PATH = DATA_DIR / "submission_anti_overfit_9010.csv"
OUTPUT_THREEWAY_PATH = DATA_DIR / "submission_anti_overfit_threeway.csv"
OUTPUT_SCALE_UP_PATH = DATA_DIR / "submission_anti_overfit_scale_up.csv"
OUTPUT_SCALE_DOWN_PATH = DATA_DIR / "submission_anti_overfit_scale_down.csv"

SCALES = [0.990, 0.995, 1.000, 1.005, 1.010]

CANDIDATE_SPECS = [
    {
        "name": "current_best",
        "path": CURRENT_BEST_PATH,
        "tags": ["anchor"],
        "description": "confirmed_public_anchor",
    },
    {
        "name": "cogs_ratio_8900",
        "path": DATA_DIR / "submission_cogs_ratio_8900.csv",
        "tags": ["anchor_like", "ratio_anchor"],
        "description": "fixed_cogs_ratio_anchor",
    },
    {
        "name": "direct_seasonal",
        "path": DATA_DIR / "submission_direct_seasonal_ratio_8900.csv",
        "tags": ["seasonal_anchor"],
        "description": "direct_seasonal_diversity",
    },
    {
        "name": "feature_union",
        "path": DATA_DIR / "submission_feature_union.csv",
        "tags": ["feature_heavy"],
        "description": "feature_union_candidate",
    },
    {
        "name": "promo_known",
        "path": DATA_DIR / "submission_promo_known.csv",
        "tags": ["promo_heavy"],
        "description": "promo_known_candidate",
    },
    {
        "name": "segment_bottomup",
        "path": DATA_DIR / "submission_m5_segment_bottomup.csv",
        "tags": ["segment_heavy"],
        "description": "segment_bottomup_candidate",
    },
    {
        "name": "funnel_seasonal",
        "path": DATA_DIR / "submission_funnel_seasonal.csv",
        "tags": ["funnel"],
        "description": "funnel_seasonal_candidate",
    },
    {
        "name": "traffic_driven_seasonal",
        "path": DATA_DIR / "submission_traffic_driven_seasonal.csv",
        "tags": ["traffic"],
        "description": "traffic_driven_candidate",
    },
    {
        "name": "stock_conservative",
        "path": DATA_DIR / "submission_stock_scale_conservative.csv",
        "tags": ["stock"],
        "description": "stock_conservative_candidate",
    },
    {
        "name": "promo_spike",
        "path": DATA_DIR / "submission_subset_promo_spike.csv",
        "tags": ["promo_heavy", "specialist_only"],
        "description": "promo_spike_specialist",
    },
]

SAFE_AUTO_EXCLUDE_TAGS = {"segment_heavy", "specialist_only", "promo_heavy"}


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
        if isinstance(frame, pd.Series):
            self.emit(frame.to_string() if not frame.empty else "(empty)")
        else:
            self.emit(frame.to_string(index=False) if not frame.empty else "(empty)")

    def save(self) -> None:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_anti_overfit_ensemble")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger


def normalize_submission(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce").dt.normalize()
    return out


def load_sample_submission() -> pd.DataFrame:
    sample = pd.read_csv(SAMPLE_SUBMISSION_PATH, parse_dates=[DATE_COL], low_memory=False)
    sample = normalize_submission(sample)
    return sample[[DATE_COL]].copy()


def validate_submission_frame(output: pd.DataFrame, sample_submission: pd.DataFrame) -> None:
    if list(output.columns) != [DATE_COL, TARGET_COL, COGS_COL]:
        raise ValueError(f"Submission columns must be exactly {DATE_COL}, {TARGET_COL}, {COGS_COL}")
    if len(output) != len(sample_submission):
        raise ValueError("Submission row count does not match sample submission")
    if not output[DATE_COL].equals(sample_submission[DATE_COL]):
        raise ValueError("Submission dates do not match sample submission order")
    if output.isna().any().any():
        raise ValueError("Submission contains missing values")
    if (output[[TARGET_COL, COGS_COL]] < 0).any().any():
        raise ValueError("Submission contains negative Revenue/COGS")


def build_submission(dates: pd.Series, revenue: np.ndarray, ratio: float = RATIO) -> pd.DataFrame:
    output = pd.DataFrame({DATE_COL: pd.to_datetime(dates).reset_index(drop=True)})
    output[TARGET_COL] = np.maximum(0.0, np.asarray(revenue, dtype=float))
    output[COGS_COL] = np.maximum(0.0, output[TARGET_COL] * ratio)
    return output[[DATE_COL, TARGET_COL, COGS_COL]]


def save_submission_no_overwrite(path: Path, submission: pd.DataFrame, sample_submission: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    validate_submission_frame(submission, sample_submission)
    submission.to_csv(path, index=False)


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    corr = pd.Series(a, dtype=float).corr(pd.Series(b, dtype=float))
    return float(corr) if pd.notna(corr) else np.nan


def safe_ratio(num: float, den: float) -> float:
    if pd.isna(num) or pd.isna(den) or abs(float(den)) < EPS:
        return np.nan
    return float(num) / float(den)


def serialize_components(weights: dict[str, float]) -> str:
    return json.dumps({key: round(float(value), 4) for key, value in weights.items() if value > 0}, sort_keys=True)


def load_candidate_submissions(sample_submission: pd.DataFrame, logger: logging.Logger) -> dict[str, pd.DataFrame]:
    candidates: dict[str, pd.DataFrame] = {}
    for spec in CANDIDATE_SPECS:
        path = Path(spec["path"])
        if not path.exists():
            logger.info("Skipping missing candidate %s (%s)", spec["name"], path)
            continue
        df = pd.read_csv(path, parse_dates=[DATE_COL], low_memory=False)
        df = normalize_submission(df)
        validate_submission_frame(df[[DATE_COL, TARGET_COL, COGS_COL]].copy(), sample_submission)
        candidates[spec["name"]] = df[[DATE_COL, TARGET_COL, COGS_COL]].copy()
    if "current_best" not in candidates:
        raise FileNotFoundError(f"Current best anchor is required but missing: {CURRENT_BEST_PATH}")
    return candidates


def build_candidate_diagnostics(candidates: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], list[str]]:
    current = candidates["current_best"][TARGET_COL].astype(float).reset_index(drop=True)
    current_mean = float(current.mean())
    current_max = float(current.max())
    current_top10_threshold = float(np.quantile(current, 0.90))
    current_top10_mask = current >= current_top10_threshold
    current_bottom50_mask = current <= float(np.quantile(current, 0.50))

    corr_matrix = pd.DataFrame(
        {
            name: frame[TARGET_COL].astype(float).reset_index(drop=True)
            for name, frame in candidates.items()
        }
    ).corr()

    diag_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    safe_auto_candidates: list[str] = []

    for spec in CANDIDATE_SPECS:
        name = spec["name"]
        if name not in candidates:
            continue
        series = candidates[name][TARGET_COL].astype(float).reset_index(drop=True)
        tags = set(spec["tags"])
        mean_ratio = safe_ratio(float(series.mean()), current_mean)
        max_ratio = safe_ratio(float(series.max()), current_max)
        top10_mean_ratio = safe_ratio(float(series[current_top10_mask].mean()), float(current[current_top10_mask].mean()))
        bottom50_mean_ratio = safe_ratio(float(series[current_bottom50_mask].mean()), float(current[current_bottom50_mask].mean()))
        ratio_series = np.divide(series, current, out=np.full(len(series), np.nan, dtype=float), where=np.abs(current.to_numpy(dtype=float)) > EPS)

        reasons: list[str] = []
        if abs(float(mean_ratio) - 1.0) > 0.08:
            reasons.append("mean_shift_gt_8pct")
        if float(max_ratio) < 0.75 or float(max_ratio) > 1.25:
            reasons.append("max_ratio_outside_0.75_1.25")
        if float(top10_mean_ratio) < 0.75 or float(top10_mean_ratio) > 1.25:
            reasons.append("top10_spike_ratio_extreme")

        flag_segment = "segment_heavy" in tags
        flag_specialist = "specialist_only" in tags
        flag_promo = "promo_heavy" in tags
        auto_safe = not reasons and not (flag_segment or flag_specialist or flag_promo) and name != "current_best"
        if auto_safe:
            safe_auto_candidates.append(name)
        if reasons:
            excluded_rows.append(
                {
                    "candidate_name": name,
                    "reason": ";".join(reasons),
                }
            )

        row = {
            "candidate_name": name,
            "description": spec["description"],
            "tags": ",".join(sorted(tags)),
            "mean_revenue": float(series.mean()),
            "std_revenue": float(series.std(ddof=1)),
            "max_revenue": float(series.max()),
            "min_revenue": float(series.min()),
            "mean_ratio_to_current": mean_ratio,
            "max_ratio_to_current": max_ratio,
            "top10_mean_ratio_to_current": top10_mean_ratio,
            "bottom50_mean_ratio_to_current": bottom50_mean_ratio,
            "corr_to_current": safe_corr(series, current),
            "days_gt_5pct_from_current": int(np.sum(np.abs(ratio_series - 1.0) > 0.05)),
            "days_gt_10pct_from_current": int(np.sum(np.abs(ratio_series - 1.0) > 0.10)),
            "flag_segment_heavy": flag_segment,
            "flag_specialist_only": flag_specialist,
            "flag_promo_heavy": flag_promo,
            "passes_diagnostics": len(reasons) == 0,
            "safe_for_two_way": auto_safe,
            "exclusion_reasons": ";".join(reasons),
        }
        for corr_name in corr_matrix.columns:
            row[f"corr__{corr_name}"] = float(corr_matrix.loc[name, corr_name])
        diag_rows.append(row)

    diagnostics = pd.DataFrame(diag_rows).sort_values(["candidate_name"]).reset_index(drop=True)
    exclusions = pd.DataFrame(excluded_rows).sort_values(["candidate_name", "reason"]).reset_index(drop=True)
    return diagnostics, corr_matrix, excluded_rows, safe_auto_candidates


def blend_revenue(candidates: dict[str, pd.DataFrame], weights: dict[str, float]) -> np.ndarray:
    length = len(next(iter(candidates.values())))
    revenue = np.zeros(length, dtype=float)
    for name, weight in weights.items():
        if weight <= 0:
            continue
        revenue += float(weight) * candidates[name][TARGET_COL].to_numpy(dtype=float)
    return revenue


def build_blend_catalog(
    candidates: dict[str, pd.DataFrame],
    diagnostics: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}

    safe_auto_set = set(diagnostics.loc[diagnostics["safe_for_two_way"], "candidate_name"].tolist())

    for aux_name in sorted(safe_auto_set):
        for current_weight, aux_weight in [(0.95, 0.05), (0.90, 0.10), (0.85, 0.15)]:
            weights = {"current_best": current_weight, aux_name: aux_weight}
            blend_id = f"two_way__{aux_name}__{int(round(current_weight * 100))}_{int(round(aux_weight * 100))}"
            catalog[blend_id] = {
                "blend_id": blend_id,
                "mode": "two_way",
                "weights": weights,
                "current_best_weight": current_weight,
                "definition": f"{current_weight:.2f} current_best + {aux_weight:.2f} {aux_name}",
                "base_revenue": blend_revenue(candidates, weights),
            }

    def add_named_blend(blend_id: str, weights: dict[str, float]) -> None:
        if not all(name in candidates for name in weights):
            return
        catalog[blend_id] = {
            "blend_id": blend_id,
            "mode": "three_way" if len(weights) == 3 else "four_way",
            "weights": weights,
            "current_best_weight": float(weights.get("current_best", 0.0)),
            "definition": " + ".join(f"{weight:.2f} {name}" for name, weight in weights.items()),
            "base_revenue": blend_revenue(candidates, weights),
        }

    anchor_name = "direct_seasonal" if "direct_seasonal" in candidates else "cogs_ratio_8900"
    add_named_blend("tiny_A", {"current_best": 0.90, "feature_union": 0.05, "promo_spike": 0.05})
    add_named_blend("tiny_B", {"current_best": 0.90, anchor_name: 0.05, "stock_conservative": 0.05})
    add_named_blend("tiny_C", {"current_best": 0.85, anchor_name: 0.10, "promo_spike": 0.05})
    add_named_blend("tiny_D", {"current_best": 0.85, "feature_union": 0.10, "stock_conservative": 0.05})
    add_named_blend("tiny_E", {"current_best": 0.80, anchor_name: 0.10, "promo_spike": 0.05, "stock_conservative": 0.05})
    return catalog


def compute_risk_row(blend_info: dict[str, Any], scale: float, current_best_revenue: np.ndarray) -> dict[str, Any]:
    revenue = np.maximum(0.0, np.asarray(blend_info["base_revenue"], dtype=float) * float(scale))
    current = np.asarray(current_best_revenue, dtype=float)
    pct_change = np.divide(revenue - current, current, out=np.zeros_like(revenue), where=np.abs(current) > EPS)

    top10_threshold = float(np.quantile(current, 0.90))
    top10_mask = current >= top10_threshold
    bottom50_threshold = float(np.quantile(current, 0.50))
    bottom50_mask = current <= bottom50_threshold

    mean_shift = abs(float(revenue.mean()) / max(float(current.mean()), EPS) - 1.0)
    std_shift = float(np.std(pct_change))
    max_shift = float(np.max(np.abs(pct_change)))
    days_gt5 = int(np.sum(np.abs(pct_change) > 0.05))
    days_gt10 = int(np.sum(np.abs(pct_change) > 0.10))
    avg_change_top10 = float(np.mean(pct_change[top10_mask])) if top10_mask.any() else 0.0
    avg_change_bottom50 = float(np.mean(pct_change[bottom50_mask])) if bottom50_mask.any() else 0.0
    avg_change_top10_abs = float(np.mean(np.abs(pct_change[top10_mask]))) if top10_mask.any() else 0.0
    avg_change_bottom50_abs = float(np.mean(np.abs(pct_change[bottom50_mask]))) if bottom50_mask.any() else 0.0

    risk_score = 0.0
    risk_score += 900.0 * max(0.0, mean_shift - 0.02)
    risk_score += 250.0 * max(0.0, std_shift - 0.025)
    risk_score += 140.0 * max(0.0, max_shift - 0.08)
    risk_score += 0.35 * days_gt5
    risk_score += 1.10 * days_gt10
    risk_score += 320.0 * max(0.0, -avg_change_top10 - 0.01)
    risk_score += 220.0 * max(0.0, avg_change_bottom50 - 0.01)
    risk_score += 100.0 * max(0.0, avg_change_bottom50_abs - 0.03)
    risk_score += 90.0 * max(0.0, avg_change_top10_abs - 0.05)
    risk_score += 120.0 * max(0.0, 0.85 - float(blend_info["current_best_weight"]))

    return {
        "blend_id": blend_info["blend_id"],
        "mode": blend_info["mode"],
        "definition": blend_info["definition"],
        "components_json": serialize_components(blend_info["weights"]),
        "current_best_weight": float(blend_info["current_best_weight"]),
        "scale": float(scale),
        "mean_shift_from_current": mean_shift,
        "std_shift_from_current": std_shift,
        "max_shift_from_current": max_shift,
        "days_changed_gt_5pct": days_gt5,
        "days_changed_gt_10pct": days_gt10,
        "avg_change_top10_current_days": avg_change_top10,
        "avg_change_bottom50_current_days": avg_change_bottom50,
        "avg_abs_change_top10_current_days": avg_change_top10_abs,
        "avg_abs_change_bottom50_current_days": avg_change_bottom50_abs,
        "future_mean_revenue": float(revenue.mean()),
        "future_min_revenue": float(revenue.min()),
        "future_max_revenue": float(revenue.max()),
        "risk_score": float(risk_score),
    }


def choose_best_row(frame: pd.DataFrame) -> pd.Series:
    return frame.sort_values(["risk_score", "days_changed_gt_10pct", "max_shift_from_current", "std_shift_from_current"]).iloc[0]


def materialize_submission(
    selected_row: pd.Series,
    blend_catalog: dict[str, dict[str, Any]],
    sample_submission: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    blend_info = blend_catalog[str(selected_row["blend_id"])]
    revenue = np.maximum(0.0, np.asarray(blend_info["base_revenue"], dtype=float) * float(selected_row["scale"]))
    submission = build_submission(sample_submission[DATE_COL], revenue, ratio=RATIO)
    return submission, revenue


def main() -> None:
    logger = setup_logging()
    reporter = Reporter(logger)
    sample_submission = load_sample_submission()
    candidates = load_candidate_submissions(sample_submission, logger)
    diagnostics, corr_matrix, excluded_rows, safe_auto_candidates = build_candidate_diagnostics(candidates)
    diagnostics.to_csv(OUTPUT_DIAGNOSTICS_PATH, index=False)

    current_best_revenue = candidates["current_best"][TARGET_COL].to_numpy(dtype=float)
    current_stats = {
        "mean_revenue": float(current_best_revenue.mean()),
        "std_revenue": float(current_best_revenue.std(ddof=1)),
        "min_revenue": float(current_best_revenue.min()),
        "max_revenue": float(current_best_revenue.max()),
    }

    blend_catalog = build_blend_catalog(candidates, diagnostics)
    if not blend_catalog:
        raise RuntimeError("No eligible anti-overfit blends could be created.")

    risk_rows = []
    for blend_info in blend_catalog.values():
        for scale in SCALES:
            risk_rows.append(compute_risk_row(blend_info, scale, current_best_revenue))
    risk_table = pd.DataFrame(risk_rows).sort_values(["risk_score", "days_changed_gt_10pct", "max_shift_from_current"]).reset_index(drop=True)
    risk_table.to_csv(OUTPUT_RISK_PATH, index=False)

    exact_scale_mask = np.isclose(risk_table["scale"], 1.0)
    scale_up_mask = risk_table["scale"] > 1.0
    scale_down_mask = risk_table["scale"] < 1.0
    two_way_9505_mask = (risk_table["mode"] == "two_way") & exact_scale_mask & np.isclose(risk_table["current_best_weight"], 0.95)
    two_way_9010_mask = (risk_table["mode"] == "two_way") & exact_scale_mask & np.isclose(risk_table["current_best_weight"], 0.90)
    threeway_mask = risk_table["blend_id"].astype(str).str.startswith("tiny_") & exact_scale_mask

    selected_rows = [
        ("submission_anti_overfit_9505.csv", OUTPUT_9505_PATH, choose_best_row(risk_table.loc[two_way_9505_mask].copy())),
        ("submission_anti_overfit_9010.csv", OUTPUT_9010_PATH, choose_best_row(risk_table.loc[two_way_9010_mask].copy())),
        ("submission_anti_overfit_threeway.csv", OUTPUT_THREEWAY_PATH, choose_best_row(risk_table.loc[threeway_mask].copy())),
        ("submission_anti_overfit_scale_up.csv", OUTPUT_SCALE_UP_PATH, choose_best_row(risk_table.loc[scale_up_mask].copy())),
        ("submission_anti_overfit_scale_down.csv", OUTPUT_SCALE_DOWN_PATH, choose_best_row(risk_table.loc[scale_down_mask].copy())),
    ]

    metadata_rows: list[dict[str, Any]] = []
    saved_stats_rows: list[dict[str, Any]] = []
    for output_name, output_path, selected in selected_rows:
        submission, revenue = materialize_submission(selected, blend_catalog, sample_submission)
        save_submission_no_overwrite(output_path, submission, sample_submission)
        blend_info = blend_catalog[str(selected["blend_id"])]
        metadata_rows.append(
            {
                "output_file": output_name,
                "blend_id": selected["blend_id"],
                "mode": selected["mode"],
                "definition": selected["definition"],
                "components_json": selected["components_json"],
                "current_best_weight": selected["current_best_weight"],
                "scale": selected["scale"],
                "risk_score": selected["risk_score"],
                "mean_shift_from_current": selected["mean_shift_from_current"],
                "std_shift_from_current": selected["std_shift_from_current"],
                "max_shift_from_current": selected["max_shift_from_current"],
                "days_changed_gt_5pct": selected["days_changed_gt_5pct"],
                "days_changed_gt_10pct": selected["days_changed_gt_10pct"],
                "avg_change_top10_current_days": selected["avg_change_top10_current_days"],
                "avg_change_bottom50_current_days": selected["avg_change_bottom50_current_days"],
                "source_weights_json": serialize_components(blend_info["weights"]),
                "source_tags": ",".join(
                    sorted(
                        {
                            tag
                            for spec in CANDIDATE_SPECS
                            if spec["name"] in blend_info["weights"]
                            for tag in spec["tags"]
                        }
                    )
                ),
            }
        )
        saved_stats_rows.append(
            {
                "output_file": output_name,
                "revenue_mean": float(revenue.mean()),
                "revenue_min": float(revenue.min()),
                "revenue_max": float(revenue.max()),
            }
        )

    metadata = pd.DataFrame(metadata_rows)
    metadata.to_csv(OUTPUT_METADATA_PATH, index=False)
    saved_stats = pd.DataFrame(saved_stats_rows)

    top_diag_columns = [
        "candidate_name",
        "tags",
        "mean_ratio_to_current",
        "max_ratio_to_current",
        "top10_mean_ratio_to_current",
        "corr_to_current",
        "safe_for_two_way",
        "exclusion_reasons",
    ]
    top_risk_columns = [
        "blend_id",
        "mode",
        "definition",
        "scale",
        "risk_score",
        "mean_shift_from_current",
        "std_shift_from_current",
        "max_shift_from_current",
        "days_changed_gt_5pct",
        "days_changed_gt_10pct",
        "avg_change_top10_current_days",
        "avg_change_bottom50_current_days",
    ]

    reporter.emit("Current best stats")
    reporter.emit(pd.Series(current_stats).to_string())
    reporter.emit("")
    reporter.emit_frame("Candidate diagnostics table", diagnostics[top_diag_columns].sort_values(["safe_for_two_way", "corr_to_current", "candidate_name"], ascending=[False, False, True]))
    reporter.emit("")
    if excluded_rows:
        reporter.emit_frame("Excluded candidates and why", pd.DataFrame(excluded_rows))
    else:
        reporter.emit("Excluded candidates and why")
        reporter.emit("(none)")
    reporter.emit("")
    created_blend_definitions = pd.DataFrame(
        [
            {
                "blend_id": blend_info["blend_id"],
                "mode": blend_info["mode"],
                "definition": blend_info["definition"],
                "components_json": serialize_components(blend_info["weights"]),
            }
            for blend_info in blend_catalog.values()
        ]
    ).sort_values(["mode", "blend_id"])
    reporter.emit_frame("Created blend definitions", created_blend_definitions)
    reporter.emit("")
    reporter.emit_frame("Risk table top 10 safest candidates", risk_table[top_risk_columns].head(10))
    reporter.emit("")
    reporter.emit_frame("Future Revenue mean/min/max for saved submissions", saved_stats)
    reporter.emit("")
    reporter.emit("Created submission files")
    for _, output_path, _selected in selected_rows:
        reporter.emit(str(output_path))
    reporter.emit("")
    reporter.emit("Recommended upload order")
    reporter.emit(str(OUTPUT_9505_PATH))
    reporter.emit(str(OUTPUT_9010_PATH))
    reporter.emit(str(OUTPUT_THREEWAY_PATH))
    reporter.emit(str(OUTPUT_SCALE_UP_PATH))
    reporter.emit(str(OUTPUT_SCALE_DOWN_PATH))
    reporter.emit("")
    reporter.emit("Leakage safety confirmation")
    reporter.emit("This pipeline does not retrain base forecasters and uses only already-generated future submissions. It applies small blends and tiny global scaling on forecast outputs only, with COGS fixed to Revenue * 0.8900.")
    reporter.save()


if __name__ == "__main__":
    main()
