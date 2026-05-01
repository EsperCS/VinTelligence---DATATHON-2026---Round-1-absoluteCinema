from __future__ import annotations

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
STOCK_PATH = DATA_DIR / "submission_stock_scale_conservative.csv"
PROMO_SPIKE_PATH = DATA_DIR / "submission_subset_promo_spike.csv"

GRID_RESULTS_PATH = DATA_DIR / "final_micro_grid_results.csv"
RISK_TABLE_PATH = DATA_DIR / "final_micro_risk_table.csv"
REPORT_PATH = LOG_DIR / "final_micro_tuning_report.txt"
LOG_PATH = LOG_DIR / "train_final_micro_tuning.log"

OUTPUT_PATHS = [
    DATA_DIR / "submission_final_micro_1.csv",
    DATA_DIR / "submission_final_micro_2.csv",
    DATA_DIR / "submission_final_micro_3.csv",
    DATA_DIR / "submission_final_micro_4.csv",
    DATA_DIR / "submission_final_micro_5.csv",
]

SCALES = [0.998, 0.999, 1.000, 1.001, 1.002]
TOP10_TARGET_BIAS = 0.002


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
    logger = logging.getLogger("train_final_micro_tuning")
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
        raise ValueError("Submission columns must be exactly Date, Revenue, COGS")
    if len(output) != len(sample_submission):
        raise ValueError("Submission row count mismatch")
    if not output[DATE_COL].equals(sample_submission[DATE_COL]):
        raise ValueError("Submission Date order mismatch")
    if output.isna().any().any():
        raise ValueError("Submission contains missing values")
    if (output[[TARGET_COL, COGS_COL]] < 0).any().any():
        raise ValueError("Submission contains negative Revenue/COGS")


def load_submission(path: Path, sample_submission: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, parse_dates=[DATE_COL], low_memory=False)
    frame = normalize_submission(frame)
    validate_submission_frame(frame[[DATE_COL, TARGET_COL, COGS_COL]].copy(), sample_submission)
    return frame[[DATE_COL, TARGET_COL, COGS_COL]].copy()


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


def format_weights(weights: dict[str, float]) -> str:
    return " + ".join(f"{weight:.3f} {name}" for name, weight in weights.items() if weight > 0)


def generate_blend_catalog(include_promo: bool) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    two_way_weights = [0.97, 0.965, 0.96, 0.955, 0.95, 0.945, 0.94, 0.935, 0.93]
    for current_weight in two_way_weights:
        stock_weight = round(1.0 - current_weight, 3)
        catalog.append(
            {
                "blend_id": f"two_way_{int(round(current_weight * 1000)):04d}_{int(round(stock_weight * 1000)):04d}",
                "blend_type": "two_way",
                "weights": {"current_best": current_weight, "stock": stock_weight},
            }
        )

    if include_promo:
        three_way_specs = [
            ("threeway_A", {"current_best": 0.94, "stock": 0.05, "promo": 0.01}),
            ("threeway_B", {"current_best": 0.93, "stock": 0.05, "promo": 0.02}),
            ("threeway_C", {"current_best": 0.92, "stock": 0.05, "promo": 0.03}),
            ("threeway_D", {"current_best": 0.95, "stock": 0.03, "promo": 0.02}),
        ]
        for blend_id, weights in three_way_specs:
            catalog.append({"blend_id": blend_id, "blend_type": "three_way", "weights": weights})
    return catalog


def blend_revenue(base_signals: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    revenue = np.zeros(len(next(iter(base_signals.values()))), dtype=float)
    for name, weight in weights.items():
        revenue += float(weight) * base_signals[name]
    return revenue


def evaluate_candidate(
    blend_id: str,
    blend_type: str,
    weights: dict[str, float],
    revenue_blend: np.ndarray,
    scale: float,
    current_best: np.ndarray,
) -> dict[str, Any]:
    revenue_final = np.maximum(0.0, revenue_blend * float(scale))
    pct_change = np.divide(
        revenue_final - current_best,
        current_best,
        out=np.zeros_like(revenue_final),
        where=np.abs(current_best) > EPS,
    )

    top10_threshold = float(np.quantile(current_best, 0.90))
    top10_mask = current_best >= top10_threshold
    bottom50_threshold = float(np.quantile(current_best, 0.50))
    bottom50_mask = current_best <= bottom50_threshold

    mean_shift_pct = float(np.mean(pct_change) * 100.0)
    abs_mean_shift_pct = abs(mean_shift_pct)
    std_shift_pct = float(np.std(pct_change) * 100.0)
    max_abs_dev_pct = float(np.max(np.abs(pct_change)) * 100.0)
    pct_days_gt_2 = float(np.mean(np.abs(pct_change) > 0.02) * 100.0)
    pct_days_gt_5 = float(np.mean(np.abs(pct_change) > 0.05) * 100.0)
    top10_change_avg_pct = float(np.mean(pct_change[top10_mask]) * 100.0) if top10_mask.any() else 0.0
    bottom50_change_avg_pct = float(np.mean(pct_change[bottom50_mask]) * 100.0) if bottom50_mask.any() else 0.0

    safe_zone = (
        abs_mean_shift_pct <= 2.0
        and max_abs_dev_pct <= 8.0
        and pct_days_gt_5 <= 10.0
    )

    top10_bias_penalty = abs(top10_change_avg_pct / 100.0 - TOP10_TARGET_BIAS)
    if top10_change_avg_pct < 0:
        top10_bias_penalty += 0.02

    return {
        "blend_id": blend_id,
        "blend_type": blend_type,
        "weights_desc": format_weights(weights),
        "current_best_weight": float(weights.get("current_best", 0.0)),
        "stock_weight": float(weights.get("stock", 0.0)),
        "promo_weight": float(weights.get("promo", 0.0)),
        "scale": float(scale),
        "mean_shift_pct": mean_shift_pct,
        "abs_mean_shift_pct": abs_mean_shift_pct,
        "std_shift_pct": std_shift_pct,
        "max_abs_dev_pct": max_abs_dev_pct,
        "pct_days_changed_gt_2": pct_days_gt_2,
        "pct_days_changed_gt_5": pct_days_gt_5,
        "top10_revenue_change_avg_pct": top10_change_avg_pct,
        "bottom50_revenue_change_avg_pct": bottom50_change_avg_pct,
        "top10_bias_penalty": float(top10_bias_penalty),
        "safe_zone": bool(safe_zone),
        "future_revenue_mean": float(revenue_final.mean()),
        "future_revenue_min": float(revenue_final.min()),
        "future_revenue_max": float(revenue_final.max()),
    }


def main() -> None:
    logger = setup_logging()
    reporter = Reporter(logger)

    sample_submission = load_sample_submission()
    current_best_df = load_submission(CURRENT_BEST_PATH, sample_submission)
    stock_df = load_submission(STOCK_PATH, sample_submission)
    promo_df = None
    if PROMO_SPIKE_PATH.exists():
        promo_df = load_submission(PROMO_SPIKE_PATH, sample_submission)

    current_best = current_best_df[TARGET_COL].to_numpy(dtype=float)
    stock = stock_df[TARGET_COL].to_numpy(dtype=float)
    base_signals: dict[str, np.ndarray] = {
        "current_best": current_best,
        "stock": stock,
    }
    if promo_df is not None:
        base_signals["promo"] = promo_df[TARGET_COL].to_numpy(dtype=float)

    base_stats = pd.Series(
        {
            "mean_revenue": float(current_best.mean()),
            "std_revenue": float(current_best.std(ddof=1)),
            "min_revenue": float(current_best.min()),
            "max_revenue": float(current_best.max()),
        }
    )

    blend_catalog = generate_blend_catalog(include_promo=("promo" in base_signals))
    candidate_rows: list[dict[str, Any]] = []
    for blend in blend_catalog:
        revenue_blend = blend_revenue(base_signals, blend["weights"])
        for scale in SCALES:
            candidate_rows.append(
                evaluate_candidate(
                    blend_id=blend["blend_id"],
                    blend_type=blend["blend_type"],
                    weights=blend["weights"],
                    revenue_blend=revenue_blend,
                    scale=scale,
                    current_best=current_best,
                )
            )

    grid_results = pd.DataFrame(candidate_rows)
    grid_results.to_csv(GRID_RESULTS_PATH, index=False)

    risk_table = grid_results.sort_values(
        [
            "safe_zone",
            "abs_mean_shift_pct",
            "max_abs_dev_pct",
            "top10_bias_penalty",
            "pct_days_changed_gt_5",
            "std_shift_pct",
        ],
        ascending=[False, True, True, True, True, True],
    ).reset_index(drop=True)
    risk_table.to_csv(RISK_TABLE_PATH, index=False)

    safe_candidates = risk_table.loc[risk_table["safe_zone"]].copy()
    if safe_candidates.empty:
        raise RuntimeError("No candidates remained inside the safe zone.")

    ranked_safe = safe_candidates.sort_values(
        [
            "abs_mean_shift_pct",
            "max_abs_dev_pct",
            "top10_bias_penalty",
            "pct_days_changed_gt_5",
            "std_shift_pct",
        ],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)

    selected = ranked_safe.head(5).copy()
    for path in OUTPUT_PATHS:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")

    saved_rows: list[dict[str, Any]] = []
    for idx, (_, row) in enumerate(selected.iterrows(), start=1):
        weights = {
            "current_best": float(row["current_best_weight"]),
            "stock": float(row["stock_weight"]),
        }
        if float(row["promo_weight"]) > 0:
            weights["promo"] = float(row["promo_weight"])
        revenue_blend = blend_revenue(base_signals, weights)
        revenue_final = np.maximum(0.0, revenue_blend * float(row["scale"]))
        submission = build_submission(sample_submission[DATE_COL], revenue_final, ratio=RATIO)
        save_submission_no_overwrite(OUTPUT_PATHS[idx - 1], submission, sample_submission)
        saved_rows.append(
            {
                "rank": idx,
                "output_file": OUTPUT_PATHS[idx - 1].name,
                "blend_id": row["blend_id"],
                "weights_desc": row["weights_desc"],
                "scale": float(row["scale"]),
                "mean_revenue": float(revenue_final.mean()),
                "min_revenue": float(revenue_final.min()),
                "max_revenue": float(revenue_final.max()),
            }
        )

    saved_stats = pd.DataFrame(saved_rows)

    reporter.emit("Base current_best stats")
    reporter.emit(base_stats.to_string())
    reporter.emit("")
    reporter.emit(f"Grid size: {len(grid_results)}")
    reporter.emit(f"Filtered safe candidates count: {len(ranked_safe)}")
    reporter.emit("")
    top_columns = [
        "blend_id",
        "blend_type",
        "weights_desc",
        "scale",
        "abs_mean_shift_pct",
        "max_abs_dev_pct",
        "pct_days_changed_gt_2",
        "pct_days_changed_gt_5",
        "top10_revenue_change_avg_pct",
        "bottom50_revenue_change_avg_pct",
    ]
    reporter.emit_frame("Top 10 safest candidates", ranked_safe[top_columns].head(10))
    reporter.emit("")
    reporter.emit_frame("Final selected submissions", saved_stats)
    reporter.emit("")
    reporter.emit_frame("Future revenue stats (mean/min/max)", saved_stats[["output_file", "mean_revenue", "min_revenue", "max_revenue"]])
    reporter.emit("")
    reporter.emit("Recommended upload order")
    reporter.emit(str(OUTPUT_PATHS[0]))
    reporter.emit(str(OUTPUT_PATHS[1]))
    reporter.emit("Stop if no improvement.")
    reporter.save()


if __name__ == "__main__":
    main()
