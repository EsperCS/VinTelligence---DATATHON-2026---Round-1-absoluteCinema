from __future__ import annotations

import logging
from pathlib import Path

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

DIAGNOSTICS_PATH = DATA_DIR / "final_sniper_grid_diagnostics.csv"
REPORT_PATH = LOG_DIR / "final_sniper_grid_report.txt"
LOG_PATH = LOG_DIR / "train_final_sniper_grid.log"

CANDIDATE_SPECS = [
    {
        "candidate_id": "sniper_1",
        "output_path": DATA_DIR / "submission_sniper_1.csv",
        "current_weight": 0.952,
        "stock_weight": 0.048,
        "scale": 1.0000,
    },
    {
        "candidate_id": "sniper_2",
        "output_path": DATA_DIR / "submission_sniper_2.csv",
        "current_weight": 0.950,
        "stock_weight": 0.050,
        "scale": 0.9995,
    },
    {
        "candidate_id": "sniper_3",
        "output_path": DATA_DIR / "submission_sniper_3.csv",
        "current_weight": 0.950,
        "stock_weight": 0.050,
        "scale": 1.0000,
    },
    {
        "candidate_id": "sniper_4",
        "output_path": DATA_DIR / "submission_sniper_4.csv",
        "current_weight": 0.948,
        "stock_weight": 0.052,
        "scale": 0.9995,
    },
    {
        "candidate_id": "sniper_5",
        "output_path": DATA_DIR / "submission_sniper_5.csv",
        "current_weight": 0.954,
        "stock_weight": 0.046,
        "scale": 1.0005,
    },
]

RECOMMENDED_UPLOAD_ORDER = [
    "submission_sniper_3.csv",
    "submission_sniper_2.csv",
    "submission_sniper_1.csv",
    "submission_sniper_4.csv",
    "submission_sniper_5.csv",
]


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
    logger = logging.getLogger("train_final_sniper_grid")
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


def build_submission(dates: pd.Series, revenue: np.ndarray) -> pd.DataFrame:
    output = pd.DataFrame({DATE_COL: pd.to_datetime(dates).reset_index(drop=True)})
    output[TARGET_COL] = np.maximum(0.0, np.asarray(revenue, dtype=float))
    output[COGS_COL] = np.maximum(0.0, output[TARGET_COL] * RATIO)
    return output[[DATE_COL, TARGET_COL, COGS_COL]]


def save_submission_no_overwrite(path: Path, submission: pd.DataFrame, sample_submission: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    validate_submission_frame(submission, sample_submission)
    submission.to_csv(path, index=False)


def build_formula(current_weight: float, stock_weight: float, scale: float) -> str:
    return (
        f"Revenue = ({current_weight:.3f} * current_best + "
        f"{stock_weight:.3f} * stock_conservative) * {scale:.4f}"
    )


def evaluate_candidate(
    candidate_id: str,
    formula: str,
    output_file: str,
    revenue: np.ndarray,
    current_best: np.ndarray,
) -> dict[str, float | int | str]:
    pct_change = np.divide(
        revenue - current_best,
        current_best,
        out=np.zeros_like(revenue),
        where=np.abs(current_best) > EPS,
    )
    top10_threshold = float(np.quantile(current_best, 0.90))
    bottom50_threshold = float(np.quantile(current_best, 0.50))
    top10_mask = current_best >= top10_threshold
    bottom50_mask = current_best <= bottom50_threshold

    return {
        "candidate_id": candidate_id,
        "output_file": output_file,
        "formula": formula,
        "mean_revenue": float(revenue.mean()),
        "min_revenue": float(revenue.min()),
        "max_revenue": float(revenue.max()),
        "mean_shift_pct": float(np.mean(pct_change) * 100.0),
        "max_abs_day_shift_pct": float(np.max(np.abs(pct_change)) * 100.0),
        "days_changed_gt_2pct": int(np.sum(np.abs(pct_change) > 0.02)),
        "days_changed_gt_5pct": int(np.sum(np.abs(pct_change) > 0.05)),
        "top10_days_avg_shift_pct": float(np.mean(pct_change[top10_mask]) * 100.0) if top10_mask.any() else 0.0,
        "bottom50_days_avg_shift_pct": float(np.mean(pct_change[bottom50_mask]) * 100.0) if bottom50_mask.any() else 0.0,
    }


def main() -> None:
    logger = setup_logging()
    reporter = Reporter(logger)

    sample_submission = load_sample_submission()
    current_best_df = load_submission(CURRENT_BEST_PATH, sample_submission)
    stock_df = load_submission(STOCK_PATH, sample_submission)

    current_best = current_best_df[TARGET_COL].to_numpy(dtype=float)
    stock = stock_df[TARGET_COL].to_numpy(dtype=float)

    base_stats = pd.Series(
        {
            "mean_revenue": float(current_best.mean()),
            "min_revenue": float(current_best.min()),
            "max_revenue": float(current_best.max()),
        }
    )

    for spec in CANDIDATE_SPECS:
        output_path = Path(spec["output_path"])
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    diagnostics_rows: list[dict[str, float | int | str]] = []
    created_files: list[str] = []

    for spec in CANDIDATE_SPECS:
        current_weight = float(spec["current_weight"])
        stock_weight = float(spec["stock_weight"])
        scale = float(spec["scale"])
        output_path = Path(spec["output_path"])

        blended = current_weight * current_best + stock_weight * stock
        revenue_final = np.maximum(0.0, blended * scale)
        submission = build_submission(sample_submission[DATE_COL], revenue_final)
        save_submission_no_overwrite(output_path, submission, sample_submission)
        created_files.append(str(output_path))

        formula = build_formula(current_weight, stock_weight, scale)
        diagnostics_rows.append(
            evaluate_candidate(
                candidate_id=str(spec["candidate_id"]),
                formula=formula,
                output_file=output_path.name,
                revenue=revenue_final,
                current_best=current_best,
            )
        )

    diagnostics = pd.DataFrame(diagnostics_rows)
    diagnostics.to_csv(DIAGNOSTICS_PATH, index=False)

    formula_frame = pd.DataFrame(
        [
            {
                "candidate_id": spec["candidate_id"],
                "output_file": Path(spec["output_path"]).name,
                "formula": build_formula(
                    float(spec["current_weight"]),
                    float(spec["stock_weight"]),
                    float(spec["scale"]),
                ),
            }
            for spec in CANDIDATE_SPECS
        ]
    )

    report_columns = [
        "candidate_id",
        "output_file",
        "mean_revenue",
        "min_revenue",
        "max_revenue",
        "mean_shift_pct",
        "max_abs_day_shift_pct",
        "days_changed_gt_2pct",
        "days_changed_gt_5pct",
        "top10_days_avg_shift_pct",
        "bottom50_days_avg_shift_pct",
    ]

    reporter.emit("All 5 candidate formulas")
    reporter.emit_frame("", formula_frame)
    reporter.emit("")
    reporter.emit("Base current_best stats")
    reporter.emit(base_stats.to_string())
    reporter.emit("")
    reporter.emit_frame("Diagnostics table", diagnostics[report_columns])
    reporter.emit("")
    reporter.emit("Created files")
    for path in created_files:
        reporter.emit(path)
    reporter.emit(str(DIAGNOSTICS_PATH))
    reporter.emit(str(REPORT_PATH))
    reporter.emit("")
    reporter.emit("Recommended upload order")
    for name in RECOMMENDED_UPLOAD_ORDER:
        reporter.emit(name)
    reporter.save()


if __name__ == "__main__":
    main()
