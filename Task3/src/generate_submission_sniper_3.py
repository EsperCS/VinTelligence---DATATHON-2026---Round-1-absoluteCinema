from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

DATE_COL = "Date"
REVENUE_COL = "Revenue"
COGS_COL = "COGS"
COGS_RATIO = 0.8900

DEFAULT_CURRENT_BEST = DATA_DIR / "submission_blend_direct_15_cogs8900.csv"
DEFAULT_STOCK = DATA_DIR / "submission_stock_scale_conservative.csv"
DEFAULT_SAMPLE = DATA_DIR / "sample_submission.csv"
DEFAULT_OUTPUT = DATA_DIR / "submission_sniper_3.csv"

CURRENT_WEIGHT = 0.950
STOCK_WEIGHT = 0.050
GLOBAL_SCALE = 1.0000


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="raise").dt.normalize()
    return out


def load_submission(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    df = pd.read_csv(path, parse_dates=[DATE_COL], low_memory=False)
    df = normalize_dates(df)
    required = [DATE_COL, REVENUE_COL, COGS_COL]
    if list(df.columns) != required:
        raise ValueError(f"{path.name} must have columns exactly: {required}")
    if df.isna().any().any():
        raise ValueError(f"{path.name} contains missing values")
    if (df[[REVENUE_COL, COGS_COL]] < 0).any().any():
        raise ValueError(f"{path.name} contains negative Revenue/COGS")
    return df[required].copy()


def load_sample(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing sample submission: {path}")
    df = pd.read_csv(path, parse_dates=[DATE_COL], low_memory=False)
    df = normalize_dates(df)
    if DATE_COL not in df.columns:
        raise ValueError("sample_submission.csv must contain Date column")
    return df[[DATE_COL]].copy()


def validate_against_sample(output: pd.DataFrame, sample: pd.DataFrame) -> None:
    if list(output.columns) != [DATE_COL, REVENUE_COL, COGS_COL]:
        raise ValueError("Output columns must be exactly Date, Revenue, COGS")
    if len(output) != len(sample):
        raise ValueError("Output row count does not match sample_submission.csv")
    if not output[DATE_COL].equals(sample[DATE_COL]):
        raise ValueError("Output Date order does not match sample_submission.csv")
    if output.isna().any().any():
        raise ValueError("Output contains missing values")
    if (output[[REVENUE_COL, COGS_COL]] < 0).any().any():
        raise ValueError("Output contains negative Revenue/COGS")


def build_submission(
    current_best: pd.DataFrame,
    stock: pd.DataFrame,
    sample: pd.DataFrame,
) -> pd.DataFrame:
    if not current_best[DATE_COL].equals(stock[DATE_COL]):
        raise ValueError("current_best and stock_conservative are not aligned by Date")
    if not current_best[DATE_COL].equals(sample[DATE_COL]):
        raise ValueError("Input files are not aligned with sample_submission.csv")

    revenue = (
        CURRENT_WEIGHT * current_best[REVENUE_COL].to_numpy(dtype=float)
        + STOCK_WEIGHT * stock[REVENUE_COL].to_numpy(dtype=float)
    ) * GLOBAL_SCALE
    revenue = np.maximum(revenue, 0.0)
    cogs = np.maximum(revenue * COGS_RATIO, 0.0)

    output = pd.DataFrame(
        {
            DATE_COL: sample[DATE_COL],
            REVENUE_COL: revenue,
            COGS_COL: cogs,
        }
    )
    validate_against_sample(output, sample)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate submission_sniper_3.csv from current_best and stock_conservative."
    )
    parser.add_argument(
        "--current-best",
        type=Path,
        default=DEFAULT_CURRENT_BEST,
        help=f"Path to current best submission (default: {DEFAULT_CURRENT_BEST})",
    )
    parser.add_argument(
        "--stock",
        type=Path,
        default=DEFAULT_STOCK,
        help=f"Path to stock conservative submission (default: {DEFAULT_STOCK})",
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=DEFAULT_SAMPLE,
        help=f"Path to sample submission (default: {DEFAULT_SAMPLE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    current_best = load_submission(args.current_best)
    stock = load_submission(args.stock)
    sample = load_sample(args.sample)

    output = build_submission(current_best, stock, sample)

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {args.output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    print("Created:", args.output)
    print(
        "Formula:",
        f"Revenue = ({CURRENT_WEIGHT:.3f} * current_best + {STOCK_WEIGHT:.3f} * stock_conservative) * {GLOBAL_SCALE:.4f}",
    )
    print("COGS ratio:", f"{COGS_RATIO:.4f}")
    print("Rows:", len(output))


if __name__ == "__main__":
    main()
