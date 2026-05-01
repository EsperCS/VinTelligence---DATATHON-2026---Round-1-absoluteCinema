from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"

DATE_COL = "Date"
REVENUE_COL = "Revenue"
COGS_COL = "COGS"
COGS_RATIO = 0.8900

SAMPLE_PATH = DATA_DIR / "sample_submission.csv"

META_SCALE_CONSERVATIVE_PATH = DATA_DIR / "submission_meta_scale_conservative.csv"
COGS_RATIO_8900_PATH = DATA_DIR / "submission_cogs_ratio_8900.csv"
DIRECT_SEASONAL_8900_PATH = DATA_DIR / "submission_direct_seasonal_ratio_8900.csv"
PRUNED_SUBMISSION_PATH = DATA_DIR / "submission_pruned_ensemble.csv"
SPIKE_SUBMISSION_PATH = DATA_DIR / "submission_spike_aware.csv"
PROMO_REGIME_SUBMISSION_PATH = DATA_DIR / "submission_promo_regime.csv"
REGIME_ULTRA_15_PATH = DATA_DIR / "submission_regime_ultra_15.csv"
CURRENT_BEST_PATH = DATA_DIR / "submission_blend_direct_15_cogs8900.csv"
STOCK_CONSERVATIVE_PATH = DATA_DIR / "submission_stock_scale_conservative.csv"
FINAL_OUTPUT_PATH = DATA_DIR / "submission_sniper_3.csv"


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="raise").dt.normalize()
    return out


def load_submission(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path, parse_dates=[DATE_COL], low_memory=False)
    df = normalize_dates(df)
    required = [DATE_COL, REVENUE_COL, COGS_COL]
    if list(df.columns) != required:
        raise ValueError(f"{path.name} must have columns exactly {required}")
    if df.isna().any().any():
        raise ValueError(f"{path.name} contains missing values")
    if (df[[REVENUE_COL, COGS_COL]] < 0).any().any():
        raise ValueError(f"{path.name} contains negative values")
    return df


def load_sample(path: Path = SAMPLE_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing sample submission: {path}")
    sample = pd.read_csv(path, parse_dates=[DATE_COL], low_memory=False)
    sample = normalize_dates(sample)
    return sample[[DATE_COL]].copy()


def validate_against_sample(output: pd.DataFrame, sample: pd.DataFrame) -> None:
    if list(output.columns) != [DATE_COL, REVENUE_COL, COGS_COL]:
        raise ValueError("Output columns must be exactly Date, Revenue, COGS")
    if len(output) != len(sample):
        raise ValueError("Output row count mismatch")
    if not output[DATE_COL].equals(sample[DATE_COL]):
        raise ValueError("Output Date order mismatch")
    if output.isna().any().any():
        raise ValueError("Output contains missing values")
    if (output[[REVENUE_COL, COGS_COL]] < 0).any().any():
        raise ValueError("Output contains negative values")


def save_submission(path: Path, submission: pd.DataFrame, sample: pd.DataFrame, force: bool) -> None:
    validate_against_sample(submission, sample)
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    submission.to_csv(path, index=False)


def run_script(script_name: str, outputs: list[Path], force: bool) -> None:
    if outputs and all(path.exists() for path in outputs) and not force:
        print(f"[skip] {script_name} -> outputs already exist")
        return
    print(f"[run ] {script_name}")
    subprocess.run(
        [sys.executable, str(SRC_DIR / script_name)],
        cwd=str(PROJECT_ROOT),
        check=True,
    )
    missing = [str(path) for path in outputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{script_name} completed but missing outputs: {missing}")


def create_cogs_ratio_8900(force: bool) -> None:
    sample = load_sample()
    source = load_submission(META_SCALE_CONSERVATIVE_PATH)
    validate_against_sample(source[[DATE_COL, REVENUE_COL, COGS_COL]].copy(), sample)
    output = source[[DATE_COL, REVENUE_COL]].copy()
    output[COGS_COL] = np.maximum(output[REVENUE_COL].to_numpy(dtype=float) * COGS_RATIO, 0.0)
    output = output[[DATE_COL, REVENUE_COL, COGS_COL]]
    save_submission(COGS_RATIO_8900_PATH, output, sample, force=force)
    print(f"[make] {COGS_RATIO_8900_PATH.name}")


def create_current_best_blend(force: bool) -> None:
    sample = load_sample()
    current_anchor = load_submission(COGS_RATIO_8900_PATH)
    direct = load_submission(DIRECT_SEASONAL_8900_PATH)
    if not current_anchor[DATE_COL].equals(direct[DATE_COL]):
        raise ValueError("cogs_ratio_8900 and direct_seasonal_8900 are not aligned by Date")

    revenue = (
        0.85 * current_anchor[REVENUE_COL].to_numpy(dtype=float)
        + 0.15 * direct[REVENUE_COL].to_numpy(dtype=float)
    )
    output = pd.DataFrame(
        {
            DATE_COL: sample[DATE_COL],
            REVENUE_COL: np.maximum(revenue, 0.0),
        }
    )
    output[COGS_COL] = np.maximum(output[REVENUE_COL].to_numpy(dtype=float) * COGS_RATIO, 0.0)
    output = output[[DATE_COL, REVENUE_COL, COGS_COL]]
    save_submission(CURRENT_BEST_PATH, output, sample, force=force)
    print(f"[make] {CURRENT_BEST_PATH.name}")


def create_regime_ultra_15(force: bool) -> None:
    sample = load_sample()
    pruned = load_submission(PRUNED_SUBMISSION_PATH)
    spike = load_submission(SPIKE_SUBMISSION_PATH)
    regime = load_submission(PROMO_REGIME_SUBMISSION_PATH)

    if not pruned[DATE_COL].equals(spike[DATE_COL]) or not pruned[DATE_COL].equals(regime[DATE_COL]):
        raise ValueError("pruned, spike, and promo_regime submissions are not aligned by Date")

    revenue = (
        0.425 * pruned[REVENUE_COL].to_numpy(dtype=float)
        + 0.425 * spike[REVENUE_COL].to_numpy(dtype=float)
        + 0.150 * regime[REVENUE_COL].to_numpy(dtype=float)
    )
    output = pd.DataFrame(
        {
            DATE_COL: sample[DATE_COL],
            REVENUE_COL: np.maximum(revenue, 0.0),
        }
    )
    output[COGS_COL] = np.maximum(output[REVENUE_COL].to_numpy(dtype=float) * COGS_RATIO, 0.0)
    output = output[[DATE_COL, REVENUE_COL, COGS_COL]]
    save_submission(REGIME_ULTRA_15_PATH, output, sample, force=force)
    print(f"[make] {REGIME_ULTRA_15_PATH.name}")


def create_submission_sniper_3(force: bool) -> None:
    sample = load_sample()
    current_best = load_submission(CURRENT_BEST_PATH)
    stock = load_submission(STOCK_CONSERVATIVE_PATH)
    if not current_best[DATE_COL].equals(stock[DATE_COL]):
        raise ValueError("current_best and stock_conservative are not aligned by Date")

    revenue = (
        0.95 * current_best[REVENUE_COL].to_numpy(dtype=float)
        + 0.05 * stock[REVENUE_COL].to_numpy(dtype=float)
    )
    output = pd.DataFrame(
        {
            DATE_COL: sample[DATE_COL],
            REVENUE_COL: np.maximum(revenue, 0.0),
        }
    )
    output[COGS_COL] = np.maximum(output[REVENUE_COL].to_numpy(dtype=float) * COGS_RATIO, 0.0)
    output = output[[DATE_COL, REVENUE_COL, COGS_COL]]
    save_submission(FINAL_OUTPUT_PATH, output, sample, force=force)
    print(f"[make] {FINAL_OUTPUT_PATH.name}")


def run_fast(force: bool) -> None:
    create_submission_sniper_3(force=force)


def run_full(force: bool) -> None:
    steps = [
        ("final_feature_prune_and_retrain.py", [DATA_DIR / "submission_pruned_ensemble.csv", DATA_DIR / "pruned_ensemble_validation_predictions.csv"]),
        ("train_spike_aware_model.py", [DATA_DIR / "submission_spike_aware.csv", DATA_DIR / "spike_model_validation_predictions.csv"]),
        ("train_promo_regime_model.py", [DATA_DIR / "submission_promo_regime.csv", DATA_DIR / "promo_regime_validation_predictions.csv"]),
    ]

    for script_name, outputs in steps:
        run_script(script_name, outputs, force=force)

    create_regime_ultra_15(force=force)

    later_steps = [
        ("train_spike_probability_gate.py", [DATA_DIR / "submission_spike_gate_aggressive.csv", DATA_DIR / "spike_gate_validation_predictions.csv"]),
        ("adaptive_scaling_layer.py", [DATA_DIR / "submission_adaptive_scale_plus.csv"]),
        ("train_meta_scaling.py", [META_SCALE_CONSERVATIVE_PATH]),
        ("train_direct_seasonal_residual_model.py", [DIRECT_SEASONAL_8900_PATH]),
        ("final_micro_calibration.py", [DATA_DIR / "final_micro_calibration_validation_predictions.csv"]),
    ]

    for script_name, outputs in later_steps:
        run_script(script_name, outputs, force=force)

    create_cogs_ratio_8900(force=force)
    create_current_best_blend(force=force)

    run_script(
        "train_stock_aware_scaling.py",
        [STOCK_CONSERVATIVE_PATH],
        force=force,
    )
    create_submission_sniper_3(force=force)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce submission_sniper_3.csv")
    parser.add_argument(
        "--mode",
        choices=["fast", "full"],
        default="fast",
        help="fast: use existing intermediate artifacts; full: regenerate upstream artifacts from raw data using existing pipeline scripts",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting generated outputs in this reproduction workflow",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "fast":
        run_fast(force=args.force)
    else:
        run_full(force=args.force)


if __name__ == "__main__":
    main()
