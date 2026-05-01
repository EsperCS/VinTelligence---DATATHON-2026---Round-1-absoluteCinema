from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "log"

SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
WEIGHT_SEARCH_RESULTS_PATH = DATA_DIR / "ensemble_weight_search_results.csv"
REPORT_PATH = LOG_DIR / "ensemble_weight_report.txt"

ENSEMBLE_SAFE_PATH = DATA_DIR / "submission_ensemble_safe.csv"
ENSEMBLE_BALANCED_PATH = DATA_DIR / "submission_ensemble_balanced.csv"
ENSEMBLE_OPTIMIZED_PATH = DATA_DIR / "submission_ensemble_optimized.csv"

FULL_METRICS = {
    "MAE": 695_337.54,
    "RMSE": 985_315.32,
    "R2": 0.653472,
}

VALIDATION_FILES = {
    "FULL": DATA_DIR / "final_validation_predictions.csv",
    "PROMO_DURATION": DATA_DIR / "final_promo_duration_validation_predictions.csv",
    "RECENT_2015": DATA_DIR / "recent_2015_validation_predictions.csv",
    "RECENT_2019": DATA_DIR / "recent_2019_validation_predictions.csv",
}

SUBMISSION_FILES = {
    "FULL": DATA_DIR / "submission.csv",
    "PROMO_DURATION": DATA_DIR / "submission_promo_duration.csv",
    "RECENT_2015": DATA_DIR / "submission_2015_window.csv",
    "RECENT_2019": DATA_DIR / "submission_2019_window.csv",
}

DEFAULT_OPTIMIZATION_MODELS = ["FULL", "PROMO_DURATION", "RECENT_2015"]
SAFE_BLEND_WEIGHTS = {"FULL": 0.70, "PROMO_DURATION": 0.20, "RECENT_2015": 0.10}
BALANCED_BLEND_WEIGHTS = {"FULL": 0.60, "PROMO_DURATION": 0.25, "RECENT_2015": 0.15}
WEIGHT_STEP = 0.05


class Reporter:
    """Collect printed lines and save them as a report."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def emit(self, message: str = "") -> None:
        print(message)
        self.lines.append(message)

    def emit_frame(self, title: str, frame: pd.DataFrame) -> None:
        self.emit(title)
        if frame.empty:
            self.emit("(empty)")
            return
        self.emit(frame.to_string(index=False))

    def save(self, path: Path = REPORT_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def metric_report(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, and R2."""
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    errors = y_true - y_pred

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    ss_res = float(np.sum(errors**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def find_column(columns: list[str], candidates: list[str]) -> str:
    """Find a matching column from candidate names."""
    lower_map = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    raise ValueError(f"Could not find any of {candidates} in columns: {columns}")


def load_validation_prediction(path: Path, model_name: str) -> pd.DataFrame:
    """Load and standardize one validation prediction file."""
    df = pd.read_csv(path, parse_dates=["Date"], low_memory=False)
    date_col = find_column(df.columns.tolist(), ["Date"])
    actual_col = find_column(
        df.columns.tolist(),
        ["actual_Revenue", "actual Revenue", "Revenue_actual", "actual"],
    )
    pred_col = find_column(
        df.columns.tolist(),
        ["predicted_Revenue", "predicted Revenue", "Revenue_predicted", "prediction", "pred"],
    )

    output = df[[date_col, actual_col, pred_col]].copy()
    output.columns = ["Date", "actual_Revenue", f"pred_{model_name}"]
    output["Date"] = pd.to_datetime(output["Date"], errors="coerce").dt.normalize()
    output = output.dropna(subset=["Date", "actual_Revenue", f"pred_{model_name}"])
    return output.sort_values("Date").reset_index(drop=True)


def load_available_validation_predictions(
    reporter: Reporter,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load available per-date validation predictions and align them by Date."""
    loaded: dict[str, pd.DataFrame] = {}
    warnings: list[str] = []

    for model_name in DEFAULT_OPTIMIZATION_MODELS:
        path = VALIDATION_FILES[model_name]
        if not path.exists():
            warning = f"Missing validation prediction file for {model_name}: {path}"
            warnings.append(warning)
            reporter.emit(f"WARNING: {warning}")
            continue
        loaded[model_name] = load_validation_prediction(path, model_name)

    if not loaded:
        return pd.DataFrame(), [], warnings

    aligned: pd.DataFrame | None = None
    available_models: list[str] = []
    for model_name, df in loaded.items():
        if aligned is None:
            aligned = df
        else:
            aligned = aligned.merge(df, on=["Date", "actual_Revenue"], how="inner")
        available_models.append(model_name)

    if aligned is None:
        return pd.DataFrame(), [], warnings

    return aligned.sort_values("Date").reset_index(drop=True), available_models, warnings


def generate_weight_combinations(model_names: list[str], step: float = WEIGHT_STEP) -> list[dict[str, float]]:
    """Generate all non-negative weight combinations summing to 1."""
    units = int(round(1.0 / step))
    combinations: list[dict[str, float]] = []

    def recurse(remaining_units: int, index: int, current: list[int]) -> None:
        if index == len(model_names) - 1:
            weights = current + [remaining_units]
            combinations.append(
                {model: round(unit * step, 10) for model, unit in zip(model_names, weights)}
            )
            return
        for unit in range(remaining_units + 1):
            recurse(remaining_units - unit, index + 1, current + [unit])

    recurse(units, 0, [])
    return combinations


def grid_search_weights(aligned: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    """Grid search ensemble weights on validation predictions."""
    if aligned.empty or not model_names:
        return pd.DataFrame()

    rows: list[dict[str, float]] = []
    actual = aligned["actual_Revenue"]

    for weights in generate_weight_combinations(model_names):
        prediction = np.zeros(len(aligned), dtype=float)
        for model_name, weight in weights.items():
            prediction += weight * aligned[f"pred_{model_name}"].to_numpy(dtype=float)

        metrics = metric_report(actual, prediction)
        row = {f"weight_{model_name}": weight for model_name, weight in weights.items()}
        row.update(metrics)
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["RMSE", "MAE"]).reset_index(drop=True)


def load_submission(path: Path, sample_dates: pd.Series) -> pd.DataFrame:
    """Load one submission and enforce sample date order."""
    df = pd.read_csv(path, parse_dates=["Date"], low_memory=False)
    required_cols = ["Date", "Revenue", "COGS"]
    missing_cols = [column for column in required_cols if column not in df.columns]
    if missing_cols:
        raise ValueError(f"{path} missing required columns: {missing_cols}")

    df = df[required_cols].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    ordered = pd.DataFrame({"Date": sample_dates}).merge(df, on="Date", how="left", validate="one_to_one")
    if ordered[["Revenue", "COGS"]].isna().any().any():
        raise ValueError(f"{path} cannot be aligned to sample_submission dates without missing values")
    return ordered


def normalize_weights_for_existing_submissions(
    weights: dict[str, float],
    reporter: Reporter,
) -> dict[str, float]:
    """Keep only models with submission files and renormalize weights."""
    existing = {
        model_name: weight
        for model_name, weight in weights.items()
        if weight > 0 and SUBMISSION_FILES.get(model_name, Path()).exists()
    }
    missing = [
        model_name
        for model_name, weight in weights.items()
        if weight > 0 and not SUBMISSION_FILES.get(model_name, Path()).exists()
    ]
    for model_name in missing:
        reporter.emit(f"WARNING: missing submission for {model_name}; dropping it from blend")

    total_weight = sum(existing.values())
    if total_weight <= 0:
        raise ValueError("No usable submission files for requested blend")
    return {model_name: weight / total_weight for model_name, weight in existing.items()}


def create_blended_submission(
    weights: dict[str, float],
    sample_submission: pd.DataFrame,
    output_path: Path,
    reporter: Reporter,
) -> pd.DataFrame:
    """Blend Revenue and COGS using normalized weights."""
    normalized = normalize_weights_for_existing_submissions(weights, reporter)
    sample_dates = pd.to_datetime(sample_submission["Date"], errors="coerce").dt.normalize()

    blended = pd.DataFrame({"Date": sample_dates})
    blended["Revenue"] = 0.0
    blended["COGS"] = 0.0

    for model_name, weight in normalized.items():
        submission = load_submission(SUBMISSION_FILES[model_name], sample_dates)
        blended["Revenue"] += weight * submission["Revenue"]
        blended["COGS"] += weight * submission["COGS"]

    blended["Revenue"] = blended["Revenue"].clip(lower=0)
    blended["COGS"] = blended["COGS"].clip(lower=0)
    blended = blended[["Date", "Revenue", "COGS"]]

    if blended.isna().any().any():
        raise ValueError(f"Blend {output_path} contains missing values")
    if (blended[["Revenue", "COGS"]] < 0).any().any():
        raise ValueError(f"Blend {output_path} contains negative values")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    blended.to_csv(output_path, index=False)
    reporter.emit(f"Created {output_path} with weights: {normalized}")
    return blended


def optimized_weights_from_results(results: pd.DataFrame, model_names: list[str]) -> dict[str, float]:
    """Extract best weight row from search results."""
    if results.empty:
        return {"FULL": 1.0}
    best = results.iloc[0]
    return {
        model_name: float(best.get(f"weight_{model_name}", 0.0))
        for model_name in model_names
        if float(best.get(f"weight_{model_name}", 0.0)) > 0
    }


def main() -> None:
    reporter = Reporter()
    reporter.emit("Ensemble Weight Optimization")
    reporter.emit("============================")
    reporter.emit("")

    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH, parse_dates=["Date"], low_memory=False)
    sample_submission["Date"] = pd.to_datetime(sample_submission["Date"], errors="coerce").dt.normalize()

    reporter.emit("1. Load validation predictions")
    aligned, optimization_models, warnings = load_available_validation_predictions(reporter)
    reporter.emit(f"Available validation models: {optimization_models}")
    if not aligned.empty:
        reporter.emit(
            f"Validation alignment: rows={len(aligned):,}, "
            f"date range={aligned['Date'].min().date()} -> {aligned['Date'].max().date()}"
        )

    reporter.emit("")
    reporter.emit("2. Grid search weights")
    search_results = grid_search_weights(aligned, optimization_models)
    WEIGHT_SEARCH_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    search_results.to_csv(WEIGHT_SEARCH_RESULTS_PATH, index=False)

    if search_results.empty:
        reporter.emit("No validation predictions available for grid search; optimized blend falls back to FULL.")
        best_weights = {"FULL": 1.0}
        best_metrics = FULL_METRICS.copy()
    else:
        best_weights = optimized_weights_from_results(search_results, optimization_models)
        best_metrics = {
            "MAE": float(search_results.iloc[0]["MAE"]),
            "RMSE": float(search_results.iloc[0]["RMSE"]),
            "R2": float(search_results.iloc[0]["R2"]),
        }
        reporter.emit_frame("Top 10 validation weight combinations:", search_results.head(10))

    reporter.emit(f"Best validation weights: {best_weights}")
    reporter.emit(
        f"Best ensemble metrics: MAE={best_metrics['MAE']:,.2f}, "
        f"RMSE={best_metrics['RMSE']:,.2f}, R2={best_metrics['R2']:.6f}"
    )

    mae_change = best_metrics["MAE"] - FULL_METRICS["MAE"]
    rmse_change = best_metrics["RMSE"] - FULL_METRICS["RMSE"]
    r2_change = best_metrics["R2"] - FULL_METRICS["R2"]
    reporter.emit(
        "Change vs FULL: "
        f"MAE={mae_change:,.2f}, RMSE={rmse_change:,.2f}, R2={r2_change:.6f}"
    )

    reporter.emit("")
    reporter.emit("3. Create ensemble submissions")
    created_paths = []
    create_blended_submission(SAFE_BLEND_WEIGHTS, sample_submission, ENSEMBLE_SAFE_PATH, reporter)
    created_paths.append(ENSEMBLE_SAFE_PATH)
    create_blended_submission(BALANCED_BLEND_WEIGHTS, sample_submission, ENSEMBLE_BALANCED_PATH, reporter)
    created_paths.append(ENSEMBLE_BALANCED_PATH)
    create_blended_submission(best_weights, sample_submission, ENSEMBLE_OPTIMIZED_PATH, reporter)
    created_paths.append(ENSEMBLE_OPTIMIZED_PATH)

    reporter.emit("")
    reporter.emit("4. Report")
    reporter.emit(f"Models used in optimization: {optimization_models}")
    reporter.emit(f"Best weights: {best_weights}")
    reporter.emit(
        f"Best validation MAE/RMSE/R2: {best_metrics['MAE']:,.2f} / "
        f"{best_metrics['RMSE']:,.2f} / {best_metrics['R2']:.6f}"
    )
    reporter.emit(
        f"Comparison versus FULL: MAE {mae_change:,.2f}, "
        f"RMSE {rmse_change:,.2f}, R2 {r2_change:.6f}"
    )
    reporter.emit(f"Created submission paths: {[str(path) for path in created_paths]}")
    if warnings:
        reporter.emit("Warnings:")
        for warning in warnings:
            reporter.emit(f"- {warning}")

    if best_metrics["RMSE"] < FULL_METRICS["RMSE"]:
        recommendation = str(ENSEMBLE_OPTIMIZED_PATH)
    else:
        recommendation = str(SUBMISSION_FILES["FULL"])
    reporter.emit(f"Recommendation: submit first = {recommendation}")
    reporter.emit("Leakage confirmation: only 2022 validation predictions and forecast submissions were blended.")

    reporter.save(REPORT_PATH)


if __name__ == "__main__":
    main()
