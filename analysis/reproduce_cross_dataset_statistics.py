#!/usr/bin/env python3
"""Validate the revised MED-OS01 grid and index authoritative statistics.

This script does not fit classifiers, regenerate predictions, or refit the
mixed-effects model. It validates each 360-row dataset summary and the combined
1,080-row table, then reports the manuscript-facing statistics already stored
in ``results/statistics``. Baseline ``RequestedIntensity`` remains missing and
is never converted to zero.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILES = {
    "PIMA": ROOT / "results" / "pima" / "summary_360_configurations.csv",
    "Heart Failure": ROOT / "results" / "heart_failure" / "summary_360_configurations.csv",
    "Thoracic Surgery": ROOT / "results" / "thoracic_surgery" / "summary_360_configurations.csv",
}
COMBINED_FILE = ROOT / "results" / "combined_summary_1080_configurations.csv"
MANIFEST_FILE = ROOT / "CSV_MANIFEST.csv"
KEY_RESULTS_FILE = ROOT / "results" / "statistics" / "key_results_summary.csv"

MODELS = {
    "Logistic Regression", "Random Forest", "MLP", "XGBoost", "LightGBM"
}
SAMPLING_METHODS = {"Baseline", "SMOTE", "ADASYN"}
SMOTE_INTENSITIES = {0.25, 0.50, 0.75, 1.00}
ADASYN_INTENSITIES = {0.50, 0.75, 1.00}
THRESHOLDS = {0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70}
KEY = ["Model", "SamplingMethod", "RequestedIntensity", "Threshold"]

THRESHOLD_DEPENDENT_METRICS = [
    "F1_Mean", "Precision_Mean", "Sensitivity_Mean", "Specificity_Mean",
    "BalancedAccuracy_Mean",
]
PROBABILITY_METRICS = [
    "ROC_AUC_Mean", "AveragePrecision_Mean", "BrierScore_Mean"
]
REQUIRED_COLUMNS = [
    "Dataset", *KEY, *THRESHOLD_DEPENDENT_METRICS, *PROBABILITY_METRICS,
    "FoldRepeatN",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pima", type=Path, default=DEFAULT_FILES["PIMA"])
    parser.add_argument(
        "--heart-failure", type=Path, default=DEFAULT_FILES["Heart Failure"]
    )
    parser.add_argument(
        "--thoracic-surgery", type=Path, default=DEFAULT_FILES["Thoracic Surgery"]
    )
    parser.add_argument("--combined", type=Path, default=COMBINED_FILE)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_FILE)
    parser.add_argument(
        "--output-dir", type=Path,
        help="Optionally save validation and descriptive summaries as CSV.",
    )
    return parser.parse_args()


def _rounded_values(series: pd.Series) -> set[float]:
    values = pd.to_numeric(series, errors="coerce").dropna().round(10)
    return set(values.astype(float))


def _display_values(values: set[float]) -> str:
    return "|".join(f"{value:.2f}" for value in sorted(values))


def expected_grid() -> set[tuple[str, str, float | None, float]]:
    expected: set[tuple[str, str, float | None, float]] = set()
    for model in MODELS:
        for threshold in THRESHOLDS:
            expected.add((model, "Baseline", None, threshold))
        for intensity in SMOTE_INTENSITIES:
            for threshold in THRESHOLDS:
                expected.add((model, "SMOTE", intensity, threshold))
        for intensity in ADASYN_INTENSITIES:
            for threshold in THRESHOLDS:
                expected.add((model, "ADASYN", intensity, threshold))
    return expected


def observed_grid(frame: pd.DataFrame) -> set[tuple[str, str, float | None, float]]:
    rows: set[tuple[str, str, float | None, float]] = set()
    for model, method, intensity, threshold in frame[KEY].itertuples(
        index=False, name=None
    ):
        canonical_intensity = None if pd.isna(intensity) else round(float(intensity), 10)
        rows.add(
            (str(model), str(method), canonical_intensity, round(float(threshold), 10))
        )
    return rows


def load_and_validate_dataset(dataset: str, path: Path) -> tuple[pd.DataFrame, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {dataset} summary: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"{dataset} is missing required columns: {missing}")
    if len(frame) != 360:
        raise ValueError(f"{dataset} must contain 360 rows; found {len(frame)}")

    frame = frame.copy()
    frame["Model"] = frame["Model"].astype(str).str.strip()
    frame["SamplingMethod"] = frame["SamplingMethod"].astype(str).str.strip()
    frame["RequestedIntensity"] = pd.to_numeric(
        frame["RequestedIntensity"], errors="coerce"
    )
    frame["Threshold"] = pd.to_numeric(frame["Threshold"], errors="coerce").round(10)
    frame["FoldRepeatN"] = pd.to_numeric(frame["FoldRepeatN"], errors="coerce")

    if set(frame["Model"]) != MODELS:
        raise ValueError(f"{dataset} model set is invalid: {sorted(set(frame['Model']))}")
    if set(frame["SamplingMethod"]) != SAMPLING_METHODS:
        raise ValueError(f"{dataset} sampling-method set is invalid")
    if _rounded_values(frame["Threshold"]) != THRESHOLDS:
        raise ValueError(f"{dataset} threshold grid is invalid")

    baseline = frame["SamplingMethod"].eq("Baseline")
    if not frame.loc[baseline, "RequestedIntensity"].isna().all():
        raise ValueError(f"{dataset} Baseline intensity must be blank/NA, never zero")
    if frame.loc[~baseline, "RequestedIntensity"].isna().any():
        raise ValueError(f"{dataset} has missing non-Baseline intensity values")
    smote = frame["SamplingMethod"].eq("SMOTE")
    adasyn = frame["SamplingMethod"].eq("ADASYN")
    if _rounded_values(frame.loc[smote, "RequestedIntensity"]) != SMOTE_INTENSITIES:
        raise ValueError(f"{dataset} SMOTE intensity grid is invalid")
    if _rounded_values(frame.loc[adasyn, "RequestedIntensity"]) != ADASYN_INTENSITIES:
        raise ValueError(f"{dataset} ADASYN intensity grid is invalid")
    if not frame["FoldRepeatN"].eq(50).all():
        raise ValueError(f"{dataset} FoldRepeatN must equal 50 in every row")
    if frame.duplicated(KEY).any():
        raise ValueError(f"{dataset} contains duplicate configuration rows")

    actual_grid = observed_grid(frame)
    wanted_grid = expected_grid()
    if actual_grid != wanted_grid:
        raise ValueError(
            f"{dataset} grid differs from the expected 360 rows "
            f"(missing={len(wanted_grid - actual_grid)}, "
            f"unexpected={len(actual_grid - wanted_grid)})"
        )

    metric_columns = THRESHOLD_DEPENDENT_METRICS + PROBABILITY_METRICS
    numeric_metrics = frame[metric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric_metrics.to_numpy(dtype=float)).all():
        raise ValueError(f"{dataset} contains missing/non-finite metric values")

    # These values come directly from predicted probabilities and must be
    # constant over the nine predefined threshold rows for each configuration.
    variation = frame.groupby(
        ["Model", "SamplingMethod", "RequestedIntensity"], dropna=False
    )[PROBABILITY_METRICS].nunique(dropna=False)
    if (variation != 1).any().any():
        raise ValueError(f"{dataset} has ROC-AUC/AP/Brier values varying by threshold")

    dataset_values = set(frame["Dataset"].astype(str).str.strip())
    if dataset_values != {dataset}:
        raise ValueError(f"{dataset} file contains Dataset values {sorted(dataset_values)}")

    audit = {
        "Dataset": dataset,
        "Rows": len(frame),
        "Thresholds": _display_values(_rounded_values(frame["Threshold"])),
        "ModelCount": frame["Model"].nunique(),
        "SamplingMethods": "|".join(sorted(set(frame["SamplingMethod"]))),
        "BaselineIntensity": "blank/NA",
        "SMOTEIntensities": _display_values(SMOTE_INTENSITIES),
        "ADASYNIntensities": _display_values(ADASYN_INTENSITIES),
        "FoldRepeatN": int(frame["FoldRepeatN"].iloc[0]),
        "Status": "PASS",
    }
    return frame, audit


def validate_combined(combined_path: Path, datasets: dict[str, pd.DataFrame]) -> None:
    if not combined_path.is_file():
        raise FileNotFoundError(f"Missing combined summary: {combined_path}")
    combined = pd.read_csv(combined_path, encoding="utf-8-sig")
    if len(combined) != 1080:
        raise ValueError(f"Combined summary must have 1,080 rows; found {len(combined)}")
    expected = pd.concat(datasets.values(), ignore_index=True)
    if list(combined.columns) != list(expected.columns):
        raise ValueError("Combined summary columns differ from the dataset summaries")
    pd.testing.assert_frame_equal(
        combined.reset_index(drop=True), expected.reset_index(drop=True),
        check_dtype=False, check_exact=True,
    )


def validate_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing CSV manifest: {path}")
    manifest = pd.read_csv(path, encoding="utf-8-sig")
    required = {"RepoPath", "DataRows", "Columns"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"CSV manifest is missing columns: {missing}")
    for row in manifest.itertuples(index=False):
        csv_path = ROOT / row.RepoPath
        if not csv_path.is_file():
            raise FileNotFoundError(f"Manifest-listed CSV is missing: {csv_path}")
        frame = pd.read_csv(csv_path, encoding="utf-8-sig")
        if len(frame) != int(row.DataRows) or len(frame.columns) != int(row.Columns):
            raise ValueError(
                f"Manifest mismatch for {row.RepoPath}: "
                f"{len(frame)} rows/{len(frame.columns)} columns; expected "
                f"{int(row.DataRows)}/{int(row.Columns)}"
            )
    return manifest


def descriptive_summary(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset, frame in datasets.items():
        for metric in THRESHOLD_DEPENDENT_METRICS + PROBABILITY_METRICS:
            rows.append(
                {
                    "Dataset": dataset,
                    "Metric": metric,
                    "MeanAcrossConfigurationRows": pd.to_numeric(frame[metric]).mean(),
                    "ConfigurationRows": len(frame),
                    "Interpretation": "descriptive only",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    files = {
        "PIMA": args.pima,
        "Heart Failure": args.heart_failure,
        "Thoracic Surgery": args.thoracic_surgery,
    }
    datasets: dict[str, pd.DataFrame] = {}
    audit_rows = []
    for dataset, path in files.items():
        frame, audit = load_and_validate_dataset(dataset, path)
        datasets[dataset] = frame
        audit_rows.append(audit)

    validate_combined(args.combined, datasets)
    manifest = validate_manifest(args.manifest)
    audit = pd.DataFrame(audit_rows)
    summary = descriptive_summary(datasets)
    if not KEY_RESULTS_FILE.is_file():
        raise FileNotFoundError(f"Missing authoritative key results: {KEY_RESULTS_FILE}")
    key_results = pd.read_csv(KEY_RESULTS_FILE, encoding="utf-8-sig")

    if args.output_dir:
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        audit.to_csv(output_dir / "revised_grid_validation.csv", index=False)
        summary.to_csv(output_dir / "descriptive_metric_summary.csv", index=False)

    print("MED-OS01 revised-result validation: PASS")
    print(audit.to_string(index=False))
    print("Combined summary: 1080 rows (PASS)")
    print(f"Manifest: {len(manifest)} CSV files verified (PASS)")
    print("\nAuthoritative manuscript-facing key results (not recomputed):")
    print(key_results.to_string(index=False))


if __name__ == "__main__":
    main()
