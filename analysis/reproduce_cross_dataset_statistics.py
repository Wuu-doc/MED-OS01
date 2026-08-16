#!/usr/bin/env python3
"""Validate the summary files and reproduce the MED-OS01 statistics.

The default analysis uses F1-score. Additional threshold-dependent metrics can
be selected from the command line. Baseline configurations are summarized for
context but are never included in the oversampling-intensity Friedman test.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILES = {
    "PIMA": ROOT / "results" / "pima" / "summary_140_configurations.csv",
    "Heart Failure": ROOT
    / "results"
    / "heart_failure"
    / "summary_140_configurations.csv",
    "Thoracic Surgery": ROOT
    / "results"
    / "thoracic_surgery"
    / "summary_140_configurations.csv",
}

KEY = ["Model", "SamplingMethod", "OversamplingIntensity", "Threshold"]
BASE_KEY = ["Model", "SamplingMethod", "OversamplingIntensity"]
MODELS = [
    "Logistic Regression",
    "Random Forest",
    "MLP",
    "XGBoost",
    "LightGBM",
]
INTENSITIES = [0.50, 0.75, 1.00]
THRESHOLDS = [0.50, 0.55, 0.60, 0.65]
TOP_K_VALUES = [5, 10, 20, 30]

METRICS = {
    "f1": ("F1-score", "F1Score_Mean"),
    "recall": ("Recall", "Recall_Mean"),
    "precision": ("Precision", "Precision_Mean"),
    "balanced_accuracy": ("Balanced Accuracy", "BalancedAccuracy_Mean"),
    "accuracy": ("Accuracy", "Accuracy_Mean"),
}
THRESHOLD_INDEPENDENT_METRICS = {
    "ROC-AUC": "ROC_AUC_Mean",
    "PR-AUC": "PR_AUC_Mean",
    "Brier score": "BrierScore_Mean",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the validated MED-OS01 cross-dataset statistics."
    )
    parser.add_argument("--pima", type=Path, default=DEFAULT_FILES["PIMA"])
    parser.add_argument(
        "--heart-failure", type=Path, default=DEFAULT_FILES["Heart Failure"]
    )
    parser.add_argument(
        "--thoracic-surgery", type=Path, default=DEFAULT_FILES["Thoracic Surgery"]
    )
    metric_group = parser.add_mutually_exclusive_group()
    metric_group.add_argument(
        "--metrics",
        nargs="+",
        choices=tuple(METRICS),
        default=["f1"],
        help="Threshold-dependent metrics to analyze (default: f1).",
    )
    metric_group.add_argument(
        "--all-metrics",
        action="store_true",
        help="Analyze all five supported threshold-dependent metrics.",
    )
    parser.add_argument(
        "--include-threshold-independent",
        action="store_true",
        help="Also correlate ROC-AUC, PR-AUC, and Brier score over 35 base configurations.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory in which to save each result table as CSV.",
    )
    return parser.parse_args()


def expected_configurations() -> set[tuple[str, str, float, float]]:
    expected: set[tuple[str, str, float, float]] = set()
    for model in MODELS:
        for threshold in THRESHOLDS:
            expected.add((model, "Baseline", 0.0, threshold))
        for method in ("SMOTE", "ADASYN"):
            for intensity in INTENSITIES:
                for threshold in THRESHOLDS:
                    expected.add((model, method, intensity, threshold))
    return expected


def _short_config_list(configs: Iterable[tuple], limit: int = 3) -> str:
    values = sorted(configs, key=str)
    preview = "; ".join(map(str, values[:limit]))
    return preview + (f"; ... ({len(values)} total)" if len(values) > limit else "")


def load_and_validate(
    files: dict[str, Path], metric_columns: list[str], include_independent: bool
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    datasets: dict[str, pd.DataFrame] = {}
    validation_rows = []
    expected = expected_configurations()

    for dataset, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing input for {dataset}: {path}")

        df = pd.read_csv(path, encoding="utf-8-sig")
        if "ExpansionRatio" in df.columns and "OversamplingIntensity" not in df.columns:
            df = df.rename(columns={"ExpansionRatio": "OversamplingIntensity"})

        required = set(KEY + metric_columns)
        if include_independent:
            required.update(THRESHOLD_INDEPENDENT_METRICS.values())
        missing_columns = sorted(required.difference(df.columns))
        if missing_columns:
            raise ValueError(
                f"{dataset} is missing required columns: {', '.join(missing_columns)}"
            )

        df = df.copy()
        df["Model"] = df["Model"].astype(str).str.strip()
        df["SamplingMethod"] = df["SamplingMethod"].astype(str).str.strip()
        df["OversamplingIntensity"] = pd.to_numeric(
            df["OversamplingIntensity"], errors="coerce"
        )
        df.loc[
            df["SamplingMethod"].eq("Baseline")
            & df["OversamplingIntensity"].isna(),
            "OversamplingIntensity",
        ] = 0.0
        df["OversamplingIntensity"] = df["OversamplingIntensity"].round(10)
        df["Threshold"] = pd.to_numeric(df["Threshold"], errors="coerce").round(10)

        numeric_columns = metric_columns.copy()
        if include_independent:
            numeric_columns.extend(THRESHOLD_INDEPENDENT_METRICS.values())
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        if len(df) != 140:
            raise ValueError(f"{dataset} must contain 140 rows; found {len(df)}")
        if df[KEY].isna().any().any():
            null_counts = df[KEY].isna().sum()
            raise ValueError(f"{dataset} has missing configuration keys:\n{null_counts}")
        if df.duplicated(KEY).any():
            duplicates = df.loc[df.duplicated(KEY, keep=False), KEY]
            raise ValueError(f"{dataset} has duplicate configurations:\n{duplicates}")
        if not np.isfinite(df[numeric_columns].to_numpy(dtype=float)).all():
            raise ValueError(f"{dataset} has missing or non-finite requested metric values")

        observed = set(df[KEY].itertuples(index=False, name=None))
        missing_configs = expected.difference(observed)
        unexpected_configs = observed.difference(expected)
        if missing_configs or unexpected_configs:
            details = []
            if missing_configs:
                details.append(f"missing: {_short_config_list(missing_configs)}")
            if unexpected_configs:
                details.append(f"unexpected: {_short_config_list(unexpected_configs)}")
            raise ValueError(f"{dataset} configuration grid is invalid ({' | '.join(details)})")

        if include_independent:
            variation = df.groupby(BASE_KEY, dropna=False)[
                list(THRESHOLD_INDEPENDENT_METRICS.values())
            ].nunique(dropna=False)
            if (variation > 1).any().any():
                raise ValueError(
                    f"{dataset} has threshold-independent metrics that vary by threshold"
                )

        # Canonical ordering makes all summaries and tie handling independent of
        # the row order in which a source CSV happens to be supplied.
        df = df.sort_values(KEY, kind="mergesort").reset_index(drop=True)
        datasets[dataset] = df
        validation_rows.append(
            {
                "Dataset": dataset,
                "File": str(path.resolve()),
                "Rows": len(df),
                "UniqueConfigurations": len(observed),
                "UniqueBaseConfigurations": len(df[BASE_KEY].drop_duplicates()),
                "Status": "valid",
            }
        )

    return datasets, pd.DataFrame(validation_rows)


def marginal_summaries(
    datasets: dict[str, pd.DataFrame], metrics: dict[str, tuple[str, str]]
) -> pd.DataFrame:
    rows = []
    for dataset, df in datasets.items():
        baseline = df[df["SamplingMethod"].eq("Baseline")]
        sampled = df[df["SamplingMethod"].isin(["SMOTE", "ADASYN"])]
        for _, (metric_label, column) in metrics.items():
            rows.append(
                {
                    "Dataset": dataset,
                    "Factor": "OversamplingIntensity",
                    "Level": "Baseline",
                    "Metric": metric_label,
                    "Mean": baseline[column].mean(),
                    "NConfigurations": len(baseline),
                }
            )
            for intensity in INTENSITIES:
                values = sampled.loc[
                    sampled["OversamplingIntensity"].eq(intensity), column
                ]
                rows.append(
                    {
                        "Dataset": dataset,
                        "Factor": "OversamplingIntensity",
                        "Level": f"{intensity:.2f}",
                        "Metric": metric_label,
                        "Mean": values.mean(),
                        "NConfigurations": len(values),
                    }
                )
            for threshold in THRESHOLDS:
                values = df.loc[df["Threshold"].eq(threshold), column]
                rows.append(
                    {
                        "Dataset": dataset,
                        "Factor": "Threshold",
                        "Level": f"{threshold:.2f}",
                        "Metric": metric_label,
                        "Mean": values.mean(),
                        "NConfigurations": len(values),
                    }
                )
    return pd.DataFrame(rows)


def matched_spearman(
    datasets: dict[str, pd.DataFrame], metrics: dict[str, tuple[str, str]]
) -> pd.DataFrame:
    rows = []
    for dataset_a, dataset_b in combinations(datasets, 2):
        columns = [column for _, column in metrics.values()]
        merged = datasets[dataset_a][KEY + columns].merge(
            datasets[dataset_b][KEY + columns],
            on=KEY,
            suffixes=("_a", "_b"),
            validate="one_to_one",
        )
        if len(merged) != 140:
            raise ValueError(
                f"Expected 140 matches for {dataset_a} vs {dataset_b}; got {len(merged)}"
            )
        for _, (metric_label, column) in metrics.items():
            result = spearmanr(merged[f"{column}_a"], merged[f"{column}_b"])
            rows.append(
                {
                    "Dataset1": dataset_a,
                    "Dataset2": dataset_b,
                    "Metric": metric_label,
                    "NMatched": len(merged),
                    "SpearmanRho": result.statistic,
                    "PValue": result.pvalue,
                }
            )
    return pd.DataFrame(rows)


def topk_overlap(
    datasets: dict[str, pd.DataFrame], metrics: dict[str, tuple[str, str]]
) -> pd.DataFrame:
    rows = []
    for dataset_a, dataset_b in combinations(datasets, 2):
        for _, (metric_label, column) in metrics.items():
            ranked = {}
            for dataset in (dataset_a, dataset_b):
                ranked[dataset] = datasets[dataset].sort_values(
                    [column, *KEY],
                    ascending=[False, True, True, True, True],
                    kind="mergesort",
                )
            for k in TOP_K_VALUES:
                top_a = set(
                    ranked[dataset_a].head(k)[KEY].itertuples(index=False, name=None)
                )
                top_b = set(
                    ranked[dataset_b].head(k)[KEY].itertuples(index=False, name=None)
                )
                overlap = len(top_a.intersection(top_b))
                union = len(top_a.union(top_b))
                rows.append(
                    {
                        "Dataset1": dataset_a,
                        "Dataset2": dataset_b,
                        "Metric": metric_label,
                        "TopK": k,
                        "Overlap": overlap,
                        "OverlapProportion": overlap / k,
                        "Jaccard": overlap / union,
                    }
                )
    return pd.DataFrame(rows)


def _friedman_table(
    df: pd.DataFrame, factor: str, metric_column: str
) -> pd.DataFrame:
    if factor == "Threshold":
        return df.pivot(index=BASE_KEY, columns="Threshold", values=metric_column)[
            THRESHOLDS
        ]
    sampled = df[df["SamplingMethod"].isin(["SMOTE", "ADASYN"])]
    return sampled.pivot(
        index=["Model", "SamplingMethod", "Threshold"],
        columns="OversamplingIntensity",
        values=metric_column,
    )[INTENSITIES]


def _wilcoxon(values_a: pd.Series, values_b: pd.Series) -> tuple[float, float]:
    differences = values_a.to_numpy() - values_b.to_numpy()
    if np.allclose(differences, 0.0):
        return 0.0, 1.0
    result = wilcoxon(values_a, values_b, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def friedman_and_posthoc(
    datasets: dict[str, pd.DataFrame], metrics: dict[str, tuple[str, str]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    friedman_rows = []
    posthoc_rows = []

    for dataset, df in datasets.items():
        for _, (metric_label, column) in metrics.items():
            for factor in ("Threshold", "OversamplingIntensity"):
                table = _friedman_table(df, factor, column)
                if table.isna().any().any():
                    raise ValueError(f"Incomplete {factor} blocks for {dataset}/{metric_label}")
                result = friedmanchisquare(*(table[level] for level in table.columns))
                n_blocks = len(table)
                n_levels = len(table.columns)
                kendalls_w = result.statistic / (n_blocks * (n_levels - 1))
                friedman_rows.append(
                    {
                        "Dataset": dataset,
                        "Factor": factor,
                        "Metric": metric_label,
                        "NBlocks": n_blocks,
                        "NLevels": n_levels,
                        "FriedmanChi2": result.statistic,
                        "PValue": result.pvalue,
                        "KendallsW": kendalls_w,
                    }
                )

                pair_rows = []
                raw_p_values = []
                for level_a, level_b in combinations(table.columns, 2):
                    statistic, raw_p = _wilcoxon(table[level_a], table[level_b])
                    pair_rows.append(
                        {
                            "Dataset": dataset,
                            "Factor": factor,
                            "Metric": metric_label,
                            "Level1": level_a,
                            "Level2": level_b,
                            "NBlocks": n_blocks,
                            "WilcoxonStatistic": statistic,
                            "MedianDifference": np.median(
                                table[level_a].to_numpy() - table[level_b].to_numpy()
                            ),
                            "RawP": raw_p,
                        }
                    )
                    raw_p_values.append(raw_p)
                rejected, adjusted, _, _ = multipletests(
                    raw_p_values, alpha=0.05, method="holm"
                )
                for row, reject, adjusted_p in zip(pair_rows, rejected, adjusted):
                    row["HolmAdjustedP"] = adjusted_p
                    row["RejectAt0.05"] = bool(reject)
                    posthoc_rows.append(row)

    return pd.DataFrame(friedman_rows), pd.DataFrame(posthoc_rows)


def threshold_independent_spearman(
    datasets: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    columns = list(THRESHOLD_INDEPENDENT_METRICS.values())
    base = {
        dataset: df.sort_values([*BASE_KEY, "Threshold"])
        .drop_duplicates(BASE_KEY)[BASE_KEY + columns]
        for dataset, df in datasets.items()
    }
    rows = []
    for dataset_a, dataset_b in combinations(base, 2):
        merged = base[dataset_a].merge(
            base[dataset_b],
            on=BASE_KEY,
            suffixes=("_a", "_b"),
            validate="one_to_one",
        )
        if len(merged) != 35:
            raise ValueError(
                f"Expected 35 base matches for {dataset_a} vs {dataset_b}; got {len(merged)}"
            )
        for metric_label, column in THRESHOLD_INDEPENDENT_METRICS.items():
            result = spearmanr(merged[f"{column}_a"], merged[f"{column}_b"])
            rows.append(
                {
                    "Dataset1": dataset_a,
                    "Dataset2": dataset_b,
                    "Metric": metric_label,
                    "NMatched": len(merged),
                    "SpearmanRho": result.statistic,
                    "PValue": result.pvalue,
                }
            )
    return pd.DataFrame(rows)


def print_table(title: str, table: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    print(table.to_string(index=False))


def save_tables(output_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8")
    print(f"\nSaved {len(tables)} tables to {output_dir.resolve()}")


def main() -> None:
    args = parse_args()
    metric_keys = list(METRICS) if args.all_metrics else args.metrics
    selected_metrics = {key: METRICS[key] for key in metric_keys}
    files = {
        "PIMA": args.pima,
        "Heart Failure": args.heart_failure,
        "Thoracic Surgery": args.thoracic_surgery,
    }

    try:
        datasets, validation = load_and_validate(
            files,
            [column for _, column in selected_metrics.values()],
            args.include_threshold_independent,
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Input validation failed: {error}") from None
    friedman, posthoc = friedman_and_posthoc(datasets, selected_metrics)
    tables = {
        "dataset_validation": validation,
        "marginal_summaries": marginal_summaries(datasets, selected_metrics),
        "spearman_correlations": matched_spearman(datasets, selected_metrics),
        "topk_overlap": topk_overlap(datasets, selected_metrics),
        "friedman_tests": friedman,
        "wilcoxon_holm": posthoc,
    }
    if args.include_threshold_independent:
        tables["threshold_independent_spearman"] = threshold_independent_spearman(
            datasets
        )

    for name, table in tables.items():
        print_table(name.replace("_", " ").title(), table)
    print(
        "\nCaution: threshold variants from one base configuration reuse the same "
        "probability predictions; treat inferential p-values as descriptive evidence, "
        "not as evidence from independent model fits."
    )
    if args.output_dir:
        save_tables(args.output_dir, tables)


if __name__ == "__main__":
    main()
