# MED-OS01: Oversampling Intensity and Decision Threshold in Imbalanced Medical Classification

## Study objective

MED-OS01 contains the analysis materials and summarized results for the study
“Effects of Oversampling Intensity and Decision Threshold on Imbalanced Medical
Classification.” The study examines how the amount of oversampling and the
classification threshold affect performance, and assesses generalization across
additional medical datasets under one experimental protocol.

## Experimental design

Each dataset uses the same 5 classifiers, 7 training-data settings, and 4
decision thresholds. This produces 35 base configurations and 140 summarized
threshold-specific configurations per dataset. Dataset-specific hyperparameter
tuning is not used. Full settings are documented in
[`docs/EXPERIMENT_DESIGN.md`](docs/EXPERIMENT_DESIGN.md).

## Dataset overview

| Dataset | Role in this study | Task type |
|---|---|---|
| PIMA Indians Diabetes | Medical tabular benchmark | Binary classification |
| Heart Failure | Medical tabular benchmark | Binary classification |
| Thoracic Surgery | Medical tabular benchmark | Binary classification |

These datasets are used for cross-dataset validation; they are not described as
external clinical validation cohorts.

## Classifiers

- Logistic Regression
- Random Forest
- Multilayer Perceptron (MLP)
- XGBoost
- LightGBM

The random state is 42 wherever the estimator or procedure exposes that option.

## Oversampling strategies

The training-data settings are Baseline (no oversampling), SMOTE, and ADASYN.
Oversampling is performed independently inside each training fold and never on
its held-out validation fold.

## Oversampling intensity

For training-fold minority and majority counts \(n_min\) and \(n_maj\), intensity
\(alpha\) defines the target minority count as:

```text
n_target = n_min + alpha * (n_maj - n_min)
```

The evaluated intensities are 0.50, 0.75, and 1.00; 1.00 targets class balance.
Baseline is reported descriptively and is represented by intensity 0 in the
summary files. The realized ADASYN count can differ slightly from its target
because samples are allocated adaptively.

## Decision thresholds

Each base configuration is evaluated at probability thresholds 0.50, 0.55,
0.60, and 0.65. The four variants reuse the same underlying fold-level
probability predictions.

## Cross-validation and statistical design

- Stratified 10-fold cross-validation
- Random state 42
- Oversampling restricted to training folds
- Primary metric: F1-score
- Optional metrics: Recall, Precision, Balanced Accuracy, and Accuracy
- Friedman tests with Kendall's W, followed by paired Wilcoxon signed-rank tests
  with Holm correction
- Spearman rank correlation and Top-5/10/20/30 configuration overlap between
  matched datasets

For intensity tests, a matched block is classifier + sampling method + decision
threshold; Baseline is excluded. For threshold tests, a matched block is
classifier + sampling method + oversampling intensity. Inferential p-values
should be interpreted cautiously because threshold variants from one base
configuration are not independent model fits.

## Repository structure

```text
MED-OS01/
├── README.md
├── requirements.txt
├── .gitignore
├── analysis/
│   └── reproduce_cross_dataset_statistics.py
├── docs/
│   ├── EXPERIMENT_DESIGN.md
│   └── CROSS_DATASET_RESULTS.md
├── figures/
│   └── pima_performance_stability_quadrant_v2.png
└── results/
    ├── pima/summary_140_configurations.csv
    ├── heart_failure/summary_140_configurations.csv
    └── thoracic_surgery/summary_140_configurations.csv
```

## Reproducibility instructions

Python 3.11 or later is recommended. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python analysis/reproduce_cross_dataset_statistics.py
```

The default command validates all three 140-row configuration grids and runs the
primary F1-score analyses. To reproduce every table represented in the results
document and save machine-readable outputs:

```bash
python analysis/reproduce_cross_dataset_statistics.py \
  --all-metrics \
  --include-threshold-independent \
  --output-dir results/cross_dataset
```

Use `--help` to supply alternative CSV paths or select individual metrics. Saved
tables include validation, marginal summaries, Spearman correlations, Top-K
overlap counts/proportions, Friedman tests, and Holm-adjusted Wilcoxon tests.

## Citation

Citation metadata will be added after manuscript and archival identifiers are
assigned. Placeholder:

```bibtex
@article{MED_OS01_TODO,
  title   = {Effects of Oversampling Intensity and Decision Threshold on
             Imbalanced Medical Classification},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO},
  doi     = {TODO}
}
```

## Data availability

This repository distributes processed experimental summaries and analysis code,
not the raw medical datasets. Raw data should be obtained from their original
public sources and used under the applicable source terms. Dataset source URLs,
versions, and access dates should be recorded before the archival release. A
software/data license must also be selected before public release.
