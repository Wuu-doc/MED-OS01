# MED-OS01: Oversampling Intensity and Decision Threshold in Imbalanced Medical Classification

## Study objective

MED-OS01 provides processed results, analysis code, figures, and supplementary
outputs for a study of how oversampling method, oversampling intensity,
classifier, and predefined decision threshold affect imbalanced medical
classification. The study also examines cross-dataset robustness and
configuration consistency.

PIMA is the primary benchmark. Heart Failure and Thoracic Surgery are
complementary cross-dataset validation datasets to which the same core
experimental protocol is applied independently.

## Experimental design

- Models: Logistic Regression, Random Forest, MLP, XGBoost, and LightGBM.
- Sampling conditions: Baseline, SMOTE, and ADASYN.
- SMOTE intensities: 0.25, 0.50, 0.75, and 1.00. The 0.25 condition is a
  low-intensity sensitivity analysis.
- ADASYN main-grid intensities: 0.50, 0.75, and 1.00. ADASYN 0.25 is excluded
  from the main grid following preliminary feasibility screening.
- Matched SMOTE–ADASYN comparison: the common 0.50, 0.75, and 1.00 grid.
- Predefined decision thresholds: 0.30 to 0.70 in steps of 0.05.
- Validation: Repeated Stratified 10-Fold Cross-Validation × 5 repetitions.
- Total: 50 held-out evaluations per model–sampling configuration.

Each dataset has 360 threshold-specific configuration rows; the combined table
has 1,080 rows. Baseline `RequestedIntensity` is blank/NA rather than zero.

Missing-value imputation, standardization, and class-imbalance handling are fit
within each training fold. The corresponding held-out fold retains its class
distribution, receives no oversampling, and is transformed only with parameters
learned from its training fold. One fitted model produces held-out probabilities
that are reused across all nine predefined thresholds.

See [the full experimental design](docs/EXPERIMENT_DESIGN.md).

## Metrics

- Primary: F1-score.
- Supporting threshold-dependent metrics: Precision, Sensitivity/Recall,
  Specificity, and Balanced Accuracy.
- Threshold-independent ranking metrics: ROC-AUC and Average Precision (AP).
- Calibration: Brier score.

ROC-AUC, AP, and Brier score are computed from predicted probabilities and do
not pass through the decision-threshold branch.

## Statistical outputs

The repository includes authoritative CSV outputs for the linear mixed-effects
interaction analysis, matched SMOTE–ADASYN comparison, cross-dataset
consistency, low-intensity sensitivity analysis, calibration, and robustness
analyses. The validation script checks the revised grids and reports those
stored results without refitting models or redefining the statistical models.

## Repository structure

```text
MED-OS01/
├── README.md
├── README_CSV_UPDATE.md
├── CSV_MANIFEST.csv
├── CSV_VALIDATION.csv
├── requirements.txt
├── analysis/
│   └── reproduce_cross_dataset_statistics.py
├── docs/
│   ├── EXPERIMENT_DESIGN.md
│   └── CROSS_DATASET_RESULTS.md
├── figures/
│   ├── Figure1_overall_research_framework.{png,pdf}
│   ├── Figure2_leakage_free_workflow.{png,pdf}
│   ├── generate_manuscript_figures.py
│   └── pima_performance_stability_quadrant_v2.png
└── results/
    ├── pima/summary_360_configurations.csv
    ├── heart_failure/summary_360_configurations.csv
    ├── thoracic_surgery/summary_360_configurations.csv
    ├── combined_summary_1080_configurations.csv
    ├── statistics/*.csv
    ├── calibration/main_ap_brier_summary.csv
    ├── robustness/*.csv
    └── supplementary/adasyn_low_intensity_feasibility.csv
```

`CSV_MANIFEST.csv` records the provenance and expected dimensions of every CSV
in the revised bundle. `CSV_VALIDATION.csv` records the supplied grid checks.

## Reproducibility and validation

Python 3.11 or later is recommended. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python analysis/reproduce_cross_dataset_statistics.py
```

The script validates all three 360-row summaries, the combined 1,080-row table,
the complete threshold and intensity grids, Baseline missing intensity,
`FoldRepeatN = 50`, threshold-independent metric consistency, and every
manifest-listed CSV. Optional descriptive validation outputs can be written to
a separate directory:

```bash
python analysis/reproduce_cross_dataset_statistics.py \
  --output-dir validation_outputs
```

The 50 held-out evaluations arise from repeated cross-validation and should not
be described as independent evaluations.

## Figures

- Figure 1: overall research framework and two-stage experimental design.
- Figure 2: leakage-free repeated cross-validation and evaluation workflow.

Both high-resolution PNG and vector PDF versions are provided.

## Data availability

This repository does not redistribute the raw medical datasets. Raw datasets
should be obtained from their original public sources and used under the
applicable source terms. The repository provides processed results, analysis
code, figures, and supplementary outputs used for manuscript-level
reproducibility.

## Citation

Citation metadata will be added after manuscript and archival identifiers are
assigned.
