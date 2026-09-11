# Unified Experimental Design

## Scope and dataset roles

The same core protocol is applied independently to three binary medical
classification datasets:

- PIMA: primary benchmark.
- Heart Failure: complementary cross-dataset validation dataset.
- Thoracic Surgery: complementary cross-dataset validation dataset.

This design assesses cross-dataset robustness and consistency. It does not
represent transfer of a model trained on PIMA to the other datasets.

## Classification models

Five model families are included:

- Logistic Regression
- Random Forest
- Multilayer Perceptron (MLP)
- XGBoost
- LightGBM

The primary grid uses the fixed model specification from the authoritative
rerun. Fixed-versus-tuned results are retained as a separate robustness analysis
in `results/robustness/fixed_vs_tuned_robustness_summary.csv`; they are not
substituted into the primary configuration ranking.

## Repeated cross-validation and leakage prevention

The evaluation uses Repeated Stratified 10-Fold Cross-Validation × 5
repetitions. Each model–sampling configuration therefore produces 50 held-out
evaluations.

For every fold:

1. Missing-value imputation parameters are learned from the training fold.
2. Standardization parameters are learned from the training fold.
3. Class-imbalance handling is performed only within the training fold.
4. The held-out fold retains its original class distribution and is never
   oversampled.
5. The held-out fold is transformed using only the preprocessing parameters
   learned from its corresponding training fold.
6. The trained classifier generates probabilities for the held-out fold.

The repeated-split and sampling procedures use deterministic, repeat/fold-aware
seeding in the authoritative rerun. There is no single random-state value that
fully describes the repeated cross-validation design. Where row-level sampler
seeds are part of a supplied output, they are retained in that CSV.

The resulting 50 held-out evaluations are repeated-CV evaluations; they are not
described as independent evaluations.

## Sampling settings and intensity

The sampling conditions are Baseline, SMOTE, and ADASYN. Within a training
fold, let `n_min` and `n_maj` be the minority and majority counts before
oversampling. For requested intensity `alpha`, the target minority count is:

```text
n_target = n_min + alpha * (n_maj - n_min)
```

Baseline uses no oversampling and is encoded with blank/NA
`RequestedIntensity`, never zero.

SMOTE intensities:

- 0.25 (low-intensity sensitivity analysis)
- 0.50
- 0.75
- 1.00

ADASYN main-grid intensities:

- 0.50
- 0.75
- 1.00

ADASYN 0.25 is excluded from the main grid. In preliminary feasibility
screening, it failed to produce the requested samples in 92 of 150 training
splits. The complete screening output is retained in
`results/supplementary/adasyn_low_intensity_feasibility.csv`.

The matched SMOTE–ADASYN analysis uses their common intensity grid: 0.50, 0.75,
and 1.00.

## Predefined decision thresholds

Each held-out probability vector is evaluated at the following nine predefined
experimental factors:

```text
0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70
```

The classifier is fitted once per fold and sampling configuration. It is not
refitted for different thresholds, and held-out performance is not used to
select the threshold grid retrospectively.

## Configuration counts

For each dataset:

```text
5 models × (1 Baseline + 4 SMOTE + 3 ADASYN conditions) = 40 base configurations
40 base configurations × 9 predefined thresholds        = 360 configuration rows
```

Across three datasets, the combined table contains 1,080 rows. Every summary
row records `FoldRepeatN = 50`.

## Evaluation metrics

Primary metric:

- F1-score

Supporting threshold-dependent metrics:

- Precision
- Sensitivity/Recall
- Specificity
- Balanced Accuracy

Threshold-independent ranking metrics:

- ROC-AUC
- Average Precision (AP)

Calibration metric:

- Brier score

ROC-AUC, AP, and Brier score are computed from the held-out predicted
probabilities. They do not pass through the decision-threshold branch and must
remain constant across the nine threshold rows belonging to one fitted
model–sampling configuration.

## Statistical analysis

The primary interaction analysis is a linear mixed-effects model containing:

- Dataset
- Model
- SamplingMethod
- `C(Intensity) * C(Threshold)`

The split/fit grouping and model definition follow the authoritative rerun. The
repository stores the corresponding omnibus and term-level outputs in
`results/statistics/`; the validation script does not redefine or refit that
model.

The matched SMOTE–ADASYN comparison uses the common 0.50/0.75/1.00 intensity
grid and all nine predefined thresholds. Cross-dataset consistency is evaluated
with matched configuration rankings and factor-wise summaries. These analyses
support conservative statements about consistency and limited transferability,
not a universally preferred configuration.

## Robustness analyses

Two separate robustness analyses are retained:

- SMOTENC versus SMOTE.
- Fixed versus tuned hyperparameters.

They are documented in `results/robustness/` and remain separate from the
primary grid and ranking results.
