# Cross-Dataset Results

This document summarizes the revised manuscript-facing results. Numerical
values are taken from the authoritative CSV files under `results/statistics/`,
`results/calibration/`, and `results/robustness/`; displayed values are rounded.
No model fitting or prediction regeneration is performed by the repository
validation script.

## Highest observed mean F1-score on the predefined grid

| Dataset | Configuration | Threshold | Mean F1-score |
|---|---|---:|---:|
| PIMA | Random Forest + Baseline | 0.35 | 0.688999 |
| Heart Failure | Random Forest + SMOTE, intensity 0.50 | 0.50 | 0.768403 |
| Thoracic Surgery | Logistic Regression + ADASYN, intensity 0.50 | 0.35 | 0.337507 |

These are the highest observed means within the predefined experimental grid.
They are not externally validated deployment thresholds and do not establish
causal superiority.

Source: `results/statistics/key_results_summary.csv`.

## Intensity × threshold interaction

The authoritative mixed-effects analysis reports an omnibus likelihood-ratio
test for `C(Intensity) × C(Threshold)`:

| Statistic | df | p-value | Convergence |
|---:|---:|---:|---|
| 141.773720 | 16 | 3.237367 × 10⁻²² | Full and reduced models converged |

The primary model converged, but the stored interpretation retains a residual
diagnostic warning. Accordingly, the interaction supports joint evaluation of
intensity and threshold but should not be overstated.

Sources:
`results/statistics/mixed_effects_interaction_omnibus.csv` and
`results/statistics/mixed_effects_intensity_threshold_terms.csv`.

## Matched SMOTE–ADASYN comparison

The matched comparison uses intensities 0.50, 0.75, and 1.00 across the nine
predefined thresholds:

| Contrast | Mean ΔF1 | 95% cluster-bootstrap CI | Wilcoxon p-value |
|---|---:|---:|---:|
| SMOTE − ADASYN | 0.000883 | [−0.001517, 0.003369] | 0.663032 |

The interval includes zero, and the paired test does not support a global
superiority claim for either method.

Source: `results/statistics/smote_adasyn_global_matched_inference.csv`.

Dataset-level supporting estimates are available in
`results/statistics/smote_adasyn_dataset_supporting_summaries.csv`.

## Cross-dataset configuration consistency

Spearman correlations over matched common-grid configuration rankings are:

| Dataset pair | Spearman ρ | Descriptive p-value | Top-10 intersection |
|---|---:|---:|---:|
| Heart Failure–PIMA | 0.618862 | 6.200921 × 10⁻³⁰ | 0 |
| Heart Failure–Thoracic Surgery | −0.009697 | 0.873984 | 0 |
| PIMA–Thoracic Surgery | 0.663224 | 1.364227 × 10⁻³⁵ | 0 |

The differing correlations and absent Top-10 intersections indicate limited
configuration-level transferability despite consistency in some broader
factor-wise patterns. They do not support broad claims beyond these datasets.

Source: `results/statistics/transferability_rank_consistency.csv`. Factor-wise
summaries for classifier, sampling method, intensity, and threshold are stored
in the corresponding `transferability_common_grid_factor_*.csv` files.

## Low-intensity sensitivity analysis

Across datasets, the descriptive mean contrast for SMOTE intensity 0.25 minus
SMOTE intensity 0.50 is −0.012725, with cluster-bootstrap 95% CI
[−0.017981, −0.007760]. This is a sensitivity analysis and is not substituted
for the common SMOTE–ADASYN grid.

Source:
`results/statistics/smote_alpha025_low_intensity_sensitivity_summary.csv`.

ADASYN 0.25 is not part of the main grid. Its preliminary screening failed to
produce the requested samples in 92 of 150 training splits; the screening rows
are retained in `results/supplementary/adasyn_low_intensity_feasibility.csv`.

## Ranking and calibration metrics

ROC-AUC and Average Precision are threshold-independent ranking metrics. Brier
score is the manuscript-facing calibration metric. These values are computed
from held-out predicted probabilities and do not depend on the predefined
decision threshold. The authoritative 120-row summary is provided at
`results/calibration/main_ap_brier_summary.csv`.

## Robustness analyses

The authoritative key-results summary reports the following descriptive
robustness estimates:

| Analysis | Scope | Mean ΔF1 |
|---|---|---:|
| SMOTENC − SMOTE | Heart Failure | 0.002740 |
| SMOTENC − SMOTE | Thoracic Surgery | 0.003647 |
| Tuned − fixed hyperparameters | All datasets | −0.000214 |

Full outputs are retained in:

- `results/robustness/smotenc_vs_smote_summary.csv`
- `results/robustness/fixed_vs_tuned_robustness_summary.csv`

These results are descriptive robustness checks and remain separate from the
primary configuration ranking.

## Interpretation

The revised results support joint consideration of oversampling intensity and
predefined decision threshold. The matched SMOTE–ADASYN estimate is near zero,
while configuration rankings vary across dataset pairs. The evidence therefore
supports dataset-specific configuration and cautious cross-dataset validation,
rather than a universally superior sampling method or setting.

The 50 held-out evaluations per model–sampling configuration arise from five
repetitions of stratified 10-fold cross-validation and are not independent
evaluations.
