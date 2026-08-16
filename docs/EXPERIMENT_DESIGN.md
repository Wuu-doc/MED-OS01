# Unified Experimental Design

## Scope

The study applies one experimental design to PIMA Indians Diabetes, Heart
Failure, and Thoracic Surgery for cross-dataset validation. No dataset-specific
hyperparameter tuning is introduced.

## Classification models

| Model | Hyperparameter settings |
|---|---|
| Logistic Regression | `max_iter=1000`, `random_state=42` |
| Random Forest | `n_estimators=300`, `random_state=42`, `n_jobs=-1`; no `class_weight="balanced"` |
| MLP | `hidden_layer_sizes=(64, 32)`, `activation="relu"`, `max_iter=1500`, `random_state=42` |
| XGBoost | `objective="binary:logistic"`, `eval_metric="logloss"`, `n_estimators=300`, `learning_rate=0.05`, `max_depth=4`, `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42` |
| LightGBM | `objective="binary"`, `n_estimators=300`, `learning_rate=0.05`, `num_leaves=31`, `random_state=42` |

Parameters not listed above retain the corresponding library defaults used by
the original experiment. Library versions for model fitting should be recorded
with the training pipeline before archival release; the present repository
contains summary-level statistical reproduction rather than model-fitting code.

## Sampling settings and intensity

Seven training-data settings are evaluated:

- Baseline: no oversampling
- SMOTE: intensity 0.50, 0.75, or 1.00
- ADASYN: intensity 0.50, 0.75, or 1.00

Within each training fold, let \(n_min\) and \(n_maj\) be the minority and
majority counts before oversampling. For intensity \(alpha\), the target minority
count is:

```text
n_target = n_min + alpha * (n_maj - n_min)
```

Thus, 0.50 closes half of the original count gap, 0.75 closes three quarters,
and 1.00 targets equal class counts. The realized ADASYN count may differ
slightly because its synthetic samples are allocated adaptively. Baseline is
encoded as intensity 0 in the supplied summaries but is not treated as an
oversampling-intensity level in inferential tests.

## Decision thresholds

Every fitted probability model is evaluated at thresholds 0.50, 0.55, 0.60,
and 0.65. Probability predictions are reused across the four thresholds.

## Cross-validation protocol

- Stratified 10-fold cross-validation
- Random state 42
- Oversampling performed only within each training fold
- Validation folds never used to fit or parameterize the oversampler
- Validation folds never oversampled

For each dataset:

```text
5 classifiers x 7 training-data settings = 35 base configurations
35 base configurations x 10 folds        = 350 model fits
35 base configurations x 4 thresholds   = 140 summarized configurations
```

## Statistical analysis plan

F1-score is the primary metric. Recall, Precision, Balanced Accuracy, and
Accuracy are supported as optional threshold-dependent outcomes.

### Oversampling intensity

- Descriptively report Baseline, 0.50, 0.75, and 1.00.
- Exclude Baseline from inferential intensity comparisons.
- Compare 0.50, 0.75, and 1.00 using the Friedman test.
- Use classifier + sampling method + decision threshold as the matched block
  (40 blocks per dataset).
- Report Kendall's W.
- Follow with all paired Wilcoxon signed-rank tests and Holm correction within
  each dataset, metric, and factor family.

### Decision threshold

- Compare 0.50, 0.55, 0.60, and 0.65 using the Friedman test.
- Use classifier + sampling method + oversampling intensity as the matched base
  configuration (35 blocks per dataset).
- Report Kendall's W.
- Follow with all paired Wilcoxon signed-rank tests and Holm correction within
  each dataset, metric, and factor family.

### Cross-dataset consistency

- Match configurations by classifier, sampling method, oversampling intensity,
  and threshold.
- Compute pairwise Spearman rank correlations over 140 matched configurations.
- For Top-K rankings (K = 5, 10, 20, 30), report intersection count,
  intersection/K, and Jaccard index.
- Break exact metric ties deterministically by the configuration key.

ROC-AUC, PR-AUC, and Brier score do not change with decision threshold. Optional
correlations for these outcomes therefore use the 35 unique base configurations,
not four duplicated threshold rows.

## Interpretation constraint

Threshold variants from a base configuration share the same underlying
probability predictions. Consequently, they should not be treated as 140
independent model fits, and inferential p-values should be interpreted as
descriptive evidence rather than overinterpreted as independent replication.
