# Cross-Dataset Statistical Results

This document summarizes results reproduced from the three supplied
140-configuration CSV files with
`analysis/reproduce_cross_dataset_statistics.py`. Displayed values are rounded;
the script reports and can save full-precision values.

## 1. Spearman rank consistency across all 140 matched configurations

| Metric | PIMA–Heart Failure | PIMA–Thoracic Surgery | Heart Failure–Thoracic Surgery |
|---|---:|---:|---:|
| F1-score | 0.6333 | 0.3182 | -0.2204 |
| Recall | 0.8383 | 0.3559 | 0.0977 |
| Balanced Accuracy | 0.6579 | 0.4550 | 0.0196 |
| Precision | 0.6020 | 0.0450 | -0.0748 |
| Accuracy | 0.3225 | 0.1786 | 0.5853 |

For F1-score, the corresponding p-values are <0.001 for PIMA–Heart Failure, <0.001 for PIMA–Thoracic Surgery, and 0.0089 for Heart Failure–Thoracic Surgery.

## 2. Top-K overlap for F1-score rankings

| Pair | Top 5 | Top 10 | Top 20 | Top 30 |
|---|---:|---:|---:|---:|
| PIMA–Heart Failure | 0 | 0 | 6 | 12 |
| PIMA–Thoracic Surgery | 1 | 4 | 9 | 13 |
| Heart Failure–Thoracic Surgery | 0 | 0 | 0 | 1 |

The overlap is the intersection count among two deterministic Top-K rankings.
Configurations are matched by model, sampling method, oversampling intensity,
and decision threshold; exact metric ties are broken by that key. The analysis
output also reports overlap/K and the Jaccard index.

## 3. Mean F1-score by oversampling intensity

| Intensity | PIMA | Heart Failure | Thoracic Surgery |
|---:|---:|---:|---:|
| Baseline | 0.5930 | 0.6930 | 0.0581 |
| 0.50 | 0.6169 | 0.7139 | 0.1322 |
| 0.75 | 0.6234 | 0.7048 | 0.1534 |
| 1.00 | 0.6300 | 0.7035 | 0.1649 |

Baseline is included only in the descriptive means above. Friedman tests compare
intensities 0.50, 0.75, and 1.00 using 40 matched classifier + sampling method +
threshold blocks per dataset:

| Dataset | Friedman chi-square | p-value | Kendall's W |
|---|---:|---:|---:|
| PIMA | 19.3500 | <0.001 | 0.2419 |
| Heart Failure | 9.6000 | 0.0082 | 0.1200 |
| Thoracic Surgery | 14.3822 | <0.001 | 0.1798 |

## 4. Mean F1-score by decision threshold

| Threshold | PIMA | Heart Failure | Thoracic Surgery |
|---:|---:|---:|---:|
| 0.50 | 0.6427 | 0.7164 | 0.1682 |
| 0.55 | 0.6294 | 0.7100 | 0.1472 |
| 0.60 | 0.6139 | 0.7032 | 0.1235 |
| 0.65 | 0.5904 | 0.6917 | 0.1093 |

Friedman tests across the four thresholds use 35 matched base configurations per
dataset:

| Dataset | Friedman chi-square | p-value | Kendall's W |
|---|---:|---:|---:|
| PIMA | 92.4514 | <0.001 | 0.8805 |
| Heart Failure | 36.9238 | <0.001 | 0.3517 |
| Thoracic Surgery | 52.0351 | <0.001 | 0.4956 |

All six pairwise threshold comparisons remain significant after Holm correction in all three datasets.

## 5. Threshold-independent metrics: 35 unique base configurations

| Metric | PIMA–Heart Failure | PIMA–Thoracic Surgery | Heart Failure–Thoracic Surgery |
|---|---:|---:|---:|
| ROC-AUC | 0.2401 | 0.7779 | 0.2319 |
| PR-AUC | 0.1056 | 0.6936 | 0.2056 |
| Brier score | 0.5852 | 0.5619 | 0.8852 |

## Interpretation

In these summaries, mean F1-score decreases as the threshold increases from 0.50
to 0.65 in all three datasets. The descriptive optimum among the tested
oversampling intensities differs by dataset. Rank correlations also differ by
dataset pair, while Top-K intersections are generally limited, so the results do
not establish a universally transferable optimal configuration.

The four threshold variants of a base configuration reuse the same probability
predictions and are therefore statistically dependent. The Friedman and
Wilcoxon p-values should be treated as descriptive evidence for this structured
comparison, not as evidence from independent model fits. These datasets support
cross-dataset validation and are not external clinical validation cohorts.
