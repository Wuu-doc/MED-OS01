# MED-OS01 revised CSV bundle

These CSV files were extracted from the authoritative `revision_v2` outputs used for the revised manuscript.

Key points:
- Each main dataset summary now has 360 threshold-specific configurations.
- Decision thresholds: 0.30 to 0.70 in steps of 0.05.
- Repeated stratified 10-fold CV x 5 repeats = 50 held-out evaluations per model/sampling configuration.
- Baseline intensity is blank/NA, not 0.
- SMOTE intensities: 0.25, 0.50, 0.75, 1.00.
- ADASYN main-grid intensities: 0.50, 0.75, 1.00.
- ADASYN 0.25 feasibility results are included separately.
- Calibration CSVs use Brier score / Average Precision; reliability-analysis outputs are intentionally not included in this GitHub bundle.
- Robustness CSVs include SMOTENC and fixed-vs-tuned hyperparameter checks.

`CSV_VALIDATION.csv` records the basic grid checks.
