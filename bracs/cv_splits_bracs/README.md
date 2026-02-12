# BRACS Cross-Validation Splits

This folder contains the patient-level train/validation/test split definitions used for
backbone fine-tuning and evaluation on the **BRACS** dataset.

---

## Split Files

### Common held-out test set

| File | Description |
|------|-------------|
| `test_split_stratified.csv` | Test set (87 WSIs) — stratified by WSI label, grouped by Patient Id |
| `test_split_multi_stratified.csv` | Test set (87 WSIs) — multi-stratified by WSI label and ROI quantiles |

### 5-fold cross-validation splits

| Pattern | Description |
|---------|-------------|
| `train_split_stratified_{0-4}.csv` | Training folds — stratified by label |
| `val_split_stratified_{0-4}.csv` | Validation folds — stratified by label |
| `train_split_multi_stratified_{0-4}.csv` | Training folds — multi-stratified (label + ROI quantiles) |
| `val_split_multi_stratified_{0-4}.csv` | Validation folds — multi-stratified (label + ROI quantiles) |

### Split generation notebooks

| File | Description |
|------|-------------|
| `generate_strat_sklearn_common_test.ipynb` | Generates stratified splits using `StratifiedGroupKFold(n_splits=5)` grouped by Patient Id |
| `generate_multi_strat_sklearn_common_test.ipynb` | Generates multi-stratified splits using a composite stratification column (label + ROI quantiles) |

---

## CSV Format

All split CSVs share the columns:

| Column | Description |
|--------|-------------|
| `WSI Filename` | BRACS WSI identifier (e.g., `BRACS_264`) |
| `Patient Id` | Patient identifier (integer) |
| `RoI` | Number of annotated regions of interest |
| `WSI label` | One of: N, PB, UDH, ADH, FEA, DCIS, IC |

The multi-stratified test split additionally contains:
`Set`, `WSI label oh` (label encoded as integer), `RoI quantiles`.

---

## Data Leak Fix

Both generation notebooks fix a data leak in the original BRACS distribution where
Patient 67 appeared in both training and testing sets. This patient was moved to the
validation set to ensure strict patient-level separation (Paper §3.1).

---

## Notes

- The BRACS dataset metadata (`BRACS.xlsx`) is required to regenerate splits but is not
  redistributed here. It is available from the [BRACS website](https://www.bracs.icar.cnr.it/).
