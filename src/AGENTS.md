# src/ — ML Pipeline

**7 files, ~600 lines. Core ML logic.**

## OVERVIEW

End-to-end ML pipeline for credit risk classification: data loading → preprocessing → model training → evaluation → prediction. All modules import from `src.config` for constants.

## FILES

| File | Lines | Role |
|------|-------|------|
| `config.py` | 134 | All paths, column lists, hyperparameter grids, risk thresholds, model version |
| `data_loader.py` | 109 | Load German Credit CSV, map target (1/2→0/1), stratified 60/20/20 split, save/load processed |
| `preprocess.py` | 79 | 3 engineered features + `ColumnTransformer` (StandardScaler + OneHotEncoder) |
| `train.py` | 145 | GridSearchCV for Logistic Regression, Random Forest, XGBoost; metadata JSON export |
| `evaluate.py` | 97 | Accuracy/precision/recall/F1/ROC-AUC + confusion matrix; model comparison DataFrame |
| `predict.py` | 135 | Model/preprocessor loading, `predict_proba`, risk band mapping (LOW/MEDIUM/HIGH), batch predict |
| `__init__.py` | 0 | Empty — no re-exports |

## CONVENTIONS

- `pandas` for all data manipulation, `joblib` for model serialization
- Functions are standalone (no classes) — pipeline composition via imports
- `src.config` is the single source of truth for all constants; no magic numbers in other modules
- Preprocessor fitted once, saved as `.pkl`; same pipeline used for train/val/test
- Random seed (`RANDOM_STATE=42`) passed explicitly for reproducibility
- Warnings filtered during training (`warnings.filterwarnings('ignore')` in Makefile)

## ANTI-PATTERNS

- **`src/predict.py`** — `required_cols` list duplicated in two functions; should be module-level constant
- **No `__main__.py`** — training not runnable as `python -m src.train`; must go through Makefile
- **No `pyproject.toml`** — imports work only from project root (no package install)

## DEPENDENCY FLOW

```
config ← data_loader → preprocess → train → evaluate → predict
   ↑__________________________________________________|
```
