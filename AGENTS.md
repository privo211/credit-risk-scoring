# PROJECT KNOWLEDGE BASE

**Generated:** 2026-06-10
**Commit:** bc27e11
**Branch:** main

## OVERVIEW

Credit risk scoring service — end-to-end ML pipeline (German Credit dataset) served via FastAPI + Docker. Binary classification (default/no-default) with three models (LR, RF, XGBoost) and GridSearchCV tuning.

**Stack:** Python 3.11+, FastAPI, scikit-learn, XGBoost, pandas, pytest, Docker

## STRUCTURE

```
credit-risk-scoring/
├── app/              # FastAPI service (endpoints, schemas, inference wrapper)
├── src/              # ML pipeline (config, data, preprocess, train, evaluate, predict)
├── tests/            # pytest unit tests (preprocessing, model, API)
├── models/           # Trained .pkl artifacts + metadata JSON
├── data/raw/         # German Credit dataset CSV
├── data/processed/   # Train/val/test splits (.gitkeep only, generated at runtime)
├── scripts/          # test_api.sh, sample_payload.json
├── notebooks/        # EDA exploration.ipynb
└── Makefile          # setup/train/test/run/docker-build/docker-run
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| API endpoints | `app/main.py` | FastAPI routes: `/health`, `/predict`, `/batch_predict` |
| Request validation | `app/schemas.py` | Pydantic models with `extra="forbid"` |
| Model loading + logging | `app/inference.py` | Module-level singleton pattern, latency logging |
| Config constants | `src/config.py` | Paths, column names, hyperparams, thresholds |
| Data loading/splitting | `src/data_loader.py` | Stratified 60/20/20 split, save/load processed |
| Feature engineering | `src/preprocess.py` | 3 engineered features + ColumnTransformer pipeline |
| Model training | `src/train.py` | GridSearchCV for LR/RF/XGBoost, metadata saving |
| Evaluation | `src/evaluate.py` | Metrics, model comparison, best-model selection |
| Prediction | `src/predict.py` | Risk bands (LOW<0.30, MEDIUM, HIGH>0.60), batch support |
| Tests | `tests/` | 15 unit tests across all layers |

## CONVENTIONS

- **No linting/formatting config** — no ruff, flake8, mypy, or pre-commit (intentional gap)
- **No pyproject.toml** — deps managed via requirements.txt only; not pip-installable
- **Semantic commits** — `feat:`, `test:`, `docs:`, `chore:` prefixes
- **Makefile as build system** — `make train` embeds inline Python (non-standard, but intentional)
- **Models tracked in git** — not gitignored, enables Docker without retraining
- **`data/processed/` gitignored** — regenerated via `make train`; directory preserved via `.gitkeep`

## ANTI-PATTERNS (THIS PROJECT)

- **Dual `src/` + `app/` packages** — creates blurry namespace boundary; `app/inference.py` is a thin wrapper around `src/predict.py`
- **Logging configured at module import time** in `app/inference.py` — executes before app lifecycle
- **Duplicated `required_cols`** list in `src/predict.py` (defined in two functions, not extracted to constant)
- **Late import** of `get_risk_band` inside function body in `app/inference.py`
- **Inline Python in Makefile** — `make train` is 46 lines of Python in a shell string, not a script
- **No environment variable configuration** — port, host, log level, model paths all hardcoded

## COMMANDS

```bash
make setup      # pip install -r requirements.txt
make train      # Full ML pipeline (load → preprocess → train 3 models → evaluate → save)
make test       # pytest tests/ -v --tb=short (15 tests)
make run        # uvicorn app.main:app --reload (port 8000)
make docker-build  # docker build -t credit-risk-scoring .
make docker-run    # docker run -p 8000:8000 credit-risk-scoring
```

## NOTES

- Best model: **Random Forest** (ROC-AUC 0.78 test, 0.73 validation)
- Risk bands: LOW < 0.30, MEDIUM 0.30–0.60, HIGH > 0.60 (<= on medium upper bound)
- Pydantic `extra="forbid"` — rejects unknown fields at API boundary
- Docker image uses `python:3.12-slim`, runs as root (no non-root user configured)
- No CI/CD pipeline exists — all testing is manual via `make test`
