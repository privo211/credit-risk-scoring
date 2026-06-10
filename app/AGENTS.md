# app/ — FastAPI Service

**4 files, ~300 lines. API layer.**

## OVERVIEW

FastAPI application serving ML predictions via `/health`, `/predict`, `/batch_predict`. Pydantic validation, structured logging, CORS middleware.

## FILES

| File | Lines | Role |
|------|-------|------|
| `main.py` | 104 | FastAPI app creation, 3 routes, lifespan (model loading), CORS |
| `schemas.py` | 124 | Pydantic models: `Applicant` (20 fields), `PredictionResult`, request/response schemas |
| `inference.py` | 68 | Module-level model singleton, `predict_single`, `predict_batch_endpoint`, latency logging |
| `__init__.py` | 0 | Empty — no re-exports |

## CONVENTIONS

- `lifespan` context manager for model loading on startup / cleanup on shutdown
- `extra="forbid"` on all Pydantic models — rejects unknown fields
- All 20 applicant fields are `Optional` with `default=None`; missing columns filled server-side
- Risk band mapping imported from `src.predict.get_risk_band`
- CORS middleware allows all origins (development config)

## ANTI-PATTERNS

- **`inference.py` logging setup at import time** — `StreamHandler` configured at module level, not in app lifespan
- **Thin wrapper layer** — `inference.py` adds minimal value over calling `src.predict` directly; main purpose is module-level `_model`/`_preprocessor` singletons
- **Late import** — `get_risk_band` imported inside `predict_single()` function body instead of top of file

## API ENDPOINTS

| Method | Path | Input | Output |
|--------|------|-------|--------|
| GET | `/health` | — | `{status, model_loaded, version}` |
| POST | `/predict` | single `Applicant` JSON | `{probability, risk_band, model_version}` |
| POST | `/batch_predict` | `{applicants: [...]}` | `{predictions: [...], count}` |
