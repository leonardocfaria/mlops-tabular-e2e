# mlops-tabular-e2e Implementation Plan

## Brainstorm (answered from SPEC + repo evidence, no human available)
- **What is it?** A portfolio-grade, reproducible tabular MLOps loop: data → validation → features → train → track → register → promote → serve → drift.
- **Success:** `python -m mlops.pipeline` runs end to end from a clean checkout; MLflow run + registered model exist; Docker image serves `/predict`; drift report flags a shifted batch.
- **Non-goals:** hyper-parameter search, a remote MLflow server, Kubernetes, auth.
- **Data:** UCI Adult is blocked by the proxy (DECISIONS #3). Use a seeded synthetic Adult-like generator with the same kind of columns (numeric + categorical, ~24 % positive). `data/raw/adult.csv` is used instead if present.
- **Registry stages:** MLflow 3 deprecates stages in favour of aliases. Promotion = set alias `production` when the new version beats a fixed AUC floor and the current production AUC.
- **Docker:** a file-store registry stores absolute paths, so the pipeline exports the production model to `serving_model/`; the image copies that folder and loads it with `mlflow.sklearn.load_model`.

## Files
- `mlops/data.py` — `make_synthetic(n, seed, shift=0.0)`, `load(path|None)`, `SCHEMA`, `validate(df)` (raises `ValueError` with all problems).
- `mlops/train.py` — `build_pipeline()` (OneHot + HistGradientBoosting), `train_and_register(df, tracking_uri) -> (run_id, version, auc)`, `promote(...)`, `export_production(...)`.
- `mlops/drift.py` — `psi(ref, cur)`, `drift_report(ref_df, cur_df) -> dict` with PSI + KS per feature and `retrain` flag; `write_report(report, dir)` (JSON + Markdown).
- `mlops/serve.py` — FastAPI app, `MODEL_DIR` env, `/health`, `/predict`.
- `mlops/pipeline.py` — one-command entry point.
- `Dockerfile`, `requirements.txt`, `requirements-serve.txt`, `tests/`.

## Tasks (TDD, commit after each)
- [x] 1. data + validation: tests for schema pass/fail (missing column, bad category, out-of-range, nulls).
- [x] 2. drift: PSI≈0 on same distribution, PSI>0.2 and retrain=True on shifted batch.
- [x] 3. train/register/promote/export against a temp MLflow file store.
- [x] 4. serve: TestClient `/predict` on an exported model; 422 on bad payload.
- [x] 5. pipeline end to end in a temp dir; README; CI workflow; Docker build + curl check (manual, documented).
