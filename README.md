# mlops-tabular-e2e

A reproducible MLOps loop for a tabular classifier. It validates data, builds features, trains a gradient-boosted model, tracks the run in MLflow, registers the model and promotes it, serves it with FastAPI in Docker, and runs a drift monitor (PSI and KS) that writes a report and flags retraining.

## Setup
```bash
cd ml/mlops-tabular-e2e
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
python -m pytest -q                 # 11 tests, ~20 s
python -m mlops.pipeline            # the whole pipeline, ~11 s on 4 CPUs
docker build -t adult-gbm .         # after the pipeline has written serving_model/
docker run -p 8000:8000 adult-gbm
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"rows":[{"age":45,"education_num":14,"hours_per_week":50,"capital_gain":0,"workclass":"Private","marital_status":"Married","occupation":"Exec","sex":"Female"}]}'
mlflow ui --backend-store-uri ./mlruns   # browse runs and the registered model
```
The pipeline writes `mlruns/` (MLflow file store with the run and the registry), `serving_model/` (the production model, exported) and `reports/drift_*.{json,md}`.

## Architecture
```
data.load ──► data.validate ──► train.build_pipeline (OneHot ► HistGradientBoosting)
                                      │
                         MLflow run (params, val_auc, model) ──► registry "adult-gbm" vN
                                      │
                 train.promote: alias "production" if AUC ≥ 0.75 and ≥ current production
                                      │
              train.export_production ► serving_model/ ► Docker ► FastAPI /predict
incoming batch ──► drift.drift_report (PSI per feature + KS for numerics) ──► reports/, retrain flag
```
| file | role |
|---|---|
| `mlops/data.py` | seeded synthetic Adult-like generator, CSV loader, schema validation (columns, nulls, ranges, categories) |
| `mlops/train.py` | sklearn pipeline, MLflow tracking and registration, alias promotion gate, export |
| `mlops/drift.py` | PSI (bins from reference quantiles) and KS two-sample test, retrain if any PSI > 0.2 |
| `mlops/serve.py` | FastAPI `/health` and `/predict`. Pydantic checks ranges and categories, so bad input gets a 422 |
| `mlops/pipeline.py` | the one-command entry point |

Design choices:
- **Data.** UCI Adult could not be downloaded from this sandbox (proxy policy). So the default data is a seeded synthetic generator with Adult-like columns and a known logistic label, about 21 % positive. Put the real file at `data/raw/adult.csv` (with the same column names) and the pipeline uses it instead.
- **Stages.** MLflow 3 deprecates registry stages, so "promotion" sets the `production` alias, which is the replacement MLflow recommends.
- **File store.** The SPEC asks for a local file store. MLflow 3 only allows it behind `MLFLOW_ALLOW_FILE_STORE=true`, which `train.py` sets. A `sqlite:///` URI also works.
- **Serialization.** Models use skops, a safe format. Only the one HistGradientBoosting tree type that skops flags is trusted, and no pickle is used.

## Results (seed 42, 20 000 synthetic rows, 4 CPUs)
| metric | value |
|---|---|
| validation AUC | 0.897 |
| pipeline wall time | ~11 s |
| Docker image | python:3.12-slim + mlflow-skinny, `/predict` verified with curl |

Drift on a batch like the training data: no retrain (all PSI < 0.01). On a shifted batch (ages +12 years, hours +8, a different occupation mix):

| feature | PSI | KS stat | KS p-value |
|---|---|---|---|
| age | 0.691 | 0.342 | 1.9e-190 |
| hours_per_week | 0.604 | 0.308 | 2e-153 |
| occupation | 0.331 | - | - |
| education_num | 0.003 | 0.019 | 0.49 |

→ **Retrain: YES**

## Not done (time box)
Hyper-parameter search, a remote MLflow server, label-based concept-drift monitoring, authentication on the API.
