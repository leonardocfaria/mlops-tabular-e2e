# mlops-tabular-e2e

## Goal
Build "End-to-end MLOps pipeline for a tabular model". Train a gradient-boosted classifier on a public tabular dataset (for example UCI Adult) with a reproducible pipeline: data validation, feature pipeline, experiment tracking (MLflow local file store), model registry stage promotion, FastAPI serving, a Docker image, and a drift monitor (PSI/KS on incoming batches) that writes a report and flags retraining. Keep it CPU-friendly (the cloud box has no GPU): small models and datasets, a fixed random seed, and a documented runtime.

## Definition of done
- [x] pipeline runs end to end with one command
- [x] MLflow run and registered model are produced
- [x] /predict endpoint serves the registered model in Docker
- [x] drift report generated on a shifted batch
- [x] Tests pass with one command; CI workflow runs them
- [x] README with setup, usage, architecture and results
- [x] Self code review done
- [ ] Draft PR open

