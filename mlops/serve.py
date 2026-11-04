"""FastAPI serving of the exported production model. MODEL_DIR points at an MLflow model folder."""
import os
from contextlib import asynccontextmanager
from typing import Literal

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .data import CATEGORICAL, CATEGORIES, NUMERIC


class Row(BaseModel):
    age: float = Field(ge=17, le=90)
    education_num: float = Field(ge=1, le=16)
    hours_per_week: float = Field(ge=1, le=99)
    capital_gain: float = Field(ge=0, le=99999)
    workclass: Literal[tuple(CATEGORIES["workclass"])]
    marital_status: Literal[tuple(CATEGORIES["marital_status"])]
    occupation: Literal[tuple(CATEGORIES["occupation"])]
    sex: Literal[tuple(CATEGORIES["sex"])]


class Request(BaseModel):
    rows: list[Row] = Field(min_length=1, max_length=1000)


state = {}


@asynccontextmanager
async def lifespan(app):
    state["model"] = mlflow.sklearn.load_model(os.environ.get("MODEL_DIR", "serving_model"))
    yield
    state.clear()


app = FastAPI(title="adult-gbm", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: Request):
    X = pd.DataFrame([r.model_dump() for r in req.rows])[NUMERIC + CATEGORICAL]
    proba = state["model"].predict_proba(X)[:, 1]
    return {"predictions": [{"probability": float(p), "label": int(p > 0.5)} for p in proba]}
