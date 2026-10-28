"""Local inference API. No training or downloads happen inside a request."""

import hashlib
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .model import FEATURES


class Instance(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)
    cement: float = Field(gt=0, le=1500)
    slag: float = Field(ge=0, le=1500)
    fly_ash: float = Field(ge=0, le=1500)
    water: float = Field(gt=0, le=1000)
    superplasticizer: float = Field(ge=0, le=200)
    coarse_aggregate: float = Field(gt=0, le=3000)
    fine_aggregate: float = Field(gt=0, le=3000)
    age: float = Field(ge=1, le=365)


class Batch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instances: list[Instance] = Field(min_length=1, max_length=64)


def out_of_range(artifact, frame):
    return {
        name: (frame[name] < artifact["reference"][name]["min"])
        | (frame[name] > artifact["reference"][name]["max"])
        for name in FEATURES
    }


def infer(artifact, instances):
    frame = pd.DataFrame([record.model_dump() for record in instances], columns=FEATURES)
    prediction = artifact["model"].predict(frame)
    flags = out_of_range(artifact, frame)
    return [
        {
            "strength_mpa": float(value),
            "interval_mpa": [float(value - artifact["radius"]), float(value + artifact["radius"])],
            "nominal_marginal_coverage": 1 - artifact["alpha"],
            "outside_training_range": [name for name in FEATURES if bool(flags[name].iloc[index])],
        }
        for index, value in enumerate(prediction)
    ]


def create_app(model_path=None):
    path = Path(model_path or os.environ.get("MODEL_PATH", "artifacts/model.joblib"))

    @asynccontextmanager
    async def lifespan(app):
        app.state.artifact = None
        app.state.model_version = None
        app.state.requests = 0
        app.state.predictions = 0
        if path.is_file():
            artifact = joblib.load(path)
            if artifact.get("format_version") != 1 or artifact.get("features") != FEATURES:
                raise ValueError("Unsupported model artifact format")
            app.state.artifact = artifact
            app.state.model_version = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        yield

    app = FastAPI(
        title="Concrete Strength Prediction",
        version="1.0.0",
        lifespan=lifespan,
        description="Portfolio inference service on historical laboratory data. Intervals assume exchangeability; this service does not validate structural designs.",
    )

    def ready():
        if app.state.artifact is None:
            raise HTTPException(
                status_code=503, detail="Model unavailable; run python -m concrete_ml train first"
            )
        return app.state.artifact

    @app.get("/health")
    def health():
        return {"status": "alive", "model_loaded": app.state.artifact is not None}

    @app.get("/ready")
    def readiness():
        ready()
        return {"status": "ready", "model_version": app.state.model_version}

    @app.post("/predict")
    def prediction(batch: Batch):
        result = infer(ready(), batch.instances)
        app.state.requests += 1
        app.state.predictions += len(result)
        return {"model_version": app.state.model_version, "predictions": result}

    @app.post("/monitor")
    def monitor(batch: Batch):
        artifact = ready()
        frame = pd.DataFrame([x.model_dump() for x in batch.instances], columns=FEATURES)
        flags = out_of_range(artifact, frame)
        return {
            "batch_size": len(frame),
            "outside_range_fraction": {name: float(flag.mean()) for name, flag in flags.items()},
            "standardized_mean_shift": {
                name: float(
                    abs(frame[name].mean() - artifact["reference"][name]["mean"])
                    / max(artifact["reference"][name]["std"], 1e-9)
                )
                for name in FEATURES
            },
            "interpretation": "Descriptive input checks; not a statistical drift test or accuracy estimate",
        }

    @app.get("/metrics")
    def metrics():
        return {
            "successful_requests": app.state.requests,
            "predictions": app.state.predictions,
            "scope": "current process lifetime",
        }

    return app


app = create_app()
