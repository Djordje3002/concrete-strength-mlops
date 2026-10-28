import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyRegressor

from concrete_ml.api import create_app
from concrete_ml.model import FEATURES, conformal_radius


@pytest.fixture
def sample():
    return dict(zip(FEATURES, [300.0, 50.0, 30.0, 180.0, 5.0, 1000.0, 750.0, 28.0]))


@pytest.fixture
def client(tmp_path, sample):
    frame = pd.DataFrame([sample, sample], columns=FEATURES)
    artifact = {
        "model": DummyRegressor().fit(frame, [30.0, 40.0]),
        "features": FEATURES,
        "radius": 5.0,
        "alpha": 0.1,
        "format_version": 1,
        "reference": {
            name: {
                "min": value * 0.5,
                "max": value * 1.5,
                "mean": value,
                "std": max(1.0, value * 0.1),
            }
            for name, value in sample.items()
        },
    }
    path = tmp_path / "model.joblib"
    joblib.dump(artifact, path)
    with TestClient(create_app(path)) as connected:
        yield connected


def test_conformal_uses_finite_sample_order_statistic():
    assert conformal_radius(np.arange(1, 10), alpha=0.2) == 8.0
    with pytest.raises(ValueError, match="Insufficient"):
        conformal_radius([1], alpha=0.1)


@pytest.mark.parametrize("values", [[], [-1, 2], [float("nan"), 2]])
def test_invalid_calibration_rejected(values):
    with pytest.raises(ValueError):
        conformal_radius(values)


def test_missing_model_has_live_health_and_failed_readiness(tmp_path, sample):
    with TestClient(create_app(tmp_path / "missing.joblib")) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.post("/predict", json={"instances": [sample]}).status_code == 503


def test_batch_prediction_interval_and_metadata(client, sample):
    assert client.get("/ready").status_code == 200
    response = client.post("/predict", json={"instances": [sample, sample]})
    assert response.status_code == 200
    body = response.json()
    assert len(body["model_version"]) == 12
    assert body["predictions"][0]["strength_mpa"] == 35.0
    assert body["predictions"][0]["interval_mpa"] == [30.0, 40.0]
    assert client.get("/metrics").json()["predictions"] == 2


@pytest.mark.parametrize(
    "patch", [{"cement": -1.0}, {"water": "180"}, {"unknown": 1}, {"age": 0.0}]
)
def test_bad_inputs_rejected(client, sample, patch):
    assert client.post("/predict", json={"instances": [sample | patch]}).status_code == 422


def test_batch_size_limits(client, sample):
    assert client.post("/predict", json={"instances": []}).status_code == 422
    assert client.post("/predict", json={"instances": [sample] * 65}).status_code == 422


def test_monitor_flags_out_of_range_input(client, sample):
    result = client.post("/monitor", json={"instances": [sample | {"cement": 700.0}]}).json()
    assert result["outside_range_fraction"]["cement"] == 1.0
    assert result["outside_range_fraction"]["age"] == 0.0
