# Prediction API and deployment

Train first, then start the service from the repository root:

```bash
python -m concrete_ml train
uvicorn concrete_ml.api:app --host 127.0.0.1 --port 8000
```

Open [interactive API documentation](http://127.0.0.1:8000/docs) or submit the complete example:

```bash
curl --fail-with-body http://127.0.0.1:8000/ready
curl --fail-with-body http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @examples/request.json
```

The request wraps 1–64 instances in an `instances` list. All eight named fields are required. Ingredients are kg/m³, age is days, and the response is MPa. Extra fields, numeric strings, NaN/infinity, negative ingredients and unsupported age ranges are rejected with HTTP 422. A valid request without a model returns 503. The model loads once at startup; requests never download data or start training.

The response includes an artifact SHA-256 prefix as `model_version`, predicted strength, a nominal 90% marginal interval, and any inputs outside the model's training range. Intervals are symmetric and untruncated, so a lower bound may be negative. Range flags do not make a prediction reliable or unreliable by themselves.

| Route | Meaning |
| --- | --- |
| `GET /health` | Process liveness; remains 200 when model is absent |
| `GET /ready` | 200 with artifact ID when loaded; 503 otherwise |
| `POST /predict` | Validated batch inference and uncertainty intervals |
| `POST /monitor` | Per-feature range fractions and standardized mean shifts |
| `GET /metrics` | Successful request and prediction counters for this process |
| `GET /docs` | OpenAPI interactive documentation |

Use `MODEL_PATH=/absolute/path/model.joblib` to select a trusted artifact at startup. Invalid artifact formats fail startup. Joblib is not safe for untrusted files. The artifact ID is an integrity identifier, not a digital signature or model registry.

## Container

After training locally:

```bash
docker build -t concrete-strength:local .
docker run --rm -p 127.0.0.1:8000:8000 concrete-strength:local
```

The container uses a non-root runtime user and checks readiness. It copies the locally trained model rather than downloading data during startup. Docker was unavailable in the authoring environment, so the container build/run remains unverified; the Python service and HTTP contracts were exercised locally.

## Operating boundaries

The default service is a local demonstration. Before public deployment, put it behind authentication, TLS, request-body limits, rate limiting and appropriate service monitoring. Keep artifacts immutable, collect consented labeled outcomes for accuracy monitoring, and validate on the actual mixture/laboratory population. Counters reset on restart and are per worker. `/monitor` describes at most 64 inputs and is not a significance test. There is no online learning or persistent request logging.

This model is an ML engineering demonstration. It cannot replace laboratory testing or engineering validation of structural materials.
