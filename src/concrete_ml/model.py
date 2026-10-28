"""Cross-validated regression and held-out split-conformal calibration."""

import io
import math

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import archive, figure, regression, save_report

FEATURES = [
    "cement",
    "slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
    "age",
]


def conformal_radius(residuals, alpha=0.1):
    residuals = np.asarray(residuals, dtype=float)
    if (
        residuals.ndim != 1
        or not len(residuals)
        or not np.isfinite(residuals).all()
        or (residuals < 0).any()
        or not 0 < alpha < 1
    ):
        raise ValueError("Residuals must be finite nonnegative values and alpha must be in (0,1)")
    rank = math.ceil((len(residuals) + 1) * (1 - alpha))
    if rank > len(residuals):
        raise ValueError("Insufficient calibration examples for a finite interval at this alpha")
    return float(np.sort(residuals)[rank - 1])


def train(root):
    with archive(root) as source:
        frame = pd.read_excel(io.BytesIO(source.read("Concrete_Data.xls")), engine="xlrd")
    frame.columns = FEATURES + ["strength"]
    raw = len(frame)
    # Remove exact duplicate feature rows before any split (keep their average target).
    frame = frame.groupby(FEATURES, as_index=False, sort=True).strength.mean()
    X, y = frame[FEATURES], frame.strength
    fit, rest = train_test_split(np.arange(len(frame)), test_size=0.4, random_state=42)
    calibration, test = train_test_split(rest, test_size=0.5, random_state=42)
    candidates = {
        "mean_baseline": DummyRegressor(),
        "scaled_ridge": make_pipeline(StandardScaler(), Ridge(alpha=10)),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=250,
            max_leaf_nodes=15,
            learning_rate=0.07,
            l2_regularization=2,
            random_state=42,
        ),
    }
    cv = {}
    for name, model in candidates.items():
        scores = -cross_val_score(
            model,
            X.iloc[fit],
            y.iloc[fit],
            cv=KFold(5, shuffle=True, random_state=42),
            scoring="neg_mean_absolute_error",
        )
        cv[name] = {"mean_mae": float(scores.mean()), "std_mae": float(scores.std())}
    selected = min(cv, key=lambda name: cv[name]["mean_mae"])
    results = {}
    for name, model in candidates.items():
        model.fit(X.iloc[fit], y.iloc[fit])
        results[name] = regression(y.iloc[test], model.predict(X.iloc[test]))
    model = candidates[selected]
    radius = conformal_radius(np.abs(y.iloc[calibration] - model.predict(X.iloc[calibration])))
    prediction = model.predict(X.iloc[test])
    coverage = float(np.mean(np.abs(y.iloc[test] - prediction) <= radius))
    order = np.argsort(y.iloc[test].to_numpy())
    plt.figure(figsize=(9, 4.5))
    plt.fill_between(
        np.arange(len(test)),
        prediction[order] - radius,
        prediction[order] + radius,
        alpha=0.2,
        color="#2563eb",
        label="Nominal 90% marginal interval",
    )
    plt.scatter(
        np.arange(len(test)),
        y.iloc[test].to_numpy()[order],
        s=12,
        label="Observed",
        color="#0f172a",
    )
    plt.plot(prediction[order], color="#2563eb", label="Predicted")
    plt.xlabel("Test examples sorted by observed strength (visualization only)")
    plt.ylabel("Compressive strength (MPa)")
    plt.legend()
    figure(root, "evaluation.png")
    result = save_report(
        root,
        {
            "raw_rows": raw,
            "unique_feature_rows": len(frame),
            "split": {"train": len(fit), "calibration": len(calibration), "test": len(test)},
            "selected": selected,
            "cross_validation": cv,
            "test": results,
            "conformal": {
                "alpha": 0.1,
                "radius_mpa": radius,
                "test_coverage": coverage,
                "mean_width_mpa": 2 * radius,
            },
        },
    )
    reference = {
        name: {
            "min": float(X.iloc[fit][name].min()),
            "max": float(X.iloc[fit][name].max()),
            "mean": float(X.iloc[fit][name].mean()),
            "std": float(X.iloc[fit][name].std()),
        }
        for name in FEATURES
    }
    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "radius": radius,
            "alpha": 0.1,
            "reference": reference,
            "format_version": 1,
        },
        root / "artifacts/model.joblib",
    )
    return result


def predict(root, payload):
    from .api import Instance, infer

    instance = Instance.model_validate(payload)
    artifact = joblib.load(root / "artifacts/model.joblib")
    return infer(artifact, [instance])[0]
