import json

from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer

from src.api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_feature_list():
    response = client.get("/features")
    assert response.status_code == 200
    assert response.json()["count"] == 30


def test_prediction():
    sample = load_breast_cancer().data[0].tolist()

    response = client.post(
        "/predict",
        json={"features": sample},
    )

    assert response.status_code == 200
    assert response.json()["prediction"] in ["malignant", "benign"]


def test_metrics_exposed():
    sample = load_breast_cancer().data[0].tolist()
    client.post("/predict", json={"features": sample})

    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.text
    assert 'http_requests_total{method="POST",path="/predict",status="200"}' in body
    assert "model_inference_duration_seconds_count" in body
    assert "model_predictions_total" in body


def test_reject_wrong_feature_count():
    response = client.post(
        "/predict",
        json={"features": [1.0, 2.0]},
    )
    assert response.status_code == 422


def test_reject_non_finite_features():
    sample = load_breast_cancer().data[0].tolist()
    sample[0] = float("nan")

    # httpx refuses to serialize NaN, so send the raw JSON body.
    response = client.post(
        "/predict",
        content=json.dumps({"features": sample}),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
