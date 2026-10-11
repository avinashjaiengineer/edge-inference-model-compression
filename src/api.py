import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"

metadata = json.loads(
    (ARTIFACTS / "metadata.json").read_text(encoding="utf-8")
)

FEATURE_NAMES = metadata["feature_names"]
TARGET_NAMES = metadata["target_names"]

session = ort.InferenceSession(
    str(ARTIFACTS / "compact.onnx"),
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]

# Prometheus metrics. Paths are recorded as route templates so arbitrary
# URLs cannot create unbounded label cardinality.
HTTP_REQUESTS = Counter(
    "http_requests_total",
    "HTTP requests handled",
    ["method", "path", "status"],
)
HTTP_LATENCY = Histogram(
    "http_request_duration_seconds",
    "End-to-end HTTP request latency inside the server",
    ["method", "path"],
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
INFERENCE_LATENCY = Histogram(
    "model_inference_duration_seconds",
    "Time spent in ONNX Runtime session.run",
    buckets=(0.00001, 0.000025, 0.00005, 0.0001, 0.00025, 0.0005, 0.001, 0.005, 0.01),
)
PREDICTIONS = Counter(
    "model_predictions_total",
    "Predictions returned, by predicted class",
    ["prediction"],
)
MODEL_INFO = Gauge(
    "model_info",
    "Loaded model (value is always 1)",
    ["model", "runtime", "onnxruntime_version"],
)
MODEL_INFO.labels("compact.onnx", "onnxruntime-cpu", ort.__version__).set(1)


class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ...,
        description="Exactly 30 numeric features in dataset order",
    )


app = FastAPI(
    title="Edge Inference API",
    version="1.0.0",
    description="CPU-based inference using a compact ONNX model.",
)


@app.middleware("http")
async def record_metrics(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        route = request.scope.get("route")
        path = route.path if route is not None else "unmatched"
        HTTP_LATENCY.labels(request.method, path).observe(
            time.perf_counter() - start
        )
        HTTP_REQUESTS.labels(request.method, path, str(status)).inc()


@app.get("/")
def root():
    return {
        "project": "Edge Inference and Model Compression",
        "model": "compact.onnx",
        "runtime": "ONNX Runtime CPU",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": session is not None,
    }


@app.get("/features")
def features():
    return {
        "count": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
    }


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(request: PredictionRequest):
    if len(request.features) != len(FEATURE_NAMES):
        raise HTTPException(
            status_code=422,
            detail=f"Expected {len(FEATURE_NAMES)} features.",
        )

    if not all(np.isfinite(value) for value in request.features):
        raise HTTPException(
            status_code=422,
            detail="All features must be finite numbers.",
        )

    values = np.asarray(
        [request.features],
        dtype=np.float32,
    )

    with INFERENCE_LATENCY.time():
        outputs = session.run(
            None,
            {input_name: values},
        )

    predicted_class = int(np.asarray(outputs[0]).reshape(-1)[0])
    PREDICTIONS.labels(TARGET_NAMES[predicted_class]).inc()

    result = {
        "prediction_id": predicted_class,
        "prediction": TARGET_NAMES[predicted_class],
    }

    if len(outputs) > 1:
        probabilities = np.asarray(outputs[1])

        # Export options disable ZipMap, so probabilities should
        # be a numeric matrix with one row per input.
        if probabilities.ndim == 2:
            result["probabilities"] = {
                TARGET_NAMES[i]: float(probabilities[0, i])
                for i in range(len(TARGET_NAMES))
            }

    return result
