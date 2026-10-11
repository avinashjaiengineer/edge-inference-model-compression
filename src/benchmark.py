import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

data = load_breast_cancer()
X = data.data.astype(np.float32)
y = data.target

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


def measure_latency(predict, sample, repetitions=300):
    # Warm-up: exclude initial startup effects.
    for _ in range(20):
        predict(sample)

    timings = []
    for _ in range(repetitions):
        start = time.perf_counter()
        predict(sample)
        timings.append((time.perf_counter() - start) * 1000)

    return {
        "p50_latency_ms": float(np.percentile(timings, 50)),
        "p95_latency_ms": float(np.percentile(timings, 95)),
        "mean_latency_ms": float(np.mean(timings)),
    }


def measure_throughput(predict, batch, iterations=100):
    # Warm-up, then time repeated full-batch predictions.
    for _ in range(5):
        predict(batch)

    start = time.perf_counter()
    for _ in range(iterations):
        predict(batch)
    elapsed = time.perf_counter() - start

    return iterations * len(batch) / elapsed


results = []

for name in ("baseline", "compact"):
    # 1. Native scikit-learn model
    model = joblib.load(ARTIFACTS / f"{name}.joblib")
    native_predict = model.predict

    native_predictions = native_predict(X_test)
    native_latency = measure_latency(native_predict, X_test[:1])

    results.append({
        "model": name,
        "runtime": "scikit-learn",
        "file_size_kb": (
            (ARTIFACTS / f"{name}.joblib").stat().st_size / 1024
        ),
        "accuracy": accuracy_score(y_test, native_predictions),
        "agreement_with_sklearn": 1.0,
        **native_latency,
        "batch_throughput_samples_sec": measure_throughput(
            native_predict, X_test
        ),
    })

    # 2. ONNX Runtime model
    session = ort.InferenceSession(
        str(ARTIFACTS / f"{name}.onnx"),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name

    def onnx_predict(batch):
        return session.run(
            [label_name],
            {input_name: np.asarray(batch, dtype=np.float32)},
        )[0]

    onnx_predictions = onnx_predict(X_test)
    onnx_latency = measure_latency(onnx_predict, X_test[:1])

    results.append({
        "model": name,
        "runtime": "ONNX Runtime",
        "file_size_kb": (
            (ARTIFACTS / f"{name}.onnx").stat().st_size / 1024
        ),
        "accuracy": accuracy_score(y_test, onnx_predictions),
        # Fraction of test records where ONNX matches scikit-learn.
        "agreement_with_sklearn": float(
            np.mean(onnx_predictions == native_predictions)
        ),
        **onnx_latency,
        "batch_throughput_samples_sec": measure_throughput(
            onnx_predict, X_test
        ),
    })

report = pd.DataFrame(results)
report.to_csv(REPORTS / "benchmark_results.csv", index=False)

pd.set_option("display.width", 200)
print("\n=== Inference Benchmark Results ===")
print(report.round(4).to_string(index=False))
print(f"\nSaved report: {REPORTS / 'benchmark_results.csv'}")
