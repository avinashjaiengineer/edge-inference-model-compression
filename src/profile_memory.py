"""Memory profiling for each model/runtime pair.

Each measurement runs in a fresh subprocess so that one model's allocations
cannot leak into another's numbers. The worker imports only the runtime it
needs (numpy + joblib/scikit-learn, or numpy + onnxruntime), which shows the
real footprint of an ONNX-only edge process.

Usage:
    python -m src.profile_memory
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"

MB = 1024 * 1024


def worker(model_name, runtime, data_path, single_calls, batch_calls):
    import psutil

    process = psutil.Process()
    peak = 0

    def rss():
        nonlocal peak
        value = process.memory_info().rss
        peak = max(peak, value)
        return value

    before_import = rss()

    import numpy as np

    if runtime == "scikit-learn":
        import joblib
        import sklearn.ensemble  # noqa: F401  (count the library itself)
    else:
        import onnxruntime as ort

    X_test = np.load(data_path)
    after_import = rss()

    if runtime == "scikit-learn":
        model = joblib.load(ARTIFACTS / f"{model_name}.joblib")
        # Single-threaded to match edge serving and avoid pool overhead.
        model.n_jobs = 1
        predict = model.predict
    else:
        session = ort.InferenceSession(
            str(ARTIFACTS / f"{model_name}.onnx"),
            providers=["CPUExecutionProvider"],
        )
        input_name = session.get_inputs()[0].name

        def predict(batch):
            return session.run(None, {input_name: batch})

    after_load = rss()

    for i in range(single_calls):
        predict(X_test[i % len(X_test):i % len(X_test) + 1])
        rss()
    for _ in range(batch_calls):
        predict(X_test)
        rss()

    after_inference = rss()

    print(json.dumps({
        "model": model_name,
        "runtime": runtime,
        "process_base_mb": before_import / MB,
        "runtime_import_mb": (after_import - before_import) / MB,
        "model_load_mb": (after_load - after_import) / MB,
        "inference_growth_mb": (after_inference - after_load) / MB,
        "peak_rss_mb": peak / MB,
    }))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-calls", type=int, default=1000)
    parser.add_argument("--batch-calls", type=int, default=100)
    parser.add_argument("--worker", nargs=3, metavar=("MODEL", "RUNTIME", "DATA"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        model_name, runtime, data_path = args.worker
        worker(model_name, runtime, data_path, args.single_calls, args.batch_calls)
        return

    # Imported only in the parent so workers start from a clean process.
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    REPORTS.mkdir(exist_ok=True)
    data = load_breast_cancer()
    _, X_test, _, _ = train_test_split(
        data.data.astype(np.float32),
        data.target,
        test_size=0.20,
        random_state=42,
        stratify=data.target,
    )

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        data_path = Path(tmp) / "x_test.npy"
        np.save(data_path, X_test)

        for model_name in ("baseline", "compact"):
            for runtime in ("scikit-learn", "ONNX Runtime"):
                output = subprocess.run(
                    [
                        sys.executable, "-m", "src.profile_memory",
                        "--single-calls", str(args.single_calls),
                        "--batch-calls", str(args.batch_calls),
                        "--worker", model_name, runtime, str(data_path),
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                results.append(json.loads(output.stdout.strip().splitlines()[-1]))

    report = pd.DataFrame(results)
    report.to_csv(REPORTS / "memory_results.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== Memory Profile (RSS, MB) ===")
    print(report.round(2).to_string(index=False))
    print(f"\nSaved report: {REPORTS / 'memory_results.csv'}")


if __name__ == "__main__":
    main()
