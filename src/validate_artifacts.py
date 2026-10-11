"""Validate trained model artifacts before they are tested, packaged or deployed.

Checks, for every model:
  * the ONNX graph passes onnx.checker
  * the input is float32 with 30 features and a dynamic batch dimension
  * outputs are a label vector and an (N, 2) probability matrix (no ZipMap)
  * ONNX labels agree 100% with the scikit-learn model on the test split
  * ONNX probabilities match scikit-learn within a small tolerance
  * test accuracy meets a minimum
and for the deployable (compact) model, that it fits a size budget and is
smaller than the baseline.

Exit code is non-zero if any check fails, so CI can gate on it.

Usage:
    python -m src.validate_artifacts
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import onnx
import onnxruntime as ort
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"

MODELS = ("baseline", "compact")
DEPLOYED_MODEL = "compact"
MIN_ACCURACY = 0.93
MIN_AGREEMENT = 1.0
PROBABILITY_TOLERANCE = 1e-4
DEPLOYED_MAX_ONNX_KB = 200


def validate():
    failures = []

    def check(condition, message):
        print(f"  [{'PASS' if condition else 'FAIL'}] {message}")
        if not condition:
            failures.append(message)

    print("metadata.json")
    metadata = json.loads((ARTIFACTS / "metadata.json").read_text(encoding="utf-8"))
    n_features = metadata["n_features"]
    check(n_features == 30, f"n_features == 30 (got {n_features})")
    check(len(metadata["feature_names"]) == n_features, "feature_names length matches n_features")
    check(metadata["target_names"] == ["malignant", "benign"], "target_names == [malignant, benign]")

    data = load_breast_cancer()
    _, X_test, _, y_test = train_test_split(
        data.data.astype(np.float32),
        data.target,
        test_size=0.20,
        random_state=42,
        stratify=data.target,
    )

    sizes = {}
    for name in MODELS:
        onnx_path = ARTIFACTS / f"{name}.onnx"
        print(f"\n{name}")

        model_proto = onnx.load(str(onnx_path))
        try:
            onnx.checker.check_model(model_proto)
            check(True, "onnx.checker passes")
        except onnx.checker.ValidationError as exc:
            check(False, f"onnx.checker passes ({exc})")

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        inputs, outputs = session.get_inputs(), session.get_outputs()
        check(len(inputs) == 1, "exactly one input")
        check(inputs[0].type == "tensor(float)", f"input dtype float32 (got {inputs[0].type})")
        check(inputs[0].shape[-1] == n_features, f"input width {n_features} (got {inputs[0].shape})")
        check(not isinstance(inputs[0].shape[0], int), "batch dimension is dynamic")
        check(len(outputs) == 2, "two outputs (label, probabilities)")
        check(outputs[1].type == "tensor(float)", "probabilities are a tensor (ZipMap disabled)")

        labels, probabilities = session.run(None, {inputs[0].name: X_test})
        model = joblib.load(ARTIFACTS / f"{name}.joblib")
        sk_labels = model.predict(X_test)
        sk_probabilities = model.predict_proba(X_test)

        agreement = float(np.mean(labels == sk_labels))
        max_diff = float(np.max(np.abs(probabilities - sk_probabilities)))
        accuracy = accuracy_score(y_test, labels)
        check(probabilities.shape == (len(X_test), 2), f"probability shape (N, 2) (got {probabilities.shape})")
        check(agreement >= MIN_AGREEMENT, f"label agreement with scikit-learn >= {MIN_AGREEMENT} (got {agreement:.4f})")
        check(max_diff <= PROBABILITY_TOLERANCE, f"max probability diff <= {PROBABILITY_TOLERANCE} (got {max_diff:.2e})")
        check(accuracy >= MIN_ACCURACY, f"accuracy >= {MIN_ACCURACY} (got {accuracy:.4f})")

        sizes[name] = onnx_path.stat().st_size / 1024

    print(f"\nsize budget ({DEPLOYED_MODEL})")
    check(sizes[DEPLOYED_MODEL] <= DEPLOYED_MAX_ONNX_KB,
          f"{DEPLOYED_MODEL}.onnx <= {DEPLOYED_MAX_ONNX_KB} KB (got {sizes[DEPLOYED_MODEL]:.1f} KB)")
    check(sizes[DEPLOYED_MODEL] < sizes["baseline"],
          f"{DEPLOYED_MODEL}.onnx smaller than baseline.onnx ({sizes['baseline']:.1f} KB)")

    return failures


def main():
    failures = validate()
    if failures:
        print(f"\n{len(failures)} check(s) failed.")
        sys.exit(1)
    print("\nAll artifact checks passed.")


if __name__ == "__main__":
    main()
