import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from skl2onnx import to_onnx

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

# Load a built-in dataset: no external download required.
dataset = load_breast_cancer()
X = dataset.data.astype(np.float32)
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

models = {
    "baseline": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    ),
    "compact": RandomForestClassifier(
        n_estimators=60,
        max_depth=6,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
}

metadata = {
    "feature_names": dataset.feature_names.tolist(),
    "target_names": dataset.target_names.tolist(),
    "n_features": int(X.shape[1]),
}

(ARTIFACTS / "metadata.json").write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)

for name, model in models.items():
    print(f"\nTraining {name} model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Save the Python model for reference benchmarking.
    joblib.dump(model, ARTIFACTS / f"{name}.joblib")

    # Export a portable ONNX model.
    # Disable ZipMap so probabilities are returned as a numeric array.
    onnx_model = to_onnx(
        model,
        X_train[:1],
        options={id(model): {"zipmap": False}},
    )

    (ARTIFACTS / f"{name}.onnx").write_bytes(
        onnx_model.SerializeToString()
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Saved {name}.joblib")
    print(f"Saved {name}.onnx")

print("\nTraining and ONNX export complete.")
print(f"Artifacts directory: {ARTIFACTS}")
