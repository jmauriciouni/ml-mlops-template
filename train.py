import os
import json
from pathlib import Path

# (si no usarás MLflow, puedes borrar estas 3 líneas)
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from model_utils import save_model_bundle
import joblib   # <--- NUEVO

METRICS_PATH = Path("metrics.json")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "model-latest.pkl"

ACCURACY_THRESHOLD = float(os.getenv("ACCURACY_THRESHOLD", "0.90"))
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "iris-ci-ct")

def train_and_eval():
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    yhat = clf.predict(Xte)
    metrics = {
        "accuracy": float(accuracy_score(yte, yhat)),
        "f1_macro": float(f1_score(yte, yhat, average="macro")),
        "classes": iris.target_names.tolist(),
    }

    # === GUARDA EL MODELO (dos caminos para asegurar) ===
    bundle = {"model": clf, "target_names": iris.target_names.tolist()}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1) tu helper (por compatibilidad con tu app)
    save_model_bundle(bundle)  # debería escribir models/model-latest.pkl

    # 2) redundante/seguro: escribir explícito con joblib
    joblib.dump(bundle, MODEL_PATH)

    # Verificación dura: si no existe, falla el CT aquí
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not created after training")

    # guarda métricas
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    # (opcional) log MLflow local
    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run():
            mlflow.log_params({"model": "LogisticRegression", "max_iter": 1000})
            mlflow.log_metrics({"accuracy": metrics["accuracy"], "f1_macro": metrics["f1_macro"]})
            mlflow.sklearn.log_model(clf, artifact_path="model")
            mlflow.log_artifact(str(METRICS_PATH))
            mlflow.log_artifact(str(MODEL_PATH))
    except Exception:
        pass

    # prints útiles para el log
    print(f"Saved model at: {MODEL_PATH.resolve()}")
    print("Train OK:", metrics)
    approved = metrics["accuracy"] >= ACCURACY_THRESHOLD
    print(f"APPROVED={str(approved).lower()}")
    return metrics

if __name__ == "__main__":
    train_and_eval()
