import os
import gradio as gr
import pandas as pd
import time as t
from datetime import datetime
from huggingface_hub import HfApi
from model_utils import load_model_bundle

# ---------- Config HF (para monitoreo) ----------
HF_DATASET_ID  = os.getenv("HF_DATASET_ID", "tuusuario/iris-data")   # cambialo en el Space y en Actions
HF_PUSH_TOKEN  = os.getenv("HF_PUSH_TOKEN")  # define Secret en el Space
LOG_LOCAL_PATH = "monitoring/prod_log.csv"
os.makedirs("monitoring", exist_ok=True)

def log_prediction(features: dict, pred: int, model_version: str = "latest",
                   y_true: int | None = None, latency_ms: float = 0.0):
    row = {
        **features,
        "prediction": int(pred),
        "y_true": (None if y_true is None else int(y_true)),
        "model_version": model_version,
        "latency_ms": float(latency_ms),
        "ts": datetime.utcnow().isoformat()
    }
    df = pd.DataFrame([row])
    header = not os.path.exists(LOG_LOCAL_PATH)
    df.to_csv(LOG_LOCAL_PATH, mode="a", header=header, index=False)

def push_logs_to_hub():
    if not HF_PUSH_TOKEN:
        return "Sin token HF_PUSH_TOKEN en Space; no se subió."
    api = HfApi(token=HF_PUSH_TOKEN)
    api.upload_file(
        path_or_fileobj=LOG_LOCAL_PATH,
        path_in_repo="monitoring/prod_log.csv",
        repo_id=HF_DATASET_ID,
        repo_type="dataset",
        commit_message="append production logs"
    )
    return "Logs subidos al Dataset repo."

# ---------- Carga de modelo ----------
bundle = load_model_bundle()
model = bundle["model"]
target_names = bundle.get("target_names", ["setosa","versicolor","virginica"])

def predict(sepal_length, sepal_width, petal_length, petal_width, y_true=None):
    features = {
        "sepal length (cm)": float(sepal_length),
        "sepal width (cm)": float(sepal_width),
        "petal length (cm)": float(petal_length),
        "petal width (cm)": float(petal_width),
    }
    X = pd.DataFrame([features])
    t0 = t.time()
    pred_idx = int(model.predict(X)[0])
    latency = (t.time() - t0) * 1000
    pred_label = target_names[pred_idx] if pred_idx < len(target_names) else str(pred_idx)
    # logging
    log_prediction(features, pred_idx, model_version="latest", y_true=None if y_true in (None,"") else int(y_true), latency_ms=latency)
    return pred_label, f"{latency:.2f} ms"

with gr.Blocks() as demo:
    gr.Markdown("# Iris Classifier — Demo MLOps")
    with gr.Row():
        sepal_length = gr.Number(label="sepal length (cm)", value=5.1)
        sepal_width  = gr.Number(label="sepal width (cm)",  value=3.5)
        petal_length = gr.Number(label="petal length (cm)", value=1.4)
        petal_width  = gr.Number(label="petal width (cm)",  value=0.2)
        y_true       = gr.Textbox(label="Label real (opcional): 0/1/2", value="")
    btn = gr.Button("Predecir")
    out_pred = gr.Textbox(label="Predicción")
    out_lat  = gr.Textbox(label="Latencia")
    btn.click(predict, inputs=[sepal_length, sepal_width, petal_length, petal_width, y_true],
              outputs=[out_pred, out_lat])

    gr.Markdown("## Monitoreo")
    flush_btn = gr.Button("Subir logs a HF Dataset")
    flush_msg = gr.Textbox(label="Estado subida", interactive=False)
    flush_btn.click(lambda: push_logs_to_hub(), outputs=flush_msg)

if __name__ == "__main__":
    demo.launch()
