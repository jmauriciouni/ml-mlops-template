import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

BUNDLE_PATH = MODELS_DIR / "model-latest.pkl"

def save_model_bundle(bundle):
    joblib.dump(bundle, BUNDLE_PATH)

def load_model_bundle():
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError("models/model-latest.pkl not found. Run CT (train.yml) first.")
    return joblib.load(BUNDLE_PATH)
