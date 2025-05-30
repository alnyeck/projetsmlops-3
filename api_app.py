from fastapi import FastAPI
from pydantic import BaseModel
import subprocess, uuid, os
from joblib import load

app = FastAPI(title="Red‑Wine MLOps API")

class TrainRequest(BaseModel):
    model: str       # elasticnet | ridge | lasso
    alpha: float
    l1_ratio: float | None = None   # ignoré hors ElasticNet

@app.post("/train")
def train(req: TrainRequest):
    """Lance un entraînement MLflow en sous‑processus et sauvegarde le modèle."""
    run_id = str(uuid.uuid4())[:8]
    model_path = f"models/model_{run_id}.pkl"
    cmd = [
        "python", "train_model.py",
        "--model", req.model,
        "--alpha", str(req.alpha),
        "--output", model_path
    ]
    if req.model == "elasticnet":
        cmd += ["--l1_ratio", str(req.l1_ratio or 0.5)]
    # On passe la variable d'env pour que le sous‑processus parle à MLflow
    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = env.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    subprocess.Popen(cmd, env=env)
    return {
        "status": "started",
        "run_id": run_id,
        "cmd": " ".join(cmd),
        "model_path": model_path
    }

class PredictRequest(BaseModel):
    alcohol: float
    volatile_acidity: float
    sulphates: float

class PredictRequest(BaseModel):
    model_name: str
    alcohol: float
    volatile_acidity: float
    sulphates: float

@app.post("/predict")
def predict(inp: PredictRequest):
    """Charge le modèle spécifié et effectue une prédiction."""
    model_path = f"models/{inp.model_name}.pkl"
    if not os.path.exists(model_path):
        return {"error": f"Model '{inp.model_name}' not found."}
    model = load(model_path)
    features = [[inp.alcohol, inp.volatile_acidity, inp.sulphates]]
    pred = model.predict(features)
    return {"quality_estimate": round(float(pred[0]), 2)}