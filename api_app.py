from fastapi import FastAPI
from pydantic import BaseModel
import subprocess, uuid, os
import mlflow.sklearn
import numpy as np

app = FastAPI(title="Red‑Wine MLOps API")

class TrainRequest(BaseModel):
    model: str       # elasticnet | ridge | lasso
    alpha: float
    l1_ratio: float | None = None   # ignoré hors ElasticNet

@app.post("/train")
def train(req: TrainRequest):
    """Lance un entraînement MLflow en sous‑processus."""
    run_id = str(uuid.uuid4())[:8]
    cmd = [
        "python", "train_model.py",
        "--model", req.model,
        "--alpha", str(req.alpha)
    ]
    if req.model == "elasticnet":
        cmd += ["--l1_ratio", str(req.l1_ratio or 0.5)]
    
    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = env.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    subprocess.Popen(cmd, env=env)
    return {"status": "started", "run_id": run_id, "cmd": " ".join(cmd)}

# --- Nouvelle version de PredictRequest avec toutes les features ---
class PredictRequest(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

@app.post("/predict")
def predict(inp: PredictRequest):
    """Prédiction en utilisant un modèle MLflow enregistré."""
    input_vector = np.array([[
        inp.fixed_acidity,
        inp.volatile_acidity,
        inp.citric_acid,
        inp.residual_sugar,
        inp.chlorides,
        inp.free_sulfur_dioxide,
        inp.total_sulfur_dioxide,
        inp.density,
        inp.pH,
        inp.sulphates,
        inp.alcohol
    ]])

    # Charger le dernier modèle enregistré dans MLflow
    model_uri = "models:/wine_quality_model/latest"  # toujours charger la dernière version dans MLflow
    model = mlflow.sklearn.load_model(model_uri)
    
    prediction = model.predict(input_vector)[0]
    return {"quality_estimate": int(prediction)} # <= classification binaire
