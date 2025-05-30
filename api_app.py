from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import subprocess, uuid, os
import pandas as pd

app = FastAPI(title="Red-Wine MLOps API")

class TrainRequest(BaseModel):
    model: str
    alpha: float
    l1_ratio: float | None = None

@app.post("/train")
def train(req: TrainRequest):
    run_id = str(uuid.uuid4())[:8]  # Identifiant local pour la requête
    cmd = [
        "python", "train_model.py",
        "--model", req.model,
        "--alpha", str(req.alpha)
    ]
    if req.model == "elasticnet":
        cmd += ["--l1_ratio", str(req.l1_ratio or 0.5)]

    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = env.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

    # Lance le script en arrière-plan
    try:
        subprocess.Popen(cmd, env=env, cwd=os.path.dirname(__file__))
        print(f"[INFO] Lancement du training avec : {' '.join(cmd)}")
    except Exception as e:
        print(f"[ERREUR] Impossible de lancer le subprocess : {e}")

    print(f"[INFO] Training lancé : {' '.join(cmd)}")
    return {"status": "started", "local_run_id": run_id, "cmd": " ".join(cmd)}

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

MODEL_URI = "models:/wine_quality_model/Production"
model = mlflow.sklearn.load_model(MODEL_URI)

@app.post("/predict")
def predict(inp: PredictRequest):
    data = pd.DataFrame([inp.dict()])
    pred = model.predict(data)[0]
    return {"quality_estimate": round(float(pred), 2)}
