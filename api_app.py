from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import subprocess, uuid, os
import mlflow.pyfunc
import pandas as pd

app = FastAPI(title="Red‑Wine MLOps API")

# ----------- TRAIN REQUEST -----------
class TrainRequest(BaseModel):
    model: str  # elasticnet | ridge | lasso | randomforest | gbrt
    alpha: Optional[float] = None
    l1_ratio: Optional[float] = None

@app.post("/train")
def train(req: TrainRequest):
    run_id = str(uuid.uuid4())[:8]
    cmd = [
        "python", "train_model.py",
        "--model", req.model
    ]
    if req.alpha is not None:
        cmd += ["--alpha", str(req.alpha)]
    if req.model == "elasticnet" and req.l1_ratio is not None:
        cmd += ["--l1_ratio", str(req.l1_ratio)]
    
    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = env.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    subprocess.Popen(cmd, env=env)
    
    return {"status": "started", "run_id": run_id, "cmd": " ".join(cmd)}

# ----------- PREDICT REQUEST -----------
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
    try:
        # Dictionnaire reçu depuis le frontend
        input_dict = inp.dict()

        # Mapping des noms reçus (underscores) vers les noms attendus par le modèle (espaces)
        rename_map = {
            "fixed_acidity": "fixed acidity",
            "volatile_acidity": "volatile acidity",
            "citric_acid": "citric acid",
            "residual_sugar": "residual sugar",
            "chlorides": "chlorides",
            "free_sulfur_dioxide": "free sulfur dioxide",
            "total_sulfur_dioxide": "total sulfur dioxide",
            "density": "density",
            "pH": "pH",
            "sulphates": "sulphates",
            "alcohol": "alcohol"
        }

        # Conversion du dictionnaire vers DataFrame avec les bons noms de colonnes
        renamed_input = {rename_map[k]: v for k, v in input_dict.items()}
        X = pd.DataFrame([renamed_input])

        # Chargement du modèle promu en production
        model_uri = "models:/RedWineModel/Production"
        model = mlflow.pyfunc.load_model(model_uri)

        # Prédiction
        pred = model.predict(X)[0]
        return {"quality_estimate": round(float(pred), 2)}

    except Exception as e:
        return {"error": str(e)}

