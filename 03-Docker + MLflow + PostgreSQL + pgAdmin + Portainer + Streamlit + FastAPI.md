# GUIDE – MLOps **Red Wine Quality**

Docker + MLflow + PostgreSQL + pgAdmin + Portainer + **FastAPI** + **Streamlit**

> **Extension par rapport à la version précédente** :
> – Ajout d’une API REST (FastAPI : port 8000)
> – Ajout d’un tableau de bord interactif (Streamlit : port 8501)
> – Possibilité de lancer – depuis le navigateur – des entraînements MLflow avec hyper‑paramètres choisis à la volée, d’observer les runs en temps réel et de servir les prédictions.


<br/>

# 0. TABLE DES PORTS

| Service    | Port conteneur | Port hôte | Description courte         |
| ---------- | -------------- | --------- | -------------------------- |
| MLflow UI  | 5000           | 5000      | Suivi des expériences      |
| FastAPI    | 8000           | 8000      | API REST (train + predict) |
| Streamlit  | 8501           | 8501      | Dashboard interactif       |
| PostgreSQL | 5432           | 5432      | Base de données MLflow     |
| pgAdmin    | 80             | 8080      | GUI PostgreSQL             |
| Portainer  | 9000           | 9000      | Supervision conteneurs     |

Assurez‑vous d’ouvrir **5000, 8000, 8501, 8080, 9000** (TCP entrants) dans votre pare‑feu / NSG.



<br/>

# 1. FICHIERS À CRÉER / MODIFIER

```
mlops-redwine/
├── Dockerfile
├── requirements.txt
├── train_model.py
├── api_app.py           <-- FastAPI
├── streamlit_app.py     <-- Streamlit
├── docker-compose.yml
├── data/
│   └── red-wine-quality.csv
└── mlruns/
```



### 1.1 `requirements.txt` (complété)

```
pandas
numpy
scikit-learn
mlflow
psycopg2-binary
fastapi
uvicorn[standard]
streamlit
```



### 1.2 `Dockerfile` (un seul conteneur « mlflow » sachant tout faire)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y git \
 && pip install --upgrade pip \
 && pip install -r requirements.txt \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Entrée par défaut : shell
CMD ["bash"]
```



### 1.3 `train_model.py`

```python

import argparse, os, sys
import pandas as pd, numpy as np
import mlflow, mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------- 1. CLI ----------
cli = argparse.ArgumentParser()
cli.add_argument("--model", required=True,
                 choices=["elasticnet", "ridge", "lasso", "randomforest", "gbrt"])
cli.add_argument("--alpha", type=float, default=0.5)         # utilisé par linéaires
cli.add_argument("--l1_ratio", type=float, default=0.5)      # utilisé uniquement par ElasticNet
args = cli.parse_args()

# ---------- 2. MLflow ----------
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(tracking_uri)
experiment_name = f"mlops_redwine_{args.model}"
mlflow.set_experiment(experiment_name)

# ---------- 3. Données ----------
csv_path = "data/red-wine-quality.csv"
if not os.path.exists(csv_path):
    sys.exit(f"Fichier introuvable : {csv_path}")
df = pd.read_csv(csv_path, sep=';')  # séparateur correct

X = df.drop("quality", axis=1)
y = df["quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

def log_metrics(y_true, y_pred):
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae":  mean_absolute_error(y_true, y_pred),
        "r2":   r2_score(y_true, y_pred)
    }

# ---------- 4. Entraînement ----------
with mlflow.start_run() as run:
    # Sélection du modèle
    if args.model == "elasticnet":
        model = ElasticNet(alpha=args.alpha, l1_ratio=args.l1_ratio, random_state=42)
        mlflow.log_param("l1_ratio", args.l1_ratio)
        mlflow.log_param("alpha", args.alpha)

    elif args.model == "ridge":
        model = Ridge(alpha=args.alpha, random_state=42)
        mlflow.log_param("alpha", args.alpha)

    elif args.model == "lasso":
        model = Lasso(alpha=args.alpha, random_state=42)
        mlflow.log_param("alpha", args.alpha)

    elif args.model == "randomforest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    elif args.model == "gbrt":
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

    else:
        sys.exit("Modèle non pris en charge.")

    # Apprentissage
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Log des métriques
    for k, v in log_metrics(y_test, preds).items():
        mlflow.log_metric(k, float(v))

    # Création de la signature
    input_example = X_train.iloc[:1]
    signature = infer_signature(X_train, model.predict(X_train))

    # Log du modèle avec signature et exemple
    mlflow.sklearn.log_model(model, "model", signature=signature, input_example=input_example)

    # Enregistrement dans le registry
    model_name = "RedWineModel"
    model_uri = f"runs:/{run.info.run_id}/model"
    mlflow.register_model(model_uri, model_name)

    # Passage à Production
    client = mlflow.tracking.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage="Production",
        archive_existing_versions=True
    )

    print(f"Modèle '{args.model}' enregistré et promu en Production.")

---

### 1.4 `api_app.py` — **FastAPI**

```python
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

```



### 1.5 `streamlit_app.py` — **Dashboard**

```python
import streamlit as st
import requests
import pandas as pd
import os
from urllib.parse import urljoin

st.set_page_config(page_title="Red Wine Quality Trainer", layout="centered")
st.title("🍷 Red Wine Quality MLOps App")

# ---------- Configuration de l'URL de l'API ----------
API_BASE_URL = os.getenv("API_URL", "http://backend:8000")  # Utilise la variable d'environnement ou la valeur par défaut
TRAIN_ENDPOINT = urljoin(API_BASE_URL, "/train")
PREDICT_ENDPOINT = urljoin(API_BASE_URL, "/predict")

# ---------- 1. Paramètres d'entraînement ----------
with st.form("train_form"):
    st.header("🔧 Entraînement du modèle")
    model_type = st.selectbox("Choisissez un modèle", ["elasticnet", "ridge", "lasso", "randomforest", "gbrt"])
    alpha = st.number_input("Alpha", value=0.5, step=0.01)
    l1_ratio = st.number_input("L1 Ratio (ElasticNet uniquement)", value=0.5, step=0.01)
    train_btn = st.form_submit_button("Lancer l'entraînement")

    if train_btn:
        payload = {"model": model_type}
        if model_type in ["elasticnet", "ridge", "lasso"]:
            payload["alpha"] = alpha
        if model_type == "elasticnet":
            payload["l1_ratio"] = l1_ratio

        with st.spinner("Entraînement en cours..."):
            try:
                r = requests.post(TRAIN_ENDPOINT, json=payload, timeout=30)
                r.raise_for_status()
                resp_json = r.json()
                st.success(f"✅ Entraînement terminé: Run ID {resp_json.get('run_id', '')}")

                if "metrics" in resp_json:
                    st.subheader("📊 Résultats d'entraînement")
                    metrics = resp_json["metrics"]
                    cols = st.columns(3)
                    cols[0].metric("RMSE", round(metrics.get("rmse", 0), 4))
                    cols[1].metric("MAE", round(metrics.get("mae", 0), 4))
                    cols[2].metric("R²", round(metrics.get("r2", 0), 4))

            except requests.exceptions.RequestException as e:
                st.error(f"❌ Erreur de connexion à l'API: {str(e)}")
            except ValueError as e:
                st.error(f"❌ Réponse JSON invalide: {str(e)}")

# ---------- 2. Prédiction ----------
st.header("🔮 Prédiction de la qualité")

input_fields = {
    "fixed_acidity": ("Fixed acidity", 5.0),
    "volatile_acidity": ("Volatile acidity", 0.5),
    "citric_acid": ("Citric acid", 0.5),
    "residual_sugar": ("Residual sugar", 2.0),
    "chlorides": ("Chlorides", 0.1),
    "free_sulfur_dioxide": ("Free sulfur dioxide", 15.0),
    "total_sulfur_dioxide": ("Total sulfur dioxide", 30.0),
    "density": ("Density", 0.995),
    "pH": ("pH", 3.5),
    "sulphates": ("Sulphates", 0.5),
    "alcohol": ("Alcohol", 10.0)
}

inputs = {}
for field, (label, default) in input_fields.items():
    inputs[field] = st.number_input(label, value=default, step=0.01)

if st.button("Prédire la qualité"):
    with st.spinner("Calcul de la prédiction..."):
        try:
            response = requests.post(PREDICT_ENDPOINT, json=inputs, timeout=10)
            response.raise_for_status()

            result = response.json()
            if "quality_estimate" in result:
                st.success(f"🎯 Qualité estimée: {result['quality_estimate']:.1f}/10")
            else:
                st.error(f"⚠️ Format de réponse inattendu: {result}")

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Erreur de connexion: {str(e)}")
        except ValueError as e:
            st.error(f"❌ Réponse JSON invalide: {str(e)}")

```



### 1.6 `docker-compose.yml` (mis à jour)

```yaml
services:
  # ---------- PostgreSQL ----------
  postgres:
    image: postgres:14
    restart: unless-stopped
    environment:
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: mlflow
      POSTGRES_DB: mlflow_db
    ports: ["5432:5432"]
    volumes: [postgres_data:/var/lib/postgresql/data]
    networks:  # Ajouté
      - mlflow_network

  # ---------- Backend MLflow + code commun ----------
  mlflow:
    build: .
    restart: unless-stopped
    depends_on: [postgres]
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
    command: >
      mlflow server
      --backend-store-uri postgresql://mlflow:mlflow@postgres:5432/mlflow_db
      --default-artifact-root /app/mlruns
      --host 0.0.0.0
    ports: ["5000:5000"]
    volumes:
      - ./mlruns:/app/mlruns
      - ./data:/app/data
    networks:  # Ajouté
      - mlflow_network

  # ---------- FastAPI ----------
  api:
    build: .
    restart: unless-stopped
    depends_on: [mlflow]
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
    command: uvicorn api_app:app --host 0.0.0.0 --port 8000 --reload
    ports: ["8000:8000"]
    volumes:
      - ./data:/app/data
      - ./mlruns:/app/mlruns
    networks:  # Ajouté
      - mlflow_network

  # ---------- Streamlit ----------
  streamlit:
    build: .
    restart: unless-stopped
    depends_on: [api]
    environment:
      STREAMLIT_SERVER_PORT: 8501
      MLFLOW_TRACKING_URI: http://mlflow:5000
      API_URL: http://api:8000  # Utilise le nom du service "api"
    command: streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
    ports: ["8501:8501"]
    volumes:
      - ./data:/app/data
      - ./mlruns:/app/mlruns
    networks:  # Ajouté
      - mlflow_network

  # ---------- pgAdmin ----------
  pgadmin:
    image: dpage/pgadmin4
    restart: unless-stopped
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@admin.com
      PGADMIN_DEFAULT_PASSWORD: admin
    ports: ["8080:80"]
    volumes: [pgadmin_data:/var/lib/pgadmin]
    networks:  # Ajouté (optionnel)
      - mlflow_network

  # ---------- Portainer ----------
  portainer:
    image: portainer/portainer-ce
    restart: unless-stopped
    command: -H unix:///var/run/docker.sock
    ports: ["9000:9000"]
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - portainer_data:/data
    networks:  # Ajouté (optionnel)
      - mlflow_network

volumes:
  postgres_data:
  pgadmin_data:
  portainer_data:

networks:  # Nouveau réseau ajouté
  mlflow_network:
    driver: bridge
```

<br/>

# 2. BUILD & LANCEMENT

```bash
sudo -s
git clone https://github.com/hrhouma/install-docker.git
cd install-docker
chmod +x install-docker.sh
./install-docker.sh           # installe Docker + compose plugin v2
apt install docker-compose
cd /home/azureuser/
git clone https://github.com/alnyeck/projetsmlops-3.git
cd projetsmlops-3/
docker-compose pull
docker-compose build --no-cache mlflow
docker-compose up -d
docker ps --format "table {{.Names}}\t{{.Ports}}"
```

Vous devez voir **6 services** : `postgres`, `mlflow`, `api`, `streamlit`, `pgadmin`, `portainer`.


<br/>

# 3. UTILISATION

| Étape | Action                                                                                                                                                                                                                                              |
| ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | Visitez **Streamlit** : `http://<IP_VM>:8501`.<br>Choisissez un modèle, réglez `alpha` et (pour ElasticNet) `l1_ratio`, cliquez **Lancer l'entraînement**.<br>Le run démarre et apparaît instantanément dans **MLflow UI** (`http://<IP_VM>:5000`). |
| 2     | Testez l’API REST :                                                                                                                                                                                                                                 |

*Depuis votre machine locale* :

```bash
curl -X POST http://<IP_VM>:8000/train \
  -H "Content-Type: application/json" \
  -d '{"model":"ridge","alpha":0.4}'
```

*Prédiction rapide* :

```bash
curl -X POST http://<IP_VM>:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"alcohol":11.0,"volatile_acidity":0.6,"sulphates":0.75}'
```

Vous obtenez un JSON avec l’estimation de qualité.

<br/>

# 4. POINTS D’AMÉLIORATION

| Idée                         | Piste de mise en œuvre                                                                            |
| ---------------------------- | ------------------------------------------------------------------------------------------------- |
| Signature MLflow             | Utiliser `mlflow.models.signature.infer_signature` pour logger `input_example` et la `signature`. |
| Prediction via modèle MLflow | Charger le dernier modèle avec `mlflow.pyfunc.load_model` dans `api_app.py`.                      |
| Authentification API         | Ajouter `fastapi.security` + JWT ou clé API simple.                                               |
| Monitoring                   | Activer Prometheus + Grafana via Portainer pour CPU/RAM.                                          |
| Tests unitaires              | Ajouter `pytest` + GitHub Actions CI/CD (build + docker push).                                    |

<br/>

#  5. Notre stack MLOps est prête !

- Page Streamlit conviviale, API REST pour scripts externes, suivi MLflow complet, base PostgreSQL, pgAdmin, Portainer — le tout en **un seul `docker-compose up -d`**.
