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
