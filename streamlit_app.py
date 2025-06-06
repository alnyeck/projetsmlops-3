import streamlit as st
import requests, json, pandas as pd

st.set_page_config(page_title="Red‑Wine Trainer", layout="wide")

st.title("🍷 Red‑Wine Trainer – Interface Streamlit")
st.markdown(
    "Lancez des entraînements MLflow et observez‑les en temps réel dans "
    "[MLflow UI](http://localhost:5000).")

with st.form("train_form"):
    model = st.selectbox("Modèle", ["elasticnet", "ridge", "lasso"])
    alpha = st.slider("alpha", 0.01, 2.0, 0.5, 0.01)
    l1_ratio = st.slider("l1_ratio (ElasticNet)", 0.0, 1.0, 0.5, 0.05)
    submitted = st.form_submit_button("🚀 Lancer l'entraînement")
    if submitted:
        payload = {"model": model, "alpha": alpha, "l1_ratio": l1_ratio}
        resp = requests.post("http://api:8000/train",
                             data=json.dumps(payload),
                             headers={"Content-Type": "application/json"})
        if resp.ok:
            st.success(f"Run démarré : {resp.json()['run_id']}")
            # Afficher le JSON complet de la réponse
            st.json(resp.json())  # <-- Streamlit affichera le JSON formaté

        else:
            st.error("Erreur lors de l’appel API")

st.header("🍷 Prévision de la qualité du vin")

# Créer les champs pour toutes les features du dataset
col1, col2, col3 = st.columns(3)
with col1:
    fixed_acidity = st.number_input("Fixed acidity", 4.0, 16.0, 7.0, 0.1)
    volatile_acidity = st.number_input("Volatile acidity", 0.1, 1.5, 0.5, 0.01)
    citric_acid = st.number_input("Citric acid", 0.0, 1.0, 0.3, 0.05)
    residual_sugar = st.number_input("Residual sugar", 0.9, 15.0, 2.5, 0.1)
with col2:
    chlorides = st.number_input("Chlorides", 0.01, 0.2, 0.045, 0.005)
    free_sulfur_dioxide = st.number_input("Free sulfur dioxide", 1, 70, 15, 1)
    total_sulfur_dioxide = st.number_input("Total sulfur dioxide", 6, 200, 45, 1)
    density = st.number_input("Density", 0.9900, 1.0050, 0.9968, 0.001)
with col3:
    pH = st.number_input("pH", 2.8, 4.0, 3.2, 0.1)
    sulphates = st.number_input("Sulphates", 0.3, 2.0, 0.6, 0.05)
    alcohol = st.number_input("Alcohol", 8.0, 15.0, 10.0, 0.1)

# Prédiction
if st.button("🔮 Prédire la qualité"):
    payload = {
        "fixed_acidity": fixed_acidity,
        "volatile_acidity": volatile_acidity,
        "citric_acid": citric_acid,
        "residual_sugar": residual_sugar,
        "chlorides": chlorides,
        "free_sulfur_dioxide": free_sulfur_dioxide,
        "total_sulfur_dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol
    }

    r = requests.post("http://api:8000/predict",
                      data=json.dumps(payload),
                      headers={"Content-Type": "application/json"})
    
    if r.status_code == 200:
        st.success(f"Qualité prédite du vin : {r.json()['quality_estimate']}")
    else:
        st.error("Erreur lors de la prédiction.")
