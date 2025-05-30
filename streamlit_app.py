import streamlit as st
import requests, json, pandas as pd
import time

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
        url = "http://api:8000/train"
        headers = {"Content-Type": "application/json"}
        max_attempts = 5
        attempt = 0
        resp = None

        while attempt < max_attempts:
            try:
                resp = requests.post(url, data=json.dumps(payload), headers=headers)
                if resp.ok:
                    st.success(f"✅ Run démarré : {resp.json()['run_id']}")
                    st.json(resp.json())
                    break
                else:
                    st.error("❌ L’API a répondu mais avec une erreur.")
                    st.json(resp.json())
                    break
            except requests.exceptions.ConnectionError:
                attempt += 1
                st.warning(f"⏳ Tentative {attempt}/{max_attempts} : en attente de l’API…")
                time.sleep(2)
        else:
            st.error("❌ Impossible de contacter l’API FastAPI après plusieurs tentatives.")


st.header("Prévision rapide (démonstration)")
inputs = {}
columns = {
    "fixed_acidity": (4.0, 16.0, 8.0, 0.1),
    "volatile_acidity": (0.1, 1.5, 0.5, 0.01),
    "citric_acid": (0.0, 1.0, 0.4, 0.01),
    "residual_sugar": (0.9, 15.0, 2.5, 0.1),
    "chlorides": (0.01, 0.2, 0.05, 0.005),
    "free_sulfur_dioxide": (1, 72, 15, 1),
    "total_sulfur_dioxide": (6, 289, 46, 1),
    "density": (0.9900, 1.0050, 0.996, 0.0001),
    "pH": (2.8, 4.0, 3.3, 0.01),
    "sulphates": (0.3, 2.0, 0.65, 0.01),
    "alcohol": (8.0, 15.0, 10.0, 0.1),
}

cols = st.columns(3)
for i, (name, (min_val, max_val, default, step)) in enumerate(columns.items()):
    with cols[i % 3]:
        inputs[name] = st.number_input(name.replace("_", " ").title(), min_val, max_val, default, step)

if st.button("🔮 Prédire la qualité"):
    r = requests.post("http://api:8000/predict",
                      data=json.dumps(inputs),
                      headers={"Content-Type": "application/json"})
    st.write(r.json())

