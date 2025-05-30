import argparse
import os
import sys
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def log_metrics(y_true, y_pred):
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred)
    }

def main():
    # --------- 1. CLI ----------
    cli = argparse.ArgumentParser(description="Entraînement modèle MLflow avec ElasticNet, Ridge ou Lasso")
    cli.add_argument("--model", required=True, choices=["elasticnet", "ridge", "lasso"])
    cli.add_argument("--alpha", type=float, default=0.5)
    cli.add_argument("--l1_ratio", type=float, default=0.5)
    args = cli.parse_args()

    # --------- 2. MLflow setup ----------
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(f"mlops_redwine_{args.model}")

    # --------- 3. Chargement données ----------
    csv_path = "data/red-wine-quality.csv"
    if not os.path.exists(csv_path):
        sys.exit(f"Fichier introuvable : {csv_path}")

    df = pd.read_csv(csv_path, sep=';')
    X = df.drop("quality", axis=1)
    y = df["quality"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # --------- 4. Entraînement et logging ----------
    with mlflow.start_run() as run:
        # Choix du modèle
        if args.model == "elasticnet":
            model = ElasticNet(alpha=args.alpha, l1_ratio=args.l1_ratio, random_state=42)
            mlflow.log_param("l1_ratio", args.l1_ratio)
        elif args.model == "ridge":
            model = Ridge(alpha=args.alpha, random_state=42)
        else:
            model = Lasso(alpha=args.alpha, random_state=42)

        mlflow.log_param("model_type", args.model)
        mlflow.log_param("alpha", args.alpha)

        # Entraînement
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Logging des métriques
        metrics = log_metrics(y_test, preds)
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))

        # Logging du modèle
        mlflow.sklearn.log_model(model, "model")

        run_id = run.info.run_id

    print(f"✅ Run MLflow terminé : {run_id}")
    print(f"Metrics : {metrics}")

    # --------- 5. Enregistrement dans le Model Registry ----------
    client = MlflowClient(tracking_uri=tracking_uri)

    # URI du modèle pour le registry
    model_uri = f"runs:/{run_id}/model"
    try:
        registered_model = client.create_registered_model("wine_quality_model")
        print("Modèle enregistré pour la première fois.")
    except Exception as e:
        # Le modèle existe probablement déjà
        print(f"Modèle déjà enregistré : {e}")

    result = client.create_model_version(
        name="wine_quality_model",
        source=model_uri,
        run_id=run_id
    )
    print(f"Version de modèle créée : {result.version}")

    # Promotion automatique en production
    client.transition_model_version_stage(
        name="wine_quality_model",
        version=result.version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"Modèle version {result.version} promu en Production.")

if __name__ == "__main__":
    main()
