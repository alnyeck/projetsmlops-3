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

