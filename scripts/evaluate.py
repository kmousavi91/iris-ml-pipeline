from sklearn.metrics import accuracy_score
import logging
from datetime import datetime
import os
import mlflow
import joblib
import numpy as np

def evaluate_model(**kwargs):
    logger = logging.getLogger("airflow.task")
    ti = kwargs["ti"]

    # Pull data from XCom
    X_test = np.array(ti.xcom_pull(key="X_test"))
    y_test = np.array(ti.xcom_pull(key="y_test"))
    model_path = ti.xcom_pull(key="model_path")

    # Safety check
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at: {model_path}")
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    # Load model
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    logger.info(f"Evaluation complete. Accuracy = {accuracy:.4f}")

    # Save metrics to CSV
    log_dir = "/tmp/iris_logs"
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "metrics.csv"), "a") as f:
        f.write(f"{datetime.now()},accuracy,{accuracy}\n")

    logger.info("Metrics saved to /tmp/iris_logs/metrics.csv")

    # Log to MLflow
    mlflow.set_tracking_uri("file:///tmp/mlruns")
    mlflow.set_experiment("IrisClassifier")

    with mlflow.start_run():
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_metric("accuracy", accuracy)
        logger.info("Metrics logged to MLflow")

