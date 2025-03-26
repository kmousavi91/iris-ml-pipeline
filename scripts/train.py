from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import logging
import joblib
import mlflow
import mlflow.sklearn
from datetime import datetime
import numpy as np
import os


def fit_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def train_model(**kwargs):
    logger = logging.getLogger("airflow.task")
    ti = kwargs["ti"]

    # Pull training data from XCom
    X_train = np.array(ti.xcom_pull(key="X_train"))
    y_train = np.array(ti.xcom_pull(key="y_train"))

    logger.info(f"Training data shapes: X={X_train.shape}, y={y_train.shape}")

    # Train model
    model = fit_model(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    logger.info(f"Training complete. Accuracy: {train_acc:.4f}")

    # Save model to models directory
    model_dir = os.path.abspath(os.path.expanduser("~/iris_ml_project/models"))
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "iris_rf_model.pkl")

    try:
        joblib.dump(model, model_path)
        logger.info(f"Model saved at {model_path}")
        logger.info(f"Model directory contents: {os.listdir(model_dir)}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")

    # Log to MLflow
    mlflow.set_tracking_uri("file:///tmp/mlruns")
    mlflow.set_experiment("IrisClassifier")
    with mlflow.start_run():
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.sklearn.log_model(model, artifact_path="iris_model")
        logger.info("Model logged to MLflow")

    # Push model path to XCom
    ti.xcom_push(key="model_path", value=model_path)
    logger.info(f"Pushed model_path to XCom: {model_path}")

