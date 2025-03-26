# scripts/predict.py

import logging
import numpy as np
import os
import joblib
from datetime import datetime
from airflow.models import Variable

def predict_model(**kwargs):
    logger = logging.getLogger("airflow.task")
    ti = kwargs["ti"]

    # Pull data and model path from XCom
    X_test = np.array(ti.xcom_pull(key="X_test"))
    model_path = ti.xcom_pull(key="model_path")

    # Load model
    model = joblib.load(model_path)

    # Predict
    y_pred = model.predict(X_test)

    # Save predictions
    output_dir = "/tmp/iris_predictions"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "predictions.csv")

    with open(output_file, "w") as f:
        f.write("prediction\n")
        for p in y_pred:
            f.write(f"{p}\n")

    logger.info(f"Predictions saved to: {output_file}")

