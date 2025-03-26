# scripts/preprocess.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import logging
import os

def preprocess_data(**kwargs):
    

    logger = logging.getLogger("airflow.task")
    logger.info("Loading and splitting Iris dataset...")

    iris = load_iris()
    X = iris.data
    y = iris.target

    from airflow.models import Variable  # optional if needed

    from datetime import datetime
    from airflow.models import XCom

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    ti = kwargs['ti']
    ti.xcom_push(key='X_train', value=X_train.tolist())
    ti.xcom_push(key='X_test', value=X_test.tolist())
    ti.xcom_push(key='y_train', value=y_train.tolist())
    ti.xcom_push(key='y_test', value=y_test.tolist())

    logger.info("Data preprocessing completed.")

