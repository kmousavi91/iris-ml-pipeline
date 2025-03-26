from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from scripts.preprocess import preprocess_data
from scripts.train import train_model
from scripts.evaluate import evaluate_model
from scripts.predict import predict_model

default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 3, 25),
    "retries": 1,
}

with DAG(
    dag_id="iris_ml_pipeline",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
    tags=["ml"],
) as dag:

    preprocess_task = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
    )

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    evaluate_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    predict_task = PythonOperator(
        task_id="predict_model",
        python_callable=predict_model,
    )

    preprocess_task >> train_task >> evaluate_task >> predict_task

dag = dag

