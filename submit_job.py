from azure.ai.ml import MLClient, Input, command
from azure.identity import DefaultAzureCredential
import os

# Load workspace config
ml_client = MLClient.from_config(DefaultAzureCredential())

# Define the job
job = command(
    code="./scripts",  # path to folder with train.py
    command="python train.py --data_path ${{inputs.iris_data}}",
    inputs={
        "iris_data": Input(
            type="uri_file",
            path="azureml:iris_dataset:1"  # Make sure this matches your dataset name and version
        )
    },
    environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu:1",  # Built-in Azure ML environment
    compute="cpu-cluster",  # Replace with your compute target
    experiment_name="iris-training-exp",
    description="Training job for Iris dataset"
)

# Submit the job
returned_job = ml_client.jobs.create_or_update(job)
print(f"Job submitted. Name: {returned_job.name}")

