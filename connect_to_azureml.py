from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import os

def get_ml_client():
    credential = DefaultAzureCredential()
    ml_client = MLClient.from_config(credential=credential)
    return ml_client
