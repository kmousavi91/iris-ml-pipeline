# register_data.py
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

# Load workspace config
ml_client = MLClient.from_config(credential=DefaultAzureCredential())

# Register the dataset
iris_dataset = Data(
    name="iris_dataset",
    version="1",
    path="./data/iris.csv",
    type=AssetTypes.URI_FILE,
    description="Iris dataset registered in Azure ML",
)

ml_client.data.create_or_update(iris_dataset)

print("✅ Iris dataset registered in Azure ML workspace.")

