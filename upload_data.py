from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from pathlib import Path

# Define project workspace info
subscription_id = "5d5c3261-a719-4a19-8fe1-b6b0cfdf7f9f"
resource_group = "ml-resource-group"
workspace = "iris-ml-workspace"

# Authenticate
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace
)

# Create a data asset from iris.csv
data_path = Path("data/iris.csv")

iris_data = Data(
    path=data_path,
    type="uri_file",
    description="Iris dataset",
    name="iris_csv",
    version="1",
)

ml_client.data.create_or_update(iris_data)
print("✅ Dataset uploaded successfully to Azure ML!")

