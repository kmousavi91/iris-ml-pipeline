import pandas as pd
from sklearn.datasets import load_iris
import os

# Load the Iris dataset
iris = load_iris(as_frame=True)
df = iris.frame

# Make sure the data directory exists
os.makedirs("data", exist_ok=True)

# Save to CSV
csv_path = os.path.join("data", "iris.csv")
df.to_csv(csv_path, index=False)

print(f"✅ Iris dataset saved to: {csv_path}")

