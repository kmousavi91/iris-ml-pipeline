
import joblib, os
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
os.makedirs("/home/kourosh/iris_ml_project/models", exist_ok=True)
joblib.dump(model, "/home/kourosh/iris_ml_project/models/test_model.pkl")
os.path.exists("/home/kourosh/iris_ml_project/models/test_model.pkl")
True
