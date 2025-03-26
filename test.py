# test.py
from scripts.train import train_model
import numpy as np

def test_model_shape():
    X_train = np.random.rand(10, 4)
    y_train = np.random.randint(0, 3, 10)
    model = train_model(X_train, y_train)
    assert hasattr(model, "predict"), "Model should have predict method"
