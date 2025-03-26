import numpy as np
from scripts.train import fit_model

def test_model_shape():
    X_train = np.random.rand(10, 4)
    y_train = np.random.randint(0, 3, 10)
    
    model = fit_model(X_train, y_train)
    
    assert hasattr(model, "predict")
    assert model.n_classes_ == 3

