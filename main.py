# main.py
from scripts import preprocess, train, evaluate

X_train, X_test, y_train, y_test = preprocess.load_and_split_data()
model = train.train_model(X_train, y_train)
evaluate.evaluate_model(model, X_test, y_test)
train.save_model(model)

