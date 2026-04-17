import joblib

def load_model(path):
    return joblib.load(path)

def make_prediction(model, data):
    return model.predict(data)