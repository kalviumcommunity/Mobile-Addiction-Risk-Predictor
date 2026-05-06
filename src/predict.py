"""
Prediction module
Responsible for inference only
"""

from src.persistence import load_model


def predict_risk(data, model_path):
    """
    Predict addiction risk.

    Parameters:
        data : Input data
        model_path : Saved model path

    Returns:
        Prediction
    """

    model = load_model(model_path)

    prediction = model.predict(data)

    return prediction