"""
Persistence module
Handles saving and loading models
"""

import joblib


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)