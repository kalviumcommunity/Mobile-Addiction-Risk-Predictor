"""
Training module
Responsible only for training ML model
"""

from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train, random_state):
    """
    Train Random Forest model.

    Parameters:
        X_train : Training features
        y_train : Training labels
        random_state : Seed value

    Returns:
        Trained model
    """

    model = RandomForestClassifier(
        random_state=random_state
    )

    model.fit(X_train, y_train)

    return model