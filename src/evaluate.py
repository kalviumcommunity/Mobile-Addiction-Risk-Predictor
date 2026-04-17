from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model, X_test, y_test):
    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)

    return {
        "accuracy": accuracy,
        "report": classification_report(y_test, pred)
    }