"""
Main orchestration file for Mobile Addiction Risk Predictor
Runs complete ML workflow:
Load Data -> Encode -> Split -> Train -> Evaluate -> Save Model
"""

from src.config import *
from src.data_loader import load_data
from src.data_preprocessing import split_data
from src.feature_engineering import encode_target
from src.train import train_model
from src.evaluate import evaluate_model
from src.persistence import save_model


def main():
    """Execute full machine learning pipeline."""

    # Load dataset
    df = load_data(DATA_PATH)

    # Encode target column
    df[TARGET_COLUMN], encoder = encode_target(df[TARGET_COLUMN])

    # Split dataset
    X_train, X_test, y_train, y_test = split_data(
        df,
        TARGET_COLUMN,
        TEST_SIZE,
        RANDOM_STATE
    )

    # Train model
    model = train_model(X_train, y_train, RANDOM_STATE)

    # Evaluate model
    result = evaluate_model(model, X_test, y_test)

    # Print evaluation results
    print("Accuracy:", result["accuracy"])
    print(result["report"])

    # Save evaluation report
    with open("reports/metrics.txt", "w") as file:
        file.write("Model Evaluation Report\n")
        file.write("========================\n")
        file.write(f"Accuracy: {result['accuracy']}\n\n")
        file.write(result["report"])

    # Save trained model using persistence module
    save_model(model, MODEL_PATH)

    # Create experiment log
    with open("logs/experiment_log.txt", "w") as log:
        log.write("Experiment Log\n")
        log.write("====================\n")
        log.write("Model: Random Forest Classifier\n")
        log.write(f"Test Size: {TEST_SIZE}\n")
        log.write(f"Random State: {RANDOM_STATE}\n")
        log.write(f"Accuracy: {result['accuracy']}\n")

    print("✅ Model Saved Successfully!")
    print("📄 Metrics Saved in reports/metrics.txt")
    print("📝 Experiment Log Saved in logs/experiment_log.txt")


if __name__ == "__main__":
    main()