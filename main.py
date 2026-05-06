"""
Main orchestration file for Mobile Addiction Risk Predictor
Runs complete ML workflow:
Load Data -> Encode -> Split -> Train -> Evaluate -> Save Model
"""

from src.config import *
from src.problem_type import identify_problem_type
from src.data_loader import load_data
from src.data_preprocessing import split_data
from src.feature_engineering import encode_target
from src.train import train_model
from src.evaluate import evaluate_model
from src.persistence import save_model


def main():
    """Execute full machine learning pipeline."""

    # Identify ML problem type
    problem_info = identify_problem_type()

    print("========== ML PROJECT INFO ==========")
    print("Learning Type:", problem_info["learning_type"])
    print("Problem Type:", problem_info["problem_type"])
    print("Target Variable:", problem_info["target_variable"])
    print("=====================================")

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
    print("\n========== MODEL EVALUATION ==========")
    print("Accuracy:", result["accuracy"])
    print(result["report"])

    # Save evaluation report
    with open("reports/metrics.txt", "w") as file:
        file.write("Model Evaluation Report\n")
        file.write("========================\n")
        file.write(
            f"Learning Type: {problem_info['learning_type']}\n"
        )
        file.write(
            f"Problem Type: {problem_info['problem_type']}\n"
        )
        file.write(
            f"Target Variable: {problem_info['target_variable']}\n\n"
        )
        file.write(f"Accuracy: {result['accuracy']}\n\n")
        file.write(result["report"])

    # Save trained model
    save_model(model, MODEL_PATH)

    # Create experiment log
    with open("logs/experiment_log.txt", "w") as log:
        log.write("Experiment Log\n")
        log.write("====================\n")
        log.write(
            f"Learning Type: {problem_info['learning_type']}\n"
        )
        log.write(
            f"Problem Type: {problem_info['problem_type']}\n"
        )
        log.write("Model: Random Forest Classifier\n")
        log.write(f"Test Size: {TEST_SIZE}\n")
        log.write(f"Random State: {RANDOM_STATE}\n")
        log.write(f"Accuracy: {result['accuracy']}\n")

    print("\n✅ Model Saved Successfully!")
    print("📄 Metrics Saved in reports/metrics.txt")
    print("📝 Experiment Log Saved in logs/experiment_log.txt")


if __name__ == "__main__":
    main()