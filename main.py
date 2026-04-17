import joblib

from src.config import *
from src.data_preprocessing import load_data, split_data
from src.feature_engineering import encode_target
from src.train import train_model
from src.evaluate import evaluate_model

# Load data
df = load_data(DATA_PATH)

# Encode full target first
df[TARGET_COLUMN], encoder = encode_target(df[TARGET_COLUMN])

# Split data
X_train, X_test, y_train, y_test = split_data(
    df,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE
)

# Train model
model = train_model(X_train, y_train, RANDOM_STATE)

# Evaluate
result = evaluate_model(model, X_test, y_test)

print("Accuracy:", result["accuracy"])
print(result["report"])

# Save model
joblib.dump(model, MODEL_PATH)

print("✅ Model Saved Successfully!")