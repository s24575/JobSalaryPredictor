import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load the modelq
model = load_model("saved/model.keras")

# Load the scalers
scaler_X = joblib.load("saved/scaler_X.joblib")
scaler_y = joblib.load("saved/scaler_y.joblib")

# Load new test data
custom_test_data = pd.read_csv("data/custom_input.csv")

# Convert categorical variables into one-hot encoding
custom_test_data = pd.get_dummies(custom_test_data, columns=['location', 'technology', 'seniority'])

columns_to_scale = ["company size", "salary employment min", "salary employment max"]
# Normalize numerical features using the same scaler from training
X_scaled = scaler_X.transform(custom_test_data[columns_to_scale])

# Make predictions on the new test data
predictions_before_scaling_custom = model.predict(X_scaled)
predictions_custom = scaler_y.inverse_transform(predictions_before_scaling_custom)

# Print or use predictions_custom as needed
print(predictions_custom)
