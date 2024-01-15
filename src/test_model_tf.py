import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Load your data
data = pd.read_csv('data/202401_soft_eng_jobs_pol.csv')

# Data Preprocessing
data['average_salary'] = (data['salary employment min'] + data['salary employment max']) / 2
data = data.drop(['salary employment min', 'salary employment max', 'salary b2b min', 'salary b2b max'], axis=1)
data = data.dropna()

# Feature Encoding
categorical_features = ['location', 'technology', 'seniority']
numeric_features = ['company size']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Train-Test Split
X = data.drop(['average_salary'], axis=1)
y = data['average_salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection and Training in TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer="adam", loss='mean_squared_error')

# Train the model
model.fit(preprocessor.transform(X_train), y_train, epochs=50, batch_size=32, validation_data=(preprocessor.transform(X_test), y_test))

# Model Evaluation
y_pred = model.predict(preprocessor.transform(X_test))
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Prediction example
new_data = pd.DataFrame({
    'company size': [100],
    'location': ['Gdańsk'],
    'technology': ['C/C++'],
    'seniority': ['mid']
})

# Feature Encoding for the new data
new_data_transformed = preprocessor.transform(new_data)

# Make predictions
predicted_salary = model.predict(new_data_transformed)

print(f'Predicted Salary: {predicted_salary[0][0]}')
