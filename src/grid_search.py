import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load data
data = pd.read_csv('data/202401_soft_eng_jobs_pol.csv')

# Data Preprocessing
data['average_salary'] = (data['salary employment min'] + data['salary employment max']) / 2
data = data.drop(['id', 'company', 'salary employment min', 'salary employment max', 'salary b2b min', 'salary b2b max'], axis=1)
data = data.dropna()

# Feature Encoding
categorical_features = ['company size', 'location', 'technology', 'seniority']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Train-Test Split
X = data.drop(['average_salary'], axis=1)
y = data['average_salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection and Training with Hyperparameter Tuning
param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5, 10],
}

model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor(random_state=42))])

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Cross-Validation
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_rmse_scores = -cv_scores  # Convert scores back to positive values

print(f'Best Hyperparameters: {best_params}')
print(f'Cross-Validation RMSE Scores: {cv_rmse_scores}')
print(f'Mean CV RMSE: {cv_rmse_scores.mean()}')

# Model Evaluation
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = math.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'RMSE: {rmse}')
print(f'R-squared: {r2}')

# Residual Analysis
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=residuals)
plt.title('Residual Analysis')
plt.xlabel('Actual Salary')
plt.ylabel('Residuals')
coefficients = np.polyfit(y_test, residuals, 1)
trendline = np.polyval(coefficients, y_test)
plt.plot(y_test, trendline, color='red', linestyle='dashed', linewidth=2)
plt.show()

plt.scatter(y_test, y_pred)
plt.xlabel('Actual values (y_test)')
plt.ylabel('Predicted values (y_pred)')
plt.title('Actual vs Predicted values')
coefficients = np.polyfit(y_test, y_pred, 1)
trendline = np.polyval(coefficients, y_test)
plt.plot(y_test, trendline, color='red', linestyle='dashed', linewidth=2)
plt.show()

# Additional Metrics
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')

# Visualization: Distribution of Average Salaries
plt.figure(figsize=(10, 6))
sns.histplot(data['average_salary'], bins=30, kde=True)
plt.title('Distribution of Average Salaries')
plt.xlabel('Average Salary')
plt.ylabel('Frequency')
plt.show()

# Feature Importance Plot
feature_names = list(best_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features))
feature_importances = best_model.named_steps['regressor'].feature_importances_

plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importances, y=feature_names)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# Visualization: Residual Distribution
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=30, kde=True)
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# Prediction example
new_data = pd.DataFrame({
    'company size': [10000],
    'location': ['Warszawa'],
    'technology': ['JavaScript'],
    'seniority': ['expert']
})

predicted_salary = best_model.predict(new_data)
print(f'Predicted Salary: {predicted_salary}')
