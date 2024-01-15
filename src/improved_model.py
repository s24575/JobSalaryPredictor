import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
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

# Model Selection and Training
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor(random_state=42))])

model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')

# Visualization: Distribution of Average Salaries
plt.figure(figsize=(10, 6))
sns.histplot(data['average_salary'], bins=30, kde=True)
plt.title('Distribution of Average Salaries')
plt.xlabel('Average Salary')
plt.ylabel('Frequency')
plt.show()

# Feature Importance Plot
feature_names = list(model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features))
feature_importances = model.named_steps['regressor'].feature_importances_

plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importances, y=feature_names)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# Prediction example
new_data = pd.DataFrame({
    'company size': [10000],
    'location': ['Warszawa'],
    'technology': ['JavaScript'],
    'seniority': ['expert']
})

predicted_salary = model.predict(new_data)
print(f'Predicted Salary: {predicted_salary}')
