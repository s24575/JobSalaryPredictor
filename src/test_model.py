import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv('data/202401_soft_eng_jobs_pol.csv')

# Data Preprocessing
data['average_salary'] = (data['salary employment min'] + data['salary employment max']) / 2
data = data.drop(['id', 'company', 'salary employment min', 'salary employment max', 'salary b2b min', 'salary b2b max'], axis=1)
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

# Model Selection and Training
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor(random_state=42))])

model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Prediction example
new_data = pd.DataFrame({
    'company size': [10000],
    'location': ['Warszawa'],
    'technology': ['JavaScript'],
    'seniority': ['expert']
})

predicted_salary = model.predict(new_data)
print(f'Predicted Salary: {predicted_salary}')
