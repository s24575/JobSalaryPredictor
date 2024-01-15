import joblib
from matplotlib import pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("data/202401_soft_eng_jobs_pol.csv")

# Drop irrelevant columns
df = df.drop(['id', 'company', 'salary b2b min', 'salary b2b max'], axis=1)

# Remove rows with no information
df = df.dropna()

numeric_columns = ['company size']
categorical_columns = ['location', 'technology', 'seniority']

# Convert categorical variables into one-hot encoding using a pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('numerical', numeric_transformer, numeric_columns),
        ('categorical', categorical_transformer, categorical_columns)])

# Combine the preprocessing steps into a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor)])

# Convert categorical variables into one-hot encoding
# df = pd.get_dummies(df, columns=categoric_columns)

df['salary_avg'] = (df['salary employment min'] + df['salary employment max']) / 2.0
df = df.drop(['salary employment min', 'salary employment max'], axis=1)

# df = df.sample(frac=1).reset_index(drop=True)

# Split the dataset into X and y
X = df.drop('salary_avg', axis=1)
y = df['salary_avg']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = clf.fit_transform(X_train)
X_test = clf.transform(X_test)

# scaler_X = StandardScaler()
# X_train = scaler_X.fit_transform(X_train)
# X_test = scaler_X.transform(X_test)

# scaler_y = StandardScaler()
# y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
# y_test = scaler_y.transform(y_test.values.reshape(-1, 1))

# np.savetxt("test/X_train.csv", X_train.toarray(), delimiter=",")
# np.savetxt("test/y_train.csv", y_train, delimiter=",")
# np.savetxt("test/X_test.csv", X_test.toarray(), delimiter=",")
# np.savetxt("test/y_test.csv", y_test, delimiter=",")

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f"Mean Squared Error on Test Set: {loss}")

# Make predictions
predictions_before_scaling = model.predict(X_test)
predictions = scaler_y.inverse_transform(predictions_before_scaling)

r2 = r2_score(y_test, predictions)
print(f"R-squared on Test Set: {r2}")

# Save the OneHotEncoder transformer
joblib.dump(pipeline.named_steps['preprocessor'].named_transformers_['encoder'], "saved/onehot_encoder.joblib")

# Save the trained model
model.save("saved/model.keras")

# Save the scalers
joblib.dump(scaler_X, "saved/scaler_X.joblib")
joblib.dump(scaler_y, "saved/scaler_y.joblib")
