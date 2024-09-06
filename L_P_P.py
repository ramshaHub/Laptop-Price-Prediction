# Laptop-Price-Prediction
import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import Normalizer, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from google.colab import files
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
uploaded = files.upload()
filename = list(uploaded.keys())[0]
data = read_csv(filename)
print(filename)

# Data Exploration
print("Data Overview:")
print(data.head())
print("\nData Description:")
print(data.describe())

# Convert relevant columns to numerical values
data['ram_gb'] = data['ram_gb'].str.replace(' GB', '').astype(float)
data['ssd'] = data['ssd'].str.replace(' GB', '').astype(float)
data['hdd'] = data['hdd'].str.replace(' GB', '').astype(float)
data['graphic_card_gb'] = data['graphic_card_gb'].str.replace(' GB', '').astype(float)
data['Price'] = data['Price'].astype(float)
data['Number of Ratings'] = data['Number of Ratings'].astype(int)
data['Number of Reviews'] = data['Number of Reviews'].astype(int)

# Feature Engineering
data['total_storage'] = data['ssd'] + data['hdd']

# Define the feature set and target variable
numerical_cols = ['ram_gb', 'ssd', 'hdd', 'graphic_card_gb', 'total_storage', 'Number of Ratings', 'Number of Reviews']
categorical_cols = ['brand', 'processor_brand', 'processor_name', 'processor_gnrtn']
X = data[numerical_cols + categorical_cols]
y = data['Price']

# Preprocessing for numerical data
numerical_transformer = StandardScaler()

# Preprocessing for categorical data
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Create and evaluate the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display the results
print("\nModel Evaluation:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Plotting the actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.title('Actual vs Predicted Prices (Random Forest)')
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()

# Feature importances
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
features = numerical_cols + list(pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_cols))

plt.figure(figsize=(10, 6))
plt.title('Feature Importances (Random Forest)')
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

