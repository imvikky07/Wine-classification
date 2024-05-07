# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:39:00 2024

@author: Vivek
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("wine_data.csv")

# Data Exploration
print(data.head())
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Visualize distributions
plt.figure(figsize=(12, 8))
sns.histplot(data['quality'], bins=10, kde=True)
plt.title('Distribution of Quality Scores')
plt.xlabel('Quality')
plt.ylabel('Count')
plt.show()

# Data Preprocessing
# Handle missing values
# Option 1: Remove rows with missing values
# data.dropna(inplace=True)

# Option 2: Fill missing values with mean, median, or mode
# data.fillna(data.mean(), inplace=True)

# Feature Selection
X = data.drop('quality', axis=1)
y = data['quality']

# Using SelectKBest for feature selection
selector = SelectKBest(score_func=f_regression, k=5)
X_new = selector.fit_transform(X, y)

# Get columns selected by SelectKBest
selected_columns = X.columns[selector.get_support()]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

# Model Selection
# Initialize and train RandomForestRegressor
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

# Model Evaluation
# Predict on the test set
y_pred = rf_model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Outlier Detection
# Example: Isolation Forest
from sklearn.ensemble import IsolationForest

outlier_detector = IsolationForest(contamination=0.1)
outlier_detector.fit(X_train)

# Predict outliers
outliers = outlier_detector.predict(X_train)
print("Number of outliers detected:", len(outliers[outliers == -1]))

# Model Fine-Tuning (Optional)
# Example: GridSearchCV for RandomForestRegressor
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Estimator:", grid_search.best_estimator_)

# Validation and Testing
# Validate model on validation set (if available)

# Test model on held-out test set

