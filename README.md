# Practice_model_building

## Introduction
This project demonstrates how to build, train, and evaluate machine learning models using Python and the Pandas and Scikit-learn libraries. The dataset used is "delaney_solubility_with_descriptors.csv", which contains molecular descriptors and solubility values. The goal is to predict solubility (logS) using different regression models.

## Libraries Used
The following libraries are used:
- pandas: For data handling and manipulation.
- sklearn.model_selection: To split the dataset into training and testing sets.
- sklearn.linear_model: Implements Linear Regression.
- sklearn.ensemble: Implements the Random Forest Regressor.
- sklearn.metrics: Provides functions for model evaluation.
- matplotlib.pyplot: For visualization.
- numpy: To perform polynomial fitting for visualization.

## Steps Involved

### 1. Load Dataset
import pandas as pd
df = pd.read_csv("delaney_solubility_with_descriptors.csv")
df
The dataset is read into a Pandas DataFrame for analysis.

### 2. Separate Target Variable from Features
y = df['logS']  # Target variable
X = df.drop('logS', axis=1)  # Feature variables
- y: contains the solubility values.
- X: contains the molecular descriptors used for prediction.
- In mathematical terms, y = f(X), meaning that the output (solubility) is a function of the input features (molecular descriptors).

### 3. Split Data into Training and Testing Sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
- The dataset is split into 80% training and 20% testing.
- random_state=100 ensures reproducibility.

### 4. Train Linear Regression Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
- A Linear Regression model is trained using the training data.
- The fit function allows the model to learn the relationship between X_train and y_train.

### 5. Make Predictions Using Linear Regression
y_model_train_pred = model.predict(X_train)
y_model_test_pred = model.predict(X_test)
- Predictions are made for both the training and test datasets.

### 6. Evaluate Linear Regression Performance
from sklearn.metrics import mean_squared_error, r2_score
model_train_mae = mean_squared_error(y_train, y_model_train_pred)
model_train_r2 = r2_score(y_train, y_model_train_pred)
model_test_mae = mean_squared_error(y_test, y_model_test_pred)
model_test_r2 = r2_score(y_test, y_model_test_pred)
- Mean Squared Error (MSE) and R-squared (R2) scores are calculated to evaluate the model.
- MSE measures the average squared difference between actual and predicted values (lower is better).
- R2 measures how well the model explains variance in the data (higher is better).

### 7. Store Linear Regression Results
model_results = pd.DataFrame(['Linear Regression', model_train_mae, model_train_r2, model_test_mae, model_test_r2]).transpose()
model_results.columns = ["Method", "Training MSE", "Training R2", "Test MSE", "Test R2"]
- The evaluation results are stored in a Pandas DataFrame for easy comparison.

### 8. Train Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(X_train, y_train)
- A Random Forest Regressor is trained with a maximum depth of 2.
- The random_state=100 ensures reproducibility.

### 9. Make Predictions Using Random Forest
y_rf_train_pred = rf.predict(X_train)
y_rf_test_pred = rf.predict(X_test)
- Predictions are made for both the training and test datasets.

### 10. Evaluate Random Forest Performance
rf_train_mae = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)
rf_test_mae = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)
- MSE and R2 scores are calculated to evaluate the Random Forest model.

### 11. Store Random Forest Results
rf_results = pd.DataFrame(['Random Forest', rf_train_mae, rf_train_r2, rf_test_mae, rf_test_r2]).transpose()
rf_results.columns = ["Method", "Training MSE", "Training R2", "Test MSE", "Test R2"]
- The evaluation results are stored for comparison.

### 12. Compare Model Performances
df_models = pd.concat([model_results, rf_results], axis=0)
df_models.reset_index(drop=True)
df_models
- Linear Regression and Random Forest results are combined into a single DataFrame.
- This allows for easy comparison of model performances.

### 13. Visualize Predictions
import matplotlib.pyplot as plt
import numpy as np

plt.scatter(x=y_train, y=y_model_train_pred, color="green", alpha=0.7)
x = np.polyfit(y_train, y_model_train_pred, 1)
p = np.poly1d(x)
plt.plot(y_train, p(y_train), color="red")
plt.ylabel('Predicted LogS')
plt.xlabel('Experimental LogS')
plt.show()
- A scatter plot is created to visualize actual vs. predicted values.
- A linear trend line is fitted to see the general trend.
- x = np.polyfit(y_train, y_model_train_pred, 1):
This line fits a first-degree (linear) polynomial to the data, meaning it finds the best-fit line that describes the relationship between the actual values (y_train) and the predicted values (y_model_train_pred). The function np.polyfit() returns the coefficients of this line.
- p = np.poly1d(x):
This creates a polynomial function p using the coefficients obtained from np.polyfit(). The function p(y_train) will generate predicted values along the fitted line.
plt.plot(y_train, p(y_train), color="red"):
This plots the best-fit line over the scatter plot, helping visualize how well the model’s predictions align with the actual values.
## Key Notes
### What is Linear Regression?
- It is a statistical method used to model the relationship between a dependent variable (y) and one or more independent variables (X).
- It finds the best-fitting line that minimizes the difference between actual and predicted values.
### Why Use Linear Regression?
- Linear Regression is a simple and interpretable model that assumes a linear relationship between input features and the target variable.
- It is useful when the relationship between the variables is approximately linear.
- It provides coefficients that indicate the importance of each feature in predicting the output.

### What is Random Forest?
- It is an ensemble learning method that constructs multiple decision trees and averages their predictions.
- It reduces variance and improves predictive performance compared to individual decision trees.

### Why Use Random Forest?
- Random Forest is a more complex model that can handle non-linear relationships between variables.
- It uses multiple decision trees to improve accuracy and reduce overfitting.
- It is useful when the dataset has complex patterns that a linear model cannot capture.

### Why Use a Scatter Plot?
- A scatter plot helps visualize the relationship between actual and predicted values.
- It allows us to assess how well the model's predictions align with real data.
- A strong correlation (points close to the red line) indicates a good model fit.

This project provides a complete guide to building and evaluating regression models using Scikit-learn. The results help determine which model performs better for predicting solubility values.

