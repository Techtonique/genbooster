import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split
from genbooster.adaboostregressor import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from time import time

# Test on California Housing dataset
print("\n\nAdaBoostRegressor California Housing dataset -----")

housing = fetch_california_housing()
X = housing.data
y = housing.target

print("Dataset features:", housing.feature_names)
print("Shape of data:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Test different base estimators
base_estimators = {
    'Ridge': Ridge(),
    'LinearRegression': LinearRegression()
}

for name, base_estimator in base_estimators.items():
    print(f"\nTesting with {name} base estimator:")
    
    # Regular base estimator
    start = time()
    base_model = base_estimator
    base_model.fit(X_train, y_train)
    base_pred = base_model.predict(X_test)
    base_rmse = np.sqrt(mean_squared_error(y_test, base_pred))
    end = time()
    print(f"{name} RMSE: {base_rmse:.4f} (Time: {end-start:.2f}s)")
    
    # AdaBoost with default settings
    start = time()
    ada = AdaBoostRegressor(base_estimator=base_estimator, random_state=42)
    ada.fit(X_train, y_train)
    ada_pred = ada.predict(X_test)
    ada_rmse = np.sqrt(mean_squared_error(y_test, ada_pred))
    end = time()
    print(f"AdaBoost with {name} RMSE: {ada_rmse:.4f} (Time: {end-start:.2f}s)")
    
    # AdaBoost with more hidden features
    start = time()
    ada_hidden = AdaBoostRegressor(
        base_estimator=base_estimator,
        n_hidden_features=10,
        random_state=42
    )
    ada_hidden.fit(X_train, y_train)
    ada_hidden_pred = ada_hidden.predict(X_test)
    ada_hidden_rmse = np.sqrt(mean_squared_error(y_test, ada_hidden_pred))
    end = time()
    print(f"AdaBoost with {name} (10 hidden) RMSE: {ada_hidden_rmse:.4f} (Time: {end-start:.2f}s)")

# Test on Diabetes dataset
print("\n\nAdaBoostRegressor diabetes dataset -----")

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nShape of training data:", X_train.shape)

# Compare different configurations
configs = [
    ('Default', AdaBoostRegressor(random_state=42)),
    ('More estimators', AdaBoostRegressor(n_estimators=200, random_state=42)),
    ('More hidden features', AdaBoostRegressor(n_hidden_features=10, random_state=42)),
    ('With dropout', AdaBoostRegressor(dropout=0.1, random_state=42)),
    ('Normal weights', AdaBoostRegressor(weights_distribution='normal', random_state=42))
]

for name, model in configs:
    start = time()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    end = time()
    print(f"{name} RMSE: {rmse:.4f} (Time: {end-start:.2f}s)")

# Optional: Plot predictions vs actual values
plt.figure(figsize=(10, 5))
plt.scatter(y_test, pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('AdaBoost Predictions vs Actual Values')
plt.tight_layout()
plt.show() 

print("\n\n BoosterRegressor boston dataset -----")

url = "https://raw.githubusercontent.com/Techtonique/datasets/refs/heads/main/tabular/regression/boston_dataset2.csv"
df = pd.read_csv(url)
print(df.head())

X = df.drop(columns=['target', 'training_index'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=142)

print("\nShape of training data:", X_train.shape)

# Compare different configurations
configs = [
    ('Default', AdaBoostRegressor(random_state=42)),
    ('More estimators', AdaBoostRegressor(n_estimators=200, random_state=42)), 
    ('More hidden features', AdaBoostRegressor(n_hidden_features=10, random_state=42)),
    ('With dropout', AdaBoostRegressor(dropout=0.1, random_state=42)),
    ('Normal weights', AdaBoostRegressor(weights_distribution='normal', random_state=42))
]

for name, model in configs:
    start = time()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    end = time()
    print(f"{name} RMSE: {rmse:.4f} (Time: {end-start:.2f}s)")

# Optional: Plot predictions vs actual values
plt.figure(figsize=(10, 5))
plt.scatter(y_test, pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('AdaBoost Predictions vs Actual Values')
plt.tight_layout()
plt.show()
