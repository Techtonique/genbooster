import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.utils.discovery import all_estimators
from sklearn.datasets import load_diabetes, load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression
from sklearn.tree import ExtraTreeRegressor
from sklearn.model_selection import train_test_split
from genbooster.genboosterclassifier import BoosterClassifier
from genbooster.genboosterregressor import BoosterRegressor
from genbooster.randombagregressor import RandomBagRegressor
from genbooster.randombagclassifier import RandomBagClassifier
from genbooster.regressionmodels import LinfaRegressor
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from time import time

print("\n\n BoosterRegressor boston dataset -----")

url = "https://raw.githubusercontent.com/Techtonique/datasets/refs/heads/main/tabular/regression/boston_dataset2.csv"
df = pd.read_csv(url)
print(df.head())

X = df.drop(columns=['target', 'training_index'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=142)

print("X_train.shape", X_train.shape)
print("y_train.shape", y_train.shape)
print("X_train.dtypes", X_train.dtypes)
print("y_train.dtypes", y_train.dtypes)

regr = LinfaRegressor(model_name="LinearRegression")
start = time()
regr.fit(X_train, y_train)
end = time()
print(f"Time taken: {end - start} seconds")
rmse = np.sqrt(mean_squared_error(y_test, regr.predict(X_test)))
print("LinfaRegressor RMSE", rmse)

regr = LinearRegression()
start = time()
regr.fit(X_train, y_train)
end = time()
print(f"Time taken: {end - start} seconds")
rmse = np.sqrt(mean_squared_error(y_test, regr.predict(X_test)))
print("LinearRegression RMSE", rmse)


X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=142)

regr = LinfaRegressor(model_name="LinearRegression")
start = time()
regr.fit(X_train, y_train)
end = time()
print(f"Time taken: {end - start} seconds")
rmse = np.sqrt(mean_squared_error(y_test, regr.predict(X_test)))
print("LinfaRegressor RMSE", rmse)


regr = LinearRegression()
start = time()
regr.fit(X_train, y_train)
end = time()
print(f"Time taken: {end - start} seconds")
rmse = np.sqrt(mean_squared_error(y_test, regr.predict(X_test)))
print("LinearRegression RMSE", rmse)


datasets = [load_iris(as_frame=True),
            load_breast_cancer(as_frame=True),
            load_wine(as_frame=True)]
datasets_names = ['iris', 'breast_cancer', 'wine']

# Booster
for dataset, dataset_name in zip(datasets, datasets_names):
    print("\n data set ", dataset_name)
    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    regr = BoosterClassifier(base_estimator=LinearRegression())
    start = time()
    regr.fit(X_train, y_train)
    end = time()
    print(f"Time taken: {end - start} seconds")
    accuracy = regr.score(X_test, y_test)
    print("BoosterClassifier LinearRegression Accuracy", accuracy)

for dataset, dataset_name in zip(datasets, datasets_names):
    print("\n data set ", dataset_name)
    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    regr = BoosterClassifier(base_estimator=LinfaRegressor(model_name="LinearRegression"))
    start = time()
    regr.fit(X_train, y_train)
    end = time()
    print(f"Time taken: {end - start} seconds")
    accuracy = regr.score(X_test, y_test)
    print("BoosterClassifier LinfaRegressor Accuracy", accuracy)

print("\n\n RandomBagRegressor boston dataset -----")
url = "https://raw.githubusercontent.com/Techtonique/datasets/refs/heads/main/tabular/regression/boston_dataset2.csv"
df = pd.read_csv(url)
print(df.head())

X = df.drop(columns=['target', 'training_index'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=142)

regr = RandomBagRegressor(base_estimator=LinearRegression())
start = time()
regr.fit(X_train, y_train)
end = time()
print(f"Time taken: {end - start} seconds")
rmse = np.sqrt(mean_squared_error(y_test, regr.predict(X_test)))
print("RandomBagRegressor RMSE", rmse)


for dataset, dataset_name in zip(datasets, datasets_names):
    print("\n data set ", dataset_name)
    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    regr = RandomBagClassifier(base_estimator=LinearRegression())
    start = time()
    regr.fit(X_train, y_train)
    end = time()
    print(f"Time taken: {end - start} seconds")
    accuracy = regr.score(X_test, y_test)
    print("RandomBagClassifier LinearRegression Accuracy", accuracy)


for dataset, dataset_name in zip(datasets, datasets_names):
    print("\n data set ", dataset_name)
    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    regr = RandomBagClassifier(base_estimator=LinfaRegressor(model_name="LinearRegression"))
    start = time()
    regr.fit(X_train, y_train)
    end = time()
    print(f"Time taken: {end - start} seconds")
    accuracy = regr.score(X_test, y_test)
    print("RandomBagClassifier LinfaRegressor Accuracy", accuracy)
