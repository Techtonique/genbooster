import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.utils.discovery import all_estimators
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.tree import ExtraTreeRegressor
from sklearn.model_selection import train_test_split
from genbooster.genboosterregressor import BoosterRegressor
from genbooster.randombagregressor import RandomBagRegressor
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

results = []

for estimator in tqdm(all_estimators(type_filter='regressor')):
    try:
        regr = BoosterRegressor(base_estimator=estimator[1]())
        start = time()
        regr.fit(X_train, y_train)
        end = time()
        print(f"Time taken: {end - start} seconds")
        rmse = np.sqrt(mean_squared_error(y_test, regr.predict(X_test)))
        print("BoosterRegressor", estimator[0], rmse)
        results.append((estimator[0], rmse, end - start))
    except Exception as e:
        print(e)
        continue

results = pd.DataFrame(results, columns=['Estimator', 'RMSE', 'Time']).sort_values(by='RMSE')
print(results)


print("\n\n RandomBagRegressor boston dataset -----")

results = []

for estimator in tqdm(all_estimators(type_filter='regressor')):
    try:
        regr = RandomBagRegressor(base_estimator=estimator[1]())
        start = time()
        regr.fit(X_train, y_train)
        end = time()
        print(f"Time taken: {end - start} seconds")
        rmse = np.sqrt(mean_squared_error(y_test, regr.predict(X_test)))
        print("RandomBagRegressor", estimator[0], rmse)
        results.append((estimator[0], rmse, end - start))
    except Exception as e:
        print(e)
        continue

results = pd.DataFrame(results, columns=['Estimator', 'RMSE', 'Time']).sort_values(by='RMSE')
print(results)