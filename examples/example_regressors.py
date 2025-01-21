import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.utils.discovery import all_estimators
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.tree import ExtraTreeRegressor
from sklearn.model_selection import train_test_split
from genbooster.genbooster import BoosterRegressor
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# Load diabetes dataset
X, y = load_diabetes(return_X_y=True)
y = y.astype(np.float64)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42)

results = []

for estimator in tqdm(all_estimators(type_filter='regressor')):
    try:
        regr = BoosterRegressor(base_estimator=estimator[1]())
        regr.fit(X_train, y_train)
        print(estimator[0])
        results.append((estimator[0], np.sqrt(mean_squared_error(y_test, regr.predict(X_test)))))
    except Exception as e:
        print(e)
        continue

results = pd.DataFrame(results, columns=['Estimator', 'RMSE']).sort_values(by='RMSE')
print(results)