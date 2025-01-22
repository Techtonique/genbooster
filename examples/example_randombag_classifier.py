import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.utils.discovery import all_estimators
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.tree import ExtraTreeRegressor
from sklearn.model_selection import train_test_split
from genbooster.randombagclassifier import RandomBagClassifier
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from time import time
from sklearn.utils.discovery import all_estimators


X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomBagClassifier(base_estimator=ExtraTreeRegressor(), 
                        n_hidden_features=10)
start = time()
clf.fit(X_train, y_train)
end = time()
print(f"Time taken: {end - start} seconds")
start = time()
preds = clf.predict(X_test)
end = time()
print(f"Time taken: {end - start} seconds")
print(np.mean(preds == y_test))

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomBagClassifier(base_estimator=ExtraTreeRegressor(), 
                        n_hidden_features=10)
start = time()
clf.fit(X_train, y_train)
end = time()
print(f"Time taken: {end - start} seconds")
start = time()
preds = clf.predict(X_test)
end = time()
print(f"Time taken: {end - start} seconds")
print(np.mean(preds == y_test))

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomBagClassifier(base_estimator=ExtraTreeRegressor(), 
                        n_hidden_features=10)
start = time()
clf.fit(X_train, y_train)
end = time()
print(f"Time taken: {end - start} seconds")
start = time()
preds = clf.predict(X_test)
end = time()
print(f"Time taken: {end - start} seconds")
print(np.mean(preds == y_test))

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomBagClassifier(base_estimator=ExtraTreeRegressor(), 
                        n_hidden_features=10)
start = time()
clf.fit(X_train, y_train)
end = time()
print(f"Time taken: {end - start} seconds")
start = time()
preds = clf.predict(X_test)
end = time()
print(f"Time taken: {end - start} seconds")
print(np.mean(preds == y_test))
