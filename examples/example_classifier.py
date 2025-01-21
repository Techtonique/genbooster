import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.utils.discovery import all_estimators
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.tree import ExtraTreeRegressor
from sklearn.model_selection import train_test_split
from genbooster.genbooster import BoosterClassifier
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from sklearn.utils.discovery import all_estimators


X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = BoosterClassifier(base_estimator=ExtraTreeRegressor(), 
                        n_hidden_features=10)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print(np.mean(preds == y_test))