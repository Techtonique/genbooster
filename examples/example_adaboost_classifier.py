import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.utils.discovery import all_estimators
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.tree import ExtraTreeRegressor
from sklearn.model_selection import train_test_split
from genbooster.adaboostclassifier import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from time import time

# Test on Iris dataset
print("\n\nAdaBoostClassifier iris dataset -----")
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=142)

# Default configuration
clf = AdaBoostClassifier(random_state=142)
start = time()
clf.fit(X_train, y_train)
end = time()
print(f"Time taken: {end - start} seconds")

preds = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print("\nClassification Report:")
print(classification_report(y_test, preds))

# Test on Breast Cancer dataset
print("\n\nAdaBoostClassifier breast cancer dataset -----")
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=142)

# Compare different configurations
configs = [
    ('Default', AdaBoostClassifier(random_state=142)),
    ('More estimators', AdaBoostClassifier(n_estimators=200, random_state=142)),
    ('More hidden features', AdaBoostClassifier(n_hidden_features=10, random_state=142)),
    ('With dropout', AdaBoostClassifier(dropout=0.1, random_state=142)),
    ('Normal weights', AdaBoostClassifier(weights_distribution='normal', random_state=142))
]

for name, model in configs:
    print(f"\nTesting {name} configuration:")
    start = time()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    end = time()
    print(f"Time taken: {end - start} seconds")
    print("Accuracy:", accuracy_score(y_test, preds))

# Test on Wine dataset
print("\n\nAdaBoostClassifier wine dataset -----")
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=142)

clf = AdaBoostClassifier(
    base_estimator=Ridge(),
    n_estimators=100,
    learning_rate=0.1,
    n_hidden_features=5,
    direct_link=True,
    random_state=142
)

start = time()
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
end = time()
print(f"Time taken: {end - start} seconds")
print("Accuracy:", accuracy_score(y_test, preds))

