import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.utils.discovery import all_estimators
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.tree import ExtraTreeRegressor
from sklearn.model_selection import train_test_split
from genbooster.genboosterclassifier import BoosterClassifier
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from time import time
from sklearn.utils.discovery import all_estimators


datasets = [load_iris(return_X_y=True),
            load_breast_cancer(return_X_y=True),
            load_wine(return_X_y=True)]
datasets_names = ['iris', 'breast_cancer', 'wine']

# Booster
for dataset, dataset_name in zip(datasets, datasets_names):
    print("\n data set ", dataset_name)
    accuracy_scores = []
    X, y = dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    for estimator in tqdm(all_estimators(type_filter='regressor')):
        try:
            clf = BoosterClassifier(base_estimator=estimator[1](), n_hidden_features=10)
            start = time()
            clf.fit(X_train, y_train)
            training_time = time() - start
            print(f'\n Time (training): {training_time}')
            start = time()
            preds = clf.predict(X_test)
            inference_time = time() - start
            print(f'\n Time (inference): {inference_time}')
            accuracy = np.mean(preds == y_test)
            print(f'\n Accuracy: {accuracy}')
            accuracy_scores.append((dataset_name, estimator[0], accuracy,
                                    training_time, inference_time))
        except Exception as e:
            continue
    accuracy_scores_df = pd.DataFrame(accuracy_scores, columns=['dataset', 'estimator', 'accuracy',
                                                                'training_time', 'inference_time'])
    accuracy_scores_df.sort_values(by='accuracy', ascending=False, inplace=True)
    print(accuracy_scores_df)