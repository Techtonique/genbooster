import numpy as np
import pandas as pd
import GPopt as gp 
from matplotlib import pyplot as plt
from sklearn.utils.discovery import all_estimators
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.tree import ExtraTreeRegressor
from sklearn.model_selection import train_test_split
from genbooster.genboosterregressor import BoosterRegressor
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from sklearn.utils.discovery import all_estimators
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

datasets = [load_diabetes(return_X_y=True), fetch_california_housing(return_X_y=True)]
datasets_names = ['diabetes', 'california_housing']
accuracy_scores = []


def genboost_cv(X_train, y_train,
               n_estimators=100,
               learning_rate=0.1,
               n_hidden_features=5,
               dropout=0):
    try:
        # Ensure parameters are valid
        if learning_rate <= 0 or n_estimators <= 0 or n_hidden_features <= 0:
            return -1e6
            
        estimator = BoosterRegressor(n_estimators=int(n_estimators),
                                    learning_rate=learning_rate,
                                    n_hidden_features=int(n_hidden_features),
                                    dropout=dropout)
        
        # Initialize KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        
        # Perform cross-validation manually
        for train_idx, val_idx in kf.split(X_train):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            # Fit and predict
            estimator.fit(X_fold_train, y_fold_train)
            preds = estimator.predict(X_fold_val)
            score = np.sqrt(mean_squared_error(y_fold_val, preds))
            scores.append(score)
        
        scores = np.array(scores)
        return np.mean(scores)
        
    except Exception as e:
        print(f"CV Error: {e}")
        return -1e6

def optimize_genboost(X_train, y_train):

  # objective function for hyperparams tuning
  def crossval_objective(x):

    return genboost_cv(
      X_train=X_train,
      y_train=y_train,
      n_estimators=int(10**x[0]),
      learning_rate=10**x[1],
      n_hidden_features=int(10**x[2]),
      dropout=x[3])

  gp_opt = gp.GPOpt(objective_func=crossval_objective,
                      lower_bound = np.array([1,   -6,   1,   0]),
                      upper_bound = np.array([3, -0.1,   3.5, 0.4]),
                      params_names=["n_estimators", "learning_rate",
                                    "n_hidden_features", "dropout"],
                      n_init=10, n_iter=90, seed=123)
  return {'parameters': gp_opt.optimize(verbose=2, abs_tol=1e-2), 'opt_object':  gp_opt}

for dataset, dataset_name in zip(datasets, datasets_names):
    print(dataset_name)
    X, y = dataset    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    optimize_genboost(X_train, y_train)

