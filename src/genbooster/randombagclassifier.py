from typing import Optional, Union
import numpy as np
import pandas as pd
import nnetsauce as ns
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from .rust_core import RustBooster as _RustBooster
    

class RandomBagClassifier(BaseEstimator, ClassifierMixin):
    """Generic Random Bagging Classifier (for any base learner).

    Parameters:

        base_estimator: Base learner to use for the booster.

        n_estimators: Number of boosting stages to perform.

        learning_rate: Learning rate shrinks the contribution of each estimator.

        n_hidden_features: Number of hidden features to use for the base learner.

        direct_link: Whether to use direct link for the base learner or not.

        weights_distribution: Distribution of the weights for the booster (uniform or normal).

        dropout: Dropout rate.

        random_state: Random state.
    
    Attributes:

        baggers_: The bagging learners.

        classes_: The classes of the target variable.

        n_classes_: The number of classes of the target variable.

    Examples:

        See https://github.com/Techtonique/genbooster/tree/main/examples

    """
    
    def __init__(self,
                base_estimator: Optional[BaseEstimator] = None,
                n_estimators: int = 100,
                learning_rate: float = 0.01,
                n_hidden_features: int = 5,
                direct_link: bool = True,
                weights_distribution: str = 'uniform',
                dropout: float = 0.0,
                random_state: Optional[int] = 42):
        if base_estimator is None:
            self.base_estimator = Ridge()
        else: 
            self.base_estimator = base_estimator        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.n_hidden_features = n_hidden_features
        self.direct_link = direct_link
        self.weights_distribution = weights_distribution
        self.dropout = dropout
        self.random_state = random_state
        self.y_mean_ = None
        self.boosters_ = None 
    
    def fit(self, X, y) -> "RandomBagClassifier":
        """Fit the bagging model.
        
        Parameters:

            X: Input data.

            y: Target data.
            
        Returns:

            self: The fitted bagging model.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame):
            y = y.values        
        y = np.asarray([int(x) for x in y]).ravel() 
        Y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # Store the results of the list comprehension
        self.boosters_ = []
        for i in range(self.n_classes_):
            booster = _RustBooster(
                self.base_estimator,
                self.n_estimators,
                self.learning_rate,
                self.n_hidden_features,
                self.direct_link,
                weights_distribution=self.weights_distribution
            )
            booster.fit_bagging(X, Y[:, i], dropout=self.dropout, seed=self.random_state)
            self.boosters_.append(booster)            
        return self
    
    def predict(self, X) -> np.ndarray:
        """Make predictions with the bagging model.
        
        Parameters:

            X: Input data.
            
        Returns:

            preds: Class predictions.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values       
        preds_proba = self.predict_proba(X)
        return np.argmax(preds_proba, axis=0)

    def predict_proba(self, X) -> np.ndarray:
        """Make probability predictions with the booster model.
        
        Parameters:

            X: Input data.
            
        Returns:
        
            preds: Probability predictions.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        raw_preds = np.asarray([booster.predict_bagging(X) for booster in self.boosters_])
        shifted_preds = raw_preds - np.max(raw_preds, axis=0)
        exp_preds = np.exp(shifted_preds)
        return exp_preds / np.sum(exp_preds, axis=0)