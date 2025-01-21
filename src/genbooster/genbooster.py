from typing import Optional, Union
import numpy as np
import pandas as pd
import nnetsauce as ns
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from .rust_core import RustBooster as _RustBooster

class BoosterRegressor(BaseEstimator, RegressorMixin):
    """A scikit-learn compatible wrapper for the Rust-based booster."""
    
    def __init__(
        self,
        base_estimator: Optional[BaseEstimator] = None,
        n_estimators: int = 100,
        learning_rate: float = 0.01,
        n_hidden_features: int = 5,
        direct_link: bool = True,
        weights_distribution: str = 'uniform',
        dropout: float = 0.0,
        random_state: Optional[int] = 42
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.n_hidden_features = n_hidden_features
        self.direct_link = direct_link
        self.weights_distribution = weights_distribution
        self.dropout = dropout
        self.random_state = random_state
        self.scaler_ = StandardScaler()
        self.y_mean_ = None

    def fit(self, X, y) -> "BoosterRegressor":
        """Fit the booster model."""        
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame):
            y = y.values
        scaled_X = self.scaler_.fit_transform(X)
        self.y_mean_ = np.mean(y)
        centered_y = y - self.y_mean_
        # Use Ridge as default base estimator if none provided
        if self.base_estimator is None:
            self.base_estimator_ = Ridge()
        else:
            self.base_estimator_ = self.base_estimator            
        # Initialize Rust booster
        self.booster_ = _RustBooster(
            self.base_estimator_,
            self.n_estimators,
            self.learning_rate,
            self.n_hidden_features,
            self.direct_link,
            weights_distribution=self.weights_distribution
        )        
        # Fit the model
        self.booster_.fit(
            np.asarray(scaled_X, dtype=np.float64), 
            np.asarray(centered_y, dtype=np.float64),
            dropout=self.dropout,
            seed=self.random_state if self.random_state is not None else 42
        )        
        return self
        
    def predict(self, X) -> np.ndarray:
        """Make predictions with the booster model."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        scaled_X = self.scaler_.transform(X)
        return self.booster_.predict(scaled_X) + self.y_mean_

class BoosterClassifier(BaseEstimator, ClassifierMixin):
    """A scikit-learn compatible wrapper for the Rust-based booster."""
    
    def __init__(self, base_estimator: Optional[BaseEstimator] = None, 
                 n_estimators: int = 100, learning_rate: float = 0.01, 
                 n_hidden_features: int = 5, direct_link: bool = True, 
                 dropout: float = 0.0, 
                 random_state: Optional[int] = 42):
        if base_estimator is None:
            base_estimator = Ridge()
        self.base_estimator = base_estimator  # Store original estimator
        self.multitask_estimator = ns.SimpleMultitaskClassifier(base_estimator)  # Create wrapped version
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.n_hidden_features = n_hidden_features
        self.direct_link = direct_link
        self.dropout = dropout
        self.random_state = random_state
        self.y_mean_ = None
    
    def fit(self, X, y) -> "BoosterClassifier":
        """Fit the booster model."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame):
            y = y.values        
        #y = np.asarray([int(x) for x in y]).ravel()   
        #print("In genbooster.py, L.103", y)
        self.multitask_estimator.fit(X, y)
        return self
    
    def predict(self, X) -> np.ndarray:
        """Make predictions with the booster model."""
        if isinstance(X, pd.DataFrame):
            X = X.values       
        return self.multitask_estimator.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Make probability predictions with the booster model."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.multitask_estimator.predict_proba(X)

    def score(self, X, y) -> float:
        """Calculate the accuracy of the model."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        preds = self.predict(X)
        return np.mean(preds == y)