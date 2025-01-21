from typing import Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from .rust_core import RustBooster as _RustBooster

class BoosterRegressor(BaseEstimator, RegressorMixin):
    """A scikit-learn compatible wrapper for the Rust-based booster."""
    
    def __init__(
        self,
        base_estimator: Optional[BaseEstimator] = None,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        n_hidden_features: int = 5,
        direct_link: bool = True,
        dropout: float = 0.0,
        random_state: Optional[int] = None
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.n_hidden_features = n_hidden_features
        self.direct_link = direct_link
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
        X = self.scaler_.fit_transform(X)
        self.y_mean_ = np.mean(y)
        y = y - self.y_mean_
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
            self.direct_link
        )        
        # Convert inputs to correct format
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)        
        # Fit the model
        self.booster_.fit(
            X, y,
            dropout=self.dropout,
            seed=self.random_state if self.random_state is not None else 42
        )        
        return self
        
    def predict(self, X) -> np.ndarray:
        """Make predictions with the booster model."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = self.scaler_.transform(X)
        return self.booster_.predict(X) + self.y_mean_
