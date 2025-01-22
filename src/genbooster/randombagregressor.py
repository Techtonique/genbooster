from typing import Optional, Union
import numpy as np
import pandas as pd
import nnetsauce as ns
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from .rust_core import RustBooster as _RustBooster

class RandomBagRegressor(BaseEstimator, RegressorMixin):
    """Generic Random Bagging Regressor (for any base learner).

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

            y_mean_: The mean of the target variable.

        Examples:

            See https://github.com/Techtonique/genbooster/tree/main/examples
                    
    """
    
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

    def fit(self, X, y) -> "RandomBagRegressor":
        """Fit the bagging model.
        
        Parameters:

            X: Input data.

            y: Target data.
            
        Returns:

            self: The fitted booster model.
        """        
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
        self.booster_.fit_bagging(
            np.asarray(scaled_X, dtype=np.float64), 
            np.asarray(centered_y, dtype=np.float64),
            dropout=self.dropout,
            seed=self.random_state if self.random_state is not None else 42
        )        
        return self
        
    def predict(self, X) -> np.ndarray:
        """Make predictions with the bagging model.

        Parameters:

            X: Input data.
            
        Returns:

            preds: Predictions.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        scaled_X = self.scaler_.transform(X)
        return self.booster_.predict_bagging(scaled_X) + self.y_mean_
