from typing import Optional, Union
import numpy as np
import pandas as pd
import nnetsauce as ns
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.tree import ExtraTreeRegressor
from .rust_core import RustBooster as _RustBooster
import random


class BoosterRegressor(BaseEstimator, RegressorMixin):
    """Generic Gradient Boosting Regressor (for any base learner).

        Parameters:

            base_estimator: Base learner to use for the booster.

            n_estimators: Number of boosting stages to perform.

            learning_rate: Learning rate shrinks the contribution of each estimator.

            n_hidden_features: Number of hidden features to use for the base learner.

            direct_link: Whether to use direct link for the base learner or not.

            weights_distribution: Distribution of the weights for the booster (uniform or normal).

            dropout: Dropout rate.

            tolerance: Tolerance for early stopping.

            random_state: Random state.

        Attributes:

            base_estimator_: The base learner.

            booster_: The boosting model.

            y_mean_: Mean of the target variable.

        Examples:

            See https://github.com/Techtonique/genbooster/tree/main/examples
                                
    """
    
    def __init__(
        self,
        base_estimator: Optional[BaseEstimator] = None,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        n_hidden_features: int = 5,
        direct_link: bool = True,
        weights_distribution: str = 'uniform',
        dropout: float = 0.0,
        tolerance: float = 1e-4,
        random_state: Optional[int] = 42
    ):
        self.base_estimator = base_estimator
        if base_estimator is None:
            self.base_estimator = ExtraTreeRegressor()
        else: 
            self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.n_hidden_features = n_hidden_features
        self.direct_link = direct_link
        self.weights_distribution = weights_distribution
        self.dropout = dropout
        self.tolerance = tolerance
        self.random_state = random_state
        self.scaler_ = StandardScaler()
        self.y_mean_ = None

    def fit(self, X, y) -> "BoosterRegressor":
        """Fit the boosting model.
        
        Parameters:

            X: Input data.

            y: Target data.
            
        Returns:

            self: The fitted boosting model.
        """        
        # Set random seed if provided
        if self.random_state is not None:
            # Convert to int for Python's random.seed
            seed_int = int(abs(self.random_state))
            # Set Python RNG seeds
            np.random.seed(seed_int)
            random.seed(seed_int)
            if hasattr(self.base_estimator, "random_state"):
                self.base_estimator.random_state = seed_int
            # Convert to u64 for Rust
            seed = np.uint64(seed_int)
        else:
            # Use a random seed if none provided
            seed_int = np.random.randint(0, 2**31 - 1)
            np.random.seed(seed_int)
            random.seed(seed_int)
            seed = np.uint64(seed_int)
            
        # Convert to numpy arrays and ensure float64 dtype with C-contiguous memory layout
        X = np.array(X, dtype=np.float64, copy=True, order='C')
        y = np.array(y, dtype=np.float64, copy=True, order='C')
        
        # Scale X and force conversion to numpy array
        scaled_X = np.array(self.scaler_.fit_transform(X), dtype=np.float64, copy=True, order='C')
        
        self.y_mean_ = float(np.mean(y))
        centered_y = np.array(y - self.y_mean_, dtype=np.float64, copy=True, order='C')
        
        # Ensure y is 1D array
        if centered_y.ndim == 2:
            if centered_y.shape[1] != 1:
                raise ValueError("y must have shape (n_samples,) or (n_samples, 1)")
            centered_y = centered_y.ravel()  # Convert to 1D array
            
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
            weights_distribution=self.weights_distribution,
            tolerance=self.tolerance
        )        
        
        # Fit the model
        self.booster_.fit_boosting(
            scaled_X,
            centered_y,
            dropout=self.dropout,
            seed=seed
        )        
        return self
        
    def predict(self, X) -> np.ndarray:
        """Make predictions with the boosting model.

        Parameters:

            X: Input data.
            
        Returns:
        
            preds: Predictions.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        scaled_X = self.scaler_.transform(X)
        return self.booster_.predict_boosting(scaled_X) + self.y_mean_