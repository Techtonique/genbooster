from typing import Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.tree import ExtraTreeRegressor
try:
    from .rust_core import AdaBoostRegressor as _AdaBoostRegressor
except ImportError:
    # Fallback for documentation generation
    class _AdaBoostRegressor:
        """Rust implementation of AdaBoostRegressor."""
        pass

class AdaBoostRegressor(BaseEstimator, RegressorMixin):
    """AdaBoost Regressor with neural network-like feature transformation.
    
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

        scaler_: StandardScaler for feature scaling.
    """
    
    def __init__(
        self,
        base_estimator: Optional[BaseEstimator] = None,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        n_hidden_features: int = 5,
        direct_link: bool = True,
        weights_distribution: str = "uniform",
        dropout: float = 0.0,
        tolerance: float = 1e-4,
        random_state: Optional[int] = None
    ):
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

    def fit(self, X, y) -> "AdaBoostRegressor":
        """Fit the AdaBoost regressor.
        
        Parameters:

            X: Input data
            
            y: Target values
        """
        # Convert inputs to numpy arrays
        X_arr = np.asarray(X.values if hasattr(X, 'values') else X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        
        # Fit and transform with StandardScaler
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_arr)
        
        # Initialize base estimator if None
        if self.base_estimator is None:
            self.base_estimator_ = ExtraTreeRegressor(random_state=self.random_state)
        else:
            self.base_estimator_ = self.base_estimator
        
        # Create and fit the booster
        self.booster_ = _AdaBoostRegressor(
            base_estimator=self.base_estimator_,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            n_hidden_features=self.n_hidden_features,
            direct_link=self.direct_link,
            weights_distribution=self.weights_distribution,
            dropout=self.dropout,
            tolerance=self.tolerance,
            random_state=self.random_state
        )
        
        self.booster_.fit(X_scaled, y_arr)
        return self

    def predict(self, X) -> np.ndarray:
        """Make predictions with the AdaBoost model.
        
        Parameters:

            X: Input data.
            
        Returns:

            predictions: Model predictions.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.array(X, dtype=np.float64, copy=True, order='C')
        scaled_X = self.scaler_.transform(X)
        return self.booster_.predict(scaled_X)