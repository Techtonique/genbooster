from typing import Optional, Union
import numpy as np
import pandas as pd
import nnetsauce as ns
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from .rust_core import RustBooster as _RustBooster


class BoosterClassifier(BaseEstimator, ClassifierMixin):
    """Generic Gradient Boosting Classifier (for any base learner).

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

        classes_: The classes of the target variable.

        n_classes_: The number of classes of the target variable.

        boosters_: Base learners.
    
    Examples:

        See https://github.com/Techtonique/genbooster/tree/main/examples

    """
    
    def __init__(self,
                base_estimator: Optional[BaseEstimator] = None,
                n_estimators: int = 100,
                learning_rate: float = 0.1,
                n_hidden_features: int = 5,
                direct_link: bool = True,
                weights_distribution: str = 'uniform',
                dropout: float = 0.0,
                tolerance: float = 1e-4,
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
        self.tolerance = tolerance
        self.random_state = random_state        
        self.boosters_ = None 
    
    def fit(self, X, y) -> "BoosterClassifier":
        """Fit the booster model.
        
        Parameters:

            X: Input data.

            y: Target data.
            
        Returns:

            self: The fitted boosting model.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame):
            y = y.values        
        y = np.asarray([int(x) for x in y]).ravel() 
        Y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        self.boosters_ = []
        for i in range(self.n_classes_):
            booster = _RustBooster(
                self.base_estimator,
                self.n_estimators,
                self.learning_rate,
                self.n_hidden_features,
                self.direct_link,
                weights_distribution=self.weights_distribution,
                tolerance=self.tolerance
            )
            # Try different shapes for y_i until one works
            y_i = Y[:, i]
            shapes_to_try = [
                lambda y: y,                    # 1D array (n_samples,)
                lambda y: y.reshape(-1, 1),     # 2D array (n_samples, 1)
                lambda y: y.reshape(1, -1),     # 2D array (1, n_samples)
                lambda y: y.reshape(-1),        # Flattened 1D
                lambda y: np.expand_dims(y, 0), # Add dimension at start
                lambda y: np.expand_dims(y, 1), # Add dimension at end
            ]
            
            success = False
            errors = []
            for shape_fn in shapes_to_try:
                try:
                    y_shaped = shape_fn(y_i)                    
                    booster.fit_boosting(X.reshape(X.shape[0], -1), y_shaped, 
                                       dropout=self.dropout, seed=self.random_state)
                    success = True
                    break
                except (ValueError, TypeError) as e:
                    errors.append(f"Shape {y_shaped.shape}: {str(e)}")
                    continue
            
            if not success:
                error_msg = "\n".join(errors)
                raise ValueError(f"Could not fit booster with any target shape for class {i}. Errors:\n{error_msg}")
                
            self.boosters_.append(booster)            
        return self
    
    def predict(self, X) -> np.ndarray:
        """Make predictions with the boosting model.
        
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
        """Make probability predictions with the boosting model.
        
        Parameters:

            X: Input data.
            
        Returns:
        
            preds: Probability predictions.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        raw_preds = np.asarray([booster.predict_boosting(X) for booster in self.boosters_])
        shifted_preds = raw_preds - np.max(raw_preds, axis=0)
        exp_preds = np.exp(shifted_preds)
        return exp_preds / np.sum(exp_preds, axis=0)