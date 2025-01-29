from typing import Optional, Union
import numpy as np
import pandas as pd
import nnetsauce as ns
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.tree import ExtraTreeRegressor
from utils import one_hot_encode2
from .genboosterregressor import BoosterRegressor


class BoosterClassifier(BaseEstimator, ClassifierMixin):
    """Generic Gradient Boosting Classifier (for any base learner).

    Parameters:

        base_estimator: Base learner to use for the booster. Default is ExtraTreeRegressor.

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
        self.boosters_ = [] 
    
    def fit(self, X, y) -> "BoosterClassifier":
        """Fit the booster model.
        
        Parameters:
            X: Input data.
            y: Target data.
            
        Returns:
            self: The fitted boosting model.
        """
        # Get unique classes and one-hot encode
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)        
        Y = one_hot_encode2(y, self.n_classes_)
        
        # Train one booster per class
        for i in range(self.n_classes_):
            booster = BoosterRegressor(
                base_estimator=self.base_estimator,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                n_hidden_features=self.n_hidden_features,
                direct_link=self.direct_link,
                weights_distribution=self.weights_distribution,
                tolerance=self.tolerance, 
                dropout=self.dropout, 
                random_state=self.random_state
            )
            
            # Convert X and y to the right format without reshaping y
            X_arr = np.asarray(X.values if hasattr(X, 'values') else X, dtype=np.float64)
            y_arr = np.asarray(Y[:, i], dtype=np.float64)
            
            # Fit the booster
            booster.fit(X=X_arr, y=y_arr)
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
        raw_preds = np.asarray([booster.predict(X) for booster in self.boosters_])
        shifted_preds = raw_preds - np.max(raw_preds, axis=0)
        exp_preds = np.exp(shifted_preds)
        return exp_preds / np.sum(exp_preds, axis=0)