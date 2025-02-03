from typing import Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.tree import ExtraTreeRegressor
from utils import one_hot_encode2
from .adaboostregressor import AdaBoostRegressor


class AdaBoostClassifier(BaseEstimator, ClassifierMixin):
    """AdaBoost Classifier using AdaBoostRegressor as a multi-task learner.
    
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

        classes_: The classes labels.

        n_classes_: The number of classes.

        boosters_: List of AdaBoostRegressor instances, one per class.
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
        self.boosters_ = []

    def fit(self, X, y) -> "AdaBoostClassifier":
        """Fit the AdaBoost classifier.
        
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
        
        # Use ExtraTreeRegressor as default base estimator if none provided
        if self.base_estimator is None:
            self.base_estimator_ = ExtraTreeRegressor(
                random_state=self.random_state
            )
        else:
            self.base_estimator_ = self.base_estimator
        
        # Train one booster per class
        for i in range(self.n_classes_):
            booster = AdaBoostRegressor(
                base_estimator=self.base_estimator_,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                n_hidden_features=self.n_hidden_features,
                direct_link=self.direct_link,
                weights_distribution=self.weights_distribution,
                dropout=self.dropout,
                tolerance=self.tolerance,
                random_state=None if self.random_state is None 
                    else self.random_state + i
            )
            
            # Convert X and y to the right format
            X_arr = np.asarray(X.values if hasattr(X, 'values') else X, dtype=np.float64)
            y_arr = np.asarray(Y[:, i], dtype=np.float64)
            
            # Fit the booster
            booster.fit(X_arr, y_arr)
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