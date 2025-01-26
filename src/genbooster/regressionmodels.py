from typing import Optional, Union
import numpy as np
import pandas as pd
import nnetsauce as ns
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from .rust_core import Regressor as _Regressor


class LinfaRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model_name="LinearRegression"):
        self.model_name = model_name
        self.model = _Regressor(model_name=self.model_name)

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values
        try: 
            self.model.fit(X, y)
        except TypeError as e:
            print(e)
            try: 
                self.model.fit(X, y.reshape(-1, 1))
            except TypeError as e:
                print(e)
                try: 
                    self.model.fit(X, y.reshape(1, -1))
                except TypeError as e:
                    print(e)
                    try: 
                        self.model.fit(X, y.ravel())
                    except TypeError as e:
                        print(e)
                        raise e
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)