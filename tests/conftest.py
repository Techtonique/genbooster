import pytest
import numpy as np
from sklearn.datasets import make_regression

@pytest.fixture
def regression_data():
    """Fixture to provide consistent regression data for tests"""
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    return X, y 