import unittest
import numpy as np
from genbooster import BoosterRegressor, BoosterClassifier
from sklearn.datasets import make_regression, make_classification

class TestBoosterRegressor(unittest.TestCase):
    def setUp(self):
        # Create a simple regression dataset
        self.X, self.y = make_regression(n_samples=100, n_features=5, random_state=42)
        self.model = BoosterRegressor(n_estimators=10, learning_rate=0.1, random_state=42)

    def test_regressor_fit(self):
        """Test if the regressor can fit without errors"""
        try:
            self.model.fit(self.X, self.y)
            fitted = True
        except Exception as e:
            fitted = False
        self.assertTrue(fitted, "Regressor should fit without errors")

    def test_regressor_predict(self):
        """Test if the regressor can make predictions"""
        self.model.fit(self.X, self.y)
        predictions = self.model.predict(self.X)
        self.assertEqual(len(predictions), len(self.y), 
                        "Predictions length should match input length")
        self.assertTrue(isinstance(predictions, np.ndarray), 
                       "Predictions should be numpy array")

class TestBoosterClassifier(unittest.TestCase):
    def setUp(self):
        # Create a simple classification dataset with compatible parameters
        self.X, self.y = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=4,  # Increased informative features
            n_redundant=1,
            n_classes=3,
            n_clusters_per_class=1,  # Reduced clusters per class
            random_state=42
        )
        self.model = BoosterClassifier(n_estimators=10, learning_rate=0.1, random_state=42)

    def test_classifier_fit(self):
        """Test if the classifier can fit without errors"""
        try:
            self.model.fit(self.X, self.y)
            fitted = True
        except Exception as e:
            fitted = False
        self.assertTrue(fitted, "Classifier should fit without errors")

    def test_classifier_predict(self):
        """Test if the classifier can make predictions"""
        self.model.fit(self.X, self.y)
        predictions = self.model.predict(self.X)
        self.assertEqual(len(predictions), len(self.y), 
                        "Predictions length should match input length")
        self.assertTrue(isinstance(predictions, np.ndarray), 
                       "Predictions should be numpy array")

    def test_classifier_predict_proba(self):
        """Test if the classifier can make probability predictions"""
        self.model.fit(self.X, self.y)
        proba = self.model.predict_proba(self.X)
        self.assertEqual(proba.shape[0], len(np.unique(self.y)), 
                        "Probability predictions should match number of classes")
        self.assertTrue(np.allclose(np.sum(proba, axis=0), 1.0), 
                       "Probabilities should sum to 1")

if __name__ == '__main__':
    unittest.main() 