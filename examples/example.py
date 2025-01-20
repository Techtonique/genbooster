import genbooster as gb 
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.tree import ExtraTreeRegressor
from sklearn.model_selection import train_test_split

# Create custom base estimator
base_estimator = ExtraTreeRegressor()

# Create booster
model = gb.RustBooster(
    base_estimator=base_estimator,
    n_estimators=100,
    learning_rate=0.1,
    n_hidden_features=5,
    direct_link=True
)

# Use it like any scikit-learn estimator
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42)
model.fit(X_train, y_train, dropout=0.1, seed=42)  # Note: dropout moved to fit method
predictions = model.predict(X_test)

print("Predictions shape:", predictions.shape)

print(np.sqrt(np.mean((predictions - y_test) ** 2)))