# Genbooster

A gradient boosting and bagging (RandomBagClassifier, similar to RandomForestClassifier) implementation using Rust and Python. Any base learner can be employed. Base learners input features are engineered using a randomized artificial neural network layer.

For more details, see also [https://www.researchgate.net/publication/386212136_Scalable_Gradient_Boosting_using_Randomized_Neural_Networks](https://www.researchgate.net/publication/386212136_Scalable_Gradient_Boosting_using_Randomized_Neural_Networks).

![PyPI](https://img.shields.io/pypi/v/genbooster) 
[![Downloads](https://pepy.tech/badge/genbooster)](https://pepy.tech/project/genbooster) 
[![Documentation](https://img.shields.io/badge/documentation-is_here-green)](https://techtonique.github.io/genbooster/)

## 1 - Installation

From PyPI:
```bash
pip install genbooster
```
From GitHub:
```bash
pip install git+https://github.com/Techtonique/genbooster.git
```

It might be required to install Rust and Cargo first (**normally, it isn't, and you can skip this step**): 

Command line:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
```

Python:
```python
import os
os.environ['PATH'] = f"/root/.cargo/bin:{os.environ['PATH']}"
```

Command line:
```bash
echo $PATH
rustc --version
cargo --version
```

## 2 - Usage

### 2.1 - Boosting

```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.utils.discovery import all_estimators
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.tree import ExtraTreeRegressor
from sklearn.model_selection import train_test_split
from genbooster.genboosterclassifier import BoosterClassifier
from genbooster.randombagclassifier import RandomBagClassifier

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = BoosterClassifier(base_estimator=ExtraTreeRegressor())
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print(np.mean(preds == y_test))
```

### 2.2 - Bagging (RandomBagClassifier, similar to RandomForestClassifier)

```python
clf = RandomBagClassifier(base_estimator=ExtraTreeRegressor())
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print(np.mean(preds == y_test))
```

