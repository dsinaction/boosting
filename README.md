# Boosting

Implementation of boosting algorithms:

- AdaBoost

## Installation

```bash
pip install -e .
```

## Usage

```python
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons

from boosting.adaboost import AdaBoostClassifier

# Generate dataset
X, y = make_moons(n_samples = 100, noise = 0.20, random_state=10)
y = np.array([ 1 if item == 0 else -1 for item in y ])

df = pd.DataFrame(dict(x1=X[:,0], x2=X[:,1], y=y))

# AdaBoost - Fit & Predict
ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), 50)
ada.fit(X, y])

df['yhat'] = ada.predict(X)
```