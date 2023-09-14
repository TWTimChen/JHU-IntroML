import numpy as np
from collections import Counter

class NullModel:
    def __init__(self, task='classification'):
        self.task = task
        self.prediction = None

    def fit(self, X, y):
        if self.task == 'classification':
            self.prediction = Counter(y).most_common(1)[0][0]
        else:
            self.prediction = np.mean(y)

    def predict(self, X):
        return np.full(X.shape[0], self.prediction)
