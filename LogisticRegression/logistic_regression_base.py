import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

LR_PARAMS_DICT = {
    'C': 1000.,
    'random_state': 777,
    'iters': 3000,
    'batch_size': 1000,
    'step': 0.02
}


class MyLogisticRegression(BaseEstimator):
    def __init__(self, C, random_state, iters, batch_size, step):
        self.C = C
        self.random_state = random_state
        self.iters = iters
        self.batch_size = batch_size
        self.step = step

    def predict(self, X):
        return self.m.predict(X)

    def fit(self, X_train, y_train):
        self.m = LogisticRegression().fit(X_train, y_train)
        return self
