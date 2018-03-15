from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
import numpy as np

TREE_PARAMS_DICT = {
    'max_depth': 7,
    "random_state": 42
}
TAU = 0.005


class SimpleGB(BaseEstimator):
    def __init__(self, tree_params_dict, iters, tau, search_iter=5):
        self.tree_params_dict = tree_params_dict
        self.iters = iters
        self.tau = tau
        self.search_iter = search_iter

    def fit(self, X, y):
        # transform [0; 1] to [-1; 1]
        y = y * 2 - 1

        def loss(y, prediction):
            return np.log(1 + np.exp(-y * prediction)).sum()

        def grad_loss(y, prediction):
            return -y / (np.exp(prediction * y) + 1)

        self.base_algo = DecisionTreeRegressor(**self.tree_params_dict).fit(X, y)
        self.estimators = []
        curr_prediction = self.base_algo.predict(X)

        for iter_num in range(self.iters):
            grad = grad_loss(y, curr_prediction)
            algo = DecisionTreeRegressor(**self.tree_params_dict).fit(X, -grad)
            grad_prediction = algo.predict(X)

            left, right = 0, self.tau
            for i in range(self.search_iter):
                lq = left + (right - left) / 3
                rq = right - (right - left) / 3
                left_loss = loss(y, curr_prediction + lq * grad_prediction)
                right_loss = loss(y, curr_prediction + rq * grad_prediction)
                if left_loss > right_loss:
                    left = lq
                else:
                    right = rq

            alp = (left + right) / 2
            curr_prediction += alp * grad_prediction
            self.estimators.append((algo, alp))
        return self

    def predict(self, X):
        res = self.base_algo.predict(X)
        for estimator, alp in self.estimators:
            res += estimator.predict(X) * alp
        return res > 0
