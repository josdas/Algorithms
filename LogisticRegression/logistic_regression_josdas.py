import numpy as np
from sklearn.base import BaseEstimator

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

    def __predict(self, X):
        return np.dot(X, self.w) + self.w0

    def predict(self, X):
        res = self.__predict(X)
        res[res > 0] = 1
        res[res < 0] = 0
        return res

    def der_reg(self):
        return self.w / self.C

    def der_loss(self, x, y):
        y = 2 * y - 1
        n_samples, n_features = x.shape

        prediction = self.__predict(x)
        der_prediction = -y / (np.exp(prediction * y) + 1)
        der_w = (der_prediction @ x) / n_samples
        der_w0 = der_prediction.mean(-1)

        return der_w, der_w0

    def fit(self, X_train, y_train):
        random_gen = np.random.RandomState(self.random_state)
        size, dim = X_train.shape

        self.w = random_gen.rand(dim)
        self.w0 = random_gen.randn()

        for _ in range(self.iters):
            rand_indices = random_gen.choice(size, self.batch_size)
            x = X_train[rand_indices]
            y = y_train[rand_indices]

            der_w, der_w0 = self.der_loss(x, y)
            der_w += self.der_reg()

            self.w -= der_w * self.step
            self.w0 -= der_w0 * self.step
        return self
