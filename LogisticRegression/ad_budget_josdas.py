from sklearn.linear_model import Ridge
from scipy.optimize import linprog
import numpy as np

EPS = 0.00001
THRESHOLD = 0.52


# Mean           |  10% percentile  |  THRESHOLD
# 7.70053852238  |  4.11012232398   |  0.6
# 7.87552883641  |  4.49596897698   |  0.55
# 7.94202240041  |  4.73869882356   |  0.52
# 7.98162243931  |  4.74235208918   |  0.5
# 8.08273456049  |  4.46617630645   |  0.45

class Optimizer:
    def __init__(self):
        pass

    def predict(self, budget):
        return np.dot(budget, self.coef) + self.const_coef

    def optimize(self, origin_budget):
        n = len(origin_budget)

        default_target = self.predict([origin_budget])

        bounds = [(x * 0.95 + EPS, x * 1.05 - EPS)
                  for x in origin_budget]

        result = linprog(np.ones(n),
                         A_ub=-self.coef,
                         b_ub=self.const_coef - default_target - THRESHOLD,
                         bounds=bounds).x

        if result is None or result.sum() > origin_budget.sum():
            return origin_budget
        return result


    def fit(self, X_data, y_data):
        model = Ridge(alpha=0.01).fit(X_data, y_data)
        self.coef = model.coef_
        self.const_coef = model.intercept_
