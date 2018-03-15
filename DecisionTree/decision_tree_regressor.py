from decision_tree import DecisionTree
from criterias import CRITERIAS
import numpy as np


class DecisionTreeRegressor(DecisionTree):
    class Leaf:
        __slots__ = ['value']

        def __init__(self, y: np.array):
            self.value = y.mean()

    def __init__(self, max_depth=np.inf, min_samples_split=2,
                 criterion='variance', debug=False):
        super().__init__(max_depth, min_samples_split, criterion, debug)

    def _leaf_value(self, y: np.array):
        return DecisionTreeRegressor.Leaf(y)

    def fit(self, X, y):
        _X = np.array(X)
        _y = np.array(y)

        assert (len(_X.shape) == 2)
        assert (_X.shape[0] == len(y))

        self._fun_criterion = CRITERIAS[self.criterion]
        self._n_feature = _X.shape[1]
        self._root = self._build_tree(_X, _y)

        return self

    def predict(self, X):
        leaves = self._get_leaves(X)
        predictions = np.array([leaf.value for leaf in leaves])
        return predictions
