import abc
import numpy as np
from sklearn.base import BaseEstimator
from collections import namedtuple


class DecisionTree(BaseEstimator):
    __metaclass__ = abc.ABCMeta

    BestSplitResult = namedtuple('result', 'feature_i threshold score')

    class Vertex:
        __slots__ = ['left', 'right', 'feature_i', 'threshold']

        def __init__(self, left, right, feature_i, threshold):
            self.left = left
            self.right = right
            self.feature_i = feature_i
            self.threshold = threshold

    def __init__(self, max_depth, min_samples_split,
                 criterion, debug=False):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.debug = debug

    def _get_predicate(self, X: np.array, feature_i, threshold):
        return X[:, feature_i] < threshold

    def _select_best_split(self, X: np.array, y: np.array):
        best_score = np.inf
        best_i, best_threshold = None, None

        for i in range(self._n_feature):
            feature_list = list(set(sorted(x[i] for x in X)))

            for j in range(len(feature_list) - 1):
                s = (feature_list[j] + feature_list[j + 1]) / 2

                predicate = self._get_predicate(X, i, s)
                l_y, r_y = y[predicate], y[~predicate]

                can_split = len(l_y) >= self.min_samples_split and \
                            len(r_y) >= self.min_samples_split
                if can_split:
                    score = self._fun_criterion(l_y) * len(l_y) + \
                            self._fun_criterion(r_y) * len(r_y)

                    if score < best_score:
                        best_i = i
                        best_threshold = s
                        best_score = score

        return DecisionTree.BestSplitResult(feature_i=best_i,
                                            threshold=best_threshold,
                                            score=best_score)

    def _build_tree(self, X: np.array, y: np.array, depth=0):
        if depth >= self.max_depth:
            if self.debug:
                print('Created a leaf by max_depth')

            return self._leaf_value(y)

        best_i, best_threshold, best_score = self._select_best_split(X, y)

        if best_i is None:
            if self.debug:
                print('Created a leaf by min_samples_split. Score', best_score)

            return self._leaf_value(y)
        else:
            if self.debug:
                print('Created a new vertex')

            predicate = self._get_predicate(X, best_i, best_threshold)
            (l_x, l_y), (r_x, r_y) = (X[predicate], y[predicate]), (X[~predicate], y[~predicate])

            l_vertex = self._build_tree(l_x, l_y, depth + 1)
            r_vertex = self._build_tree(r_x, r_y, depth + 1)

            return DecisionTree.Vertex(l_vertex, r_vertex,
                                       feature_i=best_i, threshold=best_threshold)

    def _dfs_leaves(self, X: np.array, vertex):
        if not isinstance(vertex, DecisionTree.Vertex):
            return np.array([vertex] * X.shape[0])

        predicate = self._get_predicate(X, vertex.feature_i, vertex.threshold)

        l_leaves = self._dfs_leaves(X[predicate], vertex.left)
        r_leaves = self._dfs_leaves(X[~predicate], vertex.right)

        merge = np.empty(X.shape[0], dtype=object)
        l_i, r_i = 0, 0

        for i, isLeft in enumerate(predicate):
            if isLeft:
                merge[i] = l_leaves[l_i]
                l_i += 1
            else:
                merge[i] = r_leaves[r_i]
                r_i += 1

        return merge

    def _get_leaves(self, X):
        _X = np.array(X)

        assert (len(_X.shape) == 2)
        assert (self._n_feature == _X.shape[1])

        leaves = self._dfs_leaves(_X, self._root)
        return leaves

    @abc.abstractmethod
    def _leaf_value(self, y: np.array):
        return

    @abc.abstractmethod
    def fit(self, X, y):
        return

    @abc.abstractmethod
    def predict(self, X):
        return
