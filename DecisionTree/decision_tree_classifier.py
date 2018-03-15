from decision_tree import DecisionTree
import numpy as np
from collections import namedtuple
from criterias import CRITERIAS


class DecisionTreeClassifier(DecisionTree):
    class Leaf:
        __slots__ = ['mode', 'prob']

        def __init__(self, freq: np.array):
            self.prob = freq / freq.sum()
            self.mode = np.argmax(self.prob)

    def __init__(self, max_depth=np.inf, min_samples_split=2,
                 criterion='gini', debug=False):
        super().__init__(max_depth, min_samples_split, criterion, debug)

    def _leaf_value(self, y: np.array):
        freq = y.sum(0)
        if self.debug:
            print(freq)
        return DecisionTreeClassifier.Leaf(freq)

    def _select_best_split(self, X: np.array, y: np.array):
        best_score = np.inf
        best_i, best_threshold = None, None

        for i in range(self._n_feature):
            Xy = namedtuple('xy', 'x_i y_class')

            feature_list = [Xy(x_i=x[i], y_class=y_class)
                            for x, y_class in zip(X, y)]
            feature_list.sort(key=lambda x: x.x_i)

            freq_left = np.zeros(self._n_class)
            freq_right = y.sum(0)

            for j in range(len(feature_list) - 1):
                freq_left += feature_list[j].y_class
                freq_right -= feature_list[j].y_class

                if feature_list[j].x_i == feature_list[j + 1].x_i:
                    continue

                can_split = freq_left.sum() >= self.min_samples_split and \
                            freq_right.sum() >= self.min_samples_split

                if can_split:
                    score = self._fun_criterion(freq_left) * freq_left.sum() + \
                            self._fun_criterion(freq_right) * freq_right.sum()
                    if score < best_score:
                        best_i = i
                        best_threshold = (feature_list[j].x_i + feature_list[j + 1].x_i) / 2
                        best_score = score

        return DecisionTree.BestSplitResult(feature_i=best_i,
                                            threshold=best_threshold,
                                            score=best_score)

    def fit(self, X, y):
        self._fun_criterion = CRITERIAS[self.criterion]

        _X = np.array(X)
        self._n_feature = _X.shape[1]

        assert (len(_X.shape) == 2)
        assert (_X.shape[0] == len(y))

        self._code = {}
        self._encode = []
        cur_ind = 0

        for val in y:
            if val not in self._code:
                self._code[val] = cur_ind
                self._encode.append(val)
                cur_ind += 1

        self._n_class = len(self._code)

        _y = np.zeros((len(y), self._n_class))
        for i, val in enumerate(y):
            _y[i][self._code[val]] = 1

        self._root = self._build_tree(_X, _y)

        return self

    def predict(self, X):
        leaves = self._get_leaves(X)
        predictions = [leaf.mode for leaf in leaves]
        return np.array([self._encode[prediction]
                         for prediction in predictions])

    def predict_proba(self, X):
        leaves = self._get_leaves(X)
        return np.array([leaf.prob for leaf in leaves])
