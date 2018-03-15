import unittest
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from decision_tree_classifier import DecisionTreeClassifier
import numpy as np

EPS = 0.0001
RANDOM_STATE = 42


class TestTreeClassifier(unittest.TestCase):
    def test_digits(self):
        digits = load_digits()
        X = digits.images
        X = np.array(list(map(np.ravel, X)))
        y = digits.target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=RANDOM_STATE)

        tree = self.get_fitted_tree(X_train, y_train, debug=False, max_depth=5, criterion='gini')

        y_predicted = tree.predict(X_test)
        score = accuracy_score(y_test, y_predicted)
        self.assertGreaterEqual(score, 0.5)

    def test_simple(self):
        X = np.zeros((100, 1))
        X[:, 0] = np.linspace(-30, 30, 100)
        y = np.int8(X < 0).T[0]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

        tree = self.get_fitted_tree(X_train, y_train, debug=False, max_depth=2, criterion='entropy')

        y_predicted = tree.predict(X_test)
        score = accuracy_score(y_test, y_predicted)
        self.assertGreaterEqual(score, 1 - EPS)

    def test_proba(self):
        X = np.linspace(-30, 30, 100).reshape((100, 1))
        y = (X <= 0).ravel()

        tree = self.get_fitted_tree(X, y, debug=False, max_depth=2, criterion='entropy')

        X_test = [[1000], [-1], [0], [1], [1000]]
        y_proba = tree.predict_proba(X_test)

        for i in range(y_proba.shape[0]):
            for j in [0, 1]:
                same = (X_test[i][0] <= 0) == (j == 0)
                real_proba = 1 if same else 0
                self.assertAlmostEqual(y_proba[i][j], real_proba, delta=EPS)

    @staticmethod
    def get_fitted_tree(X, y, **kwargs):
        return DecisionTreeClassifier(**kwargs).fit(X, y)


if __name__ == '__main__':
    unittest.main()
