# coding=utf-8

from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import imp
import signal
import traceback
import sys
import json

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def signal_handler(signum, frame):
    raise Exception("Timed out!")


class Checker(object):
    def __init__(self, random_state=42):
        # ВНИМАНИЕ !!!
        # При тестировании seed будет изменён
        # Не переобучитесь!
        random_gen = np.random.RandomState(random_state)

        weights = (0.05 + random_gen.exponential(0.75, size=15)) * 2

        X_data = random_gen.uniform(0., 4, size=(40, 15))
        errors = random_gen.normal(0., 2., size=40)

        split_pos = 25
        self.X_train = X_data[:split_pos]
        self.errors_train = errors[:split_pos]
        self.X_test = X_data[split_pos:]
        self.errors_test = errors[split_pos:]
        self.weights = weights

        self.applications = 0

    def check(self, script_path):
        try:
            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(120)
            algo_impl = imp.load_source('algo_impl_{}'.format(self.applications), script_path)
            self.applications += 1
            algo = algo_impl.Optimizer()
            algo.fit(np.array(self.X_train), np.dot(self.X_train, self.weights) + self.errors_train)

            real_opt = algo_impl.Optimizer()
            real_opt.set_weight(self.weights)

            saved_moneys = 0.
            for budget, target_error in zip(self.X_test, self.errors_test):
                origin_budget = np.array(budget)
                optimized_budget = np.array(algo.optimize(origin_budget))

                if ((origin_budget * 0.95 <= optimized_budget) & (optimized_budget <= origin_budget * 1.05)).all():
                    if np.dot(optimized_budget, self.weights) >= np.dot(origin_budget, self.weights):
                        saved_moneys += np.sum(origin_budget) - np.sum(optimized_budget)
                    else:
                        print('Fail. new {}. old {}'.format(np.dot(optimized_budget, self.weights),
                                                            np.dot(origin_budget, self.weights)))
                else:
                    print('Fail bounds.')

            return saved_moneys
        except:
            traceback.print_exception(*sys.exc_info())
            return None
        print('OK')


if __name__ == '__main__':
    res = np.array([Checker(i).check(SCRIPT_DIR + '/ad_budget_josdas.py')
                    for i in range(42, 90)])
    print(res)
    print(res.mean(), np.percentile(res, 10))
