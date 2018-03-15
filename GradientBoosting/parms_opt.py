from sklearn.model_selection import cross_val_score
import xgboost as xgb
import numpy as np
import signal
import os
import sys
import pandas
import json
import random
import pickle
import time
from copy import deepcopy

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def signal_handler(signum, frame):
    raise Exception("Timed out!")


class Checker:
    def __init__(self, data_path=SCRIPT_DIR + '/HR.csv'):
        df = pandas.read_csv(data_path)
        target = 'left'
        features = [c for c in df if c != target]
        self.target = np.array(df[target])
        self.data = np.array(df[features])

    def check(self, params):
        start_time = time.time()
        try:
            signal.signal(signal.SIGALRM, signal_handler)
            # Time limit = 2m
            signal.alarm(120)
            estimator = xgb.XGBClassifier(**params)
            score = np.mean(cross_val_score(
                estimator, self.data, self.target,
                scoring='accuracy',
                cv=3
            ))
        except:
            print("Unexpected error:", sys.exc_info()[0])
            score = -1
        print('Time:', time.time() - start_time)
        return score


BASE_PARAMS_PATH = 'xgboost_josdas.json'

TYPES = {
    "learning_rate": float,
    "max_depth": int,
    "n_estimators": int,
    "min_child_weight": float,
    "seed": int,
    "gamma": float
}

MAX_CHANGE_TYPE = {
    "learning_rate": 1.3,
    "max_depth": 1.2,
    "n_estimators": 1.3,
    "min_child_weight": 1.3,
    "gamma": 1.2
}

PROB_CHANGE = 0.6
MAX_CHANGE = 0.5

if __name__ == '__main__':
    random.seed(47)

    checker = Checker()

    with open(BASE_PARAMS_PATH, 'r') as f:
        base_params = json.load(f)

    checked = set()
    params_store = []

    cur_score = checker.check(base_params)
    print('Base score:', cur_score)

    while True:
        print('\n' * 2)

        cur_params = deepcopy(base_params)
        for k in MAX_CHANGE_TYPE.keys():
            if random.random() < PROB_CHANGE:
                v = cur_params[k]

                max_c = MAX_CHANGE_TYPE[k]
                min_c = 1 / MAX_CHANGE_TYPE[k]
                coef = random.random() * abs(max_c - min_c) + min_c

                new_v = TYPES[k](coef * v)
                cur_params[k] = new_v

        frozen_params = tuple(sorted(cur_params.items(), key=lambda x: x[0]))
        if frozenset(cur_params) in checked:
            continue

        checked.add(frozen_params)
        new_score = checker.check(cur_params)
        params_store.append((new_score, cur_params))

        print(cur_params)
        print('Score:', new_score)
        print('Best:', cur_score)

        if cur_score <= new_score + 0.0004:
            if cur_score < new_score:
                print('New best score:', new_score)
                cur_score = new_score
            base_params = cur_params
            print('Upd')

        with open('temp_opt3.pickle', 'wb') as fl:
            pickle.dump(params_store, fl)
