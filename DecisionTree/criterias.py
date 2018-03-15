import numpy as np


def entropy(freq: np.array):
    p = freq / freq.sum()
    return -np.sum(p * np.log2(p + 0.1))


def gini(freq: np.array):
    p = freq / freq.sum()
    return 1 - np.sum(p * p)


def variance(y: np.array):
    return np.var(y)


def mad_median(y: np.array):
    return np.abs(y - np.median(y)).sum() / len(y)


CRITERIAS = {
    'gini': gini,
    'entropy': entropy,
    'variance': variance,
    'mad_median': mad_median,
}
