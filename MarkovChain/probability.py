from markov_chain import *
from collections import Counter


def calc_probability(Q, R, b_non, b_abs):
    N = fundamental_matrix(Q)
    return b_non @ N @ R + b_abs


def probability(mchain, start_probs):
    absorbing, nonabsoring = chain_coding(mchain)

    b_abs, b_non = coding_start_probs(start_probs, absorbing, nonabsoring)
    Q, R = qr_matrix(mchain, absorbing, nonabsoring)

    probs = calc_probability(Q, R, b_non, b_abs)

    return {
        s: probs[i]
        for s, i in absorbing.items()
    }


def monte_carlo_probs(mchain, start_probs, test_n=10000):
    v2i = {v: i for i, v in enumerate(start_probs)}
    i2v = {i: v for v, i in v2i.items()}
    n = len(v2i)

    start_probs_ = np.zeros(n)
    for v, p in start_probs.items():
        start_probs_[v2i[v]] = p

    probs = np.zeros((n, n))
    for v, e in mchain.items():
        for u, p in e.items():
            probs[v2i[v]][v2i[u]] = p

    counts = Counter()
    for i in range(test_n):
        v = i2v[np.random.choice(range(n), p=start_probs_)]
        while len(mchain[v]) > 0:
            u = i2v[np.random.choice(range(n), p=probs[v])]
            v = u
        counts[v] += 1

    return {
        k: v / test_n
        for k, v in counts.items()
    }


if __name__ == '__main__':
    mchain = {
        0: {0: 0.5, 1: 0.25, 2: 0.25},
        1: {},
        2: {}
    }
    start_probs = {0: 0.5, 1: 0, 2: 0.5}
    print('Calculated value:', probability(mchain, start_probs))
    print('Monte-Carlo:', monte_carlo_probs(mchain, start_probs))
