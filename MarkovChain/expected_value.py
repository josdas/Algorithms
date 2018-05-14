from markov_chain import *


def expected_value(mchain, rewards, start_probs):
    absorbing, nonabsoring = chain_coding(mchain)

    b_abs, b_non = coding_start_probs(start_probs, absorbing, nonabsoring)
    Q, R = qr_matrix(mchain, absorbing, nonabsoring)

    WQ, WR = qr_matrix(rewards, absorbing, nonabsoring)
    WQ, WR = WQ * Q, WR * R

    N = fundamental_matrix(Q)

    ev_nonabsoring = (b_non @ N @ WQ).sum()
    ev_absoring = (b_non @ N @ WR).sum()

    return ev_nonabsoring + ev_absoring


def monte_carlo_ev(mchain, rewards, start_probs, test_n=10000):
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

    s = 0
    for i in range(test_n):
        v = i2v[np.random.choice(range(n), p=start_probs_)]
        score = 0
        while len(mchain[v]) > 0:
            u = i2v[np.random.choice(range(n), p=probs[v])]
            score += rewards[v].get(u, 0)
            v = u
        s += score

    return s / test_n


if __name__ == '__main__':
    mchain = {
        0: {0: 0.5, 1: 0.25, 2: 0.25},
        1: {},
        2: {}
    }
    rewards = {
        0: {0: -1, 1: 100, 2: 1},
        1: {},
        2: {}
    }
    start_probs = {0: 0.5, 1: 0, 2: 0.5}
    print('Calculated value:', expected_value(mchain, rewards, start_probs))
    print('Monte-Carlo:', monte_carlo_ev(mchain, rewards, start_probs))
