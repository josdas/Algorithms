from markov_chain import *
from probability import calc_probability
from collections import Counter


def conditional_ev(mchain, rewards, start_probs, n_iter=2 ** 20):
    absorbing, nonabsoring = chain_coding(mchain)

    b_abs, b_non = coding_start_probs(start_probs, absorbing, nonabsoring)
    Q, R = qr_matrix(mchain, absorbing, nonabsoring)

    WQ, WR = qr_matrix(rewards, absorbing, nonabsoring)
    WQ, WR = WQ * Q, WR * R

    n_abs, n_non = len(b_abs), len(b_non)

    ones_aa = np.eye(n_abs)
    zeros_an = np.zeros((n_abs, n_non))
    zeros_aa = np.zeros((n_abs, n_abs))
    zeros_na = np.zeros((n_non, n_abs))
    zeros_nn = np.zeros((n_non, n_non))

    prob_absorb = b_abs
    cond_ev_absorb = np.zeros(n_abs)
    prob_non = b_non
    cond_ev_non = np.zeros(n_non)

    start = np.concatenate([prob_non, cond_ev_non, cond_ev_absorb, prob_absorb])

    one_step = np.block([
        [Q, WQ, WR, R],
        [zeros_nn, Q, R, zeros_na],
        [zeros_an, zeros_an, ones_aa, zeros_aa],
        [zeros_an, zeros_an, zeros_aa, ones_aa],
    ])

    n_steps = np.linalg.matrix_power(one_step, n_iter)
    last_step = start @ n_steps

    shift = 2 * n_non
    ev = last_step[shift:shift + n_abs]
    prob = calc_probability(Q, R, b_non, b_abs)
    cond_ev = ev / prob

    return {
        v: cond_ev[i]
        for v, i in absorbing.items()
    }


def monte_carlo_cond_ev(mchain, rewards, start_probs, test_n=10000):
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
    scores = Counter()
    for i in range(test_n):
        v = i2v[np.random.choice(range(n), p=start_probs_)]
        score = 0
        while len(mchain[v]) > 0:
            u = i2v[np.random.choice(range(n), p=probs[v])]
            score += rewards[v].get(u, 0)
            v = u
        counts[v] += 1
        scores[v] += score

    return {
        v: score / counts[v]
        for v, score in scores.items()
    }


if __name__ == '__main__':
    chain = {
        0: {2: 0.8, 1: 0.2},
        1: {3: 0.3, 0: 0.5, 2: 0.2},
        2: {0: 0.4, 4: 0.6},
        3: {},
        4: {}
    }


    def gen_w(ws):
        return {
            0: {2: ws[0], 1: ws[1]},
            1: {3: ws[2], 0: ws[3], 2: ws[4]},
            2: {0: ws[5], 4: ws[6]},
            3: {},
            4: {}
        }


    for t in range(100):
        p = np.random.random(5)
        p = p / p.sum()
        s = {i: v for i, v in enumerate(p)}

        rewards = gen_w(np.random.normal(loc=3, scale=5, size=7))

        sol = conditional_ev(chain, rewards, s)
        tes = monte_carlo_cond_ev(chain, rewards, s, test_n=30000)

        print('Total diff:', abs(sol[3] - tes[3]) + abs(sol[4] - tes[4]))
        print('Calculated value:\nE[len | 3] = {:.3f}, E[len | 4] = {:.3f}'.format(sol[3], sol[4]))
        print('Monte-Carlo:\nE[len | 3] = {:.3f}, E[len | 4] = {:.3f}'.format(tes[3], tes[4]))
        print()
