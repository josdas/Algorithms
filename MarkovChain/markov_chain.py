import numpy as np


def chain_coding(mchain):
    absorbing, nonabsoring = {}, {}
    for v, e in mchain.items():
        if len(e) == 0:
            absorbing[v] = len(absorbing)
        else:
            nonabsoring[v] = len(nonabsoring)
    return absorbing, nonabsoring


def qr_matrix(mchain, absorbing, nonabsoring):
    n_abs, n_non = len(absorbing), len(nonabsoring)
    Q = np.zeros((n_non, n_non))
    R = np.zeros((n_non, n_abs))
    for v in nonabsoring:
        for u, p in mchain[v].items():
            if u in absorbing:
                R[nonabsoring[v]][absorbing[u]] = p
            else:
                Q[nonabsoring[v]][nonabsoring[u]] = p
    return Q, R


def fundamental_matrix(Q):
    return np.linalg.inv(np.identity(Q.shape[0]) - Q)


def coding_start_probs(start_probs, absorbing, nonabsoring):
    return [start_probs[v] for v in absorbing], \
           [start_probs[v] for v in nonabsoring]
