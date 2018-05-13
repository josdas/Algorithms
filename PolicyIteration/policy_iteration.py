import numpy as np
from mdp import MDP


def compute_vpi(mdp, policy, gamma):
    s2i = {s: i for i, s in enumerate(policy)}
    n = len(s2i)
    a = np.zeros((n, n))
    b = np.zeros(n)

    for s, act in policy.items():
        s_ind = s2i[s]

        a[s_ind][s_ind] = -1

        for ns in mdp.get_next_states(s, act):
            p = mdp.get_transition_prob(s, act, ns)
            r = mdp.get_reward(s, act, ns)
            b[s_ind] -= p * r

            if ns in s2i:
                a[s_ind][s2i[ns]] += p * gamma

    solution = np.linalg.solve(a, b)

    npolicy = {
        s: solution[i]
        for s, i in s2i.items()
    }

    return npolicy


def compute_state_values(mdp, vpi, gamma):
    actions = {}
    for s in vpi:
        v = -np.inf
        actions[s] = {}
        for act in mdp.get_possible_actions(s):
            val = 0
            for s_ in mdp.get_next_states(s, act):
                p = mdp.get_transition_prob(s, act, s_)
                r = mdp.get_reward(s, act, s_)
                val += p * (r + gamma * vpi.get(s_, 0))
            actions[s][act] = val
    return actions


def compute_new_policy(mdp, vpi, gamma):
    actions = {}
    state_values = compute_state_values(mdp, vpi, gamma)
    for s in vpi:
        v = -np.inf
        best = None
        for act, val in state_values[s].items():
            if v < val:
                v = val
                best = act
        if best is not None:
            actions[s] = best
    return actions


def policy_iteration(mdp, policy=None, gamma=0.9, num_iter=1000, min_difference=1e-5, verbosity=0):
    if policy is None:
        policy = {s: np.random.choice(mdp.get_possible_actions(s))
                  for s in mdp.get_all_states()
                  if len(mdp.get_possible_actions(s)) > 0}

    vpi = compute_vpi(mdp, policy, gamma)
    for it in range(num_iter):
        npolicy = compute_new_policy(mdp, vpi, gamma)
        nvpi = compute_vpi(mdp, npolicy, gamma)

        diff = max(abs(nvpi[s] - vpi[s]) for s in vpi)

        if verbosity > 0:
            print("iter %4i   |   diff: %6.5f   |   V(start): %.3f  " % (it, diff, nvpi[mdp._initial_state]))

        vpi = nvpi
        policy = npolicy
        if diff < min_difference:
            break

    return vpi, policy


if __name__ == '__main__':
    transition_probs = {
        's0': {
            'a0': {'s0': 0.989539749, 's1': 0.010460251},
            'a1': {'s1': 1}
        },
        's1': {
            'a0': {'s1': 1},
        },
    }
    rewards = {
        's0': {
            'a0': {'s0': -1, 's1': 100},
            'a1': {'s1': 0.895},
        }
    }
    mdp = MDP(transition_probs, rewards, initial_state='s0')
    state_values, policy = policy_iteration(mdp, gamma=0.95, verbosity=1)

    print('State values:', state_values)
    print('Policy:', policy)
