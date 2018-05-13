import numpy as np
import sys
sys.path.append("..")
from PolicyIteration.mdp import MDP


def get_action_value(mdp, state_values, state, action, gamma):
    q = 0
    for next_state, prob in mdp.get_next_states(state, action).items():
        q += prob * (mdp.get_reward(state, action, next_state) + gamma * state_values[next_state])
    return q


def get_new_state_value(mdp, state_values, state, gamma):
    if mdp.is_terminal(state):
        return 0
    v = -np.inf
    for action in mdp.get_possible_actions(state):
        v = max(v, get_action_value(mdp, state_values, state, action, gamma))
    return v


def get_optimal_action(mdp, state_values, state, gamma=0.9):
    if mdp.is_terminal(state):
        return None

    v = -np.inf
    best = None
    for action in mdp.get_possible_actions(state):
        val = get_action_value(mdp, state_values, state, action, gamma)
        if v < val:
            v = val
            best = action
    return best


def value_iteration(mdp, state_values=None, gamma=0.9, num_iter=1000, min_difference=1e-5, verbosity=0):
    if state_values is None:
        state_values = {s: 0 for s in mdp.get_all_states()}
    for i in range(num_iter):
        new_state_values = {
            state: get_new_state_value(mdp, state_values, state, gamma)
            for state in state_values
        }

        diff = max(abs(new_state_values[s] - state_values[s]) for s in mdp.get_all_states())

        if verbosity > 0:
            print("iter %4i   |   diff: %6.5f   |   V(start): %.3f " % (i, diff, new_state_values[mdp._initial_state]))

        state_values = new_state_values
        if diff < min_difference:
            break

    return state_values


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
    state_values = value_iteration(mdp, gamma=0.95, verbosity=1)

    print('State values:', state_values)
