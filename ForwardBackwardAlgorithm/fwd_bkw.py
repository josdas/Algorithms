from copy import copy


def forward(trans_prob, emm_prob, observations, states, start_prob):
    n = len(observations)
    alpha = [None] * n
    alpha[0] = copy(start_prob)
    for i in range(1, n):
        observation = observations[i]
        cur_a = {}
        prev_a = alpha[i - 1]
        for state in states:
            trans_sum = sum(prev_a[v] * trans_prob[v].get(state, 0)
                            for v in states)
            cur_a[state] = trans_sum * emm_prob[state].get(observation, 0)
        alpha[i] = cur_a
    return alpha


def backward(trans_prob, emm_prob, observations, states, end_st):
    n = len(observations)
    betta = [None] * n
    betta[n - 1] = {v: trans_prob[v].get(end_st, 0) for v in states}
    for i in range(n - 2, -1, -1):
        observation = observations[i]
        cur_b = {}
        prev_b = betta[i + 1]
        for state in states:
            cur_b[state] = sum(prev_b[v] * trans_prob[state].get(v, 0) * emm_prob[v].get(observation, 0)
                               for v in states)
        betta[i] = cur_b
    return betta


def forward_backward(trans_prob, emm_prob, observations, states, start_prob, end_st):
    frwrd = forward(trans_prob, emm_prob, observations, states, start_prob)
    bcwrd = backward(trans_prob, emm_prob, observations, states, end_st)

    norm = sum(frwrd[-1][k] * trans_prob[k][end_st] for k in states)
    n = len(observations)
    posterior = [None] * n
    for i in range(n):
        posterior[i] = {
            state: frwrd[i][state] * bcwrd[i][state] / norm
            for state in states
        }

    return posterior


if __name__ == '__main__':
    states = ('Healthy', 'Fever')
    end_state = 'E'

    observations = ('normal', 'cold', 'dizzy')

    start_probability = {'Healthy': 0.6, 'Fever': 0.4}

    transition_probability = {
        'Healthy': {'Healthy': 0.69, 'Fever': 0.3, 'E': 0.01},
        'Fever': {'Healthy': 0.4, 'Fever': 0.59, 'E': 0.01},
    }

    emission_probability = {
        'Healthy': {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
        'Fever': {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6},
    }

    posterior = forward_backward(transition_probability,
                                 emission_probability,
                                 observations,
                                 states,
                                 start_probability,
                                 end_state)

    print(*posterior, sep='\n')
