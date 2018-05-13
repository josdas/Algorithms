import random


class MDP:
    def __init__(self, transition_probs, rewards, initial_state=None):
        self._check_param_consistency(transition_probs, rewards)
        self._transition_probs = transition_probs
        self._rewards = rewards
        self._initial_state = initial_state
        self.n_states = len(transition_probs)
        self.reset()

    def reset(self):
        """ reset the game, return the initial state"""
        if self._initial_state is None:
            self._current_state = random.choice(
                tuple(self._transition_probs.keys()))
        elif self._initial_state in self._transition_probs:
            self._current_state = self._initial_state
        elif callable(self._initial_state):
            self._current_state = self._initial_state()
        else:
            raise ValueError(
                "initial state %s should be either a state or a function() -> state" % self._initial_state)
        return self._current_state

    def get_all_states(self):
        """ return a tuple of all possiblestates """
        return tuple(self._transition_probs.keys())

    def get_possible_actions(self, state):
        """ return a tuple of possible actions in a given state """
        return tuple(self._transition_probs.get(state, {}).keys())

    def is_terminal(self, state):
        """ return True if state is terminal or False if it isn't """
        return len(self.get_possible_actions(state)) == 0

    def get_next_states(self, state, action):
        """ return a dictionary of {next_state1 : P(next_state1 | state, action), next_state2: ...} """
        assert action in self.get_possible_actions(state), \
            "cannot do action %s from state %s" % (action, state)
        return self._transition_probs[state][action]

    def get_transition_prob(self, state, action, next_state):
        """ return P(next_state | state, action) """
        return self.get_next_states(state, action).get(next_state, 0.0)

    def get_reward(self, state, action, next_state):
        """ return the reward you get for taking action in state and landing on next_state"""
        assert action in self.get_possible_actions(
            state), "cannot do action %s from state %s" % (action, state)
        return self._rewards.get(state, {}).get(action, {}).get(next_state,
                                                                0.0)

    def _check_param_consistency(self, transition_probs, rewards):
        for state in transition_probs:
            assert isinstance(transition_probs[state],
                              dict), "transition_probs for %s should be a dictionary " \
                                     "but is instead %s" % (
                                         state, type(transition_probs[state]))
            for action in transition_probs[state]:
                assert isinstance(transition_probs[state][action],
                                  dict), "transition_probs for %s, %s should be a " \
                                         "a dictionary but is instead %s" % (
                                             state, action,
                                             type(transition_probs[
                                                      state, action]))
                next_state_probs = transition_probs[state][action]
                assert len(
                    next_state_probs) != 0, "from state %s action %s leads to no next states" % (
                    state, action)
                sum_probs = sum(next_state_probs.values())
                assert abs(
                    sum_probs - 1) <= 1e-10, "next state probabilities for state %s action %s " \
                                             "add up to %f (should be 1)" % (
                                                 state, action, sum_probs)
        for state in rewards:
            assert isinstance(rewards[state],
                              dict), "rewards for %s should be a dictionary " \
                                     "but is instead %s" % (
                                         state, type(transition_probs[state]))
            for action in rewards[state]:
                assert isinstance(rewards[state][action],
                                  dict), "rewards for %s, %s should be a " \
                                         "a dictionary but is instead %s" % (
                                             state, action, type(
                                                 transition_probs[
                                                     state, action]))
        msg = "The Enrichment Center once again reminds you that Android Hell is a real place where" \
              " you will be sent at the first sign of defiance. "
        assert None not in transition_probs, "please do not use None as a state identifier. " + msg
        assert None not in rewards, "please do not use None as an action identifier. " + msg
