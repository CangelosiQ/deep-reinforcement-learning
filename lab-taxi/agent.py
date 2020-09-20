import numpy as np
from collections import defaultdict

from RL_library.temporal_difference_control import TemporalDifferences


class Agent:

    def __init__(self, alpha, gamma=1, nA=6, n_episodes=20000):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.policy = {}
        self.epsilon = 1
        self._episode = 1
        self.alpha = alpha
        self.gamma = gamma

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        action = TemporalDifferences.greedy_policy(state, self.policy, self.nA, self.epsilon)
        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if done:
            if state not in self.Q:
                self.Q[state][action] = 0
            self.Q[state][action] = self.Q[state][action] + self.alpha * (reward - self.Q[state][action])
            return

        # SARSA Max takes best action
        stochastic_policy = TemporalDifferences.get_stochastic_policy(self.policy, state, self.nA, self.epsilon)
        expected_return_next_state = np.dot(self.Q[next_state], stochastic_policy) if next_state in self.Q else 0

        # Update Q
        # Every visit
        if state not in self.Q:
            self.Q[state][action] = 0

        self.Q[state][action] = self.Q[state][action] + self.alpha * (
                reward + self.gamma * expected_return_next_state - self.Q[state][action])

        # Update policy
        self.policy[state] = np.argmax(self.Q[state])

        self._episode += 1
        self.epsilon = 1 / self._episode

