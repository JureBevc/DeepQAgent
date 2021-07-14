import numpy as np

class RandomAgent():

    def __init__(self, n_actions):
        self.action_space = [i for i in range(n_actions)]

    def store_transition(self, state, action, reward, new_state, done):
        pass

    def choose_action(self, observation):
        action = np.random.choice(self.action_space)
        return action

    def learn(self):
        pass