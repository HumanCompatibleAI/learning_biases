class Agent(object):
    def __init__(self, gamma=1.0):
        self.gamma = gamma
        self.current_discount = 1.0
        self.reward = 0.0

    def set_mdp(self, mdp):
        self.mdp = mdp

    def get_action(self, state):
        raise NotImplentedError("get_action not implemented")

    def inform_minibatch(self, state, action, next_state, reward):
        self.reward += self.current_discount * reward
        self.current_discount *= self.gamma
