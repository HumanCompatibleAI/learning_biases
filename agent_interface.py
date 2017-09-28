class Agent:
    def __init__(self, mdp, gamma=1.0):
        self.mdp = mdp
        self.state = mdp.get_start_state()
        self.gamma = gamma
        self.current_discount = 1.0
        self.reward = 0.0

    def get_action(self):
        raise NotImplentedError("get_action not implemented")

    def inform_of_action_results(self, next_state, reward):
        self.state = next_state
        self.reward += self.current_discount * reward
        self.current_discount *= self.gamma
