class Agent(object):
    """Defines the interface that an agent playing an MDP should implement."""

    def __init__(self, gamma=1.0):
        """Initializes the agent, setting any relevant hyperparameters."""
        self.gamma = gamma
        self.current_discount = 1.0
        self.reward = 0.0

    def set_mdp(self, mdp):
        """Sets the MDP that the agent will be playing."""
        self.mdp = mdp

    def get_action(self, state):
        """Returns the action that the agent takes in the given state.

        The agent should imagine that it is in the given state when selecting an
        action. When the agent is actually acting in an environment, the
        environment will guarantee that it always passes in the current state of
        the agent. However, for other purposes, sequential calls to `get_action`
        are not required to be part of the same trajectory.

        state: State of the agent. An element of self.mdp.get_states().

        Returns: An action a such that a is in self.mdp.get_actions(state).
        """
        return self.get_action_distribution(state).sample()

    def get_action_distribution(self, state):
        """Returns a Distribution over actions that the agent takes in `state`.

        The agent should imagine that it is in the given state when selecting an
        action. When the agent is actually acting in an environment, the
        environment will guarantee that it always passes in the current state of
        the agent. However, for other purposes, sequential calls to
        `get_action_distribution` are not required to be part of the same
        trajectory.

        state: State of the agent. An element of self.mdp.get_states().

        Returns: A Distribution over actions.
        """
        raise NotImplementedError("get_action_distribution not implemented")

    def inform_minibatch(self, state, action, next_state, reward):
        """Updates the agent based on the results of the last action."""
        self.reward += self.current_discount * reward
        self.current_discount *= self.gamma
