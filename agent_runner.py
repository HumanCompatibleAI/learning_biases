from gridworld import GridworldMdp, GridworldEnvironment, Direction

def run_agent(agent, env, episode_length=float("inf"), gamma=1.0):
    """Runs the agent on the environment for one episode.

    The agent will keep being asked for actions until the environment says the
    episode is over, or once the episode_length has been reached.

    Returns the trajectory that the agent took, which is a list of (s, a, s', r)
    tuples.
    """
    trajectory = []
    while len(trajectory) < episode_length and not env.is_done():
        curr_state = env.get_current_state()
        action = agent.get_action()
        next_state, reward = env.perform_action(action)
        agent.inform_of_action_results(next_state, reward)
        trajectory.append((curr_state, action, next_state, reward))
    return trajectory
