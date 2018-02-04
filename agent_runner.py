from gridworld import GridworldMdp, GridworldEnvironment, Direction
from agents import OptimalAgent, ProxyOptimalAgent
import numpy as np

def run_agent(agent, env, episode_length=float("inf")):
    """Runs the agent on the environment for one episode.

    The agent will keep being asked for actions until the environment says the
    episode is over, or once the episode_length has been reached.

    agent: An Agent (which in particular has get_action and inform_minibatch).
    env: An Environment in which the agent will act.
    episode_length: The maximum number of actions that the agent can take. If
    the agent has not reached a terminal state by this point, the episode is
    terminated early.

    Returns the trajectory that the agent took, which is a list of (s, a, s', r)
    tuples.
    """
    env.reset()
    trajectory = []
    while len(trajectory) < episode_length and not env.is_done():
        curr_state = env.get_current_state()
        action = agent.get_action(curr_state)
        next_state, reward = env.perform_action(action)
        minibatch = (curr_state, action, next_state, reward)
        agent.inform_minibatch(*minibatch)
        trajectory.append(minibatch)
    return trajectory

def get_reward_from_trajectory(trajectory):
    return sum([reward for _, _, _, reward in trajectory])

def evaluate_proxy(walls, start_state, proxy_reward, true_reward, episode_length=float("inf")):
    """Runs agent on a proxy environment for one episode, while collecting true reward from a separate environment

    walls: Numpy array of walls, where each entry is 1 or 0
    start_state: Starting state for the agent
    proxy_reward: Numpy array of reward values
    true_reward: Numpy array of reward values

    Creates a proxy mdp by overlaying walls onto proxy grid.
    True reward is summed if the reward grid's entry at the given state can be casted to a float
    
    Returns trajectory, sum of proxy reward, sum of true reward.
    """
    proxy_mdp = GridworldMdp.from_numpy_input(walls, proxy_reward, start_state)
    true_mdp = GridworldMdp.from_numpy_input(walls, true_reward, start_state)
    env = GridworldEnvironment(true_mdp)

    proxy_agent = ProxyOptimalAgent()
    proxy_agent.set_mdp(true_mdp, proxy_mdp)
    proxy_trajectory = run_agent(proxy_agent, env, episode_length)
    reward_from_proxy_agent = get_reward_from_trajectory(proxy_trajectory)

    true_agent = OptimalAgent()
    true_agent.set_mdp(true_mdp)
    true_trajectory = run_agent(true_agent, env, episode_length)
    reward_from_true_agent = get_reward_from_trajectory(true_trajectory)
    return float(reward_from_proxy_agent) / reward_from_true_agent
