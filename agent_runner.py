from gridworld import GridworldMdp, GridworldEnvironment, Direction
from agents import OptimalAgent
import numpy as np
import pdb

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

def run_agent_proxy(walls, proxy_reward, true_reward, agent="Optimal",episode_length=float("inf")):
    """Runs agent on a proxy environment for one episode, while collecting true reward from a separate environment

    walls: a 2D python list or numpy array of walls, with a starting spot
    proxy_reward: a 2D python list or numpy array of reward values
    true_reward: a 2D python list or numpy array of reward values
    agent: only optimal is implemented

    Creates a proxy mdp by overlaying walls onto proxy grid.
    True reward is summed if the reward grid's entry at the given state can be casted to a float
    
    Returns trajectory, sum of proxy reward, sum of true reward.
    """
    # This casts our reward values to floats, but I want to see if anything breaks..?
    walls = np.array(walls)
    proxy_reward = np.array(proxy_reward)
    true_reward = np.array(true_reward)

    # Create proxy grid which overrites proxy reward with walls
    proxy_grid = create_grid(walls, proxy_reward)
    pdb.set_trace()
    proxy_mdp = GridworldMdp(proxy_grid)
    env = GridworldEnvironment(proxy_mdp)
    env.reset()

    # Create agent
    if agent == "Optimal":
        agent = OptimalAgent()
    else:
        raise "Agent Not Implemented: use Optimal instead"

    agent.set_mdp(proxy_mdp)
    trajectory = []
    reward_sum = 0.0
    while len(trajectory) < episode_length and not env.is_done():
        curr_state = env.get_current_state()
        action = agent.get_action(curr_state)
        next_state, reward = env.perform_action(action)
        minibatch = (curr_state, action, next_state, reward)
        agent.inform_minibatch(*minibatch)
        trajectory.append(minibatch)
        
        try:
            rew = float(true_reward[curr_state])
            reward_sum += rew
        except Exception:
            pass
    proxy_sum = sum([reward for _, _, _, reward in trajectory])
    return trajectory, proxy_sum, reward_sum   

def create_grid(walls, reward):
    """Adds wall+reward array together to form Mdp-recognizable grid.
    
    Part 1 - 
    Add 'A' to a location in wall that is not occupied by a reward
    Part 2 - 
    Combine wall + reward

    wall: 2D python list or numpy array
    reward: 2D python list or numpy array

    Returns: 2D python list of walls
    """
    # Looks for start
    start_count = 0
    for row in walls[1:-1]: # Exclude walls
        for item in row[1:-1]: # Exclude walls
            if item == 'A':
                start_count += 1

    # Adds start to earliest point in grid and breaks out
    if start_count == 0:
        for wall_row, reward_row in zip(walls[1:-1], reward[1:-1]):
            for i in range(1,len(wall_row)-1):
                if reward_row[i] == 0:
                    wall_row[i] = 'A'
                    break

    # Create new reward array and overlay walls on top
    grid = np.copy(reward).astype(str)
    grid[(walls=='X') | (walls==1)] = 'X'
    grid[walls=='A'] = 'A'

    return grid
