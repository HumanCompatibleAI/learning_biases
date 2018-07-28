from agent_runner import run_agent
from gridworld import GridworldMdp
from gridworld_data import print_training_example
from mdp_interface import Mdp

from agents import NaiveTimeDiscountingAgent
import fast_agents

def main():
    grid = [
        ['X','X','X','X','X','X','X','X','X'],
        ['X',' ',-90,-90,-90,-90,'8',' ','X'],
        ['X','A',' ',' ',' ',' ',' ',' ','X'],
        ['X',' ',' ',' ',' ',' ',' ',' ','X'],
        ['X',' ',' ',-99,'2',' ',' ',' ','X'],
        ['X',' ',' ',' ',' ',' ',' ',' ','X'],
        ['X',' ','1',' ',' ',' ',' ',' ','X'],
        ['X','X','X','X','X','X','X','X','X']
    ]
    mdp = GridworldMdp(grid, living_reward=-0.01, noise=0.2)
    env = Mdp(mdp)
    opt = fast_agents.FastOptimalAgent(
        gamma=0.95, num_iters=20)
    naive = NaiveTimeDiscountingAgent(
        10, 1, gamma=0.95, num_iters=20)
    soph = fast_agents.FastSophisticatedTimeDiscountingAgent(
        10, 1, gamma=0.95, num_iters=20)
    myopic = fast_agents.FastMyopicAgent(
        6, gamma=0.95, num_iters=20)
    over = fast_agents.FastUncalibratedAgent(
        gamma=0.95, num_iters=20, calibration_factor=5)
    under = fast_agents.FastUncalibratedAgent(
        gamma=0.95, num_iters=20, calibration_factor=0.5)

    agents = [opt, naive, soph, myopic, over, under]
    names = ['Optimal', 'Naive', 'Sophisticated', 'Myopic', 'Overconfident', 'Underconfident']
    for name, agent in zip(names, agents):
        print('{} agent'.format(name))
        agent.set_mdp(mdp)
        trajectory = run_agent(agent, env, episode_length=50, determinism=True)
        if agent == naive:
            print([a for _, a, _, _ in trajectory])
        print_training_example(mdp, trajectory)
    print(opt.values.T)


if __name__ == '__main__':
    main()
