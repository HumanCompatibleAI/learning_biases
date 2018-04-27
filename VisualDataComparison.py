"""
Generates 2 figures that represent the process being performed.
1) Trajectories of an agent on `N` grids
2) Corresponding inferred rewards of an optimal agent on the same grid

Possible extensions: add 1a) figure which visualizes our synthesized trajectories
"""
import utils
from utils import plot_trajectory, plot_reward, init_flags
from agents import OptimalAgent
from gridworld_data import create_agents_from_config
from gridworld import GridworldMdp

import matplotlib.pyplot as plt

grids = [
    [['X','X','X','X','X','X','X','X'],
     ['X',0.5,' ',' ',' ',' ',  4,'X'],
     ['X',' ',' ',' ',' ',' ',' ','X'],
     ['X',' ',' ',' ',' ',' ',' ','X'],
     ['X',' ',' ',' ',' ',' ',' ','X'],
     ['X',' ',' ',' ',' ',' ',' ','X'],
     ['X','A',' ',' ',' ',' ',' ','X'],
     ['X','X','X','X','X','X','X','X'],
    ],
    [['X','X','X','X','X','X','X','X'],
     ['X',' ',' ',' ',' ','X',  5,'X'],
     ['X',' ','X','X',' ',' ',' ','X'],
     ['X',' ',' ',' ','X','X','X','X'],
     ['X',' ','X',' ','X',' ',0.5,'X'],
     ['X','X','X',' ','X',' ',' ','X'],
     ['X','A',' ',' ',' ',' ',' ','X'],
     ['X','X','X','X','X','X','X','X'],
    ],
    [['X','X','X','X','X','X','X','X'],
     ['X',0.1,' ',' ',' ',' ', 10,'X'],
     ['X',' ','X','X','X','X','X','X'],
     ['X',' ','X',' ',' ',' ',' ','X'],
     ['X',' ','X',' ',' ',' ',' ','X'],
     ['X',' ','X',' ',' ',' ',' ','X'],
     ['X','A','X',' ',' ',' ',' ','X'],
     ['X','X','X','X','X','X','X','X'],
     ],
]

if __name__ == "__main__":
    config = init_flags()
    agent, _ = create_agents_from_config(config)

    num_ex = len(grids)
    fig, axes = plt.subplots(1, num_ex)

    fname = 'biasedtrajectories'

    for i, grid in enumerate(grids):
        mdp = GridworldMdp(grid)
        walls, reward, start = mdp.convert_to_numpy_input()
        axes[i].set_aspect('equal')
        plot_trajectory(walls, reward, start, agent, fig=fig, ax=axes[i])
        plot_reward(reward, walls, '', fig=fig, ax=axes[i])

    fig.suptitle("Biased Trajectories")
    fig.savefig(fname)
