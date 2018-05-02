"""
Generates 2 figures that represent the process being performed.
1) Trajectories of an agent on `N` grids
2) Corresponding inferred rewards of an optimal agent on the same grid

Possible extensions: add 1a) figure which visualizes our synthesized trajectories
"""
import numpy as np
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
     ['X',' ','X',' ','X',' ',  1,'X'],
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
    [['X','X','X','X','X','X','X','X'],
     ['X',' ',' ',' ',' ','X',  1,'X'],
     ['X',' ',' ',' ',' ',' ',' ','X'],
     ['X',' ',' ','X','X','X','X','X'],
     ['X',0.5,' ','X','X',' ',0.5,'X'],
     ['X','X',' ','X','X',' ',' ','X'],
     ['X','A',' ',' ',0.3,' ',' ','X'],
     ['X','X','X','X','X','X','X','X'],
     ]
]

if __name__ == "__main__":
    config = init_flags()
    agent, _ = create_agents_from_config(config)

    num_ex = len(grids)
    fig, axes = plt.subplots(3, num_ex)
    axes = axes.T

    fname = 'biasedtrajectories'

    for i, grid in enumerate(grids):
        mdp = GridworldMdp(grid)
        walls, reward, start = mdp.convert_to_numpy_input()
        for i, ax in enumerate(axes[i]):
            ax.set_aspect('equal')
            if i == 0:
                plot_trajectory(walls, reward, start, agent, fig=fig, ax=ax)
                plot_reward(np.zeros(reward.shape), walls, '', fig=fig, ax=ax)
            elif i == 1:
                plot_reward(reward, walls, '', fig=fig, ax=ax)
            elif i == 2:
                plot_trajectory(walls, reward, start, OptimalAgent(), fig=fig, ax=ax)
                plot_reward(np.zeros(reward.shape), walls, '', fig=fig, ax=ax)

    # Set agent title
    agent_name = config.agent
    axes[0,0].set_ylabel(agent_name)
    axes[0,1].set_ylabel("Reward")
    axes[0,2].set_ylabel("Optimal")

    # Increase vertical space btwn subplots
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle("Biased Trajectories")
    fig.savefig(fname)
    print("Saved figure to {}.png".format(fname))
