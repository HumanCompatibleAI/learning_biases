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
     ['X',10,' ',' ',0.9,0.9, 10,'X'],
     ['X',-9,'X','X','X','X','X','X'],
     ['X',' ','X',' ',' ',' ',' ','X'],
     ['X',' ','X',' ',8,  8,' ','X'],
     ['X',' ',' ',' ',' ',' ',' ','X'],
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


def problem_description():
# for i, ax in enumerate(axes[idx]):
#     if i == 0:
#         # Set agent title
#         ax.set_title(agent_names[idx])
#     elif i == 1:
#         plot_reward(reward, walls, '', fig=fig, ax=ax)
#     elif i == 2:
#         plot_reward(np.zeros(reward.shape), walls, '', fig=fig, ax=ax)
#         plot_trajectory(walls, reward, start, OptimalAgent(), fig=fig, ax=ax)
# fig.subplots_adjust(hspace=0.5)
    pass


def show_agents(grids, agent_list, agent_names, grid_names, filename='AgentComparison', figtitle=''):
    """Shows how agents perform on a gridworld

    grid - list of gridworlds (see examples in earlier part of file)
    agent_list - list of agent (objects)
    agent_names - names of agents (strings)
    """
    num_ex = len(agent_list)
    num_grids = len(grids)
    fig, axes_grid = plt.subplots(num_grids, num_ex + 1)

    if num_grids == 1:
        axes_grid = [axes_grid]

    for i, axes in enumerate(axes_grid):
        # Plot reward on first ax
        ax = axes[0]
        ax.set_aspect('equal')
        ax.set_ylabel(grid_names[i])
        # Generate MDP
        grid = grids[i]
        mdp = GridworldMdp(grid)
        walls, reward, start = mdp.convert_to_numpy_input()
        plot_reward(reward, walls, '', fig=fig, ax=ax)
        # Only write Agent names if it's the first row
        if i == 0:
            ax.set_title("Reward")

        # Plot agents on remaining axes
        axes = axes[1:]
        for idx, agent in enumerate(agent_list):
            ax = axes[idx]
            ax.set_aspect('equal')

            plot_reward(reward, walls, '', fig=fig, ax=ax)
            plot_trajectory(walls, reward, start, agent, fig=fig, ax=ax)
            # Only write Agent names if it's the first row
            if i == 0:
                ax.set_title(agent_names[idx])

        # Increase vertical space btwn subplots
    fig.subplots_adjust(hspace=0.1)
    fig.suptitle(figtitle)
    fig.savefig(filename)
    print("Saved figure to {}.png".format(filename))

if __name__ == "__main__":
    # config = init_flags()
    # agent, _ = create_agents_from_config(config)
    from fast_agents import FastMyopicAgent as Myopic, \
        FastNaiveTimeDiscountingAgent as Naive,\
        FastSophisticatedTimeDiscountingAgent as Sophisticated
    kwargs = {'max_delay': 10,
              'discount_constant': 0.9}
    agent_list = [OptimalAgent(),
                  Naive(**kwargs),
                  Sophisticated(**kwargs),
                  Myopic(horizon=10),
                  ]
    agent_names = ['Optimal',
                   'Naive',
                   'Sophisticated',
                   'Myopic',
                   ]

    # Choose grid
    grid = grids[1]
    grid_titles = ["Easy", "Medium", "Hard", "Bonus"]
    # Show how agents perform on that grid
    show_agents(grids, agent_list, agent_names, grid_titles)
