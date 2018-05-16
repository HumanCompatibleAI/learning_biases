"""
Generates 2 figures that represent the process being performed.
1) Trajectories of an agent on `N` grids
2) Corresponding inferred rewards of an optimal agent on the same grid

Possible extensions: add 1a) figure which visualizes our synthesized trajectories
"""
import numpy as np
import utils
from utils import plot_trajectory, plot_reward, plot_policy, init_flags, set_seeds
from agents import OptimalAgent
from gridworld import GridworldMdp

import matplotlib.pyplot as plt
import seaborn as sns
# sns.set(rc={'text.usetex' : True})
grids = [
    # [['X','X','X','X','X','X','X','X','X','X','X','X','X','X','X','X'],
    #  ['X',' ',' ','A',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ','X'],
    #  ['X',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ','X'],
    #  ['X',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ','X'],
    #  ['X',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ','X'],
    #  ['X',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ','X'],
    #  ['X',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ', -1,' ',' ',' ','X'],
    #  ['X',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ', -1,' ',' ',' ','X'],
    #  ['X',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ', -1,' ',' ',' ','X'],
    #  ['X',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ', -1,' ',' ',' ','X'],
    #  ['X',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ', -1,' ',' ',' ','X'],
    #  ['X',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ', -1,' ',' ',' ','X'],
    #  ['X',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ', -1,' ',' ',' ','X'],
    #  ['X',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ', -1,' ',' ',' ','X'],
    #  ['X',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',  5,' ',' ','X'],
    #  ['X',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ','X'],
    #  ['X','X','X','X','X','X','X','X','X','X','X','X','X','X','X','X'],
    #  ],
    # [['X','X','X','X','X','X','X','X','X','X','X','X','X','X','X','X'],
    #  ['X',' ',' ',' ',' ',' ','A',' ',' ',' ',' ','X',' ',' ',' ','X'],
    #  ['X',' ',' ','X',' ',' ',' ',' ',' ',' ',' ','X',' ',' ',' ','X'],
    #  ['X',' ',' ','X',' ','X','X','X','X','X',' ','X',' ',' ',' ','X'],
    #  ['X',' ',' ',  1,' ','X',' ',' ',' ',' ',' ','X',' ',' ',' ','X'],
    #  ['X',' ',' ','X',' ','X',' ','X','X','X','X','X',' ',' ',' ','X'],
    #  ['X',' ',' ','X',' ','X',' ',' ',' ',' ',' ',' ',' ',' ',' ','X'],
    #  ['X',' ',' ','X',' ','X',' ',' ',' ',' ',' ',' ',' ',' ',' ','X'],
    #  ['X',' ',' ','X',' ','X',' ',' ',' ',' ',' ',' ',' ',' ',' ','X'],
    #  ['X',' ',' ','X',' ','X',' ',' ',' ',' ',' ',' ',' ',' ',' ','X'],
    #  ['X',' ',' ','X',' ','X',' ',' ',' ',' ',' ',' ',' ',' ',' ','X'],
    #  ['X',' ',' ','X',' ','X',' ',' ',' ',' ',' ',' ',' ',' ',' ','X'],
    #  ['X',' ',' ','X',' ','X',' ',' ',' ',' ',' ',' ',' ',' ',' ','X'],
    #  ['X',' ',' ','X',' ',' ',' ',' ', 10,' ',' ',' ',' ',' ',' ','X'],
    #  ['X',' ',' ','X','X','X','X','X','X',' ',' ',' ',' ',' ',' ','X'],
    #  ['X',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ','X'],
    #  ['X','X','X','X','X','X','X','X','X','X','X','X','X','X','X','X'],
    #  ],
    # [['X','X','X','X','X','X','X','X','X','X','X','X','X','X','X','X'],
    #  ['X',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ','X','A',' ',' ','X'],
    #  ['X',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ','X',' ',' ',' ','X'],
    #  ['X',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ','X',' ',' ',' ','X'],
    #  ['X',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ','X',' ',' ',' ','X'],
    #  ['X',' ',' ',' ',' ',' ',' ','X','X','X','X','X',' ',' ',' ','X'],
    #  ['X',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ', -5, -5,'X'],
    #  ['X',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ', -5,' ','X'],
    #  ['X',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ', -5,  1,'X'],
    #  ['X',' ',' ',' ',' ',' ',' ','X','X','X','X','X','X',' ',' ','X'],
    #  ['X',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ','X'],
    #  ['X',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ','X'],
    #  ['X',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ','X'],
    #  ['X',' ',' ',' ',' ',' ',' ',' ', 10,' ',' ',' ',' ',' ',' ','X'],
    #  ['X',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ','X'],
    #  ['X',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ','X'],
    #  ['X','X','X','X','X','X','X','X','X','X','X','X','X','X','X','X'],
    #  ],
    # [['X','X','X','X','X','X','X','X'],
    #  ['X','X','X',' ',' ', 10,' ','X'],
    #  ['X','X',  5,' ','X','X',' ','X'],
    #  ['X','X','X',' ','X','X',' ','X'],
    #  ['X','X','X',' ','X','X',' ','X'],
    #  ['X',  5,' ',' ',' ',' ',' ','X'],
    #  ['X','X','X','A',' ',' ',  5,'X'],
    #  ['X','X','X','X','X','X','X','X'],
    # ],
    # [['X','X','X','X','X','X','X','X'],
    #  ['X',' ',' ',' ',' ','X',  5,'X'],
    #  ['X',' ','X',' ',' ',' ',' ','X'],
    #  ['X',  3,' ',' ','X','X','X','X'],
    #  ['X',' ','X',' ','X',' ',  1,'X'],
    #  ['X','X','X',' ','X',' ',' ','X'],
    #  ['X','A',' ',' ',' ',' ',' ','X'],
    #  ['X','X','X','X','X','X','X','X'],
    # ],
    # [['X','X','X','X','X','X','X','X'],
    #  ['X','X','X', 10,'X','X','X','X'],
    #  ['X','X',' ',' ',' ',' ','X','X'],
    #  ['X',  7,' ','X','X',' ',  4,'X'],
    #  ['X','X',' ','X',  3,' ','X','X'],
    #  ['X','X',' ','X','X',' ','X','X'],
    #  ['X','X',' ',' ',' ','A','X','X'],
    #  ['X','X','X','X','X','X','X','X'],
    #  ],
    [['X','X','X','X','X','X','X','X'],
     ['X', 10,  1,' ',' ','X', -1,'X'],
     ['X',' ',  1,' ',' ',' ',' ','X'],
     ['X','X','X',' ','X','X','X','X'],
     ['X',' ',' ',' ','X',' ',' ','X'],
     ['X',  2,' ',' ','X', -5,' ','X'],
     ['X','A',' ',' ',' ',' ',' ','X'],
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


def get_policy(self, agent, grid):
    """Returns the policy of the agent given"""
    from gridworld import GridworldMdp, Direction
    from utils import Distribution

    num_actions = len(Direction.ALL_DIRECTIONS)

    mdp = GridworldMdp(grid=grid)
    agent.set_mdp(mdp)

    def dist_to_numpy(dist):
        return dist.as_numpy_array(Direction.get_number_from_direction, num_actions)

    def action(state):
        # Walls are invalid states and the MDP will refuse to give an action for
        # them. However, the VIN's architecture requires it to provide an action
        # distribution for walls too, so hardcode it to always be STAY.
        x, y = state
        if mdp.walls[y][x]:
            return dist_to_numpy(Distribution({Direction.STAY : 1}))
        return dist_to_numpy(agent.get_action_distribution(state))

    imsize = len(grid)

    # I think it's this line that's wrong. Writing as (y, x) gives expected SVF vec
    action_dists = [[action((x, y)) for y in range(imsize)] for x in range(imsize)]
    action_dists = np.array(action_dists)
    return action_dists


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
        # Give each gridworld a name (uncomment to do so)
        # ax.set_ylabel(grid_names[i])
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
            plot_trajectory(walls, reward, start, agent, arrow_width=0.35, fig=fig, ax=ax)
            # Only write Agent names if it's the first row
            if i == 0:
                ax.set_title(agent_names[idx])

        # Increase vertical space btwn subplots
    fig.subplots_adjust(hspace=0.2)
    fig.suptitle(figtitle)
    fig.savefig(filename, bbox_inches='tight', dpi=200)
    print("Saved figure to {}.png".format(filename))


def random_gridworld_plot(agent, size, filename='RandomGrid'):
    """Plots random gridworld"""
    from gridworld import Direction
    from utils import Distribution
    if agent is None:
        raise ValueError("agent cannot be None")

    num_R = 5
    mdp = GridworldMdp.generate_random_connected(size, size, num_R, noise=0)

    walls, reward, start = mdp.convert_to_numpy_input()

    def get_policy():
        num_actions = 5
        imsize = len(walls)

        def dist_to_numpy(dist):
            return dist.as_numpy_array(Direction.get_number_from_direction, num_actions)

        def action(state):
            # Walls are invalid states and the MDP will refuse to give an action for
            # them. However, the VIN's architecture requires it to provide an action
            # distribution for walls too, so hardcode it to always be STAY.
            x, y = state
            if mdp.walls[y][x]:
                return dist_to_numpy(Distribution({Direction.STAY : 1}))
            return dist_to_numpy(agent.get_action_distribution(state))

        agent.set_mdp(mdp)
        action_dists = [[action((x, y)) for x in range(imsize)] for y in range(imsize)]
        action_dists = np.array(action_dists)
        return action_dists

    fig, axes = plt.subplots(1, 1)
    fig.set_size_inches(5, 5)

    # Reward only
    plot_reward(reward, np.zeros_like(walls), fig=fig, ax=axes, ax_title='')
    fig.savefig(filename+'R', bbox_inches='tight', dpi=100)

    # Walls only
    plot_reward(np.zeros_like(reward), walls, fig=fig, ax=axes, ax_title='')
    fig.savefig(filename+'W', bbox_inches='tight', dpi=100)

    # Trajectory + Walls + Rewards
    plot_reward(reward, walls, fig=fig, ax=axes, ax_title='')
    # plot_trajectory(walls, reward, start, agent, fig=fig, ax=axes)
    policy = get_policy()
    plot_policy(walls, policy, fig=fig, ax=axes)
    fig.savefig(filename+'T', bbox_inches='tight', dpi=100)


if __name__ == "__main__":
    # config = init_flags()
    # agent, _ = create_agents_from_config(config)
    set_seeds(0)
    from fast_agents import FastMyopicAgent as Myopic, \
        FastNaiveTimeDiscountingAgent as Naive,\
        FastSophisticatedTimeDiscountingAgent as Sophisticated,\
        FastUncalibratedAgent as Uncalibrated
    kwargs = {'gamma': 0.95,
              'max_delay': 10,
              'discount_constant': 0.9}
    agent_list = [OptimalAgent(gamma=0.95),
                  Naive(**kwargs),
                  Sophisticated(**kwargs),
                  Myopic(gamma=0.95, horizon=10),
                  Uncalibrated(gamma=0.95, calibration_factor=5),
                  Uncalibrated(gamma=0.95, calibration_factor=0.5)
                  ]
    agent_names = ['Optimal',
                   'Naive',
                   'Sophisticated',
                   'Myopic',
                   'Overconfident',
                   'Underconfident'
                   ]
    
    # Choose grid
    # grid = grids[1]
    grid_titles = ["Easy", "Medium", "Hard", "Bonus"]
    # Show how agents perform on that grid
    # show_agents(grids, agent_list, agent_names, grid_titles, figtitle='', filename='AgentComparison')

    for i in range(3):
        random_gridworld_plot(OptimalAgent(), 10, filename='random/RandomGrid-{}'.format(i))
