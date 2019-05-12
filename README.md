_Last updated Oct 12, 2017_

# planner-inference
Infer how suboptimal agents are suboptimal while planning, for example if they
are hyperbolic time discounters. Use this to do better inverse reinforcement
learning.

## Code

Note that for **SIMPLE** baseline only gridsizes of 8 & 14 work.

To run benchmark testing, run `python run_benchmarks.py --low LOW --high HIGH` etc.


### Gridworlds

`gridworld.py`: Implements the Gridworld MDP, which is used for simple
experiments.

`gridworld_data.py`: Generates example gridworlds, runs agents on the gridworlds
to generate trajectories, and collects all of the trajectories and puts them
into training and test sets used for learning.

### Agents

`agent_interface.py`: Defines the interface that agents should follow.

`agent_runner.py`: Defines `run_agent`, which given an agent and an environment,
runs the agent in the environment, producing a trajectory.

`agents.py`: Defines many different agents that can play tabular MDPs. Currently
the agents are using value iteration like approaches.

### Value Iteration Networks

The code here is taken from [Tensorflow
VINs](https://github.com/TheAbhiKumar/tensorflow-value-iteration-networks) with
a few edits.

`model.py`: Implementation of VIN and VIN with untied weights.

`train.py`: Trains a VIN using gridworld data.

### Other

`disjoint_sets.py`: An implementation of the disjoint sets data structure, used in `gridworld_data.py` to generate interesting grid worlds.

`utils.py`: Utility functions.

### Testing

All of the `*_tests.py` contain tests for the corresponding `*.py` file. All of
the tests can be run using:

    ./run_tests.sh

You may need to first give it execute permissions:

    chmod +x run_tests.sh
