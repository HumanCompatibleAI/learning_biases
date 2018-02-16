from train import run_algorithm

# We could call init_flags() from utils.py instead of creating our own config,
# but I want the specific parameters to be duplicated here, so that changes to
# flags in utils.py don't affect this experiment.
class Config(object):
    def __init__(self):
        # Parameters that we vary
        self.algorithm = None
        self.agent = None
        self.seeds = None
        # Parameters that we don't vary
        self.em_iterations = 2
        self.simple_mdp = False
        self.imsize = 8
        self.wall_prob = 0.05
        self.reward_prob = 0.05
        self.num_train = 5000
        self.num_test = 2000
        self.num_mdps = 1000
        self.model = 'VIN'
        self.vin_regularizer_C = 0.0001
        self.reward_regularizer_C = 0
        self.lr = 0.01
        self.reward_lr = 0.1
        self.epochs = 10
        self.reward_epochs = 20
        self.k = 10
        self.ch_h = 150
        self.ch_q = 5
        self.num_actions = 5
        self.batchsize = 20
        self.gamma = 0.9
        self.beta = 1.0  # If changed, change the beta in run_algorithm in train.py
        self.num_iters = 50
        self.max_delay = 10
        self.hyperbolic_constant = 1.0
        self.other_agent = None
        self.display_step = 5
        self.log = False
        self.verbosity = 2
        self.plot_rewards = False

def run_one(config, algorithm, agent, seed_lists):
    print('Running algorithm {} on data from agent {}'.format(algorithm, agent))
    percents = []
    for seed_list in seed_lists:
        config.algorithm = algorithm
        config.agent = agent
        config.seeds = list(seed_list)
        print('Using seeds {}'.format(seed_list))
        percents.append(run_algorithm(config))
    print(percents)
    average_percent = float(sum(percents)) / len(percents)
    result_string = 'Algorithm {} on agent {} achieves {}%'.format(
        algorithm, agent, average_percent * 100)
    print(result_string)
    return result_string

def run_all():
    config = Config()
    algorithms = ['given_rewards', 'optimal_planner']
    agents = ['optimal']
    seed_lists = [(1, 2, 3, 5, 8, 13, 21, 34)]
    results = []
    for algorithm in algorithms:
        for agent in agents:
            results.append(run_one(config, algorithm, agent, seed_lists))

    with open('experiment_results.txt', 'a') as f:
        f.write('RESULTS:')
        f.write('\n'.join(results))

if __name__=='__main__':
    run_all()
