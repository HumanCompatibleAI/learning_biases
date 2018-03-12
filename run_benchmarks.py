import subprocess as sp
import os

FLAGS = [
    # ['num_test',        [2000]],
    # ['num_train',       [5000]],
    # ['num_mdps',        [1000]],
    ['lr',              [0.025, 1e-4, 1e-3]],
    ['reward_lr',       [0.1, 0.01, 0.001]],
    ['epochs',          [30, 50, 70]],
    ['reward_epochs',   [50, 75]],
    ['k',               [10, 20, 30]],
    ['ch_h',            [150]],
    ['agent',           ['optimal','myopic','sophisticated','naive']],
    ['num_iters',       [50]],
    ['max_delay',       [5, 7]],
    ['hyperbolic_constant',     [0.9, 1.0]],
    # ['other_agent',     [None]], # more flags for other agent here
    ['algorithm',       ['given_rewards', 'no_rewards', 'boltzmann_planner', 'vi_inference']], # more here
    # ['action_distance_threshold', ['0.5']],
    ['reward_prob',     [0.05]],
    ['imsize',          [8,12,16]],
    # ['vin_regularizer_C',   [1e-4]],
    # ['reward_regularizer_C',    [1e-4]],
    ['model',           ['SIMPLE','VIN','VI']],
    ['seeds',           ['1,2,3,5,8,13,21,34', '89,714,10,1234,13,21,34,795', '1,2,3,4,5,75,86,907', '123,765,65,4234,5223,665,7234,897']],
    ['batchsize', [20]],
    ['use_gpu', ['True']]
]

INTERPRETER="/home/ngundotra/.conda/envs/IRL/bin/python"

def flag_generator(flags):
    """Generates (fname, string_to_run, csv_entry)"""
    if not flags:
        yield 'end.txt', "", ""
        raise StopIteration

    flag_name = flags[0][0]
    flag_values = flags[0][1]
    for value in flag_values:
        # Create relevant strings
        base_string_to_run = "--{0} {1}".format(flag_name, value)
        base_fname ='{0}-{1}'.format(flag_name, value)
        if flag_name=='seeds':
            base_csv = "-".join(value.split(','))
        else:
            base_csv = '{0}'.format(value)

        for subtuple in flag_generator(flags[1:]):
            # Concatenate the relevant examples from higher inputs
            fname = base_fname + '-' + subtuple[0]
            string_to_run = base_string_to_run + " " + subtuple[1]
            csv_entry = base_csv + ", " + subtuple[2]

            yield fname, string_to_run, csv_entry


def parse_proc(proc):
    if not proc.stdout:
        return ""

    try:
        relevant_lines = str(proc.stdout).split("\\n")[-3:-1]
        final_accuracy = relevant_lines[0].split("<1>")[1]
        performance = relevant_lines[1].split("<2>")[1]
        return final_accuracy, performance
    except Exception as e:
        print("Something went wrong while processing Popen object: {}".format(proc))
        print(e)


def run_benchmarks(interpreter, flags, dest='benchmark_data/'):
    """
    :param interpreter: path to relevant python executable
    :param flags: dictionary of flags: [benchmark_values]
    """

    if not os.path.isdir(dest):
        os.mkdir(dest)

    base_command = "{} train.py ".format(interpreter)

    count_calls = 0
    success = 0
    with open(os.path.join(dest,"index.csv"), 'w') as index_csv:
        for fname, str_to_run, csv_entry in flag_generator(flags):
            save_name = os.path.join(dest, fname)
            try:
                with open(save_name[-3:]+'err', 'w') as errtxt:
                    proc = sp.run(base_command + str_to_run, shell=True, stdout=sp.PIPE, stderr=errtxt, check=True)
                final_accuracy, performance = parse_proc(proc)
                success += 1
            except Exception as e:
                print("failed to run: {}".format(base_command+str_to_run))
                final_accuracy = "None"
                performance = "None"

            save_file = open(save_name, "w")
            save_file.write(", ".join([final_accuracy, performance]) + "\n")
            save_file.close()

            csv_entry = csv_entry + "{0}, {1}\n".format(final_accuracy, performance)
            index_csv.write(csv_entry)

            count_calls += 1

    print("{} out of {} calls were successful".format(success, count_calls))


if __name__ == '__main__':
    # for fname, str2run, csv_entry in flag_generator(flags_to_test):
    #     print("{1}".format(fname, str2run, csv_entry))

    run_benchmarks(INTERPRETER, FLAGS)
