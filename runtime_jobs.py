#!/usr/bin/env python3
"""Measure timing behavior for the single ECU case depending on number of jobs.
"""

import argparse
import random
import math
import time
# import numpy as np
import matplotlib.pyplot as plt
import utilities.generator_UUNIFAST as uunifast
import utilities.transformer as trans
import utilities.chain as ch
import utilities.analyzer as ana
import utilities.event_simulator as es
import main as main_file

debug_flag = False  # flag to have breakpoint() when errors occur

###
# Argument Parser
###
parser = argparse.ArgumentParser()

# number of the run:
parser.add_argument("-n", type=int, default=-1)

# number of tasks:
parser.add_argument("-t", type=int, default=50)

# number of tasks from list:
parser.add_argument("-tindex", type=int, default=-1)

# hyperperiod:
parser.add_argument("-p", type=int, default=2000)

# hyperperiod from list:
parser.add_argument("-pindex", type=int, default=-1)

# number of runs:
parser.add_argument("-r", type=int, default=1)

# job minimum:
parser.add_argument("-jobmin", type=int, default=-1)

# job maximum:
parser.add_argument("-jobmax", type=int, default=-1)

# flag to plot results in list:
# - j=1 hyperperiod on xaxis
# - j=2 number of tasks on xaxis
# --> for plotting, the max number of runs has to be specified by args.n
parser.add_argument("-j", type=int, default=0)

args = parser.parse_args()
del parser


###
# Main function.
###

def main():
    """Main Function."""

    try:
        ###
        # Plotting. (j == 1)
        ###
        if args.j == 1:
            if args.n == -1:
                print("ERROR: The number of runs is not specified.")
                return
            plot_results(args.n)
            return

        ###
        # Measurements. (j != 1)
        ###

        # Variables:
        periods_interval = [1, 20]
        num_runs = args.r
        jobmin = args.jobmin
        jobmax = args.jobmax

        if jobmax < jobmin:
            print("ERROR: jobmax has to be higher than jobmin.")
            return

        results = []

        # Variable to observe the number of tries.
        total_runs = 0

        while total_runs < num_runs:

            # Random values.
            utilization = random.randint(50, 90)  # random utilization in %
            num_tasks = random.randint(5, 20)  # random number of tasks

            ###
            # Task set generation.
            ###
            print("Task set generation.")
            task_sets_dic = uunifast.gen_tasksets(  # dictionary task sets
                num_tasks,
                1,
                periods_interval[0],
                periods_interval[1],
                utilization / 100.0,
                rounded=True)

            # Transform tasks to fit framework structure.
            accuracy = 10000000
            trans2 = trans.Transformer("0", task_sets_dic, accuracy)
            task_sets = trans2.transform_tasks(False)

            ###
            # Cause-effect chain generation.
            ###

            chain_len = 5  # number of tasks per chain
            task_set = task_sets[0]  # the task set under analysis

            if chain_len > len(task_set):
                print("ERROR: Not enough tasks for required chain length.")
                if debug_flag:
                    breakpoint()
                else:
                    return

            # Choose chain_len different tasks randomly and shuffled.
            ce_chain_as_list = random.sample(task_set, chain_len)

            # Transfer to ce-chain object.
            ce_chain = ch.CauseEffectChain(
                0,  # id of the chain
                ce_chain_as_list
            )

            ###
            # Time measurements.
            ###

            # Task and CE-chain Preperation.

            analyzer = ana.Analyzer("0")
            if TDA_check(task_set, analyzer) is False:  # check schedulability
                print("Task set not schedulable.")
                continue
            analyzer.davare([[ce_chain]])  # davare analysis for interval def

            # Job computation

            # Determination of the variables used to compute the stop
            # condition of the simulation
            max_e2e_latency = ce_chain.davare
            max_phase = 0  # by definition
            hyperperiod = analyzer.determine_hyper_period(task_set)
            max_period = hyperperiod  # by definition of task_set_dic

            sched_interval = (
                    2 * hyperperiod + max_phase  # interval from paper
                    + max_e2e_latency  # upper bound job chain length
                    + max_period)  # for convenience

            # Compute number of jobs.
            number_of_jobs = 0
            for task in task_set:
                number_of_jobs += sched_interval / task.period

            # Check if number of jobs is in the given range.
            if jobmin != -1 and number_of_jobs < jobmin:
                continue
            if jobmax != -1 and number_of_jobs > jobmax:
                continue

            # Information for end user.
            print("\tNumber of tasks: ", len(task_set))
            print("\tHyperperiod: ", hyperperiod / accuracy)
            print("\tNumber of jobs to schedule: ",
                  "%.2f" % number_of_jobs)

            # Start timer.
            tick = time.time()

            # # Event-based simulation.
            # print("Simulation.")
            # simulator = es.eventSimulator(task_set)
            #
            # # Stop condition: Number of jobs of lowest priority task.
            # simulator.dispatcher(
            #     int(math.ceil(sched_interval / task_set[-1].period)))
            #
            # # Simulation without early completion.
            # schedule = simulator.e2e_result()
            #
            # analyzer.reaction_our(schedule, task_set, ce_chain, max_phase,
            #                       hyperperiod)

            # main_file.our_mrt_mRda_lst([[ce_chain], task_set], [0.5])  # apply our analysis
            main_file.our_mrt_mRda_lst([[ce_chain], task_set], [0.5], calc_mda=False, calc_mrda=True)  # apply our analysis (only MRT(=E2E Latency due to equivalence) and MRDA)

            # Stop timer.
            tock = time.time()

            # Time difference.
            timing = tock - tick
            print(timing, 'seconds')

            results.append([number_of_jobs, timing])

            total_runs += 1
    except Exception as e:
        print(e)
        if debug_flag:
            breakpoint()
        else:
            return

    ###
    # Save data.
    ###

    try:
        dirname = "output/runtime/"
        filename = "result" + "_run_" + str(args.n) + ".pkl"
        main_file.check_or_make_directory(dirname)
        main_file.write_data(dirname + filename, results)
        # np.savez("output/runtime/result"
        #          + "_run_" + str(args.n)
        #          + ".npz",
        #          results=results)
    except Exception as e:
        print(e)
        print("ERROR: save")
        if debug_flag:
            breakpoint()
        else:
            return

    return


###
# Help functions.
###


def TDA_check(task_set, analyzer):
    """Check if all tasks meet their deadline."""
    for idx in range(len(task_set)):
        task_set[idx].rt = analyzer.tda(task_set[idx], task_set[:idx])
        if task_set[idx].rt > task_set[idx].deadline:
            return False
    return True


def plot_results(
        number):  # number of runs to collect data from
    """Plot the results."""
    try:
        ###
        # Load data.
        ###
        results = []
        for idx in range(number):
            filename = ("output/runtime/result"
                        + "_run_" + str(idx)
                        + ".pkl")
            results += list(main_file.load_data(filename))
            # data = np.load("output/runtime/result"
            #                + "_run_" + str(idx)
            #                + ".npz",
            #                allow_pickle=True)

            # results += list(data.f.results)

            # Close data file.
            # data.close()

    except Exception as e:
        print(e)
        print("ERROR: inputs for plotter are missing")
        if debug_flag:
            breakpoint()
        else:
            return

    ###
    # Plot result.
    ###
    draw_points(
        results,
        "output/runtime/runtime_jobs.pdf",
        xaxis_label="#Jobs",
        yaxis_label="Runtime [s]",
        convert=True)


def draw_points(
        results,
        filename,
        xaxis_label="",
        yaxis_label="",
        ylimits=None,  # [ylim_min, ylim_max]
        convert=False):
    """Draw given results as points."""

    # Convert list of results.
    if convert:
        results = list(zip(*results))

    # Size parameters:
    plt.rcParams.update({'font.size': 17})
    plt.rcParams.update({'figure.subplot.top': 0.99})
    plt.rcParams.update({'figure.subplot.bottom': 0.25})
    plt.rcParams.update({'figure.subplot.left': 0.18})
    plt.rcParams.update({'figure.subplot.right': 0.99})
    plt.rcParams.update({'figure.figsize': [7, 4.8]})

    # Draw plots:
    fig1, ax1 = plt.subplots()
    if ylimits is not None:
        ax1.set_ylim(ylimits)
    ax1.set_ylabel(yaxis_label, fontsize=22)

    plt.plot(results[0], results[1], 'o')

    ax1.set_xlabel(xaxis_label, fontsize=22)
    plt.tight_layout()

    # Save.
    plt.savefig(filename)


if __name__ == '__main__':
    main()
