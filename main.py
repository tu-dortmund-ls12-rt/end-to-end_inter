#!/usr/bin/env python3
"""Evaluation for the paper TODO .

It includes (1) local analysis (2) global analysis and (3) plotting of the
results.
"""

import gc  # garbage collector
import argparse
import getopt
import math
import re

import numpy as np
import utilities.chain as c
import utilities.communication as comm
import utilities.generator_WATERS as waters
import utilities.generator_UUNIFAST as uunifast
import utilities.transformer as trans
import utilities.event_simulator as es
import utilities.analyzer as a
import utilities.analyzer_our as a_our
import utilities.evaluation as eva
import utilities.task as task_package
import utilities.plot as plot

import time
import sys
import os
import pickle  # save and load

import random  # randomization
from multiprocessing import Pool  # multiprocessing
import itertools  # better performance

debug_flag = True  # flag to have breakpoint() when errors occur


########################
# Some help functions: #
########################

def time_now():
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    return current_time


def check_or_make_directory(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print(f'Directory {dirname} created')


def write_data(filename, data):
    file = open(filename, 'wb')
    pickle.dump(data, file)
    file.close()
    print(f'Data written to {filename}')


def load_data(filename):
    file = open(filename, 'rb')
    data = pickle.load(file)
    file.close()
    print(f'Data loaded from {filename}')
    return data


def task_set_generate(argsg, argsu, argsr):
    '''Generates task sets.
    Input:
    - argsg = benchmark to choose
    - argsu = utilization in %
    - argsr = number of task sets to generate
    Output: list of task sets.'''
    try:
        if argsg == 'waters':
            # WATERS benchmark
            print("WATERS benchmark.")

            # Statistical distribution for task set generation from table 3
            # of WATERS free benchmark paper.
            profile = [0.03 / 0.85, 0.02 / 0.85, 0.02 / 0.85, 0.25 / 0.85,
                       0.25 / 0.85, 0.03 / 0.85, 0.2 / 0.85, 0.01 / 0.85,
                       0.04 / 0.85]
            # Required utilization:
            req_uti = argsu / 100.0
            # Maximal difference between required utilization and actual
            # utilization is set to 1 percent:
            threshold = 1.0

            # Create task sets from the generator.
            # Each task is a dictionary.
            print("\tCreate task sets.")
            task_sets_waters = []
            while len(task_sets_waters) < argsr:
                task_sets_gen = waters.gen_tasksets(
                    1, req_uti, profile, True, threshold / 100.0, 4)
                task_sets_waters.append(task_sets_gen[0])

            # Transform tasks to fit framework structure.
            # Each task is an object of utilities.task.Task.
            trans1 = trans.Transformer("1", task_sets_waters, 10000000)
            task_sets = trans1.transform_tasks(False)

        elif argsg == 'uunifast':
            # UUniFast benchmark.
            print("UUniFast benchmark.")

            # Create task sets from the generator.
            print("\tCreate task sets.")

            # The following can be used for task generation with the
            # UUniFast benchmark without predefined periods.

            # # Generate log-uniformly distributed task sets:
            # task_sets_generator = uunifast.gen_tasksets(
            #         5, args.r, 1, 100, args.u, rounded=True)

            # Generate log-uniformly distributed task sets with predefined
            # periods:
            periods = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
            # Interval from where the generator pulls log-uniformly.
            min_pull = 1
            max_pull = 2000

            task_sets_uunifast = uunifast.gen_tasksets_pred(
                50, argsr, min_pull, max_pull, argsu / 100.0, periods)

            # Transform tasks to fit framework structure.
            trans2 = trans.Transformer("2", task_sets_uunifast, 10000000)
            task_sets = trans2.transform_tasks(False)

        else:
            print("Choose a benchmark")
            raise SystemExit  # exit the program

    except Exception as e:
        print(e)
        print("ERROR: task creation")
        if debug_flag:
            breakpoint()
        raise

    return task_sets


def TDA(task_set):
    """TDA analysis for a task set.
    Return True if succesful and False if not succesful."""
    try:
        ana = a.Analyzer()
        # TDA.
        i = 1
        for task in task_set:
            # Prevent WCET = 0 since the scheduler can
            # not handle this yet. This case can occur due to
            # rounding with the transformer.
            if task.wcet == 0:
                raise ValueError("WCET == 0")
            task.rt = ana.tda(task, task_set[:(i - 1)])
            if task.rt > task.deadline:
                raise ValueError(
                    "TDA Result: WCRT bigger than deadline!")
            i += 1
    except ValueError as e:
        print(e)
        return False
    return True


def schedule_task_set(ce_chains, task_set, print_status=False):
    '''Return the schedule of some task_set.
    ce_chains is a list of ce_chains that will be computed later on.
    We need this to compute latency_upper_bound to determine the additional simulation time at the end.
    Note:
    - In case of error, None is returned.
    - E2E Davare has to be computed beforehand!'''

    try:
        # Preliminary: compute latency_upper_bound
        latency_upper_bound = max([ce.davare for ce in ce_chains])

        # Main part: Simulation part
        simulator = es.eventSimulator(task_set)

        # Determination of the variables used to compute the stop
        # condition of the simulation
        max_phase = max(task_set, key=lambda task: task.phase).phase
        max_period = max(task_set, key=lambda task: task.period).period
        hyper_period = a.Analyzer.determine_hyper_period(task_set)

        sched_interval = (
                2 * hyper_period + max_phase  # interval from paper
                + latency_upper_bound  # upper bound job chain length
                + max_period)  # for convenience

        if print_status:
            # Information for end user.
            print("\tNumber of tasks: ", len(task_set))
            print("\tHyperperiod: ", hyper_period)
            number_of_jobs = 0
            for task in task_set:
                number_of_jobs += sched_interval / task.period
            print("\tNumber of jobs to schedule: ",
                  "%.2f" % number_of_jobs)

        # Stop condition: Number of jobs of lowest priority task.
        simulator.dispatcher(
            int(math.ceil(sched_interval / task_set[-1].period)))

        # Simulation without early completion.
        schedule = simulator.e2e_result()

    except Exception as e:
        print(e)
        if debug_flag:
            breakpoint()
        schedule = None

    return schedule


def schedule_taskset_as_list(ce_ts):
    '''Schedule task set and return the result as list.'''
    dict_res = schedule_task_set(ce_ts[0], ce_ts[1])
    list_res = []
    for task in ce_ts[1]:
        list_res.append(dict_res.get(task))
    return list_res


def flatten3(ce_ts_sched):
    """Used to flatten the list ce_ts_sched"""
    ce_ts_sched_flat = [(ce, ts, sched)
                        for ce_lst, ts, sched in ce_ts_sched for ce in ce_lst]
    return ce_ts_sched_flat


def flatten(ce_ts):
    """Used to flatten the list ce_ts_sched"""
    ce_ts_flat = [(ce, ts) for ce_lst, ts in ce_ts for ce in ce_lst]
    return ce_ts_flat


def change_taskset_bcet(task_set, rat):
    """Copy task set and change the wcet/bcet of each task by a given ratio."""
    new_task_set = [task.copy() for task in task_set]
    for task in new_task_set:
        task.wcet = math.ceil(rat * task.wcet)
        task.bcet = math.ceil(rat * task.bcet)
    # Note: ceiling function makes sure there is never execution of 0
    return new_task_set


def check_folder(name):
    """check if the folder exists, otherwise create it"""
    if not os.path.exists(name):
        os.makedirs(name)


####################################################
# Help functions for Analysis with multiprocessing #
####################################################
ana = a.Analyzer()


# Note:
# lst_flat = (ce, ts, sched)
# lst = ([ces], ts, sched)


def davare(lst_flat):
    ce = lst_flat[0]
    return ana.davare_single(ce)


def kloda(lst_flat):
    ce = lst_flat[0]
    hyper = ana.determine_hyper_period(lst_flat[1])
    return ana.kloda(ce, hyper)


def D19_mrt(lst_flat):
    ce = lst_flat[0]
    return ana.reaction_duerr_single(ce)


def D19_mda(lst_flat):
    ce = lst_flat[0]
    return ana.age_duerr_single(ce)


def davare_inter(inter_ch):
    '''Davare analysis for one interconnected chain.'''
    # make standard chain
    ch = []
    for entry in inter_ch:
        if isinstance(entry, c.CauseEffectChain):
            ch.extend(entry.chain)
        elif isinstance(entry, task_package.Task):
            ch.append(entry)
    return ana.davare_single(c.CauseEffectChain(0, ch))


def duerr_mrda_inter(inter_ch):
    '''Duerr analysis for one interconnected chain.'''
    # make standard chain
    ch = []
    for entry in inter_ch:
        if isinstance(entry, c.CauseEffectChain):
            ch.extend(entry.chain)
        elif isinstance(entry, task_package.Task):
            assert entry.message is True
            ch.append(entry)
    return ana.age_duerr_single(c.CauseEffectChain(0, ch))


def duerr_mrt_inter(inter_ch):
    '''Duerr analysis for one interconnected chain.'''
    # make standard chain
    ch = []
    for entry in inter_ch:
        if isinstance(entry, c.CauseEffectChain):
            ch.extend(entry.chain)
        elif isinstance(entry, task_package.Task):
            assert entry.message is True
            ch.append(entry)
    return ana.reaction_duerr_single(c.CauseEffectChain(0, ch))


def cutting_thm_implicit(inter_ch, attr_1, attr_2, bcet):
    '''Cutting Theorem for implicit communication policy.'''
    res = 0
    for entry in inter_ch[:-1]:
        if isinstance(entry, c.CauseEffectChain):
            res += getattr(entry, attr_1)[bcet]
        elif isinstance(entry, task_package.Task):
            res += getattr(entry, 'period')
            res += getattr(entry, 'rt')

    res += getattr(inter_ch[-1], attr_2)[bcet]
    return res


def cutting_thm_mrda(inter_ch, bcet):
    '''Our inter-ECU analysis when applying the Cutting theorem. (implicit communication)'''
    return cutting_thm_implicit(inter_ch, 'our_mda', 'our_mrda', bcet)


def cutting_thm_mrt(inter_ch, bcet):
    '''Our inter-ECU analysis when applying the Cutting theorem. (implicit communication)'''
    return cutting_thm_implicit(inter_ch, 'our_mrt', 'our_mrt', bcet)


def our_mrt_mRda(lst, bcet):
    '''Takes non-flattened list as input, because the schedules can be reused.'''
    ts = lst[1]  # wcet task set
    rat_ts = change_taskset_bcet(ts, bcet)  # bcet task set
    ce_lst = lst[0]  # list of ce chains
    if bcet != 0:  # the dispatcher can only handle execution != 0
        rat_sched = schedule_task_set(
            ce_lst, rat_ts, print_status=False)  # schedule with bcet
    else:
        rat_sched = a_our.execution_zero_schedule(rat_ts)
    sched = lst[2]  # schedule with wcet

    # maximum reaction time result
    mrt_res = [a_our.max_reac_local(
        ce, ts, sched, rat_ts, rat_sched) for ce in ce_lst]
    mRda_res = [a_our.max_age_local(
        ce, ts, sched, rat_ts, rat_sched) for ce in ce_lst]

    mda_res, mrda_res = list(zip(*mRda_res))
    mda_res = list(mda_res)  # maximum data age result
    mrda_res = list(mrda_res)  # maximum reduced data age result

    return (mrt_res, mda_res, mrda_res)


def our_mrt_mRda_lst(lst_ce_ts, bcet_lst, wcet=1.0):
    """lst_ce_ts[0] = list of ce-chains, lst_ce_ts[1] = task set, bet_lst = [0.0, 0.3, 0.7, 1.0]"""
    ce_lst = lst_ce_ts[0]
    ts = lst_ce_ts[1]
    wcet = 1.0

    # make schedules and store in dictionary
    schedules_todo = bcet_lst
    if wcet not in schedules_todo:
        schedules_todo.append(wcet)

    schedules = dict()  # schedules
    ts_lst = dict()  # task sets

    for et in schedules_todo:
        if et == 1.0:
            ts_et = ts  # otherwise tasks in the chain can not be allocated
        else:
            ts_et = change_taskset_bcet(ts, et)  # task set with certain execution time
        if et != 0:  # the dispatcher can only handle execution != 0
            sched_et = schedule_task_set(ce_lst, ts_et, print_status=False)  # schedule with certain execution time
        else:
            sched_et = a_our.execution_zero_schedule(ts_et)
        schedules[et] = sched_et
        ts_lst[et] = ts_et

    # do analysis for certain schedules
    results = []  # results

    for ce in ce_lst:
        results_ce = dict()
        for bcet in bcet_lst:
            results_ce[bcet] = dict()
            # breakpoint()
            ce_mrt = a_our.max_reac_local(ce, ts_lst[wcet], schedules[wcet], ts_lst[bcet], schedules[bcet])
            ce_mda, ce_mrda = a_our.max_age_local(ce, ts_lst[wcet], schedules[wcet], ts_lst[bcet], schedules[bcet])
            results_ce[bcet]['mrt'] = ce_mrt
            results_ce[bcet]['mda'] = ce_mda
            results_ce[bcet]['mrda'] = ce_mrda
        results.append(results_ce)
    return results


#################
# Main function #
#################

if __name__ == '__main__':
    # set seed for same results
    random.seed(331)
    np.random.seed(331)

    # variables
    number = 10
    processors = 1

    # =====args=====
    opts, args = getopt.getopt(sys.argv[1:], "s:u:b:n:p:",
                               ['switch=', 'util=', 'utilization=', 'bench=', 'benchmark=', 'number=', 'proc=',
                                'processors='])

    if all([o in [e[0] for e in opts] for o in ('-s', '--switch')]):
        print("Please provide code switch 'python3 main.py -s<number>'")
        sys.exit()

    for opt, arg in opts:
        if opt in ('-s', '--switch'):  # define which part of the code is being executed
            code_switch = int(arg)
        elif opt in ('-u', '--util', '--utilization'):  # utilization
            util = float(arg)
        elif opt in ('-b', '--bench', '--benchmark'):  # benchmark 'waters' or 'uunifast'
            benchmark = str(arg)
        elif opt in ('-n', '--number'):  # number of tasksets or of interconnected chains for the generations
            number = int(arg)
        elif opt in ('-p', '--proc', '--processors'):  # number of processors that are used for the computations
            processors = int(arg)
        else:
            breakpoint()

    # == Single ECU System Synthesis ==
    if code_switch == 1:
        print('=== Single ECU system synthesis')

        # == Check parameters
        assert {'benchmark', 'util', 'number'}.issubset(locals())
        assert benchmark in ('waters', 'uunifast')
        assert 0.0 <= util <= 100.0

        # == Create task set.
        print('= Task set generation')
        task_sets = task_set_generate(benchmark, util, number)

        # == CE-Chain generation.
        print('= CE-Chain generation')

        ce_chains = waters.gen_ce_chains(task_sets)
        # ce_chains contains one set of cause effect chains for each
        # task set in task_sets.

        # Match both
        assert len(task_sets) == len(ce_chains)
        ce_ts = list(zip(ce_chains, task_sets))

        # == Schedule generation
        print('= Davare Analysis (preperation for schedule generation)')

        # Preparation: TDA (for Davare)
        # Only take those that are succesful.
        ce_ts = [entry for entry in ce_ts if TDA(entry[1])]

        # Preparation: Davare (for schedule generation later on)
        ana = a.Analyzer()
        for ce, ts in ce_ts:
            ana.davare([ce])

        # # Main: Generate the schedule
        # with Pool(processors) as p:
        #     schedules_lst = p.map(schedule_taskset_as_list, ce_ts)
        # schedules_dict = []
        # for idxx, sched in enumerate(schedules_lst):
        #     schedules_dict.append(dict())
        #     for idxxx, tsk in enumerate(ce_ts[idxx][1]):
        #         schedules_dict[idxx][tsk] = sched[idxxx][:]
        #
        # schedules = schedules_dict
        #
        # # match ce_ts with schedules:
        # assert len(ce_ts) == len(schedules)
        # ce_ts_sched = [cets + (sched,)
        #                for cets, sched in zip(ce_ts, schedules)]
        # # Note: Each entry is now a 3-tuple of list of cause-effect chain,
        # # corresponding task set, and corresponding schedule
        # # - ce_ts_sched
        # #   - ce_ts_sched[0] --> one ECU
        # #     - ce_ts_sched[0][0] = set of ce-chains
        # #     - ce_ts_sched[0][1] = set of tasks (one ECU!)
        # #     - ce_ts_sched[0][2] = schedule of that ECU as dictionary
        # #   - ce_ts_sched[1]
        # #   - ...

        # == Save the results
        print("= Save data")
        check_or_make_directory('output/1generation')
        write_data(f'output/1generation/single_ce_ts_{util=}_{number=}_{benchmark=}.pkl', ce_ts)

    # == Single ECU Experiment ==
    elif code_switch == 2:

        # == Check parameters
        assert {'benchmark', 'util', 'number'}.issubset(locals())
        assert benchmark in ('waters', 'uunifast')
        assert 0.0 <= util <= 100.0

        # == Load data
        print(time_now(), "= Load data")
        ce_ts = load_data(f'output/1generation/single_ce_ts_{util=}_{number=}_{benchmark=}.pkl')
        ce_ts_flat = flatten(ce_ts)  # this one is used for other analyses

        # == Other analyses
        # - Davare
        # - Kloda
        # - D19
        # - G21

        # ana = a.Analyzer()
        print(time_now(), '= Other analyses')

        o_analyses = [  # tuple of name, function name, and attribute name of the object
            ['Davare', davare, 'davare'],
            ['Kloda', kloda, 'kloda'],
            ['D19: MDA', D19_mda, 'd19_mrda'],
            ['D19: MRT', D19_mrt, 'd19_mrt'],
            # ['G21: MDA', G21_mda, 'g21_mda'],
            # ['G21: MRDA', G21_mrda, 'g21_mrda'],
            # ['G21: MRT', G21_mrt, 'g21_mrt'],
        ]

        for name, fct, attr_name in o_analyses:
            print(time_now(), name)

            # Get result
            with Pool(processors) as p:
                results = p.map(fct, ce_ts_flat)

            # Set results
            assert len(results) == len(ce_ts_flat)
            for res, entry in zip(results, ce_ts_flat):
                setattr(entry[0], attr_name, res)

        # == Our analysis
        print(time_now(), '= Our analysis')

        bcet_ratios = [1.0, 0.7, 0.3, 0.0]  # these should be analysed

        # Add dictionary for each cause-effect chain
        for ce, _ in ce_ts_flat:
            ce.our_mrt = dict()
            ce.our_mda = dict()
            ce.our_mrda = dict()

        # Get our results
        # res_our = []
        # for entry in ce_ts:
        #     res_our.append(our_mrt_mRda_lst(entry, bcet_ratios))
        with Pool(processors) as p:
            res_our = p.starmap(our_mrt_mRda_lst, zip(ce_ts, itertools.repeat(bcet_ratios)))

        # Set our results
        assert len(res_our) == len(ce_ts)
        for res, entry in zip(res_our, ce_ts):
            for idxx, ce in enumerate(entry[0]):
                for bcet in bcet_ratios:
                    # breakpoint()
                    ce.our_mrt[bcet] = res[idxx][bcet]['mrt']
                    ce.our_mda[bcet] = res[idxx][bcet]['mda']
                    ce.our_mrda[bcet] = res[idxx][bcet]['mrda']

        # for bcet in bcet_ratios:
        #     print(time_now(), 'BCET/WCET =', bcet)
        #
        #     # Get result
        #     with Pool(processors) as p:
        #         res_our = p.starmap(our_mrt_mRda, zip(
        #             ce_ts_sched, itertools.repeat(bcet)))
        #
        #     # Set results
        #     assert len(res_our) == len(ce_ts_sched)
        #     for res, entry in zip(res_our, ce_ts_sched):
        #         for idxx, ce in enumerate(entry[0]):
        #             ce.our_mrt[bcet] = res[0][idxx]
        #             ce.our_mda[bcet] = res[1][idxx]
        #             ce.our_mrda[bcet] = res[2][idxx]

        # == Save the results
        print("= Save data")
        check_or_make_directory('output/2results')
        write_data(f'output/2results/single_ce_ts_{util=}_{number=}_{benchmark=}.pkl', ce_ts)

    # == Inter-ECU System Synthesis ==
    if code_switch == 3:
        # Note: make sure that single ECU analysis was done beforehand

        # == check parameters
        assert {'benchmark', 'number'}.issubset(locals())
        assert benchmark in ('waters', 'uunifast')

        number_interconn_ce_chains = number

        # == get all filenames
        input_dir = 'output/2results/'
        filenames = os.listdir(input_dir)
        pattern = re.compile(f'single.*{benchmark=}.*\.pkl')
        filenames = [file for file in filenames if
                     pattern.match(file)]
        filenames.sort()  # sort alphabetically (=> no random behavior when loading)

        # == load data and combine
        single_ce_ts_flat = []
        for file in filenames:
            single_ce_ts_flat.extend(flatten(load_data(input_dir + file)))

        # == make interconnected chains
        inter_chains = []
        for j in range(0, number_interconn_ce_chains):
            com_tasks = comm.generate_communication_taskset(20, 10, 1000, True)  # generate communication tasks
            com_tasks = list(np.random.choice(com_tasks, 4, replace=False))  # randomly choose 4
            choice_ces = np.random.choice(len(single_ce_ts_flat), 5, replace=False)
            ces = [single_ce_ts_flat[entry][0] for entry in choice_ces]  # randomly choose 5
            inter_chains.append([
                ces[0],
                com_tasks[0],
                ces[1],
                com_tasks[1],
                ces[2],
                com_tasks[2],
                ces[3],
                com_tasks[3],
                ces[4],
            ])
            # End user notification
            if j % 100 == 0:
                print("\t", j)

        # == Save the results
        print("= Save data")
        check_or_make_directory('output/1generation')
        write_data(f'output/1generation/inter_chains_{benchmark=}.pkl', inter_chains)

    # inter-ecu experiment
    if code_switch == 4:
        # == Check parameters
        assert {'benchmark', }.issubset(locals())
        assert benchmark in ('waters', 'uunifast')

        # == Load data
        print(time_now(), "= Load data")
        inter_chains = load_data(f'output/1generation/inter_chains_{benchmark=}.pkl')

        # == Analyses
        print(time_now(), "= Analyses")

        # davare
        with Pool(processors) as p:
            res_davare = p.map(davare_inter, inter_chains)

        # duerr MRDA
        with Pool(processors) as p:
            res_duerr_mrda = p.map(duerr_mrda_inter, inter_chains)

        # duerr MRT
        with Pool(processors) as p:
            res_duerr_mrt = p.map(duerr_mrt_inter, inter_chains)

        # Our (apply cutting)
        res_our_mrda = dict()
        res_our_mrt = dict()

        bcet_ratios = [1.0, 0.7, 0.3, 0.0]

        for bcet in bcet_ratios:
            with Pool(processors) as p:
                res_our_mrda[bcet] = p.starmap(cutting_thm_mrda, zip(inter_chains, itertools.repeat(bcet)))
                res_our_mrt[bcet] = p.starmap(cutting_thm_mrt, zip(inter_chains, itertools.repeat(bcet)))

        results_mrda = {
            'D07': res_davare,
            'D19': res_duerr_mrda,
            'Our': res_our_mrda,
        }

        results_mrt = {
            'D07': res_davare,
            'D19': res_duerr_mrt,
            'Our': res_our_mrt
        }

        # == Save the results
        print(time_now(), "= Save data")
        check_or_make_directory('output/2results')
        write_data(f'output/2results/results_inter_{benchmark=}_mrda.pkl', results_mrda)
        write_data(f'output/2results/results_inter_{benchmark=}_mrt.pkl', results_mrt)

    # single ecu plotting
    if code_switch == 5:
        # == Check parameters
        assert {'benchmark', }.issubset(locals())
        assert benchmark in ('waters', 'uunifast')

        # == get all filenames
        filenames = os.listdir('output/2results/.')
        pattern = re.compile(f'single.*{benchmark=}.*\.pkl')
        filenames = [file for file in filenames if pattern.match(file)]

        # == load data and combine
        single_ce_ts_flat = []
        for file in filenames:
            single_ce_ts_flat.extend(flatten(load_data('output/2results/' + file)))

        # == extract results:
        extract_lst_mrda = [  # name, attribute name, item
            ['D07', 'davare', None],  # Davare 2007
            ['D19', 'd19_mrda', None],  # Duerr 2019
            ['K18', 'kloda', None],  # Kloda 2018
            ['0.0', 'our_mrda', 0.0],  # Our BCET = 0.0 WCET
            ['0.3', 'our_mrda', 0.3],  # Our BCET = 0.3 WCET
            ['0.7', 'our_mrda', 0.7],  # Our BCET = 0.7 WCET
            ['1.0', 'our_mrda', 1.0],  # Our BCET = 1.0 WCET
        ]
        results_mrda = dict()
        for ana_name, attr_name, item_name in extract_lst_mrda:
            if item_name is not None:
                results_mrda[ana_name] = [getattr(ce, attr_name)[item_name] for ce, _ in single_ce_ts_flat]
            else:
                results_mrda[ana_name] = [getattr(ce, attr_name) for ce, _ in single_ce_ts_flat]

        extract_lst_mrt = [  # name, attribute name, item
            ['D07', 'davare', None],  # Davare 2007
            ['D19', 'd19_mrt', None],  # Duerr 2019
            ['K18', 'kloda', None],  # Kloda 2018
            ['0.0', 'our_mrt', 0.0],  # Our BCET = 0.0 WCET
            ['0.3', 'our_mrt', 0.3],  # Our BCET = 0.3 WCET
            ['0.7', 'our_mrt', 0.7],  # Our BCET = 0.7 WCET
            ['1.0', 'our_mrt', 1.0],  # Our BCET = 1.0 WCET
        ]
        results_mrt = dict()
        for ana_name, attr_name, item_name in extract_lst_mrt:
            if item_name is not None:
                results_mrt[ana_name] = [getattr(ce, attr_name)[item_name] for ce, _ in single_ce_ts_flat]
            else:
                results_mrt[ana_name] = [getattr(ce, attr_name) for ce, _ in single_ce_ts_flat]

        # == Plot results_mrda and results_mrt
        check_or_make_directory('output/3plots')
        plot.plot_reduction(results_mrda, 'D07', f'output/3plots/single_red_mrda_{benchmark=}.pdf')
        plot.plot_reduction(results_mrt, 'D07', f'output/3plots/single_red_mrt_{benchmark=}.pdf')

        plot.plot_gap_reduction(results_mrda, 'D07', '1.0', f'output/3plots/single_gap_mrda_{benchmark=}.pdf')
        plot.plot_gap_reduction(results_mrt, 'D07', '1.0', f'output/3plots/single_gap_mrt_{benchmark=}.pdf')

    # inter-ecu potting
    if code_switch == 6:
        # == Check parameters
        assert {'benchmark', }.issubset(locals())
        assert benchmark in ('waters', 'uunifast')

        # == Load data
        print(time_now(), "= Load data")
        results_mrda = load_data(f'output/2results/results_inter_{benchmark=}_mrda.pkl')
        results_mrt = load_data(f'output/2results/results_inter_{benchmark=}_mrt.pkl')

        bcet_ratios = [0.0, 0.3, 0.7, 1.0]

        our_mrda = results_mrda.pop('Our')
        for bcet in bcet_ratios:
            results_mrda[str(bcet)] = our_mrda[bcet]

        our_mrt = results_mrt.pop('Our')
        for bcet in bcet_ratios:
            results_mrt[str(bcet)] = our_mrt[bcet]

        # == Plot results_mrda and results_mrt
        check_or_make_directory('output/3plots')
        plot.plot_reduction(results_mrda, 'D07', f'output/3plots/inter_red_mrda_{benchmark=}.pdf')
        plot.plot_reduction(results_mrt, 'D07', f'output/3plots/inter_red_mrt_{benchmark=}.pdf')
