#!/usr/bin/env python3
"""Evaluation for the paper 'Timing Analysis of Asynchronized Distributed
Cause-Effect Chains' (2021).

It includes (1) local analysis (2) global analysis and (3) plotting of the
results.
"""

import gc  # garbage collector
import argparse
import math
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

import time
import sys
import os

import random  # randomization
from multiprocessing import Pool  # multiprocessing
import itertools  # better performance


debug_flag = True  # flag to have breakpoint() when errors occur

# set seed for same results
random.seed(331)
np.random.seed(331)


########################
# Some help functions: #
########################

def time_now():
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    return current_time


def task_set_generate(argsg, argsu, argsr):
    '''Generates task sets.
    Input:
    - argsg = benchmark to choose
    - argsu = utilization in %
    - argsr = number of task sets to generate
    Output: list of task sets.'''
    try:
        if argsg == 0:
            # WATERS benchmark
            print("WATERS benchmark.")

            # Statistical distribution for task set generation from table 3
            # of WATERS free benchmark paper.
            profile = [0.03 / 0.85, 0.02 / 0.85, 0.02 / 0.85, 0.25 / 0.85,
                       0.25 / 0.85, 0.03 / 0.85, 0.2 / 0.85, 0.01 / 0.85,
                       0.04 / 0.85]
            # Required utilization:
            req_uti = argsu/100.0
            # Maximal difference between required utilization and actual
            # utilization is set to 1 percent:
            threshold = 1.0

            # Create task sets from the generator.
            # Each task is a dictionary.
            print("\tCreate task sets.")
            task_sets_waters = []
            while len(task_sets_waters) < argsr:
                task_sets_gen = waters.gen_tasksets(
                    1, req_uti, profile, True, threshold/100.0, 4)
                task_sets_waters.append(task_sets_gen[0])

            # Transform tasks to fit framework structure.
            # Each task is an object of utilities.task.Task.
            trans1 = trans.Transformer("1", task_sets_waters, 10000000)
            task_sets = trans1.transform_tasks(False)

        elif argsg == 1:
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
                50, argsr, min_pull, max_pull, argsu/100.0, periods)

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
    '''TDA analysis for a task set.
    Return True if succesful and False if not succesful.'''
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
                number_of_jobs += sched_interval/task.period
            print("\tNumber of jobs to schedule: ",
                  "%.2f" % number_of_jobs)

        # Stop condition: Number of jobs of lowest priority task.
        simulator.dispatcher(
            int(math.ceil(sched_interval/task_set[-1].period)))

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


def flatten(ce_ts_sched):
    '''Used to flatten the list ce_ts_sched'''
    ce_ts_sched_flat = [(ce, ts, sched)
                        for ce_lst, ts, sched in ce_ts_sched for ce in ce_lst]
    return ce_ts_sched_flat


def change_taskset_bcet(task_set, rat):
    '''Copy task set and change the wcet/bcet of each task by a given ratio.'''
    new_task_set = [task.copy() for task in task_set]
    for task in new_task_set:
        task.wcet = math.ceil(rat * task.wcet)
        task.bcet = math.ceil(rat * task.bcet)
    # Note: ceiling function makes sure there is never execution of 0
    return new_task_set


def check_folder(name):
    '''check if the folder exists, otherwise create it'''
    if not os.path.exists(name):
        os.makedirs(name)


###############################
# Help functions for Analysis #
###############################
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


def G21_mda(lst_flat):
    sched = lst_flat[2]
    ts = lst_flat[1]
    ce = lst_flat[0]
    max_phase = max(t.phase for t in ts)
    hyper = ana.determine_hyper_period(ts)
    return ana.max_age_our(sched, ts, ce, max_phase, hyper, reduced=False)


def G21_mrda(lst_flat):
    sched = lst_flat[2]
    ts = lst_flat[1]
    ce = lst_flat[0]
    max_phase = max(t.phase for t in ts)
    hyper = ana.determine_hyper_period(ts)
    return ana.max_age_our(sched, ts, ce, max_phase, hyper, reduced=True)


def G21_mrt(lst_flat):
    sched = lst_flat[2]
    ts = lst_flat[1]
    ce = lst_flat[0]
    max_phase = max(t.phase for t in ts)
    hyper = ana.determine_hyper_period(ts)
    return ana.reaction_our(sched, ts, ce, max_phase, hyper)


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


def let_mrt(lst_flat):
    '''analysis with LET communication policy'''
    if lst_flat[0].length() == 0:
        return 0
    return a_our.mrt_let(lst_flat[0], lst_flat[1])


def let_mRda(lst_flat):
    '''analysis with LET communication policy
    Note: Returns tuple of mda and mrda result.'''
    if lst_flat[0].length() == 0:
        return (0, 0)
    return a_our.mda_let(lst_flat[0], lst_flat[1])


def analyze_mixed_mrt(lst_inter, scenario):
    '''Analyze a mixed setup. 
    lst_inter is a list where each entry is a list of ce chain, task set and schedule
    '''

    e2e_latency = 0

    for entry, sc in zip(lst_inter, scenario):
        if sc == 'impl':
            e2e_latency += entry[0].our_mrt[0]
        elif sc == 'let':
            e2e_latency += entry[0].let_mrt

    return e2e_latency


def analyze_mixed_mda(lst_inter, scenario):
    '''Analyze a mixed setup. 
    lst_inter is a list where each entry is a list of ce chain, task set and schedule
    '''

    e2e_latency = 0

    for entry, sc in zip(lst_inter, scenario):
        if sc == 'impl':
            e2e_latency += entry[0].our_mda[0]
        elif sc == 'let':
            e2e_latency += entry[0].let_mda

    return e2e_latency


def analyze_mixed_mrda(lst_inter, scenario):
    '''Analyze a mixed setup. 
    lst_inter is a list where each entry is a list of ce chain, task set and schedule
    '''

    # if last chain is empty chain, remove it!
    while lst_inter[-1][0].length() == 0:
        lst_inter = lst_inter[:-1]
        scenario = scenario[:-1]

    e2e_latency = 0

    # first entries
    for entry, sc in zip(lst_inter[:-1], scenario[:-1]):
        if sc == 'impl':
            e2e_latency += entry[0].our_mda[0]
        elif sc == 'let':
            e2e_latency += entry[0].let_mda

    # last entry
    sc = scenario[-1]
    entry = lst_inter[-1]
    if sc == 'impl':
        e2e_latency += entry[0].our_mrda[0]
    elif sc == 'let':
        e2e_latency += entry[0].let_mrda

    return e2e_latency


def divide_chain_4(chain, nmb):
    '''divides chain into two parts, where the cut is at nmb/4-th.'''
    cut_at = int(len(chain.chain)*nmb/4)
    c1 = c.CauseEffectChain(1, chain.chain[:cut_at])
    c2 = c.CauseEffectChain(2, chain.chain[cut_at:])
    return (c1, c2)


def cut_chain(ce_ts_sched_entry, nmb):
    '''makes a list of two chains for one chain by cutting at nmb/4-th of the length'''
    ce, ts, sched = ce_ts_sched_entry
    ce1, ce2 = divide_chain_4(ce, nmb)
    return ((ce1, ts, sched), (ce2, ts, sched))

#################
# Main function #
#################


def main():
    """Main Function."""
    ###
    # Argument Parser
    ###
    parser = argparse.ArgumentParser()

    # which part of code should be executed:
    parser.add_argument("-j", type=int, default=0)
    # utilization in 0 to 100 [percent]:
    parser.add_argument("-u", type=float, default=50)
    # task generation (0: WATERS Benchmark, 1: UUnifast):
    parser.add_argument("-g", type=int, default=0)

    # number of concurrent processes:
    parser.add_argument("-p", type=int, default=1)

    # name of the run:
    parser.add_argument("-n", type=int, default=-1)
    # number of task sets to generate:
    parser.add_argument("-r", type=int, default=5)
    # number mixed ce chains to generate:
    parser.add_argument("-m", type=int, default=20)

    args = parser.parse_args()
    del parser

    if args.j == 10:
        """Create task sets, local cause-effect chains and produce schedule."""

        ###
        # Create task set.
        # output:
        ###
        print('=Task set generation')

        task_sets = task_set_generate(args.g, args.u, args.r)

        ###
        # CE-Chain generation.
        ###
        print('=CE-Chain generation')

        ce_chains = waters.gen_ce_chains(task_sets)
        # ce_chains contains one set of cause effect chains for each
        # task set in task_sets.

        # match both
        assert len(task_sets) == len(ce_chains)
        ce_ts = list(zip(ce_chains, task_sets))

        # breakpoint()

        ###
        # Schedule generation
        ###
        print('=Schedule generation')

        # Preparation: TDA (for Davare)
        # Only take those that are succesful.
        ce_ts = [entry for entry in ce_ts if TDA(entry[1])]

        # Preparation: Davare (for schedule generation)
        ana = a.Analyzer()
        for ce, ts in ce_ts:
            ana.davare([ce])

        # Main: Generate the schedule
        with Pool(args.p) as p:
            schedules_lst = p.map(schedule_taskset_as_list, ce_ts)
        schedules_dict = []
        for idxx, sched in enumerate(schedules_lst):
            schedules_dict.append(dict())
            for idxxx, tsk in enumerate(ce_ts[idxx][1]):
                schedules_dict[idxx][tsk] = sched[idxxx][:]

        schedules = schedules_dict

        # breakpoint()

        # for entry in ce_ts:
        #     sched1 = schedule_task_set(*entry)
        #     with Pool(1) as p:
        #         sched2 = p.map(schedule_taskset_as_list, [entry])
        #     sched2 = sched2[0]
        #     # sched2 = schedule_taskset_as_list(entry)
        #     sched3 = dict()
        #     for t, s in zip(entry[1], sched2):
        #         sched3[t] = s
        #     breakpoint()
        # with Pool(args.p) as p:
        #     schedules = p.starmap(schedule_task_set, ce_ts)
        # schedules = [schedule_task_set(ts, ce) for ce, ts in ce_ts]

        # match ce_ts with schedules:
        assert len(ce_ts) == len(schedules)
        ce_ts_sched = [cets + (sched,)
                       for cets, sched in zip(ce_ts, schedules)]
        # Note: Each entry is now a 3-tuple of list of cause-effect chain,
        # corresponding task set, and corresponding schedule

        # breakpoint()

        ###
        # Save the results
        ###
        print("=Save data.=")

        try:
            folder = "output/1generation/"
            filename = ("ce_ts_sched_u="+str(args.u) +
                        "_n=" + str(args.n) + "_g=" + str(args.g) + ".npz")
            check_folder(folder)
            np.savez(folder + filename, gen=ce_ts_sched)

        except Exception as e:
            print(e)
            print("ERROR: save")
            if debug_flag:
                breakpoint()
            else:
                return

    elif args.j == 11:
        '''Implicit communication analyses.

        Input:
        - args.u
        - args.n
        - args.g
        - args.p
        '''

        ###
        # Load data
        ###
        print(time_now(), "= Load data =")

        filename = ("output/1generation/ce_ts_sched_u="+str(args.u)
                    + "_n=" + str(args.n)
                    + "_g=" + str(args.g) + ".npz")
        data = np.load(filename, allow_pickle=True)
        ce_ts_sched = data.f.gen  # this one is used

        ce_ts_sched_flat = flatten(ce_ts_sched)  # this one is used

        ###
        # Other analyses
        # - Davare
        # - Kloda
        # - D19
        # - G21
        ###
        # ana = a.Analyzer()
        print(time_now(), '= Other analyses =')

        ###
        # ==Davare
        print(time_now(), 'Davare')

        # Get result
        with Pool(args.p) as p:
            res_davare = p.map(davare, ce_ts_sched_flat)

        # Set results
        assert len(res_davare) == len(ce_ts_sched_flat)
        for res, entry in zip(res_davare, ce_ts_sched_flat):
            entry[0].davare = res

        ###
        # ==Kloda
        print(time_now(), 'Kloda')

        # Get result
        with Pool(args.p) as p:
            res_kloda = p.map(kloda, ce_ts_sched_flat)

        # Set results
        assert len(res_kloda) == len(ce_ts_sched_flat)
        for res, entry in zip(res_kloda, ce_ts_sched_flat):
            entry[0].kloda = res

        ###
        # ==Duerr (D19): MDA
        print(time_now(), 'D19: MDA')

        # Get result
        with Pool(args.p) as p:
            res_d19_mda = p.map(D19_mda, ce_ts_sched_flat)

        # Set results
        assert len(res_d19_mda) == len(ce_ts_sched_flat)
        for res, entry in zip(res_d19_mda, ce_ts_sched_flat):
            entry[0].d19_mrda = res

        # ==Duerr (D19): MRT
        print(time_now(), 'D19: MRT')

        # Get result
        with Pool(args.p) as p:
            res_d19_mrt = p.map(D19_mrt, ce_ts_sched_flat)

        # Set results
        assert len(res_d19_mrt) == len(ce_ts_sched_flat)
        for res, entry in zip(res_d19_mrt, ce_ts_sched_flat):
            entry[0].d19_mrt = res

        ###
        # ==Guenzel (G21): MDA
        print(time_now(), 'G21: MDA')

        # Get result
        with Pool(args.p) as p:
            res_g21_mda = p.map(G21_mda, ce_ts_sched_flat)

        # Set results
        assert len(res_g21_mda) == len(ce_ts_sched_flat)
        for res, entry in zip(res_g21_mda, ce_ts_sched_flat):
            entry[0].g21_mda = res

        # ==Guenzel (G21): MRDA
        print(time_now(), 'G21: MRDA')

        # Get result
        with Pool(args.p) as p:
            res_g21_mrda = p.map(G21_mrda, ce_ts_sched_flat)

        # Set results
        assert len(res_g21_mrda) == len(ce_ts_sched_flat)
        for res, entry in zip(res_g21_mrda, ce_ts_sched_flat):
            entry[0].g21_mrda = res

        # ==Guenzel (G21): MRT
        print(time_now(), 'G21: MRT')

        # Get result
        with Pool(args.p) as p:
            res_g21_mrt = p.map(G21_mrt, ce_ts_sched_flat)

        # Set results
        assert len(res_g21_mrt) == len(ce_ts_sched_flat)
        for res, entry in zip(res_g21_mrt, ce_ts_sched_flat):
            entry[0].g21_mrt = res

        # print(ce_ts_sched_flat[0][0].g21_mda, ce_ts_sched_flat[0][0].g21_mrda,
        #       ce_ts_sched_flat[0][0].g21_mrt)
        # breakpoint()

        ###
        # Our analysis
        ###

        # Note: given some bcet ratio, make new schedule, analyse, put value to ce chain.

        print(time_now(), '= Our analysis =')

        bcet_ratios = [1.0, 0.7, 0.3, 0.0]

        # Add dictionary for each cause-effect chain
        for ce, _, _ in ce_ts_sched_flat:
            ce.our_mrt = dict()
            ce.our_mda = dict()
            ce.our_mrda = dict()

        for bcet in bcet_ratios:
            print(time_now(), 'BCET/WCET =', bcet)

            # Get result
            with Pool(args.p) as p:
                res_our = p.starmap(our_mrt_mRda, zip(
                    ce_ts_sched, itertools.repeat(bcet)))

            # Set results
            assert len(res_our) == len(ce_ts_sched)
            for res, entry in zip(res_our, ce_ts_sched):
                for idxx, ce in enumerate(entry[0]):
                    ce.our_mrt[bcet] = res[0][idxx]
                    ce.our_mda[bcet] = res[1][idxx]
                    ce.our_mrda[bcet] = res[2][idxx]

        # breakpoint()

        ###
        # Store data
        ###
        print(time_now(), '= Store data =')

        folder = "output/2implicit/"
        output_filename = ("ce_ts_sched_u=" + str(args.u) +
                           "_n=" + str(args.n) + "_g=" + str(args.g) + ".npz")
        check_folder(folder)
        np.savez(folder+output_filename, gen=ce_ts_sched)

        print(time_now(), '= Done =')

    if args.j == 12:
        '''mixed setup evaluation -- Global
        - make interconnected chains with 4 ce each
        - different scenarios for which local chains have implicit communication or LET
        - apply our analysis
        Note:
        - for implicit we assume BCET/WCET = 0
        Output: 'final_results' 
        - list
        - each entry describes results from one scenario
        - the results consist of 3 lists covering the mrt, mda, mrda for each interconnected ce chain
        '''

        scenarios = [
            ['impl', 'impl', 'impl', 'impl'],
            ['let', 'impl', 'impl', 'impl'],
            ['let', 'let', 'impl', 'impl'],
            ['let', 'let', 'let', 'impl'],
            ['let', 'let', 'let', 'let'],
        ]

        ###
        # Load data
        ###
        print(time_now(), "= Load data =")

        filename = ("output/2implicit/ce_ts_sched_u="+str(args.u)
                    + "_n=" + str(args.n)
                    + "_g=" + str(args.g) + ".npz")
        data = np.load(filename, allow_pickle=True)

        ce_ts_sched = data.f.gen  # this one is used
        ce_ts_sched_flat = flatten(ce_ts_sched)  # this one is used

        # breakpoint()

        ###
        # Make interconnected mixed chain
        ###
        nmb_inter = args.m  # number of chains for the analysis

        # list of lists of ce chains
        # (each list of ce chains is one interconnected chain)
        ce_ts_sched_inter = [random.sample(
            ce_ts_sched_flat, 4) for _ in range(nmb_inter)]

        assert len(ce_ts_sched_inter) == nmb_inter

        ###
        # Analyze
        ###
        print(time_now(), '= Analysis =')

        # == LET: MRT
        # We do LET analysis to use that for the interconnected analysis
        print(time_now(), 'LET: MRT')

        # Get result
        with Pool(args.p) as p:
            res_let_mrt = p.map(let_mrt, ce_ts_sched_flat)

        # Set results
        assert len(res_let_mrt) == len(ce_ts_sched_flat)
        for res, entry in zip(res_let_mrt, ce_ts_sched_flat):
            entry[0].let_mrt = res

        # == LET: M(R)DA
        # We do LET analysis to use that for the interconnected analysis
        print(time_now(), 'LET: M(R)DA')

        # Get result
        with Pool(args.p) as p:
            res_let_mRda = p.map(let_mRda, ce_ts_sched_flat)

        # Set results
        assert len(res_let_mRda) == len(ce_ts_sched_flat)
        for res, entry in zip(res_let_mRda, ce_ts_sched_flat):
            entry[0].let_mda = res[0]
            entry[0].let_mrda = res[1]

        # breakpoint()

        # == Mixed scenario
        print(time_now(), 'Mixed Scenarios')

        final_results = []

        for sc in scenarios:
            # Get results
            with Pool(args.p) as p:
                res_mix_mrt = p.starmap(analyze_mixed_mrt, zip(
                    ce_ts_sched_inter, itertools.repeat(sc)))
                res_mix_mda = p.starmap(analyze_mixed_mda, zip(
                    ce_ts_sched_inter, itertools.repeat(sc)))
                res_mix_mrda = p.starmap(analyze_mixed_mrda, zip(
                    ce_ts_sched_inter, itertools.repeat(sc)))

            # Save results
            assert len(res_mix_mrt) == len(ce_ts_sched_inter)
            assert len(res_mix_mda) == len(ce_ts_sched_inter)
            assert len(res_mix_mrda) == len(ce_ts_sched_inter)

            final_results.append(
                [res_mix_mrt[:], res_mix_mda[:], res_mix_mrda[:]])

        # # DEBUG
        # this, this_ts, _ = ce_ts_sched_inter[0][0][1]
        # print(this.chain, [t.period for t in this.chain],
        #       this.let_mrt, this.let_mda, this.let_mrda)
        # breakpoint()

        # a_our.mda_let(this, this_ts)

        ###
        # Store data
        ###
        print(time_now(), '= Store data =')
        folder = "output/3mixedinter/"
        output_filename = ("inter_res_u=" + str(args.u) +
                           "_n=" + str(args.n) + "_g=" + str(args.g) + ".npz")
        check_folder(folder)
        np.savez(folder+output_filename,
                 result=final_results, scenarios=scenarios)

        print(time_now(), '= Done =')

    if args.j == 13:
        '''mixed setup evaluation -- Local
        '''

        scenarios = [
            0, 1, 2, 3, 4
        ]  # 1 means 1/4-th LET, 2 means 2/4=1/2-th LET, ...

        ###
        # Load data
        ###
        print(time_now(), "= Load data =")

        filename = ("output/2implicit/ce_ts_sched_u="+str(args.u)
                    + "_n=" + str(args.n)
                    + "_g=" + str(args.g) + ".npz")
        data = np.load(filename, allow_pickle=True)

        ce_ts_sched = data.f.gen  # this one is used
        ce_ts_sched_flat = flatten(ce_ts_sched)  # this one is used

        ###
        # Make intraconnected mixed chain
        ###

        # == Choose those chains which are divideable by 4
        ce_ts_sched_flat_div4 = [
            entry for entry in ce_ts_sched_flat if len(entry[0].chain) % 4 == 0]

        # == Randomly choose 100 from them
        nmb_intra = args.m

        if len(ce_ts_sched_flat_div4) < nmb_intra:
            print('chains required:', nmb_intra)
            print('chains found:', len(ce_ts_sched_flat_div4))
            print('continue with "c" ...')
            breakpoint()
            ce_ts_sched_intra = ce_ts_sched_flat_div4
        else:  # choose 100 at random
            ce_ts_sched_intra = random.sample(ce_ts_sched_flat_div4, nmb_intra)

        # == Cut the chains
        ce_ts_sched_intra_dict = dict()
        for sc in scenarios:
            ce_ts_sched_intra_dict[sc] = [
                cut_chain(entry, sc) for entry in ce_ts_sched_intra]

        # flatten the dict
        # Note: we do this to apply the analysis
        ce_ts_sched_intra_dict_flat = dict()
        for sc in scenarios:
            ce_ts_sched_intra_dict_flat[sc] = [
                entry for intra_entry in ce_ts_sched_intra_dict[sc] for entry in intra_entry]

        # breakpoint()

        ###
        # Analyze
        ###

        final_results = []

        for sc in scenarios:
            print(time_now(), 'Scenario:', sc)

            ce_ts_sched_analysis = ce_ts_sched_intra_dict[sc]
            ce_ts_sched_flat_analysis = ce_ts_sched_intra_dict_flat[sc]

            # breakpoint()

            # == LET: MRT
            # We do LET analysis to use that for the mixed analysis
            print(time_now(), 'LET: MRT')

            # Get result
            with Pool(args.p) as p:
                res_let_mrt = p.map(let_mrt, ce_ts_sched_flat_analysis)

            # Set results
            assert len(res_let_mrt) == len(ce_ts_sched_flat_analysis)
            for res, entry in zip(res_let_mrt, ce_ts_sched_flat_analysis):
                entry[0].let_mrt = res

            # == LET: M(R)DA
            # We do LET analysis to use that for the mixed analysis
            print(time_now(), 'LET: M(R)DA')

            # Get result
            with Pool(args.p) as p:
                res_let_mRda = p.map(let_mRda, ce_ts_sched_flat_analysis)

            # Set results
            assert len(res_let_mRda) == len(ce_ts_sched_flat_analysis)
            for res, entry in zip(res_let_mRda, ce_ts_sched_flat_analysis):
                entry[0].let_mda = res[0]
                entry[0].let_mrda = res[1]

            # == IMPL
            print(time_now(), 'IMPLICIT')

            bcet = 0

            for ce, _, _ in ce_ts_sched_flat_analysis:
                ce.our_mrt = dict()
                ce.our_mda = dict()
                ce.our_mrda = dict()

            # breakpoint()

            # Get result
            analysis_impl = [([ce], ts, sched)
                             for ce, ts, sched in ce_ts_sched_flat_analysis]
            with Pool(args.p) as p:
                res_our = p.starmap(our_mrt_mRda, zip(
                    analysis_impl, itertools.repeat(bcet)))

            # Set results
            assert len(res_our) == len(analysis_impl)
            for res, entry in zip(res_our, analysis_impl):
                for idxx, ce in enumerate(entry[0]):
                    ce.our_mrt[bcet] = res[0][idxx]
                    ce.our_mda[bcet] = res[1][idxx]
                    ce.our_mrda[bcet] = res[2][idxx]

            # == mixed analysis
            print(time_now(), 'MIXED')
            queue = ['let', 'impl']
            with Pool(args.p) as p:
                res_mix_mrt = p.starmap(analyze_mixed_mrt, zip(
                    ce_ts_sched_analysis, itertools.repeat(queue)))
                res_mix_mda = p.starmap(analyze_mixed_mda, zip(
                    ce_ts_sched_analysis, itertools.repeat(queue)))
                res_mix_mrda = p.starmap(analyze_mixed_mrda, zip(
                    ce_ts_sched_analysis, itertools.repeat(queue)))

            # Save results
            assert len(res_mix_mrt) == len(ce_ts_sched_analysis)
            assert len(res_mix_mda) == len(ce_ts_sched_analysis)
            assert len(res_mix_mrda) == len(ce_ts_sched_analysis)

            final_results.append(
                [res_mix_mrt[:], res_mix_mda[:], res_mix_mrda[:]])

        # breakpoint()

        # # DEBUG
        # for idxx in range(len(final_results[0][2])):
        #     en_0 = final_results[0][2][idxx]
        #     en_14 = final_results[1][2][idxx]
        #     if en_0 > en_14:
        #         c1 = ce_ts_sched_intra_dict[0][idxx]
        #         c2 = ce_ts_sched_intra_dict[1][idxx]
        #         breakpoint()

        #         # analysis of c1[1][0]
        #         our_mrt_mRda(c1, 0.0)

        ###
        # Store data
        ###
        print(time_now(), '= Store data =')
        folder = "output/4mixedintra/"
        output_filename = ("intra_res_u=" + str(args.u) +
                           "_n=" + str(args.n) + "_g=" + str(args.g) + ".npz")
        check_folder(folder)
        np.savez(folder+output_filename,
                 result=final_results, scenarios=scenarios)

        print(time_now(), '= Done =')

    elif args.j == 100:
        '''Evaluation -- global'''
        ###
        # Load data
        ###
        print(time_now(), "= Load data =")

        # == files from implicit communication evaluation
        filename_implicit = ("output/2implicit/ce_ts_sched_u="+str(args.u)
                             + "_n=" + str(args.n)
                             + "_g=" + str(args.g) + ".npz")
        data_implicit = np.load(filename_implicit, allow_pickle=True)

        ce_ts_sched_implicit = data_implicit.f.gen
        ce_ts_sched_implicit_flat = flatten(ce_ts_sched_implicit)

        # == files from mixed local evaluation
        filename_ml = ("output/4mixedintra/intra_res_u="+str(args.u)
                       + "_n=" + str(args.n)
                       + "_g=" + str(args.g) + ".npz")
        data_ml = np.load(filename_ml, allow_pickle=True)

        res_ml = data_ml.f.result
        sce_ml = data_ml.f.scenarios

        # == files from mixed global evaluation
        filename_mg = ("output/3mixedinter/inter_res_u="+str(args.u)
                       + "_n=" + str(args.n)
                       + "_g=" + str(args.g) + ".npz")
        data_mg = np.load(filename_mg, allow_pickle=True)

        res_mg = data_mg.f.result
        sce_mg = data_mg.f.scenarios

        ###
        # Generate plots
        ###
        print("=Draw plots.=")

        myeva = eva.Evaluation()

        folder = "output/5plots/"
        check_folder(folder)

        # == change our values from dict to direct values
        for ch, _, _ in ce_ts_sched_implicit_flat:
            ch.our0_mrt = ch.our_mrt[0.0]
            ch.our1_mrt = ch.our_mrt[0.3]
            ch.our2_mrt = ch.our_mrt[0.7]
            ch.our3_mrt = ch.our_mrt[1.0]
            #
            ch.our0_mda = ch.our_mda[0.0]
            ch.our1_mda = ch.our_mda[0.3]
            ch.our2_mda = ch.our_mda[0.7]
            ch.our3_mda = ch.our_mda[1.0]
            #
            ch.our0_mrda = ch.our_mrda[0.0]
            ch.our1_mrda = ch.our_mrda[0.3]
            ch.our2_mrda = ch.our_mrda[0.7]
            ch.our3_mrda = ch.our_mrda[1.0]

        # == implicit communication evaluation
        # MRT
        myeva.boxplot_impl(
            [ch for ch, _, _ in ce_ts_sched_implicit_flat],
            folder+"implicit_eval_mrt_u=" +
            str(args.u) + "_n=" + str(args.n) + "_g=" + str(args.g) + ".pdf",
            ['d19_mrt', 'kloda', 'our0_mrt', 'our1_mrt',
                'our2_mrt', 'our3_mrt', 'g21_mrt'],
            ['D19', 'K18', '0.0', '0.3', '0.7', '1.0', 'G21']
        )

        # MRDA
        myeva.boxplot_impl(
            [ch for ch, _, _ in ce_ts_sched_implicit_flat],
            folder+"implicit_eval_mrda_u=" +
            str(args.u) + "_n=" + str(args.n) + "_g=" + str(args.g) + ".pdf",
            ['d19_mrda', 'kloda', 'our0_mrda', 'our1_mrda',
                'our2_mrda', 'our3_mrda', 'g21_mrda'],
            ['D19', 'K18', '0.0', '0.3', '0.7', '1.0', 'G21']
        )

        # MDA
        myeva.boxplot_impl(
            [ch for ch, _, _ in ce_ts_sched_implicit_flat],
            folder+"implicit_eval_mda_u=" +
            str(args.u) + "_n=" + str(args.n) + "_g=" + str(args.g) + ".pdf",
            ['kloda', 'our0_mda', 'our1_mda',
                'our2_mda', 'our3_mda', 'g21_mda'],
            ['K18', '0.0', '0.3', '0.7', '1.0', 'G21']
        )

        # == Mixed local
        # MRT:
        ml_mrt_val = [
            [(e/eimpl) for e, eimpl in zip(entry[0], res_ml[0][0])] for entry in res_ml[1:]]

        # worse than than the case with only implicit
        # breakpoint()
        myeva.boxplot_values(
            list(ml_mrt_val),
            ['1/4', '2/4', '3/4', '4/4'],
            folder+"mixed_local_mrt_u=" +
            str(args.u) + "_n=" + str(args.n) +
            "_g=" + str(args.g) + ".pdf",
            yticks=[1.0, 2.0, 3.0, 4.0],
            ylimits=[0.8, 4.2]
        )

        # MDA:
        ml_mda_val = [
            [(e/eimpl) for e, eimpl in zip(entry[1], res_ml[0][1])] for entry in res_ml[1:]]

        # for entry in ml_mda_val[0]:
        #     if entry < 1:
        #         breakpoint()

        # worse than than the case with only implicit
        # breakpoint()
        myeva.boxplot_values(
            list(ml_mda_val),
            ['1/4', '2/4', '3/4', '4/4'],
            folder+"mixed_local_mda_u=" +
            str(args.u) + "_n=" + str(args.n) +
            "_g=" + str(args.g) + ".pdf",
            yticks=[1.0, 2.0, 3.0, 4.0],
            ylimits=[0.8, 4.2]
        )

        # MRDA:
        ml_mrda_val = [
            [(e/eimpl) for e, eimpl in zip(entry[2], res_ml[0][2])] for entry in res_ml[1:]]

        # worse than than the case with only implicit
        # breakpoint()
        myeva.boxplot_values(
            list(ml_mrda_val),
            ['1/4', '2/4', '3/4', '4/4'],
            folder+"mixed_local_mrda_u=" +
            str(args.u) + "_n=" + str(args.n) +
            "_g=" + str(args.g) + ".pdf",
            yticks=[1.0, 2.0, 3.0, 4.0],
            ylimits=[0.8, 4.2]
        )

        # == Mixed global
        # MRT:
        mg_mrt_val = [
            [(e/eimpl) for e, eimpl in zip(entry[0], res_mg[0][0])] for entry in res_mg[1:]]

        # worse than than the case with only implicit
        # breakpoint()
        myeva.boxplot_values(
            list(mg_mrt_val),
            ['1/4', '2/4', '3/4', '4/4'],
            folder+"mixed_global_mrt_u=" +
            str(args.u) + "_n=" + str(args.n) +
            "_g=" + str(args.g) + ".pdf",
            yticks=[1.0, 2.0, 3.0, 4.0],
            ylimits=[0.8, 4.2]
        )

        # MDA:
        mg_mda_val = [
            [(e/eimpl) for e, eimpl in zip(entry[1], res_mg[0][1])] for entry in res_mg[1:]]

        # worse than than the case with only implicit
        # breakpoint()
        myeva.boxplot_values(
            list(mg_mda_val),
            ['1/4', '2/4', '3/4', '4/4'],
            folder+"mixed_global_mda_u=" +
            str(args.u) + "_n=" + str(args.n) +
            "_g=" + str(args.g) + ".pdf",
            yticks=[1.0, 2.0, 3.0, 4.0],
            ylimits=[0.8, 4.2]
        )

        # MRDA:
        mg_mrda_val = [
            [(e/eimpl) for e, eimpl in zip(entry[2], res_mg[0][2])] for entry in res_mg[1:]]

        # worse than than the case with only implicit
        # breakpoint()
        myeva.boxplot_values(
            list(mg_mrda_val),
            ['1/4', '2/4', '3/4', '4/4'],
            folder+"mixed_global_mrda_u=" +
            str(args.u) + "_n=" + str(args.n) +
            "_g=" + str(args.g) + ".pdf",
            yticks=[1.0, 2.0, 3.0, 4.0],
            ylimits=[0.8, 4.2]
        )

    elif args.j == 101:
        '''Evaluation -- global -- Combined utilizations'''
        ###
        # Load data
        ###
        print(time_now(), "= Load data =")

        ce_ts_sched_implicit = []
        ce_ts_sched_implicit_flat = []

        res_ml = [
            [[], [], []],
            [[], [], []],
            [[], [], []],
            [[], [], []],
            [[], [], []],
        ]
        res_mg = [
            [[], [], []],
            [[], [], []],
            [[], [], []],
            [[], [], []],
            [[], [], []],
        ]

        for util in ['50.0', '60.0', '70.0', '80.0', '90.0']:

            # == files from implicit communication evaluation
            filename_implicit = ("output/2implicit/ce_ts_sched_u="+str(util)
                                 + "_n=" + str(args.n)
                                 + "_g=" + str(args.g) + ".npz")
            data_implicit = np.load(filename_implicit, allow_pickle=True)

            ce_ts_sched_implicit.extend(data_implicit.f.gen)
            ce_ts_sched_implicit_flat.extend(flatten(data_implicit.f.gen))

            # == files from mixed local evaluation
            filename_ml = ("output/4mixedintra/intra_res_u="+str(util)
                           + "_n=" + str(args.n)
                           + "_g=" + str(args.g) + ".npz")
            data_ml = np.load(filename_ml, allow_pickle=True)

            for idx in range(len(res_ml)):
                for idy in range(len(res_ml[idx])):
                    res_ml[idx][idy].extend(data_ml.f.result[idx][idy])

            # == files from mixed global evaluation
            filename_mg = ("output/3mixedinter/inter_res_u="+str(util)
                           + "_n=" + str(args.n)
                           + "_g=" + str(args.g) + ".npz")
            data_mg = np.load(filename_mg, allow_pickle=True)

            for idx in range(len(res_mg)):
                for idy in range(len(res_mg[idx])):
                    res_mg[idx][idy].extend(data_mg.f.result[idx][idy])

        ###
        # Generate plots
        ###
        print("=Draw plots.=")

        myeva = eva.Evaluation()

        folder = "output/6plots_combined_util/"
        check_folder(folder)

        # == change our values from dict to direct values
        for ch, _, _ in ce_ts_sched_implicit_flat:
            ch.our0_mrt = ch.our_mrt[0.0]
            ch.our1_mrt = ch.our_mrt[0.3]
            ch.our2_mrt = ch.our_mrt[0.7]
            ch.our3_mrt = ch.our_mrt[1.0]
            #
            ch.our0_mda = ch.our_mda[0.0]
            ch.our1_mda = ch.our_mda[0.3]
            ch.our2_mda = ch.our_mda[0.7]
            ch.our3_mda = ch.our_mda[1.0]
            #
            ch.our0_mrda = ch.our_mrda[0.0]
            ch.our1_mrda = ch.our_mrda[0.3]
            ch.our2_mrda = ch.our_mrda[0.7]
            ch.our3_mrda = ch.our_mrda[1.0]

        # == implicit communication evaluation
        # MRT
        myeva.boxplot_impl(
            [ch for ch, _, _ in ce_ts_sched_implicit_flat],
            folder+"implicit_eval_mrt" + "_n=" +
            str(args.n) + "_g=" + str(args.g) + ".pdf",
            ['d19_mrt', 'kloda', 'our0_mrt', 'our1_mrt',
                'our2_mrt', 'our3_mrt', 'g21_mrt'],
            ['D19', 'K18', '0.0', '0.3', '0.7', '1.0', 'G21'],
            ylabel='Latency Reduction (%)'
        )

        # MRDA
        myeva.boxplot_impl(
            [ch for ch, _, _ in ce_ts_sched_implicit_flat],
            folder+"implicit_eval_mrda" + "_n=" +
            str(args.n) + "_g=" + str(args.g) + ".pdf",
            ['d19_mrda', 'kloda', 'our0_mrda', 'our1_mrda',
                'our2_mrda', 'our3_mrda', 'g21_mrda'],
            ['D19', 'K18', '0.0', '0.3', '0.7', '1.0', 'G21'],
            ylabel='Latency Reduction (%)'
        )

        # MDA
        myeva.boxplot_impl(
            [ch for ch, _, _ in ce_ts_sched_implicit_flat],
            folder+"implicit_eval_mda" + "_n=" +
            str(args.n) + "_g=" + str(args.g) + ".pdf",
            ['kloda', 'our0_mda', 'our1_mda',
                'our2_mda', 'our3_mda', 'g21_mda'],
            ['K18', '0.0', '0.3', '0.7', '1.0', 'G21'],
            ylabel='Latency Reduction (%)'
        )

        # == Mixed local
        # MRT:
        ml_mrt_val = [
            [(e/eimpl)*100 for e, eimpl in zip(entry[0], res_ml[0][0])] for entry in res_ml[1:]]

        # worse than than the case with only implicit
        # breakpoint()
        myeva.boxplot_values(
            list(ml_mrt_val),
            ['1/4', '2/4', '3/4', '4/4'],
            folder+"mixed_local_mrt" + "_n=" + str(args.n) +
            "_g=" + str(args.g) + ".pdf",
            yticks=[100.0, 200.0, 300.0, 400.0],
            ylimits=[80.0, 420.0],
            ylabel='Normalized Latency (%)'
        )

        # MDA:
        ml_mda_val = [
            [(e/eimpl)*100 for e, eimpl in zip(entry[1], res_ml[0][1])] for entry in res_ml[1:]]

        # for entry in ml_mda_val[0]:
        #     if entry < 1:
        #         breakpoint()

        # worse than than the case with only implicit
        # breakpoint()
        myeva.boxplot_values(
            list(ml_mda_val),
            ['1/4', '2/4', '3/4', '4/4'],
            folder+"mixed_local_mda" + "_n=" + str(args.n) +
            "_g=" + str(args.g) + ".pdf",
            yticks=[100.0, 200.0, 300.0, 400.0],
            ylimits=[80.0, 420.0],
            ylabel='Normalized Latency (%)'
        )

        # MRDA:
        ml_mrda_val = [
            [(e/eimpl)*100 for e, eimpl in zip(entry[2], res_ml[0][2])] for entry in res_ml[1:]]

        # worse than than the case with only implicit
        # breakpoint()
        myeva.boxplot_values(
            list(ml_mrda_val),
            ['1/4', '2/4', '3/4', '4/4'],
            folder+"mixed_local_mrda" + "_n=" + str(args.n) +
            "_g=" + str(args.g) + ".pdf",
            yticks=[100.0, 200.0, 300.0, 400.0],
            ylimits=[80.0, 420.0],
            ylabel='Normalized Latency (%)'
        )

        # == Mixed global
        # MRT:
        mg_mrt_val = [
            [(e/eimpl)*100 for e, eimpl in zip(entry[0], res_mg[0][0])] for entry in res_mg[1:]]

        # worse than than the case with only implicit
        # breakpoint()
        myeva.boxplot_values(
            list(mg_mrt_val),
            ['1/4', '2/4', '3/4', '4/4'],
            folder+"mixed_global_mrt" + "_n=" + str(args.n) +
            "_g=" + str(args.g) + ".pdf",
            yticks=[100.0, 200.0, 300.0, 400.0],
            ylimits=[80.0, 420.0],
            ylabel='Normalized Latency (%)'
        )

        # MDA:
        mg_mda_val = [
            [(e/eimpl)*100 for e, eimpl in zip(entry[1], res_mg[0][1])] for entry in res_mg[1:]]

        # worse than than the case with only implicit
        # breakpoint()
        myeva.boxplot_values(
            list(mg_mda_val),
            ['1/4', '2/4', '3/4', '4/4'],
            folder+"mixed_global_mda" + "_n=" + str(args.n) +
            "_g=" + str(args.g) + ".pdf",
            yticks=[100.0, 200.0, 300.0, 400.0],
            ylimits=[80.0, 420.0],
            ylabel='Normalized Latency (%)'
        )

        # MRDA:
        mg_mrda_val = [
            [(e/eimpl)*100 for e, eimpl in zip(entry[2], res_mg[0][2])] for entry in res_mg[1:]]

        # worse than than the case with only implicit
        # breakpoint()
        myeva.boxplot_values(
            list(mg_mrda_val),
            ['1/4', '2/4', '3/4', '4/4'],
            folder+"mixed_global_mrda" + "_n=" + str(args.n) +
            "_g=" + str(args.g) + ".pdf",
            yticks=[100.0, 200.0, 300.0, 400.0],
            ylimits=[80.0, 420.0],
            ylabel='Normalized Latency (%)'
        )


if __name__ == '__main__':
    main()
