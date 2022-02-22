"""Our End-To-End analysis.
- implicit communication only
- periodic tasks 
"""

import itertools
import utilities.analyzer

# Method to compute the hyperperiod
compute_hyper = utilities.analyzer.Analyzer.determine_hyper_period


class re_we_analyzer():
    def __init__(self, bcet_schedule, wcet_schedule, hyperperiod):
        self.bc = bcet_schedule
        self.wc = wcet_schedule
        self.hyperperiod = hyperperiod

    def _get_entry(self, nmb, lst, tsk):
        '''get nmb-th entry of the list lst with task tsk.'''
        if nmb < 0:  # Case: out of range
            raise IndexError('nbm<0')
        # Case: index too high, has to be made smaller # TODO not sure if this is a good idea since the last entries could be wrong depending on the implementation of the scheduler ...
        elif nmb >= len(lst):
            # check one hyperperiod earlier
            # make new_nmb an integer value
            div, rem = divmod(self.hyperperiod, tsk.period)
            assert rem == 0
            new_nmb = nmb - div
            # add one hyperperiod
            # TODO this is not very efficient since we need the values several times.
            return [self.hyperperiod + entry for entry in self._get_entry(new_nmb, lst, tsk)]
        else:  # Case: entry can be used
            try:
                return lst[nmb]
            except:
                breakpoint()

    def remin(self, task, nmb):
        '''returns the upper bound on read-event of the nbm-th job of a task.'''
        lst = self.bc[task]  # list that has the read-even minimum
        # choose read-event from list
        return self._get_entry(nmb, lst, task)[0]

    def remax(self, task, nmb):
        '''returns the upper bound on read-event of the nbm-th job of a task.'''
        lst = self.wc[task]  # list that has the read-even maximum
        # choose read-event from list
        return self._get_entry(nmb, lst, task)[0]

    def wemin(self, task, nmb):
        '''returns the upper bound on read-event of the nbm-th job of a task.'''
        lst = self.bc[task]  # list that has the write-even minimum
        # choose write-event from list
        return self._get_entry(nmb, lst, task)[1]

    def wemax(self, task, nmb):
        '''returns the upper bound on read-event of the nbm-th job of a task.'''
        lst = self.wc[task]  # list that has the write-even maximum
        # choose write-event from list
        return self._get_entry(nmb, lst, task)[1]

    def find_next_fw(self, curr_task_wc, nxt_task_bc, curr_index):
        '''Find next index for the abstract representation in forward manner.'''
        # wemax of current task
        curr_wemax = self.wemax(curr_task_wc, curr_index)
        curr_rel = curr_task_wc.phase + curr_index * \
            curr_task_wc.period  # release of current task

        for idx in itertools.count():
            idx_remin = self.remin(nxt_task_bc, idx)

            if (
                idx_remin >= curr_wemax  # first property
                # second property (lower priority value means higher priority!)
                or (curr_task_wc.priority < nxt_task_bc.priority and idx_remin >= curr_rel)
            ):
                return idx

    def find_next_bw(self, curr_task_bc, prev_task_wc, curr_index):
        '''Find next index for the abstract representation in backward manner.
        Note: returns -1 if no index can be found.'''
        # remin of current task
        curr_remin = self.remin(curr_task_bc, curr_index)

        # # wemax of current task
        # curr_wemax = self.wemax(curr_task_wc, curr_index)
        # curr_rel = curr_task_wc.phase + curr_index * \
        #     curr_task_wc.period  # release of current task

        for idxprev, idx in zip(itertools.count(start=-1), itertools.count(start=0)):
            # wemax and release of job idx of prev_task
            idx_wemax = self.wemax(prev_task_wc, idx)
            idx_rel = prev_task_wc.phase + idx * prev_task_wc.period

            if not (
                curr_remin >= idx_wemax  # first property
                # second property
                or (prev_task_wc.priority < curr_task_bc.priority and curr_remin >= idx_rel)
            ):
                # if properties are NOT fulfilled, return the previous index
                return idxprev

    def len_abstr(self, abstr, last_tsk_wc, first_tsk_bc):
        '''Length of an abstract representation.'''
        return self.wemax(last_tsk_wc, abstr[-1])-self.remin(first_tsk_bc, abstr[0])

    def len_abstr_reduced(self, abstr, last_tsk_wc, first_tsk_bc):
        '''REDUCED Length of an abstract representation.'''
        return self.wemax(last_tsk_wc, abstr[-2])-self.remin(first_tsk_bc, abstr[0])

    def incomplete_bound(self, abstr, last_tsk_wc, first_tsk_bc):
        '''Second backward bound'''
        return self.wemax(last_tsk_wc, abstr[-1]) - self.remin(first_tsk_bc, 0)

    def incomplete_bound_reduced(self, abstr, last_tsk_wc, first_tsk_bc):
        '''REDUCED Second backward bound'''
        return self.wemax(last_tsk_wc, abstr[-2]) - self.remin(first_tsk_bc, 0)


def max_reac_local(chain, task_set_wcet, schedule_wcet, task_set_bcet, schedule_bcet):
    '''Main method for maximum reaction time.

    We construct all abstract represenations and compute the maximal length among them.
    - chain: cause-effect chain as list of tasks
    - task_set: the task set of the ECU that the ce chain lies on
    - schedule: the schedule of task_set (simulated beforehand)

    we distinguish between bcet and wcet task set and schedule.'''

    if chain.length() == 0:  # corner case
        return 0

    # Make analyzer
    ana = re_we_analyzer(schedule_bcet, schedule_wcet,
                         compute_hyper(task_set_wcet))

    # Chain of indeces that describes the cause-effect chain
    index_chain = [task_set_wcet.index(entry) for entry in chain.chain]

    # Set of all abstract representations
    all_abstr = []

    # useful values for break-condition
    hyper = compute_hyper(task_set_wcet)
    max_phase = max([task.phase for task in task_set_wcet])

    for idx in itertools.count():
        # Compute idx-th abstract integer representation.
        abstr = []
        abstr.append(idx)  # first entry
        abstr.append(idx+1)  # second entry

        for idtsk, nxt_idtsk in zip(index_chain[:-1], index_chain[1:]):
            abstr.append(ana.find_next_fw(
                task_set_wcet[idtsk], task_set_bcet[nxt_idtsk], abstr[-1]))  # intermediate entries

        abstr.append(abstr[-1])  # last entry

        assert len(abstr) == chain.length() + 2

        all_abstr.append(abstr[:])

        # Break loop
        if (chain.chain[0].phase + idx * chain.chain[0].period) >= (max_phase + 2*hyper):
            break

        # print([task_set_wcet[i].priority for i in index_chain])

        # print([(schedule_bcet[task_set_bcet[i]][j][0], schedule_wcet[task_set_wcet[i]][j][1])
        #       for i, j in zip(index_chain, abstr[1:-1])])

        # breakpoint()

    # maximal length
    max_length = max([ana.len_abstr(abstr, task_set_wcet[index_chain[-1]],
                     task_set_bcet[index_chain[0]]) for abstr in all_abstr] + [0])
    chain.our_new_local_mrt = max_length
    return max_length


def max_age_local(chain, task_set_wcet, schedule_wcet, task_set_bcet, schedule_bcet):
    '''Main method for maximum data age.
    Returns a list of two values. First is the maximum data age bound, second is
    the maximum REDUCED data age bound.

    We construct all abstract represenations and compute the maximal length among them.
    - chain: cause-effect chain as list of tasks
    - task_set: the task set of the ECU that the ce chain lies on
    - schedule: the schedule of task_set (simulated beforehand)

    we distinguish between bcet and wcet task set and schedule.'''

    if chain.length() == 0:  # corner case
        return (0, 0)

    # Make analyzer
    ana = re_we_analyzer(schedule_bcet, schedule_wcet,
                         compute_hyper(task_set_wcet))

    # Chain of indeces that describes the cause-effect chain
    index_chain = [task_set_wcet.index(entry) for entry in chain.chain]

    # Set of all abstract representations
    all_abstr = []
    complete_abstr = []
    incomplete_abstr = []

    # useful values for break-condition
    hyper = compute_hyper(task_set_wcet)
    max_phase = max([task.phase for task in task_set_wcet])

    for idx in itertools.count(start=1):
        # Compute idx-th abstract integer representation.
        # In backwards manner!
        # We start be filling the tuple abstr from left to right and switch the direction afterwards

        abstr = []
        abstr.append(idx)  # last entry
        abstr.append(idx-1)  # second last entry

        for idtsk, prev_idtsk in zip(index_chain[::-1][:-1], index_chain[::-1][1:]):
            indx = ana.find_next_bw(
                task_set_bcet[idtsk], task_set_wcet[prev_idtsk], abstr[-1])
            abstr.append(indx)  # intermediate entries

            if indx == -1:  # check if incomplete
                break

        abstr.append(abstr[-1])  # first entry

        # Turn around the chain
        abstr = abstr[::-1]

        # assert len(abstr) == chain.length() + 2 # Note: this is not true anymore, since we have imcomplete chains.

        all_abstr.append(abstr[:])
        if abstr[0] == -1:
            incomplete_abstr.append(abstr[:])
        else:
            complete_abstr.append(abstr[:])

        # Break loop
        if (chain.chain[0].phase + abstr[0] * chain.chain[0].period) >= (max_phase + 2*hyper):
            break

        # print([task_set_wcet[i].priority for i in index_chain])

        # print([(schedule_bcet[task_set_bcet[i]][j][0], schedule_wcet[task_set_wcet[i]][j][1])
        #       for i, j in zip(index_chain, abstr[1:-1])])

        # breakpoint()

    # maximal length
    max_length_compl = max(
        [ana.len_abstr(abstr, task_set_wcet[index_chain[-1]],
                       task_set_bcet[index_chain[0]]) for abstr in complete_abstr] + [0]
    )
    max_length_incompl = max(
        [ana.incomplete_bound(abstr, task_set_wcet[index_chain[-1]],
                              task_set_bcet[index_chain[0]]) for abstr in incomplete_abstr] + [0]
    )
    max_length = max(max_length_compl, max_length_incompl)

    # maximal reduced length
    max_length_compl_red = max(
        [ana.len_abstr_reduced(abstr, task_set_wcet[index_chain[-1]],
                               task_set_bcet[index_chain[0]]) for abstr in complete_abstr] + [0]
    )
    max_length_incompl_red = max(
        [ana.incomplete_bound_reduced(abstr, task_set_wcet[index_chain[-1]],
                                      task_set_bcet[index_chain[0]]) for abstr in incomplete_abstr] + [0]
    )
    max_length_red = max(max_length_compl_red, max_length_incompl_red)

    chain.our_new_local_mda = max_length
    chain.our_new_local_mrda = max_length_red
    return (max_length, max_length_red)


def execution_zero_schedule(task_set):
    '''Since the dispatcher can only handle execution time >0, we generate a "faked" schedule.'''
    hyperperiod = compute_hyper(task_set)
    max_phase = max([task.phase for task in task_set])

    # Initialize result dictionary.
    result = dict()
    for task in task_set:
        result[task] = []

    for task in task_set:
        curr_time = task.phase
        while curr_time <= max_phase + 2 * hyperperiod:
            # start and finish directly at release
            result[task].append((curr_time, curr_time))
            curr_time += task.period

    return result


class rel_dl_analyzer:
    def __init__(self):
        pass

    def rel(self, task, nmb):
        return task.phase + nmb * task.period

    def dl(self, task, nmb):
        return self.rel(task, nmb) + task.deadline

    def find_next_rel(self, task, bound):
        '''Find the index of the first job with release after the bound.'''
        for idx in itertools.count():
            if self.rel(task, idx) >= bound:
                return idx

    def find_prev_dl(self, task, bound):
        '''Find the index of the latest job with deadline before the bound.
        Note: returns -1 if no such job can be found.'''
        for idx, idx_next in zip(itertools.count(start=-1), itertools.count(start=0)):
            if not (self.dl(task, idx_next) <= bound):
                return idx


def mrt_let(chain, task_set):
    '''Compute maximum reaction time when all tasks adhere to LET.
    This is an exact analysis.'''

    # Make analyzer
    ana = rel_dl_analyzer()

    # Set of forward chains
    fw = []

    # useful values for break-condition and valid check
    hyper = compute_hyper(task_set)
    max_phase = max([task.phase for task in task_set])
    max_first_read_event = max([ana.rel(task, 0) for task in task_set])

    for idx in itertools.count():
        # check valid
        if not (ana.rel(chain.chain[0], idx + 1) >= max_first_read_event):
            continue

        # Compute idx-th fw chain
        fwidx = []

        fwidx.append(ana.rel(chain.chain[0], idx))  # external activity
        fwidx.append(idx+1)  # first job

        for curr_task, nxt_task in zip(chain.chain[:-1], chain.chain[1:]):
            fwidx.append(  # add next release
                ana.find_next_rel(nxt_task, ana.dl(curr_task, fwidx[-1]))
            )

        fwidx.append(ana.dl(chain.chain[-1], fwidx[-1]))  # actuation

        assert len(fwidx) == chain.length() + 2

        fw.append(fwidx[:])

        # break condition
        if ana.rel(chain.chain[0], idx) >= (max_phase + 2*hyper):
            break

    max_length = max(fwidx[-1] - fwidx[0] for fwidx in fw)

    chain.mrt_let = max_length
    return max_length


def mda_let(chain, task_set):
    '''Compute maximum data age and maximum reduced data age when all tasks adhere to LET.
    This is an exact analysis.'''

    # Make analyzer
    ana = rel_dl_analyzer()

    # Set of backward chains
    bw = []

    # useful values for break-condition and valid check
    hyper = compute_hyper(task_set)
    max_phase = max([task.phase for task in task_set])
    max_first_read_event = max([ana.rel(task, 0) for task in task_set])

    for idx in itertools.count(start=1):

        # Compute idx-th bw chain
        bwidx = []
        # Note: fill by append and reverse afterwards.

        bwidx.append(ana.dl(chain.chain[-1], idx))  # actuation
        bwidx.append(idx-1)  # last job

        for curr_task, prev_task in zip(chain.chain[::-1][:-1], chain.chain[::-1][1:]):
            indx = ana.find_prev_dl(prev_task, ana.rel(curr_task, bwidx[-1]))
            bwidx.append(indx)  # add next release

            # Check if incomplete:
            if indx == -1:
                break

        # Remove if incomplete:
        if bwidx[-1] == -1:
            continue

        bwidx.append(ana.rel(chain.chain[0], bwidx[-1]))  # actuation

        # Turn around the chain:
        bwidx = bwidx[::-1]

        # Note: here we only have complete chains. The others are removed already
        assert len(bwidx) == chain.length() + 2

        bw.append(bwidx[:])

        # check valid
        if not (ana.rel(chain.chain[0], bwidx[1]) >= max_first_read_event):
            continue

        # break condition
        if ana.rel(chain.chain[0], bwidx[1]) >= (max_phase + 2*hyper):
            break

    # maximal length
    max_length = max(bwidx[-1] - bwidx[0] for bwidx in bw)

    # maximal reduced length
    max_length_red = max(
        ana.dl(chain.chain[-1], bwidx[-2]) - bwidx[0] for bwidx in bw)

    chain.mda_let = max_length
    chain.mrda_let = max_length_red

    return (max_length, max_length_red)
