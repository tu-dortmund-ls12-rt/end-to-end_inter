"""End-to-End (e2e) Analysis.

Source:
https://github.com/tu-dortmund-ls12-rt/end-to-end"""
import math
import utilities.task
import utilities.augmented_job_chain as aug


debug_flag = False  # flag to have breakpoint() when errors occur


class Analyzer:
    """Analyzer to do the analysis."""

    def __init__(self, e_id=0):
        """Creates an analyzer represented by ID."""
        self.id = e_id  # unique identifier

    @staticmethod
    def determine_hyper_period(task_set):
        """Determine the hyperperiod of task_set."""
        # Collect periods.
        periods = []
        for task in task_set:
            if task.period not in periods:
                periods.append(task.period)
        # Compute least common multiple = hyperperiod.
        lcm = periods[0]
        for i in periods[1:]:
            lcm = int(lcm * i / math.gcd(lcm, i))
        return lcm

    @staticmethod
    def workload(period, wcet, time):
        """Workload function for TDA.

        Help function for tda().
        """
        return wcet * math.ceil(float(time) / period)

    def tda(self, task, hp_tasks):
        """Implementation of TDA to calculate worst-case response time.

        Source:
        https://github.com/kuanhsunchen/MissRateSimulator/blob/master/TDA.py
        """
        c = task.wcet  # WCET
        r = c  # WCRT
        while True:
            i = 0  # interference
            for itask in hp_tasks:
                i = i + self.workload(itask.period, itask.wcet, r)

            if r < i + c:
                r = i + c
            else:
                return r

    ###
    # Our analyses from 'Timing Analysis of Asynchronized Distributed
    # Cause-Effect Chains' (2021).
    ###

    def max_age_our(self, schedule, task_set, chain, max_phase, hyper_period,
                    reduced=False):
        """Our maximum data age time analysis.

        We construct all immediate backward augmented job chains and then
        choose the maximal length of them.
        Note: The schedule has to be build beforehand with the event scheduler.
        """
        # Compute maximal first read.
        first_jobs = []
        for task in task_set:
            first_jobs.append(schedule.get(task)[0])
        max_first_read = max(first_jobs, key=lambda first_job: first_job[0])[0]

        # Construct all valid immediate backward augmented job chains.
        candidates = []

        # Position for the last job in the chain.
        position = -1
        while True:
            # We start with position = 0 (1st job).
            position += 1

            # Checking for mistakes.
            if len(schedule.get(chain.chain[-1])) < position:
                if debug_flag:
                    breakpoint()
                else:
                    return

            # Last job in the job chain:
            next_job = schedule.get(chain.chain[-1])[position]

            # Find actuation.
            if reduced:
                actuation = schedule.get(chain.chain[-1])[position][1]
            else:
                actuation = schedule.get(chain.chain[-1])[position+1][1]

            # Construct augmented job chain with help function.
            job_chain = self.imm_bw_jc(next_job, chain.length()-1, schedule,
                                       chain, key=0)

            # Handle incomplete job chains.
            if job_chain is None:
                continue

            # Define external activity.
            ext_activity = job_chain[0][0]

            # Find first job after ext_activity
            job_after_ext_activity = None
            flag = False
            for job in schedule.get(chain.chain[0]):
                if job[0] > ext_activity:
                    flag = True
                    break
            if flag is False:  # no event after ext_activity could be found
                if debug_flag:
                    breakpoint()
                else:
                    return
            else:
                job_after_ext_activity = job

            # Check if the augmented job chain is valid.
            if job_after_ext_activity[0] > max_first_read:
                pass
            else:
                continue

            # End condition.
            if ext_activity < max_phase + 2*hyper_period:
                pass
            else:
                break

            # Add augmented job chain to candidates.
            candidates.append(aug.AugJobChain(
                job_chain=job_chain,
                ext_activity=ext_activity,
                actuation=actuation))

        # Compare length of candidates.
        max_cand = max(candidates, key=lambda cand: cand.length())
        max_length = max_cand.length()

        # Results.
        if reduced:
            chain.our_red_age = max_length
        else:
            chain.our_age = max_length
        return max_length

    def imm_bw_jc(self, current_job, c_len, schedule, chain, key=0):
        """Compute immediate forward job chain recursively.

        Used as help function for max_age_our(). Returns None if the job chain
        is incomplete.
        """
        # Initial case.
        if key == 0:
            res = self.imm_bw_jc(current_job, c_len, schedule, chain,
                                 key=key+1)
            if res is None:  # incomplete job chain
                return None
            else:
                return res + [current_job]  # build from right to left

        # Intermediate cases. Adding one job.
        elif key <= c_len:
            flag_found = False
            # Search in reversed schedule for next job.
            for next_job in schedule.get(chain.chain[-key-1])[::-1]:
                if current_job[0] >= next_job[1]:  # condition for next job
                    flag_found = True
                    break
            # Case: No job was found.
            if flag_found is False:
                return None  # indicate incomplete job chain
            # Case: Job was found.
            else:
                res = self.imm_bw_jc(next_job, c_len, schedule, chain,
                                     key=key+1)
                if res is None:  # incomplete job chain.
                    return None
                else:
                    return res + [next_job]  # build from right to left

        # Final case. (key > c_len)
        else:
            return []

    def reaction_our(self, schedule, task_set, chain, max_phase, hyper_period):
        """Our maximum reaction time analysis.

        We construct all immediate forward augmented job chains and then
        choose the maximal length of them.
        Note: The schedule has to be build beforehand with the event scheduler.
        """
        # Compute maximal first read.
        first_jobs = []
        for task in task_set:
            first_jobs.append(schedule.get(task)[0])
        max_first_read = max(first_jobs, key=lambda first_job: first_job[0])[0]

        # Construct all valid immediate forward augmented job chains.
        candidates = []

        # Position for the first job in the chain.
        position = 0
        while True:
            # We start with position = 1 (2nd job) because we need one previous
            # job for the definition of external activity.
            position += 1

            # Checking for mistakes.
            if len(schedule.get(chain.chain[0])) < position:
                if debug_flag:
                    breakpoint()
                else:
                    return

            # First job in the job chain.
            next_job = schedule.get(chain.chain[0])[position]

            # External activity.
            ext_activity = schedule.get(chain.chain[0])[position-1][0]

            # Check if valid
            if next_job[0] > max_first_read:
                pass
            else:
                continue

            # End condition.
            if ext_activity < max_phase + 2*hyper_period:
                pass
            else:
                break

            # Construct augmented job chain with help function.
            job_chain = self.imm_fw_jc(next_job, chain.length()-1, schedule,
                                       chain, key=0)

            # Compute actuation.
            actuation = job_chain[-1][1]

            # Add augmented job chain to candidates.
            candidates.append(aug.AugJobChain(
                job_chain=job_chain,
                ext_activity=ext_activity,
                actuation=actuation))

        # Compare length of candidates.
        max_cand = max(candidates, key=lambda cand: cand.length())
        max_length = max_cand.length()

        # Results.
        chain.our_react = max_length
        return max_length

    def imm_fw_jc(self, current_job, c_len, schedule, chain, key=0):
        """Compute immediate forward job chain recursively

        Used as help function for reaction_our().
        """
        # Initial case.
        if key == 0:
            # Build from left to right:
            return [current_job] + self.imm_fw_jc(current_job, c_len, schedule,
                                                  chain, key=key+1)

        # Intermediate cases. Adding one job.
        elif key <= c_len:
            flag_found = False
            # Search for next job.
            for next_job in schedule.get(chain.chain[key]):
                if current_job[1] <= next_job[0]:  # condition for next job
                    flag_found = True
                    break
            # Case: No job was found.
            if flag_found is False:
                print("ERROR")
            # Case: Job was found.
            else:
                return [next_job] + self.imm_fw_jc(next_job, c_len, schedule,
                                                   chain, key=key+1)

        # Final case. (key > c_len)
        else:
            return []

    def reaction_inter_our(self, chain_set):
        """Our maximum reaction time analysis for interconnected cause-effect
        chains.

        Input: chain_set is a list of cause-effect chains with entry at
        interconnected.
        Note: The chains have to be analyzed by our single ECU maximum reaction
        time analysis beforehand. ( reaction_our() )
        """
        for chain in chain_set:
            inter_our_react = 0  # total reaction time
            for i in range(0, len(chain.interconnected)):
                # Case: i is a communication task.
                if isinstance(chain.interconnected[i], utilities.task.Task):
                    inter_our_react += (chain.interconnected[i].period
                                        + chain.interconnected[i].rt)
                # Case: i is a cause-effect chain.
                else:
                    inter_our_react += chain.interconnected[i].our_react
            # Store result.
            chain.inter_our_react = inter_our_react

    def max_age_inter_our(self, chain_set, reduced=False):
        """Our reduced maximum data age analysis for interconnected
        cause-effect chains.

        Input: chain_set is a list of cause-effect chains with entry at
        interconnected.
        Note: The chains have to be analyzed by our single ECU maximum data age
        analysis beforehand. ( max_age_our() and max_age_our(reduced=True) )
        """
        for chain in chain_set:
            m = len(chain.interconnected)  # chain length
            inter_our_red_age = 0  # total data age
            for i in range(0, m-1):
                # Case: i is a communication task.
                if isinstance(chain.interconnected[i], utilities.task.Task):
                    inter_our_red_age += (chain.interconnected[i].period
                                          + chain.interconnected[i].rt)
                # Case: i is a cause-effect chain.
                else:
                    inter_our_red_age += chain.interconnected[i].our_age

            # Handle the last cause-effect chain in the list.
            if reduced:
                inter_our_red_age += chain.interconnected[m-1].our_red_age
            else:
                inter_our_red_age += chain.interconnected[m-1].our_age

            # Store result.
            chain.inter_our_red_age = inter_our_red_age

    ###
    # Davare analysis from 'Period Optimization for Hard Real-time Distributed
    # Automotive Systems' (2007).
    ###

    def davare(self, chain_sets):
        """End-to-end latency analysis from Davare.

        Input: chain_sets is a list of lists of chains.
        """
        for chain_set in chain_sets:
            for chain in chain_set:
                # Compute the latency for chain.
                latency = 0
                for task in chain.chain:
                    latency += task.period + task.rt
                # Store result.
                chain.davare = latency

    def davare_single(self, chain):  # Added in October 21
        """End-to-end latency analysis from Davare.

        Input: One chain.
        """
        latency = 0
        for task in chain.chain:
            latency += task.period + task.rt
        # Store result.
        chain.davare = latency

        return latency

    ###
    # Duerr analysis from 'End-to-End Timing Analysis of Sporadic Cause-Effect
    # Chains in Distributed Systems' (2019).
    ###
    def reaction_duerr(self, chain_sets):
        """Maximum reaction time analysis from Duerr.

        Input: chain_sets is a list of lists of chains.
        """
        for chain_set in chain_sets:
            for chain in chain_set:
                # Compute latency.
                latency = chain.chain[-1].rt + chain.chain[0].period
                for task, next_task in zip(chain.chain[:-1], chain.chain[1:]):
                    if (task.priority > next_task.priority
                            or next_task.message or task.message):
                        part2 = task.rt
                    else:
                        part2 = 0
                    latency += max(task.rt, next_task.period + part2)
                # Store result.
                chain.duerr_react = latency

    def reaction_duerr_single(self, chain):  # added in Oct 21
        # Compute latency.
        latency = chain.chain[-1].rt + chain.chain[0].period
        for task, next_task in zip(chain.chain[:-1], chain.chain[1:]):
            if (task.priority > next_task.priority
                    or next_task.message or task.message):
                part2 = task.rt
            else:
                part2 = 0
            latency += max(task.rt, next_task.period + part2)
        # Store result.
        chain.duerr_react = latency

        return latency

    def age_duerr(self, chain_sets):
        """Maximum data age analysis from Duerr.

        Input: chain_sets is a list of lists of chains.
        """
        for chain_set in chain_sets:
            for chain in chain_set:
                # Compute latency.
                latency = chain.chain[-1].rt
                for task, next_task in zip(chain.chain[:-1], chain.chain[1:]):
                    if (task.priority > next_task.priority
                            or next_task.message or task.message):
                        part2 = task.rt
                    else:
                        part2 = 0
                    latency += task.period + part2
                # Store result.
                chain.duerr_age = latency

    def age_duerr_single(self, chain):  # added in Oct 21
        # Compute latency.
        latency = chain.chain[-1].rt
        for task, next_task in zip(chain.chain[:-1], chain.chain[1:]):
            if (task.priority > next_task.priority
                    or next_task.message or task.message):
                part2 = task.rt
            else:
                part2 = 0
            latency += task.period + part2
        # Store result.
        chain.duerr_age = latency

        return latency

    ###
    # Kloda analysis from 'Latency analysis for data chains of real-time
    # periodic tasks' (2018).
    ###

    def kloda(self, chain, hyper_period):
        """Kloda analysis for the single ECU case with synchronous releases.

        Input: chain is one cause-effect chain. hyper_period is the hyperperiod
        of the underlying task set.
        """
        for release_first_task_in_chain in range(0, max(1, hyper_period),
                                                 chain.chain[0].period):
            # Compute latency for a given first job.
            kloda = self.kloda_rec(chain.chain, release_first_task_in_chain,
                                   beginning=True)
            # Compare and store the results.
            if chain.kloda < kloda:
                chain.kloda = kloda
        return chain.kloda

    def kloda_rec(self, chain, rel_producer, beginning=True):
        """Recursive function to compute the reaction time by klodas analysis.

        Note: The additional period is already added with the beginning=True
        option.
        """
        add = 0
        # Additional period at the beginning. (This is only done for the
        # initial case.)
        if beginning:
            add += chain[0].period

        producer_task = chain[0]  # producer

        # Final case
        if len(chain) == 1:
            return producer_task.rt + add

        rem_chain = chain[1::]  # remaining chain
        consumer_task = rem_chain[0]  # consumer

        # Intermediate cases. Compute difference between producer and consumer.
        q = 0
        # Case: Producer has lower priority than consumer, i.e., the priority
        # value is higher. Note: We do not implement a processor change since
        # we consider only the single ECU case. (Kloda cannot be applied to
        # asynchronized ECUs.)
        if (producer_task.priority > consumer_task.priority):
            q = producer_task.rt
        rel_consumer = (math.ceil((rel_producer + q) / consumer_task.period)
                        * consumer_task.period)
        return (add + rel_consumer - rel_producer
                + self.kloda_rec(rem_chain, rel_consumer, beginning=False))
