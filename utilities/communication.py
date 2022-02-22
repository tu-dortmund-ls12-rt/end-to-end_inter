"""Creation of communication tasks.

Source:
https://github.com/tu-dortmund-ls12-rt/end-to-end"""

import numpy as np
import random
import math
import utilities.task as task

# CAN-bus description.
CAN_BUS = {
    'MESSAGE_BIT': 130,
    'BANDWIDTH_MBPS': 1
}


# Main function
def generate_communication_taskset(num_tasks, min_period, max_period,
                                   rounded=False, max_trials=100):
    """Generate a set of communication tasks.

    num_tasks: number of communication tasks in the set
    min_period: lower bound for the periods
    max_period: upper bound for the periods
    rounded: flag for rounding the period values
    max_trials: maximal tries to create a proper task set
    """
    # If no proper taskset was created, we try again.
    trials = 0
    while (trials < max_trials):
        # Create candidates.
        taskset = generate_communication_candidate_taskset(
            num_tasks, min_period, max_period, rounded)
        taskset = sorted(taskset, key=lambda x: x.priority)

        # Compute WCRT.
        if non_preemptive_response_time(taskset):
            return taskset
        trials += 1
    # The creation failed too many times.
    return False


# Help functions
def generate_communication_candidate_taskset(num_tasks, min_period, max_period,
                                             rounded=False):
    """Generate candidate for the set of communication tasks."""
    taskset = []
    # Generate WCET and periods.
    wcet = (float(CAN_BUS['MESSAGE_BIT'])/CAN_BUS['BANDWIDTH_MBPS'])/10**3
    periods = np.exp(np.random.uniform(
        low=np.log(min_period), high=np.log(max_period), size=num_tasks))
    if rounded:  # round to nearest integer.
        periods = np.rint(periods).tolist()
    # Generate priorities.
    prio = list(range(num_tasks))
    random.shuffle(prio)
    # Create tasks.
    for i in range(num_tasks):
        taskset.append(task.Task(i, 0, wcet, wcet, periods[i], periods[i],
                                 prio[i], True))
    return taskset


def non_preemptive_response_time(taskset):
    """Compute the worst-case response time of the communication tasks."""
    # Help function.
    def tda_np(pivot, lower_prio_tasks=[], higher_prio_tasks=[]):
        """Non-preemptive time demand analysis."""
        # Blocking by lower priority tasks due to non-preemptiveness.
        if lower_prio_tasks:
            blocked = max(task.wcet for task in lower_prio_tasks)
        else:
            blocked = 0

        # Increase time according to TDA:
        time = (blocked + pivot.wcet
                + sum(task.wcet for task in higher_prio_tasks))
        workload = 0
        while time <= pivot.deadline:
            workload = (blocked + task.wcet + sum(
                math.ceil(float(time)/task.period) * task.wcet
                for task in higher_prio_tasks))
            if (workload <= time):  # stop property
                return workload
            else:
                time = workload
        return False  # TDA failed

    # Compute WCRT of each task.
    for i, task in enumerate(taskset):
        rt = tda_np(task, taskset[i+1:], taskset[:i])
        if rt is False:  # tda_np failed
            return False
        elif rt > task.deadline:  # WCRT > deadline is not allowed
            return False
        else:
            # Set task WCRT
            task.rt = rt
    return True
