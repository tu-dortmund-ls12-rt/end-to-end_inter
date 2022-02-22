"""Transform task from dictionaries to task objects for the event simulator.

Source:
https://github.com/tu-dortmund-ls12-rt/end-to-end"""
import utilities.task as t
from scipy import stats


class Transformer:
    """Transformer class."""

    def __init__(self, t_id, t_task_sets, time_scale=10000000):
        """Creates a transformer object."""
        self.id = str(t_id)  # unique identifier
        self.task_sets = t_task_sets  # task set as dictionary
        self.time_scale = time_scale  # scaling factor for period, WCET, etc.

    def transform_tasks(self, phase):
        """Transform the given tasks.

        The flag phase specifies if phases should be introduced to the task
        set.
        - bring task so object definition
        - sort tasks for RM and set priorities
        - time_scale is the exactness of the values (task values are integers
                only afterwards.)
        - set phase
        """
        # Distribution of task phases
        distribution_phase = stats.uniform()

        # Initialization of the transformed task sets
        transformed_task_sets = []

        for task_set in self.task_sets:
            # Sort tasks set by periods.
            sorted_task_set = sorted(task_set, key=lambda task: task['period'])
            transformed_task_set = []
            i = 0
            # Transform each task individually.
            for task in sorted_task_set:
                # Set phase.
                if phase:
                    phase = int(float(format(distribution_phase.rvs() * 1000,
                                             ".7f")) * self.time_scale)
                else:
                    phase = 0
                # Scale values and make a task object.
                transformed_task_set.append(
                    t.Task(i, phase, 0,
                           int(float(format(task['execution'], ".7f"))
                               * self.time_scale),
                           int(float(format(task['period'], ".7f"))
                               * self.time_scale),
                           int(float(format(task['period'], ".7f"))
                               * self.time_scale), i))
                i += 1
            transformed_task_sets.append(transformed_task_set)
        return transformed_task_sets
