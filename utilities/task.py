"""Representation of Tasks.

Source:
https://github.com/tu-dortmund-ls12-rt/end-to-end"""


class Task:
    """A task."""

    def __init__(self, task_id, task_phase, task_bcet, task_wcet, task_period,
                 task_deadline, priority=0, message=False):
        """Creates a task represented by ID, Phase, BCET, WCET, Period and
        Deadline.
        """
        self.id = str(task_id)
        self.phase = task_phase  # phase
        self.bcet = task_bcet  # best-case execution time
        self.wcet = task_wcet  # worst-case execution time
        self.period = task_period  # period
        self.deadline = task_deadline  # deadline
        self.priority = priority  # a lower value means higher priority
        self.message = message  # flag for communication tasks

        self.rt = 0  # Worst-case response time, specified during analysis

    def __str__(self):
        """Print a task."""
        return (" Type: {type:^}\n ID: {id:^}\n Priority: {priority:^}\n"
                + " Phase: {phase:^} \n BCET: {bcet:^} \n WCET: {wcet:^} \n"
                + " Period: {period:^} \n Deadline: {deadline:^} \n"
                + " Response: {response:^}").format(
            type=str('Message') if self.message else str('Task'),
            id=self.id, priority=self.priority, phase=self.phase,
            bcet=self.bcet, wcet=self.wcet, period=self.period,
            deadline=self.deadline, response=self.rt)

    def copy(self):
        tsk = Task(self.id, self.phase, self.bcet, self.wcet,
                   self.period, self.deadline, self.priority, self.message)
        tsk.rt = self.rt
        return tsk
