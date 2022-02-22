"""Simulator to create the schedule.

Source: https://github.com/tu-dortmund-ls12-rt/MissRateSimulator/blob/master/
simulator.py

Copied from:
https://github.com/tu-dortmund-ls12-rt/end-to-end"""
from __future__ import division
import operator


class eventSimulator:
    """The event simulator with periodic job behavior, fixed execution time>0,
    constrained deadline and synchronous releases for the single ECU case.
    """

    def __init__(self, tasks):
        """Initialize the event simulator.

        We assume that the tasks are sorted by their priority (highest priority
        first).
        """
        self.tasks = tasks  # list of tasks
        self.h = -1  # index of the active task with the highest workload
        self.n = len(tasks)  # number of tasks
        self.systemTick = float(0)  # current time

        self.statusTable = [[float(0.0) for x in range(5)]
                            for y in range(self.n)]
        # The status table for the simulator has 5 columns per row:
        # 0. remaining workload of task
        # 1. number of release
        # 2. number of deadlines misses
        # 3. number of deadlines = this should be less than release
        # 4. flag to determine starting time of a job

        self.eventList = []  # List of events the simulator has to process

        # # Sorting:
        # tasks = sorted(tasks, key=operator.attrgetter('priority'))

        # Analysis result.
        self.raw_result = dict()

        # Fill statusTable, raw_result and eventList the first time.
        self.initState()

    class eventClass(object):
        """One Event."""

        def __init__(self, case, delta, idx):
            """Initialize the event.

            case = 0 is a release and case = 1 is a deadline.
            delta is the remaining time until the event.
            idx is the corresponding task index for that event.
            """
            self.eventType = case
            self.delta = delta
            self.idx = idx

        def case(self):
            """Return the case of that event."""
            if self.eventType == 0:
                return "release"
            elif self.eventType == 1:
                return "deadline"

        def updateDelta(self, elapsedTime):
            """Update remaining time until the event."""
            self.delta = self.delta - elapsedTime

    def tableReport(self):
        """Print eventList and statusTable."""
        # Print eventList.
        for i, e in enumerate(self.eventList):
            print("Event " + str(i) + " from task " + str(e.idx))
            print(e.case())
            print(e.delta)

        # Print statusTable.
        for x in range(self.n):
            print("task" + str(x) + ": ")
            for y in range(5):
                print(self.statusTable[x][y])

    def findTheHighestWithWorkload(self):
        """Find active task with highest priority.

        Returns index of the task. If there is no active task, it returns -1.
        """
        hidx = -1
        # Find first row with non-zero entry at 0 = remaining workload
        for i in range(self.n):
            if self.statusTable[i][0] != 0:
                hidx = i
                break
        return hidx

    def release(self, idx):
        """Behavior at job release of task with index idx."""
        # Set deadline event.
        self.eventList.append(self.eventClass(
            1, self.tasks[idx].deadline, idx))

        # Set next release event.
        self.eventList.append(self.eventClass(
            0, self.tasks[idx].period, idx))

        # Sort the eventList.
        self.eventList = sorted(self.eventList,
                                key=operator.attrgetter('delta'))

        # Add the workload to corresponding entry in statusTable.
        self.statusTable[idx][0] += float(self.tasks[idx].wcet)

        # Initialiue the flag to indicate the first execution.
        self.statusTable[idx][4] = 1

        # Decide the highest priority task in the system.
        self.h = self.findTheHighestWithWorkload()
        if self.h == -1:
            print("BUG: after release, there must be at least one task with"
                  " workload.")

        # Record the job release in the statusTable.
        self.statusTable[idx][1] += 1

    def deadline(self, idx):
        """Behavior at job deadline of task with index idx."""
        # Check for deadline misses.
        if self.workload(idx) != 0:
            print("task" + str(idx) + " misses deadline")
            self.statusTable[idx][2] += 1
        self.statusTable[idx][3] += 1

    def dispatcher(self, targetedNumber):
        """Main function of the scheduler.

        Stops when the number of released jobs of the lowest priority task is
        equal to targetedNumber.
        """
        while (targetedNumber != self.numDeadlines(self.n - 1)):
            if len(self.eventList) == 0:
                print("BUG: there is no event in the dispatcher")
                break
            else:
                # Get next event from the eventList.
                e = self.getNextEvent()
                # Process the event.
                self.event_to_dispatch(e)

    def event_to_dispatch(self, event):
        """Process the given event."""
        # Process the elapsed time until the event.
        self.elapsedTime(event)

        # Find the corresponding function for the event.
        switcher = {
            0: self.release,
            1: self.deadline,
        }
        func = switcher.get(event.eventType, lambda: "ERROR")

        # Execute the event.
        func(event.idx)

    def elapsedTime(self, event):
        """Process the elapsed time until the event."""
        # Determine the elapsed time until the event.
        delta = event.delta

        # Update deltas of remaining events in eventList.
        for e in self.eventList:
            e.updateDelta(delta)

        # Update the workloads in statusTable.
        while (delta):
            self.h = self.findTheHighestWithWorkload()

            if self.h == -1:
                # Case: Processor idles for the remaining time.
                self.systemTick += delta
                delta = 0

            elif delta >= self.statusTable[self.h][0]:
                # Case: Task with index h finishes during remaining time.

                if self.statusTable[self.h][4] == 1:
                    # Case: First time execution of task hidx
                    # Put start of the job to raw_result.
                    self.raw_result[self.tasks[self.h]].append(self.systemTick)
                    # Set flag to 0.
                    self.statusTable[self.h][4] = 0

                # Edit delta and systemTick.
                delta -= self.statusTable[self.h][0]
                self.systemTick += self.statusTable[self.h][0]
                # Set remaining workload to 0.
                self.statusTable[self.h][0] = 0

                # Put finish of the job to raw_result.
                self.raw_result[self.tasks[self.h]].append(self.systemTick)

            elif delta < self.statusTable[self.h][0]:
                # Case: Task with index h finishes not during remaining time.

                if self.statusTable[self.h][4] == 1:
                    # Case: First time execution of task hidx
                    # Put start of the job to raw_result.
                    self.raw_result[self.tasks[self.h]].append(self.systemTick)
                    # Set flag to 0.
                    self.statusTable[self.h][4] = 0

                # Edit remaining workload.
                self.statusTable[self.h][0] -= delta
                # Edit delta and systemTick.
                self.systemTick += delta
                delta = 0

    def getNextEvent(self):
        """Get the next event from eventList."""
        event = self.eventList.pop(0)
        return event

    def e2e_result(self):
        """Provide necessary information for the end to end analysis.

        The result of the scheduler is pre-handled to represent in [start, end]
        format (result[task] is a list of tuples describing the start and end
        of each job).
        Note: The scheduler returns an empty list for a task if it has
        execution time = 0.
        """
        # Initialize result dictionary.
        result = dict()
        for task in self.tasks:
            result[task] = []

        # Fill entries of the dictionary.
        for task in self.tasks:
            # Traverse the raw_result.
            job_start = -1
            job_end = -1
            for x in self.raw_result[task]:
                if (job_start < 0):  # fill start
                    job_start = x
                else:  # fill finish
                    job_end = x
                if job_start > -1 and job_end > -1:  # put job to result
                    result[task].append((job_start, job_end))
                    job_start = -1
                    job_end = -1

        return result

    def missRate(self, idx):
        """Return the miss rate of task idx."""
        return self.statusTable[idx][2] / self.statusTable[idx][1]

    def totalMissRate(self):
        """Return the total miss rate of the system."""
        sumRelease = 0
        sumMisses = 0
        for idx in range(self.n):
            sumRelease += self.statusTable[idx][1]
            sumMisses += self.statusTable[idx][2]
        return sumMisses / sumRelease

    def releasedJobs(self, idx):
        """Return the number of released jobs of idx task in the table."""
        return self.statusTable[idx][1]

    def numDeadlines(self, idx):
        """Return the number of past deadlines of idx task in the table."""
        return self.statusTable[idx][3]

    def releasedMisses(self, idx):
        """Return the number of misses of idx task in the table."""
        return self.statusTable[idx][2]

    def workload(self, idx):
        """Return the remaining workload of idx task in the table."""
        return self.statusTable[idx][0]

    def initState(self):
        """Specify the initial state of the simulator."""
        # Make one entry for each task in the result dictionary.
        for task in self.tasks:
            self.raw_result[task] = []

        for idx in range(len(self.tasks)):
            # Fill the status Table.
            self.statusTable[idx][0] = 0
            self.statusTable[idx][3] = self.statusTable[idx][1]
            # Put release events to the eventList.
            self.eventList.append(self.eventClass(
                0, self.tasks[idx].phase, idx))

        # Sort eventList by remaining time.
        # In case phase not 0 anymore, we need this one.
        self.eventList = sorted(
            self.eventList, key=operator.attrgetter('delta'))
