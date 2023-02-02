"""End-to-End (e2e) Analysis by Becker et al.
https://doi.org/10.1016/j.sysarc.2017.09.004
https://doi.org/10.1109/RTCSA.2016.41

(own implementation for maximum data age only, under the assumption that WCRT is known)
"""
from utilities.chain import CauseEffectChain
from utilities.task import Task
from math import ceil


# ===== Help functions =====
def Rmin(tsk: Task, idx: int):
    return tsk.phase + idx * tsk.period


def Rmax(tsk: Task, idx: int):
    return Rmin(tsk, idx + 1) - tsk.wcet


def Dmin(tsk: Task, idx: int):
    return Rmin(tsk, idx) + tsk.wcet


def Dmax(tsk: Task, idx: int):
    return Rmax(tsk, idx + 1) + tsk.wcet


def build_tree(chain: CauseEffectChain, current_position: int, current_job_idx: int):
    if current_position + 1 >= chain.length():
        return []

    current_task: Task = chain.chain[current_position]
    next_task: Task = chain.chain[current_position + 1]

    # With this index the below property is safely fulfilled:
    next_job_idx = (
        ceil(
            (Dmax(current_task, current_position) - next_task.phase) / next_task.period
        )
        - 1
    )

    assert Rmin(next_task, next_job_idx) < Dmax(current_task, current_job_idx)

    next_job_idx = max(next_job_idx, 0)

    while Rmin(next_task, next_job_idx + 1) < Dmax(current_task, current_job_idx):
        next_job_idx = next_job_idx + 1

    # Note that in https://doi.org/10.1016/j.sysarc.2017.09.004 it is not specified what happens if there is no reachable job due to large phase of next_task.
    # We assume that in that case the first job is still reachable.

    return [next_job_idx] + build_tree(chain, current_position + 1, next_job_idx)


# ===== Analysis =====
def mrda(chain: CauseEffectChain):
    return
