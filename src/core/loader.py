from core.acceleration_table import AccelerationTable
from core.simulator import Simulator
from models.problem import Problem


def load_problem(path: str):
    problem = Problem(path)

    accel_table = AccelerationTable(problem.acceleration_ranges)

    all_gifts_map = {gift.name: gift for gift in problem.gifts}

    simulator = Simulator(
        t_limit=problem.T,
        range_d=problem.D,
        accel_table=accel_table,
        all_gifts_map=all_gifts_map,
    )

    return problem, simulator
