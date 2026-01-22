import math

from models.coordinate import Coordinate


def distance(source: Coordinate, target: Coordinate) -> int:
    dx = source.c - target.c
    dy = source.r - target.r
    return math.sqrt(dx * dx + dy * dy)


def calculate_braking_signal(
    dist: float, velocity_towards: float, max_acc: float
) -> float:
    if velocity_towards <= 0 or max_acc == 0:
        return 0.0

    stopping_dist = (velocity_towards**2) / (2.0 * max_acc)

    if stopping_dist >= dist * 0.9:
        return 1.0
    return 0.0
