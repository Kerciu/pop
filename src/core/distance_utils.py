import math
from models.coordinate import Coordinate


def distance(source: Coordinate, target: Coordinate) -> int:
    dx = source.c - target.c
    dy = source.r - target.r
    return math.sqrt(dx * dx + dy * dy)
