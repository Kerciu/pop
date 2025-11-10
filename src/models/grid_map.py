from dataclasses import dataclass
from models.coordinate import Coordinate


@dataclass
class GridMap:
    rows: int
    cols: int
    base: 'Coordinate'
