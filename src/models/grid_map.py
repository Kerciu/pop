from dataclasses import dataclass
from src.models.coordinate import Coordinate


@dataclass
class GridMap:
    rows: int
    cols: int
    base: 'Coordinate'
