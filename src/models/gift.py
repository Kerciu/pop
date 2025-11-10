from dataclasses import dataclass
from models.coordinate import Coordinate


@dataclass
class Gift:
    name:   str
    score:  int
    weight: int
    destination: Coordinate
