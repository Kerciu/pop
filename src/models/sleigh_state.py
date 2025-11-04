from dataclasses import dataclass
from src.models.velocity import Velocity
from src.models.coordinate import Coordinate
from src.models.gift import Gift


@dataclass
class SleighState:
    current_time: int
    position:     'Coordinate'
    velocity:     'Velocity'
    sleigh_weight: int
    carrots_count: int
    loaded_gifts: list['Gift']
    available_gifts: list['Gift']
    delivered_gifts: list['Gift']
    last_action_was_acceleration: bool

    @classmethod
    def clone() -> 'SleighState':
        return SleighState(
            current_time=0,
            position=Coordinate(0, 0),
            velocity=Velocity(0, 0),
            sleigh_weight=0,
            carrots_count=0,
            loaded_gifts=[],
            available_gifts=[],
            delivered_gifts=[],
            last_action_was_acceleration=False
        )