from dataclasses import dataclass
from src.models.velocity import Velocity
from src.models.coordinate import Coordinate


@dataclass
class SleighState:
    current_time: int
    position:     'Coordinate'
    velocity:     'Velocity'
    sleigh_weight: int
    carrot_count: int
    loaded_gifts: list[str]
    available_gifts: list[str]
    delivered_gifts: list[str]
    last_action_was_acceleration: bool

    def clone(self) -> 'SleighState':
        return SleighState(
            current_time=self.current_time,
            position=self.position,
            velocity=self.velocity,
            sleigh_weight=self.sleigh_weight,
            carrot_count=self.carrot_count,
            loaded_gifts=self.loaded_gifts,
            available_gifts=self.available_gifts,
            delivered_gifts=self.delivered_gifts,
            last_action_was_acceleration=self.last_action_was_acceleration
        )