from typing import Mapping

from core.acceleration_table import AccelerationTable
from core.actions import (
    Action,
    Direction,
    accelerate,
    deliver_gift,
    floating,
    load_carrots,
    load_gifts,
)
from models.coordinate import Coordinate
from models.gift import Gift
from models.sleigh_state import SleighState


class Simulator:
    def __init__(
        self,
        t_limit: int,
        range_d: int,
        accel_table: AccelerationTable,
        all_gifts_map: Mapping[str, Gift],
        lapland_pos: Coordinate = Coordinate(0, 0),
    ):
        self.t_limit = t_limit
        self.range_d = range_d
        self.accel_table = accel_table
        self.all_gifts_map = all_gifts_map
        self.lapland_pos = lapland_pos

    def apply_action(
        self, state: SleighState, action: Action, parameter: int
    ) -> SleighState:
        new_state = state.clone()
        # Wywoływanie odpowiednich funkcji z actions.py
        match action:
            case Action.AccUp:
                accelerate(new_state, self.accel_table, parameter, Direction.UP)
            case Action.AccDown:
                accelerate(new_state, self.accel_table, parameter, Direction.DOWN)
            case Action.AccLeft:
                accelerate(new_state, self.accel_table, parameter, Direction.LEFT)
            case Action.AccRight:
                accelerate(new_state, self.accel_table, parameter, Direction.RIGHT)
            case Action.Floating:
                for _ in range(parameter):
                    floating(new_state)
            case Action.LoadGifts:
                # parameter to INDEX w liście dostępnych prezentów
                load_gifts(
                    new_state,
                    parameter,
                    self.all_gifts_map,
                    self.lapland_pos,
                    self.range_d,
                )
            case Action.DeliverGift:
                # parameter to INDEX w liście załadowanych prezentów
                deliver_gift(new_state, parameter, self.all_gifts_map, self.range_d)
            case Action.LoadCarrots:
                load_carrots(new_state, parameter, self.lapland_pos, self.range_d)
            case _:
                raise InvalidActionError(action)
        return new_state


class InvalidActionError(Exception):
    def __init__(self, action):
        self.action = action
        super().__init__(f"Action {action} is invalid")
