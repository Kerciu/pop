from models.gift import Gift
from models.coordinate import Coordinate
from models.sleigh_state import SleighState
from core.acceleration_table import AccelerationTable
from core.actions import Action, accelerate, floating, load_gifts, deliver_gift, load_carrots, Direction
from typing import Mapping


class Simulator:
    def __init__(self, t_limit: int, range_d: int,
                 accel_table: AccelerationTable,
                 all_gifts_map: Mapping[str, Gift],
                 lapland_pos: Coordinate = Coordinate(0, 0)):
        self.t_limit = t_limit
        self.range_d = range_d
        self.accel_table = accel_table
        self.all_gifts_map = all_gifts_map
        self.lapland_pos = lapland_pos

    def apply_action(self, state: SleighState, action: Action, parameter: int) -> SleighState:
        new_state = state.clone()
        match action:
            case action.AccUp:
                accelerate(new_state, self.accel_table, parameter, Direction.UP)
            case action.AccDown:
                accelerate(new_state, self.accel_table, parameter, Direction.DOWN)
            case action.AccLeft:
                accelerate(new_state, self.accel_table, parameter, Direction.LEFT)
            case action.AccRight:
                accelerate(new_state, self.accel_table, parameter, Direction.RIGHT)
            case action.Floating:
                for _ in range(parameter):
                    floating(new_state)
            case action.LoadGifts:
                load_gifts(new_state, parameter, self.all_gifts_map, self.lapland_pos, self.range_d)
            case action.DeliverGift:
                deliver_gift(new_state, parameter, self.all_gifts_map, self.lapland_pos, self.range_d)
            case action.LoadCarrots:
                load_carrots(new_state, parameter, self.lapland_pos, self.range_d)
            case _:
                raise InvalidActionError
        return new_state


class InvalidActionError(Exception):
    def __init__(self, action):
        self.action = action
        super.__init__(f"Action {action.name} is invalid")