from collections import Map
from src.models.gift import Gift
from src.models.action import Action
from src.models.coordinate import Coordinate
from src.models.sleigh_state import SleighState
from src.core.acceleration_table import AccelerationTable

from src.core.distance_utils import distance


class Simulator:
    def __init__(self, t_limit: int, range_d: int,
                 accel_table: 'AccelerationTable', all_gifts_map: Map[str, 'Gift'],
                 lapland_pos: 'Coordinate' = Coordinate(0,0)):
        self.t_limit = t_limit
        self.range_d = range_d
        self.accel_table = accel_table
        self.all_gifts_map = Map[str, 'Gift']
        self.lapland_pos = lapland_pos

    def apply_action(self, state: 'SleighState', action: 'Action') -> 'SleighState':
        new_state = state.clone()
        match action:
            case action.AccUp:
                # need to inject 'a' as a parameter
                a = 0 # just for now
                assert not state.last_action_was_acceleration
                assert state.carrots_count >= 1
                max_a = self.accel_table.get_max_accel(state.sleigh_weight)
                assert a >= 0 and a <= max_a

                new_state.velocity.vr += a
                new_state.carrot_count -= 1
                new_state.sleigh_weight -= 1
                new_state.last_action_was_acceleration = True
            case action.AccDown:
                # need to inject 'a' as a parameter
                a = 0 # just for now
                assert not state.last_action_was_acceleration
                assert state.carrots_count >= 1
                max_a = self.accel_table.get_max_accel(state.sleigh_weight)
                assert a >= 0 and a <= max_a

                new_state.velocity.vr -= a
                new_state.carrot_count -= 1
                new_state.sleigh_weight -= 1
                new_state.last_action_was_acceleration = True
            case action.AccLeft:
                # need to inject 'a' as a parameter
                a = 0 # just for now
                assert not state.last_action_was_acceleration
                assert state.carrots_count >= 1
                max_a = self.accel_table.get_max_accel(state.sleigh_weight)
                assert a >= 0 and a <= max_a

                new_state.velocity.vc -= a
                new_state.carrot_count -= 1
                new_state.sleigh_weight -= 1
                new_state.last_action_was_acceleration = True
            case action.AccRight:
                # need to inject 'a' as a parameter
                a = 2 # just for now
                assert not state.last_action_was_acceleration
                assert state.carrots_count >= 1
                max_a = self.accel_table.get_max_accel(state.sleigh_weight)
                assert a >= 0 and a <= max_a

                new_state.velocity.vc += a
                new_state.carrot_count -= 1
                new_state.sleigh_weight -= 1
                new_state.last_action_was_acceleration = True
            case action.Float:
                assert state.current_time + 1 <= self.t_limit

                new_state.position.c += state.velocity.vc
                new_state.position.r += state.velocity.vr
                new_state.current_time += 1
                new_state.last_action_was_acceleration = False
            case action.LoadGifts:
                # need to inject 'name' as a parameter
                name = 'Toby' # just for now
                assert distance(state.position, self.lapland_pos) <= self.range_d
                assert name in state.available_gifts

                gift = self.all_gifts_map[name]
                state.available_gifts.remove(name)
                state.loaded_gifts.append(name)
                state.sleigh_weight += gift.weight
            case action.DeliverGift:
                # need to inject 'name' as a parameter
                name = 'Toby' # just for now
                assert name in state.loaded_gifts
                gift = self.all_gifts_map[name]
                assert distance(state.position, gift.destination) <= self.range_d

                state.loaded_gifts.remove(name)
                state.delivered_gifts.append(name)
                state.sleigh_weight -= gift.weight
            case action.LoadCarrots:
                # need to inject 'n' as a parameter
                n = 1 # just for now
                assert distance(state.position, self.lapland_pos) <= self.range_d
                assert n >= 1

                new_state.carrot_count += n
                new_state.sleigh_weight += n
            case _:
                raise InvalidActionError
        return new_state


class InvalidActionError(Exception):
    def __init__(self, action):
        self.action = action
        super.__init__(f"Action {action.name} is invalid")