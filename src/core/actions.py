from enum import Enum, auto
from collections import Map
from src.models.gift import Gift
from src.models.coordinate import Coordinate
from src.models.sleigh_state import SleighState
from src.core.distance_utils import distance
from src.core.acceleration_table import AccelerationTable


class Action(Enum):
    AccUp: auto
    AccDown: auto
    AccLeft: auto
    AccRight:auto
    Floating: auto
    LoadGifts: auto
    DeliverGift: auto
    LoadCarrots: auto


class Direction(Enum):
    UP = auto
    DOWN = auto
    LEFT = auto
    RIGHT = auto


def accelerate(new_state: 'SleighState', accel_table: 'AccelerationTable', acceleration: int, direction: 'Direction') -> 'SleighState':
    assert not new_state.last_action_was_acceleration
    assert new_state.carrots_count >= 1
    max_a = accel_table.get_max_accel(new_state.sleigh_weight)
    assert acceleration >= 0 and acceleration <= max_a

    if (direction == direction.UP):
        new_state.velocity.vr += acceleration
    elif (direction == direction.DOWN):
        new_state.velocity.vr -= acceleration
    elif (direction == direction.LEFT):
        new_state.velocity.vc -= acceleration
    elif (direction == direction.RIGHT):
        new_state.velocity.vc += acceleration
    new_state.carrot_count -= 1
    new_state.sleigh_weight -= 1
    new_state.last_action_was_acceleration = True

def floating(new_state: 'SleighState') -> 'SleighState':
    new_state.position.c += new_state.velocity.vc
    new_state.position.r += new_state.velocity.vr
    new_state.current_time += 1
    new_state.last_action_was_acceleration = False

def load_gifts(new_state: 'SleighState', gift_index: int, all_gifts_map: 'Map[str, Gift]', lapland_pos: 'Coordinate', range_d: int) -> 'SleighState':
    assert distance(new_state.position, lapland_pos) <= range_d
    assert gift_index < new_state.available_gifts.len() and gift_index >= 0

    gift = all_gifts_map[gift_name := new_state.available_gifts[gift_index]]
    new_state.available_gifts.remove(gift_name)
    new_state.loaded_gifts.append(gift_name)
    new_state.sleigh_weight += gift.weight

def deliver_gift(new_state: 'SleighState', gift_index: int, all_gifts_map: 'Map[str, Gift]', range_d: int) -> 'SleighState':
    assert gift_index < new_state.loaded_gifts.len() and gift_index >= 0
    gift = all_gifts_map[gift_name := new_state.loaded_gifts[gift_index]]
    assert distance(new_state.position, gift.destination) <= range_d

    new_state.loaded_gifts.remove(gift_name)
    new_state.delivered_gifts.append(gift_name)
    new_state.sleigh_weight -= gift.weight

def load_carrots(new_state: 'SleighState', n: int, lapland_pos: 'Coordinate', range_d: int) -> 'SleighState':
    assert distance(new_state.position, lapland_pos) <= range_d
    assert n >= 1

    new_state.carrot_count += n
    new_state.sleigh_weight += n