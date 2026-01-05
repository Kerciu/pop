from enum import Enum, auto
from models.gift import Gift
from models.coordinate import Coordinate
from models.sleigh_state import SleighState
from core.distance_utils import distance
from core.acceleration_table import AccelerationTable
from typing import Mapping


class Action(Enum):
    AccUp = auto()
    AccDown = auto()
    AccLeft = auto()
    AccRight = auto()
    Floating = auto()
    LoadGifts = auto()
    DeliverGift = auto()
    LoadCarrots = auto()


class Direction(Enum):
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()


def accelerate(new_state: SleighState, accel_table: AccelerationTable, acceleration: int, direction: Direction) -> SleighState:
    assert not new_state.last_action_was_acceleration
    assert new_state.carrot_count >= 1
    max_a = accel_table.get_max_acceleration_for_weight(new_state.sleigh_weight)
    assert acceleration >= 0 and acceleration <= max_a

    if (direction == Direction.UP):
        new_state.velocity.vr += acceleration
    elif (direction == Direction.DOWN):
        new_state.velocity.vr -= acceleration
    elif (direction == Direction.LEFT):
        new_state.velocity.vc -= acceleration
    elif (direction == Direction.RIGHT):
        new_state.velocity.vc += acceleration
    new_state.carrot_count -= 1
    new_state.sleigh_weight -= 1
    new_state.last_action_was_acceleration = True
    return new_state


def floating(new_state: SleighState) -> SleighState:
    new_state.position.c += new_state.velocity.vc
    new_state.position.r += new_state.velocity.vr
    new_state.current_time += 1
    new_state.last_action_was_acceleration = False
    return new_state


def load_gifts(
        new_state: SleighState, gift_index: int,
        all_gifts_map: Mapping[str, Gift],
        lapland_pos: Coordinate, range_d: int) -> SleighState:
    assert distance(new_state.position, lapland_pos) <= range_d
    assert gift_index < len(new_state.available_gifts) and gift_index >= 0

    gift = all_gifts_map[gift_name := new_state.available_gifts[gift_index]]
    new_state.available_gifts.remove(gift_name)
    new_state.loaded_gifts.append(gift_name)
    new_state.sleigh_weight += gift.weight
    return new_state


def deliver_gift(
        new_state: SleighState, gift_index: int,
        all_gifts_map: Mapping[str, Gift], range_d: int
        ) -> SleighState:
    assert gift_index < len(new_state.loaded_gifts) and gift_index >= 0
    gift = all_gifts_map[gift_name := new_state.loaded_gifts[gift_index]]
    assert distance(new_state.position, gift.destination) <= range_d

    new_state.loaded_gifts.remove(gift_name)
    new_state.delivered_gifts.append(gift_name)
    new_state.sleigh_weight -= gift.weight
    return new_state


def load_carrots(
        new_state: SleighState, n: int, lapland_pos: Coordinate, range_d: int
        ) -> SleighState:
    assert distance(new_state.position, lapland_pos) <= range_d
    assert n >= 1

    new_state.carrot_count += n
    new_state.sleigh_weight += n
    return new_state
