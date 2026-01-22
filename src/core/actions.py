from enum import Enum, IntEnum, auto
from typing import Mapping

from core.acceleration_table import AccelerationTable
from core.distance_utils import distance
from models.coordinate import Coordinate
from models.gift import Gift
from models.sleigh_state import SleighState


class Action(IntEnum):
    AccUp = 0
    AccDown = 1
    AccLeft = 2
    AccRight = 3
    Floating = 4
    LoadCarrots = 5
    LoadGifts = 6
    DeliverGift = 7


class Direction(Enum):
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()


def accelerate(
    new_state: SleighState,
    accel_table: AccelerationTable,
    acceleration: int,
    direction: Direction,
) -> SleighState:
    assert not new_state.last_action_was_acceleration
    assert new_state.carrot_count >= 1

    max_a = accel_table.get_max_acceleration_for_weight(new_state.sleigh_weight)
    actual_acceleration = min(acceleration, max_a)

    assert actual_acceleration >= 0

    if direction == Direction.UP:
        new_state.velocity.vr += actual_acceleration
    elif direction == Direction.DOWN:
        new_state.velocity.vr -= actual_acceleration
    elif direction == Direction.LEFT:
        new_state.velocity.vc -= actual_acceleration
    elif direction == Direction.RIGHT:
        new_state.velocity.vc += actual_acceleration

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
    new_state: SleighState,
    gift_index: int,
    all_gifts_map: Mapping[str, Gift],
    lapland_pos: Coordinate,
    range_d: int,
) -> SleighState:
    assert distance(new_state.position, lapland_pos) <= range_d
    assert gift_index < len(new_state.available_gifts) and gift_index >= 0

    gift_name = new_state.available_gifts[gift_index]
    gift = all_gifts_map[gift_name]

    new_state.available_gifts.remove(gift_name)
    new_state.loaded_gifts.append(gift_name)
    new_state.sleigh_weight += gift.weight

    return new_state


def deliver_gift(
    new_state: SleighState,
    gift_index: int,
    all_gifts_map: Mapping[str, Gift],
    range_d: int,
) -> SleighState:
    assert gift_index < len(new_state.loaded_gifts) and gift_index >= 0

    gift_name = new_state.loaded_gifts[gift_index]
    gift = all_gifts_map[gift_name]

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


def solve_knapsack_greedy(available_gifts_ids, gifts_map, max_weight, current_weight):
    remaining_capacity = max_weight - current_weight
    if remaining_capacity <= 0:
        return []

    candidates = []
    for g_id in available_gifts_ids:
        gift = gifts_map[g_id]
        if gift.weight <= remaining_capacity:
            weight = max(0.1, gift.weight)
            ratio = gift.score / weight
            candidates.append((g_id, gift, ratio))

    candidates.sort(key=lambda x: (x[2], -x[1].weight), reverse=True)

    to_load = []
    for g_id, gift, ratio in candidates:
        if gift.weight <= remaining_capacity:
            to_load.append(g_id)
            remaining_capacity -= gift.weight

    return to_load
