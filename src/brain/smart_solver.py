from enum import Enum, auto

from brain.motion_control import get_move_action
from brain.route_planner import plan_delivery_batch, sort_route_tsp
from core.actions import Action
from core.distance_utils import distance
from models.coordinate import Coordinate


class MissionState(Enum):
    AT_BASE = auto()
    DELIVERING = auto()
    RETURNING = auto()


class SmartSolver:
    def __init__(self):
        self.mission_state = MissionState.AT_BASE
        self.delivery_queue = []

    def resolve(self, state, problem, accel_table, all_gifts_map):
        if state.last_action_was_acceleration:
            return Action.Floating, 1

        lapland = Coordinate(0, 0)

        if self.mission_state == MissionState.AT_BASE:
            if state.carrot_count < 10:
                return Action.LoadCarrots, 20

            if not state.loaded_gifts:
                if not state.available_gifts:
                    return Action.Floating, 1

                to_load = plan_delivery_batch(
                    state.available_gifts,
                    all_gifts_map,
                    state.sleigh_weight,
                    accel_table,
                )

                if not to_load:
                    return Action.Floating, 1

                try:
                    target_gift_name = to_load[0]
                    target_idx = state.available_gifts.index(target_gift_name)
                    return Action.LoadGifts, target_idx
                except ValueError:
                    return Action.Floating, 1

            if not self.delivery_queue:
                self.delivery_queue = sort_route_tsp(
                    state.loaded_gifts, all_gifts_map, lapland
                )
                self.mission_state = MissionState.DELIVERING

        if self.mission_state == MissionState.DELIVERING:
            if not self.delivery_queue:
                self.mission_state = MissionState.RETURNING
                return Action.Floating, 1

            target_gift_name = self.delivery_queue[0]
            target_gift = all_gifts_map[target_gift_name]

            curr_dist = distance(state.position, target_gift.destination)

            if curr_dist <= problem.D:
                try:
                    idx = state.loaded_gifts.index(target_gift_name)
                    self.delivery_queue.pop(0)
                    return Action.DeliverGift, idx
                except ValueError:
                    self.mission_state = MissionState.RETURNING

            return get_move_action(state, target_gift.destination)

        if self.mission_state == MissionState.RETURNING:
            if distance(state.position, lapland) <= problem.D:
                self.mission_state = MissionState.AT_BASE
                self.delivery_queue = []
                return Action.Floating, 1

            return get_move_action(state, lapland)

        return Action.Floating, 1
