from core.actions import Action
from core.distance_utils import distance
from models.coordinate import Coordinate


class GreedySolver:
    def __init__(self):
        pass

    def resolve(self, current_state, problem, accel_table, all_gift_map):
        lapland_pos = Coordinate(0, 0)

        if current_state.last_action_was_acceleration:
            return (Action.Floating, 1)

        for i, gift_name in enumerate(current_state.loaded_gifts):
            gift = all_gift_map[gift_name]
            if distance(current_state.position, gift.destination) <= problem.D:
                return (Action.DeliverGift, i)

        dist_to_lapland = distance(current_state.position, lapland_pos)
        if dist_to_lapland <= 1.0:
            if current_state.carrot_count < 10:
                return (Action.LoadCarrots, 50)

            if len(current_state.available_gifts) > 0:
                next_gift = all_gift_map[current_state.available_gifts[0]]
                new_weight = current_state.sleigh_weight + next_gift.weight
                if accel_table.get_max_acceleration_for_weight(new_weight) > 0:
                    return (Action.LoadGifts, 0)

        target_pos = lapland_pos

        if len(current_state.loaded_gifts) > 0:
            first_gift_name = current_state.loaded_gifts[0]
            target_pos = all_gift_map[first_gift_name].destination

        max_accel = accel_table.get_max_acceleration_for_weight(
            current_state.sleigh_weight
        )

        if max_accel == 0 or current_state.carrot_count == 0:
            return (Action.Floating, 1)

        accel_val = min(1, max_accel)

        if target_pos.c > current_state.position.c:
            return (Action.AccRight, accel_val)
        if target_pos.c < current_state.position.c:
            return (Action.AccLeft, accel_val)
        if target_pos.r > current_state.position.r:
            return (Action.AccUp, accel_val)
        if target_pos.r < current_state.position.r:
            return (Action.AccDown, accel_val)

        return (Action.Floating, 1)
