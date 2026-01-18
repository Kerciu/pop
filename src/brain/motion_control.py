from core.actions import Action, Direction
from models.coordinate import Coordinate
from models.sleigh_state import SleighState


def get_stopping_distance(velocity: int) -> int:
    v = abs(velocity)
    return (v * (v + 1)) // 2


def get_move_action(state: SleighState, target: Coordinate, accel_table) -> tuple[Action, int]:
    dc = target.c - state.position.c
    dr = target.r - state.position.r

    vc = state.velocity.vc
    vr = state.velocity.vr

    def solve_axis(dist, vel):
        if abs(dist) <= 0.5:
            dist = 0

        if dist == 0:
            if vel == 0:
                return 0
            return -1 if vel > 0 else 1

        if dist * vel > 0:
            stop_dist = get_stopping_distance(vel)
            if abs(dist) <= stop_dist:
                return -1 if vel > 0 else 1
            return 0

        return 1 if dist > 0 else -1

    action_c = solve_axis(dc, vc)
    action_r = solve_axis(dr, vr)

    braking_c = action_c * vc < 0
    braking_r = action_r * vr < 0

    final_dir = None

    if braking_c:
        final_dir = Direction.LEFT if action_c == -1 else Direction.RIGHT
    elif braking_r:
        final_dir = Direction.UP if action_r == 1 else Direction.DOWN
    else:
        if abs(dc) >= abs(dr) and action_c != 0:
            final_dir = Direction.RIGHT if action_c == 1 else Direction.LEFT
        elif action_r != 0:
            final_dir = Direction.UP if action_r == 1 else Direction.DOWN
        else:
            return Action.Floating, 1

    max_power = accel_table.get_max_acceleration_for_weight(state.sleigh_weight)
    power = max(1, max_power)

    if final_dir is None:
        return Action.Floating, power

    if final_dir == Direction.UP:
        return Action.AccUp, power
    if final_dir == Direction.DOWN:
        return Action.AccDown, power
    if final_dir == Direction.LEFT:
        return Action.AccLeft, power
    if final_dir == Direction.RIGHT:
        return Action.AccRight, power

    return Action.Floating, power
