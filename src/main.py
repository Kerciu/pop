from models.problem import Problem
from models.sleigh_state import SleighState
from models.coordinate import Coordinate
from models.velocity import Velocity
from models.gift import Gift
from core.actions import Action
from core.acceleration_table import AccelerationTable
from core.simulator import Simulator
from core.distance_utils import distance
from typing import Tuple, Any, Mapping
import os

INPUT_DATA_PATH = os.path.join("data", "a_an_example.in.txt")
print("Path: ", INPUT_DATA_PATH)
OUTPUT_DATA_PATH = os.path.join("data", "output", "output_data.txt")


def solver_function(
        current_state: SleighState,
        problem: Problem,
        accel_table: AccelerationTable,
        all_gift_map: Mapping[str, Gift]) -> Tuple[Action, Any]:

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


def write_output_file(actions_list: list[str], output_path: str):
    with open(output_path, 'w') as f:
        f.write(f"{len(actions_list)}\n")
        for action_str in actions_list:
            f.write(f"{action_str}\n")
    print(f"Saved solution in {output_path}")


def main() -> None:
    problem: Problem = Problem(INPUT_DATA_PATH)
    accel_table: AccelerationTable = AccelerationTable(
        problem.acceleration_ranges
    )

    all_gift_map = {gift.name: gift for gift in problem.gifts}

    simulator: Simulator = Simulator(
        t_limit=problem.T,
        range_d=problem.D,
        accel_table=accel_table,
        all_gifts_map=all_gift_map,
    )

    solver = solver_function

    initial_state: SleighState = SleighState(
        current_time=0,
        position=Coordinate(0, 0),
        velocity=Velocity(0, 0),
        sleigh_weight=0,
        carrot_count=0,
        loaded_gifts=[],
        available_gifts=[gift.name for gift in all_gift_map.values()],
        delivered_gifts=[],
        last_action_was_acceleration=False,
    )

    current_state = initial_state
    action_list = []

    while current_state.current_time < problem.T:

        action, parameter = solver(
            current_state, problem, accel_table, all_gift_map
        )

        if action == Action.Floating:
            if current_state.current_time + parameter > problem.T:
                parameter = problem.T - current_state.current_time

        if parameter <= 0 and action == Action.Floating:
            break

        try:
            current_state = simulator.apply_action(
                current_state, action, parameter
            )

            action_list.append(f"{action.name} {parameter}")

        except Exception as e:
            print(f"Error while action {action.name} {parameter}: {e}")
            if not current_state.last_action_was_acceleration:
                try:
                    current_state = simulator.apply_action(
                        current_state,
                        Action.Floating,
                        1
                    )
                    action_list.append("Floating 1")
                except Exception as e2:
                    print(f"Critical error, cannot even float: {e2}")
                    break
            else:
                action_list.append("Floating 1")

    output_dir = os.path.dirname(OUTPUT_DATA_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    write_output_file(action_list, OUTPUT_DATA_PATH)
    print("Simulation ended.")


if __name__ == "__main__":
    main()
