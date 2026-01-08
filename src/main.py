import os

from brain.smart_solver import SmartSolver
from core.acceleration_table import AccelerationTable
from core.actions import Action
from core.simulator import Simulator
from models.coordinate import Coordinate
from models.problem import Problem
from models.sleigh_state import SleighState
from models.velocity import Velocity

INPUT_DATA_PATH = os.path.join("data", "a_an_example.in.txt")
print("Path: ", INPUT_DATA_PATH)
OUTPUT_DATA_PATH = os.path.join("data", "output", "output_data.txt")


def write_output_file(actions_list: list[str], output_path: str):
    with open(output_path, "w") as f:
        f.write(f"{len(actions_list)}\n")
        for action_str in actions_list:
            f.write(f"{action_str}\n")
    print(f"Saved solution in {output_path}")


def main() -> None:
    problem: Problem = Problem(INPUT_DATA_PATH)
    accel_table: AccelerationTable = AccelerationTable(problem.acceleration_ranges)

    all_gift_map = {gift.name: gift for gift in problem.gifts}

    simulator: Simulator = Simulator(
        t_limit=problem.T,
        range_d=problem.D,
        accel_table=accel_table,
        all_gifts_map=all_gift_map,
    )

    solver = SmartSolver()
    # solver = GreedySolver()

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
        action, parameter = solver.resolve(
            current_state, problem, accel_table, all_gift_map
        )

        if action == Action.Floating:
            if current_state.current_time + parameter > problem.T:
                parameter = problem.T - current_state.current_time

        if parameter <= 0 and action == Action.Floating:
            break

        try:
            current_state = simulator.apply_action(current_state, action, parameter)

            action_list.append(f"{action.name} {parameter}")

        except Exception as e:
            print(f"Error while action {action.name} {parameter}: {e}")
            if not current_state.last_action_was_acceleration:
                try:
                    current_state = simulator.apply_action(
                        current_state, Action.Floating, 1
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
