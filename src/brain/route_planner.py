from core.distance_utils import distance
from models.coordinate import Coordinate
from models.gift import Gift


def plan_delivery_batch(
    available_gifts: list[str],
    all_gifts_map: dict[str, Gift],
    current_weight: int,
    accel_table,
) -> list[str]:
    selected_gifts = []
    MAX_SAFE_WEIGHT = 60

    simulated_weight = current_weight

    lapland = Coordinate(0, 0)
    sorted_gifts = sorted(
        available_gifts, key=lambda g: distance(lapland, all_gifts_map[g].destination)
    )

    for g_name in sorted_gifts:
        gift = all_gifts_map[g_name]
        if simulated_weight + gift.weight < MAX_SAFE_WEIGHT:
            selected_gifts.append(g_name)
            simulated_weight += gift.weight
        else:
            break

    return selected_gifts


def sort_route_tsp(
    loaded_gifts: list[str], all_gifts_map: dict[str, Gift], start_pos: Coordinate
) -> list[str]:
    route = []
    current_pos = start_pos
    remaining = loaded_gifts[:]

    while remaining:
        nearest = min(
            remaining, key=lambda g: distance(current_pos, all_gifts_map[g].destination)
        )
        route.append(nearest)
        remaining.remove(nearest)
        current_pos = all_gifts_map[nearest].destination

    return route


def plan_dynamic_mission(
    available_gifts: list[str],
    all_gifts_map: dict[str, Gift],
    current_weight: int,
    accel_table,
):
    MAX_TOTAL_WEIGHT = 450
    FUEL_SAFETY_MARGIN = 20
    FUEL_PER_UNIT_DIST = 1.5

    lapland = Coordinate(0, 0)

    def score_heuristic(g_name):
        g = all_gifts_map[g_name]
        dist = distance(lapland, g.destination)
        if dist == 0: dist = 1
        return (g.score) / (g.weight * dist)

    sorted_gifts = sorted(available_gifts, key=score_heuristic, reverse=True)

    selected_gifts = []
    simulated_weight = current_weight

    current_pos = lapland
    total_dist = 0

    for g_name in sorted_gifts:
        gift = all_gifts_map[g_name]

        dist_leg = distance(current_pos, gift.destination)
        dist_return = distance(gift.destination, lapland)

        new_total_dist = total_dist + dist_leg + dist_return
        needed_fuel = int(new_total_dist * FUEL_PER_UNIT_DIST) + FUEL_SAFETY_MARGIN

        total_mass_prediction = simulated_weight + gift.weight + needed_fuel

        if total_mass_prediction < MAX_TOTAL_WEIGHT:
            selected_gifts.append(g_name)
            simulated_weight += gift.weight
            current_pos = gift.destination
            total_dist += dist_leg
        else:
            continue

    final_return_dist = distance(current_pos, lapland)
    total_mission_dist = total_dist + final_return_dist
    final_fuel_needed = int(total_mission_dist * FUEL_PER_UNIT_DIST) + FUEL_SAFETY_MARGIN

    return selected_gifts, final_fuel_needed
