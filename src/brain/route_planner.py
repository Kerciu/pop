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
