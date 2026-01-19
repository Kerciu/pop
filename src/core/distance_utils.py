import math

from models.coordinate import Coordinate


def distance(source: Coordinate, target: Coordinate) -> int:
    dx = source.c - target.c
    dy = source.r - target.r
    return math.sqrt(dx * dx + dy * dy)


def calculate_braking_signal(
    dist: float, velocity_towards: float, max_acc: float
) -> float:
    """
    Zwraca 1.0 jeśli trzeba natychmiast hamować, żeby nie przelecieć celu.
    Fizyka: droga hamowania s = v^2 / (2*a)
    """
    if velocity_towards <= 0 or max_acc == 0:
        return 0.0

    # Droga potrzebna do zatrzymania się z obecnej prędkości
    stopping_dist = (velocity_towards**2) / (2.0 * max_acc)

    # Jeśli droga hamowania to 90% dystansu do celu (lub więcej) -> ALARM
    # Dajemy 10% marginesu błędu, bo dyskretny czas symulacji
    if stopping_dist >= dist * 0.9:
        return 1.0
    return 0.0
