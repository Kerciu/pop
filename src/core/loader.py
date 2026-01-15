from core.acceleration_table import AccelerationTable
from core.simulator import Simulator
from models.problem import Problem


def load_problem(path: str):
    """
    Wczytuje problem używając Twojej istniejącej infrastruktury (InputParser -> Problem).
    Zwraca obiekt problemu i zainicjalizowany symulator.
    """
    # 1. Wczytanie problemu (Twoja klasa Problem używa w środku InputParser)
    problem = Problem(path)

    # 2. Inicjalizacja tabeli przyspieszeń (tak jak w Twoim starym main.py)
    # Zakładam, że klasa AccelerationTable jest w core/acceleration_table.py
    # i przyjmuje listę ranges
    accel_table = AccelerationTable(problem.acceleration_ranges)

    # 3. Mapa prezentów
    all_gifts_map = {gift.name: gift for gift in problem.gifts}

    # 4. Inicjalizacja Symulatora (tak jak w Twoim starym main.py)
    simulator = Simulator(
        t_limit=problem.T,
        range_d=problem.D,
        accel_table=accel_table,
        all_gifts_map=all_gifts_map,
    )

    return problem, simulator
