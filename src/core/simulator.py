from typing import Mapping

from core.acceleration_table import AccelerationTable
from models.coordinate import Coordinate
from models.gift import Gift
from models.sleigh_state import SleighState
from models.velocity import Velocity


class Simulator:
    def __init__(
        self,
        t_limit: int,
        range_d: int,
        accel_table: AccelerationTable,
        all_gifts_map: Mapping[str, Gift],
        lapland_pos: Coordinate = Coordinate(0, 0),
    ):
        self.t_limit = t_limit
        self.range_d = range_d
        self.accel_table = accel_table
        self.all_gifts_map = all_gifts_map
        self.lapland_pos = lapland_pos

        # RL State holder
        self.state: SleighState = None

    def reset(self) -> SleighState:
        """Inicjalizuje stan symulacji dla nowego epizodu."""
        # Tworzymy nową instancję Coordinate ręcznie
        start_pos = Coordinate(self.lapland_pos.c, self.lapland_pos.r)

        self.state = SleighState(
            current_time=0,
            position=start_pos,
            velocity=Velocity(0, 0),
            carrot_count=0,
            sleigh_weight=10.0,  # Waga bazowa sań
            available_gifts=list(self.all_gifts_map.keys()),
            loaded_gifts=[],
            delivered_gifts=[],
            last_action_was_acceleration=False,
        )
        return self.state

    def step(self):
        """Przesuwa czas o 1 (Floating) i aktualizuje pozycję."""
        # x = x + vx, y = y + vy
        self.state.position.c += self.state.velocity.vc
        self.state.position.r += self.state.velocity.vr
        self.state.current_time += 1

        # Floating resetuje flagę przyspieszenia
        self.state.last_action_was_acceleration = False

    def handle_action(self, ax: float, ay: float, load_cmd: int, fuel_cmd: int):
        """
        Obsługuje natychmiastowe zmiany stanu (bez upływu czasu).
        """
        # 1. Fizyka (Przyspieszenie)
        if ax != 0 or ay != 0:
            self.state.velocity.vc += ax
            self.state.velocity.vr += ay
            self.state.last_action_was_acceleration = True

        # 2. Paliwo
        if fuel_cmd > 0:
            self.state.carrot_count = 20
            self.state.last_action_was_acceleration = False

        # 3. Ładowanie prezentów (POPRAWKA: Sprawdzamy fizykę zamiast pola max_weight)
        if load_cmd == 1:
            self.state.last_action_was_acceleration = False

            gifts_to_load = []
            # Symulujemy dodawanie prezentów
            current_simulated_weight = self.state.sleigh_weight

            for gift_id in list(self.state.available_gifts):
                gift = self.all_gifts_map[gift_id]
                new_weight = current_simulated_weight + gift.weight

                # Sprawdzamy, czy po dodaniu tego prezentu sanie będą miały jakiekolwiek przyspieszenie
                # Jeśli get_max_acceleration zwraca 0, to znaczy, że jesteśmy przeciążeni.
                if self.accel_table.get_max_acceleration_for_weight(new_weight) > 0:
                    gifts_to_load.append(gift_id)
                    current_simulated_weight += gift.weight

            # Zatwierdzamy zmiany
            for g in gifts_to_load:
                gift = self.all_gifts_map[g]
                self.state.available_gifts.remove(g)
                self.state.loaded_gifts.append(g)
                self.state.sleigh_weight += gift.weight

        # 4. Dostarczanie prezentów
        elif load_cmd == -1:
            self.state.last_action_was_acceleration = False
            if self.state.loaded_gifts:
                gift_id = self.state.loaded_gifts[0]
                gift = self.all_gifts_map[gift_id]

                self.state.loaded_gifts.pop(0)
                self.state.delivered_gifts.append(gift_id)
                self.state.sleigh_weight -= gift.weight
