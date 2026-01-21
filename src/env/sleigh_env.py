import math

import torch

from core.distance_utils import distance


class SleighEnv:
    # 0-3: Acc +1 (N, S, E, W) -> AUTO FLOAT
    # 4-7: Acc MAX (N, S, E, W) -> AUTO FLOAT
    # 8: Float (Coast) -> FLOAT
    # 9: Load Gifts (Once per visit)
    # 10: Fuel (Once per visit)
    # 11: Deliver
    ACTION_SPACE_SIZE = 12

    def __init__(self, problem, simulator):
        self.problem = problem
        self.sim = simulator
        self.state = None

        # Flagi logiczne
        self.base_interaction_locked = False  # Bezpiecznik bazy
        self.last_dist_to_target = 0

        # Parametry normalizacji (do tensora)
        self.max_coord = 200000.0  # Przybliżony rozmiar mapy
        self.max_vel = 200.0  # Przybliżona max prędkość
        FUEL_RESERVE_THRESHOLD = 5

    def reset(self):
        self.sim.reset()
        self.state = self.sim.state
        self.base_interaction_locked = False
        self.last_dist_to_target = self._get_distance_to_current_target()
        return self._get_observation()

    def step(self, action_id):
        reward = 0
        done = False
        time_passed = 0

        # 1. Zarządzanie bezpiecznikiem bazy (RESET flagi po wylocie)
        dist_to_base = distance(self.state.position, self.sim.lapland_pos)
        if self.base_interaction_locked and dist_to_base > self.problem.D:
            self.base_interaction_locked = False

        # 2. Wykonanie akcji
        prev_dist = self.last_dist_to_target

        # Grupa A: Ruch (0-8) -> Zmienia fizykę I przesuwa czas
        if 0 <= action_id <= 8:
            # Ustawianie przyspieszenia
            ax, ay = 0, 0
            if action_id < 8:
                direction = action_id % 4  # 0=N, 1=S, 2=E, 3=W
                is_max = action_id >= 4

                acc_val = 1.0
                if is_max:
                    # Pobieramy max możliwe przyspieszenie dla aktualnej wagi
                    acc_val = self.sim.accel_table.get_max_acceleration_for_weight(
                        self.state.sleigh_weight
                    )

                if direction == 0:
                    ay = acc_val
                elif direction == 1:
                    ay = -acc_val
                elif direction == 2:
                    ax = acc_val
                elif direction == 3:
                    ax = -acc_val

                # Aplikujemy zmianę sterowania
                self.sim.handle_action(ax, ay, 0, 0)  # Acc only

            # AUTOMATYCZNY FLOAT DLA KAŻDEJ AKCJI RUCHOWEJ
            self.sim.step()
            time_passed = 1
            reward -= 0.5  # Lekka kara za upływ czasu (presja)

        # Grupa B: Interakcje w bazie (9, 10)
        elif action_id == 9:  # Load Gifts
            if dist_to_base <= self.problem.D and not self.base_interaction_locked:
                if not self.state.loaded_gifts and self.state.available_gifts:
                    self.sim.handle_action(0, 0, 1, 0)  # Load
                    reward += 50.0
                    self.base_interaction_locked = True
                else:
                    reward -= 1.0  # Kara za spamowanie bez sensu
            else:
                reward -= 1.0  # Kara za próbę ładowania poza bazą/zablokowaną

        elif action_id == 10:  # Fuel
            if dist_to_base <= self.problem.D and not self.base_interaction_locked:
                if self.state.carrot_count < 20:  # Tylko jak potrzebuje
                    self.sim.handle_action(0, 0, 0, 1)  # Fuel
                    reward += 20.0
                    self.base_interaction_locked = (
                        True  # Blokujemy też tankowanie w tej wizycie
                    )
                else:
                    reward -= 1.0
            else:
                reward -= 1.0

        # Grupa C: Dostarczanie (11)
        elif action_id == 11:  # Deliver
            # Sprawdzamy czy jesteśmy u celu
            if self.state.loaded_gifts:
                gift_id = self.state.loaded_gifts[0]
                target_gift = self.sim.all_gifts_map[gift_id]

                # POPRAWKA: target_gift to obiekt Gift, musimy wziąć .destination
                dist_to_gift = distance(self.state.position, target_gift.destination)

                if dist_to_gift <= self.problem.D:
                    self.sim.handle_action(0, 0, -1, 0)  # Deliver
                    reward += 1000.0
                    self.base_interaction_locked = False
                else:
                    reward -= 2.0  # Kara za zrzut w polu
            else:
                reward -= 2.0  # Nie masz co zrzucać

        # 3. Aktualizacja stanu i nagrody za postęp
        self.state = self.sim.state
        current_dist = self._get_distance_to_current_target()

        # Nagroda za zbliżanie się (Shaping) - tylko jeśli czas płynął
        if time_passed > 0:
            diff = prev_dist - current_dist
            # Jeśli zbliżył się o 100 jednostek -> nagroda +10
            # Jeśli oddalił się -> kara
            reward += diff * 0.1

        self.last_dist_to_target = current_dist

        # Kary śmiertelne
        if self.state.carrot_count <= 0:
            reward -= 200.0
            done = True

        # Sprawdzenie końca czasu lub zadań
        if self.state.current_time >= self.problem.T or (
            not self.state.loaded_gifts and not self.state.available_gifts
        ):
            done = True
            if not self.state.available_gifts and not self.state.loaded_gifts:
                reward += 2000.0  # Bonus za wyczyszczenie mapy

        return self._get_observation(), reward, done, {}

    def _get_distance_to_current_target(self):
        # PRIORYTET 1: PALIWO
        if self.state.carrot_count <= self.FUEL_RESERVE_THRESHOLD:
            # Włączamy tryb paniki - cel to Baza
            return distance(self.state.position, self.sim.lapland_pos)

        # PRIORYTET 2: DOSTARCZANIE
        if self.state.loaded_gifts:
            gift_id = self.state.loaded_gifts[0]
            target_pos = self.sim.all_gifts_map[gift_id].destination
            return distance(self.state.position, target_pos)

        # PRIORYTET 3: POWRÓT PO TOWAR
        else:
            return distance(self.state.position, self.sim.lapland_pos)

    def _get_observation(self):
        s = self.state

        # Logika wyboru celu dla tensora
        target_pos = self.sim.lapland_pos
        target_type = -1.0  # Domyślnie baza

        if s.carrot_count <= self.FUEL_RESERVE_THRESHOLD:
            # REZERWA: Cel to baza, typ celu to Baza (-1.0)
            target_pos = self.sim.lapland_pos
            target_type = -1.0
        elif s.loaded_gifts:
            # PRACA: Cel to klient
            target_pos = self.sim.all_gifts_map[s.loaded_gifts[0]].destination
            target_type = 1.0

        dx = target_pos.c - s.position.c
        dy = target_pos.r - s.position.r
        dist = math.sqrt(dx**2 + dy**2)

        # Fizyka hamowania (Czy muszę hamować?)
        # Droga hamowania = v^2 / (2*a). Przyjmijmy średnie a=5.0
        speed_sq = s.velocity.vc**2 + s.velocity.vr**2
        braking_dist = speed_sq / (2.0 * 5.0 + 1e-5)
        brake_warning = 1.0 if dist < (braking_dist + self.problem.D + 50) else -1.0

        # Normalizacja
        feats = [
            s.position.c / self.max_coord,
            s.position.r / self.max_coord,
            s.velocity.vc / self.max_vel,
            s.velocity.vr / self.max_vel,
            dx / self.max_coord,
            dy / self.max_coord,
            dist / self.max_coord,
            s.sleigh_weight / 5000.0,
            s.carrot_count / 20.0,
            1.0 if s.loaded_gifts else -1.0,  # Czy mam paczkę
            target_type,  # Gdzie lecę
            1.0 if dist <= self.problem.D else -1.0,  # Czy jestem w zasięgu interakcji
            brake_warning,  # CZY HAMOWAĆ?!
            1.0 if self.base_interaction_locked else -1.0,  # Czy bezpiecznik aktywny
        ]

        return torch.FloatTensor(feats)

    @property
    def input_size(self):
        return 14  # Rozmiar listy feats

    @property
    def gifts_map(self):
        """Przekierowanie dla Visualizera i starego kodu."""
        return self.sim.all_gifts_map
