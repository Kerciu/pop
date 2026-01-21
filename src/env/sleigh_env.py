import math

import torch

from core.distance_utils import distance


class SleighEnv:
    ACTION_SPACE_SIZE = 12

    # Parametry normalizacji
    MAX_COORD = 150000.0  # Dostosowane do huge_challenge
    MAX_VEL = 250.0

    def __init__(self, problem, simulator):
        self.problem = problem
        self.sim = simulator
        self.state = None
        self.base_interaction_locked = False
        self.last_dist_to_target = 0

        # --- DYNAMICZNE SKALOWANIE MAPY ---
        # Skanujemy prezenty, aby znaleźć granice świata (jak w visualizerze)
        max_dist = 1.0
        for g in problem.gifts:
            # Szukamy najdalszej współrzędnej (promień mapy)
            d = max(abs(g.destination.c), abs(g.destination.r))
            if d > max_dist:
                max_dist = d

        # Dodajemy margines (np. 10%)
        self.map_limit = max_dist * 1.1

        # Ustawiamy parametry na podstawie odczytanych danych
        self.MAX_COORD = float(self.map_limit)
        self.MAX_VEL = 250.0

        # Definiujemy dynamiczny "dzielnik" dla rezerwy.
        # Chcemy, aby na krańcu mapy (self.map_limit) dodatek do rezerwy wynosił np. 10 marchewek.
        # Wzór: Dodatek = Dystans / DZIELNIK
        # Więc: 10 = map_limit / DZIELNIK  =>  DZIELNIK = map_limit / 10
        self.reserve_divisor = self.map_limit / 10.0

    def reset(self):
        self.sim.reset()
        self.state = self.sim.state
        self.base_interaction_locked = False
        self.last_dist_to_target = self._get_distance_to_current_target()
        return self._get_observation()

    def _calculate_dynamic_reserve(self):
        """
        Oblicza próg paliwa w oparciu o faktyczny rozmiar mapy.
        """
        dist_to_base = distance(self.state.position, self.sim.lapland_pos)

        # Baza: żelazna rezerwa na manewry przy lądowaniu
        base_reserve = 3.0

        # Dodatek dystansowy skalowany do wielkości mapy
        # Dzięki temu na krańcu mapy zawsze będziesz miał +10 marchewek rezerwy,
        # niezależnie czy mapa ma rozmiar 1000 czy 100000.
        distance_buffer = dist_to_base / self.reserve_divisor

        return base_reserve + distance_buffer

    def step(self, action_id):
        reward = 0
        done = False
        time_passed = 0

        dist_to_base = distance(self.state.position, self.sim.lapland_pos)
        if self.base_interaction_locked and dist_to_base > self.problem.D:
            self.base_interaction_locked = False

        prev_dist = self.last_dist_to_target

        # A. RUCH
        if 0 <= action_id <= 8:
            ax, ay = 0, 0
            if action_id < 8:
                direction = action_id % 4
                is_max = action_id >= 4
                acc_val = 1.0
                if is_max:
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

                self.sim.handle_action(ax, ay, 0, 0)

            self.sim.step()
            time_passed = 1
            reward -= 0.5

        # B. LOAD (Tylko w bazie)
        elif action_id == 9:
            if dist_to_base <= self.problem.D and not self.base_interaction_locked:
                if not self.state.loaded_gifts and self.state.available_gifts:
                    self.sim.handle_action(0, 0, 1, 0)
                    reward += 50.0
                    self.base_interaction_locked = True
                else:
                    reward -= 1.0
            else:
                reward -= 1.0

        # C. FUEL (Tylko w bazie)
        elif action_id == 10:
            if dist_to_base <= self.problem.D and not self.base_interaction_locked:
                # Tankujemy, jeśli mamy mniej niż 80% baku
                if self.state.carrot_count < (self.sim.MAX_FUEL * 0.8):
                    self.sim.handle_action(0, 0, 0, 1)
                    reward += 30.0  # Wyższa nagroda za tankowanie
                    self.base_interaction_locked = True
                else:
                    reward -= 1.0
            else:
                reward -= 1.0

        # D. DELIVER
        elif action_id == 11:
            if self.state.loaded_gifts:
                target_gift = self.sim.all_gifts_map[self.state.loaded_gifts[0]]
                if (
                    distance(self.state.position, target_gift.destination)
                    <= self.problem.D
                ):
                    self.sim.handle_action(0, 0, -1, 0)
                    reward += 1000.0
                    self.base_interaction_locked = False
                else:
                    reward -= 2.0
            else:
                reward -= 2.0

        self.state = self.sim.state
        current_dist = self._get_distance_to_current_target()

        if time_passed > 0:
            diff = prev_dist - current_dist
            reward += diff * 0.1

        self.last_dist_to_target = current_dist

        if self.state.carrot_count <= 0:
            reward -= 500.0  # Większa kara za śmierć
            done = True

        if self.state.current_time >= self.problem.T:
            done = True

        if not self.state.available_gifts and not self.state.loaded_gifts:
            reward += 5000.0
            done = True

        return self._get_observation(), reward, done, {}

    def _get_distance_to_current_target(self):
        s = self.state
        reserve_threshold = self._calculate_dynamic_reserve()  # <--- DYNAMICZNA REZERWA

        # 1. PALIWO
        if s.carrot_count <= reserve_threshold:
            return distance(s.position, self.sim.lapland_pos)
        # 2. DOSTAWA
        if s.loaded_gifts:
            target_pos = self.sim.all_gifts_map[s.loaded_gifts[0]].destination
            return distance(s.position, target_pos)
        # 3. POWRÓT
        else:
            return distance(s.position, self.sim.lapland_pos)

    def _get_observation(self):
        s = self.state
        reserve_threshold = self._calculate_dynamic_reserve()  # <--- DYNAMICZNA REZERWA

        target_pos = self.sim.lapland_pos

        is_panic = s.carrot_count <= reserve_threshold
        has_gifts = len(s.loaded_gifts) > 0

        if is_panic:
            target_pos = self.sim.lapland_pos
        elif has_gifts:
            target_pos = self.sim.all_gifts_map[s.loaded_gifts[0]].destination
        else:
            target_pos = self.sim.lapland_pos

        dx = target_pos.c - s.position.c
        dy = target_pos.r - s.position.r
        dist = math.sqrt(dx**2 + dy**2)

        speed_sq = s.velocity.vc**2 + s.velocity.vr**2
        braking_dist = speed_sq / (2.0 * 5.0 + 1e-5)
        must_brake = 1.0 if dist < (braking_dist + self.problem.D + 100) else 0.0

        in_base_range = (
            1.0 if distance(s.position, self.sim.lapland_pos) <= self.problem.D else 0.0
        )

        features = [
            1.0 if is_panic else 0.0,
            1.0 if has_gifts else 0.0,
            in_base_range,
            max(-1.0, min(1.0, dx / self.MAX_COORD)),
            max(-1.0, min(1.0, dy / self.MAX_COORD)),
            s.velocity.vc / self.MAX_VEL,
            s.velocity.vr / self.MAX_VEL,
            must_brake,
            s.sleigh_weight / 5000.0,
            # Dodajemy informację o stanie paliwa (znormalizowaną do max_fuel)
            s.carrot_count / float(self.sim.MAX_FUEL),
        ]

        return torch.tensor(features, dtype=torch.float32)

    @property
    def input_size(self):
        return 10  # Dodaliśmy jedną cechę (poziom paliwa)

    @property
    def gifts_map(self):
        return self.sim.all_gifts_map
