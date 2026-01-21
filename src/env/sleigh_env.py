import math

import torch

from core.distance_utils import distance


class SleighEnv:
    ACTION_SPACE_SIZE = 12

    def __init__(self, problem, simulator):
        self.problem = problem
        self.sim = simulator
        self.state = None
        self.base_interaction_locked = False

        # --- 1. SKALOWANIE MAPY I FIZYKI ---
        max_dist = 1.0
        for g in problem.gifts:
            d = max(abs(g.destination.c), abs(g.destination.r))
            if d > max_dist:
                max_dist = d

        # Margines mapy
        self.map_limit = max_dist * 1.2
        self.MAX_COORD = float(self.map_limit)

        # DYNAMICZNA PRĘDKOŚĆ: Max prędkość to 1/2 wielkości mapy
        self.MAX_VEL = self.map_limit * 0.5

        # Rezerwa: 5 + 10% dystansu mapy
        self.sim.MAX_FUEL = 100
        self.reserve_divisor = self.map_limit / 10.0

    def reset(self):
        self.sim.reset()
        self.state = self.sim.state
        self.base_interaction_locked = False
        return self._get_observation()

    def _calculate_dynamic_reserve(self):
        dist_to_base = distance(self.state.position, self.sim.lapland_pos)
        return 5.0 + (dist_to_base / self.reserve_divisor)

    def step(self, action_id):
        reward = 0
        done = False

        # 1. Obliczenia wstępne (PRZED IF-ami!)
        dist_to_base = distance(self.state.position, self.sim.lapland_pos)
        in_base = dist_to_base <= self.problem.D

        # Reset blokady bazy po wylocie
        if self.base_interaction_locked and not in_base:
            self.base_interaction_locked = False

        # --- A. RUCH (0-8) ---
        if action_id <= 8:
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
                    ay = acc_val  # N
                elif direction == 1:
                    ay = -acc_val  # S
                elif direction == 2:
                    ax = acc_val  # E
                elif direction == 3:
                    ax = -acc_val  # W

                self.sim.handle_action(ax, ay, 0, 0)

            self.sim.step()
            reward -= 1.0  # Mała kara za czas

        # --- B. AKCJE LOGICZNE (Teraz elif jest bezpośrednio po if!) ---

        # LOAD (9)
        elif action_id == 9:
            if in_base and not self.base_interaction_locked:
                if not self.state.loaded_gifts and self.state.available_gifts:
                    self.sim.handle_action(0, 0, 1, 0)
                    reward += 50.0  # Nagroda za załadunek
                    self.base_interaction_locked = True
                else:
                    reward -= 500.0  # Błąd logiczny
            else:
                reward -= 500.0  # Klikanie w polu

        # FUEL (10)
        elif action_id == 10:
            if in_base and not self.base_interaction_locked:
                if self.state.carrot_count < (self.sim.MAX_FUEL * 0.9):
                    self.sim.handle_action(0, 0, 0, 1)
                    reward += 50.0  # Nagroda za tankowanie
                    self.base_interaction_locked = True
                else:
                    reward -= 500.0
            else:
                reward -= 500.0

        # DELIVER (11)
        elif action_id == 11:
            success = False
            if self.state.loaded_gifts:
                target = self.sim.all_gifts_map[self.state.loaded_gifts[0]]
                if distance(self.state.position, target.destination) <= self.problem.D:
                    self.sim.handle_action(0, 0, -1, 0)
                    reward += 20000.0  # DUŻA NAGRODA
                    self.base_interaction_locked = False
                    success = True

            if not success:
                reward -= 500.0  # BARDZO DUŻA KARA ZA PUDŁO

        # --- AKTUALIZACJA I KARA ZA DYSTANS ---
        self.state = self.sim.state
        dist_to_target = self._get_current_target_dist()

        # Kara za odległość (im dalej od celu, tym gorzej)
        reward -= dist_to_target / self.MAX_COORD

        # --- WARUNKI KOŃCA ---
        if self.state.carrot_count <= 0:
            reward -= 1000.0
            done = True

        if self.state.current_time >= self.problem.T:
            done = True

        if not self.state.available_gifts and not self.state.loaded_gifts:
            reward += 100000.0  # Jackpot
            done = True

        return self._get_observation(), reward, done, {}

    def _get_current_target_dist(self):
        """Zwraca dystans do aktualnego celu logicznego."""
        s = self.state
        limit = self._calculate_dynamic_reserve()

        if s.carrot_count <= limit:
            return distance(s.position, self.sim.lapland_pos)
        if s.loaded_gifts:
            tgt = self.sim.all_gifts_map[s.loaded_gifts[0]]
            return distance(s.position, tgt.destination)
        return distance(s.position, self.sim.lapland_pos)

    def _get_observation(self):
        s = self.state
        limit = self._calculate_dynamic_reserve()

        # Wyznaczanie celu
        target_pos = self.sim.lapland_pos
        is_panic = s.carrot_count <= limit
        has_gifts = len(s.loaded_gifts) > 0

        if is_panic:
            target_pos = self.sim.lapland_pos
        elif has_gifts:
            target_pos = self.sim.all_gifts_map[s.loaded_gifts[0]].destination

        # Wektory
        dx = target_pos.c - s.position.c
        dy = target_pos.r - s.position.r
        dist = math.sqrt(dx**2 + dy**2)

        # Fizyka hamowania
        speed_sq = s.velocity.vc**2 + s.velocity.vr**2
        brake_dist = speed_sq / 10.0
        must_brake = (
            1.0 if dist < (brake_dist + self.problem.D + self.map_limit * 0.1) else 0.0
        )

        dist_base = distance(s.position, self.sim.lapland_pos)
        in_base = dist_base <= self.problem.D

        # --- 3. PODPOWIEDZI AKCJI (Action Masks) ---
        can_load = (
            1.0
            if (
                in_base
                and not s.loaded_gifts
                and s.available_gifts
                and not self.base_interaction_locked
            )
            else 0.0
        )
        can_fuel = (
            1.0
            if (in_base and s.carrot_count < 90 and not self.base_interaction_locked)
            else 0.0
        )
        can_deliver = 0.0
        if has_gifts:
            tgt = self.sim.all_gifts_map[s.loaded_gifts[0]]
            if distance(s.position, tgt.destination) <= self.problem.D:
                can_deliver = 1.0

        features = [
            1.0 if is_panic else 0.0,
            1.0 if has_gifts else 0.0,
            1.0 if in_base else 0.0,
            max(-1.0, min(1.0, dx / self.MAX_COORD)),
            max(-1.0, min(1.0, dy / self.MAX_COORD)),
            s.velocity.vc / self.MAX_VEL,
            s.velocity.vr / self.MAX_VEL,
            must_brake,
            s.sleigh_weight / 50.0,
            s.carrot_count / 100.0,
            can_load,
            can_fuel,
            can_deliver,
        ]

        return torch.tensor(features, dtype=torch.float32)

    @property
    def input_size(self):
        return 13

    @property
    def gifts_map(self):
        return self.sim.all_gifts_map
