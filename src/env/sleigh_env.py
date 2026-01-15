import gymnasium as gym

from core.actions import Action
from core.distance_utils import distance
from core.simulator import Simulator
from env.state_encoder import StateEncoder
from models.coordinate import Coordinate
from models.problem import Problem
from models.sleigh_state import SleighState
from models.velocity import Velocity


class SleighEnv(gym.Env):
    def __init__(self, problem: Problem, simulator: Simulator):
        self.problem = problem
        self.sim = simulator
        self.encoder = StateEncoder(problem)
        self.state = None
        self.prev_dist = 0.0
        self.gifts_map = {g.name: g for g in problem.gifts}
        self.map_limit = 500.0
        self.useless_interact_count = 0

        # Limit baku (zabezpieczenie przed reward hacking)
        self.MAX_FUEL = 300

        # --- DEFINICJA AKCJI Z PARAMETRAMI ---
        self.ACTION_MAPPING = {
            # --- RUCH ---
            0: (Action.AccUp, 1),
            1: (Action.AccUp, 3),
            2: (Action.AccDown, 1),
            3: (Action.AccDown, 3),
            4: (Action.AccLeft, 1),
            5: (Action.AccLeft, 3),
            6: (Action.AccRight, 1),
            7: (Action.AccRight, 3),
            # --- INTERAKCJE ---
            8: (Action.Floating, 1),
            9: (Action.LoadCarrots, 20),
            10: (Action.LoadCarrots, 100),
            11: (Action.LoadGifts, 0),
            12: (Action.DeliverGift, 0),
        }

        self.action_space_size = len(self.ACTION_MAPPING)

    def reset(self):
        initial_gifts = [g.name for g in self.problem.gifts]

        self.state = SleighState(
            current_time=0,
            position=Coordinate(0, 0),
            velocity=Velocity(0, 0),
            sleigh_weight=0,
            # ZMIANA: Startujemy z małą ilością paliwa (np. 10),
            # żeby mógł chociaż ruszyć do tankowania, jeśli sieć zgłupieje.
            carrot_count=10,
            loaded_gifts=[],
            available_gifts=initial_gifts,
            delivered_gifts=[],
            last_action_was_acceleration=False,
        )
        target_pos = self._get_target_pos()
        self.prev_dist = distance(self.state.position, target_pos)
        self.useless_interact_count = 0
        return self.encoder.encode(self.state)

    def _get_target_pos(self):
        if self.state.loaded_gifts:
            g_name = self.state.loaded_gifts[0]
            if g_name in self.gifts_map:
                return self.gifts_map[g_name].destination
        return self.sim.lapland_pos

    def step(self, action_id: int):
        reward = 0.0
        step_penalty = -0.1

        action_enum, param = self.ACTION_MAPPING[action_id]

        # --- ZABEZPIECZENIA I WALIDACJA ---

        # A. Walidacja Ruchu
        if action_enum in [
            Action.AccUp,
            Action.AccDown,
            Action.AccLeft,
            Action.AccRight,
        ]:
            max_acc = self.sim.accel_table.get_max_acceleration_for_weight(
                self.state.sleigh_weight
            )
            if param > max_acc:
                param = max_acc

            if self.state.carrot_count <= 0:
                reward -= 1.0
                action_enum = Action.Floating
                param = 1

        # B. Walidacja Interakcji
        action_success = True

        if action_enum in [Action.LoadCarrots, Action.LoadGifts, Action.DeliverGift]:
            speed = abs(self.state.velocity.vc) + abs(self.state.velocity.vr)
            dist_base = distance(self.state.position, self.sim.lapland_pos)

            # 1. Czy stoi w miejscu?
            if speed > 1.0:
                reward -= 1.0
                action_enum = Action.Floating
                action_success = False

            # 2. Logika dla LoadCarrots (TUTAJ BYŁ BŁĄD)
            elif action_enum == Action.LoadCarrots:
                if dist_base > self.problem.D:
                    action_success = False  # Nie jesteśmy w bazie
                elif self.state.carrot_count >= self.MAX_FUEL:
                    # FIX: Jeśli bak pełny, zabraniamy tankowania i dajemy karę!
                    action_success = False
                    reward -= 1.0  # Kara za chciwość
                else:
                    reward += 1.0  # Nagroda za tankowanie (tylko gdy potrzebne)

            # 3. Logika dla LoadGifts
            elif action_enum == Action.LoadGifts:
                if dist_base > self.problem.D or not self.state.available_gifts:
                    action_success = False
                # FIX: Jeśli mamy już prezenty na saniach, a próbujemy dobrać (przy param=0 ładowane są wszystkie),
                # to symulator może to zignorować, ale agent nie powinien spamować.
                elif len(self.state.loaded_gifts) > 0:
                    # Pozwalamy dobrać, ale mniejsza nagroda, żeby nie spamował w kółko
                    reward += 1.0
                else:
                    reward += 5.0  # Duża nagroda za PIERWSZY załadunek

            # 4. Logika dla DeliverGift
            elif action_enum == Action.DeliverGift:
                if not self.state.loaded_gifts:
                    action_success = False
                else:
                    target = self.gifts_map[self.state.loaded_gifts[0]]
                    if (
                        distance(self.state.position, target.destination)
                        > self.problem.D
                    ):
                        action_success = False

        # Kara za nieudaną akcję
        if not action_success and action_enum != Action.Floating:
            action_enum = Action.Floating
            reward -= 1.0
            self.useless_interact_count += 1
        elif action_success:
            self.useless_interact_count = 0

        # --- WYKONANIE W SYMULATORZE ---
        prev_delivered = len(self.state.delivered_gifts)

        try:
            self.state = self.sim.apply_action(self.state, action_enum, param)
        except Exception:
            return self.encoder.encode(self.state), -50.0, True, {}

        # --- REWARD SHAPING ---
        target_pos = self._get_target_pos()
        curr_dist = distance(self.state.position, target_pos)

        # Reset dystansu po zmianie celu
        if action_enum in [Action.LoadGifts, Action.DeliverGift]:
            self.prev_dist = curr_dist
        else:
            dist_improvement = self.prev_dist - curr_dist
            reward += dist_improvement * 0.5
            self.prev_dist = curr_dist

        curr_delivered = len(self.state.delivered_gifts)
        if curr_delivered > prev_delivered:
            reward += 100.0
            new_target = self._get_target_pos()
            self.prev_dist = distance(self.state.position, new_target)

        reward += step_penalty

        done = self.state.current_time >= self.problem.T
        if curr_delivered == len(self.problem.gifts):
            done = True
            reward += 200.0

        if (
            abs(self.state.position.c) > self.map_limit
            or abs(self.state.position.r) > self.map_limit
        ):
            done = True
            reward -= 50.0

        if self.useless_interact_count > 30:
            done = True
            reward -= 20.0

        return self.encoder.encode(self.state), reward, done, {}
