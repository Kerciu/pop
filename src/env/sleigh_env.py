import gymnasium as gym

from core.actions import Action, solve_knapsack_greedy
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
        self.encoder = StateEncoder(problem, simulator)
        self.state = None
        self.prev_dist = 0.0
        self.gifts_map = {g.name: g for g in problem.gifts}
        self.map_limit = 100000.0
        self.MAX_FUEL = 300

        self.ACTION_MAPPING = {
            0: Action.AccUp,
            1: Action.AccDown,
            2: Action.AccLeft,
            3: Action.AccRight,
            4: Action.Floating,
            5: Action.LoadCarrots,
            6: Action.LoadGifts,
            7: Action.DeliverGift,
        }
        self.action_space_size = len(self.ACTION_MAPPING)

    def reset(self):
        initial_gifts = [g.name for g in self.problem.gifts]
        self.state = SleighState(
            current_time=0,
            position=Coordinate(0, 0),
            velocity=Velocity(0, 0),
            sleigh_weight=10,
            carrot_count=100,
            loaded_gifts=[],
            available_gifts=initial_gifts,
            delivered_gifts=[],
            last_action_was_acceleration=False,
        )
        target_pos = self._get_target_pos()
        self.prev_dist = distance(self.state.position, target_pos)
        return self.encoder.encode(self.state)

    def _get_target_pos(self):
        if self.state.loaded_gifts:
            g_name = self.state.loaded_gifts[0]
            if g_name in self.gifts_map:
                return self.gifts_map[g_name].destination
        return self.sim.lapland_pos

    def step(self, action_id: int):
        reward = 0.0
        step_penalty = -0.01
        action_enum = self.ACTION_MAPPING[action_id]
        param = 1
        sim_action = action_enum

        # 1. RUCH
        if action_enum in [
            Action.AccUp,
            Action.AccDown,
            Action.AccLeft,
            Action.AccRight,
        ]:
            max_acc = self.sim.accel_table.get_max_acceleration_for_weight(
                self.state.sleigh_weight
            )
            param = max_acc
            if self.state.carrot_count <= 0 or max_acc == 0:
                sim_action = Action.Floating
                param = 1
                reward -= 0.5

        # 2. TANKOWANIE
        elif action_enum == Action.LoadCarrots:
            dist_base = distance(self.state.position, self.sim.lapland_pos)
            if dist_base > self.problem.D:
                sim_action = Action.Floating
                reward -= 1.0
            else:
                needed = max(0, 100 - self.state.carrot_count)
                if needed > 0:
                    param = needed
                    reward += 1.0
                else:
                    sim_action = Action.Floating
                    reward -= 0.5

        # 3. DOSTARCZANIE
        elif action_enum == Action.DeliverGift:
            if not self.state.loaded_gifts:
                sim_action = Action.Floating
                reward -= 1.0
            else:
                target = self.gifts_map[self.state.loaded_gifts[0]]
                if distance(self.state.position, target.destination) > self.problem.D:
                    sim_action = Action.Floating
                    reward -= 1.0
                else:
                    param = 0

        # 4. ŁADOWANIE (BEZ UPŁYWU CZASU)
        elif action_enum == Action.LoadGifts:
            dist_base = distance(self.state.position, self.sim.lapland_pos)

            if dist_base > self.problem.D or not self.state.available_gifts:
                sim_action = Action.Floating
                reward -= 1.0
            else:
                real_max_weight = self.sim.accel_table.ranges[-1].max_weight_inclusive
                gifts_to_load_ids = solve_knapsack_greedy(
                    self.state.available_gifts,
                    self.gifts_map,
                    real_max_weight,
                    self.state.sleigh_weight,
                )

                if not gifts_to_load_ids:
                    sim_action = Action.Floating
                    reward -= 1.0
                else:
                    try:
                        # MANUALNA ZMIANA STANU (BYPASS SYMULATORA)
                        loaded_count = 0
                        loaded_set = set(gifts_to_load_ids)
                        new_available = []

                        for g_name in self.state.available_gifts:
                            if g_name in loaded_set:
                                gift = self.gifts_map[g_name]
                                self.state.loaded_gifts.append(g_name)
                                self.state.sleigh_weight += gift.weight
                                # UWAGA: TU JUŻ NIE DODAJEMY CZASU!
                                # self.state.current_time += 1 <--- USUNIĘTE
                                loaded_count += 1
                            else:
                                new_available.append(g_name)

                        self.state.available_gifts = new_available

                        reward += 5.0 + loaded_count * 0.5
                        return self._finalize_step(
                            reward, step_penalty, len(self.state.delivered_gifts)
                        )

                    except Exception as e:
                        print(f"Błąd LoadGifts: {e}")
                        return self.encoder.encode(self.state), -50.0, True, {}

        # WYKONANIE
        if action_enum != Action.LoadGifts:
            prev_delivered = len(self.state.delivered_gifts)
            try:
                self.state = self.sim.apply_action(self.state, sim_action, param)
            except Exception:
                return self.encoder.encode(self.state), -50.0, True, {}
            return self._finalize_step(reward, step_penalty, prev_delivered)

        return self.encoder.encode(self.state), reward, False, {}

    def _finalize_step(self, reward, step_penalty, prev_delivered):
        target_pos = self._get_target_pos()
        curr_dist = distance(self.state.position, target_pos)

        dist_improvement = self.prev_dist - curr_dist
        if abs(dist_improvement) < 1000:
            reward += dist_improvement * 1.0
        self.prev_dist = curr_dist

        curr_delivered = len(self.state.delivered_gifts)
        if curr_delivered > prev_delivered:
            reward += 500.0
            new_target = self._get_target_pos()
            self.prev_dist = distance(self.state.position, new_target)

        reward += step_penalty

        # Koniec gry?
        done = self.state.current_time >= self.problem.T
        if curr_delivered == len(self.problem.gifts):
            done = True
            reward += 1000.0

        if (
            abs(self.state.position.c) > self.map_limit
            or abs(self.state.position.r) > self.map_limit
        ):
            done = True
            reward -= 50.0

        return self.encoder.encode(self.state), reward, done, {}
