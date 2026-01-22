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

        max_dist = 1.0
        for g in problem.gifts:
            d = max(abs(g.destination.c), abs(g.destination.r))
            if d > max_dist:
                max_dist = d

        self.map_limit = max_dist * 1.2
        self.MAX_COORD = float(self.map_limit)

        self.MAX_VEL = max(100.0, self.map_limit / 5.0)

    def reset(self):
        self.sim.reset()
        self.state = self.sim.state
        self.base_interaction_locked = False
        self.last_dist_to_target = self._get_distance_to_current_target()
        return self._get_observation()

    def step(self, action_id):
        reward = 0
        done = False

        reward -= 0.1

        dist_to_base = distance(self.state.position, self.sim.lapland_pos)
        in_base = dist_to_base <= self.problem.D

        if self.base_interaction_locked and not in_base:
            self.base_interaction_locked = False

        prev_dist = self.last_dist_to_target

        # MOVE
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
            if action_id < 8 and self.state.carrot_count < self.sim.state.carrot_count:
                reward -= 0.5

        # LOAD (9)
        elif action_id == 9:
            if in_base and not self.base_interaction_locked:
                if not self.state.loaded_gifts and self.state.available_gifts:
                    self.sim.handle_action(0, 0, 1, 0)
                    reward += 200.0
                    self.base_interaction_locked = True
                else:
                    reward -= 5.0
            else:
                reward -= 5.0

        # FUEL (10)
        elif action_id == 10:
            if in_base and not self.base_interaction_locked:
                if self.state.carrot_count < self.sim.MAX_FUEL:
                    self.sim.handle_action(0, 0, 0, 1)
                    reward += 50.0
                    self.base_interaction_locked = True
                else:
                    reward -= 5.0
            else:
                reward -= 5.0

        # DELIVER (11)
        elif action_id == 11:
            if self.state.loaded_gifts:
                target = self.sim.all_gifts_map[self.state.loaded_gifts[0]]
                dist_to_gift = distance(self.state.position, target.destination)

                if dist_to_gift <= self.problem.D:
                    self.sim.handle_action(0, 0, -1, 0)
                    reward += 5000.0
                    self.base_interaction_locked = False
                else:
                    reward -= 10.0
            else:
                reward -= 10.0

        self.state = self.sim.state
        current_dist = self._get_distance_to_current_target()

        diff = prev_dist - current_dist
        reward += diff * 0.005

        self.last_dist_to_target = current_dist

        if self.state.carrot_count <= 0:
            reward -= 500.0
            done = True

        if self.state.current_time >= self.problem.T:
            done = True

        if not self.state.available_gifts and not self.state.loaded_gifts:
            reward += 10000.0
            done = True

        return self._get_observation(), reward, done, {}

    def _get_distance_to_current_target(self):
        """Metoda pomocnicza do obliczania dystansu dla shaping reward."""
        s = self.state
        fuel_ratio = s.carrot_count / float(self.sim.MAX_FUEL)

        if fuel_ratio < 0.2:
            target_pos = self.sim.lapland_pos
        elif s.loaded_gifts:
            target_pos = self.sim.all_gifts_map[s.loaded_gifts[0]].destination
        else:
            target_pos = self.sim.lapland_pos

        return distance(s.position, target_pos)

    def _get_observation(self):
        s = self.state

        target_pos = self.sim.lapland_pos
        fuel_ratio = s.carrot_count / float(self.sim.MAX_FUEL)

        has_gift = len(s.loaded_gifts) > 0

        if fuel_ratio < 0.2:
            target_pos = self.sim.lapland_pos
        elif has_gift:
            target_pos = self.sim.all_gifts_map[s.loaded_gifts[0]].destination
        else:
            target_pos = self.sim.lapland_pos

        dx = target_pos.c - s.position.c
        dy = target_pos.r - s.position.r
        dist = math.sqrt(dx**2 + dy**2)

        in_range_of_target = 1.0 if dist <= self.problem.D else 0.0

        features = [
            1.0 if has_gift else 0.0,
            fuel_ratio,
            1.0
            if distance(s.position, self.sim.lapland_pos) <= self.problem.D
            else 0.0,
            in_range_of_target,
            max(-1.0, min(1.0, dx / self.MAX_COORD)),
            max(-1.0, min(1.0, dy / self.MAX_COORD)),
            s.velocity.vc / self.MAX_VEL,
            s.velocity.vr / self.MAX_VEL,
            1.0 if dist < (s.velocity.vc**2 + s.velocity.vr**2) / 10.0 else 0.0,
            s.sleigh_weight / 100.0,
            dist / self.MAX_COORD,
            1.0,
        ]

        return torch.tensor(features, dtype=torch.float32)

    @property
    def input_size(self):
        return 12

    @property
    def gifts_map(self):
        return self.sim.all_gifts_map
