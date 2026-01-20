import math

import torch

from core.distance_utils import calculate_braking_signal
from models.problem import Problem
from models.sleigh_state import SleighState


class StateEncoder:
    # Dodajemy parametr map_limit do __init__
    def __init__(self, problem: Problem, simulator, map_limit=100000.0):
        self.problem = problem
        self.sim = simulator
        self.gifts_map = {g.name: g for g in problem.gifts}

        # Używamy dynamicznego limitu!
        self.MAX_COORD = float(map_limit)
        self.MAX_VELOCITY = 100.0  # To można zostawić, albo też skalować

        self.output_size = 8

    def _get_active_target(self, state: SleighState):
        if state.loaded_gifts:
            tgt_name = state.loaded_gifts[0]
            if tgt_name in self.gifts_map:
                dest = self.gifts_map[tgt_name].destination
                return dest.c, dest.r
        return 0.0, 0.0

    def encode(self, state: SleighState) -> torch.Tensor:
        target_x, target_y = self._get_active_target(state)

        dx = target_x - state.position.c
        dy = target_y - state.position.r
        dist = math.sqrt(dx**2 + dy**2)

        velocity_towards = 0.0
        if dist > 0:
            dir_x, dir_y = dx / dist, dy / dist
            velocity_towards = state.velocity.vc * dir_x + state.velocity.vr * dir_y

        max_acc = self.sim.accel_table.get_max_acceleration_for_weight(
            state.sleigh_weight
        )
        must_brake = calculate_braking_signal(dist, velocity_towards, max_acc)

        # Normalizacja
        features = [
            1.0 if state.loaded_gifts else 0.0,
            1.0 if state.carrot_count < 20 else 0.0,
            # Kluczowe: używamy self.MAX_COORD
            max(-1.0, min(1.0, dx / self.MAX_COORD)),
            max(-1.0, min(1.0, dy / self.MAX_COORD)),
            state.velocity.vc / self.MAX_VELOCITY,
            state.velocity.vr / self.MAX_VELOCITY,
            must_brake,
            1.0 if state.last_action_was_acceleration else 0.0,
        ]

        return torch.tensor(features, dtype=torch.float32)
