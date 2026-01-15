import torch

from models.problem import Problem
from models.sleigh_state import SleighState


class StateEncoder:
    def __init__(self, problem: Problem):
        self.problem = problem
        # ZWIĘKSZONE dla mapy huge (żeby wartości były znormalizowane < 1.0)
        self.max_coord = 500.0
        self.max_time = float(problem.T)
        self.max_weight = 1000.0
        self.output_size = 12
        self.gifts_map = {g.name: g for g in problem.gifts}

    def _get_target_pos(self, state: SleighState):
        target_x, target_y = 0.0, 0.0
        if state.loaded_gifts:
            tgt_name = state.loaded_gifts[0]
            if tgt_name in self.gifts_map:
                dest = self.gifts_map[tgt_name].destination
                target_x, target_y = dest.c, dest.r
        return target_x, target_y

    def encode(self, state: SleighState) -> torch.Tensor:
        target_x, target_y = self._get_target_pos(state)

        # Normalizacja względem powiększonej mapy
        pos_c = max(-self.max_coord, min(self.max_coord, state.position.c))
        pos_r = max(-self.max_coord, min(self.max_coord, state.position.r))

        dx = (target_x - state.position.c) / self.max_coord
        dy = (target_y - state.position.r) / self.max_coord

        vc = max(-50, min(50, state.velocity.vc))
        vr = max(-50, min(50, state.velocity.vr))

        features = [
            pos_c / self.max_coord,
            pos_r / self.max_coord,
            vc / 20.0,
            vr / 20.0,
            state.sleigh_weight / self.max_weight,
            state.carrot_count / 100.0,  # Normalizacja do 100 marchewek
            state.current_time / self.max_time,
            dx,
            dy,
            1.0 if state.last_action_was_acceleration else 0.0,
            1.0 if len(state.loaded_gifts) > 0 else 0.0,
            1.0 if len(state.available_gifts) == 0 else 0.0,
        ]
        return torch.tensor(features, dtype=torch.float32)
