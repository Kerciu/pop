import numpy as np


class SantaPhysics:
    def __init__(self, acc_ranges):
        self.acc_ranges = sorted(acc_ranges, key=lambda x: x["max_weight"])

    def get_max_acc(self, current_weight):
        """Zwraca dopuszczalne przyspieszenie dla danej wagi."""
        for r in self.acc_ranges:
            if current_weight <= r["max_weight"]:
                return r["max_acc"]
        return 0

    def apply_action(self, pos, vel, weight, action_idx):
        """
        Symuluje 1 sekundę.
        Akcje: 0:Wait, 1:Up, 2:Down, 3:Right, 4:Left
        Zwraca: (new_pos, new_vel, new_weight, acc_vector)
        """
        max_acc = self.get_max_acc(weight)
        acc_vec = np.array([0, 0])
        cost = 0  # Zużycie marchewki

        if max_acc > 0:
            if action_idx == 1:
                acc_vec = np.array([0, max_acc])
                cost = 1
            elif action_idx == 2:
                acc_vec = np.array([0, -max_acc])
                cost = 1
            elif action_idx == 3:
                acc_vec = np.array([max_acc, 0])
                cost = 1
            elif action_idx == 4:
                acc_vec = np.array([-max_acc, 0])
                cost = 1

        # Aktualizacja prędkości (natychmiastowa na początku sekundy) [cite: 89]
        new_vel = vel + acc_vec
        new_weight = weight - cost

        # Ruch (dryfowanie przez 1 sek) [cite: 58]
        new_pos = pos + new_vel

        return new_pos, new_vel, new_weight, acc_vec

    def get_action_string(self, action_idx, acc_val):
        """Mapuje indeks akcji na format wyjściowy [cite: 243-247]."""
        if action_idx == 0 or acc_val == 0:
            return "Float 1"
        if action_idx == 1:
            return f"AccUp {acc_val}"
        if action_idx == 2:
            return f"AccDown {acc_val}"
        if action_idx == 3:
            return f"AccRight {acc_val}"
        if action_idx == 4:
            return f"AccLeft {acc_val}"
        return "Float 1"
