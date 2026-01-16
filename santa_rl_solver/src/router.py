import numpy as np

from src.physics import SantaPhysics


class SantaRouter:
    def __init__(self, data, pilot_agent):
        self.T = data["T"]
        self.D = data["D"]
        self.gifts = data["gifts"]
        self.pilot = pilot_agent
        self.physics = SantaPhysics(data["acc_ranges"])

    def solve(self):
        # Prosta heurystyka kolejkowania: Najbliższy prezent (Greedy)
        # Można tu wpiąć Algorytm Genetyczny dla lepszych wyników.
        pending_gifts = self.gifts.copy()

        # Sortowanie wstępne: gęstość punktów (score / dist od bazy)
        pending_gifts.sort(key=lambda g: g["score"], reverse=True)

        commands = []

        # Stan Mikołaja
        curr_pos = np.array([0.0, 0.0])
        curr_vel = np.array([0.0, 0.0])
        curr_weight = 10.0  # Waga sań (pusta)
        time_elapsed = 0

        # Pobieramy paczkę -> jedziemy do dziecka -> wracamy po następną
        # (Strategia gwiazdy dla uproszczenia, można optymalizować batching)

        while pending_gifts and time_elapsed < self.T:
            gift = pending_gifts.pop(0)

            # 1. Załaduj marchewki (strategicznie: tyle ile trzeba na podróż + margines)
            needed_carrots = 30
            commands.append(f"LoadCarrots {needed_carrots}")
            curr_weight += needed_carrots

            # 2. Załaduj prezent
            commands.append(f"LoadGift {gift['name']}")  # [cite: 250]
            curr_weight += gift["weight"]

            # 3. Leć do celu (używając Pilota RL)
            target = np.array([gift["c"], gift["r"]])
            flight_cmds, curr_pos, curr_vel, curr_weight, steps = self._fly(
                curr_pos, curr_vel, curr_weight, target
            )
            commands.extend(flight_cmds)
            time_elapsed += steps

            # 4. Dostarcz
            commands.append(f"DeliverGift {gift['name']}")  # [cite: 251]
            curr_weight -= gift["weight"]

            # 5. Wróć do bazy (0,0)
            flight_cmds, curr_pos, curr_vel, curr_weight, steps = self._fly(
                curr_pos, curr_vel, curr_weight, np.array([0, 0])
            )
            commands.extend(flight_cmds)
            time_elapsed += steps

        return len(commands), commands

    def _fly(self, pos, vel, weight, target):
        """Używa modelu RL do generowania trasy."""
        cmds = []
        steps = 0
        sim_pos = pos.copy()
        sim_vel = vel.copy()
        sim_w = weight

        # Pętla symulacji (max 500s na przelot)
        for _ in range(500):
            dist = np.linalg.norm(target - sim_pos)
            if dist <= self.D:
                break

            # Agent widzi wektor DO celu
            rel = target - sim_pos
            action = self.pilot.get_action(
                rel[0], rel[1], sim_vel[0], sim_vel[1], sim_w
            )

            # Wykonaj krok fizyki
            sim_pos, sim_vel, sim_w, acc_vec = self.physics.apply_action(
                sim_pos, sim_vel, sim_w, action
            )
            acc_val = int(np.linalg.norm(acc_vec))

            cmd_str = self.physics.get_action_string(action, acc_val)
            cmds.append(cmd_str)
            steps += 1

        return cmds, sim_pos, sim_vel, sim_w, steps
