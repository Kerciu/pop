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

    def _sort_loaded_gifts(self):
        """Sortuje załadowane prezenty od najbliższego."""
        if not self.state.loaded_gifts:
            return
        current_pos = self.state.position
        self.state.loaded_gifts.sort(
            key=lambda g_name: distance(current_pos, self.gifts_map[g_name].destination)
        )

    def step(self, action_id: int):
        # Domyślna kara za upływ czasu (motywacja do pośpiechu)
        reward = -0.1
        step_penalty = 0.0

        action_enum = self.ACTION_MAPPING[action_id]
        param = 1

        # --- 1. RUCH (Akceleracja) ---
        if action_enum in [
            Action.AccUp,
            Action.AccDown,
            Action.AccLeft,
            Action.AccRight,
        ]:
            max_acc = self.sim.accel_table.get_max_acceleration_for_weight(
                self.state.sleigh_weight
            )

            # KARA: Próba ruchu bez paliwa lub przy przeciążeniu
            if self.state.carrot_count <= 0 or max_acc == 0:
                reward -= 10.0  # Duża kara za próbę ruchu "na pusto"
                # Fizycznie nic się nie dzieje (lub dryfujemy), ale agent musi się nauczyć
                # W symulatorze wywołamy Floating, żeby gra się nie sypała, ale kara jest
                try:
                    self.state = self.sim.apply_action(self.state, Action.Floating, 1)
                except:
                    pass
            else:
                param = max_acc
                # Wykonanie normalnego ruchu
                try:
                    self.state = self.sim.apply_action(self.state, action_enum, param)
                except Exception:
                    # Np. złamanie zasady Acc -> Float -> Acc
                    reward -= 20.0
                    # Musimy "odczekać" karę dryfując
                    try:
                        self.state = self.sim.apply_action(
                            self.state, Action.Floating, 1
                        )
                    except:
                        pass

        # --- 2. TANKOWANIE ---
        elif action_enum == Action.LoadCarrots:
            dist_base = distance(self.state.position, self.sim.lapland_pos)

            # KARA: Tankowanie poza bazą
            if dist_base > self.problem.D:
                reward -= 50.0
                # Czas płynie (zmarnowana tura na próbie tankowania)
                self.state.current_time += 1
            else:
                needed = max(0, 100 - self.state.carrot_count)
                # KARA/NAGRODA: Tankowanie
                if needed > 0:
                    param = needed
                    # Nagroda jest mała, żeby nie farmił punktów tankowaniem
                    reward += 1.0
                    self.state = self.sim.apply_action(self.state, action_enum, param)
                else:
                    # Pełny bak - kara za marnowanie czasu
                    reward -= 10.0
                    self.state.current_time += 1  # Czas płynie

        # --- 3. DOSTARCZANIE ---
        elif action_enum == Action.DeliverGift:
            if not self.state.loaded_gifts:
                # KARA: Próba dostarczenia pustego worka
                reward -= 20.0
                self.state.current_time += 1
            else:
                target = self.gifts_map[self.state.loaded_gifts[0]]
                dist_to_target = distance(self.state.position, target.destination)

                if dist_to_target > self.problem.D:
                    # KARA: Próba zrzutu za daleko od celu
                    reward -= 20.0
                    self.state.current_time += 1
                else:
                    # SUKCES: Dostarczamy
                    param = 0
                    prev_delivered = len(self.state.delivered_gifts)
                    self.state = self.sim.apply_action(self.state, action_enum, param)

                    # Sortujemy resztę, żeby nowy cel był [0]
                    self._sort_loaded_gifts()

                    # Sprawdzamy czy się udało (dla pewności)
                    if len(self.state.delivered_gifts) > prev_delivered:
                        reward += 5000.0  # WIELKA NAGRODA

        # --- 4. ŁADOWANIE PREZENTÓW ---
        elif action_enum == Action.LoadGifts:
            dist_base = distance(self.state.position, self.sim.lapland_pos)

            # KARA: Ładowanie poza bazą
            if dist_base > self.problem.D:
                reward -= 50.0
                self.state.current_time += 1

            # KARA: Ładowanie, gdy już mamy prezenty (w tym uproszczonym modelu lecimy full -> empty -> full)
            # To zapobiega pętli ładowania w nieskończoność
            elif self.state.loaded_gifts:
                reward -= 50.0  # Bardzo bolesna kara za pętlę!
                self.state.current_time += 1

            # KARA: Brak prezentów do wzięcia
            elif not self.state.available_gifts:
                reward -= 10.0
                self.state.current_time += 1

            else:
                # SUKCES: Próba załadunku
                real_max_weight = self.sim.accel_table.ranges[-1].max_weight_inclusive
                gifts_to_load_ids = solve_knapsack_greedy(
                    self.state.available_gifts,
                    self.gifts_map,
                    real_max_weight,
                    self.state.sleigh_weight,
                )

                if not gifts_to_load_ids:
                    # Nic się nie zmieściło (mało prawdopodobne przy pustych saniach)
                    reward -= 10.0
                    self.state.current_time += 1
                else:
                    # Ładujemy manualnie (bez upływu czasu)
                    try:
                        loaded_count = 0
                        loaded_set = set(gifts_to_load_ids)
                        new_available = []
                        for g_name in self.state.available_gifts:
                            if g_name in loaded_set:
                                gift = self.gifts_map[g_name]
                                self.state.loaded_gifts.append(g_name)
                                self.state.sleigh_weight += gift.weight
                                loaded_count += 1
                            else:
                                new_available.append(g_name)
                        self.state.available_gifts = new_available

                        self._sort_loaded_gifts()
                        reward += 10.0 + loaded_count * 0.5  # Nagroda za załadunek
                    except:
                        reward -= 50.0  # Krytyczny błąd

        # --- 5. FLOATING (Dryfowanie) ---
        elif action_enum == Action.Floating:
            self.state = self.sim.apply_action(self.state, action_enum, 1)
            # Mała kara za stratę czasu, chyba że czekamy na coś sensownego
            reward -= 0.1

        # --- FINALIZE STEP (Wspólne obliczenia) ---
        return self._finalize_step_logic(reward)

    def _finalize_step_logic(self, reward):
        target_pos = self._get_target_pos()
        curr_dist = distance(self.state.position, target_pos)

        # Prędkość
        velocity_mag = (self.state.velocity.vc**2 + self.state.velocity.vr**2) ** 0.5

        # Nagroda za zbliżanie się (tylko przy rozsądnej prędkości)
        # Aby zachęcić do latania w dobrą stronę, ale nie premiować pędu
        dist_improvement = self.prev_dist - curr_dist
        if abs(dist_improvement) < 1000 and velocity_mag < 80:
            reward += dist_improvement * 0.2

        self.prev_dist = curr_dist

        # Warunki końca gry
        done = False

        # 1. Koniec czasu
        if self.state.current_time >= self.problem.T:
            done = True

        # 2. Wszystko dostarczone
        if len(self.state.delivered_gifts) == len(self.problem.gifts):
            done = True
            reward += 100000.0  # Jackpot

        # 3. Wylot poza mapę
        if (
            abs(self.state.position.c) > self.map_limit
            or abs(self.state.position.r) > self.map_limit
        ):
            done = True
            reward -= 200.0  # Śmierć

        return self.encoder.encode(self.state), reward, done, {}
