import math
import os

from agents.dqn_agent import DQNAgent
from core.actions import Action
from core.distance_utils import distance
from core.loader import load_problem
from env.sleigh_env import SleighEnv

INPUT_DATA_PATH = "data/b_better_hurry.in.txt"
MODEL_PATH = "models_saved/dqn_santa_best.pth"


def get_navigation_action(env, target_pos):
    """
    Pomocniczy algorytm nawigacji (Autopilot).
    Zwraca ID akcji (lub None), aby bezpiecznie dolecieć do celu.
    """
    s = env.state

    # Wektor do celu
    dx = target_pos.c - s.position.c
    dy = target_pos.r - s.position.r
    dist = math.sqrt(dx**2 + dy**2)

    # Aktualna prędkość
    vx = s.velocity.vc
    vy = s.velocity.vr
    speed = math.sqrt(vx**2 + vy**2)

    # 1. Jeśli jesteśmy bardzo blisko i mamy małą prędkość -> Czekamy na Deliver
    if dist <= env.problem.D:
        return 4  # Floating (czekamy na logikę Deliver)

    # 2. Fizyka hamowania: Droga hamowania = v^2 / (2*a).
    # Przyjmijmy a ok. 10 (zależy od wagi, ale bezpieczniej założyć mniej).
    # Jeśli dist < speed^2 / 10, to musimy hamować NATYCHMIAST.
    braking_threshold = (speed**2) / 10.0

    # Margines bezpieczeństwa (zwalniamy wcześniej)
    if dist < braking_threshold + 500:
        # Logika hamowania (kontrowanie prędkości)
        if abs(vx) > abs(vy):
            return (
                2 if vx > 0 else 3
            )  # AccLeft jeśli lecimy w prawo, AccRight jeśli w lewo
        else:
            return 0 if vy > 0 else 1  # AccUp jeśli lecimy w dół, AccDown jeśli w górę

    # 3. Jeśli jesteśmy daleko, a prędkość jest mała -> Przyspieszamy w stronę celu
    if speed < 50:  # Limit prędkości przelotowej
        if abs(dx) > abs(dy):
            return 3 if dx > 0 else 2  # AccRight / AccLeft
        else:
            return 1 if dy > 0 else 0  # AccDown / AccUp

    # 4. Jeśli lecimy szybko i mniej więcej w dobrą stronę -> Dryfujemy
    return 4  # Floating


def main():
    if not os.path.exists(MODEL_PATH):
        print("Brak modelu! Uruchom najpierw train.py")
        return

    print("Wczytywanie problemu...")
    problem, simulator = load_problem(INPUT_DATA_PATH)
    env = SleighEnv(problem, simulator)

    # Sortowanie na starcie (żeby pierwszy cel był blisko)
    env.reset()
    # Hack: wywołujemy sortowanie ręcznie, bo w evaluate nie ma pętli treningowej
    if env.state.available_gifts:
        env._sort_loaded_gifts()

    agent = DQNAgent(env.encoder.output_size, env.action_space_size)

    try:
        agent.load(MODEL_PATH)
        print(f"Wczytano model z {MODEL_PATH}")
    except:
        print("Błąd wczytywania modelu.")
        # W evaluate możemy lecieć nawet bez modelu, jeśli mamy dobrą logikę nawigacji!

    state = env.reset()
    state = state.unsqueeze(0)
    done = False
    total_reward = 0
    step = 0
    last_action_was_load = False

    print("\n--- START EWALUACJI Z AUTOPILOTEM ---")

    while not done:
        # --- HYBRID LOGIC + AUTOPILOT ---
        forced_action = None
        current_state_obj = env.state
        dist_to_base = distance(current_state_obj.position, simulator.lapland_pos)

        # A. Ładowanie (Baza)
        if (
            dist_to_base <= problem.D
            and not current_state_obj.loaded_gifts
            and current_state_obj.available_gifts
            and not last_action_was_load
        ):
            forced_action = 6  # LoadGifts

        # B. Tankowanie (Baza)
        elif dist_to_base <= problem.D and current_state_obj.carrot_count < 50:
            forced_action = 5  # LoadCarrots

        # C. Dostarczanie i Nawigacja (W terenie)
        elif current_state_obj.loaded_gifts:
            # Cel to zawsze pierwszy załadowany (bo są posortowane przez env)
            target_name = current_state_obj.loaded_gifts[0]
            target_gift = env.gifts_map[target_name]
            dist_to_target = distance(
                current_state_obj.position, target_gift.destination
            )

            # 1. Jesteśmy w zasięgu -> Oddaj
            if dist_to_target <= problem.D:
                forced_action = 7  # DeliverGift
            else:
                # 2. Jesteśmy w trasie -> Użyj Autopilota zamiast sieci
                forced_action = get_navigation_action(env, target_gift.destination)

        # D. Powrót do bazy (gdy pusto)
        elif (
            not current_state_obj.loaded_gifts and not current_state_obj.available_gifts
        ):
            # Koniec gry, czekamy
            forced_action = 4
        elif not current_state_obj.loaded_gifts:
            # Wracamy do bazy po nową partię
            forced_action = get_navigation_action(env, simulator.lapland_pos)

        # Wybór akcji
        if forced_action is not None:
            action_id = forced_action
            source = "LOGIC"
        else:
            # Fallback do sieci (rzadko używane przy pełnym autopilocie)
            action_id = agent.get_action(state, epsilon=0.0)
            source = "AI_NET"

        # Wykonanie
        next_state, reward, done, _ = env.step(action_id)

        # Logika flagi
        if action_id == 6 and not env.state.loaded_gifts:
            last_action_was_load = True
        else:
            last_action_was_load = False

        # Logowanie (tylko co 100 kroków lub przy ważnych akcjach)
        action_enum = env.ACTION_MAPPING[action_id]
        if (
            action_enum in [Action.LoadGifts, Action.DeliverGift, Action.LoadCarrots]
            or step % 100 == 0
        ):
            pos = env.state.position
            print(
                f"Step {step:4d} | [{source:7}] {action_enum.name:13} | "
                f"Pos: {pos.c:6.0f},{pos.r:6.0f} | "
                f"Gifts: {len(env.state.loaded_gifts):3} | "
                f"Delivered: {len(env.state.delivered_gifts):3} | "
                f"Reward: {reward:7.2f}"
            )

        state = next_state.unsqueeze(0)
        total_reward += reward
        step += 1

    print(f"\nKONIEC. Wynik: {total_reward:.2f}")
    print(f"Dostarczono: {len(env.state.delivered_gifts)} / {len(problem.gifts)}")


if __name__ == "__main__":
    main()
