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
    Inteligentny Autopilot uwzględniający masę sań.
    """
    s = env.state

    # 1. Odczekanie po przyspieszeniu (wymóg fizyki)
    if s.last_action_was_acceleration:
        return 4  # Floating

    # Wektor i dystans do celu
    dx = target_pos.c - s.position.c
    dy = target_pos.r - s.position.r
    dist = math.sqrt(dx**2 + dy**2)

    # Prędkość
    vx = s.velocity.vc
    vy = s.velocity.vr
    speed = math.sqrt(vx**2 + vy**2)

    # 1. Jesteśmy na miejscu -> Czekaj na Deliver
    if dist <= env.problem.D:
        return 4

    # --- KLUCZOWA ZMIANA: FIZYKA HAMOWANIA ---
    # Pobieramy aktualne możliwości przyspieszenia dla obecnej wagi
    max_acc = env.sim.accel_table.get_max_acceleration_for_weight(s.sleigh_weight)

    # Jeśli jesteśmy przeciążeni (max_acc=0), nic nie zrobimy, dryfujemy
    if max_acc == 0:
        return 4

    # Droga hamowania = v^2 / (2*a).
    # Uwzględniamy, że hamowanie to cykl Acc+Float (efektywne a = max_acc / 2)
    effective_acc = max_acc / 2.0
    braking_distance = (speed**2) / (2 * effective_acc)

    # Margines bezpieczeństwa (np. 10% + stała wartość na manewry)
    safety_margin = braking_distance * 0.1 + 200
    threshold = braking_distance + safety_margin

    # Debug (opcjonalny, odkomentuj jeśli chcesz widzieć fizykę w akcji)
    # if env.state.current_time % 10 == 0:
    #    print(f"Dist: {dist:.0f} | Speed: {speed:.1f} | BrakeDist: {braking_distance:.0f} | MaxAcc: {max_acc}")

    # 2. Logika Hamowania
    if dist < threshold:
        # Musimy zbić prędkość w osi, w której poruszamy się najszybciej
        if abs(vx) > abs(vy):
            # Kontra pozioma: v > 0 (lewo), v < 0 (prawo)
            return 2 if vx > 0 else 3
        else:
            # Kontra pionowa: v > 0 (dół), v < 0 (góra) (w systemie gry: AccDown zmniejsza vr)
            return 1 if vy > 0 else 0

    # 3. Logika Rozpędzania
    # Limit prędkości dostosowany do wagi - im ciężej, tym wolniej latamy, żeby wyrobić na zakrętach
    speed_limit = 50 if s.sleigh_weight > 1000 else 150

    if speed < speed_limit:
        if abs(dx) > abs(dy):
            return 3 if dx > 0 else 2  # Prawo / Lewo
        else:
            return (
                0 if dy > 0 else 1
            )  # Góra / Dół (zgodnie z logiką: cel wyżej -> AccUp)

    return 4  # Dryfuj (utrzymuj prędkość)


def main():
    if not os.path.exists(MODEL_PATH):
        print("Brak modelu (używamy autopilota).")

    print("Wczytywanie problemu...")
    problem, simulator = load_problem(INPUT_DATA_PATH)
    env = SleighEnv(problem, simulator)

    # Sortowanie początkowe
    env.reset()
    if env.state.available_gifts:
        env._sort_loaded_gifts()

    agent = DQNAgent(env.encoder.output_size, env.action_space_size)
    try:
        agent.load(MODEL_PATH)
        print("Model wczytany.")
    except:
        pass

    state = env.reset()
    state = state.unsqueeze(0)
    done = False
    total_reward = 0
    step = 0
    last_action_was_load = False

    print("\n--- START EWALUACJI ---")

    while not done:
        forced_action = None
        current_state_obj = env.state
        dist_to_base = distance(current_state_obj.position, simulator.lapland_pos)

        # A. Baza: Ładuj
        if (
            dist_to_base <= problem.D
            and not current_state_obj.loaded_gifts
            and current_state_obj.available_gifts
            and not last_action_was_load
        ):
            forced_action = 6

        # B. Baza: Tankuj
        elif dist_to_base <= problem.D and current_state_obj.carrot_count < 20:
            forced_action = 5

        # C. W terenie: Dostarczaj i Leć
        elif current_state_obj.loaded_gifts:
            target_name = current_state_obj.loaded_gifts[0]
            target_gift = env.gifts_map[target_name]
            dist_target = distance(current_state_obj.position, target_gift.destination)

            if dist_target <= problem.D:
                forced_action = 7  # Deliver
            else:
                forced_action = get_navigation_action(
                    env, target_gift.destination
                )  # Autopilot

        # D. Pusto: Wracaj
        elif not current_state_obj.loaded_gifts:
            forced_action = get_navigation_action(env, simulator.lapland_pos)

        if forced_action is not None:
            action_id = forced_action
            source = "AUTO"
        else:
            action_id = agent.get_action(state, epsilon=0.0)
            source = "NET"

        next_state, reward, done, _ = env.step(action_id)

        # Logika flagi
        if action_id == 6 and not env.state.loaded_gifts:
            last_action_was_load = True
        else:
            last_action_was_load = False

        action_enum = env.ACTION_MAPPING[action_id]

        # Logowanie
        if (
            action_enum in [Action.LoadGifts, Action.DeliverGift]
            or step % 50 == 0
            or done
        ):
            pos = env.state.position
            print(
                f"Step {step:4d} | [{source}] {action_enum.name:13} | "
                f"Pos: {pos.c:6.0f},{pos.r:6.0f} | "
                f"Gifts: {len(env.state.loaded_gifts):3} | "
                f"Deliv: {len(env.state.delivered_gifts):3} | "
                f"Reward: {reward:7.2f} | Time: {env.state.current_time}"
            )

        state = next_state.unsqueeze(0)
        total_reward += reward
        step += 1

    print(f"\nKONIEC. Wynik: {total_reward:.2f}")
    print(f"Dostarczono: {len(env.state.delivered_gifts)} / {len(problem.gifts)}")


if __name__ == "__main__":
    main()
