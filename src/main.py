import argparse
import math
import os
import sys

# Dodajemy katalog src do ścieżki, żeby importy działały
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from agents.dqn_agent import DQNAgent
from core.distance_utils import distance
from core.loader import load_problem
from env.sleigh_env import SleighEnv

# --- KONFIGURACJA ---
DEFAULT_INPUT = "data/b_better_hurry.in.txt"
MODEL_PATH = "models_saved/dqn_santa_unified.pth"


def get_autopilot_action(env, target_pos):
    """
    Inteligentny Autopilot z fizyką hamowania (Smart Braking).
    """
    s = env.state

    # 1. Fizyka: Odczekanie po przyspieszeniu
    if s.last_action_was_acceleration:
        return 4  # Floating

    dx = target_pos.c - s.position.c
    dy = target_pos.r - s.position.r
    dist = math.sqrt(dx**2 + dy**2)

    vx = s.velocity.vc
    vy = s.velocity.vr
    speed = math.sqrt(vx**2 + vy**2)

    # 1. Jesteśmy na miejscu -> Czekaj na Deliver
    if dist <= env.problem.D:
        return 4

    # 2. Fizyka hamowania
    max_acc = env.sim.accel_table.get_max_acceleration_for_weight(s.sleigh_weight)
    if max_acc == 0:
        return 4  # Przeciążenie

    effective_acc = max_acc / 2.0
    braking_distance = (speed**2) / (2 * effective_acc) if effective_acc > 0 else 99999

    # Margines bezpieczeństwa
    threshold = braking_distance + 200

    # Logika Hamowania
    if dist < threshold:
        if abs(vx) > abs(vy):
            return 2 if vx > 0 else 3
        else:
            return 1 if vy > 0 else 0

    # 3. Logika Rozpędzania
    speed_limit = 50 if s.sleigh_weight > 1000 else 150
    if speed < speed_limit:
        if abs(dx) > abs(dy):
            return 3 if dx > 0 else 2
        else:
            return 0 if dy > 0 else 1

    return 4  # Dryfuj


def get_hybrid_action(
    env, agent, epsilon, use_autopilot=False, last_action_was_load=False
):
    """
    Wspólna logika decyzyjna dla treningu i ewaluacji.
    Zwraca: (action_id, source_string)
    """
    problem = env.problem
    sim = env.sim
    state = env.state

    dist_to_base = distance(state.position, sim.lapland_pos)

    # --- 1. LOGIKA BAZOWA (Sztywna) ---

    # A. Ładowanie
    if (
        dist_to_base <= problem.D
        and not state.loaded_gifts
        and state.available_gifts
        and not last_action_was_load
    ):
        return 6, "LOGIC_LOAD"  # LoadGifts

    # B. Tankowanie (jeśli mało paliwa i jesteśmy w bazie)
    if dist_to_base <= problem.D and state.carrot_count < 20:
        return 5, "LOGIC_FUEL"  # LoadCarrots

    # --- 2. LOGIKA TERENOWA (Dostarczanie) ---

    if state.loaded_gifts:
        # Dzięki sortowaniu w Environment, [0] to zawsze najbliższy cel
        target_name = state.loaded_gifts[0]
        target_gift = env.gifts_map[target_name]
        dist_target = distance(state.position, target_gift.destination)

        # C. Dostarczanie (jeśli w zasięgu)
        if dist_target <= problem.D:
            return 7, "LOGIC_DELIV"  # DeliverGift

        # D. Nawigacja do celu (Autopilot lub Sieć)
        if use_autopilot:
            return get_autopilot_action(env, target_gift.destination), "AUTO_NAV"

    # --- 3. LOGIKA POWROTU ---

    elif not state.loaded_gifts:
        # Wracamy do bazy
        if use_autopilot:
            return get_autopilot_action(env, sim.lapland_pos), "AUTO_HOME"

    # --- 4. SIEĆ NEURONOWA (DQN) ---
    # Jeśli żadna sztywna reguła nie zadziałała (lub autopilot wyłączony)
    action = agent.get_action(env.encoder.encode(state).unsqueeze(0), epsilon)
    return action, "AI_NET"


def run_training(env, agent, args):
    print(f"--- START TRENINGU ({args.episodes} epizodów) ---")

    save_dir = os.path.dirname(MODEL_PATH)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.999
    best_score = -float("inf")

    for e in range(args.episodes):
        env.reset()
        # Sortowanie na starcie
        if env.state.available_gifts:
            env._sort_loaded_gifts()

        state_tensor = env.encoder.encode(env.state).unsqueeze(0)
        done = False
        total_reward = 0
        last_action_was_load = False

        while not done:
            # W treningu rzadko używamy autopilota, żeby sieć się uczyła,
            # chyba że chcemy "Behavior Cloning", ale tu uczymy od zera.
            action_id, _ = get_hybrid_action(
                env,
                agent,
                epsilon,
                use_autopilot=False,  # Sieć ma się uczyć latać!
                last_action_was_load=last_action_was_load,
            )

            next_state_tensor, reward, done, _ = env.step(action_id)
            next_state_tensor = next_state_tensor.unsqueeze(0)

            # Aktualizacja flagi
            last_action_was_load = action_id == 6 and not env.state.loaded_gifts

            # Zapis do pamięci i uczenie
            agent.update(state_tensor, action_id, reward, next_state_tensor, done)

            state_tensor = next_state_tensor
            total_reward += reward

        # Po epizodzie
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if total_reward > best_score:
            best_score = total_reward
            agent.save(MODEL_PATH)
            print(f"🚀 NOWY REKORD: {best_score:.2f} (Epizod {e})")

        if e % 10 == 0:
            print(
                f"Ep {e} | Score: {total_reward:.2f} | Best: {best_score:.2f} | Eps: {epsilon:.2f}"
            )
            agent.update_target_network()


def run_evaluation(env, agent, args):
    print("--- START EWALUACJI ---")
    if not os.path.exists(MODEL_PATH):
        print("⚠️  UWAGA: Brak zapisanego modelu. Używam losowych wag (lub autopilota).")
    else:
        try:
            agent.load(MODEL_PATH)
            print(f"✅ Wczytano model: {MODEL_PATH}")
        except:
            print("❌ Błąd wczytywania modelu.")

    env.reset()
    if env.state.available_gifts:
        env._sort_loaded_gifts()

    done = False
    total_reward = 0
    step = 0
    last_action_was_load = False

    while not done:
        # W ewaluacji chcemy najlepszy możliwy wynik.
        # Jeśli flaga --autopilot jest włączona, używamy algorytmu nawigacji.
        action_id, source = get_hybrid_action(
            env,
            agent,
            epsilon=0.0,
            use_autopilot=args.autopilot,  # Tu decydujemy czy sieć czy matematyka
            last_action_was_load=last_action_was_load,
        )

        next_state_tensor, reward, done, _ = env.step(action_id)

        last_action_was_load = action_id == 6 and not env.state.loaded_gifts

        # Logowanie
        action_enum = env.ACTION_MAPPING[action_id]
        if step % 50 == 0 or action_id in [5, 6, 7] or done:
            pos = env.state.position
            print(
                f"Step {step:4d} | [{source:11}] {action_enum.name:13} | "
                f"Pos: {pos.c:6.0f},{pos.r:6.0f} | "
                f"Gifts: {len(env.state.loaded_gifts):3} | "
                f"Deliv: {len(env.state.delivered_gifts):3} | "
                f"Time: {env.state.current_time}"
            )

        total_reward += reward
        step += 1

    print(f"\nKONIEC. Wynik: {total_reward:.2f}")
    print(f"Dostarczono: {len(env.state.delivered_gifts)} / {len(env.problem.gifts)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Santa Sleigh RL Controller")
    parser.add_argument("mode", choices=["train", "eval"], help="Tryb działania")
    parser.add_argument(
        "--episodes", type=int, default=5000, help="Liczba epizodów treningowych"
    )
    parser.add_argument(
        "--autopilot",
        action="store_true",
        help="Włącz autopilota w trybie eval (omija sieć)",
    )

    args = parser.parse_args()

    # Ładowanie środowiska
    if not os.path.exists(DEFAULT_INPUT):
        print(f"Błąd: Brak pliku {DEFAULT_INPUT}")
        exit(1)

    problem, simulator = load_problem(DEFAULT_INPUT)
    env = SleighEnv(problem, simulator)

    # Inicjalizacja agenta
    state_size = env.encoder.output_size
    action_size = env.action_space_size
    agent = DQNAgent(state_size, action_size)

    if args.mode == "train":
        run_training(env, agent, args)
    else:
        run_evaluation(env, agent, args)
