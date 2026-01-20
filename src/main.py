import argparse
import math
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from agents.dqn_agent import DQNAgent
from core.distance_utils import distance
from core.loader import load_problem
from env.sleigh_env import SleighEnv
from visualizer import Visualizer

# --- KONFIGURACJA ---
DEFAULT_INPUT = "data/huge_challenge.in.txt"
MODEL_PATH = "models_saved/dqn_santa_huge.pth"
# DEFAULT_INPUT = "data/b_better_hurry.in.txt"
# MODEL_PATH = "models_saved/dqn_santa_pure.pth"


def get_autopilot_action(env, target_pos):
    """
    Czysty algorytm matematyczny (do porównania lub debugowania).
    """
    s = env.state
    if s.last_action_was_acceleration:
        return 4

    dx = target_pos.c - s.position.c
    dy = target_pos.r - s.position.r
    dist = math.sqrt(dx**2 + dy**2)
    vx = s.velocity.vc
    vy = s.velocity.vr
    speed = math.sqrt(vx**2 + vy**2)

    if dist <= env.problem.D:
        return 4  # Czekaj na strzał

    # Fizyka hamowania
    max_acc = env.sim.accel_table.get_max_acceleration_for_weight(s.sleigh_weight)
    if max_acc == 0:
        return 4

    effective_acc = max_acc / 2.0
    braking_dist = (speed**2) / (2 * effective_acc) if effective_acc > 0 else 999999

    if dist < braking_dist + 300:
        if abs(vx) > abs(vy):
            return 2 if vx > 0 else 3
        else:
            return 1 if vy > 0 else 0

    speed_limit = 60 if s.sleigh_weight > 2000 else 150
    if speed < speed_limit:
        if abs(dx) > abs(dy):
            return 3 if dx > 0 else 2
        else:
            return 0 if dy > 0 else 1

    return 4


def get_action_for_context(env, agent, epsilon, use_autopilot):
    """
    Wybiera akcję.
    W trybie TRENINGU: Używa Hybrid Logic (sztywne reguły) LUB sieci.
    W trybie AUTOPILOT: Używa tylko matematyki.
    """
    if use_autopilot:
        # 1. Logika Autopilota (Matematyczna)
        dist_to_base = distance(env.state.position, env.sim.lapland_pos)

        # Baza: Ładuj/Tankuj
        if dist_to_base <= env.problem.D:
            if not env.state.loaded_gifts and env.state.available_gifts:
                return 6, "AUTO_LOAD"
            if env.state.carrot_count < 20:
                return 5, "AUTO_FUEL"

        # Trasa
        if env.state.loaded_gifts:
            target = env.gifts_map[env.state.loaded_gifts[0]]
            if distance(env.state.position, target.destination) <= env.problem.D:
                return 7, "AUTO_DELIV"
            return get_autopilot_action(env, target.destination), "AUTO_NAV"
        else:
            return get_autopilot_action(env, env.sim.lapland_pos), "AUTO_HOME"

    else:
        # 2. Logika Treningowa (Hybrid: Sztywne ramy + Sieć)
        # Zostawiamy minimalną pomoc, żeby agent w ogóle wiedział co robić w kluczowych punktach
        # ale usuwamy "niańczenie" pętli. Niech uczy się na karach.

        state = env.state
        dist_to_base = distance(state.position, env.sim.lapland_pos)

        # A. Jeśli jesteśmy w bazie i pusto -> Sugestia: Ładuj (ale sieć może nadpisać epsilonem)
        # UWAGA: Tu pozwalamy sieci działać, ale dla przyspieszenia nauki
        # wymuszamy te krytyczne momenty, bo inaczej uczenie trwa wieki.
        if (
            dist_to_base <= env.problem.D
            and not state.loaded_gifts
            and state.available_gifts
        ):
            return 6, "RULE_LOAD"

        # B. Jeśli jesteśmy idealnie u celu -> Sugestia: Oddaj
        if state.loaded_gifts:
            target = env.gifts_map[state.loaded_gifts[0]]
            if distance(state.position, target.destination) <= env.problem.D:
                return 7, "RULE_DELIV"

        # C. Reszta (99% czasu) -> SIEĆ NEURONOWA
        # Sieć decyduje jak latać, kiedy tankować (poza bazą i tak nie zatankuje)
        state_tensor = env.encoder.encode(state).unsqueeze(0)
        action = agent.get_action(state_tensor, epsilon)
        return action, "AI_NET"


def run_training(env, agent, args):
    print(f"--- START TRENINGU ({args.episodes} epizodów) ---")
    save_dir = os.path.dirname(MODEL_PATH)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.9995
    best_score = -float("inf")

    for e in range(args.episodes):
        env.reset()
        if env.state.available_gifts:
            env._sort_loaded_gifts()

        state_tensor = env.encoder.encode(env.state).unsqueeze(0)
        done = False
        total_reward = 0

        while not done:
            action_id, _ = get_action_for_context(
                env, agent, epsilon, use_autopilot=False
            )

            next_state_tensor, reward, done, _ = env.step(action_id)
            next_state_tensor = next_state_tensor.unsqueeze(0)

            agent.update(state_tensor, action_id, reward, next_state_tensor, done)
            state_tensor = next_state_tensor
            total_reward += reward

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if total_reward > best_score:
            best_score = total_reward
            agent.save(MODEL_PATH)
            print(f"🚀 NOWY REKORD: {best_score:.2f} (Epizod {e})")

        if e % 50 == 0:
            print(
                f"Ep {e} | Score: {total_reward:.2f} | Best: {best_score:.2f} | Eps: {epsilon:.2f}"
            )
            agent.update_target_network()


def run_evaluation(env, agent, args):
    print("--- START EWALUACJI ---")
    viz = None
    if args.render:
        viz = Visualizer(env.problem)

    if os.path.exists(MODEL_PATH):
        try:
            agent.load(MODEL_PATH)
            print(f"✅ Wczytano model: {MODEL_PATH}")
        except:
            print("❌ Błąd modelu")

    env.reset()
    if env.state.available_gifts:
        env._sort_loaded_gifts()

    done = False
    total_reward = 0
    step = 0

    while not done:
        action_id, source = get_action_for_context(
            env, agent, epsilon=0.0, use_autopilot=args.autopilot
        )

        _, reward, done, _ = env.step(action_id)
        action_enum = env.ACTION_MAPPING[action_id]

        if viz:
            viz.render(env, f"{action_enum.name} ({source})", reward, step)

        if step % 50 == 0 or action_id in [5, 6, 7] or done:
            pos = env.state.position
            print(
                f"Step {step:4d} | [{source:10}] {action_enum.name:13} | "
                f"Pos: {pos.c:5.0f},{pos.r:5.0f} | "
                f"Gifts: {len(env.state.loaded_gifts):3} | "
                f"Deliv: {len(env.state.delivered_gifts):3} | "
                f"Time: {env.state.current_time}/{env.problem.T}"
            )

        total_reward += reward
        step += 1

    print(f"\nKONIEC. Wynik: {total_reward:.2f}")
    print(f"Dostarczono: {len(env.state.delivered_gifts)} / {len(env.problem.gifts)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "eval"])
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--autopilot", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(DEFAULT_INPUT):
        exit(1)
    problem, simulator = load_problem(DEFAULT_INPUT)
    env = SleighEnv(problem, simulator)
    state_size = env.encoder.output_size
    action_space_size = env.action_space_size
    agent = DQNAgent(state_size, action_space_size)

    if args.mode == "train":
        run_training(env, agent, args)
    else:
        run_evaluation(env, agent, args)
