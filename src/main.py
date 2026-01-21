import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from agents.dqn_agent import DQNAgent
from core.loader import load_problem
from env.sleigh_env import SleighEnv
from output.output_writer import OutputWriter  # Upewnij się, że masz ten plik
from visualizer import Visualizer

# MODEL_PATH = "models_saved/santa_fuel_fixed.pth"
# INPUT_FILE = "data/huge_challenge.in.txt"

MODEL_PATH = "models_saved/santa_mini_test.pth"
INPUT_FILE = "data/mini_challenge.in.txt"


def run_training(env, agent, args):
    print(f"--- START TRENINGU DUELING DQN ({args.episodes} odcinków) ---")

    if not os.path.exists("models_saved"):
        os.makedirs("models_saved")

    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05

    best_avg_reward = -float("inf")
    recent_rewards = []

    for e in range(1, args.episodes + 1):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done:
            action = agent.get_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            total_reward += reward
            steps += 1

            if steps > 2500:
                break

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        recent_rewards.append(total_reward)
        if len(recent_rewards) > 50:
            recent_rewards.pop(0)

        avg_reward = sum(recent_rewards) / len(recent_rewards)

        if e % 10 == 0:
            print(
                f"Ep {e:4d} | Avg Score: {avg_reward:8.2f} | Epsilon: {epsilon:.2f} | Steps: {steps} | Deliv: {len(env.state.delivered_gifts)}"
            )
            agent.update_target_network()

        if avg_reward > best_avg_reward and e > 100:
            best_avg_reward = avg_reward
            agent.save(MODEL_PATH)
            print(f"💾 Zapisano model! Nowy rekord średniej: {best_avg_reward:.2f}")


def run_evaluation(env, agent, args):
    print("--- START EWALUACJI ---")
    if os.path.exists(MODEL_PATH):
        try:
            agent.load(MODEL_PATH)
            print(f"✅ Wczytano model: {MODEL_PATH}")
        except:
            print("❌ Błąd ładowania modelu. Używam losowych wag.")

    agent.policy_net.eval()
    writer = OutputWriter()
    viz = Visualizer(env.problem) if args.render else None

    state = env.reset()
    done = False
    total_reward = 0
    step = 0

    action_names = [
        "ACC_N",
        "ACC_S",
        "ACC_E",
        "ACC_W",
        "MAX_N",
        "MAX_S",
        "MAX_E",
        "MAX_W",
        "FLOAT",
        "LOAD",
        "FUEL",
        "DELIVER",
    ]

    while not done:
        # 1. Pobieramy dane symulacji PRZED krokiem (z obiektu env.state, nie z tensora)
        carrots_before = env.state.carrot_count
        loaded_before = set(env.state.loaded_gifts)
        delivered_before = set(env.state.delivered_gifts)

        # Zapisujemy prędkość, aby obliczyć przyspieszenie
        vc_before = env.state.velocity.vc
        vr_before = env.state.velocity.vr

        # 2. Wykonanie akcji
        action = agent.get_action(state, epsilon=0.0)
        next_state_tensor, reward, done, _ = env.step(action)  # next_state to tensor!
        action_name = action_names[action]

        # 3. LOGIKA ZAPISU (Porównujemy env.state po kroku z zapisanymi before)

        # A. Ruch
        if action < 8:
            # Porównujemy aktualną prędkość z poprzednią
            dv_c = abs(env.state.velocity.vc - vc_before)
            dv_r = abs(env.state.velocity.vr - vr_before)
            acc_val = max(dv_c, dv_r)

            writer.record_move(action_name, int(acc_val))
            writer.record_move("FLOAT", 1)

        # B. Float
        elif action == 8:
            writer.record_move("FLOAT", 1)

        # C. Load Gifts
        elif action == 9:
            current_loaded = set(env.state.loaded_gifts)
            new_gifts = current_loaded - loaded_before
            for gift_id in new_gifts:
                writer.record_load_gift(gift_id)

        # D. Fuel
        elif action == 10:
            diff = env.state.carrot_count - carrots_before
            if diff > 0:
                writer.record_load_carrots(diff)

        # E. Deliver
        elif action == 11:
            current_delivered = set(env.state.delivered_gifts)
            new_delivered = current_delivered - delivered_before
            for gift_id in new_delivered:
                writer.record_deliver_gift(gift_id)

        # 4. Wizualizacja
        pos = env.state.position
        if viz:
            viz.render(env, action_name, reward, step)

        if action >= 8 or step % 50 == 0:
            print(
                f"Step {step:4d} | {action_name:10} | Pos: {pos.c:4.0f},{pos.r:4.0f} | R: {reward:6.1f} | Gifts: {len(env.state.loaded_gifts)}"
            )

        state = next_state_tensor
        total_reward += reward
        step += 1

        if step > 2500:
            print("❌ Przekroczono limit kroków.")
            break

    print(
        f"Koniec. Wynik: {total_reward:.2f}. Dostarczono: {len(env.state.delivered_gifts)}"
    )
    writer.save("solution.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "eval"])
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    problem, simulator = load_problem(INPUT_FILE)
    env = SleighEnv(problem, simulator)
    agent = DQNAgent(env.input_size, env.ACTION_SPACE_SIZE)

    if args.mode == "train":
        run_training(env, agent, args)
    else:
        run_evaluation(env, agent, args)
