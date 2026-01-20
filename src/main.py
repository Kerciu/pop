import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from agents.dqn_agent import DQNAgent
from core.loader import load_problem
from env.sleigh_env import SleighEnv
from visualizer import Visualizer

MODEL_PATH = "models_saved/santa_dueling_smart.pth"
INPUT_FILE = "data/huge_challenge.in.txt"  # Upewnij się, że masz ten plik


def run_training(env, agent, args):
    print(f"--- START TRENINGU DUELING DQN ({args.episodes} odcinków) ---")

    if not os.path.exists("models_saved"):
        os.makedirs("models_saved")

    epsilon = 1.0
    epsilon_decay = 0.997  # Wolniejszy decay, niech więcej eksploruje
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

            # Safety break jeśli kręci się w kółko
            if steps > 2500:
                break

        # Logika po epizodzie
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
    agent.load(MODEL_PATH)
    agent.policy_net.eval()  # Tryb ewaluacji (wyłącza dropout itp jeśli są)

    viz = Visualizer(env.problem) if args.render else None

    state = env.reset()
    done = False
    total_reward = 0
    step = 0

    # Mapping nazw akcji do wyświetlania
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
        action = agent.get_action(state, epsilon=0.0)  # Greedy
        next_state, reward, done, _ = env.step(action)

        action_name = action_names[action]
        pos = env.state.position

        if viz:
            viz.render(env, action_name, reward, step)

        # Logowanie co jakiś czas lub przy ważnych akcjach
        if action >= 8 or step % 20 == 0:
            print(
                f"Step {step:4d} | Action: {action_name:10} | Pos: {pos.c:4.0f},{pos.r:4.0f} | R: {reward:6.1f} | Gifts: {len(env.state.loaded_gifts)}"
            )

        state = next_state
        total_reward += reward
        step += 1

        if step > 3000:
            print("❌ Przekroczono limit kroków ewaluacji.")
            break

    print(
        f"Koniec. Wynik: {total_reward:.2f}. Dostarczono: {len(env.state.delivered_gifts)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "eval"])
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    problem, simulator = load_problem(INPUT_FILE)
    env = SleighEnv(problem, simulator)

    # Inicjalizacja agenta z dynamicznym rozmiarem wejścia
    agent = DQNAgent(env.input_size, env.ACTION_SPACE_SIZE)

    if args.mode == "train":
        run_training(env, agent, args)
    else:
        run_evaluation(env, agent, args)
