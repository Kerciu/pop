import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from agents.dqn_agent import DQNAgent
from core.loader import load_problem
from env.sleigh_env import SleighEnv
from output.output_writer import OutputWriter
from visualizer import Visualizer

MODEL_PATH = "models_saved/santa_final_try.pth"
INPUT_FILE = "data/huge_challenge.in.txt"


def run_training(env, agent, args):
    print(f"--- START TRENINGU ({args.episodes} odcinków) ---")
    if not os.path.exists("models_saved"):
        os.makedirs("models_saved")

    epsilon = 1.0
    epsilon_decay = 0.999
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
            if steps > 2000:
                break

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        recent_rewards.append(total_reward)
        if len(recent_rewards) > 50:
            recent_rewards.pop(0)
        avg_reward = sum(recent_rewards) / len(recent_rewards)

        if e % 10 == 0:
            print(
                f"Ep {e:4d} | Avg: {avg_reward:8.2f} | Eps: {epsilon:.2f} | Steps: {steps} | Deliv: {len(env.state.delivered_gifts)}"
            )
            agent.update_target_network()

        if avg_reward > best_avg_reward and e > 20:
            best_avg_reward = avg_reward
            agent.save(MODEL_PATH)
            print(f"💾 Zapisano rekord: {best_avg_reward:.2f}")


def run_evaluation(env, agent, args):
    print("--- EWALUACJA ---")
    try:
        agent.load(MODEL_PATH)
        print(f"✅ Wczytano: {MODEL_PATH}")
    except:
        print("❌ Brak modelu!")
        return

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
        carrots_before = env.state.carrot_count
        loaded_before = set(env.state.loaded_gifts)
        delivered_before = set(env.state.delivered_gifts)
        vc_before = env.state.velocity.vc
        vr_before = env.state.velocity.vr

        action = agent.get_action(state, epsilon=0.0)
        next_state, reward, done, _ = env.step(action)

        if action < 8:
            dv_c = abs(env.state.velocity.vc - vc_before)
            dv_r = abs(env.state.velocity.vr - vr_before)
            val = int(max(dv_c, dv_r))
            if val == 0:
                val = 1
            writer.record_move(action_names[action], val)
            writer.record_move("FLOAT", 1)
        elif action == 8:
            writer.record_move("FLOAT", 1)
        elif action == 9:
            for gid in set(env.state.loaded_gifts) - loaded_before:
                writer.record_load_gift(gid)
        elif action == 10:
            if env.state.carrot_count > carrots_before:
                writer.record_load_carrots(env.state.carrot_count - carrots_before)
        elif action == 11:
            for gid in set(env.state.delivered_gifts) - delivered_before:
                writer.record_deliver_gift(gid)

        if viz:
            viz.render(env, action_names[action], reward, step)
        if step % 20 == 0 or action >= 9:
            print(
                f"Step {step:4d} | {action_names[action]:10} | Pos: {env.state.position.c:.0f},{env.state.position.r:.0f} | R: {reward:6.1f}"
            )

        state = next_state
        total_reward += reward
        step += 1
        if step > 3000:
            break

    print(
        f"Koniec. Wynik: {total_reward}. Dostarczono: {len(env.state.delivered_gifts)}"
    )
    writer.save("solution.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "eval"])
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    problem, simulator = load_problem(INPUT_FILE)
    env = SleighEnv(problem, simulator)
    agent = DQNAgent(env.input_size, env.ACTION_SPACE_SIZE)

    if args.mode == "train":
        run_training(env, agent, args)
    else:
        run_evaluation(env, agent, args)
