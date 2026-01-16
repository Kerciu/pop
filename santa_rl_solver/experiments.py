import matplotlib.pyplot as plt
import numpy as np
from src.agent import PilotAgent

from src.env import SantaPilotEnv


def run_convergence_experiment(acc_ranges):
    """Eksperyment 1: Sprawdzenie czy agent się uczy."""
    print("Running Convergence Experiment...")
    # Tutaj normalnie użylibyśmy callbacków w SB3 do logowania rewardu
    # Symulacja: Trenujemy krótko i sprawdzamy średni reward
    rewards = []
    checkpoints = [1000, 10000, 50000]

    for steps in checkpoints:
        agent = PilotAgent(acc_ranges, model_path=f"data/models/exp_{steps}")
        agent.train(timesteps=steps)

        # Ewaluacja
        env = SantaPilotEnv(acc_ranges)
        total_r = 0
        for _ in range(10):
            obs, _ = env.reset()
            done = False
            while not done:
                action = agent.get_action(*obs)
                obs, r, term, trunc, _ = env.step(action)
                total_r += r
                done = term or trunc
        rewards.append(total_r / 10)

    plt.plot(checkpoints, rewards, marker="o")
    plt.title("Mean Reward vs Training Steps")
    plt.xlabel("Steps")
    plt.ylabel("Avg Reward")
    plt.savefig("data/outputs/convergence.png")
    print("Convergence plot saved.")


def run_trajectory_visualization(acc_ranges, agent_path):
    """Eksperyment 2: Wizualizacja trasy agenta."""
    print("Visualizing Trajectory...")
    agent = PilotAgent(acc_ranges, model_path=agent_path)
    agent.load()

    env = SantaPilotEnv(acc_ranges)
    obs, _ = env.reset()
    # Wymuś konkretny cel dla czytelności
    env.target = np.array([100.0, 100.0])
    env.pos = np.array([0.0, 0.0])

    path_x, path_y = [], []

    done = False
    while not done:
        path_x.append(env.pos[0])
        path_y.append(env.pos[1])
        action = agent.get_action(*obs)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc

    plt.figure()
    plt.plot(path_x, path_y, label="Sleigh Path")
    plt.scatter([100], [100], c="red", label="Target")
    plt.scatter([0], [0], c="green", label="Start")
    plt.legend()
    plt.title("Agent Trajectory (Frictionless)")
    plt.savefig("data/outputs/trajectory.png")
    print("Trajectory plot saved.")


if __name__ == "__main__":
    # Mock data for ranges
    ranges = [{"max_weight": 20, "max_acc": 5}, {"max_weight": 100, "max_acc": 2}]
    run_convergence_experiment(ranges)
    # run_trajectory_visualization(ranges, "data/models/pilot_ppo")
