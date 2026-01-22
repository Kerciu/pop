import os
import random
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from agents.genetic_agent import GeneticAgent
from core.distance_utils import distance
from core.loader import load_problem
from env.sleigh_env import SleighEnv

INPUT_FILE = "data/huge_challenge.in.txt"
MODEL_PATH = "models_saved/santa_genetic_best.pth"

POPULATION_SIZE = 50
GENERATIONS = 1000
ELITE_SIZE = 5
MUTATION_POWER = 0.05


def evaluate_agent(env, agent):
    """Puszcza jednego agenta na mapę i zwraca jego wynik."""
    state_tensor = env.reset()
    if env.state.available_gifts:
        env._sort_loaded_gifts()

    done = False
    total_reward = 0
    steps = 0

    max_steps = env.problem.T + 100

    while not done and steps < max_steps:
        dist_to_base = distance(env.state.position, env.sim.lapland_pos)
        action_id = None

        if dist_to_base <= env.problem.D:
            if not env.state.loaded_gifts and env.state.available_gifts:
                action_id = 6  # Load
            elif env.state.carrot_count < 20:
                action_id = 5  # Fuel

        if action_id is None:
            if env.state.loaded_gifts:
                tgt = env.gifts_map[env.state.loaded_gifts[0]]
                if distance(env.state.position, tgt.destination) <= env.problem.D:
                    action_id = 7  # Deliver

            if action_id is None:
                action_id = agent.get_action(state_tensor)

        state_tensor, reward, done, _ = env.step(action_id)

        if action_id in [6, 7]:
            env._sort_loaded_gifts()

        total_reward += reward
        steps += 1

    return total_reward, len(env.state.delivered_gifts)


def main():
    if not os.path.exists("models_saved"):
        os.makedirs("models_saved")

    print("Ładowanie mapy...")
    problem, simulator = load_problem(INPUT_FILE)
    env = SleighEnv(problem, simulator)

    state_size = env.encoder.output_size
    action_size = env.action_space_size

    print(f"Tworzenie populacji {POPULATION_SIZE} agentów...")
    population = [GeneticAgent(state_size, action_size) for _ in range(POPULATION_SIZE)]

    if os.path.exists(MODEL_PATH):
        print("Wczytano poprzedniego mistrza!")
        master = GeneticAgent(state_size, action_size)
        master.load_state_dict(torch.load(MODEL_PATH))
        population[0] = master
        for i in range(1, POPULATION_SIZE):
            population[i] = master.mutate(power=0.1)

    best_global_score = -float("inf")

    for gen in range(GENERATIONS):
        scores = []

        for i, agent in enumerate(population):
            score, delivered = evaluate_agent(env, agent)
            scores.append((score, agent, delivered))

        scores.sort(key=lambda x: x[0], reverse=True)

        best_score = scores[0][0]
        best_agent = scores[0][1]
        best_delivered = scores[0][2]

        if best_score > best_global_score:
            best_global_score = best_score
            torch.save(best_agent.state_dict(), MODEL_PATH)
            print(f"\n🚀 NOWY REKORD: {best_score:.2f} (Dostarczono: {best_delivered})")

        print(
            f"\rGen {gen:3d} | Best: {best_score:10.2f} | Avg: {np.mean([s[0] for s in scores]):10.2f} | Deliv: {best_delivered}"
        )

        new_population = []

        elites = [s[1] for s in scores[:ELITE_SIZE]]
        new_population.extend(elites)

        spots_left = POPULATION_SIZE - ELITE_SIZE
        for i in range(spots_left):
            parent = random.choice(elites)
            child = parent.mutate(mutation_power=MUTATION_POWER)
            new_population.append(child)

        population = new_population


if __name__ == "__main__":
    main()
