import os
import random
import sys

import numpy as np
import torch

# Dodajemy katalog src do ścieżki
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from agents.genetic_agent import GeneticAgent
from core.distance_utils import distance
from core.loader import load_problem
from env.sleigh_env import SleighEnv

# --- KONFIGURACJA ---
INPUT_FILE = "data/huge_challenge.in.txt"
# INPUT_FILE = "data/b_better_hurry.in.txt"
MODEL_PATH = "models_saved/santa_genetic_best.pth"

POPULATION_SIZE = 50  # Ile Mikołajów lata naraz
GENERATIONS = 1000  # Ile pokoleń
ELITE_SIZE = 5  # Ilu najlepszych przechodzi bez zmian
MUTATION_POWER = 0.05  # Jak bardzo zmieniamy wagi (0.05 = 5% zmian)


def evaluate_agent(env, agent):
    """Puszcza jednego agenta na mapę i zwraca jego wynik."""
    state_tensor = env.reset()
    if env.state.available_gifts:
        env._sort_loaded_gifts()

    done = False
    total_reward = 0
    steps = 0

    # Limiter kroków, żeby słabi nie latali w nieskończoność
    max_steps = env.problem.T + 100

    while not done and steps < max_steps:
        # 1. Logika Hybrydowa (Baza) - pomagamy tylko w bazie
        dist_to_base = distance(env.state.position, env.sim.lapland_pos)
        action_id = None

        # Prosta pomoc w bazie (ładowanie/tankowanie)
        if dist_to_base <= env.problem.D:
            if not env.state.loaded_gifts and env.state.available_gifts:
                action_id = 6  # Load
            elif env.state.carrot_count < 20:
                action_id = 5  # Fuel

        # 2. Sieć neuronowa (Latanie i Dostarczanie)
        if action_id is None:
            # Wymuszamy Deliver jeśli jesteśmy idealnie w celu
            if env.state.loaded_gifts:
                tgt = env.gifts_map[env.state.loaded_gifts[0]]
                if distance(env.state.position, tgt.destination) <= env.problem.D:
                    action_id = 7  # Deliver

            if action_id is None:
                action_id = agent.get_action(state_tensor)

        state_tensor, reward, done, _ = env.step(action_id)

        # Sortowanie po zmianach inwentarza
        if action_id in [6, 7]:
            env._sort_loaded_gifts()

        total_reward += reward
        steps += 1

    return total_reward, len(env.state.delivered_gifts)


def main():
    if not os.path.exists("models_saved"):
        os.makedirs("models_saved")

    # Ładowanie środowiska
    print("Ładowanie mapy...")
    problem, simulator = load_problem(INPUT_FILE)
    env = SleighEnv(
        problem, simulator
    )  # Upewnij się, że masz wersję z auto-skalowaniem!

    state_size = env.encoder.output_size
    action_size = env.action_space_size

    # 1. Inicjalizacja populacji
    print(f"Tworzenie populacji {POPULATION_SIZE} agentów...")
    population = [GeneticAgent(state_size, action_size) for _ in range(POPULATION_SIZE)]

    # Jeśli mamy stary model, wczytajmy go do części populacji (żeby nie zaczynać od zera)
    if os.path.exists(MODEL_PATH):
        print("Wczytano poprzedniego mistrza!")
        master = GeneticAgent(state_size, action_size)
        master.load_state_dict(torch.load(MODEL_PATH))
        # Pierwszy agent to mistrz, reszta to jego mutacje
        population[0] = master
        for i in range(1, POPULATION_SIZE):
            population[i] = master.mutate(power=0.1)

    best_global_score = -float("inf")

    # --- PĘTLA EWOLUCJI ---
    for gen in range(GENERATIONS):
        scores = []

        # Ocena każdego agenta
        for i, agent in enumerate(population):
            score, delivered = evaluate_agent(env, agent)
            scores.append((score, agent, delivered))
            # print(f"\rGen {gen} | Agent {i+1}/{POPULATION_SIZE}", end="")

        # Sortowanie: Najlepszy wynik na początku
        scores.sort(key=lambda x: x[0], reverse=True)

        best_score = scores[0][0]
        best_agent = scores[0][1]
        best_delivered = scores[0][2]

        # Zapisywanie rekordu
        if best_score > best_global_score:
            best_global_score = best_score
            torch.save(best_agent.state_dict(), MODEL_PATH)
            print(f"\n🚀 NOWY REKORD: {best_score:.2f} (Dostarczono: {best_delivered})")

        print(
            f"\rGen {gen:3d} | Best: {best_score:10.2f} | Avg: {np.mean([s[0] for s in scores]):10.2f} | Deliv: {best_delivered}"
        )

        # Selekcja i Mutacja (Tworzenie nowego pokolenia)
        new_population = []

        # 1. Elity (przechodzą dalej bez zmian)
        elites = [s[1] for s in scores[:ELITE_SIZE]]
        new_population.extend(elites)

        # 2. Dzieci (mutacje elit)
        # Resztę miejsc wypełniamy mutacjami najlepszych jednostek
        spots_left = POPULATION_SIZE - ELITE_SIZE
        for i in range(spots_left):
            parent = random.choice(elites)  # Losujemy rodzica z elity
            child = parent.mutate(mutation_power=MUTATION_POWER)
            new_population.append(child)

        population = new_population


if __name__ == "__main__":
    main()
