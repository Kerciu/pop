import os

from agents.dqn_agent import DQNAgent
from core.acceleration_table import AccelerationTable
from core.simulator import Simulator
from env.sleigh_env import SleighEnv
from models.problem import Problem

INPUT_DATA_PATH = os.path.join("data", "huge_challenge.in.txt")
MODEL_PATH = os.path.join("models_saved", "dqn_santa_best.pth")


def main():
    # 1. Setup środowiska
    problem = Problem(INPUT_DATA_PATH)
    # Sprawdzamy czy Problem poprawnie załadował ranges, jeśli nie to fallback (dla bezpieczeństwa)
    accel_ranges = (
        problem.acceleration_ranges if hasattr(problem, "acceleration_ranges") else []
    )
    accel_table = AccelerationTable(accel_ranges)

    all_gift_map = {g.name: g for g in problem.gifts}

    sim = Simulator(problem.T, problem.D, accel_table, all_gift_map)
    env = SleighEnv(problem, sim)

    # 2. Inicjalizacja Agenta
    agent = DQNAgent(env.encoder.output_size, env.action_space_size)

    # 3. Wczytanie wytrenowanego mózgu
    if os.path.exists(MODEL_PATH):
        print(f"Wczytywanie modelu z {MODEL_PATH}...")
        try:
            agent.load(MODEL_PATH)
        except RuntimeError:
            print(
                "BŁĄD: Nie można wczytać modelu. Prawdopodobnie różnica w rozmiarze akcji."
            )
            print(
                f"Model na dysku ma inny rozmiar niż obecne środowisko ({env.action_space_size} akcji)."
            )
            print("Musisz wytrenować model od nowa (uruchom train.py).")
            return
    else:
        print("Brak zapisanego modelu! Uruchom najpierw train.py")
        return

    # 4. Pętla testowa
    state = env.reset()

    # --- POPRAWKA: Dodanie wymiaru batcha [1, 12] ---
    state = state.unsqueeze(0)

    done = False
    total_reward = 0
    step_count = 0

    print("\n--- START SYMULACJI POKAZOWEJ ---")

    while not done:
        # Agent podejmuje decyzję (bez losowości)
        action_id = agent.get_action(state, epsilon=0.0)

        # Wykonanie kroku
        next_state, reward, done, info = env.step(action_id)

        # --- POPRAWKA: Dodanie wymiaru batcha dla następnego stanu ---
        next_state = next_state.unsqueeze(0)

        # Pobieranie czytelnej nazwy akcji z mapowania środowiska
        action_enum, param = env.ACTION_MAPPING[action_id]
        action_name = f"{action_enum.name}({param})"

        # Logowanie
        pos = env.state.position
        # print(
        #     f"Krok {step_count}: {action_name:<20} -> Poz: ({pos.c:.1f}, {pos.r:.1f}) | "
        #     f"Prezenty: {len(env.state.loaded_gifts)} | Paliwo: {env.state.carrot_count} | Nagroda: {reward:.2f}"
        # )

        state = next_state
        total_reward += reward
        step_count += 1

        # Opcjonalnie: Zwolnij tempo, żeby widzieć co się dzieje
        # time.sleep(0.05)

    print(f"\nKoniec! Wynik końcowy: {total_reward:.2f}")
    print(
        f"Dostarczone prezenty: {len(env.state.delivered_gifts)} / {len(problem.gifts)}"
    )


if __name__ == "__main__":
    main()
