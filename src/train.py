import os

from core.loader import load_problem

from agents.dqn_agent import DQNAgent
from env.sleigh_env import SleighEnv


def main():
    # Ścieżka do problemu
    problem_path = "data/huge_challenge.in.txt"  # Upewnij się, że plik istnieje
    print(f"Ładowanie problemu: {problem_path}")

    problem, simulator = load_problem(problem_path)
    env = SleighEnv(problem, simulator)

    state_size = env.encoder.output_size
    action_size = env.action_space_size

    # Inicjalizacja Agenta
    agent = DQNAgent(state_size, action_size)

    # Próba wczytania zapisanego modelu (checkpoint)
    save_path = "models_saved/dqn_santa_best.pth"
    # Odkomentuj linię poniżej, jeśli chcesz kontynuować trening
    # if os.path.exists(save_path): agent.load(save_path)

    # --- PARAMETRY TRENINGU ---
    episodes = 3000  # Więcej epizodów dla dużej mapy!
    batch_size = 64  # Mniejszy batch = częstsze aktualizacje
    epsilon = 1.0  # Start od pełnej losowości
    # KLUCZOWA ZMIANA: Bardzo wolny spadek epsilona.
    # Przy 0.999 w 1000 epizodzie nadal będzie ~36% losowości.
    epsilon_decay = 0.999
    epsilon_min = 0.05  # Zawsze zostawiamy 5% na szaleństwo

    best_score = -float("inf")

    if not os.path.exists("models_saved"):
        os.makedirs("models_saved")

    print(f"--- START TRENINGU NA DUŻEJ MAPIE (Epsilon decay: {epsilon_decay}) ---")

    for e in range(episodes):
        state = env.reset()
        # Obsługa tensora (unsqueeze dla batcha)
        state = state.unsqueeze(0)

        total_reward = 0
        done = False

        while not done:
            # Wybór akcji
            action = agent.get_action(state, epsilon)

            # Krok symulacji
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.unsqueeze(0)

            # Zapamiętanie w buforze
            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

        # Aktualizacja Epsilona (po każdym epizodzie)
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Logowanie wyników
        if e % 10 == 0:
            print(
                f"Ep {e} | Score: {total_reward:.2f} | Best: {best_score:.2f} | Eps: {epsilon:.2f}"
            )

        # Zapisywanie najlepszego modelu
        if total_reward > best_score:
            best_score = total_reward
            agent.save(save_path)
            print(f"🚀 NOWY REKORD: {best_score:.2f} (Zapisano model)")

        # Co jakiś czas aktualizujemy sieć docelową (Target Network)
        if e % 10 == 0:
            agent.update_target_network()


if __name__ == "__main__":
    main()
