import os

from agents.dqn_agent import DQNAgent
from core.distance_utils import distance
from core.loader import load_problem
from env.sleigh_env import SleighEnv


def main():
    problem_path = "data/b_better_hurry.in.txt"
    problem, simulator = load_problem(problem_path)
    env = SleighEnv(problem, simulator)

    state_size = env.encoder.output_size
    action_size = env.action_space_size
    agent = DQNAgent(state_size, action_size)

    save_path = "models_saved/dqn_santa_best.pth"
    if not os.path.exists("models_saved"):
        os.makedirs("models_saved")

    episodes = 10000

    # --- PARAMETRY EPSILON ---
    epsilon = 1.0
    epsilon_decay = 0.997  # Wolniejszy spadek, żeby dłużej eksplorował
    epsilon_min = 0.05

    # --- PARAMETRY WYCHODZENIA Z OPTIMUM (NOWOŚĆ) ---
    patience = 100  # Ile epizodów czekamy na rekord
    episodes_without_progress = 0
    epsilon_boost_value = 0.6  # Do ilu podbijamy losowość jak utknie

    best_score = -float("inf")

    print("--- START TRENINGU (Adaptive Epsilon) ---")

    for e in range(episodes):
        state = env.reset()
        state = state.unsqueeze(0)
        done = False
        total_reward = 0
        last_action_was_load = False

        while not done:
            # --- OVERRIDE LOGIC (Baza) ---
            forced_action = None
            current_state_obj = env.state
            dist_to_base = distance(current_state_obj.position, simulator.lapland_pos)

            # 1. ŁADUJ (Baza + Pusto)
            if (
                dist_to_base <= problem.D
                and not current_state_obj.loaded_gifts
                and current_state_obj.available_gifts
                and not last_action_was_load
            ):
                forced_action = 6
            # 2. TANKUJ (Baza + Pusty bak)
            elif dist_to_base <= problem.D and current_state_obj.carrot_count < 20:
                forced_action = 5
            # 3. DOSTARCZ (Blisko celu) - to pomaga sieci trafić
            elif current_state_obj.loaded_gifts:
                target_name = current_state_obj.loaded_gifts[
                    0
                ]  # Env sortuje, więc [0] to najbliższy
                target_gift = env.gifts_map[target_name]
                if (
                    distance(current_state_obj.position, target_gift.destination)
                    <= problem.D
                ):
                    forced_action = 7

            if forced_action is not None:
                action = forced_action
            else:
                # Tu działa losowość (Epsilon)
                action = agent.get_action(state, epsilon)

            next_state, reward, done, _ = env.step(action)

            # Flaga zabezpieczająca przed pętlą ładowania
            if action == 6 and not env.state.loaded_gifts:
                last_action_was_load = True
            else:
                last_action_was_load = False

            next_state = next_state.unsqueeze(0)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        # --- LOGIKA ADAPTACYJNA ---
        if total_reward > best_score:
            best_score = total_reward
            agent.save(save_path)
            print(f"🚀 NOWY REKORD: {best_score:.2f}")
            episodes_without_progress = 0  # Reset licznika cierpliwości
        else:
            episodes_without_progress += 1

        # Mechanizm "Wstrząsu" (Simulated Annealing / Restart)
        if episodes_without_progress >= patience:
            print(
                f"⚠️ UTKNIĘCIE! Zwiększam losowość (Epsilon {epsilon:.2f} -> {epsilon_boost_value})"
            )
            epsilon = epsilon_boost_value
            episodes_without_progress = 0  # Dajemy mu nową szansę
            # Opcjonalnie: Zmniejszamy learning rate, jeśli agent jest już "stary"
            # agent.optimizer.param_groups[0]['lr'] *= 0.9

        # Standardowy spadek epsilona
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if e % 20 == 0:
            print(
                f"Ep {e} | Score: {total_reward:.2f} | Best: {best_score:.2f} | Eps: {epsilon:.2f} | NoProg: {episodes_without_progress}"
            )
            agent.update_target_network()


if __name__ == "__main__":
    main()
