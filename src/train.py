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

    # if os.path.exists(save_path): agent.load(save_path)

    episodes = 3000
    epsilon = 1.0
    epsilon_decay = 0.999
    epsilon_min = 0.05
    best_score = -float("inf")

    print("--- START TRENINGU (Hybrid Logic) ---")

    for e in range(episodes):
        state = env.reset()
        state = state.unsqueeze(0)
        done = False
        total_reward = 0

        # step_count = 0  # <--- DODAJ LICZNIK
        last_action_was_load = False

        while not done:
            # step_count += 1
            # # DEBUG: Wypisz co 10 kroków, żeby widzieć, że żyje
            # if step_count % 10 == 0:
            #     print(
            #         f"\rEpizod {e}, Krok {step_count}, Paliwo: {env.state.carrot_count}, Waga: {env.state.sleigh_weight}",
            #         end="",
            #     )
            # --- OVERRIDE LOGIC ---
            forced_action = None
            current_state_obj = env.state
            dist_to_base = distance(current_state_obj.position, simulator.lapland_pos)

            # 1. W BAZIE I PUSTO -> ŁADUJ PREZENTY (ID 6)
            # ZMIANA: Dodajemy warunek "and not last_action_was_load"
            # Jeśli ostatnio próbowaliśmy ładować i nadal jest pusto (czyli się nie udało),
            # to nie wymuszajmy tego ponownie - niech agent spróbuje czegoś innego (np. losowej akcji)
            if (
                dist_to_base <= problem.D
                and not current_state_obj.loaded_gifts
                and current_state_obj.available_gifts
                and not last_action_was_load  # <--- NOWY WARUNEK
            ):
                forced_action = 6

            # 2. W BAZIE I PUSTY BAK -> ŁADUJ MARCHEWKI (ID 5)
            elif dist_to_base <= problem.D and current_state_obj.carrot_count < 20:
                forced_action = 5

            # 3. U CELU -> DOSTARCZ (ID 7)
            elif current_state_obj.loaded_gifts:
                target_name = current_state_obj.loaded_gifts[0]
                target_gift = env.gifts_map[target_name]
                if (
                    distance(current_state_obj.position, target_gift.destination)
                    <= problem.D
                ):
                    forced_action = 7

            if forced_action is not None:
                action = forced_action
            else:
                action = agent.get_action(state, epsilon)

            next_state, reward, done, _ = env.step(action)

            if action == 6 and not env.state.loaded_gifts:
                last_action_was_load = True
            else:
                last_action_was_load = False

            next_state = next_state.unsqueeze(0)

            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if e % 10 == 0:
            print(
                f"Ep {e} | Score: {total_reward:.2f} | Best: {best_score:.2f} | Eps: {epsilon:.2f}"
            )
            agent.update_target_network()

        if total_reward > best_score:
            best_score = total_reward
            agent.save(save_path)
            print(f"🚀 NOWY REKORD: {best_score:.2f}")


if __name__ == "__main__":
    main()
