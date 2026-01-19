import os

from agents.dqn_agent import DQNAgent
from core.distance_utils import distance
from core.loader import load_problem
from env.sleigh_env import SleighEnv

# Upewnij się, że ścieżki są poprawne
INPUT_DATA_PATH = os.path.join("data", "b_better_hurry.in.txt")
MODEL_PATH = os.path.join("models_saved", "dqn_santa_best.pth")


def main():
    # 1. Używamy load_problem, tak jak w train.py, żeby setup był identyczny
    if not os.path.exists(INPUT_DATA_PATH):
        print(f"Błąd: Nie znaleziono pliku danych: {INPUT_DATA_PATH}")
        return

    problem, simulator = load_problem(INPUT_DATA_PATH)
    env = SleighEnv(problem, simulator)

    state_size = env.encoder.output_size
    action_space_size = env.action_space_size

    agent = DQNAgent(state_size, action_space_size)

    # 2. Wczytywanie modelu
    if os.path.exists(MODEL_PATH):
        print(f"Wczytywanie modelu z {MODEL_PATH}...")
        try:
            agent.load(MODEL_PATH)
            agent.policy_net.eval()  # Ustawiamy sieć w tryb ewaluacji (wyłącza dropout itp.)
        except Exception as e:
            print(f"Błąd wczytywania modelu: {e}")
            return
    else:
        print("Błąd: Brak zapisanego modelu. Najpierw uruchom train.py.")
        return

    # 3. Reset środowiska
    state = env.reset()
    state = state.unsqueeze(0)
    done = False
    total_reward = 0
    step = 0

    # Zmienna pomocnicza z train.py zapobiegająca pętlom ładowania
    last_action_was_load = False

    print("\n--- START SYMULACJI (EWALUACJA) ---")

    while not done:
        # --- HYBRID LOGIC (Kluczowe dla poprawnego działania) ---
        # Musi być identyczna jak w train.py

        forced_action = None
        current_state_obj = env.state
        dist_to_base = distance(current_state_obj.position, simulator.lapland_pos)

        # 1. W BAZIE I PUSTO -> ŁADUJ PREZENTY (ID 6)
        if (
            dist_to_base <= problem.D
            and not current_state_obj.loaded_gifts
            and current_state_obj.available_gifts
            and not last_action_was_load
        ):
            forced_action = 6

        # 2. W BAZIE I PUSTY BAK -> ŁADUJ MARCHEWKI (ID 5)
        # (Opcjonalnie, jeśli w train.py to miałeś)
        elif dist_to_base <= problem.D and current_state_obj.carrot_count < 20:
            forced_action = 5

        # 3. U CELU -> DOSTARCZ (ID 7)
        elif current_state_obj.loaded_gifts:
            target_name = current_state_obj.loaded_gifts[0]
            target_gift = env.gifts_map[target_name]
            dist = distance(current_state_obj.position, target_gift.destination)

            if dist <= problem.D:
                forced_action = 7  # Deliver

        # --- WYBÓR AKCJI ---
        if forced_action is not None:
            action_id = forced_action
            is_ai_action = False
        else:
            # Epsilon = 0.0 -> Czysta eksploatacja wiedzy sieci (bez losowości)
            action_id = agent.get_action(state, epsilon=0.0)
            is_ai_action = True

        # --- WYKONANIE KROKU ---
        next_state, reward, done, _ = env.step(action_id)
        next_state = next_state.unsqueeze(0)

        # Aktualizacja flagi logicznej
        if action_id == 6 and not env.state.loaded_gifts:
            last_action_was_load = True
        else:
            last_action_was_load = False

        # --- LOGOWANIE ---
        action_enum = env.ACTION_MAPPING[action_id]
        pos = env.state.position
        source = "LOGIC" if not is_ai_action else "AI_NET"

        # Wypisujemy co krok (lub co 10/100, jeśli za szybko leci)
        print(
            f"Step {step:4d} | [{source}] {action_enum.name:13} | "
            f"Pos: {pos.c:6.0f},{pos.r:6.0f} | "
            f"Gifts: {len(env.state.loaded_gifts):2} | "
            f"Fuel: {env.state.carrot_count:3} | "
            f"Reward: {reward:6.2f}"
        )

        state = next_state
        total_reward += reward
        step += 1

        # Opcjonalne spowolnienie, żebyś widział co się dzieje
        # time.sleep(0.05)

    print("\n" + "=" * 30)
    print("KONIEC SYMULACJI")
    print(f"Całkowity wynik: {total_reward:.2f}")
    print(
        f"Dostarczono prezentów: {len(env.state.delivered_gifts)} / {len(problem.gifts)}"
    )
    print("=" * 30)


if __name__ == "__main__":
    main()
