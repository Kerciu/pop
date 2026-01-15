import os
import random

OUTPUT_FILE = os.path.join("data", "huge_challenge.in.txt")


def generate():
    # Parametry wyzwania
    T = 2000  # Dużo czasu
    D = 10  # Zasięg rzutu
    N_ACC = 5  # Ilość poziomów obciążenia
    N_GIFTS = 100  # Aż 100 prezentów!

    os.makedirs("data", exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        # Nagłówek
        f.write(f"{T} {D} {N_ACC} {N_GIFTS}\n")

        # Tabela przyspieszeń (standardowa)
        # Waga | Przyspieszenie
        f.write("20 10\n")
        f.write("50 8\n")
        f.write("100 6\n")
        f.write("200 4\n")
        f.write("500 2\n")

        # Generowanie losowych prezentów w klastrach
        for i in range(1, N_GIFTS + 1):
            # Tworzymy 4 "miasta" (skupiska prezentów), żeby agent musiał latać między nimi
            cluster = random.choice([(50, 50), (-50, 50), (50, -50), (-50, -50)])
            cx, cy = cluster

            # Rozrzut wokół miasta
            x = cx + random.randint(-30, 30)
            y = cy + random.randint(-30, 30)
            weight = random.choice([5, 10, 15, 20, 50])

            f.write(f"Gift_{i} {i} {x} {y} {weight}\n")

    print(f"Wygenerowano plik: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate()
