def parse_input(file_path):
    with open(file_path, "r") as f:
        lines = f.read().splitlines()

    # Linia 1: T, D, W, G
    T, D, W, G = map(int, lines[0].split())

    # Zakresy akceleracji
    acc_ranges = []
    current_line = 1
    for _ in range(W):
        l, a = map(int, lines[current_line].split())
        acc_ranges.append({"max_weight": l, "max_acc": a})
        current_line += 1

    # Prezenty
    gifts = []
    for i in range(G):
        parts = lines[current_line].split()
        # Format: name score weight c r
        gifts.append(
            {
                "id": i,
                "name": parts[0],
                "score": int(parts[1]),
                "weight": int(parts[2]),
                "c": int(parts[3]),
                "r": int(parts[4]),
            }
        )
        current_line += 1

    return {"T": T, "D": D, "acc_ranges": acc_ranges, "gifts": gifts}
