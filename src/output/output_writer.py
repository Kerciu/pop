class OutputWriter:
    def __init__(self):
        self.commands = []

    def record_move(self, action_enum, value):
        """
        Zapisuje akcje ruchu: AccUp, AccDown, AccLeft, AccRight, Float.
        action_enum: nazwa z enuma (np. ACC_N) lub string
        value: wartość liczbowa (np. 1 lub calculated_max)
        """
        cmd_str = ""
        # Mapowanie naszych kierunków na format wyjściowy
        # Zakładamy: N=Up, S=Down, E=Right, W=Left (zależnie od układu współrzędnych)
        # W SleighEnv: 0=N (ay+), 1=S (ay-), 2=E (ax+), 3=W (ax-)

        name = str(action_enum).upper()

        if "ACC_N" in name or "MAX_N" in name:
            cmd_str = f"AccUp {int(value)}"
        elif "ACC_S" in name or "MAX_S" in name:
            cmd_str = f"AccDown {int(value)}"
        elif "ACC_E" in name or "MAX_E" in name:
            cmd_str = f"AccRight {int(value)}"
        elif "ACC_W" in name or "MAX_W" in name:
            cmd_str = f"AccLeft {int(value)}"
        elif "FLOAT" in name:
            cmd_str = f"Float {int(value)}"

        if cmd_str:
            self.commands.append(cmd_str)

    def record_load_carrots(self, amount):
        """Zapisuje LoadCarrots N"""
        self.commands.append(f"LoadCarrots {amount}")

    def record_load_gift(self, gift_name):
        """Zapisuje LoadGift ChildName"""
        self.commands.append(f"LoadGift {gift_name}")

    def record_deliver_gift(self, gift_name):
        """Zapisuje DeliverGift ChildName"""
        self.commands.append(f"DeliverGift {gift_name}")

    def save(self, filepath):
        """Zapisuje plik zgodnie ze specyfikacją (pierwsza linia to liczba akcji)"""
        if not self.commands:
            print("⚠️ Brak komend do zapisu.")
            return

        try:
            with open(filepath, "w") as f:
                # 1. Liczba akcji C
                f.write(f"{len(self.commands)}\n")
                # 2. Akcje linia po linii
                for cmd in self.commands:
                    f.write(f"{cmd}\n")
            print(f"✅ Zapisano rozwiązanie do: {filepath}")
        except Exception as e:
            print(f"❌ Błąd zapisu pliku: {e}")
