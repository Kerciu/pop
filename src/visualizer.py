import sys

import pygame


class Visualizer:
    def __init__(self, problem, width=800, height=800):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Santa Sleigh RL Visualization")
        self.font = pygame.font.SysFont("Arial", 18)
        self.clock = pygame.time.Clock()
        self.problem = problem

        # --- AUTOMATYCZNE SKALOWANIE WIDOKU ---
        max_coord = 10.0
        for g in problem.gifts:
            max_coord = max(max_coord, abs(g.destination.c), abs(g.destination.r))

        # Margines wizualny (np. 1.5x)
        self.map_limit = max_coord * 1.5
        self.scale = width / (self.map_limit * 2)
        print(f"🖥️  Skala wizualizacji: 1 jednostka = {self.scale:.2f} px")
        # --------------------------------------

        # Kolory (bez zmian)
        self.COLOR_BG = (30, 30, 30)
        self.COLOR_BASE = (50, 150, 255)
        self.COLOR_GIFT = (0, 200, 100)
        self.COLOR_GIFT_DELIVERED = (100, 100, 100)
        self.COLOR_SANTA = (255, 50, 50)
        self.COLOR_TRAIL = (255, 255, 50)
        self.COLOR_TEXT = (255, 255, 255)

        self.trail = []

    def _to_screen(self, c, r):
        """Konwertuje współrzędne świata gry na piksele ekranu."""
        x = (c + self.map_limit) * self.scale
        # Odwracamy Y, bo w pygame Y rośnie w dół, a na mapie w górę (zależy od definicji, tu przyjmujemy standard)
        y = self.height - (r + self.map_limit) * self.scale
        return int(x), int(y)

    def render(self, env, action_name, reward, step):
        # Obsługa zdarzeń (żeby okno się nie zawiesiło i można je było zamknąć)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill(self.COLOR_BG)

        state = env.state

        # 1. Rysowanie prezentów
        # Optymalizacja: Rysujemy tylko te niedostarczone jako kropki
        # (Można to zoptymalizować, rysując na osobnej powierzchni raz na jakiś czas)
        for gift in self.problem.gifts:
            if gift.name in state.delivered_gifts:
                color = self.COLOR_GIFT_DELIVERED
                size = 1
            else:
                color = self.COLOR_GIFT
                size = 2
                # Jeśli prezent jest załadowany na saniach, nie rysujemy go na mapie (jest w saniach)
                if gift.name in state.loaded_gifts:
                    continue

            px, py = self._to_screen(gift.destination.c, gift.destination.r)
            pygame.draw.circle(self.screen, color, (px, py), size)

        # 2. Rysowanie Bazy (Lapland)
        bx, by = self._to_screen(0, 0)
        pygame.draw.circle(self.screen, self.COLOR_BASE, (bx, by), 5)
        pygame.draw.rect(self.screen, self.COLOR_BASE, (bx - 10, by - 10, 20, 20), 1)

        # 3. Rysowanie Szlaku (Trail)
        santa_pos = state.position
        sx, sy = self._to_screen(santa_pos.c, santa_pos.r)

        self.trail.append((sx, sy))
        if len(self.trail) > 500:  # Ograniczamy długość ogona
            self.trail.pop(0)

        if len(self.trail) > 1:
            pygame.draw.lines(self.screen, self.COLOR_TRAIL, False, self.trail, 1)

        # 4. Rysowanie Mikołaja
        pygame.draw.circle(self.screen, self.COLOR_SANTA, (sx, sy), 6)

        # Rysowanie celu (linii do najbliższego prezentu)
        if state.loaded_gifts:
            target_name = state.loaded_gifts[0]
            if target_name in env.gifts_map:
                tgt = env.gifts_map[target_name].destination
                tx, ty = self._to_screen(tgt.c, tgt.r)
                pygame.draw.line(self.screen, (255, 0, 255), (sx, sy), (tx, ty), 1)

        # 5. HUD (Napisy)
        info_lines = [
            f"Step: {step}",
            f"Action: {action_name}",
            f"Pos: {santa_pos.c:.0f}, {santa_pos.r:.0f}",
            f"Vel: {state.velocity.vc:.1f}, {state.velocity.vr:.1f}",
            f"Gifts Loaded: {len(state.loaded_gifts)}",
            f"Delivered: {len(state.delivered_gifts)} / {len(self.problem.gifts)}",
            f"Fuel: {state.carrot_count}",
            f"Weight: {state.sleigh_weight}",
            f"Time: {state.current_time} / {self.problem.T}",
            f"Reward: {reward:.2f}",
        ]

        for i, line in enumerate(info_lines):
            text = self.font.render(line, True, self.COLOR_TEXT)
            self.screen.blit(text, (10, 10 + i * 20))

        pygame.display.flip()

        # Ograniczenie klatek, żeby symulacja była czytelna dla oka
        # Zmień na wyższą wartość (np. 60 lub wywal), jeśli chcesz fast-forward
        self.clock.tick(30)
