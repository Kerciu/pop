import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.physics import SantaPhysics


class SantaPilotEnv(gym.Env):
    def __init__(self, acc_ranges, delivery_radius=3.0):
        super(SantaPilotEnv, self).__init__()
        self.physics = SantaPhysics(acc_ranges)
        self.D = delivery_radius

        # Akcje: Float, Up, Down, Right, Left
        self.action_space = spaces.Discrete(5)

        # Obs: [rel_x, rel_y, vel_x, vel_y, weight]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )

        self.max_steps = 200

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Losujemy scenariusz "doleć do celu"
        dist = np.random.uniform(20, 300)
        angle = np.random.uniform(0, 2 * np.pi)

        self.target = np.array([0.0, 0.0])
        # Startujemy z losowej pozycji względem celu
        self.pos = np.array([dist * np.cos(angle), dist * np.sin(angle)])
        self.vel = np.array([0.0, 0.0])
        self.weight = np.random.uniform(15, 60)  # Losowa waga sań

        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        # Agent widzi cel względem siebie
        rel = self.target - self.pos
        return np.array(
            [rel[0], rel[1], self.vel[0], self.vel[1], self.weight], dtype=np.float32
        )

    def step(self, action):
        self.steps += 1

        prev_dist = np.linalg.norm(self.target - self.pos)

        # Symulacja fizyki
        self.pos, self.vel, self.weight, acc_vec = self.physics.apply_action(
            self.pos, self.vel, self.weight, action
        )

        dist = np.linalg.norm(self.target - self.pos)

        # --- Reward Function ---
        reward = -0.5  # Stała kara za czas (zachęta do szybkości)

        # Nagroda za zbliżanie się (shaping)
        reward += (prev_dist - dist) * 1.5

        terminated = False
        truncated = False

        # Sukces: Jesteśmy w zasięgu i prędkość jest mała (bezpieczne parkowanie)
        # Choć w zadaniu "parkowanie" nie jest wymagane, ułatwia to manewry przy kolejnych prezentach.
        if dist <= self.D:
            reward += 200
            terminated = True

        if self.steps >= self.max_steps:
            truncated = True
            reward -= 50  # Kara za timeout

        # Kara za ucieczkę bardzo daleko
        if dist > 2000:
            truncated = True
            reward -= 100

        return self._get_obs(), reward, terminated, truncated, {}
