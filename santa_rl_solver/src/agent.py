import os

from stable_baselines3 import PPO

from src.env import SantaPilotEnv


class PilotAgent:
    def __init__(self, acc_ranges, model_path="data/models/pilot_ppo"):
        self.acc_ranges = acc_ranges
        self.model_path = model_path
        self.model = None

    def train(self, timesteps=100000):
        env = SantaPilotEnv(self.acc_ranges)
        self.model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)
        self.model.learn(total_timesteps=timesteps)

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        print("Model saved.")

    def load(self):
        if os.path.exists(self.model_path + ".zip"):
            self.model = PPO.load(self.model_path)
            print("Model loaded.")
        else:
            print("No model found. Please train first.")

    def get_action(self, rel_x, rel_y, vx, vy, w):
        if self.model:
            obs = [rel_x, rel_y, vx, vy, w]
            action, _ = self.model.predict(obs, deterministic=True)
            return int(action)
        return 0  # Fallback: Wait
