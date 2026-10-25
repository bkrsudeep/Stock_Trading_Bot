from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from .trading_environment import TradingEnv


def train_agent(states: np.ndarray, prices: np.ndarray, dates=None, initial_cash: float = 100_000,
                transaction_cost: float = 0.001, timesteps: int = 5_000, model_path: str | Path = "models/ppo_trader") -> tuple[PPO, TradingEnv]:
    env = TradingEnv(states, prices, dates, initial_cash, transaction_cost)
    n_steps = min(256, max(64, len(prices) - 1))
    batch_size = 64 if n_steps % 64 == 0 else 32
    agent = PPO("MlpPolicy", env, seed=42, verbose=0, n_steps=n_steps, batch_size=min(batch_size, n_steps))
    agent.learn(total_timesteps=timesteps, progress_bar=False)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(model_path))
    return agent, env
