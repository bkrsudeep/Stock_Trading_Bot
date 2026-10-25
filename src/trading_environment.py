from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TradingEnv(gym.Env):
    """Long-only, discrete-action trading simulator using precomputed LSTM states."""
    metadata = {"render_modes": []}

    def __init__(self, states: np.ndarray, prices: np.ndarray, dates=None, initial_cash: float = 100_000,
                 transaction_cost: float = 0.001, slippage: float = 0.0005):
        super().__init__()
        if len(states) != len(prices) or len(prices) < 2:
            raise ValueError("States and prices must have equal length and at least two rows.")
        self.states = np.asarray(states, dtype=np.float32)
        self.prices = np.asarray(prices, dtype=np.float32)
        self.dates = dates
        self.initial_cash = float(initial_cash)
        self.transaction_cost = float(transaction_cost)
        self.slippage = float(slippage)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.states.shape[1] + 3,), dtype=np.float32)
        self.reset()

    def _observation(self):
        price = max(float(self.prices[self.index]), 1e-8)
        value = self.cash + self.shares * price
        return np.concatenate([self.states[self.index], [self.cash / self.initial_cash, self.shares * price / self.initial_cash, value / self.initial_cash]]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.index, self.cash, self.shares = 0, self.initial_cash, 0.0
        self.history = []
        return self._observation(), {}

    def step(self, action):
        action = int(action)
        price = float(self.prices[self.index])
        before = self.cash + self.shares * price
        executed = False
        if action == 1 and self.cash > 0:
            fill = price * (1 + self.slippage)
            shares = self.cash / (fill * (1 + self.transaction_cost))
            self.cash -= shares * fill * (1 + self.transaction_cost)
            self.shares += shares
            executed = shares > 0
        elif action == 2 and self.shares > 0:
            fill = price * (1 - self.slippage)
            self.cash += self.shares * fill * (1 - self.transaction_cost)
            self.shares = 0.0
            executed = True
        self.index += 1
        terminated = self.index >= len(self.prices) - 1
        next_price = float(self.prices[self.index])
        portfolio = self.cash + self.shares * next_price
        reward = float(np.log(max(portfolio, 1e-8) / max(before, 1e-8)))
        record = {"date": self.dates[self.index] if self.dates is not None else self.index, "action": action,
                  "executed": executed, "price": next_price, "shares": self.shares,
                  "cash": self.cash, "portfolio_value": portfolio}
        self.history.append(record)
        return self._observation(), reward, terminated, False, {"portfolio_value": portfolio, "executed": executed}
