from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(values: pd.Series) -> float:
    drawdown = values / values.cummax() - 1
    return float(drawdown.min()) if len(drawdown) else 0.0


def sharpe_ratio(values: pd.Series) -> float:
    returns = values.pct_change().dropna()
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(np.sqrt(252) * returns.mean() / returns.std())


def run_backtest(model, env) -> tuple[pd.DataFrame, dict]:
    observation, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, done, _, _ = env.step(action)
    trades = pd.DataFrame(env.history)
    if trades.empty:
        raise ValueError("Backtest produced no observations.")
    values = trades["portfolio_value"]
    initial = env.initial_cash
    metrics = {"initial_capital": initial, "final_value": float(values.iloc[-1]),
               "total_return": float(values.iloc[-1] / initial - 1),
               "max_drawdown": max_drawdown(values), "sharpe": sharpe_ratio(values),
               "buy_actions": int((trades.action == 1).sum()), "sell_actions": int((trades.action == 2).sum()),
               "hold_actions": int((trades.action == 0).sum()), "executed_trades": int(trades.executed.sum())}
    return trades, metrics


def buy_and_hold(prices: pd.Series, initial_cash: float) -> pd.Series:
    first = float(prices.iloc[0])
    return initial_cash * prices / first
