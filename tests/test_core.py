import numpy as np
import pandas as pd

from src.backtest import max_drawdown
from src.preprocessing import prepare_data
from src.trading_environment import TradingEnv


def test_environment_executes_buy_and_sell():
    env = TradingEnv(np.zeros((4, 2), dtype=np.float32), np.array([10, 12, 11, 13], dtype=np.float32), initial_cash=100)
    env.reset(seed=1)
    env.step(1)
    assert env.shares > 0
    env.step(2)
    assert env.shares == 0


def test_max_drawdown():
    assert max_drawdown(pd.Series([100, 120, 90, 110])) == -0.25


def test_preprocessing_keeps_test_sequences_after_training_boundary():
    dates = pd.date_range("2020-01-01", periods=120, freq="D")
    close = np.linspace(100, 150, len(dates))
    data = pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1,
                         "Close": close, "Volume": np.full(len(dates), 1000)}, index=dates)
    prepared = prepare_data(data, lookback=10, train_ratio=0.8)
    assert prepared.train_mask.any()
    assert (~prepared.train_mask).any()
    assert prepared.sequence_dates[prepared.train_mask].max() < prepared.sequence_dates[~prepared.train_mask].min()
