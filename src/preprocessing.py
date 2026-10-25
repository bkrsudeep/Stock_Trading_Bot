from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = ["Return", "MA_10", "MA_30", "Volatility_10", "Range", "VolumeChange"]


@dataclass
class PreparedData:
    features: pd.DataFrame
    scaled_features: np.ndarray
    scaled_rows: np.ndarray
    scaler: StandardScaler
    train_end: int
    lookback: int
    sequence_dates: pd.DatetimeIndex
    targets: np.ndarray
    train_mask: np.ndarray


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    frame["Return"] = frame["Close"].pct_change()
    frame["MA_10"] = frame["Close"].rolling(10).mean()
    frame["MA_30"] = frame["Close"].rolling(30).mean()
    frame["Volatility_10"] = frame["Return"].rolling(10).std()
    frame["Range"] = (frame["High"] - frame["Low"]) / frame["Close"]
    frame["VolumeChange"] = frame["Volume"].pct_change()
    return frame.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS)


def prepare_data(data: pd.DataFrame, lookback: int = 30, train_ratio: float = 0.8) -> PreparedData:
    if not 0.5 <= train_ratio < 1:
        raise ValueError("train_ratio must be between 0.5 and 1.")
    featured = add_features(data)
    if len(featured) <= lookback + 10:
        raise ValueError("Not enough feature rows for the selected lookback window.")
    train_end = max(lookback + 1, int(len(featured) * train_ratio))
    scaler = StandardScaler().fit(featured.iloc[:train_end][FEATURE_COLUMNS])
    scaled = scaler.transform(featured[FEATURE_COLUMNS]).astype(np.float32)
    sequences, targets, dates, masks = [], [], [], []
    for target_index in range(lookback, len(featured)):
        sequences.append(scaled[target_index - lookback:target_index])
        targets.append(featured["Return"].iloc[target_index])
        dates.append(featured.index[target_index])
        masks.append(target_index < train_end)
    return PreparedData(featured, np.asarray(sequences, dtype=np.float32), scaled, scaler, train_end, lookback,
                        pd.DatetimeIndex(dates), np.asarray(targets, dtype=np.float32), np.asarray(masks))
