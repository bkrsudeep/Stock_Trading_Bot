from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf


REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def validate_market_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize OHLCV data without inventing observations."""
    frame = data.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [column[0] for column in frame.columns]
    frame.columns = [str(column).strip() for column in frame.columns]
    if "Date" in frame.columns:
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame = frame.set_index("Date")
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame[~frame.index.isna()].sort_index()
    frame = frame[~frame.index.duplicated(keep="first")]
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    for column in REQUIRED_COLUMNS + (["Adj Close"] if "Adj Close" in frame else []):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=REQUIRED_COLUMNS)
    if (frame[REQUIRED_COLUMNS] <= 0).any().any():
        raise ValueError("OHLCV data contains non-positive numeric values.")
    if len(frame) < 80:
        raise ValueError("At least 80 valid daily observations are required.")
    frame.index.name = "Date"
    return frame


def load_csv(path: str | Path) -> pd.DataFrame:
    return validate_market_data(pd.read_csv(path))


def download_data(ticker: str, period: str = "5y", output_dir: str | Path = "data") -> tuple[pd.DataFrame, dict[str, Any]]:
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Ticker symbol cannot be empty.")
    raw = yf.download(ticker, period=period, auto_adjust=False, progress=False, threads=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"No historical data was returned for {ticker}. Upload a valid CSV instead.")
    data = validate_market_data(raw)
    output_path = Path(output_dir) / f"{ticker.replace('.', '_')}_history.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path)
    return data, {"ticker": ticker, "source": "Yahoo Finance", "path": str(output_path)}


def load_uploaded_csv(uploaded_file: Any) -> tuple[pd.DataFrame, dict[str, str]]:
    data = validate_market_data(pd.read_csv(uploaded_file))
    return data, {"ticker": "Uploaded dataset", "source": "Local uploaded CSV", "path": "uploaded"}
