# Intelligent Stock Trading Bot

An educational, local-only project titled **Develop an Intelligent Stock Trading Bot Using RL with LSTM State Modeling**. It downloads real historical OHLCV data, builds leakage-safe market features, trains an LSTM to predict next-day returns, uses its hidden representation as the state for a PPO trading agent, and evaluates the agent on a chronological holdout period.

## Requirements

Windows 10/11, Python 3.11-3.13, and internet access for Yahoo Finance downloads. No brokerage connection or real-money trading is used.

## Install and run

```powershell
cd C:\Stock_Bot
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`. The main page opens immediately. Choose a ticker, download data, and click **Start training and testing** when you are ready. The default uses AAPL and five years of daily data. Use the sidebar controls to reduce epochs or PPO timesteps for a quick demonstration. Streamlit caches completed runs during the session and does not retrain on every rerun.

If Yahoo Finance is unavailable, upload a CSV with `Date`, `Open`, `High`, `Low`, `Close`, and `Volume` columns. The app identifies uploaded data clearly and never substitutes synthetic prices.

## Deploy on Render

This repository includes [`render.yaml`](./render.yaml), which configures a Render web service.

1. Create a new GitHub repository and push this project to it. Do not commit `.env`, API keys, or other secrets.
2. Sign in to [Render](https://render.com) and choose **New > Blueprint**.
3. Connect the GitHub repository and select the branch containing `render.yaml`.
4. Confirm the service settings and click **Apply**. Render will install `requirements.txt` and start Streamlit on Render's assigned port.
5. Open the generated `onrender.com` URL after the deploy finishes.

The free Render service may sleep when idle. The first request after sleeping can take time while the service starts. Training also uses CPU and can take several minutes; reduce the epochs and PPO timesteps in the sidebar for a demonstration.

## Project layout

- `app.py`: Streamlit interface and orchestration.
- `src/data_loader.py`: download, CSV upload, and validation.
- `src/preprocessing.py`: chronological features and training-only scaling.
- `src/lstm_model.py`: Keras LSTM, hidden-state extraction, and metrics.
- `src/trading_environment.py`: Gymnasium long-only portfolio simulator.
- `src/train_rl.py`: PPO training and model persistence.
- `src/backtest.py`: portfolio metrics and buy-and-hold baseline.
- `tests/`: accounting and environment smoke tests.
- `data/`, `models/`, `outputs/`: generated artifacts.

## Demonstration flow

1. Explain that the market data is historical, not live.
2. Show the feature and price charts.
3. Run the LSTM and discuss MAE/RMSE and the actual-vs-predicted return chart.
4. Explain that the LSTM hidden vector plus portfolio state is the PPO observation.
5. Compare RL, buy-and-hold, and cash-only curves on unseen dates.
6. Export the trade history CSV and discuss limitations: historical backtests do not guarantee future profitability, and this is not financial advice.

## Common issues

- **PowerShell blocks activation:** run `Set-ExecutionPolicy -Scope Process Bypass` in that terminal, then activate again.
- **No data returned:** check the ticker spelling or upload a valid CSV.
- **TensorFlow installation fails:** use a fresh 64-bit Python 3.11-3.13 environment and upgrade pip first.
- **Training is slow:** lower `LSTM epochs` and `PPO timesteps` in the sidebar; CPU training is supported.
