from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from stable_baselines3 import PPO

from src.backtest import buy_and_hold, run_backtest
from src.data_loader import download_data, load_uploaded_csv
from src.lstm_model import extract_states, train_lstm
from src.preprocessing import prepare_data
from src.trading_environment import TradingEnv
from src.train_rl import train_agent


st.set_page_config(page_title="Intelligent Stock Trading Bot", page_icon="📈", layout="wide")
st.markdown("""<style>
.stApp { background: #0b1220; color: #e6edf7; }
[data-testid="stMetric"] { background: #121d2f; border: 1px solid #243550; padding: 14px; border-radius: 8px; }
[data-testid="stSidebar"] { background: #101a2b; }
h1, h2, h3 { color: #f2f6fc; }
</style>""", unsafe_allow_html=True)


def metric_card(label, value):
    st.metric(label, value)


@st.cache_data(show_spinner=False)
def cached_download(ticker: str):
    return download_data(ticker)


def get_bot_suggestion(predicted_return: float) -> str:
    if predicted_return > 0.005:
        return "BUY"
    if predicted_return < -0.005:
        return "SELL"
    return "HOLD"


def run_pipeline(data: pd.DataFrame, ticker: str, epochs: int, timesteps: int, initial_cash: float, cost: float):
    prepared = prepare_data(data)
    lstm, history, lstm_info = train_lstm(prepared.scaled_features, prepared.targets, prepared.train_mask, epochs=epochs)
    states = extract_states(lstm, prepared.scaled_features)
    next_return = float(lstm.predict(prepared.scaled_rows[-prepared.lookback:][None, ...], verbose=0).reshape(-1)[0])
    train_mask = prepared.train_mask
    test_mask = ~prepared.train_mask
    train_prices = prepared.features.loc[prepared.sequence_dates[train_mask], "Close"].to_numpy()
    train_dates = prepared.sequence_dates[train_mask]
    test_states = states[test_mask]
    test_prices = prepared.features.loc[prepared.sequence_dates[test_mask], "Close"].to_numpy()
    test_dates = prepared.sequence_dates[test_mask]
    agent, _ = train_agent(states[train_mask], train_prices, train_dates, initial_cash, cost, timesteps)
    env = TradingEnv(test_states, test_prices, test_dates, initial_cash, cost)
    trades, metrics = run_backtest(agent, env)
    trades["buy_and_hold"] = buy_and_hold(pd.Series(test_prices), initial_cash).iloc[1:].to_numpy()
    metrics["buy_hold_return"] = float(trades["buy_and_hold"].iloc[-1] / initial_cash - 1)
    lstm_info["dates"] = test_dates
    lstm_info["next_return"] = next_return
    lstm_info["projected_close"] = float(data["Close"].iloc[-1] * (1 + next_return))
    lstm_info["suggestion"] = get_bot_suggestion(next_return)
    return {"ticker": ticker, "data": data, "prepared": prepared, "history": history,
            "lstm_info": lstm_info, "trades": trades, "metrics": metrics}


st.title("Intelligent Stock Trading Bot")
st.caption("A learning program that studies past prices and tests trading decisions | Past data only")

with st.sidebar:
    st.header("Choose your test")
    ticker = st.text_input("Stock symbol", "AAPL").strip().upper()
    uploaded = st.file_uploader("Use your own price file (optional)", type="csv")
    epochs = st.slider("How many times should the price model learn?", 3, 40, 12)
    timesteps = st.slider("How many decisions should the trading bot practise?", 500, 20_000, 3_000, step=500)
    initial_cash = st.number_input("Starting money", min_value=1000.0, value=100000.0, step=5000.0)
    cost = st.slider("Fee paid for each trade", 0.0, 0.02, 0.001, step=0.0005, format="%.3f")
    run = st.button("Start training and testing", type="primary", use_container_width=True)

if "result" not in st.session_state:
    st.session_state.result = None
needs_run = run

if uploaded is not None and run:
    try:
        data, source = load_uploaded_csv(uploaded)
        selected_ticker = source["ticker"]
    except Exception as exc:
        st.error(f"CSV error: {exc}")
        st.stop()
else:
    data = None
    selected_ticker = ticker
    if run:
        try:
            with st.spinner(f"Getting real past price data for {ticker}..."):
                data, source = cached_download(ticker)
        except Exception as exc:
            st.error(f"Data download failed: {exc}")
            st.info("Upload a CSV containing Date, Open, High, Low, Close, and Volume columns.")
            st.stop()

if run or needs_run:
    try:
        with st.spinner("Teaching the bot, then testing it on later dates..."):
            st.session_state.result = run_pipeline(data, selected_ticker, epochs, timesteps, initial_cash, cost)
    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")
        st.exception(exc)
        st.stop()

if st.session_state.result is None:
    st.subheader("Welcome")
    st.write(
        "Choose a stock symbol or upload historical prices, adjust the settings, "
        "and click **Start training and testing** to explore the bot."
    )
    st.info(
        "The main page is ready. Training is started manually so the app opens "
        "quickly and does not begin a long computation automatically."
    )
    st.stop()

result = st.session_state.result
data = result["data"]
metrics = result["metrics"]
trades = result["trades"]

st.info(f"Using {'your uploaded file' if selected_ticker == 'Uploaded dataset' else 'real Yahoo Finance data'} | Dates: {data.index.min().date()} to {data.index.max().date()} | Price records: {len(data):,}")
cols = st.columns(5)
with cols[0]: metric_card("Ticker", result["ticker"])
with cols[1]: metric_card("Latest close", f"{data['Close'].iloc[-1]:,.2f}")
with cols[2]: metric_card("Bot result", f"{metrics['total_return']:.2%}")
with cols[3]: metric_card("Buy and keep result", f"{metrics['buy_hold_return']:.2%}")
with cols[4]: metric_card("Money at the end", f"{metrics['final_value']:,.2f}")

market_tab, lstm_tab, rl_tab, about_tab = st.tabs(["Prices and data", "Bot prediction", "Results", "How it works"])
with market_tab:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=data.index, y=data.Close, name="Close", line={"color": "#55aaff"}), secondary_y=False)
    fig.add_trace(go.Bar(x=data.index, y=data.Volume, name="Volume", opacity=0.25), secondary_y=True)
    fig.update_layout(template="plotly_dark", height=430, margin={"t": 20, "b": 20})
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(data.tail(20), use_container_width=True)

with lstm_tab:
    info = result["lstm_info"]
    st.write("The bot looks at the last 30 trading days to estimate what the next return may be. It then uses this learned information when choosing BUY, SELL, or HOLD.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bot suggestion", info["suggestion"])
    c2.metric("Estimated next closing price", f"{info['projected_close']:,.2f}")
    c3.metric("Average prediction error", f"{info['mae']:.6f}")
    c4.metric("Typical prediction error", f"{info['rmse']:.6f}")
    st.write(f"The bot expects the next-day change to be **{info['next_return']:.2%}**.")
    st.caption("BUY means the estimate is above +0.5%, SELL means it is below -0.5%, and HOLD means it is between those values. This is an educational estimate based on past prices, not financial advice.")
    pred = pd.DataFrame({"Date": info["dates"], "Actual return": info["predictions"]["actual"], "Predicted return": info["predictions"]["predicted"]}).set_index("Date")
    st.line_chart(pred)
    st.line_chart(pd.DataFrame(result["history"]))

with rl_tab:
    st.metric("Largest drop from a previous high", f"{metrics['max_drawdown']:.2%}")
    st.write(f"Bot choices: BUY {metrics['buy_actions']} | SELL {metrics['sell_actions']} | HOLD {metrics['hold_actions']} | Completed trades {metrics['executed_trades']}")
    curve = trades.set_index("date")[["portfolio_value", "buy_and_hold"]].rename(columns={"portfolio_value": "RL strategy", "buy_and_hold": "Buy and hold"})
    st.line_chart(curve)
    st.dataframe(trades, use_container_width=True)
    st.download_button("Download results as CSV", trades.to_csv(index=False), "backtest_results.csv", "text/csv")

with about_tab:
    st.markdown("""### How it works
- **LSTM:** learns sequential patterns in historical market features and emits a fixed-size hidden state.
- **PPO:** learns whether to HOLD, BUY, or SELL from that state plus cash, holdings, and portfolio value.
- **Backtest:** evaluates only on later, unseen dates and compares against buy-and-hold.

This is an educational historical simulation. It does not place trades, connect to a broker, or guarantee future profitability. Results depend on the selected historical period, data quality, costs, and random seed; this is not financial advice.
""")
