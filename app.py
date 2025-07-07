import streamlit as st
import requests
from textblob import TextBlob
from datetime import datetime
import pandas as pd
import numpy as np
from telethon import TelegramClient
from streamlit_autorefresh import st_autorefresh
import asyncio
import plotly.graph_objects as go
import openai

# Set OpenAI API key from secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Auto-refresh: price every 10 seconds, full data every 60 seconds
st_autorefresh(interval=10000, limit=None, key="price_refresh")
st_autorefresh(interval=60000, limit=None, key="data_refresh")

# Configs and secrets
TWELVEDATA_API_KEY = st.secrets["TWELVEDATA_API_KEY"]
NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
TELEGRAM_API_ID = int(st.secrets["TELEGRAM_API_ID"])
TELEGRAM_API_HASH = st.secrets["TELEGRAM_API_HASH"]
TELEGRAM_CHANNEL = 'Gary_TheTrader'  # without @

LOG_FILE = "signal_history.csv"

def fetch_latest_xauusd_price():
    url = f"https://api.twelvedata.com/price?symbol=XAU/USD&apikey={TWELVEDATA_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        return float(data['price'])
    except Exception:
        return None

def fetch_weekly_chart_data():
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1week&apikey={TWELVEDATA_API_KEY}&outputsize=50"
    try:
        resp = requests.get(url)
        data = resp.json()
        if "values" not in data:
            return pd.DataFrame()
        df = pd.DataFrame(data['values'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].astype(float)
        return df
    except Exception:
        return pd.DataFrame()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series):
    exp1 = series.ewm(span=12, adjust=False).mean()
    exp2 = series.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal

def analyze_technical_indicators(df):
    df['RSI'] = calculate_rsi(df['close'])
    df['MACD_HIST'] = calculate_macd(df['close'])
    last = df.iloc[-1]
    return last['RSI'], last['MACD_HIST']

def fetch_news():
    url = f"https://newsapi.org/v2/everything?q=gold OR XAUUSD&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    try:
        resp = requests.get(url)
        data = resp.json()
        return data.get('articles', [])[:5]
    except Exception:
        return []

def analyze_news_sentiment(articles):
    sentiments = []
    for art in articles:
        text = (art.get('title', '') or '') + ". " + (art.get('description', '') or '')
        sentiment = TextBlob(text).sentiment.polarity
        sentiments.append(sentiment)
    return np.mean(sentiments) if sentiments else 0

async def fetch_telegram_signal():
    try:
        client = TelegramClient('session', TELEGRAM_API_ID, TELEGRAM_API_HASH)
        await client.start()
        channel = await client.get_entity(TELEGRAM_CHANNEL)
        messages = await client.get_messages(channel, limit=30)
        for msg in messages:
            text = msg.message.lower()
            if "gold buy now" in text:
                return "buy", msg.message, msg.date.strftime("%Y-%m-%d %H:%M:%S")
            elif "gold sell now" in text:
                return "sell", msg.message, msg.date.strftime("%Y-%m-%d %H:%M:%S")
        return "none", None, None
    except Exception as e:
        return "error", str(e), None

def get_telegram_signal_sync():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    signal, full_msg, dt = loop.run_until_complete(fetch_telegram_signal())
    loop.close()
    return signal, full_msg, dt

def classify_trade(rsi, macd_hist, news_sentiment, telegram_signal):
    if telegram_signal == "error":
        return "Error fetching Telegram signal"
    if telegram_signal == "buy" and rsi < 70 and macd_hist > 0 and news_sentiment > 0.1:
        return "Trade"
    if telegram_signal == "sell" and rsi > 30 and macd_hist < 0 and news_sentiment < -0.1:
        return "Trade"
    if telegram_signal == "none":
        if rsi < 30 and macd_hist > 0 and news_sentiment > 0.2:
            return "Trade"
        elif abs(news_sentiment) < 0.1:
            return "Risk"
        else:
            return "Don't Trade"
    return "Risk"

def plot_candles(df):
    fig = go.Figure(data=[go.Candlestick(
        x=df['datetime'],
        open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        increasing_line_color='green', decreasing_line_color='red'
    )])
    fig.update_layout(title="XAU/USD Weekly Candle Chart", xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig, use_container_width=True)

def log_signal(decision, telegram_msg, dt):
    import os
    import csv
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["timestamp", "decision", "telegram_msg"])
        writer.writerow([dt or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), decision, telegram_msg or ""])

def show_signal_history():
    import os
    if not os.path.isfile(LOG_FILE):
        st.info("No signal history found.")
        return
    df = pd.read_csv(LOG_FILE)
    if df.empty:
        st.info("Signal history is empty.")
        return
    st.subheader("Signal History")
    st.dataframe(df.sort_values(by="timestamp", ascending=False))

def generate_news_summary(articles):
    if not articles:
        return "No recent news available."
    combined_text = " ".join((art.get('title', '') + ". " + art.get('description', '') for art in articles))
    prompt = f"Summarize the following news about gold and XAU/USD in 3 concise bullet points:\n\n{combined_text}"
    try:
        completion = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100,
            temperature=0.5,
            n=1,
        )
        return completion.choices[0].text.strip()
    except Exception as e:
        return f"OpenAI API error: {e}"

# === Streamlit UI ===
st.title("🚀 XAU/USD AI Signal Bot")

# Live price
price = fetch_latest_xauusd_price()
if price:
    st.metric(label="Live XAU/USD Price (USD)", value=f"${price:.2f}")
else:
    st.warning("Failed to fetch live XAU/USD price")

# Weekly candle chart
chart_data = fetch_weekly_chart_data()
if chart_data.empty:
    st.error("Failed to fetch weekly chart data.")
    st.stop()
plot_candles(chart_data)

# Technical indicators
rsi, macd_hist = analyze_technical_indicators(chart_data)
st.write(f"**RSI:** {rsi:.2f}")
st.write(f"**MACD Histogram:** {macd_hist:.4f}")

# News sentiment (no headlines)
news_articles = fetch_news()
sentiment = analyze_news_sentiment(news_articles)
st.write(f"**News Sentiment Score:** {sentiment:.3f}")

# OpenAI news summary (optional, comment out if undesired)
with st.expander("AI Generated News Summary"):
    summary = generate_news_summary(news_articles)
    st.write(summary)

# Telegram Signal
telegram_signal, telegram_msg, telegram_dt = get_telegram_signal_sync()
st.subheader("Telegram Signal")
if telegram_signal == "error":
    st.error(f"Error fetching Telegram signal: {telegram_msg}")
elif telegram_signal == "none":
    st.info("No recent Gold Buy/Sell signal found in Telegram.")
else:
    st.write(f"**Signal:** {telegram_signal.capitalize()}")
    st.write(f"**Message:** {telegram_msg}")
    st.write(f"**Date:** {telegram_dt}")

# Final AI Trade Decision
decision = classify_trade(rsi, macd_hist, sentiment, telegram_signal)
st.header(f"AI Trade Decision: {decision}")

# Log decision
log_signal(decision, telegram_msg, telegram_dt)

# Show history checkbox
if st.checkbox("Show signal history"):
    show_signal_history()
