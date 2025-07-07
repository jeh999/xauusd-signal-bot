import streamlit as st
import requests
from textblob import TextBlob
import pandas as pd
import numpy as np
from telethon import TelegramClient
from streamlit_autorefresh import st_autorefresh
import asyncio
import plotly.graph_objects as go
import re
import os
from datetime import datetime

st_autorefresh(interval=60000, limit=None, key="datarefresh")

# --- Configurations ---
TRADINGVIEW_XAUUSD_FEED_WEEKLY = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1week&apikey={st.secrets['TWELVEDATA_API_KEY']}"
TRADINGVIEW_XAUUSD_FEED_HOURLY = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&apikey={st.secrets['TWELVEDATA_API_KEY']}"
NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
NEWS_API_URL = f"https://newsapi.org/v2/everything?q=gold+OR+XAUUSD&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"

TELEGRAM_API_ID = int(st.secrets["TELEGRAM_API_ID"])
TELEGRAM_API_HASH = st.secrets["TELEGRAM_API_HASH"]
TELEGRAM_CHANNEL = 'Gary_TheTrader'

log_file = "signal_log.csv"

# --- Functions ---
def fetch_chart_data(url):
    response = requests.get(url)
    data = response.json()
    if 'values' not in data:
        return pd.DataFrame()
    df = pd.DataFrame(data['values'])
    df = df.rename(columns={"datetime": "date", "close": "close", "open": "open", "high": "high", "low": "low"})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(float)
    return df

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

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
    return {
        'RSI': last['RSI'],
        'MACD_HIST': last['MACD_HIST']
    }

def fetch_news():
    response = requests.get(NEWS_API_URL)
    return response.json().get('articles', [])[:5]

def analyze_news_sentiment(articles):
    sentiment_scores = []
    for article in articles:
        content = article.get('title', '') + ". " + article.get('description', '')
        sentiment = TextBlob(content).sentiment.polarity
        sentiment_scores.append(sentiment)
    return np.mean(sentiment_scores) if sentiment_scores else 0

async def fetch_telegram_signal():
    try:
        client = TelegramClient('session_gary', TELEGRAM_API_ID, TELEGRAM_API_HASH)
        await client.start()
        channel = await client.get_entity(TELEGRAM_CHANNEL)
        messages = await client.get_messages(channel, limit=50)

        for message in messages:
            if message.message:
                msg = message.message.upper()
                if 'GOLD BUY NOW' in msg or 'GOLD SELL NOW' in msg:
                    price = None
                    match = re.search(r"@ ?([\d.]+)", msg)
                    if match:
                        price = float(match.group(1))
                    await client.disconnect()
                    return msg, 'buy' if 'BUY' in msg else 'sell', price

        await client.disconnect()
        return None, 'uncertain', None
    except Exception as e:
        return f"error: {e}", 'error', None

def get_latest_telegram_signal():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    message, signal, price = loop.run_until_complete(fetch_telegram_signal())
    loop.close()
    return message, signal, price

def classify_signal(rsi, macd_hist, sentiment, telegram_signal):
    score = 0
    if rsi < 30: score += 25
    if macd_hist > 0: score += 25
    if sentiment > 0.2: score += 25
    if telegram_signal == "buy": score += 25
    elif telegram_signal == "sell": score += 25
    if telegram_signal == 'uncertain': return "Risk", score
    return ("Trade" if score >= 75 else "Don't Trade"), score

def log_signal(signal, rsi, macd, sentiment, price):
    now = datetime.utcnow().isoformat()
    row = pd.DataFrame([[now, signal, round(rsi,2), round(macd,4), round(sentiment,3), price]],
                       columns=["timestamp", "signal", "RSI", "MACD", "sentiment", "entry_price"])
    if os.path.exists(log_file):
        row.to_csv(log_file, mode='a', header=False, index=False)
    else:
        row.to_csv(log_file, index=False)

def show_signal_history():
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        st.subheader("📜 Signal History")
        st.dataframe(df.tail(10))

# --- UI ---
st.title("📈 XAU/USD AI Signal Bot")

chart_week = fetch_chart_data(TRADINGVIEW_XAUUSD_FEED_WEEKLY)
chart_hour = fetch_chart_data(TRADINGVIEW_XAUUSD_FEED_HOURLY)
if chart_week.empty or chart_hour.empty:
    st.error("❌ Failed to fetch chart data.")
    st.stop()

st.subheader("Weekly Candlestick Chart")
fig = go.Figure(data=[go.Candlestick(x=chart_week['date'],
                                     open=chart_week['open'], high=chart_week['high'],
                                     low=chart_week['low'], close=chart_week['close'])])
fig.update_layout(xaxis_rangeslider_visible=False)
st.plotly_chart(fig)

indicators = analyze_technical_indicators(chart_hour)
st.write(f"**RSI (1H):** {indicators['RSI']:.2f}")
st.write(f"**MACD Histogram (1H):** {indicators['MACD_HIST']:.4f}")

news = fetch_news()
sentiment_score = analyze_news_sentiment(news)
st.write(f"**News Sentiment Score:** {sentiment_score:.3f}")

message, signal, entry_price = get_latest_telegram_signal()
if signal == 'error':
    st.error(f"Telegram error: {message}")
elif signal == 'uncertain':
    st.warning("❗️ No 'Gold Buy now' or 'Gold Sell now' signal found in recent messages.")
else:
    st.success(f"📢 Telegram Signal: {signal.upper()} @ {entry_price if entry_price else 'Unknown'}")
    st.code(message)

decision, confidence = classify_signal(indicators['RSI'], indicators['MACD_HIST'], sentiment_score, signal)
st.header(f"🚦 Trade Decision: {decision}")
st.progress(confidence / 100)

log_signal(signal, indicators['RSI'], indicators['MACD_HIST'], sentiment_score, entry_price)
show_signal_history()

st.markdown("---")
st.subheader("📰 Top Headlines")
for article in news:
    st.markdown(f"- [{article['title']}]({article['url']})")
