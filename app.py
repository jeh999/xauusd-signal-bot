import streamlit as st
import requests
import pandas as pd
import numpy as np
from telethon import TelegramClient
from streamlit_autorefresh import st_autorefresh
import asyncio
import plotly.graph_objects as go
import re
from datetime import datetime, timezone
import openai
import os
from textblob import TextBlob

# --- CONFIG --- #

st.set_page_config(page_title="XAU/USD AI Signal Bot Plus", layout="wide")

# Auto-refresh every 60 seconds
st_autorefresh(interval=60000, limit=None, key="datarefresh")

# Load secrets
TWELVEDATA_API_KEY = st.secrets["TWELVEDATA_API_KEY"]
TELEGRAM_API_ID = int(st.secrets["TELEGRAM_API_ID"])
TELEGRAM_API_HASH = st.secrets["TELEGRAM_API_HASH"]
TELEGRAM_CHANNEL = 'Gary_TheTrader'  # without @
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
openai.api_key = OPENAI_API_KEY

LOG_FILE = "signal_log.csv"

# API URLs
TWELVEDATA_WEEKLY = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1week&apikey={TWELVEDATA_API_KEY}"
TWELVEDATA_HOURLY = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&apikey={TWELVEDATA_API_KEY}"
NEWS_API_URL = f"https://newsapi.org/v2/everything?q=gold+OR+XAUUSD&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"

# --- FUNCTIONS --- #

def fetch_chart_data(url):
    res = requests.get(url)
    data = res.json()
    if 'values' not in data:
        return pd.DataFrame()
    df = pd.DataFrame(data['values'])
    df.rename(columns={"datetime": "date"}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(float)
    return df

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series):
    exp12 = series.ewm(span=12, adjust=False).mean()
    exp26 = series.ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - signal
    return macd_hist

def calculate_atr(df, period=14):
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    atr = df['TR'].rolling(window=period).mean().iloc[-1]
    return atr

def analyze_technical_indicators(df):
    df['RSI'] = calculate_rsi(df['close'])
    df['MACD_HIST'] = calculate_macd(df['close'])
    last = df.iloc[-1]
    return last['RSI'], last['MACD_HIST']

async def fetch_telegram_signal():
    try:
        client = TelegramClient('session_gary', TELEGRAM_API_ID, TELEGRAM_API_HASH)
        await client.start()
        channel = await client.get_entity(TELEGRAM_CHANNEL)
        messages = await client.get_messages(channel, limit=50)
        for msg in messages:
            if msg.message:
                text = msg.message.upper()
                if "GOLD BUY NOW" in text or "GOLD SELL NOW" in text:
                    price_match = re.search(r'@ ?([\d.]+)', text)
                    price = float(price_match.group(1)) if price_match else None
                    timestamp = msg.date.replace(tzinfo=timezone.utc)
                    await client.disconnect()
                    signal = 'buy' if "BUY" in text else 'sell'
                    return text, signal, price, timestamp
        await client.disconnect()
        return None, 'uncertain', None, None
    except Exception as e:
        return f"Error: {e}", 'error', None, None

def get_latest_telegram_signal():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    msg, sig, price, time = loop.run_until_complete(fetch_telegram_signal())
    loop.close()
    return msg, sig, price, time

def fetch_news():
    response = requests.get(NEWS_API_URL)
    articles = response.json().get('articles', [])[:10]
    return articles

def analyze_news_sentiment(articles):
    scores = []
    for art in articles:
        content = art.get('title', '') + ". " + art.get('description', '')
        blob = TextBlob(content)
        scores.append(blob.sentiment.polarity)
    return np.mean(scores) if scores else 0

def detect_news_events(articles):
    events_keywords = ['geopolitical', 'inflation', 'FED', 'war', 'sanction', 'conflict', 'crisis', 'demand', 'supply']
    event_score = 0
    for art in articles:
        text = (art.get('title','') + " " + art.get('description','')).lower()
        for kw in events_keywords:
            if kw in text:
                event_score += 1
    return min(event_score / 5, 1)  # Normalize 0 to 1

def simple_lstm_predict_stub(df):
    # Placeholder: you can integrate your trained model here
    # Return price movement prediction between -1 and 1 (neg to pos)
    return np.random.uniform(-1, 1)

def classify_signal(rsi, macd_hist, telegram_signal, news_sentiment, news_event_score, lstm_pred, atr, risk_tolerance):
    score = 50  # start at neutral 50

    # RSI & MACD
    if rsi < 30:
        score += 10
    elif rsi > 70:
        score -= 10

    if macd_hist > 0:
        score += 10
    else:
        score -= 10

    # Telegram signal
    if telegram_signal == 'buy':
        score += 15
    elif telegram_signal == 'sell':
        score -= 15

    # News sentiment and event
    score += news_sentiment * 20  # positive sentiment adds score
    score += news_event_score * 10  # relevant news adds confidence

    # LSTM prediction
    score += lstm_pred * 15

    # Adjust by ATR and risk tolerance (0 to 1)
    if atr > 0:
        volatility_factor = min(atr / 50, 1)  # Normalize ATR (example)
        score -= volatility_factor * 10 * (1 - risk_tolerance)

    # Clamp
    score = max(0, min(100, score))

    if score >= 70:
        return "Trade", score
    elif score >= 40:
        return "Risk", score
    else:
        return "Don't Trade", score

def log_signal(signal, rsi, macd, price, decision, confidence):
    now = datetime.utcnow().isoformat()
    row = pd.DataFrame([[now, signal, round(rsi,2), round(macd,4), price, decision, confidence]],
                       columns=["timestamp", "signal", "RSI", "MACD", "entry_price", "decision", "confidence"])
    if os.path.exists(LOG_FILE):
        row.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        row.to_csv(LOG_FILE, index=False)

def show_signal_history():
    if not os.path.exists(LOG_FILE):
        st.info("No signal history yet.")
        return
    try:
        df = pd.read_csv(LOG_FILE, on_bad_lines='skip')
        if df.empty:
            st.info("Signal history is empty.")
            return
        st.subheader("📜 Signal History (Last 10)")
        st.dataframe(df.tail(10))
    except Exception as e:
        st.error(f"Error reading signal history: {e}")

def gpt_validate_signal(message):
    if not message or message.startswith("Error") or message == "None":
        return "No valid message for GPT analysis."
    prompt = (
        "You are an expert trading assistant.\n"
        "Analyze this Telegram message about Gold trading:\n"
        f"\"\"\"\n{message}\n\"\"\"\n"
        "Is this a clear, reliable trading signal? Provide a short rationale."
    )
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"GPT API error: {e}"

# --- UI --- #

st.title("🚀 XAU/USD AI Signal Bot Plus")

# User risk tolerance slider
risk_tolerance = st.slider("Select your risk tolerance (0 = very conservative, 1 = very aggressive):", 0.0, 1.0, 0.5)

# Fetch charts
weekly_df = fetch_chart_data(TWELVEDATA_WEEKLY)
hourly_df = fetch_chart_data(TWELVEDATA_HOURLY)

if weekly_df.empty or hourly_df.empty:
    st.error("Failed to fetch price data.")
    st.stop()

# Weekly candlestick chart
st.subheader("Weekly Candlestick Chart")
fig = go.Figure(data=[go.Candlestick(
    x=weekly_df['date'],
    open=weekly_df['open'], high=weekly_df['high'],
    low=weekly_df['low'], close=weekly_df['close']
)])
fig.update_layout(xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# Hourly indicators
rsi, macd_hist = analyze_technical_indicators(hourly_df)
st.write(f"**RSI (1H):** {rsi:.2f}")
st.write(f"**MACD Histogram (1H):** {macd_hist:.4f}")

# ATR for risk management
atr = calculate_atr(hourly_df)
st.write(f"**ATR (1H):** {atr:.4f}")

# Telegram signal
message, signal, price, signal_time = get_latest_telegram_signal()
if signal == 'error':
    st.error(message)
elif signal == 'uncertain':
    st.warning("No recent 'Gold Buy now' or 'Gold Sell now' signal found in Telegram channel.")
else:
    minutes_ago = int((datetime.utcnow().replace(tzinfo=timezone.utc) - signal_time).total_seconds() / 60)
    st.success(f"📢 Telegram Signal: **{signal.upper()}** at price: {price if price else 'Unknown'}")
    st.info(f"Signal time: {signal_time.strftime('%Y-%m-%d %H:%M:%S UTC')} ({minutes_ago} minutes ago)")
    st.code(message)

# News
news_articles = fetch_news()
news_sentiment = analyze_news_sentiment(news_articles)
news_event_score = detect_news_events(news_articles)
st.write(f"**News Sentiment:** {news_sentiment:.3f}")
st.write(f"**News Event Score:** {news_event_score:.2f}")

# LSTM prediction stub
lstm_pred = simple_lstm_predict_stub(hourly_df)
st.write(f"**LSTM Price Prediction (stub):** {lstm_pred:.3f} (pos = up, neg = down)")

# Classification
decision, confidence = classify_signal(rsi, macd_hist, signal, news_sentiment, news_event_score, lstm_pred, atr, risk_tolerance)
st.header(f"🚦 Trade Decision: {decision} (Confidence: {confidence:.1f}%)")
st.progress(confidence / 100)

# GPT validation
if message and not message.startswith("Error") and signal != 'uncertain':
    st.subheader("GPT-4o-mini Signal Validation & Rationale")
    gpt_result = gpt_validate_signal(message)
    st.write(gpt_result)

# Log
if signal not in ['error', 'uncertain']:
    log_signal(signal, rsi, macd_hist, price, decision, confidence)

show_signal_history()
