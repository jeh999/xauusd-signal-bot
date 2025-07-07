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

# Auto-refresh every 60 seconds
st_autorefresh(interval=60000, limit=None, key="datarefresh")

# Load API keys from Streamlit secrets
TWELVEDATA_API_KEY = st.secrets["TWELVEDATA_API_KEY"]
TELEGRAM_API_ID = int(st.secrets["TELEGRAM_API_ID"])
TELEGRAM_API_HASH = st.secrets["TELEGRAM_API_HASH"]
TELEGRAM_CHANNEL = 'Gary_TheTrader'  # without @
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY

# API URLs
TWELVEDATA_WEEKLY = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1week&apikey={TWELVEDATA_API_KEY}"
TWELVEDATA_HOURLY = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&apikey={TWELVEDATA_API_KEY}"

# Log file for signals
LOG_FILE = "signal_log.csv"

# --- Technical analysis functions ---

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

def analyze_technical_indicators(df):
    df['RSI'] = calculate_rsi(df['close'])
    df['MACD_HIST'] = calculate_macd(df['close'])
    last = df.iloc[-1]
    return last['RSI'], last['MACD_HIST']

# --- Telegram functions ---

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

# --- OpenAI GPT validation ---

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
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"GPT API error: {e}"

# --- Classification & Confidence ---

def classify_signal(rsi, macd_hist, telegram_signal):
    score = 0
    rationale = []

    # RSI contribution
    if rsi < 30:
        score += 30
        rationale.append("RSI < 30 suggests oversold (bullish) conditions.")
    elif rsi > 70:
        score -= 30
        rationale.append("RSI > 70 suggests overbought (bearish) conditions.")

    # MACD contribution
    if macd_hist > 0:
        score += 30
        rationale.append("Positive MACD histogram indicates bullish momentum.")
    else:
        score -= 30
        rationale.append("Negative MACD histogram indicates bearish momentum.")

    # Telegram signal contribution
    if telegram_signal == 'buy':
        score += 40
        rationale.append("Telegram signal indicates BUY.")
    elif telegram_signal == 'sell':
        score -= 40
        rationale.append("Telegram signal indicates SELL.")
    else:
        rationale.append("No clear Telegram trading signal.")

    # Normalize score 0-100
    score = max(0, min(100, score + 50))

    # Decision threshold
    if score >= 70:
        decision = "Trade"
    elif score >= 40:
        decision = "Risk"
    else:
        decision = "Don't Trade"

    return decision, score, rationale

# --- Logging ---

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
        df = pd.read_csv(LOG_FILE)
        if df.empty:
            st.info("Signal history is empty.")
            return
        st.subheader("📜 Signal History (Last 10)")
        st.dataframe(df.tail(10))
    except pd.errors.EmptyDataError:
        st.info("Signal history file is empty.")
    except Exception as e:
        st.error(f"Error reading signal history: {e}")

# --- Streamlit UI ---

st.title("📈 XAU/USD AI Signal Bot with GPT Validation")

# Fetch data
weekly_df = fetch_chart_data(TWELVEDATA_WEEKLY)
hourly_df = fetch_chart_data(TWELVEDATA_HOURLY)

if weekly_df.empty or hourly_df.empty:
    st.error("Failed to fetch price data. Please check API keys and internet connection.")
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

# Indicators on hourly data
rsi, macd_hist = analyze_technical_indicators(hourly_df)
st.write(f"**RSI (1H):** {rsi:.2f}")
st.write(f"**MACD Histogram (1H):** {macd_hist:.4f}")

# Get Telegram signal
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

    # Classify and get confidence
    decision, confidence, rationale = classify_signal(rsi, macd_hist, signal)

    st.header(f"🚦 Trade Decision: {decision} (Confidence: {confidence}%)")

    st.subheader("Decision Rationale")
    for line in rationale:
        st.write("- " + line)

    # GPT validation
    st.subheader("GPT-4o-mini Signal Validation & Rationale")
    gpt_result = gpt_validate_signal(message)
    st.write(gpt_result)

    # Confidence bar
    st.progress(confidence / 100)

    # Log signal for history
    log_signal(signal, rsi, macd_hist, price, decision, confidence)
    show_signal_history()
