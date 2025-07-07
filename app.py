import streamlit as st
import requests
from textblob import TextBlob
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from telethon import TelegramClient
from streamlit_autorefresh import st_autorefresh
import asyncio

# Auto-refresh every 60 seconds
st_autorefresh(interval=60000, limit=None, key="datarefresh")

# --- Configurations ---
TRADINGVIEW_XAUUSD_FEED = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=15min&apikey={st.secrets['TWELVEDATA_API_KEY']}"
NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
NEWS_API_URL = f"https://newsapi.org/v2/everything?q=gold+OR+XAUUSD&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"

# Telegram API credentials
TELEGRAM_API_ID = int(st.secrets["TELEGRAM_API_ID"])
TELEGRAM_API_HASH = st.secrets["TELEGRAM_API_HASH"]
TELEGRAM_CHANNEL = 'gary_thetrader'  # Without @

# --- Functions ---

def fetch_chart_data():
    response = requests.get(TRADINGVIEW_XAUUSD_FEED)
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
    articles = response.json().get('articles', [])[:5]
    return articles

def analyze_news_sentiment(articles):
    sentiment_scores = []
    for article in articles:
        content = article.get('title', '') + ". " + article.get('description', '')
        sentiment = TextBlob(content).sentiment.polarity
        sentiment_scores.append(sentiment)
    return np.mean(sentiment_scores) if sentiment_scores else 0

# --- Telegram Signal Fetching with Event Loop Fix ---
async def fetch_telegram_signal():
    try:
        client = TelegramClient('session_gary', TELEGRAM_API_ID, TELEGRAM_API_HASH)
        await client.start()
        channel = await client.get_entity(TELEGRAM_CHANNEL)
        messages = await client.get_messages(channel, limit=10)
        for message in messages:
            msg = message.message.upper()
            if 'XAUUSD' in msg:
                if 'BUY' in msg:
                    return 'buy'
                elif 'SELL' in msg:
                    return 'sell'
                elif 'WAIT' in msg or 'AVOID' in msg:
                    return 'uncertain'
        return 'uncertain'
    except Exception as e:
        return f"error: {e}"

def get_latest_telegram_signal():
    loop = asyncio.new_event_loop()  # Create a new event loop
    asyncio.set_event_loop(loop)  # Set it as the current event loop
    signal = loop.run_until_complete(fetch_telegram_signal())  # Run the async function
    loop.close()  # Close the loop after use
    return signal

# --- Signal Classification ---
def classify_signal(rsi, macd_hist, news_sentiment, telegram_signal):
    if rsi < 30 and macd_hist > 0 and news_sentiment > 0.2 and telegram_signal == "buy":
        return "Trade"
    elif telegram_signal == "uncertain" or abs(news_sentiment) < 0.1:
        return "Risk"
    else:
        return "Don't Trade"

# --- Streamlit UI ---
st.title("📈 XAU/USD AI Signal Bot")

chart_data = fetch_chart_data()
if chart_data.empty:
    st.error("❌ Failed to fetch chart data.")
    st.stop()

st.subheader("Live Price (15-min)")
st.line_chart(chart_data.set_index('date')['close'])

indicators = analyze_technical_indicators(chart_data)
st.write(f"**RSI**: {indicators['RSI']:.2f}")
st.write(f"**MACD Histogram**: {indicators['MACD_HIST']:.4f}")

articles = fetch_news()
sentiment_score = analyze_news_sentiment(articles)
st.write(f"**News Sentiment Score**: {sentiment_score:.3f}")

telegram_signal = get_latest_telegram_signal()
st.write(f"**Telegram Signal:** {telegram_signal.capitalize()}")

signal = classify_signal(indicators['RSI'], indicators['MACD_HIST'], sentiment_score, telegram_signal)

st.header(f"🚦 Trade Decision: {signal}")
if signal == "Trade":
    st.success("✅ Conditions look good for trading.")
elif signal == "Risk":
    st.warning("⚠️ Risky. Mixed or weak signals.")
else:
    st.error("❌ Avoid trading now.")
