import streamlit as st
import requests
from textblob import TextBlob
import pandas as pd
import numpy as np
from telethon import TelegramClient
from streamlit_autorefresh import st_autorefresh
import asyncio
import plotly.graph_objects as go

# Auto-refresh every 60 seconds
st_autorefresh(interval=60000, limit=None, key="datarefresh")

# --- Configurations ---
TRADINGVIEW_XAUUSD_FEED = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1week&apikey={st.secrets['TWELVEDATA_API_KEY']}"
NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
NEWS_API_URL = f"https://newsapi.org/v2/everything?q=gold+OR+XAUUSD&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"

# Telegram API credentials
TELEGRAM_API_ID = int(st.secrets["TELEGRAM_API_ID"])
TELEGRAM_API_HASH = st.secrets["TELEGRAM_API_HASH"]
TELEGRAM_CHANNEL = 'Gary_TheTrader'  # Channel username without @

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

# --- Telegram Signal Fetching (last 50 messages, newest first) ---
async def fetch_telegram_signal():
    try:
        client = TelegramClient('session_gary', TELEGRAM_API_ID, TELEGRAM_API_HASH)
        await client.start()

        channel = await client.get_entity(TELEGRAM_CHANNEL)
        messages = await client.get_messages(channel, limit=50)

        for message in messages:
            if message.message:
                msg = message.message.upper()

                if 'GOLD BUY NOW' in msg:
                    await client.disconnect()
                    return 'buy', message.sender_id, message.message
                elif 'GOLD SELL NOW' in msg:
                    await client.disconnect()
                    return 'sell', message.sender_id, message.message

        await client.disconnect()
        return 'uncertain', None, None
    except Exception as e:
        return f"error: {e}", None, None

def get_latest_telegram_signal():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    signal, sender, message = loop.run_until_complete(fetch_telegram_signal())
    loop.close()
    return signal, sender, message

def classify_signal(rsi, macd_hist, news_sentiment, telegram_signal):
    if rsi < 30 and macd_hist > 0 and news_sentiment > 0.2 and telegram_signal == "buy":
        return "Trade"
    elif telegram_signal == "uncertain" or abs(news_sentiment) < 0.1:
        return "Risk"
    else:
        return "Don't Trade"

# --- Streamlit UI ---
st.title("📈 XAU/USD AI Signal Bot")

# Fetch Chart Data
chart_data = fetch_chart_data()
if chart_data.empty:
    st.error("❌ Failed to fetch chart data.")
    st.stop()

# Candlestick chart - weekly
fig = go.Figure(data=[go.Candlestick(
    x=chart_data['date'],
    open=chart_data['open'],
    high=chart_data['high'],
    low=chart_data['low'],
    close=chart_data['close'],
    name="Candlestick"
)])
fig.update_layout(
    title="XAU/USD Candlestick Chart (Weekly)",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    xaxis_rangeslider_visible=False
)
st.plotly_chart(fig)

# Technical Indicators
indicators = analyze_technical_indicators(chart_data)
st.write(f"**RSI**: {indicators['RSI']:.2f}")
st.write(f"**MACD Histogram**: {indicators['MACD_HIST']:.4f}")

# Fetch News Sentiment
articles = fetch_news()
sentiment_score = analyze_news_sentiment(articles)
st.write(f"**News Sentiment Score**: {sentiment_score:.3f}")

# Fetch Telegram Signal and show only if 'Gold Buy now' or 'Gold Sell now' found
telegram_signal, sender, telegram_message = get_latest_telegram_signal()

if telegram_signal.startswith("error"):
    st.error(f"Telegram Signal Error: {telegram_signal}")
elif telegram_signal == 'uncertain' or telegram_message is None:
    st.warning("❗️ No 'Gold Buy now' or 'Gold Sell now' signal found in recent messages.")
else:
    st.write(f"**Telegram Signal:** {telegram_signal.capitalize()}")
    st.write(f"**Telegram Message:** {telegram_message}")

# Final trade decision
signal = classify_signal(indicators['RSI'], indicators['MACD_HIST'], sentiment_score, telegram_signal)
st.header(f"🚦 Trade Decision: {signal}")

if signal == "Trade":
    st.success("✅ Conditions look good for trading.")
elif signal == "Risk":
    st.warning("⚠️ Risky. Mixed or weak signals.")
else:
    st.error("❌ Avoid trading now.")
