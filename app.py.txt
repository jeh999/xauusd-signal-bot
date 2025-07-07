import streamlit as st
import requests
from textblob import TextBlob
from datetime import datetime
import pandas as pd
import numpy as np
from telethon.sync import TelegramClient
from streamlit_autorefresh import st_autorefresh

# Auto-refresh every 60 seconds
st_autorefresh(interval=60000, limit=None, key="datarefresh")

# --- API Config ---
TRADINGVIEW_XAUUSD_FEED = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=15min&apikey={st.secrets['TWELVEDATA_API_KEY']}"
NEWS_API_URL = f"https://newsapi.org/v2/everything?q=gold+OR+XAUUSD&language=en&sortBy=publishedAt&apiKey={st.secrets['NEWS_API_KEY']}"

# Telegram setup
TELEGRAM_API_ID = int(st.secrets["TELEGRAM_API_ID"])
TELEGRAM_API_HASH = st.secrets["TELEGRAM_API_HASH"]
TELEGRAM_CHANNEL = 'gary_thetrader'  # No '@'

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
    return response.json().get('articles', [])[:5]

def analyze_news_sentiment(articles):
    scores = []
    for article in articles:
        content = article.get('title', '') + " " + article.get('description', '')
        polarity = TextBlob(content).sentiment.polarity
        scores.append(polarity)
    return np.mean(scores) if scores else 0

def get_latest_telegram_signal():
    try:
        client = TelegramClient('session_gary', TELEGRAM_API_ID, TELEGRAM_API_HASH)
        client.connect()
        if not client.is_user_authorized():
            return 'uncertain'
        channel = client.get_entity(TELEGRAM_CHANNEL)
        messages = client.get_messages(channel, limit=10)
        for msg in messages:
            content = msg.message.upper()
            if 'XAUUSD' in content:
                if 'BUY' in content:
                    return 'buy'
                elif 'SELL' in content:
                    return 'sell'
                elif 'WAIT' in content or 'AVOID' in content:
                    return 'uncertain'
        return 'uncertain'
    except Exception as e:
        st.warning(f"Failed to fetch Telegram signal: {e}")
        return 'uncertain'

def classify_signal(rsi, macd_hist, sentiment, telegram_signal):
    if rsi < 30 and macd_hist > 0 and sentiment > 0.2 and telegram_signal == 'buy':
        return "Trade"
    elif telegram_signal == "uncertain" or abs(sentiment) < 0.1:
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
sentiment = analyze_news_sentiment(articles)
st.write(f"**News Sentiment Score**: {sentiment:.3f}")

telegram_signal = get_latest_telegram_signal()
st.write(f"**Telegram Signal:** {telegram_signal.capitalize()}")

decision = classify_signal(indicators['RSI'], indicators['MACD_HIST'], sentiment, telegram_signal)

st.header(f"🚦 Trade Decision: {decision}")
if decision == "Trade":
    st.success("✅ Conditions look good for trading.")
elif decision == "Risk":
    st.warning("⚠️ Risky. Mixed or weak signals.")
else:
    st.error("❌ Avoid trading now.")
