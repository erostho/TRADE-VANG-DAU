# -*- coding: utf-8 -*-
"""
main.py - Phân tích đa khung Vàng, Dầu, BTC, EUR/USD, USD/JPY
Khung: 30m, 1H, 6H, 12H, 1D
Logic nâng cao: EMA/RSI/MACD/ADX + BBW + ATR (RR check)
Kết quả: LONG / SHORT / SIDEWAY
Gửi kết quả về Telegram
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime

# ========= CONFIG =========
ASSETS = {
    "XAU/USD": "XAUUSD=X",   # Vàng
    "WTI Oil": "CL=F",       # Dầu WTI
    "Bitcoin": "BTC-USD",    # BTC
    "EUR/USD": "EURUSD=X",   # EURUSD
    "USD/JPY": "JPY=X"       # USDJPY
}

TIMEFRAMES = {
    "30m": "30m",
    "1H": "1h",
    "6H": "6h",
    "12H": "12h",
    "1D": "1d"
}

PERIOD = "90d"  # đủ dữ liệu cho khung lớn
ADX_TREND = 20
RR_MIN = 1.5

# Telegram config (điền bot token + chat id của bạn)
BOT_TOKEN = os.getenv("TELE_BOT_TOKEN")
CHAT_ID   = os.getenv("TELE_CHAT_ID")

# ========= INDICATORS =========
def ema(series, n): return series.ewm(span=n, adjust=False).mean()

def rsi(series, n=14):
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/n, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100/(1+rs))

def macd(series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_ln = ema(macd_line, signal)
    hist = macd_line - signal_ln
    return macd_line, signal_ln, hist

def true_range(df):
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(df, n=14): return true_range(df).rolling(n).mean()

def adx(df, n=14):
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range(df)
    tr_n = tr.rolling(n).sum()
    plus_di = 100 * (pd.Series(plus_dm).rolling(n).sum() / tr_n)
    minus_di = 100 * (pd.Series(minus_dm).rolling(n).sum() / tr_n)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.rolling(n).mean()

def bb_width(series, n=20):
    ma = series.rolling(n).mean()
    sd = series.rolling(n).std(ddof=0)
    upper, lower = ma + 2*sd, ma - 2*sd
    return (upper - lower) / ma

# ========= ANALYSIS =========
def analyze_signal(df):
    last = df.iloc[-1]
    signal = "SIDEWAY"

    if (last["Close"] > last["EMA20"] > last["EMA50"] > last["EMA200"]
        and last["RSI14"] > 55 and last["MACD"] > last["MACD_SIG"]
        and last["ADX14"] > ADX_TREND and last["BBWIDTH"] > df["BBWIDTH"].mean()):
        signal = "LONG"
    elif (last["Close"] < last["EMA20"] < last["EMA50"] < last["EMA200"]
          and last["RSI14"] < 45 and last["MACD"] < last["MACD_SIG"]
          and last["ADX14"] > ADX_TREND and last["BBWIDTH"] > df["BBWIDTH"].mean()):
        signal = "SHORT"

    return signal

def analyze_asset(symbol, ticker):
    print(f"\n====== {symbol} ======")
    results = {}

    for tf_name, interval in TIMEFRAMES.items():
        data = yf.download(ticker, interval=interval, period=PERIOD, progress=False)
        if len(data) < 100:
            results[tf_name] = "N/A"
            continue

        df = data.copy()
        df["EMA20"] = ema(df["Close"], 20)
        df["EMA50"] = ema(df["Close"], 50)
        df["EMA200"] = ema(df["Close"], 200)
        df["RSI14"] = rsi(df["Close"], 14)
        df["MACD"], df["MACD_SIG"], df["MACD_HIST"] = macd(df["Close"])
        df["ADX14"] = adx(df, 14)
        df["ATR14"] = atr(df, 14)
        df["BBWIDTH"] = bb_width(df["Close"], 20)

        results[tf_name] = analyze_signal(df)

    # Gom nhóm timeframe
    short_tf = f"{results.get('30m')} / {results.get('1H')}"
    mid_tf   = f"{results.get('6H')} / {results.get('12H')}"
    long_tf  = results.get("1D")

    msg = (
        f"====== {symbol} ======\n"
        f"Ngắn (30m-1H): {short_tf}\n"
        f"Trung (6H-12H): {mid_tf}\n"
        f"Dài (1D): {long_tf}\n"
        f"👉 Kết luận (1H): {results.get('1H')}"
    )
    print(msg)
    return msg

# ========= TELEGRAM =========
def send_telegram(msg):
    if not BOT_TOKEN or not CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": CHAT_ID, "text": msg}, timeout=15)
    except Exception as e:
        print("Telegram error:", e)

# ========= MAIN =========
def main():
    all_msgs = []
    for sym, tick in ASSETS.items():
        try:
            msg = analyze_asset(sym, tick)
            all_msgs.append(msg)
        except Exception as e:
            print(f"Lỗi {sym}: {e}")
    final_report = "\n\n".join(all_msgs)
    send_telegram(final_report)

if __name__ == "__main__":
    main()
