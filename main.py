# -*- coding: utf-8 -*-
"""
Scanner (Yahoo Finance)
Frames: 30m, 1H, 1D
Symbols chuẩn:
  - Gold (GC=F), WTI (CL=F), Bitcoin (BTC-USD), EUR/USD (EURUSD=X), USD/JPY (JPY=X)
Logic: EMA20/50/200 + RSI14 + MACD + ADX + BBWidth
Output per symbol to Telegram:
===SYMBOL===
30m-1H: <status or Mixed(...)>
1H: <status>
1D: <status>

ENV: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import requests

# ---------- CONFIG ----------
ASSETS = {
    "XAU/USD (Gold)": "GC=F",
    "WTI Oil": "CL=F",
    "Bitcoin": "BTC-USD",
    "EUR/USD": "EURUSD=X",
    "USD/JPY": "JPY=X",
}

# khung dùng trực tiếp từ Yahoo
INTRADAY_PERIOD = "60d"  # giới hạn intraday của Yahoo
ADX_TREND = 20           # lọc sideway
SEND_TELE = True

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

# ---------- INDICATORS ----------
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(close, n=14):
    d = close.diff()
    gain, loss = d.clip(lower=0), (-d).clip(lower=0)
    ag = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    al = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(close, fast=12, slow=26, sig=9):
    m = ema(close, fast) - ema(close, slow)
    s = ema(m, sig)
    return m, s, m - s

def true_range(df):
    pc = df["Close"].shift(1)
    return pd.concat([(df["High"]-df["Low"]),
                      (df["High"]-pc).abs(),
                      (df["Low"]-pc).abs()], axis=1).max(axis=1)

def adx(df, n=14):
    up, dn = df["High"].diff(), -df["Low"].diff()
    plus  = up.where((up > dn) & (up > 0), 0.0)
    minus = dn.where((dn > up) & (dn > 0), 0.0)
    trn = true_range(df).rolling(n).sum()
    pdi = 100 * (plus.rolling(n).sum() / trn)
    mdi = 100 * (minus.rolling(n).sum() / trn)
    dx  = ((pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)) * 100
    return dx.rolling(n).mean()

def bb_width(close, n=20):
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    return ((ma + 2*sd) - (ma - 2*sd)) / ma

def decorate(df):
    out = df.copy()
    out["EMA20"]  = ema(out["Close"], 20)
    out["EMA50"]  = ema(out["Close"], 50)
    out["EMA200"] = ema(out["Close"], 200)
    out["RSI14"]  = rsi(out["Close"], 14)
    out["MACD"], out["MACD_SIG"], out["MACD_HIST"] = macd(out["Close"])
    out["ADX14"]  = adx(out, 14)
    out["BBWIDTH"] = bb_width(out["Close"], 20)
    return out

def decide(df):
    last = df.iloc[-1]
    trending = (last["ADX14"] > ADX_TREND) and \
               (last["BBWIDTH"] > df["BBWIDTH"].rolling(50).mean().iloc[-1])
    if (last["Close"] > last["EMA20"] > last["EMA50"] > last["EMA200"]
        and last["RSI14"] > 55
        and last["MACD"] > last["MACD_SIG"]
        and trending):
        return "Long".upper()
    if (last["Close"] < last["EMA20"] < last["EMA50"] < last["EMA200"]
        and last["RSI14"] < 45
        and last["MACD"] < last["MACD_SIG"]
        and trending):
        return "Short".upper()
    return "Sideway".upper()

# ---------- DATA ----------
def fetch(symbol, interval, period):
    df = yf.download(symbol, interval=interval, period=period,
                     auto_adjust=False, progress=False)
    if df is None or df.empty or not {"Open","High","Low","Close"}.issubset(df.columns):
        raise RuntimeError(f"No data for {symbol} @ {interval}/{period}")
    df = df.dropna().copy()
    df.index = pd.to_datetime(df.index)
    return df

# ---------- TELEGRAM ----------
def send_tele(text):
    if not SEND_TELE: return
    if not BOT_TOKEN or not CHAT_ID:
        print("⚠️ Missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID"); return
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text},
            timeout=20
        )
    except Exception as e:
        print("Telegram error:", e)

# ---------- ANALYZE ONE ----------
def analyze_one(name, ticker):
    # 30m & 1h (<=60d), 1d (5y)
    d30 = decorate(fetch(ticker, "30m", INTRADAY_PERIOD))
    d1h = decorate(fetch(ticker, "1h",  INTRADAY_PERIOD))
    d1d = decorate(fetch(ticker, "1d",  "5y"))

    s30 = decide(d30) if len(d30) >= 100 else "N/A"
    s1h = decide(d1h) if len(d1h) >= 100 else "N/A"
    s1d = decide(d1d) if len(d1d) >= 200 else "N/A"

    # tổng hợp short-term (30m-1H)
    if s30 == s1h and s30 not in ("N/A",):
        short_line = s30
    else:
        short_line = f"Mixed (30m:{s30}, 1H:{s1h})"

    msg = (
        f"==={name}===\n"
        f"30m-1H: {short_line}\n"
        f"1H: {s1h}\n"
        f"1D: {s1d}"
    )
    print(msg)
    return msg

# ---------- MAIN ----------
def main():
    reports = []
    for name, ticker in ASSETS.items():
        try:
            reports.append(analyze_one(name, ticker))
        except Exception as e:
            reports.append(f"==={name}===\nLỗi: {e}")
    send_tele("\n\n".join(reports))

if __name__ == "__main__":
    main()
