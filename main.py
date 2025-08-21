# -*- coding: utf-8 -*-
"""
Scanner Yahoo Finance
Khung: 30m, 1H (native), 3H (resample từ 1H), 1D (native)
Tài sản: XAU/USD, WTI Oil, BTC, EUR/USD, USD/JPY
Logic: EMA20/50/200 + RSI14 + MACD + ADX + BBWidth
Kết quả: LONG / SHORT / SIDEWAY
ENV: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
"""

import os, numpy as np, pandas as pd, yfinance as yf, requests

# -------- Config --------
ASSETS = {
    "XAU/USD": {"intraday": ["XAUUSD=X", "GC=F"], "daily": ["XAUUSD=X", "GC=F"]},
    "WTI Oil": {"intraday": ["CL=F"],                 "daily": ["CL=F"]},
    "Bitcoin": {"intraday": ["BTC-USD"],              "daily": ["BTC-USD"]},
    "EUR/USD": {"intraday": ["EURUSD=X"],             "daily": ["EURUSD=X"]},
    "USD/JPY": {"intraday": ["JPY=X"],                "daily": ["JPY=X"]},
}
ADX_TREND = 20
SEND_TELE = True
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

# -------- Indicators --------
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(close, n=14):
    delta = close.diff()
    gain, loss = delta.clip(lower=0), (-delta).clip(lower=0)
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

# -------- Yahoo helpers --------
def fetch_yf(tickers, interval, period):
    last_err = None
    for t in tickers:
        try:
            df = yf.download(t, interval=interval, period=period,
                             auto_adjust=False, progress=False)
            if isinstance(df, pd.DataFrame) and not df.empty and {"Open","High","Low","Close"}.issubset(df.columns):
                df = df.dropna().copy()
                df.index = pd.to_datetime(df.index)
                return df, t
        except Exception as e:
            last_err = e
    raise last_err or RuntimeError(f"No data for {tickers} @ {interval}/{period}")

def resample_ohlc(df, rule):
    o = df["Open"].resample(rule).first()
    h = df["High"].resample(rule).max()
    l = df["Low"].resample(rule).min()
    c = df["Close"].resample(rule).last()
    v = df.get("Volume", pd.Series(index=df.index, dtype=float)).resample(rule).sum()
    out = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v}).dropna()
    return out

# -------- Core analysis --------
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
        return "LONG"
    if (last["Close"] < last["EMA20"] < last["EMA50"] < last["EMA200"]
        and last["RSI14"] < 45
        and last["MACD"] < last["MACD_SIG"]
        and trending):
        return "SHORT"
    return "SIDEWAY"

def analyze_asset(name, mapping):
    # 30m & 1H (<=60d)
    df30, _ = fetch_yf(mapping["intraday"], "30m", "60d")
    df1h,  _ = fetch_yf(mapping["intraday"], "1h",  "60d")
    # 3H: resample từ 1H
    df3h = resample_ohlc(df1h, "3H")
    # 1D: 5y
    df1d, _ = fetch_yf(mapping["daily"], "1d", "5y")

    frames = {
        "30m": df30, "1H": df1h, "3H": df3h, "1D": df1d
    }
    results = {k: (decide(decorate(v)) if len(v) >= 100 else "N/A")
               for k, v in frames.items()}

    msg = (
        f"====== {name} ======\n"
        f"30m-1H: {results['30m']} / {results['1H']}\n"
        f"1H: {results['1H']}"
        f"1D: {results['1D']}\n"
    )
    print(msg)
    return msg

# -------- Telegram --------
def send_tele(text):
    if not SEND_TELE: return
    if not BOT_TOKEN or not CHAT_ID:
        print("⚠️ Missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID"); return
    try:
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      json={"chat_id": CHAT_ID, "text": text}, timeout=20)
    except Exception as e:
        print("Telegram error:", e)

# -------- Main --------
def main():
    all_msgs = []
    for name, mapping in ASSETS.items():
        try:
            all_msgs.append(analyze_asset(name, mapping))
        except Exception as e:
            all_msgs.append(f"====== {name} ======\nLỗi: {e}")
    send_tele("\n\n".join(all_msgs))

if __name__ == "__main__":
    main()
