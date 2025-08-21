# -*- coding: utf-8 -*-
"""
Phân tích đa khung: Vàng, Dầu, BTC, EUR/USD, USD/JPY (Yahoo Finance)
Khung: 30m, 1H, 6H, 12H, 1D
Logic: EMA20/50/200 + RSI14 + MACD + ADX + BBWidth
Kết luận: LONG / SHORT / SIDEWAY
Gửi báo cáo Telegram (ENV: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import requests

# ============== CONFIG ==============
ASSETS = {
    "XAU/USD": ["XAUUSD=X", "GC=F"],  # vàng: ưu tiên spot; fallback futures
    "WTI Oil": ["CL=F"],              # dầu WTI futures
    "Bitcoin": ["BTC-USD"],
    "EUR/USD": ["EURUSD=X"],
    "USD/JPY": ["JPY=X"],
}

TIMEFRAMES = {
    "30m": "30m",
    "1H":  "1h",
    "6H":  "6h",
    "12H": "12h",
    "1D":  "1d",
}

# Ngưỡng
ADX_TREND = 20
RR_MIN = 1.5  # (giữ placeholder – chưa tính SL/TP ở bản báo cáo)
SEND_TELE = True

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

# ============== INDICATORS (ỔN ĐỊNH 1-D) ==============
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val

def macd(close: pd.Series, fast=12, slow=26, signal=9):
    m = ema(close, fast) - ema(close, slow)
    sig = ema(m, signal)
    return m, sig, m - sig

def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - pc).abs(),
        (df["Low"] - pc).abs()
    ], axis=1).max(axis=1)
    return tr

def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()
    plus_dm  = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    trn = true_range(df).rolling(n).sum()
    plus_di  = 100 * (plus_dm.rolling(n).sum() / trn)
    minus_di = 100 * (minus_dm.rolling(n).sum() / trn)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.rolling(n).mean()

def bb_width(close: pd.Series, n=20) -> pd.Series:
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    upper = ma + 2*sd
    lower = ma - 2*sd
    return (upper - lower) / ma

# ============== DATA FETCH (CÓ FALLBACK + PERIOD HỢP LỆ) ==============
def period_for_interval(interval: str) -> str:
    # Intraday (<=1h) phải <= 60d
    if interval in ("30m", "1h"):
        return "60d"
    # 6h/12h lấy dài hơn
    if interval in ("6h", "12h"):
        return "730d"  # ~2 năm
    return "5y"       # 1D có thể lấy dài
def fetch_one(ticker: str, interval: str) -> pd.DataFrame:
    per = period_for_interval(interval)
    df = yf.download(ticker, interval=interval, period=per, progress=False, auto_adjust=False)
    if isinstance(df, pd.DataFrame) and not df.empty and {"Open","High","Low","Close"}.issubset(df.columns):
        return df
    raise ValueError(f"No data for {ticker} @ {interval} (period={per})")

def fetch_with_fallback(symbol_aliases, interval: str) -> pd.DataFrame:
    last_err = None
    for tk in symbol_aliases:
        try:
            return fetch_one(tk, interval)
        except Exception as e:
            last_err = e
    raise last_err or RuntimeError("fetch_with_fallback failed")

# ============== CORE ANALYSIS ==============
def decorate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["EMA20"]  = ema(out["Close"], 20)
    out["EMA50"]  = ema(out["Close"], 50)
    out["EMA200"] = ema(out["Close"], 200)
    out["RSI14"]  = rsi(out["Close"], 14)
    out["MACD"], out["MACD_SIG"], out["MACD_HIST"] = macd(out["Close"])
    out["ADX14"]  = adx(out, 14)
    out["BBWIDTH"] = bb_width(out["Close"], 20)
    return out

def decide_signal(df: pd.DataFrame) -> str:
    last = df.iloc[-1]
    # Filter sideway: ADX & BBWidth > mean
    trending = (last["ADX14"] > ADX_TREND) and (last["BBWIDTH"] > df["BBWIDTH"].rolling(50).mean().iloc[-1])
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

def analyze_asset(name: str, aliases: list[str]) -> str:
    results = {}
    for tf_name, interval in TIMEFRAMES.items():
        try:
            raw = fetch_with_fallback(aliases, interval)
            df = decorate(raw)
            results[tf_name] = decide_signal(df)
        except Exception as e:
            results[tf_name] = f"N/A"
            print(f"{name} {tf_name} -> {e}")

    short = f"{results.get('30m')} / {results.get('1H')}"
    mid   = f"{results.get('6H')} / {results.get('12H')}"
    long_ = results.get("1D")

    msg = (
        f"====== {name} ======\n"
        f"Ngắn (30m-1H): {short}\n"
        f"Trung (6H-12H): {mid}\n"
        f"Dài (1D): {long_}\n"
        f"👉 Kết luận (1H): {results.get('1H')}"
    )
    print(msg)
    return msg

# ============== TELEGRAM ==============
def send_telegram(text: str):
    if not SEND_TELE:
        return
    if not BOT_TOKEN or not CHAT_ID:
        print("⚠️ TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID chưa cấu hình.")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text},
            timeout=20,
        )
    except Exception as e:
        print("Telegram error:", e)

# ============== MAIN ==============
def main():
    reports = []
    for name, aliases in ASSETS.items():
        reports.append(analyze_asset(name, aliases))
    send_telegram("\n\n".join(reports))

if __name__ == "__main__":
    main()
