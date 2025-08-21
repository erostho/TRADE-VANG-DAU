# -*- coding: utf-8 -*-
"""
Frames: 30m, 1H, 1D (Yahoo Finance)
Symbols: GC=F (Gold), CL=F (WTI), BTC-USD, EURUSD=X, JPY=X
Logic: EMA20/50/200 + RSI14 + MACD + ADX + BBWidth
Output Telegram:
===SYMBOL===
30m-1H: <Sideway/Mixed/...>
1H: <Long/Short/Sideway>
1D: <Long/Short/Sideway>
(+ thêm dòng ⚠ note khi khung nào đó thiếu / fallback)

ENV:
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
  LOG_LEVEL=INFO/DEBUG (tùy chọn)
  INCLUDE_ERRORS_IN_TELEGRAM=1 (tùy chọn)
"""

import os, sys, time, logging
import numpy as np
import pandas as pd
import yfinance as yf
import requests

# ---------------- CONFIG ----------------
ASSETS = {
    "XAU/USD (Gold)": "GC=F",
    "WTI Oil": "CL=F",
    "Bitcoin": "BTC-USD",
    "EUR/USD": "EURUSD=X",
    "USD/JPY": "JPY=X",
}

# Yahoo intraday chỉ cho <= 60d; có khi 60d không có -> thử ngắn dần
INTRADAY_PERIODS = ["60d", "30d", "14d", "7d", "5d", "2d", "1d"]
DAILY_PERIODS    = ["5y", "3y", "1y"]

ADX_TREND = 20
SEND_TELE = True

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
INC_ERR_TG = os.getenv("INCLUDE_ERRORS_IN_TELEGRAM", "0") == "1"

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------- INDICATORS ----------------
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
        return "LONG"
    if (last["Close"] < last["EMA20"] < last["EMA50"] < last["EMA200"]
        and last["RSI14"] < 45
        and last["MACD"] < last["MACD_SIG"]
        and trending):
        return "SHORT"
    return "SIDEWAY"

# ---------------- DATA FETCH (with fallbacks) ----------------
def _download(symbol, interval, period):
    logging.debug(f"Try: {symbol} @ {interval}/{period}")
    df = yf.download(symbol, interval=interval, period=period,
                     auto_adjust=False, progress=False)
    if df is None or df.empty or not {"Open","High","Low","Close"}.issubset(df.columns):
        raise RuntimeError(f"No data for {symbol} @ {interval}/{period}")
    df = df.dropna().copy()
    df.index = pd.to_datetime(df.index)
    logging.info(f"OK: {symbol} @ {interval}/{period} -> {len(df)} rows "
                 f"[{df.index[0]} .. {df.index[-1]}]")
    return df

def fetch_intraday(symbol, interval):
    """
    interval: '30m' or '1h'
    Tries multiple periods. If 30m fails, fallback to 1h and mark degraded.
    Returns: (df, used_interval, used_period, note)
    """
    notes = []
    if interval == "30m":
        for p in INTRADAY_PERIODS:
            try:
                df = _download(symbol, "30m", p)
                return df, "30m", p, "; ".join(notes)
            except Exception as e:
                notes.append(f"30m/{p} fail: {e}")
        # fallback 1h
        for p in INTRADAY_PERIODS:
            try:
                df = _download(symbol, "1h", p)
                notes.append("fallback 30m→1h")
                return df, "1h", p, "; ".join(notes)
            except Exception as e:
                notes.append(f"1h/{p} fail: {e}")
        raise RuntimeError("; ".join(notes))
    elif interval == "1h":
        last_err = None
        for p in INTRADAY_PERIODS:
            try:
                df = _download(symbol, "1h", p)
                return df, "1h", p, "; ".join(notes)
            except Exception as e:
                last_err = e
                notes.append(f"1h/{p} fail: {e}")
        raise RuntimeError("; ".join(notes))
    else:
        raise ValueError("interval must be 30m or 1h")

def fetch_daily(symbol):
    for p in DAILY_PERIODS:
        try:
            df = _download(symbol, "1d", p)
            return df, "1d", p, ""
        except Exception as e:
            logging.warning(f"Daily {symbol} {p} fail: {e}")
    raise RuntimeError(f"Daily fail for {symbol}")

# ---------------- TELEGRAM ----------------
def send_tele(text):
    if not SEND_TELE: return
    if not BOT_TOKEN or not CHAT_ID:
        logging.warning("Missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID")
        return
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text},
            timeout=20
        )
        if r.status_code != 200:
            logging.warning(f"Telegram send failed: {r.status_code} {r.text}")
    except Exception as e:
        logging.exception(f"Telegram error: {e}")

# ---------------- ANALYZE ONE ----------------
def analyze_one(name, ticker):
    logging.info(f"=== Start {name} ({ticker}) ===")
    notes = []

    # 30m
    try:
        d30, i30, p30, n30 = fetch_intraday(ticker, "30m")
        if n30: notes.append(f"30m note: {n30}")
        d30 = decorate(d30)
        s30 = decide(d30) if len(d30) >= 100 else "N/A"
    except Exception as e:
        s30 = "N/A"
        notes.append(f"30m ERR: {e}")

    # 1H
    try:
        d1h, i1h, p1h, n1h = fetch_intraday(ticker, "1h")
        if n1h: notes.append(f"1H note: {n1h}")
        d1h = decorate(d1h)
        s1h = decide(d1h) if len(d1h) >= 100 else "N/A"
    except Exception as e:
        s1h = "N/A"
        notes.append(f"1H ERR: {e}")

    # 1D
    try:
        d1d, i1d, p1d, _ = fetch_daily(ticker)
        d1d = decorate(d1d)
        s1d = decide(d1d) if len(d1d) >= 200 else "N/A"
    except Exception as e:
        s1d = "N/A"
        notes.append(f"1D ERR: {e}")

    # short line (30m-1H)
    short_line = s30 if (s30 == s1h and s30 != "N/A") else f"Mixed (30m:{s30}, 1H:{s1h})"

    msg = (
        f"==={name}===\n"
        f"30m-1H: {short_line}\n"
        f"1H: {s1h}\n"
        f"1D: {s1d}"
    )

    # append errors (compact) if any
    if notes and INC_ERR_TG:
        compact = "; ".join(notes)
        # Telegram message length friendly
        if len(compact) > 600:
            compact = compact[:600] + "..."
        msg += f"\n⚠ {compact}"

    logging.info(msg.replace("\n", " | "))
    return msg

# ---------------- MAIN ----------------
def main():
    report_parts = []
    for name, ticker in ASSETS.items():
        try:
            report_parts.append(analyze_one(name, ticker))
        except Exception as e:
            logging.exception(f"{name} fatal")
            report_parts.append(f"==={name}===\nLỗi nghiêm trọng: {e}")
    send_tele("\n\n".join(report_parts))

if __name__ == "__main__":
    main()
