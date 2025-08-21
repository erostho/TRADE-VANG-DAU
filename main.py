# -*- coding: utf-8 -*-
"""
Only-TwelveData Scanner
Frames: 30m, 1H, 2H, 4H, 1D
Assets:
  - XAU/USD (Gold)  -> XAU/USD
  - WTI Oil         -> CL (continuous)
  - Bitcoin         -> BTC/USD
  - EUR/USD         -> EUR/USD
  - USD/JPY         -> USD/JPY
Logic: EMA20/50/200 + RSI14 + MACD + ADX + BBWidth
Telegram format:
===SYMBOL===
30m-1H: <Sideway/Mixed/...>
2H-4H: <Long/Short/Sideway or Mixed(...)>
1D: <Long/Short/Sideway>

ENV REQUIRED:
  TWELVE_DATA_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
Optional:
  LOG_LEVEL=INFO|DEBUG
  INCLUDE_ERRORS_IN_TELEGRAM=1
"""

import os
import logging
import requests
import pandas as pd
import numpy as np

# ----------- ENV / LOG -----------
TD_KEY = os.getenv("TWELVE_DATA_KEY")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
INC_ERR_TG = os.getenv("INCLUDE_ERRORS_IN_TELEGRAM", "0") == "1"

if not TD_KEY:
    raise SystemExit("❌ Missing TWELVE_DATA_KEY env.")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ----------- CONFIG -----------
ASSETS = {
    "XAU/USD (Gold)": "XAU/USD",
    "WTI Oil": "CL",            # WTI continuous
    "Bitcoin": "BTC/USD",
    "EUR/USD": "EUR/USD",
    "USD/JPY": "USD/JPY",
}

INTERVALS = {  # TD intervals
    "30m": "30min",
    "1H":  "1h",
    "2H":  "2h",
    "4H":  "4h",
    "1D":  "1day",
}
OUTPUT_SIZES = [5000, 2000, 1000, 300]  # thử giảm dần để lấy đủ nến

# Ngưỡng lọc / số nến tối thiểu
ADX_TREND = 20
MIN_BARS_INTRADAY = 60   # cho 30m/1h/2h/4h
MIN_BARS_DAILY    = 120  # cho 1d

# ----------- Indicators -----------
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

# ----------- Twelve Data fetch -----------
def td_download(symbol: str, interval_label: str) -> pd.DataFrame:
    iv = INTERVALS[interval_label]
    last_err = None
    for size in OUTPUT_SIZES:
        try:
            logging.info(f"TD fetch {symbol} {iv} size={size}")
            r = requests.get(
                "https://api.twelvedata.com/time_series",
                params={
                    "symbol": symbol,
                    "interval": iv,
                    "outputsize": size,
                    "apikey": os.environ["TWELVE_DATA_KEY"],
                },
                timeout=25,
            )
            r.raise_for_status()
            js = r.json()
            if "values" not in js or not js["values"]:
                raise RuntimeError(js)
            df = pd.DataFrame(js["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").set_index("datetime")
            df = df.rename(columns={"open":"Open","high":"High","low":"Low",
                                    "close":"Close","volume":"Volume"})
            df[["Open","High","Low","Close"]] = df[["Open","High","Low","Close"]].astype(float)
            if "Volume" in df.columns:
                df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
            logging.info(f"TD OK {symbol} {iv} -> {len(df)} rows [{df.index[0]} .. {df.index[-1]}]")
            return df
        except Exception as e:
            last_err = e
            logging.warning(f"TD fail {symbol} {iv} size={size}: {e}")
    raise RuntimeError(f"TD final fail {symbol} {iv}: {last_err}")

# ----------- Telegram ----------
def send_tele(text: str):
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

# ----------- Analyze one ----------
def analyze_one(name: str, td_symbol: str) -> str:
    logging.info(f"=== Start {name} ({td_symbol}) ===")
    notes = []

    frames = {}
    for lbl in ["30m", "1H", "2H", "4H", "1D"]:
        try:
            frames[lbl] = td_download(td_symbol, lbl)
        except Exception as e:
            frames[lbl] = None
            notes.append(f"{lbl} ERR: {e}")

    def sig_for(df, is_daily=False):
        if isinstance(df, pd.DataFrame):
            n = len(df)
            need = MIN_BARS_DAILY if is_daily else MIN_BARS_INTRADAY
            logging.info(f"bars {name} -> {'daily' if is_daily else 'intra'} {n}")
            if n >= need:
                return decide(decorate(df))
        return "N/A"

    s30 = sig_for(frames["30m"])
    s1h = sig_for(frames["1H"])
    s2h = sig_for(frames["2H"])
    s4h = sig_for(frames["4H"])
    s1d = sig_for(frames["1D"], is_daily=True)

    line_short = s30 if (s30 == s1h and s30 != "N/A") else f"Mixed (30m:{s30}, 1H:{s1h})"
    line_mid   = s2h if (s2h == s4h and s2h != "N/A") else f"Mixed (2H:{s2h}, 4H:{s4h})"

    msg = f"==={name}===\n30m-1H: {line_short}\n2H-4H: {line_mid}\n1D: {s1d}"
    if notes and INC_ERR_TG:
        compact = "; ".join(notes)
        if len(compact) > 700: compact = compact[:700] + "..."
        msg += f"\n⚠ {compact}"

    logging.info(msg.replace("\n", " | "))
    return msg

# ----------- Main ----------
def main():
    reports = []
    for name, sym in ASSETS.items():
        try:
            reports.append(analyze_one(name, sym))
        except Exception as e:
            logging.exception(f"{name} fatal")
            reports.append(f"==={name}===\nLỗi nghiêm trọng: {e}")
    send_tele("\n\n".join(reports))

if __name__ == "__main__":
    main()
