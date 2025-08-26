#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import requests
import pandas as pd
from datetime import datetime
import logging

# ============== Logging ==============
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ============== ENV ==============
TD_KEY = os.getenv("TD_KEY", "")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ============== CONST ==============
HTTP_TIMEOUT = 30
TITLE_PREFIX = "💵 TRADE GOODS"
watchlist = {
    "XAU/USD": "XAUUSD",
    "WTI Oil": "CL",
    "Bitcoin": "BTC/USD",
    "Ethereum": "ETH/USD",
    "EUR/USD": "EUR/USD",
    "USD/JPY": "USD/JPY",
}
intervals = ["15min", "30min", "1h", "2h", "4h", "1day"]

# ============== Utils ==============
def fetch_ohlcv(symbol: str, interval: str, n: int = 100):
    url = f"https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": n,
        "apikey": TD_KEY,
    }
    r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    j = r.json()
    if "values" not in j:
        logging.warning("No data for %s-%s: %s", symbol, interval, j)
        return None
    df = pd.DataFrame(j["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    df["close"] = df["close"].astype(float)
    return df

def signal_from(df: pd.DataFrame):
    if df is None or len(df) < 50:
        return "N/A"
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    last = df.iloc[-1]
    if last["ema20"] > last["ema50"]:
        return "LONG"
    elif last["ema20"] < last["ema50"]:
        return "SHORT"
    return "SIDEWAY"

def telegram_send(text: str):
    if not BOT_TOKEN or not CHAT_ID:
        logging.warning("Thiếu TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID – bỏ qua gửi Telegram.")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": CHAT_ID, "text": text}, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        logging.info("Telegram: sent")
    except Exception as e:
        logging.error("Telegram error: %s", e)

# ============== Main ==============
def main():
    lines = [f"{TITLE_PREFIX}"]
    for name, symbol in watchlist.items():
        try:
            sigs = {}
            for tf in intervals:
                df = fetch_ohlcv(symbol, tf, 100)
                sigs[tf] = signal_from(df)
            lines.append(f"==={name}===")
            lines.append(f"15m-30m: {sigs['15min']}-{sigs['30min']}")
            lines.append(f"1H-2H: {sigs['1h']}-{sigs['2h']}")
            lines.append(f"4H-1D: {sigs['4h']}-{sigs['1day']}")
            lines.append("")
        except Exception as e:
            logging.error("Error %s: %s", name, e)
    msg = "\n".join(lines)
    logging.info("Message:\n%s", msg)
    telegram_send(msg)

if __name__ == "__main__":
    main()
