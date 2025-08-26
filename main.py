#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main bot:
- Fetch EACH timeframe separately from TwelveData (no resampling).
- Intervals: 15m, 30m, 1h, 2h, 4h, 1day.
- Pairs: BTC/USD, ETH/USD, XAU/USD (Gold), WTI Oil (CL or WTI/USD), EUR/USD, USD/JPY.
- Rate limiting: default 7 requests/minute to stay under free 8 credits/min (configurable).
- Only send Telegram when at least one timeframe has LONG/SHORT (not all SIDEWAY/N/A).
- Output format:

💵 TRADE GOODS
⏱ 2025-08-26 15:16:15

===Bitcoin===
15m-30m: SIDEWAY - SIDEWAY
1h-2h:   SIDEWAY - LONG
4h-1D:   SHORT
"""

import os
import time
import math
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional

import requests

# ---------- Config via ENV ----------
TD_API_KEY = os.getenv("TD_API_KEY", "").strip()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TZ_NAME = os.getenv("TZ", "Asia/Ho_Chi_Minh")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
# requests per minute cap (credits). keep <=7 to be safe on free tier (limit=8)
RPM = int(os.getenv("TD_MAX_RPM", "7"))
# how many candles to fetch per timeframe
LIMIT = int(os.getenv("TD_LIMIT", "120"))

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

TD_BASE = "https://api.twelvedata.com/time_series"
HTTP_TIMEOUT = 30

TITLE = "💵 TRADE GOODS"

# ---------- Symbols & display names ----------
PAIRS: Dict[str, List[str]] = {
    "Bitcoin": ["BTC/USD", "BTCUSDT"],
    "Ethereum": ["ETH/USD", "ETHUSDT"],
    "XAU/USD (Gold)": ["XAU/USD", "XAUUSD"],
    "WTI Oil": ["WTI/USD", "CL=F", "USOIL", "CL"],
    "EUR/USD": ["EUR/USD", "EURUSD"],
    "USD/JPY": ["USD/JPY", "USDJPY"],
}

FRAMES = ["15min", "30min", "1h", "2h", "4h", "1day"]

# ---------- Utilities ----------
def now_local() -> datetime:
    # Simple TZ helper: use +7 if Asia/Ho_Chi_Minh
    if TZ_NAME == "Asia/Ho_Chi_Minh":
        return datetime.now(timezone(timedelta(hours=7)))
    return datetime.now()

def http_get(url: str, params: Dict) -> Dict:
    r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    j = r.json()
    return j

# ---------- TwelveData fetch per timeframe ----------
def td_fetch_series(symbol: str, interval: str) -> List[Dict]:
    """
    Fetch time series for a symbol at given interval. Returns list of bars newest->oldest.
    """
    if not TD_API_KEY:
        raise RuntimeError("Missing TD_API_KEY env")
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": LIMIT,
        "format": "JSON",
        "apikey": TD_API_KEY,
        "dp": 8,
        "timezone": "UTC",
        "order": "desc",
    }
    try:
        j = http_get(TD_BASE, params)
    except Exception as e:
        raise RuntimeError(f"HTTP fail {symbol} {interval}: {e}")

    # Handle errors from API
    if isinstance(j, dict) and j.get("status") == "error":
        code = str(j.get("code"))
        msg = j.get("message", "")
        # Rate limit 429: wait and retry once
        if code == "429":
            logging.warning("Rate limit hit. Sleeping 65s then retry: %s %s", symbol, interval)
            time.sleep(65)
            j = http_get(TD_BASE, params)
            if isinstance(j, dict) and j.get("status") == "error":
                raise RuntimeError(f"TD error after retry {symbol} {interval}: {j}")
        else:
            raise RuntimeError(f"TD error {symbol} {interval}: {j}")

    data = j.get("values") if isinstance(j, dict) else None
    if not data:
        raise RuntimeError(f"No data {symbol}-{interval}: {j}")
    return data  # newest first

# ---------- Throttler ----------
class Throttler:
    """
    Simple rpm throttler: ensure <= RPM requests per 60 seconds.
    """
    def __init__(self, rpm: int):
        self.rpm = max(1, rpm)
        self.window_start = time.time()
        self.count = 0

    def hit(self):
        now = time.time()
        # reset window after 60s
        if now - self.window_start >= 60:
            self.window_start = now
            self.count = 0
        # if would exceed, sleep the remaining time
        if self.count >= self.rpm:
            wait = 60 - (now - self.window_start)
            if wait > 0:
                logging.info("Throttle: sleeping %.1fs to respect RPM=%s", wait, self.rpm)
                time.sleep(wait)
            self.window_start = time.time()
            self.count = 0
        self.count += 1

throttler = Throttler(RPM)

# ---------- Signal logic per timeframe ----------
def decide_signal(values: List[Dict]) -> str:
    """
    Decide LONG/SHORT/SIDEWAY from OHLC list (newest first).
    Rule (simple & robust):
      - Compute ema20 on close (reverse to old->new first).
      - slope = ema[-1] - ema[-6] (approx over 5 bars)
      - price position vs ema:
          up if slope>0 and close[-1] > ema[-1]
          down if slope<0 and close[-1] < ema[-1]
          else SIDEWAY
    """
    if not values or len(values) < 25:
        return "N/A"
    # reverse
    closes = [float(x["close"]) for x in values[::-1]]
    # ema20
    k = 2 / (20 + 1)
    ema = []
    e = closes[0]
    for c in closes:
        e = c * k + e * (1 - k)
        ema.append(e)
    slope = ema[-1] - ema[-6] if len(ema) >= 6 else ema[-1] - ema[0]
    last_close = closes[-1]
    last_ema = ema[-1]
    if slope > 0 and last_close > last_ema:
        return "LONG"
    if slope < 0 and last_close < last_ema:
        return "SHORT"
    return "SIDEWAY"

# ---------- Try a list of tickers for one label ----------
def get_best_symbol(label: str) -> Optional[str]:
    for s in PAIRS.get(label, []):
        return s
    return None

def analyze_one(label: str) -> Dict[str, str]:
    """
    For one instrument label, fetch EACH timeframe separately.
    Return dict interval->signal.
    """
    symbols = PAIRS.get(label, [])
    if not symbols:
        return {}
    signals: Dict[str, str] = {}
    # Try each candidate symbol until success for a given interval
    for iv in FRAMES:
        got = False
        for sym in symbols:
            try:
                throttler.hit()
                vals = td_fetch_series(sym, iv)
                sig = decide_signal(vals)
                signals[iv] = sig
                got = True
                break
            except Exception as e:
                logging.warning("No data for %s-%s: %s", sym, iv, e)
                continue
        if not got:
            signals[iv] = "N/A"
    return signals

# ---------- Telegram ----------
def send_telegram(text: str):
    if not BOT_TOKEN or not CHAT_ID:
        logging.info("Telegram disabled (missing env).")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": CHAT_ID, "text": text}, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        logging.info("Telegram: sent")
    except Exception as e:
        logging.exception("Telegram error: %s", e)

# ---------- Format message ----------
def fmt_block(label: str, sigs: Dict[str, str]) -> str:
    # map frames
    s15 = sigs.get("15min", "N/A")
    s30 = sigs.get("30min", "N/A")
    s1h = sigs.get("1h", "N/A")
    s2h = sigs.get("2h", "N/A")
    s4h = sigs.get("4h", "N/A")
    s1d = sigs.get("1day", "N/A")

    lines = []
    lines.append(f"==={label}===")
    lines.append(f"15m-30m: {s15} - {s30}")
    lines.append(f"1h-2h:   {s1h} - {s2h}")
    # if 4h and 1D same, show single; else show both
    if s4h == s1d:
        lines.append(f"4h-1D:   {s4h}")
    else:
        lines.append(f"4h-1D:   {s4h} - {s1d}")
    return "\n".join(lines)

def all_sideway(blocks: Dict[str, Dict[str, str]]) -> bool:
    # Return True if all intervals for all labels are SIDEWAY or N/A
    for sigs in blocks.values():
        for v in sigs.values():
            if v in ("LONG", "SHORT"):
                return False
    return True

# ---------- Main ----------
def main():
    logging.info("Start")
    results: Dict[str, Dict[str, str]] = {}
    for label in PAIRS.keys():
        try:
            sigs = analyze_one(label)
            results[label] = sigs
        except Exception as e:
            logging.exception("Analyze fail %s: %s", label, e)
            results[label] = {}

    if all_sideway(results):
        logging.info("All SIDEWAY/N/A -> skip Telegram")
        return

    t = now_local().strftime("%Y-%m-%d %H:%M:%S")
    lines = [TITLE, f"⏱ {t}", ""]
    for label in results.keys():
        lines.append(fmt_block(label, results[label]))
        lines.append("")
    msg = "\n".join(lines).rstrip()
    logging.info("Message:\n%s", msg)
    send_telegram(msg)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("Fatal: %s", e)
        try:
            send_telegram(f"{TITLE}\n❌ ERROR: {e}")
        except Exception:
            pass
        raise
