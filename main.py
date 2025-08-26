#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main bot: fetch OHLC from TwelveData in batch, analyze trend per timeframe,
and send compact summary to Telegram.

ENV required:
- TWELVEDATA_API_KEY
- TELEGRAM_BOT_TOKEN
- TELEGRAM_CHAT_ID
Optional:
- LOG_LEVEL (default INFO)
- TZ (default Asia/Ho_Chi_Minh)
"""

import os
import sys
import time
import math
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Any

import requests

# ---------------- Logging ----------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ---------------- ENV ----------------
TD_KEY = os.getenv("TWELVEDATA_API_KEY", "").strip()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TITLE_PREFIX = "💵 TRADE GOODS"

if not TD_KEY:
    logging.warning("Missing TWELVEDATA_API_KEY")
if not BOT_TOKEN or not CHAT_ID:
    logging.warning("Missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID")

# ---------------- Const ----------------
TD_BASE = "https://api.twelvedata.com"
HTTP_TIMEOUT = 30

# Telegram formatting switch
USE_MD = False

# Standardized symbol map for TwelveData
SYMBOL_MAP: Dict[str, str] = {
    "XAUUSD": "XAU/USD",
    "WTI": "WTI/USD",        # Spot WTI index at TwelveData
    "CL": "CL",              # Alternative: futures continuous (may require paid)
    "BTCUSD": "BTC/USD",
    "ETHUSD": "ETH/USD",
    "EURUSD": "EUR/USD",
    "USDJPY": "USD/JPY",
}

# Default watchlist
WATCH_SYMBOLS = ["XAUUSD", "WTI", "BTCUSD", "ETHUSD", "EURUSD", "USDJPY"]

# Intervals (pair formatting in output)
INTERVALS = ["15min", "30min", "1h", "2h", "4h", "1day"]

# ---------------- Time helpers ----------------
def now_vn() -> datetime:
    """Return now in VN timezone if TZ=Asia/Ho_Chi_Minh else local."""
    tzname = os.getenv("TZ", "Asia/Ho_Chi_Minh")
    try:
        if tzname == "Asia/Ho_Chi_Minh":
            return datetime.now(timezone(timedelta(hours=7)))
        return datetime.now()
    except Exception:
        return datetime.utcnow()

# ---------------- HTTP helpers ----------------
def http_get(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        logging.debug("Raw response not JSON: %s", r.text[:500])
        raise

def td_batch_time_series(symbols: List[str], interval: str, outputsize: int = 120, retries: int = 2) -> dict:
    """
    Call TwelveData time_series with comma-joined symbols to minimize API calls.
    Handles 429 by brief sleep & retry.
    Returns raw JSON.
    """
    if not symbols:
        return {}

    url = f"{TD_BASE}/time_series"
    params = {
        "symbol": ",".join(symbols),
        "interval": interval,
        "outputsize": outputsize,
        "format": "JSON",
        "apikey": TD_KEY,
    }

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            j = http_get(url, params)
            # If response is a single timeseries (dict with "values"), normalize into dict
            if "values" in j and "meta" in j:
                sym = j.get("meta", {}).get("symbol")
                return {sym: j}
            return j
        except requests.HTTPError as e:
            last_err = e
            code = e.response.status_code if e.response is not None else None
            if code == 429:
                wait = 3 + attempt * 2
                logging.warning("429 rate limited at interval=%s, sleeping %ss...", interval, wait)
                time.sleep(wait)
                continue
            raise
        except Exception as e:
            last_err = e
            logging.warning("Fetch error interval=%s symbols=%s: %s", interval, symbols, e)
            time.sleep(1.0 + attempt * 0.5)
    if last_err:
        raise last_err
    return {}

def normalize_symbol(sym: str) -> str:
    return SYMBOL_MAP.get(sym.upper(), sym)

# ---------------- TA helpers ----------------
def ema(values: List[float], period: int) -> List[float]:
    """Simple EMA implementation. Returns list of same length, with Nones for warmup replaced by SMA for first seed."""
    if not values or period <= 1:
        return values[:]
    k = 2.0 / (period + 1.0)
    out: List[float] = []
    ema_prev = None
    for i, v in enumerate(values):
        if v is None:
            out.append(ema_prev if ema_prev is not None else None)
            continue
        if ema_prev is None:
            # seed by SMA of first P points (or first value if not enough)
            start = max(0, i - period + 1)
            window = [x for x in values[start:i+1] if x is not None]
            ema_prev = sum(window) / len(window) if window else v
        else:
            ema_prev = (v - ema_prev) * k + ema_prev
        out.append(ema_prev)
    return out

def decide_signal_from_candles(values: List[dict]) -> str:
    """
    Decide LONG/SHORT/SIDEWAY from last 40 candles using EMA20 & its slope.
    TwelveData values are newest-first; we reverse to oldest-first.
    """
    if not values or len(values) < 25:
        return "N/A"
    # Reverse to chronological
    closes = [float(x.get("close")) for x in reversed(values)]
    # Compute EMA20
    e20 = ema(closes, 20)
    c_last = closes[-1]
    e_last = e20[-1]
    # slope from last 5 bars
    e_prev = e20[-6]
    slope = e_last - e_prev

    # thresholds
    band = max(0.0001, 0.003 * abs(e_last))  # ~0.3% band
    if c_last > e_last + band and slope > 0:
        return "LONG"
    if c_last < e_last - band and slope < 0:
        return "SHORT"
    return "SIDEWAY"

# ---------------- Fetch & analyze ----------------
def fetch_all(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """Return dict {interval: {symbol: series_json}}"""
    norm = [normalize_symbol(s) for s in symbols]
    out: Dict[str, Dict[str, Any]] = {}
    for iv in INTERVALS:
        try:
            data = td_batch_time_series(norm, iv, outputsize=120)
            out[iv] = data
            time.sleep(0.7)  # stay under free limit
        except Exception as e:
            logging.warning("Interval %s failed: %s", iv, e)
            out[iv] = {}
    return out

def analyze(all_data: Dict[str, Dict[str, Any]], symbols: List[str]) -> Dict[str, Dict[str, str]]:
    """Return {symbol_display: {interval: signal}}"""
    results: Dict[str, Dict[str, str]] = {}
    # Build reverse map from TD symbol to display key
    rev_map = {normalize_symbol(k): k for k in symbols}
    for iv, blob in all_data.items():
        if not isinstance(blob, dict):
            continue
        for td_sym, series in blob.items():
            # Handle API error object
            if isinstance(series, dict) and series.get("status") == "error":
                logging.warning("No data for %s %s: %s", td_sym, iv, series)
                continue
            # Some batch response nests as {"BTC/USD": {"meta":..., "values":[...] } }
            values = None
            if isinstance(series, dict) and "values" in series:
                values = series["values"]
            elif isinstance(series, dict) and td_sym in series and "values" in series[td_sym]:
                values = series[td_sym]["values"]
            if values is None:
                continue
            sig = decide_signal_from_candles(values)
            key = rev_map.get(td_sym, td_sym)
            results.setdefault(key, {})[iv] = sig
    return results

# ---------------- Telegram ----------------
def telegram_send(text: str) -> None:
    if not BOT_TOKEN or not CHAT_ID:
        logging.info("Telegram creds missing, print message instead:\n%s", text)
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text}
    if USE_MD:
        payload["parse_mode"] = "Markdown"
    try:
        r = requests.post(url, json=payload, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        logging.info("Telegram: sent")
    except Exception as e:
        logging.exception("Telegram send fail: %s", e)

# ---------------- Format message ----------------
def format_report(analysis: Dict[str, Dict[str, str]]) -> str:
    # Ordered display by our watchlist order
    lines: List[str] = [f"{TITLE_PREFIX}",]
    for sym in WATCH_SYMBOLS:
        data = analysis.get(sym, {})
        if not data:
            continue
        lines.append(f"==={sym.replace('USD','/USD')}===")
        g = lambda iv: data.get(iv, "N/A")
        # pairs
        lines.append(f"15m-30m: {pair_text(g('15min'), g('30min'))}")
        lines.append(f"1H-2H: {pair_text(g('1h'), g('2h'))}")
        lines.append(f"4H-1D: {pair_text(g('4h'), g('1day'))}")
        lines.append("")

    lines.append(f"⏰ {now_vn().strftime('%Y-%m-%d %H:%M:%S')}")
    return "\n".join(lines).strip()

def pair_text(a: str, b: str) -> str:
    if a == b:
        return a
    if a == "N/A" and b == "N/A":
        return "N/A"
    return f"Mixed ({a},{b})"

# ---------------- Main ----------------
def main():
    logging.info("Start bot")
    if not TD_KEY:
        raise RuntimeError("TWELVEDATA_API_KEY missing")

    data = fetch_all(WATCH_SYMBOLS)
    analysis = analyze(data, WATCH_SYMBOLS)
    msg = format_report(analysis)
    logging.info("\n" + msg)
    telegram_send(msg)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("Fatal: %s", e)
        try:
            telegram_send(f"{TITLE_PREFIX}\n❌ ERROR: {e}")
        except Exception:
            pass
        sys.exit(1)
