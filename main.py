#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TRADE GOODS bot — batched TwelveData fetch + compact signal + Telegram
- Intervals: 15m, 30m, 1h, 2h, 4h, 1day
- Symbols: BTC/USD, ETH/USD, XAU/USD (Gold), WTI/USD (Oil), EUR/USD, USD/JPY
- Groups in message: 15m-30m, 1h-2h, 4h-1D
- Only send when there is at least one LONG/SHORT. If all are SIDEWAY → skip Telegram.
Environment variables:
  TD_KEY                TwelveData API key
  TELEGRAM_BOT_TOKEN    Telegram bot token
  TELEGRAM_CHAT_ID      Telegram chat id
  LOG_LEVEL             INFO (default) / DEBUG
"""

import os
import time
import math
import json
import logging
from typing import Dict, List, Tuple
from datetime import datetime, timezone, timedelta

import requests

# ------------ Logging ------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ------------ ENV ------------
TD_KEY = os.getenv("TD_KEY", "").strip()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

TITLE = "💵 TRADE GOODS"

API_TS = "https://api.twelvedata.com/time_series"
HTTP_TIMEOUT = 30

# Symbols mapping: {display: twelvedata_symbol}
SYMBOLS: Dict[str, str] = {
    "Bitcoin": "BTC/USD",
    "Ethereum": "ETH/USD",
    "XAU/USD (Gold)": "XAU/USD",
    "WTI Oil": "WTI/USD",
    "EUR/USD": "EUR/USD",
    "USD/JPY": "USD/JPY",
}

INTERVALS = ["15min", "30min", "1h", "2h", "4h", "1day"]

# ------------ Helpers ------------
def now_vn() -> datetime:
    try:
        return datetime.now(timezone(timedelta(hours=7)))
    except Exception:
        return datetime.utcnow()

def ema(series: List[float], span: int) -> List[float]:
    """Simple EMA without pandas."""
    if not series:
        return []
    alpha = 2 / (span + 1.0)
    out = []
    e = series[0]
    out.append(e)
    for x in series[1:]:
        e = alpha * x + (1 - alpha) * e
        out.append(e)
    return out

def classify(prices: List[float]) -> str:
    """
    Classify trend using EMA20/EMA50 and slope.
    LONG  if EMA20>EMA50 and slope_up
    SHORT if EMA20<EMA50 and slope_down
    else SIDEWAY
    """
    if not prices or len(prices) < 60:
        return "N/A"
    fast = ema(prices, 20)
    slow = ema(prices, 50)
    f1, s1 = fast[-1], slow[-1]
    # slope via last 5 bars
    tail = prices[-5:]
    slope = tail[-1] - tail[0]
    if f1 > s1 and slope > 0:
        return "LONG"
    if f1 < s1 and slope < 0:
        return "SHORT"
    return "SIDEWAY"

def fetch_timeframe(interval: str, td_symbols: List[str], retries: int = 2, sleep_on_429: int = 65) -> Dict[str, List[float]]:
    """
    Fetch close prices for many symbols in one request for a single interval.
    Returns dict: {td_symbol: [closes newest_last]}
    """
    params = {
        "symbol": ",".join(td_symbols),
        "interval": interval,
        "outputsize": 120,  # enough for EMA50
        "apikey": TD_KEY,
        "format": "JSON",
    }
    for attempt in range(retries + 1):
        try:
            r = requests.get(API_TS, params=params, timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            j = r.json()
            # Handle error shape (when querying multiple, errors may be per key or global)
            if isinstance(j, dict) and j.get("status") == "error":
                msg = str(j)
                if "429" in msg or "credits" in msg.lower():
                    if attempt < retries:
                        logger.warning("Rate limited (global). Sleeping %ss then retry...", sleep_on_429)
                        time.sleep(sleep_on_429)
                        continue
                logger.warning("No data for %s: %s", interval, msg)
                return {}

            out: Dict[str, List[float]] = {}

            # Multiple symbols → object with keys per symbol or "data" list
            if isinstance(j, dict) and "data" in j and isinstance(j["data"], list):
                for item in j["data"]:
                    sym = item.get("symbol")
                    values = item.get("values") or []
                    closes = [float(v["close"]) for v in reversed(values) if "close" in v]  # oldest→newest
                    out[sym] = closes
            else:
                # fallback: each key is a symbol
                for sym in td_symbols:
                    obj = j.get(sym) if isinstance(j, dict) else None
                    if not obj or "values" not in obj:
                        # Maybe error object
                        if isinstance(obj, dict) and obj.get("status") == "error":
                            logger.warning("%s %s: %s", sym, interval, obj)
                        continue
                    closes = [float(v["close"]) for v in reversed(obj["values"]) if "close" in v]
                    out[sym] = closes

            return out
        except requests.HTTPError as e:
            # 429 or others
            txt = ""
            try:
                txt = r.text[:200]
            except Exception:
                pass
            msg = f"{e} | {txt}"
            if "429" in msg or "credits" in msg.lower():
                if attempt < retries:
                    logger.warning("Rate limited. Sleeping %ss then retry...", sleep_on_429)
                    time.sleep(sleep_on_429)
                    continue
            logger.exception("HTTP error @%s: %s", interval, msg)
            return {}
        except Exception as e:
            logger.exception("Fetch error @%s: %s", interval, e)
            return {}
        finally:
            # safety pacing between intervals
            time.sleep(1)
    return {}

def combine_label(a: str, b: str, la: str, lb: str) -> str:
    if a == b and a in ("LONG", "SHORT"):
        return a
    if a == "N/A" and b == "N/A":
        return "N/A"
    return f"Mixed ({la}:{a}, {lb}:{b})"

def send_telegram(text: str) -> None:
    if not BOT_TOKEN or not CHAT_ID:
        logger.info("No Telegram env; printing only.\n%s", text)
        return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        r = requests.post(url, json={"chat_id": CHAT_ID, "text": text}, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        logger.info("Telegram: sent")
    except Exception as e:
        logger.exception("Telegram error: %s", e)

def main():
    if not TD_KEY:
        raise RuntimeError("Missing TD_KEY env")

    td_list = list(SYMBOLS.values())

    # Fetch each interval in <=6 requests (under 8/min). Each request returns all symbols.
    data_by_iv: Dict[str, Dict[str, List[float]]] = {}
    for iv in INTERVALS:
        data_by_iv[iv] = fetch_timeframe(iv, td_list)

    # Build signals per symbol
    report_lines: List[str] = []
    any_trade_signal = False

    header = [f"{TITLE}"]
    ts = now_vn().strftime("%Y-%m-%d %H:%M:%S")
    header.append(f"⏱ {ts}")
    report_lines.extend(header)

    for disp, td_sym in SYMBOLS.items():
        iv_map = data_by_iv
        sig_15 = classify(iv_map.get("15min", {}).get(td_sym, []))
        sig_30 = classify(iv_map.get("30min", {}).get(td_sym, []))
        sig_1h = classify(iv_map.get("1h", {}).get(td_sym, []))
        sig_2h = classify(iv_map.get("2h", {}).get(td_sym, []))
        sig_4h = classify(iv_map.get("4h", {}).get(td_sym, []))
        sig_1d = classify(iv_map.get("1day", {}).get(td_sym, []))

        # Determine if this symbol has any actionable signal
        symbol_has_trade = any(s in ("LONG", "SHORT") for s in [sig_15, sig_30, sig_1h, sig_2h, sig_4h, sig_1d])
        if not symbol_has_trade:
            continue  # skip pure SIDEWAY/N/A symbols

        any_trade_signal = True
        report_lines.append("")
        report_lines.append(f"==={disp}===")
        report_lines.append(f"15m-30m: {combine_label(sig_15, sig_30)}")
        report_lines.append(f"1h-2h:   {combine_label(sig_1h, sig_2h)}")
        report_lines.append(f"4h-1D:   {combine_label(sig_4h, sig_1d)}")

    if not any_trade_signal:
        logger.info("All symbols SIDEWAY/N/A → skip Telegram.")
        return

    message = "\n".join(report_lines)
    logger.info("Message:\n%s", message)
    send_telegram(message)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Fatal: %s", e)
