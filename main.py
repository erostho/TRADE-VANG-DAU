#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional

import requests

# ========== CONFIG ==========
VN_TZ = timezone(timedelta(hours=7))
STATE_FILE = "state_rr.json"      # lưu con trỏ round-robin & cache 1D
MAX_CALLS_PER_RUN = int(os.getenv("MAX_CALLS_PER_RUN", "7"))  # tối đa 7 API/run

# Symbols bạn muốn theo dõi (có thể chỉnh)
SYMBOLS: List[str] = [
    "BTC/USD",
    "ETH/USD",
    "XAU/USD",
    "WTI Oil": "CL",
    "USD/JPY",
]

# TwelveData
TD_API_KEY = os.getenv("TWELVE_DATA_KEY", "") or os.getenv("TD_API_KEY", "")
TD_BASE = "https://api.twelvedata.com"

# Telegram
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

# Logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s | %(levelname)s | %(message)s")

# ========== Utilities ==========
def now_vn() -> datetime:
    return datetime.now(VN_TZ)

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_state(st: dict) -> None:
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(st, f)
    except Exception:
        pass

def td_get(path: str, params: dict) -> dict:
    if not TD_API_KEY:
        raise RuntimeError("Missing TWELVEDATA_API_KEY / TD_API_KEY")
    p = dict(params or {})
    p["apikey"] = TD_API_KEY
    url = f"{TD_BASE.rstrip('/')}/{path.lstrip('/')}"
    r = requests.get(url, params=p, timeout=30)
    if r.status_code == 429:
        logging.warning("TD 429: wait 65s then retry once...")
        time.sleep(65)
        r = requests.get(url, params=p, timeout=30)
    r.raise_for_status()
    return r.json()

# ========== TA (simple) ==========
def ema_last(values: List[float], span: int) -> Optional[float]:
    if len(values) < span:
        return None
    k = 2 / (span + 1)
    e = values[0]
    for x in values[1:]:
        e = x * k + e * (1 - k)
    return e

def classify_trend(closes: List[float]) -> str:
    if not closes or len(closes) < 60:
        return "N/A"
    arr = list(reversed(closes))
    e20_prev = ema_last(arr[:-1], 20)
    e20 = ema_last(arr, 20)
    e50 = ema_last(arr, 50)
    if e20 is None or e50 is None or e20_prev is None:
        return "N/A"
    slope = e20 - e20_prev
    last = arr[-1]
    if last > e20 and e20 > e50 and slope > 0:
        return "LONG"
    if last < e20 and e20 < e50 and slope < 0:
        return "SHORT"
    return "SIDEWAY"

def fetch_trend(symbol: str, interval: str, outputsize: int = 120) -> str:
    try:
        j = td_get("/time_series", {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "order": "DESC",
        })
        vals = j.get("values", [])
        closes = [float(v["close"]) for v in vals]
        return classify_trend(closes)
    except Exception as e:
        logging.warning("Fetch fail %s %s: %s", symbol, interval, e)
        return "N/A"

# ========== Telegram ==========
def telegram_send(text: str):
    if not BOT_TOKEN or not CHAT_ID:
        logging.warning("Telegram missing config")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": CHAT_ID, "text": text, "disable_web_page_preview": True}, timeout=30)
        if r.status_code != 200:
            logging.warning("Telegram error %s: %s", r.status_code, r.text[:400])
        else:
            logging.info("Telegram: sent")
    except Exception as e:
        logging.warning("Telegram request error: %s", e)

# ========== Round-robin scheduling ==========
SLOT_TFS = {
    0: ["30min"],
    1: ["1h", "2h"],
    2: ["4h"],
}

def get_slot(now: Optional[datetime] = None) -> int:
    now = now or now_vn()
    return now.minute % 3

def get_daily_should_refresh(now: Optional[datetime] = None) -> bool:
    now = now or now_vn()
    return now.hour == 7 and now.minute == 0

def run_once():
    st = load_state()
    now = now_vn()
    slot = get_slot(now)
    tfs = SLOT_TFS[slot]

    rr_key = f"rr_slot_{slot}"
    start_idx = int(st.get(rr_key, 0)) % max(1, len(SYMBOLS))

    daily_refetched = False
    if get_daily_should_refresh(now):
        logging.info("Refetch daily (1D) for all symbols at 07:00 VN")
        for sym in SYMBOLS:
            trend_1d = fetch_trend(sym, "1day", 200)
            st[f"{sym}__1d"] = {"trend": trend_1d, "ts": now.isoformat()}
            time.sleep(0.2)
        save_state(st)
        daily_refetched = True

    daily_cache = {sym: (st.get(f"{sym}__1d", {}).get("trend") or "N/A") for sym in SYMBOLS}

    calls_per_symbol = len(tfs)
    max_symbols_this_run = max(1, MAX_CALLS_PER_RUN // calls_per_symbol)

    ordered = SYMBOLS[start_idx:] + SYMBOLS[:start_idx]
    pick_syms = ordered[:max_symbols_this_run]

    logging.info("Slot %s -> TFs %s | start_idx=%s | processing symbols: %s",
                 slot, tfs, start_idx, ", ".join(pick_syms))

    result_map: Dict[str, Dict[str, str]] = {s: {"30min": "...", "1h": "...", "2h": "...", "4h": "...", "1day": daily_cache.get(s, "N/A")} for s in SYMBOLS}

    for sym in pick_syms:
        for tf in tfs:
            trend = fetch_trend(sym, tf, 200)
            result_map[sym][tf] = trend
            time.sleep(0.2)

    st[rr_key] = (start_idx + len(pick_syms)) % max(1, len(SYMBOLS))
    save_state(st)

    header = ["💵 TRADE GOODS", f"⏱ {now.strftime('%Y-%m-%d %H:%M:%S')} (VN)"]
    lines: List[str] = header
    if daily_refetched:
        lines.append("Daily 1D: refreshed ✅")
    lines.append("")

    for sym in SYMBOLS:
        rec = result_map[sym]
        lines.append(f"==={sym}===")
        lines.append(f"30m: {rec['30min']}")
        lines.append(f"1h:  {rec['1h']}")
        lines.append(f"2h:  {rec['2h']}")
        lines.append(f"4h:  {rec['4h']}")
        lines.append(f"1d:  {rec['1day']}")
        lines.append("")

    msg = "\n".join(lines).rstrip()
    logging.info("Message:\n%s", msg)
    telegram_send(msg)

def main():
    try:
        run_once()
    except Exception as e:
        logging.exception("Fatal error: %s", e)
        try:
            telegram_send(f"💵 TRADE GOODS\n❌ ERROR: {e}")
        except Exception:
            pass
        raise

if __name__ == "__main__":
    main()
