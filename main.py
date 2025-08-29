#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MAIN (round-robin + daily-at-7am cache)
- Keeps technical logic (EMA20/50 trend + 1H ATR Entry/SL/TP)
- Round-robin per minute (slot 0: 30m; slot 1: 1h+2h; slot 2: 4h)
- Cap API per run: MAX_CALLS_PER_RUN (default 7)
- Daily (1D) fetch only at 07:00 VN; otherwise use cached 1D per symbol
ENV:
  TWELVEDATA_API_KEY (or TD_API_KEY)
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
  MAX_CALLS_PER_RUN (default 7)
  LOG_LEVEL (default INFO)
"""

import os, json, time, logging, math
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import requests

# ========== CONFIG ==========
VN_TZ = timezone(timedelta(hours=7))
STATE_FILE = "state_rr.json"
MAX_CALLS_PER_RUN = int(os.getenv("MAX_CALLS_PER_RUN", "7"))

SYMBOLS = {
    "Bitcoin": "BTC/USD",
    "Ethereum": "ETH/USD",
    "XAU/USD (Gold)": "XAU/USD",
    "WTI Oil": "CL",
    "USD/JPY": "USD/JPY",
}
TD_API_KEY = os.getenv("TWELVE_DATA_KEY", "") or os.getenv("TD_API_KEY", "")
TD_BASE = "https://api.twelvedata.com"
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("main")

# ========== Utilities ==========
def now_vn() -> datetime:
    return datetime.now(VN_TZ)

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            return json.load(open(STATE_FILE, "r", encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_state(st: dict) -> None:
    try:
        json.dump(st, open(STATE_FILE, "w", encoding="utf-8"))
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
        log.warning("TD 429: wait 65s then retry once...")
        time.sleep(65)
        r = requests.get(url, params=p, timeout=30)
    r.raise_for_status()
    return r.json()

# ========== TA logic (kept) ==========
def ema_last(values: List[float], span: int) -> Optional[float]:
    if len(values) < span: return None
    k = 2 / (span + 1)
    e = values[0]
    for x in values[1:]:
        e = x * k + e * (1 - k)
    return e

def classify_trend(closes_desc: List[float]) -> str:
    """closes_desc: newest->oldest"""
    if not closes_desc or len(closes_desc) < 60: return "N/A"
    arr = list(reversed(closes_desc))  # oldest->newest
    e20_prev = ema_last(arr[:-1], 20); e20 = ema_last(arr, 20); e50 = ema_last(arr, 50)
    if None in (e20_prev, e20, e50): return "N/A"
    slope = e20 - e20_prev
    last = arr[-1]
    if last > e20 and e20 > e50 and slope > 0: return "LONG"
    if last < e20 and e20 < e50 and slope < 0: return "SHORT"
    return "SIDEWAY"

def fetch_trend(symbol: str, interval: str, outputsize: int = 200) -> str:
    try:
        j = td_get("/time_series", {
            "symbol": symbol, "interval": interval, "outputsize": outputsize, "order": "DESC"
        })
        vals = j.get("values", [])
        closes = [float(v["close"]) for v in vals]
        return classify_trend(closes)
    except Exception as e:
        log.warning("Fetch fail %s %s: %s", symbol, interval, e)
        return "N/A"

def atr14_from_series(vals_desc: List[dict]) -> Optional[float]:
    """Compute ATR14 from newest->oldest values list."""
    try:
        # convert to oldest->newest
        vals = list(reversed(vals_desc))
        highs = [float(v["high"]) for v in vals]
        lows  = [float(v["low"])  for v in vals]
        closes= [float(v["close"]) for v in vals]
        if len(closes) < 15: return None
        trs = []
        prev = closes[0]
        for i in range(1, len(closes)):
            tr = max(highs[i] - lows[i], abs(highs[i] - prev), abs(lows[i] - prev))
            trs.append(tr); prev = closes[i]
        if len(trs) < 14: return None
        atr = sum(trs[:14])/14.0
        for tr in trs[14:]:
            atr = (atr*13 + tr)/14.0
        return atr
    except Exception:
        return None

def one_hour_plan(symbol: str):
    """Return (direction, entry, sl, tp, atr) using 1h series; keep previous logic style."""
    try:
        j = td_get("/time_series", {"symbol": symbol, "interval": "1h", "outputsize": 200, "order": "DESC"})
        vals = j.get("values", [])
        if len(vals) < 60: return "N/A", None
        closes = [float(v["close"]) for v in vals]
        direction = classify_trend(closes)
        if direction in ("N/A", "SIDEWAY"): return direction, None
        a = atr14_from_series(vals)
        if a is None: return direction, None
        entry = closes[0]
        sl_mult = float(os.getenv("SL_ATR_MULT", "1.5"))
        tp_mult = float(os.getenv("TP_ATR_MULT", "2.5"))
        if direction == "LONG":
            sl = entry - sl_mult*a; tp = entry + tp_mult*a
        else:
            sl = entry + sl_mult*a; tp = entry - tp_mult*a
        return direction, {"entry": entry, "sl": sl, "tp": tp, "atr": a}
    except Exception as e:
        log.warning("1H plan fail %s: %s", symbol, e)
        return "N/A", None

# ========== Telegram ==========
def telegram_send(text: str):
    if not BOT_TOKEN or not CHAT_ID:
        log.warning("Telegram missing config")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": CHAT_ID, "text": text, "disable_web_page_preview": True}, timeout=30)
        r.raise_for_status()
        log.info("Telegram sent")
    except Exception as e:
        log.warning("Telegram error: %s", e)

# ========== Round-robin scheduler ==========
SLOT_TFS = {0: ["30min"], 1: ["1h","2h"], 2: ["4h"]}

def get_slot(now=None) -> int:
    now = now or now_vn()
    return now.minute % 3

def need_refresh_daily(now=None) -> bool:
    now = now or now_vn()
    return now.hour == 7 and now.minute == 0

def run_once():
    st = load_state()
    now = now_vn()
    slot = get_slot(now)
    tfs = SLOT_TFS[slot]

    # Daily 1D
    if need_refresh_daily(now):
        for sym in SYMBOLS:
            trend1d = fetch_trend(sym, "1day", 200)
            st[f"{sym}::1d"] = {"trend": trend1d, "ts": now.isoformat()}
            time.sleep(0.2)
        save_state(st)
        daily_refreshed = True
    else:
        daily_refreshed = False
    daily_cache = {s: st.get(f"{s}::1d", {}).get("trend", "N/A") for s in SYMBOLS}

    # Round-robin symbol window to cap calls/run
    rr_key = f"rr_slot_{slot}"
    start_idx = int(st.get(rr_key, 0)) % max(1, len(SYMBOLS))
    calls_per_sym = len(tfs)
    max_syms = max(1, MAX_CALLS_PER_RUN // calls_per_sym)
    order = SYMBOLS[start_idx:] + SYMBOLS[:start_idx]
    pick = order[:max_syms]

    log.info("Slot=%s TFs=%s start=%s pick=%s", slot, tfs, start_idx, ",".join(pick))

    # Collect result
    res: Dict[str, Dict[str, str]] = {s: {"30min":"...", "1h":"...", "2h":"...", "4h":"...", "1day": daily_cache.get(s,"N/A")} for s in SYMBOLS}

    # Fetch for picked symbols
    for s in pick:
        for tf in tfs:
            res[s][tf] = fetch_trend(s, tf, 200)
            time.sleep(0.2)

    # 1H plan (Entry/SL/TP) — optional: compute only when slot contains 1h to save calls
    plans: Dict[str, str] = {}
    if "1h" in tfs:
        for s in pick:
            dir1h, plan = one_hour_plan(s)
            if plan and dir1h in ("LONG","SHORT"):
                plans[s] = f"Entry {plan['entry']:.2f} | SL {plan['sl']:.2f} | TP {plan['tp']:.2f}"
            else:
                plans[s] = ""
    else:
        # reuse last plan from state if exists (optional)
        for s in SYMBOLS:
            plans[s] = st.get(f"{s}::plan1h", "")

    # persist rr index and latest plans
    st[rr_key] = (start_idx + len(pick)) % max(1, len(SYMBOLS))
    for s, text in plans.items():
        if text:
            st[f"{s}::plan1h"] = text
    save_state(st)

    # Build message
    lines = ["💵 TRADE GOODS", f"⏱ {now.strftime('%Y-%m-%d %H:%M:%S')} (VN)"]
    if daily_refreshed: lines.append("Daily 1D: refreshed ✅")
    lines.append("")

    for s in SYMBOLS:
        r = res[s]
        lines.append(f"==={s}===")
        lines.append(f"30m: {r['30min']}")
        lines.append(f"1h:  {r['1h']}")
        lines.append(f"2h:  {r['2h']}")
        lines.append(f"4h:  {r['4h']}")
        lines.append(f"1d:  {r['1day']}")
        if st.get(f"{s}::plan1h"):
            lines.append(st[f"{s}::plan1h"])
        lines.append("")

    msg = "\n".join(lines).rstrip()
    log.info("Message:\n%s", msg)
    telegram_send(msg)

def main():
    try:
        run_once()
    except Exception as e:
        log.exception("Fatal: %s", e)
        try:
            telegram_send(f"💵 TRADE GOODS\n❌ ERROR: {e}")
        except Exception:
            pass
        raise

if __name__ == "__main__":
    main()
