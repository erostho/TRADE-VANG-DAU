#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main bot for trend snapshot + 1H Entry/SL/TP suggestions.
Environment:
  - TD_API_KEY: TwelveData API key
  - TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
  - TZ=Asia/Ho_Chi_Minh (recommended)
  - LOG_LEVEL=INFO
Runtime notes:
  * Free TwelveData plan is 8 req/min and 800 req/day.
    We cache responses to /tmp/td_cache.json and auto-throttle to 7 rpm.
"""
import os, json, time, math, logging
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import requests

# ============== Logging ==============
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("bot")

# ============== ENV ==============
TD_API_KEY  = os.getenv("TWELVE_DATA_KEY", "")
BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID     = os.getenv("TELEGRAM_CHAT_ID", "")
TITLE_PREFIX = "💵 TRADE GOODS"
TZ_NAME     = os.getenv("TZ", "Asia/Ho_Chi_Minh")

# ============== TwelveData client with cache & throttle ==============
TD_BASE = "https://api.twelvedata.com/time_series"
HEADERS = {"User-Agent": "trend-bot/1.1"}
CACHE_FILE = "/tmp/td_cache.json"
RPM_LIMIT = 7  # requests per minute (safe under 8)
_last_min = [0, 0]  # minute, count

def now_tz():
    if TZ_NAME == "Asia/Ho_Chi_Minh":
        return datetime.now(timezone(timedelta(hours=7)))
    return datetime.now()

def _throttle():
    """Keep at most RPM_LIMIT requests per minute."""
    global _last_min
    m = int(time.time() // 60)
    if _last_min[0] != m:
        _last_min = [m, 0]
    else:
        if _last_min[1] >= RPM_LIMIT:
            sleep_s = 60 - (time.time() % 60) + 1
            log.info("Throttle: sleeping %.1fs to respect RPM<=%d", sleep_s, RPM_LIMIT)
            time.sleep(sleep_s)
    _last_min[1] += 1

def _load_cache():
    try:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_cache(c):
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(c, f)
    except Exception:
        pass

def td_fetch(symbol: str, interval: str, outputsize: int = 120):
    """Fetch OHLC series; cached 55s to avoid bursts."""
    if not TD_API_KEY:
        raise RuntimeError("Missing TD_API_KEY")
    cache = _load_cache()
    key = f"{symbol}|{interval}|{outputsize}"
    now = time.time()
    ent = cache.get(key)
    if ent and now - ent.get("ts", 0) <= 55:
        return ent["data"]

    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TD_API_KEY,
        "order": "asc",
        "format": "JSON",
    }
    _throttle()
    r = requests.get(TD_BASE, params=params, headers=HEADERS, timeout=30)
    try:
        j = r.json()
    except Exception:
        log.warning("TD raw: %s", r.text[:300])
        raise
    if "values" not in j:
        raise RuntimeError(f"TD error {symbol}-{interval}: {j}")
    cache[key] = {"ts": now, "data": j}
    _save_cache(cache)
    return j

# ============== Indicators ==============
def sma(arr, n):
    if len(arr) < n or n <= 0: return None
    return sum(arr[-n:]) / n

def ema(arr, n):
    if len(arr) < n or n <= 0: return None
    k = 2 / (n + 1)
    e = sum(arr[:n]) / n
    for x in arr[n:]:
        e = x * k + e * (1 - k)
    return e

def atr(highs, lows, closes, n=14):
    if len(closes) < n + 1: return None
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i]  - closes[i - 1]),
        )
        trs.append(tr)
    if len(trs) < n: return None
    # Wilder's smoothing
    atr_val = sum(trs[:n]) / n
    for tr in trs[n:]:
        atr_val = (atr_val * (n - 1) + tr) / n
    return atr_val

# ============== Trend logic ==============
def detect_trend(closes):
    """
    Simple regime:
      LONG  if close > EMA20 and EMA20 > EMA50 and slope(EMA20)>0
      SHORT if close < EMA20 and EMA20 < EMA50 and slope(EMA20)<0
      else SIDEWAY
    """
    if len(closes) < 60: return "N/A"
    e20 = ema(closes, 20); e50 = ema(closes, 50)
    if (e20 is None) or (e50 is None): return "N/A"
    slope = e20 - ema(closes[:-1], 20)
    last = closes[-1]
    if last > e20 and e20 > e50 and slope > 0:
        return "LONG"
    if last < e20 and e20 < e50 and slope < 0:
        return "SHORT"
    return "SIDEWAY"

# ============== Entry/SL/TP (1H) ==============
def one_hour_plan(ohlc):
    """
    Given TD series dict, compute entry/sl/tp from last bar & ATR(14).
    SL = 1.5 * ATR, TP = 2.5 * ATR (≈ RR 1:1.67).
    Direction comes from detect_trend on 1H closes.
    """
    vals = ohlc["values"]
    highs = [float(v["high"]) for v in vals]
    lows  = [float(v["low"])  for v in vals]
    closes= [float(v["close"])for v in vals]
    direction = detect_trend(closes)
    if direction in ("N/A", "SIDEWAY"):
        return direction, None
    a = atr(highs, lows, closes, 14)
    if a is None:
        return direction, None
    entry = closes[-1]
    sl_mult = float(os.getenv("SL_ATR_MULT", "1.5"))
    tp_mult = float(os.getenv("TP_ATR_MULT", "2.5"))
    if direction == "LONG":
        sl = entry - sl_mult * a
        tp = entry + tp_mult * a
    else:
        sl = entry + sl_mult * a
        tp = entry - tp_mult * a
    return direction, {"entry": entry, "sl": sl, "tp": tp, "atr": a}

def fmt_price(x):
    try:
        if abs(x) >= 100:
            return f"{x:.2f}"
        return f"{x:.3f}"
    except Exception:
        return "N/A"

# ============== Telegram ==============
def tg_send(text: str):
    if not BOT_TOKEN or not CHAT_ID:
        log.warning("Missing TELEGRAM_BOT_TOKEN/CHAT_ID; skip Telegram.")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": CHAT_ID, "text": text}, timeout=30)
        r.raise_for_status()
        log.info("Telegram: sent")
    except Exception as e:
        log.exception("Telegram error: %s", e)

# ============== Symbol config ==============
SYMBOLS = {
    "Bitcoin":   ["BTC/USD"],
    "Ethereum":  ["ETH/USD"],
    "XAU/USD (Gold)": ["XAU/USD"],
    "WTI Oil":   ["WTI/USD", "CL", "USOIL", "WTICOUSD"],  # try in order
    "USD/JPY":   ["USD/JPY"],
}
# Intervals for trend blocks
BLOCKS = [
    ("30m-1H", [("30m","30min"), ("1h","1h")]),
    ("2H-4H",  [("2h","2h"), ("4h","4h")]),
    ("1D",     [("1d","1day")]),
]

def first_working_symbol(candidates):
    for s in candidates:
        try:
            td_fetch(s, "1h", 5)  # cheap probe
            return s
        except Exception as e:
            log.debug("Probe fail %s: %s", s, e)
    return None

def trend_for_intervals(symbol):
    res = {}
    for label, ivs in BLOCKS:
        parts = []
        mix = set()
        for nice, iv in ivs:
            try:
                j = td_fetch(symbol, iv, 120)
                closes = [float(v["close"]) for v in j["values"]]
                t = detect_trend(closes)
            except Exception as e:
                log.warning("No data for %s-%s: %s", symbol, iv, e)
                t = "N/A"
            mix.add(t)
            parts.append(f"{nice}:{t}")
        if len(ivs) == 1:
            res[label] = parts[0].split(":")[1]
        else:
            if len(mix) == 1:
                res[label] = next(iter(mix))
            else:
                res[label] = f"Mixed ({', '.join(parts)})"
    return res

# ============== Main ==============
def main():
    lines = [TITLE_PREFIX, f"⏱ {now_tz().strftime('%Y-%m-%d %H:%M:%S')}"]
    any_signal = False

    for title, cands in SYMBOLS.items():
        symbol = first_working_symbol(cands)
        if not symbol:
            lines.append(f"\\n==={title}===\\nNo supported symbol on free plan.")
            continue

        # Trend blocks
        tb = trend_for_intervals(symbol)

        # 1H plan
        plan = None; direction = "N/A"
        try:
            j1h = td_fetch(symbol, "1h", 200)
            direction, plan = one_hour_plan(j1h)
        except Exception as e:
            log.warning("Plan 1H fail %s: %s", symbol, e)

        # Compose section
        sec = [f"\\n==={title}==="]
        for label, _ in BLOCKS:
            sec.append(f"{label}: {tb.get(label,'N/A')}")
        if plan:
            any_signal = True if direction in ("LONG","SHORT") else any_signal
            sec.append(
                f"1H plan: {direction} | Entry {fmt_price(plan['entry'])} | "
                f"SL {fmt_price(plan['sl'])} | TP {fmt_price(plan['tp'])} "
                f"(ATR14 {fmt_price(plan['atr'])})"
            )
        else:
            sec.append(f"1H plan: {direction}")
        lines.append("\\n".join(sec))

    msg = "\\n".join(lines)
    # Only send if at least one non-sideway signal exists
    if any_signal:
        tg_send(msg)
    else:
        log.info("Only SIDEWAY/N/A signals -> not sending.\\n%s", msg)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.exception("Fatal: %s", e)
        # push error to Telegram to debug
        try:
            tg_send(f"{TITLE_PREFIX}\\n❌ ERROR: {e}")
        except Exception:
            pass
        raise
