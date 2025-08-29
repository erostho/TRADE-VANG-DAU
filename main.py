# main.py
# -*- coding: utf-8 -*-

import os, sys, json, time, math, logging, traceback
from datetime import datetime, timedelta, timezone
import requests

# =============== CONFIG ===============
API_KEY          = os.getenv("TWELVE_DATA_KEY", "")
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

BASE_URL = "https://api.twelvedata.com/time_series"
TZ_NAME  = os.getenv("TZ", "Asia/Ho_Chi_Minh")

RPM = int(os.getenv("RPM", "7"))  # throttle per minute

SYMBOLS = {
    "Bitcoin":   "BTC/USD",
    "Ethereum":  "ETH/USD",
    "XAU/USD (Gold)": "XAU/USD",
    "WTI Oil":   "CL",
    "USD/JPY":   "USD/JPY",
}

GROUP_LABELS = [("30m-1H", ["30min", "1h"]),
                ("2H-4H",  ["2h", "4h"])]

ALL_INTERVALS = ["30min", "1h", "2h", "4h", "1day"]

ROUND_ROBIN = [
    ["30min"],
    ["1h", "2h"],
    ["4h"],
]

STATE_PATH = "state.json"

def load_state():
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_state(st):
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False)
    os.replace(tmp, STATE_PATH)

STATE = load_state()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("tradebot")

try:
    import zoneinfo
    VN_TZ = zoneinfo.ZoneInfo(TZ_NAME)
except Exception:
    VN_TZ = timezone(timedelta(hours=7))

def now_vn():
    return datetime.now(VN_TZ)

def is_candle_close(interval: str, t: datetime) -> bool:
    m = t.minute; h = t.hour
    if interval == "30min": return (m % 30) == 0
    if interval == "1h":    return m == 0
    if interval == "2h":    return (h % 2 == 0) and (m == 0)
    if interval == "4h":    return (h % 4 == 0) and (m == 0)
    return False

def should_fetch_daily(t: datetime) -> bool:
    return (t.hour == 7 and t.minute == 0)

def td_get(symbol: str, interval: str, count: int = 200):
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": count,
        "apikey": API_KEY,
        "format": "JSON",
        "source": "docs",
        "dp": 8,
        "order": "desc",
        "ext_hours": "false",
    }
    res = requests.get(BASE_URL, params=params, timeout=20)
    if res.status_code != 200:
        raise RuntimeError(f"TD HTTP {res.status_code}: {res.text[:200]}")
    data = res.json()
    if "values" not in data:
        raise RuntimeError(f"TD error {symbol} {interval}: {data}")
    out = []
    for v in data["values"]:
        try:
            out.append({
                "datetime": v["datetime"],
                "open": float(v["open"]),
                "high": float(v["high"]),
                "low":  float(v["low"]),
                "close":float(v["close"]),
            })
        except Exception:
            pass
    return out

def sma(closes, n):
    if len(closes) < n: return None
    return sum(closes[:n]) / n

def classify_trend(closes):
    if len(closes) < 25: return "N/A"
    sma9  = sma(closes, 9)
    sma21 = sma(closes, 21)
    if sma9 is None or sma21 is None: return "N/A"
    if sma9 > sma21 * 1.001: return "LONG"
    if sma9 < sma21 * 0.999: return "SHORT"
    return "SIDEWAY"

def atr14(ohlc):
    if len(ohlc) < 15: return None
    trs = []
    prev_close = ohlc[1]["close"]
    for i in range(0, 15):
        h = ohlc[i]["high"]; l = ohlc[i]["low"]
        c_prev = ohlc[i+1]["close"] if i+1 < len(ohlc) else prev_close
        tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
        trs.append(tr)
    return sum(trs) / len(trs)

def cache_get(sym: str, interval: str):
    return STATE.get(f"{sym}:{interval}")

def cache_put(sym: str, interval: str, trend: str, entry=None, sl=None, tp=None, asof=None):
    key = f"{sym}:{interval}"
    STATE[key] = {
        "trend": trend,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "asof": asof or now_vn().strftime("%Y-%m-%d %H:%M:%S"),
    }

def telegram_send(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram env missing, skip send"); return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text,
               "parse_mode": "HTML", "disable_web_page_preview": True}
    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code != 200:
            log.warning("Telegram send fail: %s", r.text[:200])
    except Exception as e:
        log.warning("Telegram error: %s", e)

def fmt_num(x):
    if x is None: return "N/A"
    if isinstance(x, (int,)) or abs(x) >= 1000:
        return f"{x:,.2f}"
    return f"{x:.4f}"

def build_message(results, plan1h):
    t = now_vn().strftime("%Y-%m-%d %H:%M:%S (VN)")
    lines = []
    lines.append("💵 TRADE GOODS")
    lines.append(f"🕰 {t}")
    lines.append("")
    for name in SYMBOLS.keys():
        sym = SYMBOLS[name]
        rs  = results.get(sym, {})
        lines.append(f"==={name}===")
        g1 = ["30min","1h"]
        if rs.get("30min") == rs.get("1h"):
            lines.append(f"30m-1H: {rs.get('30min','N/A')}")
        else:
            lines.append(f"30m-1H: Mixed ({rs.get('30min','N/A')},{rs.get('1h','N/A')})")
        g2 = ["2h","4h"]
        if rs.get("2h") == rs.get("4h"):
            lines.append(f"2H-4H: {rs.get('2h','N/A')}")
        else:
            lines.append(f"2H-4H: Mixed ({rs.get('2h','N/A')},{rs.get('4h','N/A')})")
        lines.append(f"1D: {rs.get('1day','N/A')}")
        p1 = plan1h.get(sym)
        if p1 and (p1.get("entry") is not None):
            lines.append(f"Entry {fmt_num(p1['entry'])} | SL {fmt_num(p1['sl'])} | TP {fmt_num(p1['tp'])}")
        lines.append("")
    return "\n".join(lines).rstrip()

def run_once():
    t = now_vn()
    minute_slot = t.minute % len(ROUND_ROBIN)
    planned_intervals = set(ROUND_ROBIN[minute_slot])
    if should_fetch_daily(t):
        planned_intervals.add("1day")

    results = {}; plan1h = {}; calls = 0

    for name, sym in SYMBOLS.items():
        results[sym] = {}
        for iv in ALL_INTERVALS:
            try:
                can_fetch = (iv in planned_intervals and
                             (iv == "1day" and should_fetch_daily(t) or iv != "1day") and
                             (iv == "1day" or is_candle_close(iv, t)))
                if can_fetch and calls < RPM:
                    ohlc = td_get(sym, iv, count=200)
                    calls += 1
                    closes = [bar["close"] for bar in ohlc]
                    trend = classify_trend(closes)
                    results[sym][iv] = trend
                    cache_put(sym, iv, trend)
                    if iv == "1h":
                        atr = atr14(ohlc); last = ohlc[0]["close"]
                        if atr and last:
                            if trend == "LONG":
                                entry, sl, tp = last, last-atr, last+atr
                            elif trend == "SHORT":
                                entry, sl, tp = last, last+atr, last-atr
                            else:
                                entry=sl=tp=last
                            cache_put(sym, "1h_plan", trend, entry, sl, tp)
                if not results[sym].get(iv):
                    c = cache_get(sym, iv)
                    results[sym][iv] = c["trend"] if c else "N/A"
                if iv == "1h" and sym not in plan1h:
                    pc = cache_get(sym, "1h_plan")
                    if pc:
                        plan1h[sym] = {"entry": pc.get("entry"),"sl": pc.get("sl"),"tp": pc.get("tp")}
            except Exception as e:
                log.warning("Fetch fail %s %s: %s", sym, iv, e)
                c = cache_get(sym, iv)
                results[sym][iv] = c["trend"] if c else "N/A"
                if iv == "1h" and sym not in plan1h:
                    pc = cache_get(sym, "1h_plan")
                    if pc:
                        plan1h[sym] = {"entry": pc.get("entry"),"sl": pc.get("sl"),"tp": pc.get("tp")}
        if sym not in plan1h:
            plan1h[sym] = {"entry": None,"sl": None,"tp": None}

    save_state(STATE)
    msg = build_message(results, plan1h)
    telegram_send(msg)

def main():
    try:
        run_once()
    except Exception as e:
        log.error("Fatal: %s", e)
        try: telegram_send(f"⚠️ Bot error: {e}")
        except: pass

if __name__ == "__main__":
    main()
          
