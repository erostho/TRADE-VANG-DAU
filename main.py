#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, time, math, logging, datetime as dt
from typing import Dict, Any, List, Tuple, Optional
import urllib.parse, urllib.request

# ============== LOG ==============
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("main")

# ============== CONFIG ==============
API_KEY          = os.getenv("TWELEVE_DATA_KEY", "") or os.getenv("TWELVE_DATA_KEY", "")
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
BASE_URL         = "https://api.twelvedata.com/time_series"
TZ_NAME          = os.getenv("TZ", "Asia/Ho_Chi_Minh")
RPM              = int(os.getenv("RPM", "7"))  # max calls per minute
REDIS_URL        = os.getenv("REDIS_URL", "")

# Hiển thị -> API symbol (CHỈ value được dùng gọi API & cache)
SYMBOLS: Dict[str, str] = {
    "Bitcoin":         "BTC/USD",
    "Ethereum":        "ETH/USD",
    "XAU/USD (Gold)":  "XAU/USD",
    "WTI Oil":         "WTI/USD",   # nếu rỗng hãy thử "WTICO/USD"
    "USD/JPY":         "USD/JPY",
}

# nhóm khung – để render dòng gọn
PAIR_GROUPS: List[Tuple[str,str]] = [("30min","1h"), ("2h","4h")]
DAILY_TF = "1day"

# ============== TZ ==============
try:
    import zoneinfo
    VN_TZ = zoneinfo.ZoneInfo(TZ_NAME)
except Exception:
    # fallback UTC+7
    class _Fixed(dt.tzinfo):
        def utcoffset(self, _): return dt.timedelta(hours=7)
        def tzname(self, _): return "UTC+7"
        def dst(self, _): return dt.timedelta(0)
    VN_TZ = _Fixed()

def now_vn() -> dt.datetime:
    return dt.datetime.now(VN_TZ)

# ============== CACHE ==============
class CacheIF:
    def get(self, k: str) -> Any: ...
    def set(self, k: str, v: Any, ex: int = None) -> None: ...

class FileCache(CacheIF):
    def __init__(self, path="/tmp/cache.json"):
        self.path = path
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except Exception:
            self.data = {}
    def _flush(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f)
        except Exception:
            pass
    def get(self, k):
        item = self.data.get(k)
        if not item: return None
        if "exp" in item and item["exp"] and time.time() > item["exp"]:
            self.data.pop(k, None)
            self._flush()
            return None
        return item["val"]
    def set(self, k, v, ex=None):
        exp = time.time() + ex if ex else None
        self.data[k] = {"val": v, "exp": exp}
        self._flush()

_cache: CacheIF
if REDIS_URL:
    try:
        import redis
        r = redis.from_url(REDIS_URL, decode_responses=True)
        class RedisCache(CacheIF):
            def get(self, k):
                s = r.get(k)
                return json.loads(s) if s else None
            def set(self, k, v, ex=None):
                r.set(k, json.dumps(v), ex=ex)
        _cache = RedisCache()
        log.info("Cache: Redis")
    except Exception as e:
        log.warning("Redis not available, fallback to file cache: %s", e)
        _cache = FileCache()
else:
    _cache = FileCache()

def api_symbol(display_name: str) -> str:
    return SYMBOLS[display_name]

def cache_key(api_sym: str, tf: str) -> str:
    return f"{api_sym}:{tf}"

def store(k: str, v: Any, ttl: int):
    _cache.set(k, v, ex=ttl)

def load(k: str) -> Any:
    return _cache.get(k)

# ============== HTTP helper ==============
def http_get(url: str, params: Dict[str, str]) -> Dict[str, Any]:
    q = urllib.parse.urlencode(params)
    full = f"{url}?{q}"
    req = urllib.request.Request(full, headers={"User-Agent":"Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        text = resp.read().decode("utf-8", "ignore")
    return json.loads(text)

# ============== DATA / INDICATORS ==============
def fetch_series(symbol_api: str, interval: str, outputsize: int=120) -> Optional[List[Dict[str, Any]]]:
    params = {
        "symbol": symbol_api,
        "interval": interval,
        "apikey": API_KEY,
        "outputsize": str(outputsize),
        "order": "desc",
        "format": "JSON",
        "dp": "8",
    }
    try:
        data = http_get(BASE_URL, params)
    except Exception as e:
        log.warning("HTTP fail %s %s: %s", symbol_api, interval, e)
        return None

    if isinstance(data, dict) and data.get("status") == "error":
        log.warning("TD error %s %s: %s", symbol_api, interval, data)
        return None

    vals = data.get("values")
    if not vals: 
        return None

    # normalize to floats
    for v in vals:
        for k in ("open","high","low","close"):
            v[k] = float(v[k])
    return vals  # newest first

def sma(seq: List[float], n: int) -> float:
    n = min(n, len(seq))
    return sum(seq[:n]) / max(n,1)

def atr14(ohlc: List[Dict[str,Any]]) -> float:
    # expects newest first
    n = min(15, len(ohlc))
    if n < 15:
        return None
    trs = []
    prev_close = ohlc[::-1][0]["close"]  # oldest close
    for x in ohlc[::-1][1:15]:           # 14 periods
        high, low = x["high"], x["low"]
        tr = max(high-low, abs(high-prev_close), abs(low-prev_close))
        trs.append(tr)
        prev_close = x["close"]
    return sum(trs)/len(trs) if trs else None

def infer_trend(ohlc: List[Dict[str,Any]]) -> str:
    # dùng SMA(5) vs SMA(20) – đủ ổn định, sideway nếu chênh < 0.05%
    closes = [x["close"] for x in ohlc]
    if len(closes) < 20:
        return "N/A"
    s5, s20 = sma(closes,5), sma(closes,20)
    if s20 == 0: return "N/A"
    diff = (s5 - s20)/s20
    if diff > 0.0005:   # >0.05%
        return "LONG"
    if diff < -0.0005:
        return "SHORT"
    return "SIDEWAY"

def compute_plan_1h(ohlc_1h: List[Dict[str,Any]]) -> Optional[Dict[str,Any]]:
    if not ohlc_1h: return None
    trend = infer_trend(ohlc_1h)
    if trend not in ("LONG","SHORT"):
        return None
    entry = ohlc_1h[0]["close"]  # last close (newest first)
    a14 = atr14(ohlc_1h)
    if not a14 or a14 <= 0: 
        return None
    if trend == "LONG":
        sl = entry - a14
        tp = entry + a14
    else:
        sl = entry + a14
        tp = entry - a14
    return {"trend":"", "entry":entry, "sl":sl, "tp":tp, "atr":a14}

# ============== ROUND-ROBIN ==============
def make_rr_list() -> List[Tuple[str,str]]:
    lst = []
    for name in SYMBOLS.keys():
        for tf in ["30min","1h","2h","4h"]:
            lst.append((name, tf))
    return lst

def rr_index_key() -> str:
    return "rr:index"

def rr_next_batch(limit: int) -> List[Tuple[str,str]]:
    all_pairs = make_rr_list()
    idx = load(rr_index_key()) or 0
    out = []
    for i in range(limit):
        out.append(all_pairs[(idx+i) % len(all_pairs)])
    store(rr_index_key(), (idx+limit) % len(all_pairs), ttl=24*3600)
    return out

# ============== FETCH/STORE HELPERS ==============
def get_or_fetch(api_sym: str, tf: str, ttl: int) -> Optional[Dict[str,Any]]:
    k = cache_key(api_sym, tf)
    val = load(k)
    if val:
        return val
    series = fetch_series(api_sym, tf)
    if not series: 
        return None
    # build tiny payload: trend + last_close + maybe ATR (for 1h)
    payload = {
        "trend": infer_trend(series),
        "last": series[0]["close"],
    }
    if tf == "1h":
        p1 = compute_plan_1h(series)
        if p1:
            payload.update({
                "entry": p1["entry"],
                "sl": p1["sl"],
                "tp": p1["tp"],
                "atr": p1["atr"],
            })
    # TTL: 30m=25m, 1h=50m, 2h=110m, 4h=230m, 1day=26h
    tf_ttl = {
        "30min": 25*60, "1h": 50*60, "2h": 110*60, "4h": 230*60, "1day": 26*3600
    }.get(tf, 3600)
    store(k, payload, tf_ttl)
    return payload

# ============== TELEGRAM ==============
def telegram_send(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.info("Telegram disabled (missing env)")
        return
    try:
        api = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
        }
        req = urllib.request.Request(api, data=urllib.parse.urlencode(data).encode("utf-8"))
        urllib.request.urlopen(req, timeout=15).read()
        log.info("Telegram: sent")
    except Exception as e:
        log.warning("Telegram send fail: %s", e)

# ============== MAIN CYCLE ==============
def fetch_daily_if_7am():
    now = now_vn()
    # chỉ fetch trong phút 0–1 để tránh spam
    if now.hour == 7 and now.minute <= 1:
        for disp, api_sym in SYMBOLS.items():
            get_or_fetch(api_sym, DAILY_TF, ttl=26*3600)

def throttled_round_robin():
    # mỗi lần chạy chỉ gọi tối đa (RPM) cái; trừ hao 1–2 cho fallback
    budget = max(1, RPM - 1)
    batch = rr_next_batch(budget)
    log.info("Batch: %s", batch)
    used = 0
    for disp, tf in batch:
        api_sym = api_symbol(disp)
        # chỉ fetch khi nến "gần đóng" để giảm noise
        # (nhưng vẫn cache nếu chưa có)
        get_or_fetch(api_sym, tf, ttl=3600)
        used += 1
    return used

def render_message() -> str:
    now = now_vn().strftime("%Y-%m-%d %H:%M:%S (VN)")
    lines = []
    lines.append("💵 TRADE GOODS")
    lines.append(f"🕰 {now}")
    lines.append("")
    for disp in SYMBOLS.keys():
        api_sym = api_symbol(disp)
        lines.append(f"==={disp}===")
        # group 30m-1h
        g1_vals = []
        g1_show = []
        for tf in ["30min","1h"]:
            v = load(cache_key(api_sym, tf))
            if not v:
                v = get_or_fetch(api_sym, tf, ttl=3600)  # fallback 1 lần
            trend = v.get("trend") if v else "N/A"
            g1_vals.append(trend)
            g1_show.append(f"{tf.replace('min','m')}:{trend}")
        if g1_vals[0] == g1_vals[1] and g1_vals[0] in ("LONG","SHORT","SIDEWAY"):
            lines.append(f"30m–1H: {g1_vals[0]}")
        else:
            lines.append(f"30m–1H: Mixed ({', '.join(g1_show)})")

        # group 2h-4h
        g2_vals = []
        g2_show = []
        for tf in ["2h","4h"]:
            v = load(cache_key(api_sym, tf))
            if not v:
                v = get_or_fetch(api_sym, tf, ttl=7200)
            trend = v.get("trend") if v else "N/A"
            g2_vals.append(trend)
            g2_show.append(f"{tf}:{trend}")
        if g2_vals[0] == g2_vals[1] and g2_vals[0] in ("LONG","SHORT","SIDEWAY"):
            lines.append(f"2H–4H: {g2_vals[0]}")
        else:
            lines.append(f"2H–4H: Mixed ({', '.join(g2_show)})")

        # 1D (chỉ fetch 7h; còn lại lấy cache)
        d = load(cache_key(api_sym, DAILY_TF))
        day_trend = d.get("trend") if d else "N/A"
        lines.append(f"1D: {day_trend}")

        # 1H Entry/SL/TP – chỉ khi trend 1h = LONG/SHORT
        v1h = load(cache_key(api_sym, "1h"))
        if v1h and v1h.get("entry") and (load(cache_key(api_sym, "1h")).get("trend") in ("LONG","SHORT") or infer_trend([{"close":v1h["last"],"open":v1h["last"],"high":v1h["last"],"low":v1h["last"]}] ) ):
            entry = v1h["entry"]; sl = v1h["sl"]; tp = v1h["tp"]
            lines.append(f"Entry {entry:.2f} | SL {sl:.2f} | TP {tp:.2f}")
        lines.append("")
    return "\n".join(lines).rstrip()

def main():
    if not API_KEY:
        log.error("Missing TwelveData API key")
    fetch_daily_if_7am()
    throttled_round_robin()
    msg = render_message()
    telegram_send(msg)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.exception("Fatal: %s", e)
        sys.exit(1)
