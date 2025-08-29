# main.py
import os, json, time, math, logging, requests, datetime as dt
from typing import Dict, Tuple, List

# ============ CONFIG ============
API_KEY = os.getenv("TWELVE_DATA_KEY", "")
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT  = os.getenv("TELEGRAM_CHAT_ID", "")
TIMEZ    = os.getenv("TZ", "Asia/Ho_Chi_Minh")
RPM      = int(os.getenv("RPM", "7"))            # tối đa request/phút (<= 7 để an toàn free plan)
BASE_URL = "https://api.twelvedata.com/time_series"

# Lấy nến gần nhất (không đợi đóng)
FETCH_LIVE = True

# Symbols (hiển thị -> mã TwelveData)
symbols = {
    "Bitcoin": "BTC/USD",
    "Ethereum": "ETH/USD",
    "XAU/USD (Gold)": "XAU/USD",
    "WTI Oil": "CL",
    "USD/JPY": "USD/JPY",
}

# Các khung cần theo dõi
I_30M, I_1H, I_2H, I_4H, I_1D = "30min", "1h", "2h", "4h", "1day"

# File lưu cache/state
STATE_FILE = "state.json"      # lưu round-robin + cache
TIMEOUT_S  = 15

# ============ LOG ============
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("main")

# ============ STATE (RR + CACHE) ============
def _now_vn() -> dt.datetime:
    # không có pytz: dùng offset VN +7
    return dt.datetime.utcnow() + dt.timedelta(hours=7)

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    # default state
    rr = []  # hàng đợi round-robin
    for name in symbols.keys():
        rr += [(name, I_30M), (name, I_1H), (name, I_2H), (name, I_4H)]
    return {
        "rr": rr,              # hàng đợi còn lại lần chạy hiện tại
        "cursor": 0,           # con trỏ vòng lặp
        "cache": {},           # {(symbol, interval): {"trend": "...", "ts": "...", "entry":..., "sl":..., "tp":... (chỉ 1h)}}
        "daily": {"dateVN": "", "data": {}},  # 1D lưu theo ngày VN (YYYY-MM-DD)
    }

def save_state(st: dict):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(st, f, ensure_ascii=False)
    except Exception as e:
        log.warning("Cannot save state: %s", e)

# ============ TwelveData fetch ============
def fetch_series(symbol_code: str, interval: str) -> Tuple[List[dict], dict]:
    """
    Trả về (list_values_desc, last_bar)
    values: order=desc (mới -> cũ)
    last_bar: nến gần nhất (có thể đang chạy)
    """
    need = 60 if interval == I_1H else 50
    params = {
        "symbol": symbol_code,
        "interval": interval,
        "outputsize": max(need, 2),
        "order": "desc",
        "timezone": TIMEZ,
        "apikey": API_KEY,
    }
    r = requests.get(BASE_URL, params=params, timeout=TIMEOUT_S)
    r.raise_for_status()
    js = r.json()
    if "values" not in js:
        raise RuntimeError(f"TD no values: {js}")
    values = js["values"]
    if not values:
        raise RuntimeError("TD empty values")
    last_bar = values[0]
    return values, last_bar

# ============ TA helpers ============
def to_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def calc_atr(values_desc: List[dict], period: int = 14) -> float:
    """values_desc: list mới->cũ"""
    # chuyển sang cũ->mới cho dễ tính
    vals = list(reversed(values_desc))
    trs = []
    prev_close = None
    for v in vals:
        high = to_float(v["high"])
        low  = to_float(v["low"])
        close = to_float(v["close"])
        if math.isnan(high) or math.isnan(low) or math.isnan(close):
            continue
        if prev_close is None:
            tr = high - low
        else:
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
        prev_close = close
    if len(trs) < period:
        return float("nan")
    return sum(trs[-period:]) / period

def sma(closes: List[float], n: int) -> float:
    if len(closes) < n:
        return float("nan")
    return sum(closes[-n:]) / n

def detect_trend(values_desc: List[dict]) -> str:
    """Trend đơn giản: SMA20 vs SMA50 + hướng close gần nhất"""
    closes = [to_float(v["close"]) for v in reversed(values_desc)]
    if len(closes) < 50:
        return "N/A"
    s20 = sma(closes, 20)
    s50 = sma(closes, 50)
    if math.isnan(s20) or math.isnan(s50):
        return "N/A"
    if s20 > s50 * 1.001:
        return "LONG"
    if s20 < s50 * 0.999:
        return "SHORT"
    return "SIDEWAY"

# ============ Round-robin pick ============
def build_rr_list() -> List[Tuple[str, str]]:
    rr = []
    for name in symbols.keys():
        rr += [(name, I_30M), (name, I_1H), (name, I_2H), (name, I_4H)]
    return rr

def pick_batch(st: dict, limit: int) -> List[Tuple[str, str]]:
    rr = st.get("rr") or []
    cur = st.get("cursor", 0)
    if not rr:
        rr = build_rr_list()
        cur = 0
    batch = []
    for _ in range(limit):
        pair = rr[cur % len(rr)]
        batch.append(pair)
        cur += 1
    st["rr"] = rr
    st["cursor"] = cur
    return batch

# ============ Business rules ============
def update_interval_cache(st: dict, disp_name: str, interval: str):
    code = symbols[disp_name]
    try:
        values, last_bar = fetch_series(code, interval)
        trend = detect_trend(values)

        # update cache
        key = f"{disp_name}|{interval}"
        entry = st["cache"].get(key, {})
        entry.update({
            "trend": trend,
            "ts": _now_vn().isoformat(timespec="seconds"),
        })

        # Tính Entry/SL/TP cho 1H nếu có
        if interval == I_1H and trend in ("LONG", "SHORT"):
            close = to_float(last_bar["close"])
            atr = calc_atr(values, 14)
            if not math.isnan(close) and not math.isnan(atr):
                if trend == "LONG":
                    sl = close - atr
                    tp = close + atr
                else:
                    sl = close + atr
                    tp = close - atr
                entry["entry"] = round(close, 2)
                entry["sl"]    = round(sl, 2)
                entry["tp"]    = round(tp, 2)
            else:
                entry.pop("entry", None); entry.pop("sl", None); entry.pop("tp", None)
        st["cache"][key] = entry

    except requests.HTTPError as e:
        log.warning("TD HTTP %s for %s %s", e, disp_name, interval)
    except Exception as e:
        log.warning("Update fail %s %s: %s", disp_name, interval, e)

def update_daily_if_needed(st: dict):
    today_vn = _now_vn().strftime("%Y-%m-%d")
    is_7am_window = _now_vn().hour == 7  # chạy ở phút nào cũng được trong giờ 7
    # nếu đã có cùng ngày thì không fetch lại
    if st["daily"].get("dateVN") == today_vn:
        return
    if not is_7am_window:
        return
    data = {}
    for name, code in symbols.items():
        try:
            vals, _ = fetch_series(code, I_1D)
            data[name] = detect_trend(vals)
            time.sleep(max(0, 60.0 / RPM / 2))  # nương tay
        except Exception as e:
            log.warning("Daily fetch fail %s: %s", name, e)
    if data:
        st["daily"]["dateVN"] = today_vn
        st["daily"]["data"] = data

def get_cached(st: dict, name: str, interval: str) -> str:
    key = f"{name}|{interval}"
    item = st["cache"].get(key)
    if item and item.get("trend"):
        return item["trend"]
    return "N/A"

def get_1h_plan_line(st: dict, name: str) -> str:
    key = f"{name}|{I_1H}"
    item = st["cache"].get(key) or {}
    trend = item.get("trend")
    if trend in ("LONG", "SHORT") and all(k in item for k in ("entry","sl","tp")):
        return f"Entry {item['entry']} | SL {item['sl']} | TP {item['tp']}"
    return ""  # SIDEWAY hoặc chưa đủ dữ liệu -> không in

# ============ Telegram ============
def tg_send(text: str):
    if not (TG_TOKEN and TG_CHAT):
        log.info("Telegram skipped (no token/chat)")
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT, "text": text}
    try:
        r = requests.post(url, json=payload, timeout=TIMEOUT_S)
        if r.status_code != 200:
            log.warning("Telegram send failed: %s %s", r.status_code, r.text)
        else:
            log.info("Telegram: sent")
    except Exception as e:
        log.warning("Telegram error: %s", e)

# ============ Compose message ============
def combine_pair(a: str, b: str, label_a: str, label_b: str) -> str:
    if a == "N/A" and b == "N/A":
        return "N/A"
    if a == b:
        return a
    return f"Mixed ({label_a}:{a}, {label_b}:{b})"

def build_message(st: dict) -> str:
    now = _now_vn().strftime("%Y-%m-%d %H:%M:%S (VN)")
    lines = [ "💵 TRADE GOODS", f"🕰 {now}", "" ]
    something_actionable = False

    for name in symbols.keys():
        m30 = get_cached(st, name, I_30M)
        h1  = get_cached(st, name, I_1H)
        h2  = get_cached(st, name, I_2H)
        h4  = get_cached(st, name, I_4H)

        pair1 = combine_pair(m30, h1, "30min", "1h")
        pair2 = combine_pair(h2, h4, "2h", "4h")

        d1 = st["daily"]["data"].get(name, "N/A")

        # đánh dấu để quyết định có gửi hay không (ít nhất 1 LONG/SHORT)
        for v in (m30, h1, h2, h4, d1):
            if v in ("LONG", "SHORT"):
                something_actionable = True

        lines += [f"==={name}===",
                  f"30m–1H: {pair1}",
                  f"2H–4H: {pair2}",
                  f"1D: {d1}"]

        plan_line = get_1h_plan_line(st, name)
        if plan_line:
            lines.append(plan_line)

        lines.append("")

    msg = "\n".join(lines).rstrip()

    # chỉ gửi khi có ít nhất 1 LONG/SHORT (theo yêu cầu)
    if not something_actionable:
        log.info("Only SIDEWAY/N/A signals -> not sending.")
        return ""
    return msg

# ============ MAIN LOOP (1 lần chạy) ============
# ============ MAIN LOOP (1 lần chạy) ============
def main():
    if not API_KEY:
        log.error("Missing TWELVE_DATA_KEY")
    st = load_state()

    # 1D: chỉ fetch 07:00 VN
    update_daily_if_needed(st)

    # Round-robin: chọn tối đa RPM (<=7) cặp (symbol, interval) để fetch mỗi lần chạy
    batch = pick_batch(st, RPM)
    log.info("Batch: %s", batch)

    calls = 0
    start_min = int(time.time() // 60)
    for (name, interval) in batch:
        # tôn trọng giới hạn 8/min: nếu quá, dừng
        if calls >= RPM:
            break
        # fetch cập nhật cache
        update_interval_cache(st, name, interval)
        calls += 1
        # spacing nhẹ để an toàn
        time.sleep(max(0, 60.0 / RPM / 2))

    save_state(st)

    # Xây tin nhắn & gửi
    msg = build_message(st)
    if msg:
        tg_send(msg)

if __name__ == "__main__":
    main()
