import os
import time
import json
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# ================ LOGGING ================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)

# ================ CONFIG ================
API_KEY = os.getenv("TWELVE_DATA_KEY", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

BASE_URL = "https://api.twelvedata.com/time_series"
TIMEZ = os.getenv("TZ", "Asia/Ho_Chi_Minh")

# Request per minute throttle
RPM = int(os.getenv("RPM", 7))

# Cache 1D (chỉ fetch 1 lần/ngày lúc 00:05)
DAILY_CACHE_PATH = os.getenv("DAILY_CACHE_PATH", "/tmp/daily_cache.json")

symbols = {
    "Bitcoin": "BTC/USD",
    "Ethereum": "ETH/USD",
    "XAU/USD (Gold)": "XAU/USD",
    "WTI Oil": "CL",
    "EUR/USD": "EUR/USD",
}

interval_groups = {
    "15m-30m": ["15min", "30min"],
    "1H-2H": ["1h", "2h"],
    "4H": ["4h"]
}
# ====== STABILITY SETTINGS ======
CONFIRM_TF = ["1h", "2h", "4h"]   # TF làm gốc cho Direction/Entry
CONF_THRESHOLD = 55               # % tối thiểu để xuất Entry/SL/TP
HYSTERESIS_PCT = 6               # chênh lệch % tối thiểu mới cho phép đảo chiều
MIN_HOLD_MIN = 90                # phải giữ hướng tối thiểu 120 phút mới cho phép đảo
COOLDOWN_MIN = 60                 # sau khi đảo, chờ 60 phút mới được đảo nữa
STATE_PATH = os.getenv("STATE_PATH", "/tmp/signal_state.json")
SMOOTH_ALPHA = 0.5                # làm mượt confidence giữa các lần chạy (0..1)
# ================ HELPERS ================
def fetch_candles(symbol, interval, retries=3):
    url = f"{BASE_URL}?symbol={symbol}&interval={interval}&apikey={API_KEY}&outputsize=200"
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 429:
                logging.warning(f"Rate limit hit for {symbol}-{interval}, sleeping 65s then retry...")
                time.sleep(65)
                continue
            data = r.json()
            if "values" not in data:
                logging.warning(f"No data for {symbol}-{interval}: {data}")
                return None
            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").reset_index(drop=True)
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col])
            return df
        except Exception as e:
            logging.error(f"Fetch error {symbol}-{interval}: {e}")
            time.sleep(3)
    return None

def atr(df, period=14):
    high = df["high"].values
    low = df["low"].values
    close_prev = df["close"].shift(1).fillna(df["close"])
    tr = np.maximum(high - low, np.maximum(abs(high - close_prev), abs(low - close_prev)))
    atr_vals = pd.Series(tr).rolling(period).mean()
    return float(atr_vals.iloc[-1])

def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def adx(df, n=14):
    if df is None or len(df) < n + 20:
        return np.nan
    up = df['high'].diff()
    dn = -df['low'].diff()
    plus  = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift()).abs()
    tr3 = (df['low']  - df['close'].shift()).abs()
    tr  = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_ = pd.Series(tr).rolling(n).mean()
    plus_di  = 100 * pd.Series(plus).rolling(n).mean()  / atr_
    minus_di = 100 * pd.Series(minus).rolling(n).mean() / atr_
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return float(dx.rolling(n).mean().iloc[-1])

def rsi(series, n=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    dn = -delta.clip(upper=0).rolling(n).mean()
    rs = up / (dn.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def macd_hist(series, fast=12, slow=26, sig=9):
    ema_f = series.ewm(span=fast, adjust=False).mean()
    ema_s = series.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    signal = macd.ewm(span=sig, adjust=False).mean()
    return float((macd - signal).iloc[-1])

def donchian_mid(df, n=20):
    hi = df['high'].rolling(n).max()
    lo = df['low' ].rolling(n).min()
    mid = (hi + lo) / 2
    return float(hi.iloc[-1]), float(lo.iloc[-1]), float(mid.iloc[-1])

def keltner_mid(df, n=20, atr_mult=1.0):
    mid_val = float(ema(df['close'], n).iloc[-1])
    a = atr(df, 14)
    return mid_val, mid_val + atr_mult*a, mid_val - atr_mult*a

def strong_trend(df):
    """Trả LONG/SHORT/SIDEWAY cho 1 khung thời gian (dùng nến đã ĐÓNG)."""
    if df is None or len(df) < 65:
        return "N/A"

    e20 = ema(df["close"], 20)
    e50 = ema(df["close"], 50)
    if len(e20) < 60 or np.isnan(e20.iloc[-2]) or np.isnan(e50.iloc[-2]):
        return "N/A"

    last = float(df["close"].iloc[-2])
    # dốc EMA20 ~5 nến (đều là closed bar)
    slope = (e20.iloc[-2] - e20.iloc[-7]) / max(1e-9, e20.iloc[-7]) * 100.0

    # ADX (không có thì coi như pass)
    try:
        adx_val = adx(df, 14)
        adx_ok = (not np.isnan(adx_val)) and adx_val >= 18
    except Exception:
        adx_ok = True

    SLOPE_UP = 0.15
    SLOPE_DN = -0.15

    long_cond  = (last > e20.iloc[-2] > e50.iloc[-2]) and (slope > SLOPE_UP) and adx_ok
    short_cond = (last < e20.iloc[-2] < e50.iloc[-2]) and (slope < SLOPE_DN) and adx_ok

    if long_cond:  return "LONG"
    if short_cond: return "SHORT"
    return "SIDEWAY"
def strong_trend(df):
    """Trả LONG/SHORT/SIDEWAY cho 1 khung thời gian (dùng nến đã ĐÓNG)."""
    if df is None or len(df) < 65:
        return "N/A"

    e20 = ema(df["close"], 20)
    e50 = ema(df["close"], 50)
    if len(e20) < 60 or np.isnan(e20.iloc[-2]) or np.isnan(e50.iloc[-2]):
        return "N/A"

    last = float(df["close"].iloc[-2])
    # dốc EMA20 ~5 nến (đều là closed bar)
    slope = (e20.iloc[-2] - e20.iloc[-7]) / max(1e-9, e20.iloc[-7]) * 100.0

    # ADX (không có thì coi như pass)
    try:
        adx_val = adx(df, 14)
        adx_ok = (not np.isnan(adx_val)) and adx_val >= 18
    except Exception:
        adx_ok = True

    SLOPE_UP = 0.15
    SLOPE_DN = -0.15

    long_cond  = (last > e20.iloc[-2] > e50.iloc[-2]) and (slope > SLOPE_UP) and adx_ok
    short_cond = (last < e20.iloc[-2] < e50.iloc[-2]) and (slope < SLOPE_DN) and adx_ok

    if long_cond:  return "LONG"
    if short_cond: return "SHORT"
    return "SIDEWAY"
    
def detect_fast_flip_2h(symbol):
    df2h = fetch_candles(symbol, "2h")
    if df2h is None or len(df2h) < 60:
        return False
    e20 = df2h['close'].ewm(span=20, adjust=False).mean()
    two_red  = (df2h['close'].iloc[-1] < df2h['open'].iloc[-1]) and (df2h['close'].iloc[-2] < df2h['open'].iloc[-2])
    below_e20 = df2h['close'].iloc[-1] < e20.iloc[-1]
    slope_neg = (e20.iloc[-1] - e20.iloc[-6]) < 0
    return bool(two_red and below_e20 and slope_neg)
    
def swing_levels(df, lookback=20):
    if df is None or len(df) < lookback + 2:
        return (np.nan, np.nan)
    swing_hi = df['high'].rolling(lookback).max().iloc[-2]
    swing_lo = df['low' ].rolling(lookback).min().iloc[-2]
    return float(swing_hi), float(swing_lo)

def market_regime(df):
    """TRND / RANGE dùng cho hiển thị và lọc."""
    if df is None or len(df) < 60:
        return "UNKNOWN"
    e20 = ema(df['close'], 20)
    e50 = ema(df['close'], 50)
    a = adx(df, 14)
    if np.isnan(a): 
        return "UNKNOWN"
    if a >= 20 and abs((e20.iloc[-1] - e50.iloc[-1]) / e50.iloc[-1]) > 0.001:
        return "TREND"
    return "RANGE"

def score_frame(df, bias):
    """Chấm điểm 0..1 cho một khung thời gian dựa trên confluence."""
    if df is None or len(df) < 60 or bias == "N/A":
        return 0.0
    e20 = ema(df['close'], 20)
    e50 = ema(df['close'], 50)
    adx_v = adx(df, 14)
    mh   = macd_hist(df['close'])
    r    = rsi(df['close'], 14).iloc[-1]
    _, _, dmid = donchian_mid(df, 20)

    score = 0.0
    # regime + EMA alignment
    if adx_v >= 20:
        if bias == "LONG"  and e20.iloc[-1] > e50.iloc[-1]: score += 0.35
        if bias == "SHORT" and e20.iloc[-1] < e50.iloc[-1]: score += 0.35
    # MACD hist ủng hộ
    if (bias == "LONG" and mh > 0) or (bias == "SHORT" and mh < 0):
        score += 0.25
    # RSI vùng khỏe
    if (bias == "LONG" and 50 <= r <= 65) or (bias == "SHORT" and 35 <= r <= 50):
        score += 0.2
    # Vị trí so với Donchian mid
    last = df['close'].iloc[-1]
    if (bias == "LONG" and last >= dmid) or (bias == "SHORT" and last <= dmid):
        score += 0.2

    return float(min(score, 1.0))

def confluence_score(results_dict):
    """điểm đồng thuận 0–3: giữa 15–30, 1H–2H, 4H (để hiển thị cũ giữ nguyên)."""
    g15_30 = results_dict.get("15m-30m", "N/A")
    g1_2   = results_dict.get("1H-2H",   "N/A")
    g4     = results_dict.get("4H",      "N/A")

    def norm(x):
        if isinstance(x, str) and x.startswith("Mixed"):
            return "MIX"
        return x

    a, b, c = norm(g15_30), norm(g1_2), norm(g4)
    score = 0
    if a in ("LONG","SHORT") and b == a: score += 1
    if b in ("LONG","SHORT") and c == b: score += 1
    if a in ("LONG","SHORT") and c == a: score += 1
    return score

def format_price(sym, val):
    if val is None or np.isnan(val): return "N/A"
    # FX thì 2 chữ số thập phân với JPY, 5 với EURUSD; hàng hóa/crypto để 2
    if "JPY" in sym:
        return f"{val:.2f}"
    if sym in ("EUR/USD",):
        return f"{val:.5f}"
    return f"{val:.2f}"

# ============ Daily cache (1D) ============
def load_daily_cache():
    try:
        with open(DAILY_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"date": None, "data": {}}

def save_daily_cache(cache):
    try:
        with open(DAILY_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Save daily cache failed: {e}")

def maybe_refresh_daily_cache():
    """
    Gọi lúc mỗi lần chạy.
    Chỉ fetch 1D khi: hôm nay khác cache['date'] và thời điểm >= 00:05 (theo TIMEZ).
    """
    cache = load_daily_cache()
    now_local = datetime.now(timezone.utc).astimezone()
    today_str = now_local.strftime("%Y-%m-%d")

    # đổi múi giờ hiển thị thôi; 00:05 theo server local (đã astimezone())
    if cache.get("date") == today_str:
        return cache  # đã có hôm nay

    # chỉ làm sau 00:05
    if now_local.hour == 0 and now_local.minute < 5:
        logging.info("Before 00:05 — skip daily 1D refresh this run.")
        return cache

    logging.info("Refreshing daily 1D cache ...")
    new_data = {}
    for name, sym in symbols.items():
        df1d = fetch_candles(sym, "1day")
        if df1d is None or len(df1d) < 60:
            trend_1d = "N/A"
        else:
            trend_1d = strong_trend(df1d)
        new_data[sym] = {"trend": trend_1d}
        time.sleep(60.0 / RPM)

    cache = {"date": today_str, "data": new_data}
    save_daily_cache(cache)
    return cache

import json
from datetime import datetime, timezone

def load_state():
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(state):
    try:
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
    except Exception as e:
        logging.warning(f"save_state failed: {e}")

def smooth_conf(prev_conf, curr_conf):
    if prev_conf is None or np.isnan(prev_conf):
        return curr_conf
    return SMOOTH_ALPHA*curr_conf + (1.0-SMOOTH_ALPHA)*prev_conf

def decide_with_memory(sym, raw_dir, raw_conf, state):
    """Áp dụng hysteresis/min-hold/cooldown + làm mượt confidence."""
    now = datetime.now(timezone.utc)
    s = state.get(sym, {})
    prev_dir  = s.get("dir")
    prev_conf = s.get("conf")
    prev_ts_s = s.get("ts")
    prev_ts   = datetime.fromisoformat(prev_ts_s) if prev_ts_s else None

    # mượt hoá confidence
    smoothed_conf = smooth_conf(prev_conf, raw_conf if raw_conf is not None else 0)

    # lần đầu chưa có state
    if prev_dir is None or prev_ts is None:
        final_dir  = raw_dir
        final_conf = smoothed_conf
        state[sym] = {"dir": final_dir, "conf": final_conf, "ts": now.isoformat()}
        return final_dir, final_conf

    held_min = (now - prev_ts).total_seconds() / 60.0
    flip = (raw_dir in ("LONG","SHORT")) and (prev_dir in ("LONG","SHORT")) and (raw_dir != prev_dir)
    big_enough_move = abs((smoothed_conf or 0) - (prev_conf or 0)) >= HYSTERESIS_PCT
    can_flip_time   = held_min >= MIN_HOLD_MIN and held_min >= COOLDOWN_MIN

    if flip and big_enough_move and can_flip_time:
        # chấp nhận đảo chiều
        final_dir  = raw_dir
        final_conf = smoothed_conf
        state[sym] = {"dir": final_dir, "conf": final_conf, "ts": now.isoformat()}
        return final_dir, final_conf

    # giữ nguyên hướng trước
    final_dir  = prev_dir
    final_conf = smoothed_conf
    state[sym]["conf"] = final_conf
    return final_dir, final_conf
import re

def _norm_dir(x: str) -> str:
    """Chuẩn hoá text trend về LONG/SHORT/SIDEWAY/N/A."""
    if not isinstance(x, str): 
        return "N/A"
    x = x.upper()
    if x.startswith("MIXED"): 
        return "MIXED"
    for t in ("LONG","SHORT","SIDEWAY"):
        if t in x: 
            return t
    return "N/A"

def _extract_subdir(mixed_text: str, key: str) -> str:
    """
    Lấy hướng của 1 khung trong chuỗi Mixed, ví dụ 'Mixed (1h:LONG, 2h:SHORT)'
    key = '1h' hoặc '2h'
    """
    if not isinstance(mixed_text, str): 
        return "N/A"
    m = re.search(fr"{key}\s*:\s*(LONG|SHORT|SIDEWAY)", mixed_text, re.IGNORECASE)
    return m.group(1).upper() if m else "N/A"
    
def compact_label(group: str, trend: str) -> str:
    """Rút gọn Mixed(...) thành 'A-B' theo cặp khung; còn lại giữ nguyên."""
    if not isinstance(trend, str):
        return "N/A"
    if not trend.upper().startswith("MIXED"):
        return trend

    up = trend.upper()
    if group == "15m-30m":
        a = _extract_subdir(up, "15MIN")
        b = _extract_subdir(up, "30MIN")
        return f"{a}-{b}" if a != "N/A" and b != "N/A" else "MIXED"
    if group == "1H-2H":
        a = _extract_subdir(up, "1H")
        b = _extract_subdir(up, "2H")
        return f"{a}-{b}" if a != "N/A" and b != "N/A" else "MIXED"
    return "MIXED"
    
def detect_pullback(results: dict) -> str:
    """
    Trả về '', hoặc 'UP', 'DOWN'
    - Pullback DOWN: 4H==LONG & 1H==SHORT
    - Pullback UP  : 4H==SHORT & 1H==LONG
    Ưu tiên 1D nếu có (1D trùng 4H thì cảnh báo mạnh hơn – mình chỉ trả hướng để bạn in).
    """
    g12 = results.get("1H-2H", "N/A")
    d4  = _norm_dir(results.get("4H", "N/A"))
    d1  = _norm_dir(results.get("1D", "N/A"))

    # Lấy hướng 1H trong group 1H-2H
    if _norm_dir(g12) == "MIXED":
        d1h = _extract_subdir(g12, "1h")
    else:
        d1h = _norm_dir(g12)

    if d4 == "LONG" and d1h == "SHORT":
        return "DOWN"   # pullback giảm trong xu hướng tăng
    if d4 == "SHORT" and d1h == "LONG":
        return "UP"     # pullback tăng trong xu hướng giảm
    return ""
CONFIRM_STRONG = 70   # >=70%: mạnh
CONFIRM_OK     = 55   # 55–69%: trung bình

def smart_sl_tp(entry, atr, swing_hi, swing_lo, kup, kdn, side, is_fx):
    """
    Tính SL/TP:
      - SL: ATR-based + dựa swing gần nhất (lấy xa hơn để an toàn).
      - TP: ưu tiên 1.2R, nhưng KHÔNG vượt quá Keltner band và swing đối diện.
    """
    base_mult = 2.5 if is_fx else 1.5
    buf = 0.5 * atr

    if side == "LONG":
        sl_candidates = [
            entry - base_mult * atr,
            (swing_lo - buf) if not np.isnan(swing_lo) else entry - base_mult * atr,
        ]
        sl = min(sl_candidates)
        R = entry - sl

        # trần TP bởi band/swing (nếu có)
        caps = [1.2 * R, 1.5 * atr]
        if not np.isnan(kup):        # khoảng tới Keltner trên
            caps.append(max(0.0, kup - entry))
        if not np.isnan(swing_hi):   # khoảng tới swing đỉnh
            caps.append(max(0.0, swing_hi - entry - buf))

        tp = entry + max(0.0, min(caps))

    else:  # SHORT
        sl_candidates = [
            entry + base_mult * atr,
            (swing_hi + buf) if not np.isnan(swing_hi) else entry + base_mult * atr,
        ]
        sl = max(sl_candidates)
        R = sl - entry

        caps = [1.2 * R, 1.5 * atr]
        if not np.isnan(kdn):
            caps.append(max(0.0, entry - kdn))
        if not np.isnan(swing_lo):
            caps.append(max(0.0, entry - swing_lo - buf))

        tp = entry - max(0.0, min(caps))

    return sl, tp


def decide_signal_color(results: dict, final_dir: str, final_conf: int):
    """
    Trả về (emoji, label_size)
    - 🟢 'FULL'    : final_conf>=70 và 4H trùng final_dir và (1D trùng hoặc N/A)
    - 🟡 'HALF'    : final_conf 55–69, hoặc 1H ngược 4H nhưng 4H==final_dir
    - 🔴 'SKIP'    : còn lại
    """
    d4  = _norm_dir(results.get("4H", "N/A"))
    d1  = _norm_dir(results.get("1D", "N/A"))
    g12 = results.get("1H-2H", "N/A")
    d1h = _extract_subdir(g12, "1h") if _norm_dir(g12)=="MIXED" else _norm_dir(g12)

    # GREEN – mạnh
    if final_dir in ("LONG","SHORT") and final_conf >= CONFIRM_STRONG \
       and d4 == final_dir and d1 in (final_dir, "N/A"):
        return "🟢", "FULL"

    # YELLOW – trung bình
    if (CONFIRM_OK <= final_conf < CONFIRM_STRONG) or (d4 == final_dir and d1h not in ("N/A","SIDEWAY") and d1h != d4):
        return "🟡", "HALF"

    # RED – yếu/không rõ
    return "🔴", "SKIP"
    
# ================ CORE ANALYZE ================
def analyze_symbol(name, symbol, daily_cache):
    results = {}
    has_data = False
    fast_bear = False

    # 1) Trend text theo nhóm khung như cũ (dùng strong_trend)
    for group, intervals in interval_groups.items():
        trends = []
        for iv in intervals:
            df = fetch_candles(symbol, iv)
            trend = strong_trend(df)
            trends.append(f"{iv}:{trend}")
            time.sleep(60.0 / RPM)
        if len(intervals) == 1:
            res = trends[0].split(":")[1]
        else:
            uniq = set([t.split(":")[1] for t in trends])
            if len(uniq) == 1:
                res = uniq.pop()
            else:
                res = "Mixed (" + ", ".join(trends) + ")"
        results[group] = res
        if res != "N/A":
            has_data = True

    # 1D từ cache (không tốn call)
    daily_trend = daily_cache.get("data", {}).get(symbol, {}).get("trend", "N/A")
    results["1D"] = daily_trend

    # ===== Sau khi đã có 'results' cho các khung =====
    # Bỏ phiếu từ 1h/2h/4h (bỏ qua Mixed/N/A)
    votes = []
    for g in ["1H-2H", "4H"]:
        v = results.get(g, "N/A")
        if isinstance(v, str) and v.startswith("Mixed"):
            continue
        if v in ("LONG", "SHORT"):
            votes.append(v)

    if votes.count("LONG") > votes.count("SHORT"):
        raw_dir = "LONG"
    elif votes.count("SHORT") > votes.count("LONG"):
        raw_dir = "SHORT"
    else:
        raw_dir = "SIDEWAY"

    # Tính confidence đơn giản (mỗi phiếu = 33%), bonus nếu 1D trùng
    raw_conf = 0
    raw_conf += 33 * votes.count(raw_dir)
    d1 = results.get("1D", "N/A")
    if d1 == raw_dir:
        raw_conf += 20
    raw_conf = max(0, min(100, raw_conf))
    # Nếu có fast_bear mà memory vẫn giữ conf cao, ép kẹp xuống mức vừa tính
    fast_bear = detect_fast_flip_2h(symbol)
    if fast_bear:
        raw_conf = max(0, raw_conf - 25)
        hi_bias = results.get("4H", "N/A")
        d1_bias = results.get("1D", "N/A")
        if raw_dir == "LONG" and ("LONG" in (hi_bias, d1_bias)):
            raw_dir  = "SIDEWAY"
            raw_conf = min(raw_conf, 50)
        else:
            raw_dir  = "SHORT"
            raw_conf = min(raw_conf, 35)
    # ===== 2H override để phản ứng nhanh khi 2H đang giảm mạnh =====
    df2h = fetch_candles(symbol, "2h")
    if df2h is not None and len(df2h) >= 60:
        e20_2h = df2h['close'].ewm(span=20, adjust=False).mean()
        # 2 nến 2H liền kề là nến giảm
        two_red = (df2h['close'].iloc[-1] < df2h['open'].iloc[-1]) and \
                  (df2h['close'].iloc[-2] < df2h['open'].iloc[-2])
        # giá dưới EMA20 và slope EMA20 đang âm
        below_e20 = df2h['close'].iloc[-1] < e20_2h.iloc[-1]
        slope_neg = (e20_2h.iloc[-1] - e20_2h.iloc[-6]) < 0
    
        if two_red and below_e20 and slope_neg:
            raw_conf = max(0, raw_conf - 25)
            print(f"⚠️ 2H đảo chiều giảm mạnh – hạ confidence {symbol} xuống {raw_conf}")
            # --- 2H override để phản ứng nhanh khi 2H giảm mạnh ---

            df2h = fetch_candles(symbol, "2h")
            if df2h is not None and len(df2h) > 60:
                e20_2h = df2h['close'].ewm(span=20, adjust=False).mean()
                two_red = (df2h['close'].iloc[-1] < df2h['open'].iloc[-1]) and \
                          (df2h['close'].iloc[-2] < df2h['open'].iloc[-2])
                below_e20 = df2h['close'].iloc[-1] < e20_2h.iloc[-1]
                slope_neg = (e20_2h.iloc[-1] - e20_2h.iloc[-6]) < 0
            
                if two_red and below_e20 and slope_neg:
                    # nếu 4H/1D vẫn LONG thì hạ về SIDEWAY, còn không thì cho SHORT nhẹ
                    hi_bias  = results.get("4H", "N/A")
                    d1_bias  = results.get("1D", "N/A")
            
                    if raw_dir == "LONG" and ("LONG" in (hi_bias, d1_bias)):
                        raw_dir  = "SIDEWAY"
                        raw_conf = min(raw_conf, 50)
                    else:
                        raw_dir  = "SHORT"
                        raw_conf = min(raw_conf, 35)
                    fast_bear = True
                    print(f"⚠️ 2H đảo chiều mạnh -> raw_dir={raw_dir}, raw_conf={raw_conf}")
            
            # Nếu khung lớn vẫn LONG -> hạ xuống SIDEWAY & kẹp conf
            hi_bias = results.get("4H", "N/A")
            d1_bias = results.get("1D", "N/A")
            if raw_dir == "LONG" and ("LONG" in (hi_bias, d1_bias)):
                raw_dir  = "SIDEWAY"
                raw_conf = min(raw_conf, 50)
            # Nếu khung lớn không ủng hộ LONG -> cho phép lật SHORT nhạy hơn
            else:
                raw_dir  = "SHORT"
                raw_conf = max(raw_conf, 65)
    
    # Áp dụng hysteresis & memory
    state = load_state()
    final_dir, final_conf = decide_with_memory(symbol, raw_dir, raw_conf, state)
    
    # Nếu có fast_bear mà memory vẫn giữ conf cao, thì ép kẹp xuống mức vừa tính
    if fast_bear and final_conf > raw_conf:
        final_conf = raw_conf
        # cập nhật luôn vào state để lần sau không bật lại 86%
        state.setdefault(symbol, {})
        state[symbol]["dir"]  = final_dir
        state[symbol]["conf"] = final_conf
    save_state(state)

    # ===== Entry/SL/TP từ khung CHÍNH (mặc định 2H, có thể đổi qua biến môi trường MAIN_TF) =====
    plan = "SIDEWAY"
    entry = sl = tp = atrval = None
    
    MAIN_TF = os.getenv("MAIN_TF", "2h")  # "2h" / "4h" / "1h" tùy bạn
    df_main = fetch_candles(symbol, MAIN_TF)
    
    if df_main is not None and len(df_main) > 60:
        entry = float(df_main["close"].iloc[-2])
        atrval = atr(df_main, 14)
        swing_hi, swing_lo = swing_levels(df_main, 20)  # đỉnh/đáy gần
    
        # hệ số ATR theo loại sản phẩm (giữ quy ước cũ)
        is_fx = name in ("EUR/USD")
        base_mult = 2.5 if is_fx else 1.5
    
        # ===== LONG =====
        if final_dir == "LONG" and final_conf >= CONF_THRESHOLD:
            plan = "LONG"
            # SL: lấy xa hơn giữa ATR*mult và đáy gần - buffer
            sl_candidates = [
                entry - base_mult * atrval,
                (swing_lo - 0.3 * atrval) if not np.isnan(swing_lo) else entry - base_mult * atrval
            ]
            sl = min(sl_candidates)
            R = entry - sl
    
            # TP “thông minh”:
            #   - giữ RR hợp lý (1.2–1.8R)
            #   - không vượt 3×ATR
            #   - không chọc quá đỉnh gần + buffer
            rr_tp = min(max(1.2 * R, 1.5 * atrval), 3.0 * atrval)
    
            cap = None
            if not np.isnan(swing_hi):
                cap = max(0.8 * atrval, (swing_hi + 0.4 * atrval) - entry)
            tp_dist = min(rr_tp, cap) if (cap is not None and cap > 0) else rr_tp
            tp = entry + tp_dist
    
        # ===== SHORT =====
        elif final_dir == "SHORT" and final_conf >= CONF_THRESHOLD:
            plan = "SHORT"
            # SL: lấy xa hơn giữa ATR*mult và đỉnh gần + buffer
            sl_candidates = [
                entry + base_mult * atrval,
                (swing_hi + 0.3 * atrval) if not np.isnan(swing_hi) else entry + base_mult * atrval
            ]
            sl = max(sl_candidates)
            R = sl - entry
    
            rr_tp = min(max(1.2 * R, 1.5 * atrval), 3.0 * atrval)
    
            cap = None
            if not np.isnan(swing_lo):
                cap = max(0.8 * atrval, entry - (swing_lo - 0.4 * atrval))
            tp_dist = min(rr_tp, cap) if (cap is not None and cap > 0) else rr_tp
            tp = entry - tp_dist
    # ===== Hết block SL/TP =====

    # Trả thêm 'final_conf' để in ra Telegram (nếu bạn muốn)
    return results, plan, entry, sl, tp, atrval, True, final_dir, int(round(final_conf))

def send_telegram(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram not configured")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
        if r.status_code != 200:
            logging.error(f"Telegram error: {r.text}")
    except Exception as e:
        logging.error(f"Telegram send failed: {e}")

# ================ MAIN =================
def main():
    # luôn kiểm tra/làm mới cache 1D (chỉ fetch khi tới giờ/đúng ngày)
    daily_cache = maybe_refresh_daily_cache()

    lines = []
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    lines.append("💵 TRADE GOODS")
    lines.append(f"⏱ {now}\n")

    any_symbol_has_data = False

    for name, sym in symbols.items():
        results, plan, entry, sl, tp, atrval, has_data, final_dir, final_conf = analyze_symbol(name, sym, daily_cache)
        # Cảnh báo fast-flip 2H nếu có
        df_2h_check = fetch_candles(sym, "2h")
        fast_flip = False
        if df_2h_check is not None and len(df_2h_check) > 60:
            e20_2h = df_2h_check["close"].ewm(span=20, adjust=False).mean()
            two_red = (df_2h_check["close"].iloc[-1] < df_2h_check["open"].iloc[-1]) and (df_2h_check["close"].iloc[-2] < df_2h_check["open"].iloc[-2])
            below_e20 = df_2h_check["close"].iloc[-1] < e20_2h.iloc[-1]
            slope_neg = (e20_2h.iloc[-1] - e20_2h.iloc[-6]) < 0
            if two_red and below_e20 and slope_neg:
                fast_flip = True
        if fast_flip:
            lines.append("⚡ Fast-flip 2H active — chờ nến đóng xác nhận")
        if has_data:
            any_symbol_has_data = True

        lines.append(f"==={name}===")
        for group, trend in results.items():
            lines.append(f"{group}: {compact_label(group, trend)}")

        # —— Pullback & Color
        pb = detect_pullback(results)
        emoji, size_label = decide_signal_color(results, final_dir, int(round(final_conf)))
        
        regime = "TREND" if results.get("4H") in ("LONG","SHORT") else "RANGE"
        if pb == "DOWN":
            lines.append("⚠️ Pullback: 1H ngược 4H/1D (DOWN) – cân nhắc chờ xác nhận")
        elif pb == "UP":
            lines.append("⚠️ Pullback: 1H ngược 4H/1D (UP) – cân nhắc chờ xác nhận")
        
        # dòng Confidence có màu & size gợi ý
        lines.append(f"{emoji} Confidence: {int(round(final_conf))}% | Regime: {regime}")
        #| Size: {size_label}")
        
        # thêm Confidence + Regime (không ảnh hưởng logic cũ)
        #regime = "TREND" if results.get("4H") in ("LONG","SHORT") else "RANGE"
        #lines.append(f"Confidence: {final_conf}% | Regime: {regime}")

        if entry is not None and sl is not None and tp is not None:
            lines.append(
                f"Entry {format_price(name if name in ('EUR/USD','USD/JPY') else sym, entry)} | "
                f"SL {format_price(name if name in ('EUR/USD','USD/JPY') else sym, sl)} | "
                f"TP {format_price(name if name in ('EUR/USD','USD/JPY') else sym, tp)}"
            )
        lines.append("")

        # dàn request để không vượt quota
        time.sleep(10)

    # Nếu tất cả đều N/A/SIDEWAY & không có Entry -> vẫn gửi để biết trạng thái; nếu muốn có thể chặn tại đây
    #msg = "\n".join(lines)
    #send_telegram(msg)
    # Chỉ gửi nếu có ít nhất 1 symbol có dữ liệu thật sự
    # Chỉ gửi nếu có ít nhất 1 symbol KHÔNG N/A
    # Chỉ gửi nếu có ít nhất 1 symbol có Entry thật (không phải N/A)
    valid_msg = any(
    ("Entry" in l and not any(x in l for x in ["N/A", "None", "NaN"]))
    for l in lines
)
    if valid_msg:
        msg = "\n".join(lines)
        send_telegram(msg)
    else:
        print("🚫 Tất cả đều N/A, không gửi Telegram")

if __name__ == "__main__":
    main()
    
