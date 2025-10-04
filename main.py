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
    "USD/JPY": "USD/JPY",
}

interval_groups = {
    "15m-30m": ["15min", "30min"],
    "1H-2H": ["1h", "2h"],
    "4H": ["4h"]
}
# ====== STABILITY SETTINGS ======
CONFIRM_TF = ["1h", "2h", "4h"]   # TF làm gốc cho Direction/Entry
CONF_THRESHOLD = 60               # % tối thiểu để xuất Entry/SL/TP
HYSTERESIS_PCT = 10               # chênh lệch % tối thiểu mới cho phép đảo chiều
MIN_HOLD_MIN = 120                # phải giữ hướng tối thiểu 120 phút mới cho phép đảo
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
    """Regime + slope cơ bản để gán LONG/SHORT/SIDEWAY"""
    if df is None or len(df) < 60:
        return "N/A"
    e20 = ema(df['close'], 20)
    e50 = ema(df['close'], 50)
    last = df['close'].iloc[-1]
    adx_val = adx(df, 14)
    if len(e20) < 6 or np.isnan(adx_val):
        return "N/A"
    slope = (e20.iloc[-1] - e20.iloc[-6]) / max(1e-9, e20.iloc[-6]) * 100
    if (e20.iloc[-1] > e50.iloc[-1]) and (last > e20.iloc[-1]) and (adx_val >= 20) and (slope > 0.02):
        return "LONG"
    if (e20.iloc[-1] < e50.iloc[-1]) and (last < e20.iloc[-1]) and (adx_val >= 20) and (slope < -0.02):
        return "SHORT"
    return "SIDEWAY"

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

# ================ CORE ANALYZE ================
def analyze_symbol(name, symbol, daily_cache):
    results = {}
    has_data = False

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

    # Áp dụng hysteresis & memory
    state = load_state()
    final_dir, final_conf = decide_with_memory(symbol, raw_dir, raw_conf, state)
    save_state(state)

    # ===== Entry/SL/TP: chỉ khi đủ ngưỡng =====
    plan = "SIDEWAY"
    entry = sl = tp = atrval = None

    df1h = fetch_candles(symbol, "1h")
    if df1h is not None and len(df1h) > 40:
        atrval = atr(df1h, 14)
        entry  = df1h["close"].iloc[-1]
        swing_hi, swing_lo = swing_levels(df1h, 20)  # giữ hàm swing_levels bạn đang có

        # hệ số ATR (giữ quy ước cũ của bạn)
        is_fx = name in ("EUR/USD", "USD/JPY")
        base_mult = 2.5 if is_fx else 1.5

        if final_dir == "LONG" and final_conf >= CONF_THRESHOLD:
            plan = "LONG"
            sl_candidates = [
                entry - base_mult*atrval,
                swing_lo - 0.5*atrval if not np.isnan(swing_lo) else entry - base_mult*atrval
            ]
            sl = min(sl_candidates)
            R  = entry - sl
            # kẹp TP: nhỏ hơn giữa 1.2R và 1.5*ATR
            tp_dist = min(1.2*R, 1.5*atrval)
            tp = entry + tp_dist

        elif final_dir == "SHORT" and final_conf >= CONF_THRESHOLD:
            plan = "SHORT"
            sl_candidates = [
                entry + base_mult*atrval,
                swing_hi + 0.5*atrval if not np.isnan(swing_hi) else entry + base_mult*atrval
            ]
            sl = max(sl_candidates)
            R  = sl - entry
            tp_dist = min(1.2*R, 1.5*atrval)
            tp = entry - tp_dist

    # Trả thêm 'final_conf' để in ra Telegram (nếu bạn muốn)
    return results, plan, entry, sl, tp, atrval, True, final_dir, int(round(raw_conf)), int(round(final_conf))

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
        results, plan, entry, sl, tp, atrval, has_data, final_dir, raw_conf, final_conf = analyze_symbol(name, sym, daily_cache)
        if has_data:
            any_symbol_has_data = True

        lines.append(f"==={name}===")
        for group, trend in results.items():
            lines.append(f"{group}: {trend}")

        # thêm Confidence + Regime (không ảnh hưởng logic cũ)
        regime = "TREND" if results.get("4H") in ("LONG","SHORT") else "RANGE"
        lines.append(f"Confidence: {final_conf}% | Regime: {regime}")

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
    msg = "\n".join(lines)
    send_telegram(msg)

if __name__ == "__main__":
    main()
    
