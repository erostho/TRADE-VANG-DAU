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
from collections import deque

# cache cho 1 lần chạy
RUN_CACHE = {}

# token bucket đơn giản cho quota theo phút
_last_min_calls = deque()   # lưu timestamps các call trong 60s gần nhất
def _throttle():
    # số call/phút cho phép
    limit = max(1, int(os.getenv("RPM", 7)))
    now = time.monotonic()
    # bỏ timestamps cũ hơn 60s
    while _last_min_calls and now - _last_min_calls[0] > 60:
        _last_min_calls.popleft()
    if len(_last_min_calls) >= limit:
        # đợi tới khi đủ chỗ
        sleep_for = 60 - (now - _last_min_calls[0]) + 0.01
        time.sleep(max(0.0, sleep_for))
    _last_min_calls.append(time.monotonic())
    
symbols = {
    "Bitcoin": "BTC/USD",
    "Ethereum": "ETH/USD",
    "XAU/USD (Gold)": "XAU/USD",
    "WTI Oil": "CL",
    "USD/JPY": "USD/JPY",
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

# --- Oil calibration (CL futures -> Exness XTIUSD) ---
EXNESS_OIL_TICKER   = os.getenv("EXNESS_OIL_TICKER", "")  # vd: "XTIUSD" nếu provider của bạn có
OIL_CALIB_CACHE     = os.getenv("OIL_CALIB_CACHE", "/tmp/oil_calib.json")
OIL_CALIB_TTL_MIN   = int(os.getenv("OIL_CALIB_TTL_MIN", "60"))  # hiệu chuẩn lại mỗi 60 phút

# Fallback thủ công nếu không auto được
OIL_PRICE_SCALE_ENV  = float(os.getenv("OIL_PRICE_SCALE", "1.0"))   # a
OIL_PRICE_OFFSET_ENV = float(os.getenv("OIL_PRICE_OFFSET", "-16.0"))# b

# ====== PROP DESK SETTINGS ======
# Confidence → risk map
CONF_RISK_TABLE = [
    (85, 0.012),  # ≥85%: 1.2% equity
    (70, 0.008),  # 70–84%: 0.8%
    (55, 0.005),  # 55–69%: 0.5%
    (0,  0.003),  # <55%: 0.3% (nhưng thường không trade)
]

# Daily risk cap & circuit breakers
DAILY_RISK_CAP_PCT   = float(os.getenv("DAILY_RISK_CAP_PCT", "0.04"))  # 4%/ngày
MAX_LOSING_STREAK    = int(os.getenv("MAX_LOSING_STREAK", "3"))        # thua 3 lệnh/ngày thì ngưng
CB_COOLDOWN_MIN      = int(os.getenv("CB_COOLDOWN_MIN", "120"))         # nghỉ 120'
STATS_PATH           = os.getenv("STATS_PATH", "/tmp/prop_stats.json")  # track risk, streak

# News filter (tuỳ chọn – có là dùng, không có thì bỏ qua)
NEWS_FILTER_ON       = os.getenv("NEWS_FILTER_ON", "1") == "1"            # 1 là mở
NEWS_LOOKAHEAD_MIN   = int(os.getenv("NEWS_LOOKAHEAD_MIN", "60"))       # 60' trước/sau tin
TRADING_ECON_API_KEY = os.getenv("TRADING_ECON_API_KEY", "")            # optional
NEWS_CACHE_PATH      = os.getenv("NEWS_CACHE_PATH", "/tmp/news_today.json")

# Signal log (để backtest/expectancy offline)
SIGNAL_CSV_PATH      = os.getenv("SIGNAL_CSV_PATH", "/tmp/signals.csv")
# ===== INTRABAR GUARDS (real-time adaptation) =====
INTRABAR_PRICE_DEVIATION_ATR = float(os.getenv("INTRABAR_DEV_ATR", "0.5"))  # lệch > 0.5*ATR -> bỏ tín hiệu
ENTRY_WINDOW_MIN             = 80
MICROTREND_TF                 = os.getenv("MICROTREND_TF", "15min")          # khung xác nhận micro
MICROTREND_ALLOW_SIDEWAY      = os.getenv("MICROTREND_ALLOW_SIDEWAY", "1") == "1"

# Volatility regime
VOL_BW_HIGH   = float(os.getenv("VOL_BW_HIGH", "0.025"))  # BBWidth 2h > 2.5% coi là biến động cao
VOL_ATR_MULT  = float(os.getenv("VOL_ATR_MULT", "1.25"))  # ATR hiện tại > 1.25×ATR_20 coi là cao

# Bias tracking
BIAS_INVALIDATE_MIN = int(os.getenv("BIAS_INVALIDATE_MIN", "60"))  # 60' sau tín hiệu nếu mất bias thì huỷ
# ================ HELPERS ================
def fetch_candles(symbol, interval, retries=3):
    key = (symbol, interval)
    if key in RUN_CACHE:
        return RUN_CACHE[key]

    url = f"{BASE_URL}?symbol={symbol}&interval={interval}&apikey={API_KEY}&outputsize=200"
    for attempt in range(retries):
        try:
            _throttle()  # GIỮA MỌI LẦN GỌI
            r = requests.get(url, timeout=10)
            if r.status_code == 429:
                logging.warning(f"429 {symbol}-{interval} -> sleep 65s & retry...")
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
            RUN_CACHE[key] = df  # lưu cache cho cùng lần chạy
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
    if sym in ("EUR/USD","USD/JPY"):
        return f"{val:.5f}"
    return f"{val:.2f}"

# ================= EXTRA HELPERS (ADD) =================
def bb_width(df, n=20):
    """Bollinger Band Width = (upper - lower) / middle (tỷ lệ, vd 0.02 = 2%)"""
    if df is None or len(df) < n + 2:
        return np.nan
    m = df['close'].rolling(n).mean()
    s = df['close'].rolling(n).std()
    upper = m + 2*s
    lower = m - 2*s
    mid = m
    bw = (upper.iloc[-2] - lower.iloc[-2]) / max(1e-9, mid.iloc[-2])
    return float(bw)

def is_fx(sym_name_or_symbol: str) -> bool:
    up = sym_name_or_symbol.upper()
    return ("EUR/USD" in up) or ("USD/JPY" in up) or ("/" in up and "XAU" not in up and "BTC" not in up and "ETH" not in up)

def is_crypto(sym_name_or_symbol: str) -> bool:
    up = sym_name_or_symbol.upper()
    return ("BTC" in up) or ("ETH" in up)

def is_commodity(sym_name_or_symbol: str) -> bool:
    up = sym_name_or_symbol.upper()
    return ("XAU" in up) or ("GOLD" in up) or ("CL" in up) or ("OIL" in up)

# ——— Chuẩn hoá TF score theo confluence chỉ báo (0..1)
def calc_tf_score(df, direction: str) -> float:
    if df is None or len(df) < 60 or direction not in ("LONG","SHORT"):
        return 0.0
    try:
        e20 = ema(df["close"], 20)
        e50 = ema(df["close"], 50)
        a14 = adx(df, 14)
        mh  = macd_hist(df["close"])
        r14 = rsi(df["close"], 14).iloc[-2]
        last = df["close"].iloc[-2]
    except Exception:
        return 0.0

    score = 0.0
    # ADX: trend strength
    if not np.isnan(a14) and a14 >= 20: 
        score += 0.25
    # EMA alignment
    if direction == "LONG"  and e20.iloc[-2] > e50.iloc[-2]: score += 0.25
    if direction == "SHORT" and e20.iloc[-2] < e50.iloc[-2]: score += 0.25
    # MACD hist
    if (direction == "LONG" and mh > 0) or (direction == "SHORT" and mh < 0):
        score += 0.25
    # RSI vùng thuận
    if (direction == "LONG" and r14 >= 50) or (direction == "SHORT" and r14 <= 50):
        score += 0.25

    return float(min(score, 1.0))

def weighted_confidence(symbol, raw_dir: str) -> int:
    """Chuẩn hoá confidence theo trọng số TF: 4H=0.45, 2H=0.35, 1H=0.20 → 0..100"""
    if raw_dir not in ("LONG","SHORT"):
        return 0
    weights = [("1h", 0.20), ("2h", 0.35), ("4h", 0.45)]
    total = 0.0
    for tf, w in weights:
        df_tf = fetch_candles(symbol, tf)
        sc = calc_tf_score(df_tf, raw_dir)
        total += sc * w
    return int(round(min(100, max(0, total * 100))))

# ——— Position sizing (lot) theo % rủi ro & khoảng SL
# ENV: ACCOUNT_EQUITY (mặc định 1000), RISK_PER_TRADE (mặc định 0.02 = 2%)
ACCOUNT_EQUITY = float(os.getenv("ACCOUNT_EQUITY", "1000"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.02"))
# Contract size mặc định (có thể override bằng JSON ở env CONTRACT_SIZES)
_DEFAULT_CONTRACT_SIZES = {
    "BTC/USD": 1,       # crypto CFD thường contract 1
    "ETH/USD": 1,
    "XAU/USD": 100,     # Gold 100 oz/lot
    "XAU/USD (GOLD)": 100,
    "CL": 1000,         # WTI oil 1000 barrels/lot (CFD thường scale theo broker)
    "WTI OIL": 1000,
    "EUR/USD": 100000,  # FX 100k units/lot
    "USD/JPY": 100000,
}
def _load_contract_sizes():
    try:
        j = os.getenv("CONTRACT_SIZES", "")
        if j:
            user_map = json.loads(j)
            for k,v in user_map.items():
                _DEFAULT_CONTRACT_SIZES[str(k).upper()] = float(v)
    except Exception:
        pass
    return _DEFAULT_CONTRACT_SIZES
_CONTRACT_SIZES = _load_contract_sizes()

def _lookup_contract(symbol: str, name: str) -> float:
    # tra cứu theo symbol trước, rồi theo name
    key1 = symbol.upper()
    key2 = name.upper()
    return float(_CONTRACT_SIZES.get(key1, _CONTRACT_SIZES.get(key2, 100000.0)))
RISK_PCT = float(os.getenv("RISK_PCT", 0.02))     # 2%/lệnh
BALANCE_USD = float(os.getenv("BALANCE_USD", 120))
CONTRACT_SIZES={"XAU/USD":100,"XAG/USD":5000,"EUR/USD":100000,"USD/JPY":100000,"BTC/USD":1,"CL":1000}
def is_fx_name(n): return n in ("EUR/USD","USD/JPY")

def account_equity_usd():
    try:
        v=float(os.getenv("BALANCE_USD","0"))
        if v>0: return v
    except Exception: pass
    return 100.0

def compute_lot_size(entry, sl, symbol, name, risk_pct=0.005):
    if entry is None or sl is None: return 0.0
    dist=abs(entry-sl); 
    if dist<=0: return 0.0
    key=name if name in CONTRACT_SIZES else symbol
    contract=CONTRACT_SIZES.get(key,100000)
    if "JPY" in key: pipsize=0.01
    elif "XAU" in key: pipsize=0.1
    elif key=="CL": pipsize=0.01
    elif "/" in key: pipsize=0.0001
    else: pipsize=1.0
    value_per_point=contract*pipsize
    risk_money=account_equity_usd()*risk_pct
    lots=risk_money/max(1e-9,(dist*value_per_point))
    return round(max(lots,0.01),3)
    
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
    
import re

def compact_label(group: str, trend: str) -> str:
    """Rút gọn/ghi rõ Mixed của cặp khung; các trường hợp khác giữ nguyên."""
    if not isinstance(trend, str):
        return "N/A"

    up = trend.upper()
    if not up.startswith("MIXED"):
        return trend  # đã là LONG/SHORT/SIDEWAY thì trả nguyên

    # lấy hướng của 1 khung từ chuỗi Mixed (case-insensitive)
    def pick(key: str) -> str:
        m = re.search(rf"{key}\s*:\s*(LONG|SHORT|SIDEWAY)", up, re.IGNORECASE)
        return m.group(1).upper() if m else "N/A"

    if group == "1H-2H":
        d1h = pick("1H")
        d2h = pick("2H")
        if d1h != "N/A" and d2h != "N/A":
            return d1h if d1h == d2h else f"MIXED ({d1h} – {d2h})"
        return "MIXED"

    if group == "15m-30m":
        d15 = pick("15MIN")
        d30 = pick("30MIN")
        if d15 != "N/A" and d30 != "N/A":
            return d15 if d15 == d30 else f"MIXED ({d15} – {d30})"
        return "MIXED"

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
    # nhẹ tay hơn
    base_mult = 2.0 if is_fx else 1.2      # (trước: 2.5 / 1.5)
    buf = 0.3 * atr                         # (trước: 0.5 * ATR)

    if side == "LONG":
        # chọn SL gần hơn, nhưng không chặt hơn 0.8*ATR
        sl_candidates = [
            entry - base_mult * atr,
            (swing_lo - buf) if not np.isnan(swing_lo) else entry - base_mult * atr,
        ]
        sl = max(sl_candidates)             # GẦN HƠN cho LONG
        sl = min(sl, entry - 0.8 * atr)     # vẫn cách tối thiểu 0.8*ATR

        R = entry - sl

        # TP “thông minh” giữ nguyên
        caps = [1.2 * R, 1.5 * atr]
        if not np.isnan(kup):  caps.append(max(0.0, kup - entry))
        if not np.isnan(swing_hi): caps.append(max(0.0, swing_hi - entry - buf))
        tp = entry + max(0.0, min(caps))

    else:  # SHORT
        sl_candidates = [
            entry + base_mult * atr,
            (swing_hi + buf) if not np.isnan(swing_hi) else entry + base_mult * atr,
        ]
        sl = min(sl_candidates)             # GẦN HƠN cho SHORT
        sl = max(sl, entry + 0.8 * atr)     # vẫn cách tối thiểu 0.8*ATR

        R = sl - entry

        caps = [1.2 * R, 1.5 * atr]
        if not np.isnan(kdn):      caps.append(max(0.0, entry - kdn))
        if not np.isnan(swing_lo): caps.append(max(0.0, entry - swing_lo - buf))
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

def is_wti_name(name: str) -> bool:
    return name in ("WTI Oil", "USOIL", "XTIUSD")

def _load_oil_calib_cache():
    try:
        with open(OIL_CALIB_CACHE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_oil_calib_cache(d):
    try:
        with open(OIL_CALIB_CACHE, "w", encoding="utf-8") as f:
            json.dump(d, f)
    except Exception:
        pass

def _minutes_since(ts_iso: str) -> float:
    try:
        t = datetime.fromisoformat(ts_iso)
        return (datetime.now(timezone.utc) - t).total_seconds()/60.0
    except Exception:
        return 1e9

def compute_oil_calibration() -> tuple[float, float]:
    """
    Trả về (scale, offset) để chuyển giá CL -> Exness.
    Nếu có EXNESS_OIL_TICKER:
      - Lấy CL và EXNESS_OIL_TICKER (cùng interval 1h), căn chỉnh theo datetime
      - offset = median(exness - cl) trên 50 nến gần nhất
      - scale = 1.0 (đơn giản, đủ tốt vì khác biệt chủ yếu là mặt bằng)
    Nếu fail -> dùng ENV fallback.
    Có cache theo TTL để đỡ tốn API.
    """
    # cache
    cache = _load_oil_calib_cache()
    if cache.get("ts") and _minutes_since(cache["ts"]) < OIL_CALIB_TTL_MIN:
        return float(cache.get("scale", OIL_PRICE_SCALE_ENV)), float(cache.get("offset", OIL_PRICE_OFFSET_ENV))

    # nếu không chỉ định ticker exness -> fallback ENV
    if not EXNESS_OIL_TICKER:
        return (OIL_PRICE_SCALE_ENV, OIL_PRICE_OFFSET_ENV)

    df_cl  = fetch_candles("CL", "1h")
    df_ex  = fetch_candles(EXNESS_OIL_TICKER, "1h")
    if df_cl is None or df_ex is None or len(df_cl) < 10 or len(df_ex) < 10:
        return (OIL_PRICE_SCALE_ENV, OIL_PRICE_OFFSET_ENV)

    # join theo datetime
    x = df_cl[["datetime", "close"]].rename(columns={"close":"cl"}).copy()
    y = df_ex[["datetime", "close"]].rename(columns={"close":"ex"}).copy()
    z = pd.merge_asof(x.sort_values("datetime"), y.sort_values("datetime"),
                      on="datetime", direction="nearest", tolerance=pd.Timedelta("30min")).dropna()

    if len(z) < 10:
        return (OIL_PRICE_SCALE_ENV, OIL_PRICE_OFFSET_ENV)

    z = z.tail(50)  # 50 điểm gần nhất
    offset = float(np.median(z["ex"] - z["cl"]))
    scale  = 1.0

    _save_oil_calib_cache({"ts": datetime.now(timezone.utc).isoformat(),
                           "scale": scale, "offset": offset})
    return (scale, offset)

# sẽ được set khi chạy main()
_OIL_SCALE = OIL_PRICE_SCALE_ENV
_OIL_OFFSET = OIL_PRICE_OFFSET_ENV

def oil_adjust(p: float) -> float:
    if p is None or (isinstance(p, float) and np.isnan(p)): 
        return p
    return p * _OIL_SCALE + _OIL_OFFSET
# ---------- day stats / circuit breaker ----------
def load_stats():
    try:
        with open(STATS_PATH, "r", encoding="utf-8") as f: s = json.load(f)
    except Exception: s = {}
    today = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
    if s.get("date") != today:
        s = {"date": today, "risk_used": 0.0, "losing_streak": 0, "cb_until_ts": None}
    return s

def save_stats(s):
    try:
        with open(STATS_PATH, "w", encoding="utf-8") as f: json.dump(s, f, ensure_ascii=False)
    except Exception: pass

def is_circuit_breaker_on(stats):
    ts = stats.get("cb_until_ts")
    if not ts: return False
    try: until = datetime.fromisoformat(ts)
    except Exception: return False
    return datetime.now(timezone.utc) < until

def trigger_circuit_breaker(stats, minutes=CB_COOLDOWN_MIN):
    stats["cb_until_ts"] = (datetime.now(timezone.utc)+timedelta(minutes=minutes)).isoformat()

# ---------- confidence calibration ----------
def _tf_weighted_conf(results):
    def norm(x): return "MIX" if isinstance(x,str) and x.startswith("Mixed") else x
    g15, g12, g4, d1 = (norm(results.get(k,"N/A")) for k in ("15m-30m","1H-2H","4H","1D"))
    score=w=0.0
    for v,wt in [(g15,0.5),(g12,1.0),(g4,1.3),(d1,0.7)]:
        if v in ("LONG","SHORT"): score+=wt
        elif v=="MIX": score+=0.3*wt
        w+=wt
    return 100.0*(score/max(1e-6,w))

def calibrate_confidence(raw_conf, results, final_dir):
    tfc = _tf_weighted_conf(results)
    pen = 0
    if isinstance(results.get("1H-2H",""),str) and results["1H-2H"].startswith("Mixed"): pen+=8
    if results.get("4H")=="SIDEWAY": pen+=12
    conf = 0.6*raw_conf + 0.4*tfc - pen
    if final_dir=="SIDEWAY": conf=min(conf,50)
    return int(max(0,min(100,round(conf))))

def dynamic_risk_pct(conf_pct, regime):
    base=0.003
    for th,r in CONF_RISK_TABLE:
        if conf_pct>=th: base=r; break
    if regime=="RANGE": base*=0.6
    return base

# ---------- news filter (optional) ----------
def _relevant_currencies(symbol_name):
    s=symbol_name.upper()
    if "XAU" in s or "GOLD" in s: return ["USD"]
    if "XAG" in s or "SILVER" in s: return ["USD"]
    if "BTC" in s or "ETH" in s: return ["USD"]
    if "EUR" in s and "USD" in s: return ["EUR","USD"]
    if "USD" in s and "JPY" in s: return ["USD","JPY"]
    if "OIL" in s or s=="CL": return ["USD"]
    return ["USD"]

def _load_news_cache():
    try: 
        with open(NEWS_CACHE_PATH,"r",encoding="utf-8") as f: return json.load(f)
    except Exception: return {"date":None,"events":[]}

def _save_news_cache(c):
    try:
        with open(NEWS_CACHE_PATH,"w",encoding="utf-8") as f: json.dump(c,f,ensure_ascii=False)
    except Exception: pass

def refresh_news_today():
    cache=_load_news_cache()
    today=datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
    if cache.get("date")==today: return cache["events"]
    if not TRADING_ECON_API_KEY:
        c={"date":today,"events":[]}; _save_news_cache(c); return []
    try:
        url=f"https://api.tradingeconomics.com/calendar?d1={today}&d2={today}&c=all&format=json&client={TRADING_ECON_API_KEY}"
        r=requests.get(url,timeout=10); raw=r.json() if r.status_code==200 else []
    except Exception: raw=[]
    ev=[]
    for e in raw:
        try:
            imp=(e.get("Importance","") or e.get("Impact","")).lower()
            if "high" not in imp: continue
            cc=(e.get("Country","") or e.get("Currency","") or "").upper()
            dt=e.get("DateTime","") or e.get("Date","")
            when=pd.to_datetime(dt,utc=True).to_pydatetime()
            ev.append({"ccy":cc,"when":when.isoformat()})
        except Exception: continue
    _save_news_cache({"date":today,"events":ev})
    return ev

def news_blackout(symbol_name, lookahead_min=NEWS_LOOKAHEAD_MIN):
    if not NEWS_FILTER_ON: return (False,"")
    events=refresh_news_today()
    if not events: return (False,"")
    now=datetime.now(timezone.utc); rel=_relevant_currencies(symbol_name)
    for e in events:
        try: when=datetime.fromisoformat(e["when"])
        except Exception: continue
        if e["ccy"] in rel and abs((when-now).total_seconds())<=lookahead_min*60:
            return (True,f"Tin mạnh ({e['ccy']}) ±{lookahead_min}’")
    return (False,"")

def log_signal(symbol, plan, entry, sl, tp, conf, regime, lots, reason=""):
    need_hdr = not os.path.exists(SIGNAL_CSV_PATH)
    try:
        with open(SIGNAL_CSV_PATH,"a",encoding="utf-8") as f:
            if need_hdr: f.write("ts,symbol,plan,entry,sl,tp,conf,regime,lots,reason\n")
            t=datetime.now(timezone.utc).isoformat()
            f.write(f"{t},{symbol},{plan},{entry},{sl},{tp},{conf},{regime},{lots},{reason}\n")
    except Exception: pass
# ---------- Fetch open positions risk ----------
def fetch_open_positions_risk():
    """
    Giả lập tổng rủi ro của các lệnh đang mở (dựa theo file logs signals.csv).
    Trong bản thực chiến, bạn có thể kết nối API của sàn (OKX, Exness, v.v.)
    để lấy chính xác các lệnh đang mở và SL tương ứng.
    """
    total_risk_money = 0.0
    try:
        if not os.path.exists(SIGNAL_CSV_PATH):
            return 0.0
        df = pd.read_csv(SIGNAL_CSV_PATH)
        df = df[df["plan"].isin(["LONG", "SHORT"])]
        df = df.sort_values("ts", ascending=False).drop_duplicates("symbol")

        eq = account_equity_usd()
        for _, row in df.iterrows():
            lots = float(row.get("lots", 0))
            conf = float(row.get("conf", 0))
            rpct = dynamic_risk_pct(conf, row.get("regime", "TREND"))
            total_risk_money += eq * rpct
        return total_risk_money
    except Exception as e:
        logging.warning(f"fetch_open_positions_risk failed: {e}")
        return 0.0
def get_realtime_price(symbol: str) -> float | None:
    """Lấy giá gần nhất (close của nến nhỏ) để kiểm tra lệch so với Entry."""
    try:
        df = fetch_candles(symbol, "15min")  # an toàn cho quota hơn 1m
        if df is None or len(df) < 2:
            return None
        return float(df["close"].iloc[-1])   # giá hiện hành ~ close nến mới nhất
    except Exception:
        return None

def micro_trend_ok(symbol: str, expect: str) -> bool:
    """Xác nhận micro-trend (15m mặc định) cùng/không ngược với hướng kỳ vọng."""
    df = fetch_candles(symbol, MICROTREND_TF)
    d  = strong_trend(df)
    if MICROTREND_ALLOW_SIDEWAY and d == "SIDEWAY":
        return True
    if expect in ("LONG","SHORT") and d in ("LONG","SHORT"):
        return d == expect
    return False

def is_high_vol(symbol: str) -> bool:
    """Định danh biến động cao theo BBWidth & ATR trên 2h."""
    df2 = fetch_candles(symbol, "2h")
    if df2 is None or len(df2) < 60:
        return False
    bw  = bb_width(df2, 20)
    a   = atr(df2, 14)
    a20 = pd.Series(df2["close"]).diff().abs().rolling(14).mean().iloc[-1]  # proxy mềm
    cond_bw  = (not np.isnan(bw)) and bw > VOL_BW_HIGH
    cond_atr = (not np.isnan(a)) and a20 and (a > VOL_ATR_MULT * a20)
    return bool(cond_bw or cond_atr)

def bias_invalidation(symbol: str, expect: str) -> bool:
    """
    Trả True nếu bias bị vô hiệu trong ~1h: RSI/MACD trái hướng liên tục 3 bar trên MICROTREND_TF.
    """
    df = fetch_candles(symbol, MICROTREND_TF)
    if df is None or len(df) < 25:
        return False
    mh = df["close"].ewm(span=12, adjust=False).mean() - df["close"].ewm(span=26, adjust=False).mean()
    sig= mh.ewm(span=9, adjust=False).mean()
    hist = (mh - sig).tail(5)
    rsi15 = rsi(df["close"], 14).tail(5)
    if expect == "LONG":
        return (hist.lt(0).tail(3).all()) and (rsi15.tail(3).lt(50).all())
    if expect == "SHORT":
        return (hist.gt(0).tail(3).all()) and (rsi15.tail(3).gt(50).all())
    return False
# ==== FILTER HELPERS (thêm) ====
def has_volume_spike(df, n=20, mult=1.2):
    """Volume nến đã ĐÓNG >= mult * MA(n). Nếu không có cột volume thì bỏ qua."""
    if df is None or "volume" not in df.columns or len(df) < n + 2:
        return True  # không có volume thì không chặn
    v = pd.to_numeric(df["volume"], errors="coerce")
    vma = v.rolling(n).mean().iloc[-2]
    if np.isnan(v.iloc[-2]) or np.isnan(vma) or vma <= 0:
        return True
    return v.iloc[-2] >= mult * vma

def is_bullish_engulfing(df):
    """Nến -2 xanh phủ thân nến -3 đỏ."""
    if df is None or len(df) < 3: return False
    o2,c2 = float(df["open"].iloc[-2]),  float(df["close"].iloc[-2])
    o3,c3 = float(df["open"].iloc[-3]),  float(df["close"].iloc[-3])
    return (c2 > o2) and (c3 < o3) and (o2 <= c3) and (c2 >= o3)

def is_bearish_engulfing(df):
    """Nến -2 đỏ phủ thân nến -3 xanh."""
    if df is None or len(df) < 3: return False
    o2,c2 = float(df["open"].iloc[-2]),  float(df["close"].iloc[-2])
    o3,c3 = float(df["open"].iloc[-3]),  float(df["close"].iloc[-3])
    return (c2 < o2) and (c3 > o3) and (o2 >= c3) and (c2 <= o3)

def is_bullish_pinbar(df, tol=0.33):
    """Pin bar tăng: đuôi dưới dài (>= 2/3 toàn nến), đóng > mở."""
    if df is None or len(df) < 2: return False
    o,c,h,l = [float(df[x].iloc[-2]) for x in ["open","close","high","low"]]
    rng = max(1e-9, h - l)
    lower_tail = min(o,c) - l
    body = abs(c - o)
    return (c > o) and (lower_tail / rng >= tol) and (body / rng <= 1 - tol)

def is_bearish_pinbar(df, tol=0.33):
    """Pin bar giảm: đuôi trên dài (>= 2/3 toàn nến), đóng < mở."""
    if df is None or len(df) < 2: return False
    o,c,h,l = [float(df[x].iloc[-2]) for x in ["open","close","high","low"]]
    rng = max(1e-9, h - l)
    upper_tail = h - max(o,c)
    body = abs(c - o)
    return (c < o) and (upper_tail / rng >= tol) and (body / rng <= 1 - tol)
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
            #time.sleep(60.0 / RPM)
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
    
    # === (NEW) Chuẩn hoá confidence có trọng số TF
    raw_conf = weighted_confidence(symbol, raw_dir)
    # === (NEW) Filter sideway chặt hơn theo ADX & BBWidth (2H làm chính)
    df2 = fetch_candles(symbol, "2h")
    bw2 = bb_width(df2, 20) if df2 is not None else np.nan
    a2  = adx(df2, 14) if df2 is not None else np.nan
    # Ngưỡng khác nhau theo loại sản phẩm
    if is_crypto(symbol) or is_crypto(name):
        BW_MIN = 0.020   # 2.0%
    elif is_commodity(symbol) or is_commodity(name):
        BW_MIN = 0.015   # 1.5%
    else:  # FX
        BW_MIN = 0.012   # 1.2%

    sideway_block = (np.isnan(a2) or a2 < 20) or (np.isnan(bw2) or bw2 < BW_MIN)    
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
    # === PROP DESK GUARDS ===
    raw_conf = calibrate_confidence(raw_conf, results, raw_dir)
    regime = "TREND" if results.get("4H") in ("LONG","SHORT") else "RANGE"
    
    blocked_news, news_msg = news_blackout(name)
    stats = load_stats()
    cb_on  = is_circuit_breaker_on(stats)
    block_reason = ""
    if blocked_news: block_reason = f"NEWS: {news_msg}"
    elif cb_on:      block_reason = "Circuit breaker cooling"
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

    # ===== Entry/SL/TP (GIỮ NGUYÊN cấu trúc cũ của bạn) =====
    plan = "SIDEWAY"
    entry = sl = tp = atrval = None
    lots = 0.0

    MAIN_TF = os.getenv("MAIN_TF", "2h")  # giữ env như bạn đang dùng
    df_main = fetch_candles(symbol, MAIN_TF)

    if df_main is not None and len(df_main) > 60:
        entry  = float(df_main["close"].iloc[-2])  # dùng nến đã đóng
        atrval = atr(df_main, 14)
        swing_hi, swing_lo = swing_levels(df_main, 20)

        # hệ số ATR theo loại sản phẩm (giữ quy ước cũ)
        base_mult = 2.5 if is_fx(symbol) or is_fx(name) else 1.5

        # (NEW) sideway filter: nếu sideway_block → KHÔNG đề xuất lệnh
        if sideway_block:
            plan = "SIDEWAY"
            entry = sl = tp = None
        else:
            if final_dir == "LONG" and final_conf >= CONF_THRESHOLD:
                plan = "LONG"
                sl_candidates = [
                    entry - base_mult * atrval,
                    (swing_lo - 0.3 * atrval) if not np.isnan(swing_lo) else entry - base_mult * atrval
                ]
                sl = min(sl_candidates)
                R = entry - sl
                rr_tp = min(max(1.2 * R, 1.5 * atrval), 3.0 * atrval)
                cap = None
                if not np.isnan(swing_hi):
                    cap = max(0.8 * atrval, (swing_hi + 0.4 * atrval) - entry)
                tp_dist = min(rr_tp, cap) if (cap is not None and cap > 0) else rr_tp
                tp = entry + tp_dist

            elif final_dir == "SHORT" and final_conf >= CONF_THRESHOLD:
                plan = "SHORT"
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
        # … sau khi đã có entry/sl/tp …
        if is_wti_name(name) and all(v is not None for v in (entry, sl, tp)):
            entry = oil_adjust(entry)
            sl    = oil_adjust(sl)
            tp    = oil_adjust(tp)
        # === ENTRY WINDOW: reset khi đã sang nến CHÍNH mới (dựa đúng timestamp của df_main) ===
        try:
            # Lấy timestamp của nến đã ĐÓNG dùng cho entry (phải là index -2)
            last_closed_ts = pd.to_datetime(df_main["datetime"].iloc[-2], utc=True).to_pydatetime()
            # Bảo đảm có tzinfo (UTC)
            if last_closed_ts.tzinfo is None:
                last_closed_ts = last_closed_ts.replace(tzinfo=timezone.utc)
        
            # Suy ra độ dài khung MAIN_TF
            tf_minutes_map = {"15min": 15, "30min": 30, "1h": 60, "2h": 120, "4h": 240}
            tf_minutes = tf_minutes_map.get(MAIN_TF, 120)
        
            next_candle_ts = last_closed_ts + timedelta(minutes=tf_minutes)
        
            now_utc = datetime.now(timezone.utc)
        
            # Nếu đã qua thời điểm mở nến kế tiếp -> reset tín hiệu để chờ setup mới
            if now_utc >= next_candle_ts:
                entry = sl = tp = None
                plan = "SIDEWAY"
                # (tùy chọn) ghi log nhẹ để theo dõi
                logging.info(f"[ENTRY] reset: last_closed={last_closed_ts.isoformat()} "
                             f"next={next_candle_ts.isoformat()} now={now_utc.isoformat()}")
        except Exception as e:
            logging.warning(f"[WARN] ENTRY WINDOW check failed: {e}")
            # Không chặn tín hiệu nếu check lỗi
            pass
        # ====== 5 FILTER NÂNG WINRATE (thêm ngay sau khi đã có entry/sl/tp) ======
        # Gom lý do chặn vào block_reason (nếu đã có sẵn thì nối thêm)
        reasons = []

        # 4H bias phải trùng hướng trade
        bias4 = _norm_dir(results.get("4H", "N/A"))
        if final_dir in ("LONG","SHORT") and bias4 in ("LONG","SHORT") and bias4 != final_dir:
            reasons.append("4H bias mismatch")

        # RSI + MACD phải cùng hướng với final_dir (nến đã đóng)
        try:
            rsi_last = rsi(df_main["close"], 14).iloc[-2]
            macd_h   = macd_hist(df_main["close"])
            if final_dir == "LONG"  and (rsi_last < 55 or macd_h <= 0): reasons.append("RSI/MACD not aligned")
            if final_dir == "SHORT" and (rsi_last > 45 or macd_h >= 0): reasons.append("RSI/MACD not aligned")
        except Exception:
            pass  # nếu lỗi chỉ báo thì không chặn

        # Volume spike xác nhận (nếu có cột volume)
        if not has_volume_spike(df_main, n=20, mult=1.2):
            reasons.append("No volume confirmation")

        # Price Action: Engulfing / Pin bar theo hướng
        bull_ok = is_bullish_engulfing(df_main) or is_bullish_pinbar(df_main)
        bear_ok = is_bearish_engulfing(df_main) or is_bearish_pinbar(df_main)
        if final_dir == "LONG"  and not bull_ok:  reasons.append("No bullish PA (engulf/pin)")
        if final_dir == "SHORT" and not bear_ok:  reasons.append("No bearish PA (engulf/pin)")

        # RR phải đạt tối thiểu 1.8
        if entry is not None and sl is not None and tp is not None:
            rr_ratio = abs(tp - entry) / max(1e-9, abs(entry - sl))
            if rr_ratio < 1.8:
                reasons.append(f"RR too low ({rr_ratio:.2f} < 1.8)")

        # Nếu có bất kỳ lý do -> huỷ đề xuất lệnh, ghi reason
        if reasons:
            plan = "SIDEWAY"
            entry = sl = tp = None
            lots = 0.0
            # nếu bạn đã có biến block_reason trước đó:
            try:
                if block_reason:
                    block_reason = block_reason + " | " + " | ".join(reasons)
                else:
                    block_reason = " | ".join(reasons)
            except NameError:
                block_reason = " | ".join(reasons)
        # ==== (NEW) 5 GUARDS chống “đảo chiều giữa nến” ====        
        # 1) Dynamic candle validation: giá hiện tại lệch quá xa Entry -> bỏ
        if entry is not None and sl is not None and tp is not None:
            px_now = get_realtime_price(symbol)
            if px_now is not None and abs(px_now - entry) > INTRABAR_PRICE_DEVIATION_ATR * atrval:
                plan = "SIDEWAY"; entry = sl = tp = None
                block_reason = f"Price deviated > {INTRABAR_PRICE_DEVIATION_ATR}×ATR"

        # 2) Micro-trend confirmation (15m cùng hướng hoặc ít nhất không ngược)
        if entry is not None and sl is not None and tp is not None:
            if not micro_trend_ok(symbol, final_dir):
                plan = "SIDEWAY"; entry = sl = tp = None
                block_reason = f"Micro-trend {MICROTREND_TF} disagrees"

        # 3) Adaptive volatility regime: biến động cao thì bắt buộc micro-confirm chặt hơn
        if entry is not None and sl is not None and tp is not None and is_high_vol(symbol):
            # Trong chế độ high-vol chỉ cho đi khi micro-trend đúng hướng KÈM khoảng lệch rất nhỏ
            px_now = px_now if 'px_now' in locals() and px_now is not None else get_realtime_price(symbol)
            if (not micro_trend_ok(symbol, final_dir)) or (px_now is not None and abs(px_now - entry) > 0.3 * atrval):
                plan = "SIDEWAY"; entry = sl = tp = None
                block_reason = "High-volatility guard"

        # 4) Real-time bias tracking: nếu bias bị vô hiệu trong ~1h -> bỏ
        if entry is not None and sl is not None and tp is not None:
            if bias_invalidation(symbol, final_dir):
                plan = "SIDEWAY"; entry = sl = tp = None
                block_reason = "Bias invalidated intrabar"

        # 5) (giữ nguyên) — nếu tên là dầu WTI thì hiệu chỉnh sang quote Exness
        if is_wti_name(name) and all(v is not None for v in (entry, sl, tp)):
            entry = oil_adjust(entry); sl = oil_adjust(sl); tp = oil_adjust(tp)
        # (NEW) Position sizing — chỉ khi có SL/TP hợp lệ
        if entry is not None and sl is not None and tp is not None:
            lots = compute_lot_size(entry, sl, symbol, name)
        if entry is not None and sl is not None and tp is not None and not block_reason:
            rpct = dynamic_risk_pct(final_conf, regime)
            lots = compute_lot_size(entry, sl, symbol, name, risk_pct=rpct)
        
            est_risk_money = account_equity_usd()*rpct
            # --- New: Daily risk cap based on *actual open trades* ---
            open_risk_money = fetch_open_positions_risk()
            day_cap_money   = account_equity_usd() * DAILY_RISK_CAP_PCT
            
            if open_risk_money + (account_equity_usd() * rpct) > day_cap_money:
                plan = "SIDEWAY"
                entry = sl = tp = None
                block_reason = f"Daily risk cap reached ({int(DAILY_RISK_CAP_PCT*100)}%)"
            else:
                # Không cộng risk_used nữa (vì chỉ tính lệnh thực)
                log_signal(name, plan, entry, sl, tp, final_conf, regime, lots)
    # Trả thêm 'final_conf' để in ra Telegram (nếu bạn muốn)
    return results, plan, entry, sl, tp, atrval, True, final_dir, int(round(final_conf)), lots, block_reason

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
    # === Hiệu chuẩn dầu tự động (nếu có ticker bên Exness) ===
    global _OIL_SCALE, _OIL_OFFSET
    _OIL_SCALE, _OIL_OFFSET = compute_oil_calibration()
    logging.info(f"Oil calibration: scale={_OIL_SCALE:.4f}, offset={_OIL_OFFSET:.4f}")
    lines = []
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    lines.append("💵 TRADE GOODS")
    lines.append(f"⏱ {now}\n")

    any_symbol_has_data = False

    for name, sym in symbols.items():
        results, plan, entry, sl, tp, atrval, has_data, final_dir, final_conf, lots, block_reason = analyze_symbol(name, sym, daily_cache)
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
        if fast_flip:
            lines.append(f"⚡ Fast-flip 2H active — chờ nến kế tiếp")        

        # thêm Confidence + Regime (không ảnh hưởng logic cũ)
        #regime = "TREND" if results.get("4H") in ("LONG","SHORT") else "RANGE"
        #lines.append(f"Confidence: {final_conf}% | Regime: {regime}")

        if entry is not None and sl is not None and tp is not None:
            lines.append(
                f"Entry {format_price(name if name in ('EUR/USD','USD/JPY') else sym, entry)} | "
                f"SL {format_price(name if name in ('EUR/USD','USD/JPY') else sym, sl)} | "
                f"TP {format_price(name if name in ('EUR/USD','USD/JPY') else sym, tp)}"
                + (f" | Size {lots:.3f} lot" if lots and lots>0 else "")
            )
        elif block_reason:
            lines.append(f"⛔ {block_reason}")

        # dàn request để không vượt quota
        time.sleep(10)

    # Nếu tất cả đều N/A/SIDEWAY & không có Entry -> vẫn gửi để biết trạng thái; nếu muốn có thể chặn tại đây
    #msg = "\n".join(lines)
    #send_telegram(msg)
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
