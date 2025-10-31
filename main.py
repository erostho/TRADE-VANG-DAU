import re
import os
import time
import json
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials
import gdown
from pydrive2.auth import ServiceAccountCredentials
import os, logging, mimetypes
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import os, json, logging, time
from google.oauth2 import service_account   # <-- BẮT BUỘC
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
ALLOW_RANGE_IN_BACKTEST = True
CONF_MIN_BACKTEST = 0.55
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
    #"USD/JPY": "USD/JPY",
    "EUR/USD": "EUR/USD",
    #"AUD/USD": "AUD/USD",
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
# === AUTO OFFSET EXNESS ALIGNER ===
def get_price_twelvedata(symbol: str, api_key: str) -> float | None:
    try:
        url = f"https://api.twelvedata.com/price?symbol={symbol}&apikey={api_key}"
        r = requests.get(url, timeout=8).json()
        if "price" in r:
            return float(r["price"])
    except Exception as e:
        logging.warning(f"TwelveData error {symbol}: {e}")
    return None

def get_manual_price_exness(symbol: str) -> float | None:
    """
    Lấy giá Exness thủ công (nhập tay qua biến môi trường).
    Ví dụ: EXNESS_XAU=61.35
    """
    key = f"EXNESS_{symbol.replace('/', '_').upper()}"
    val = os.getenv(key)
    try:
        return float(val) if val else None
    except:
        return None
# cache 1 lần chạy
RUN_CACHE = {}
def fetch_candles(symbol, interval, retries=3, use_cache=True):
    key = (symbol, interval)
    if use_cache and key in RUN_CACHE:
        return RUN_CACHE[key]

    url = f"{BASE_URL}?symbol={symbol}&interval={interval}&apikey={API_KEY}&outputsize=200"
    for attempt in range(retries):
        try:
            _throttle()
            r = requests.get(url, timeout=10)
            if r.status_code == 429:
                logging.warning(f"429 {symbol}-{interval} -> sleep 65s & retry...")
                time.sleep(65); continue
            data = r.json()
            if "values" not in data:
                logging.warning(f"No data for {symbol}-{interval}: {data}")
                return None
            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)  # BẮT BUỘC UTC
            df = df.sort_values("datetime").reset_index(drop=True)
            for col in ["open","high","low","close"]:
                df[col] = pd.to_numeric(df[col])
            RUN_CACHE[key] = df
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

# Giới hạn lot & đòn bẩy
MAX_LOT = float(os.getenv("MAX_LOT", "1.0"))     # tuỳ sàn, ví dụ 0.5 cho crypto
MIN_LOT = float(os.getenv("MIN_LOT", "0.001"))
LEVERAGE = float(os.getenv("LEVERAGE", "10"))    # nếu futures có đòn bẩy

def compute_lot_size(entry, sl, symbol, name, risk_pct=0.005):
    if entry is None or sl is None:
        return 0.0
    dist = abs(entry - sl)
    if dist <= 0:
        return 0.0

    key = name if name in CONTRACT_SIZES else symbol
    contract = CONTRACT_SIZES.get(key, 100000)

    # bước giá
    if "JPY" in key:       pipsize = 0.01
    elif "XAU" in key:     pipsize = 0.1
    elif key == "CL":      pipsize = 0.01
    elif "/" in key:       pipsize = 0.0001
    else:                  pipsize = 1.0

    value_per_point = contract * pipsize
    risk_money = account_equity_usd() * risk_pct

    lots = risk_money / max(1e-9, (dist * value_per_point))
    # hiệu chỉnh theo leverage nếu cần
    #lots = lots / max(1.0, LEVERAGE)
    # Theo chuẩn risk-by-SL không cần chia theo leverage.
    # Nếu muốn cực kỳ bảo thủ, bật ENV APPLY_LEVERAGE_ON_SIZE=1
    if os.getenv("APPLY_LEVERAGE_ON_SIZE", "0") == "1":
        lots = lots / max(1.0, LEVERAGE)
    # kẹp biên
    if np.isnan(lots) or lots <= 0:
        lots = MIN_LOT
    lots = max(MIN_LOT, min(MAX_LOT, lots))
    return round(lots, 3)

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
    """
    SL/TP thông minh:
    - SL ưu tiên gần nhưng không chặt hơn 0.8*ATR
    - TP luôn có khoảng cách dương và RR tối thiểu 1.8R (fallback)
    - Có 'cap' bởi swing/keltner nếu phù hợp
    """
    if entry is None or atr is None or atr <= 0:
        return None, None

    base_mult = 2.0 if is_fx else 1.2   # SL “vừa tay”
    buf = 0.3 * atr

    if side == "LONG":
        # --- SL
        sl_candidates = [
            entry - base_mult * atr,
            (swing_lo - buf) if not np.isnan(swing_lo) else entry - base_mult * atr,
        ]
        sl = max(sl_candidates)                          # gần hơn cho LONG
        sl = min(sl, entry - 0.8 * atr)                  # vẫn cách tối thiểu 0.8ATR
        R = max(1e-9, entry - sl)                        # risk per unit

        # --- RR mục tiêu + cap
        rr_tp = min(max(1.8 * R, 1.2 * atr), 3.0 * atr)  # khung an toàn
        cap = None
        if not np.isnan(kup):       cap = max(cap or 0.0, max(0.0, kup - entry))
        if not np.isnan(swing_hi):  cap = max(cap or 0.0, max(0.0, swing_hi - entry - buf))

        tp_dist = rr_tp if (cap is None or cap <= 0) else min(rr_tp, cap)

        # --- Fallback cứng: luôn dương & đủ xa
        if (tp_dist is None) or (tp_dist <= 0) or np.isnan(tp_dist):
            tp_dist = max(1.8 * R, 1.2 * atr)

        tp = entry + tp_dist
        return sl, tp

    elif side == "SHORT":
        # --- SL
        sl_candidates = [
            entry + base_mult * atr,
            (swing_hi + buf) if not np.isnan(swing_hi) else entry + base_mult * atr,
        ]
        sl = min(sl_candidates)                          # gần hơn cho SHORT
        sl = max(sl, entry + 0.8 * atr)                  # vẫn cách tối thiểu 0.8ATR
        R = max(1e-9, sl - entry)

        # --- RR mục tiêu + cap
        rr_tp = min(max(1.8 * R, 1.2 * atr), 3.0 * atr)
        cap = None
        if not np.isnan(kdn):       cap = max(cap or 0.0, max(0.0, entry - kdn))
        if not np.isnan(swing_lo):  cap = max(cap or 0.0, max(0.0, entry - swing_lo - buf))

        tp_dist = rr_tp if (cap is None or cap <= 0) else min(rr_tp, cap)

        # --- Fallback cứng
        if (tp_dist is None) or (tp_dist <= 0) or np.isnan(tp_dist):
            tp_dist = max(1.8 * R, 1.2 * atr)

        tp = entry - tp_dist
        return sl, tp

    return None, None

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
# === AUTO CALIBRATION: Align TV -> Exness for GOLD, SILVER, OIL ==========
CALIB_TTL_MIN = int(os.getenv("CALIB_TTL_MIN", "60"))
CALIB_CACHE_PATH = os.getenv("CALIB_CACHE_PATH", "/tmp/calib_align.json")

def _load_calib_cache():
    try:
        return json.load(open(CALIB_CACHE_PATH, "r", encoding="utf-8"))
    except Exception:
        return {}

def _save_calib_cache(data):
    try:
        json.dump(data, open(CALIB_CACHE_PATH, "w", encoding="utf-8"))
    except Exception:
        pass

def _minutes_since(ts_iso: str) -> float:
    try:
        t = datetime.fromisoformat(ts_iso)
        return (datetime.now(timezone.utc) - t).total_seconds()/60.0
    except Exception:
        return 1e9

def compute_symbol_calibration(symbol: str, exness_symbol: str) -> float:
    """
    Tính offset trung vị (median) giữa TwelveData (TV) và Exness.
    offset = median(exness - tv)
    Trả về offset (float).
    """
    cache = _load_calib_cache()
    key = f"{symbol}-{exness_symbol}"
    if key in cache and _minutes_since(cache[key]["ts"]) < CALIB_TTL_MIN:
        return cache[key]["offset"]

    df_tv = fetch_candles(symbol, "1h")
    df_ex = fetch_candles(exness_symbol, "1h")
    if df_tv is None or df_ex is None:
        return 0.0

    z = pd.merge_asof(
        df_tv[["datetime", "close"]].rename(columns={"close": "tv"}),
        df_ex[["datetime", "close"]].rename(columns={"close": "ex"}),
        on="datetime", direction="nearest", tolerance=pd.Timedelta("30min")
    ).dropna()

    if len(z) < 10:
        return 0.0

    offset = float(np.median(z.tail(50)["ex"] - z.tail(50)["tv"]))
    cache[key] = {"offset": offset, "ts": datetime.now(timezone.utc).isoformat()}
    _save_calib_cache(cache)
    # --- Debug log giá Exness vs TV ---
    try:
        latest_ex = z["ex"].iloc[-1]
        latest_tv = z["tv"].iloc[-1]
        logging.info(f"[CALIB DEBUG] {symbol}: Exness={latest_ex:.2f}, TV={latest_tv:.2f}, Offset={offset:+.2f}")
    except Exception as e:
        logging.warning(f"[CALIB DEBUG] Could not log calibration data for {symbol}: {e}")
    # --- end debug log ---
    return offset

def apply_symbol_calibration(symbol: str, entry, sl, tp):
    """
    Tinh chỉnh Entry/SL/TP về giá Exness cho kim loại (XAU, XAG).
    An toàn: nếu thiếu dữ liệu thì trả về nguyên trạng.
    """
    # Chưa có lệnh -> không làm gì
    if not all(v is not None for v in (entry, sl, tp)):
        return entry, sl, tp, ""

    # Chỉ align cho vàng/bạc; dầu đã có oil_adjust rồi
    if symbol not in ("XAU/USD", "XAG/USD"):
        return entry, sl, tp, ""

    # Map đích sang Exness (nếu bạn dùng ký hiệu khác thì sửa ở đây)
    exness_symbol = symbol  # "XAU/USD" hoặc "XAG/USD"

    # Tính offset; nếu lỗi/None thì coi như 0
    try:
        offset = compute_symbol_calibration(symbol, exness_symbol)
    except Exception:
        offset = 0.0

    if offset is None or (isinstance(offset, float) and np.isnan(offset)):
        offset = 0.0

    # Nếu lệch quá lớn thì bỏ qua để tránh dữ liệu sai
    if abs(offset) > 30:
        return entry, sl, tp, f"skip_large_offset {offset:+.2f}"

    return entry + offset, sl + offset, tp + offset, f"aligned {offset:+.2f}"
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

def session_ok(name_or_sym: str, now_utc: datetime | None = None) -> bool:
    """
    Lọc phiên giao dịch theo sản phẩm để giảm nhiễu:
    - XAU/WTI: 07:00–20:00 UTC (London+NY)
    - USD/JPY (FX): (00:00–11:00) hoặc (12:00–19:00) UTC
    - Crypto (BTC/ETH): tránh 02:00–06:00 UTC
    """
    up = (name_or_sym or "").upper()
    now_utc = now_utc or datetime.now(timezone.utc)
    h = now_utc.hour

    if ("XAU" in up) or ("GOLD" in up) or ("CL" in up) or ("OIL" in up):
        return 7 <= h <= 20
    if ("USD/JPY" in up) or ("EUR/USD" in up):
        return (0 <= h <= 11) or (12 <= h <= 19)
    if ("BTC" in up) or ("ETH" in up):
        return not (2 <= h <= 6)
    return True
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

def tf_to_timedelta(tf: str) -> timedelta:
    tf = tf.lower()
    if tf.endswith("min"):
        return timedelta(minutes=int(tf.replace("min","")))
    if tf.endswith("h"):
        return timedelta(hours=int(tf.replace("h","")))
    if tf in ("1day","1d","day"):
        return timedelta(days=1)
    # mặc định 2h
    return timedelta(hours=2)
# ================ CORE ANALYZE ================
def analyze_symbol(name, symbol, daily_cache):
    results = {}
    has_data = False
    fast_bear = False
    block_reason = ""
    reasons: list[str] = []           # dùng 'reason' cho thống nhất
    blocked_news: bool = False
    news_msg: str = ""
    cb_on: bool = False
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
    #block_reason = ""
    #if blocked_news: block_reason = f"NEWS: {news_msg}"
    #elif cb_on:      block_reason = "Circuit breaker cooling"
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
    MAIN_TF = os.getenv("MAIN_TF", "2h")           # TF chính
    df_main = fetch_candles(symbol, MAIN_TF, use_cache=False)   # luôn lấy nến mới
    
    if df_main is not None and len(df_main) > 60:
        # đảm bảo datetime là UTC
        if df_main["datetime"].dt.tz is None:
            df_main["datetime"] = pd.to_datetime(df_main["datetime"], utc=True)

        # --- tính entry/ATR/swing như cũ, dùng nến đã đóng (-2) ---
        entry   = float(df_main["close"].iloc[-2])
        atrval  = atr(df_main, 14)
        swing_hi, swing_lo = swing_levels(df_main, 20)

        # hệ số ATR theo loại sản phẩm
        base_mult = 2.5 if (is_fx(symbol) or is_fx(name)) else 1.5

        # sideway filter
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
                # ---- Smart SL/TP tự động theo ATR + Keltner + Swing ----
                try:
                    k_mid, kup, kdn = keltner_mid(df_main, 20, atr_mult=1.0)
                    sl, tp = smart_sl_tp(
                        entry, atrval, swing_hi, swing_lo, kup, kdn, final_dir,
                        is_fx(symbol) or is_fx(name)
                    )
                except Exception as e:
                    logging.warning(f"⚠️ smart_sl_tp fallback: {e}")

            elif final_dir == "SHORT" and final_conf >= CONF_THRESHOLD:
                plan = "SHORT"
                sl_candidates = [
                    entry + base_mult * atrval,
                    (swing_hi + 0.3 * atrval) if not np.isnan(swing_hi) else entry + base_mult * atrval
                ]
                # ---- Smart SL/TP tự động theo ATR + Keltner + Swing ----
                try:
                    k_mid, kup, kdn = keltner_mid(df_main, 20, atr_mult=1.0)
                    sl, tp = smart_sl_tp(
                        entry, atrval, swing_hi, swing_lo, kup, kdn, final_dir,
                        is_fx(symbol) or is_fx(name)
                    )
                except Exception as e:
                    logging.warning(f"⚠️ smart_sl_tp fallback: {e}")
        # Fallback cuối: đảm bảo TP cách Entry đủ xa
        if sl is not None and tp is not None and entry is not None:
            R = max(1e-9, abs(entry - sl))
            if abs(tp - entry) < max(1.8 * R, 1.2 * atrval):
                if final_dir == "LONG":
                    tp = entry + max(1.8 * R, 1.2 * atrval)
                elif final_dir == "SHORT":
                    tp = entry - max(1.8 * R, 1.2 * atrval)
        # === Nếu là dầu/XAU/XAG thì hiệu chỉnh giá ===
        if is_wti_name(name) and all(v is not None for v in (entry, sl, tp)):
            entry = oil_adjust(entry)
            sl    = oil_adjust(sl)
            tp    = oil_adjust(tp)
        #if symbol in ("XAU/USD", "XAG/USD"):
            entry, sl, tp, align_note = apply_symbol_calibration(symbol, entry, sl, tp)
            if align_note:
                logging.info(f"[ALIGN] {symbol} {align_note}")
            reasons = []

        # === UPGRADE MR/TRADING FILTER 2025 =========================================
        # Mục tiêu:
        # 1) Cho phép Mean-Reversion (MR) an toàn cho FX khi RANGE
        # 2) Nới RSI/MACD khi RANGE, đặc biệt với MR
        # 3) Bỏ bắt buộc Volume/PA cho FX RANGE (vì dữ liệu volume FX thường sai)
        # 4) RR tối thiểu co giãn theo ADX (1.5 ↔ 2.2)
        # 5) Watchdog: nếu 36h chưa ra kèo -> tạm nới 1 vòng
        # 6) Price Deviation Guard động theo loại sản phẩm (FX chặt hơn)
        
        from dataclasses import dataclass
        
        # ---- cấu hình mềm (có thể đổi bằng env nếu muốn) ----
        RELAX_AFTER_H = int(os.getenv("RELAX_AFTER_H", "36"))   # watchdog 36h
        WDOG_PATH     = os.getenv("WDOG_PATH", "/tmp/signal_watchdog.json")
        
        def _load_wdog():
            try:
                with open(WDOG_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        def _save_wdog(d):
            try:
                with open(WDOG_PATH, "w", encoding="utf-8") as f:
                    json.dump(d, f, ensure_ascii=False)
            except Exception:
                pass
        
        def _minutes_since_iso(ts_iso):
            try:
                t = datetime.fromisoformat(ts_iso)
                return (datetime.now(timezone.utc) - t).total_seconds()/60.0
            except Exception:
                return 1e9
        
        @dataclass
        class MrSetup:
            enabled: bool = False
            side: str = "SIDEWAY"   # LONG/SHORT
            entry: float | None = None
            sl: float | None = None
            tp: float | None = None
            note: str = ""
        
        def _fx_symbol(name_or_sym: str) -> bool:
            up = name_or_sym.upper()
            return ("EUR/USD" in up) or ("USD/JPY" in up) or ("/" in up and "XAU" not in up and "BTC" not in up and "ETH" not in up)
        
        def _price_deviation_limit(symbol_or_name: str, atrv: float) -> float:
            up = symbol_or_name.upper()
            # FX nhạy → lệch tối đa 0.25*ATR; Commodity 0.4; Crypto 0.6
            if _fx_symbol(up):          return 0.25 * atrv
            if ("XAU" in up) or ("CL" in up) or ("OIL" in up):  return 0.40 * atrv
            return 0.60 * atrv
        
        def _rr_min_by_adx(adx_val: float) -> float:
            if np.isnan(adx_val):    return 1.8
            if adx_val < 18:         return 1.50   # yếu => chốt gần
            if adx_val < 25:         return 1.70
            if adx_val < 32:         return 1.85
            return 2.20              # rất mạnh => giữ xa
        
        def _mr_fx_range(df2h, df_main, atrv) -> MrSetup:
            """
            Mean-Reversion cho FX khi RANGE:
              - giá nến -2 chạm/vượt 2σ Bollinger (20) trên 2H
              - entry = close[-2], SL ngoài band ~0.8*ATR, TP về mid Keltner/BB
            """
            setup = MrSetup(enabled=False)
            try:
                if df2h is None or len(df2h) < 60 or df_main is None or len(df_main) < 60:
                    return setup
        
                # BB 20 trên 2H (dùng nến đã đóng)
                close2 = df2h["close"]
                ma = close2.rolling(20).mean()
                sd = close2.rolling(20).std()
                upper = ma + 2*sd
                lower = ma - 2*sd
                c2 = float(close2.iloc[-2])
        
                # Keltner mid để chọn TP
                kmid, kup, kdn = keltner_mid(df_main, 20, atr_mult=1.0)
                # Ngưỡng "chạm biên" mềm: c2 >= upper[-2] * 0.999… để chịu sai số API
                hit_upper = (not np.isnan(upper.iloc[-2])) and (c2 >= float(upper.iloc[-2]) * 0.999)
                hit_lower = (not np.isnan(lower.iloc[-2])) and (c2 <= float(lower.iloc[-2]) * 1.001)
        
                if hit_upper:
                    setup.enabled = True
                    setup.side = "SHORT"
                    setup.entry = float(df_main["close"].iloc[-2])
                    # SL phía trên band 0.8*ATR
                    band_cap = max(0.8*atrv, setup.entry - float(lower.iloc[-2]))
                    setup.sl = setup.entry + band_cap
                    # TP về mid (Keltner hoặc BB ma), đảm bảo tối thiểu 1.5R
                    R = abs(setup.sl - setup.entry)
                    tp_target = min(setup.entry - 1.5*R, float(kmid) if not np.isnan(kmid) else float(ma.iloc[-2]))
                    setup.tp = tp_target
                    setup.note = "MR-FX (upper band)"
                elif hit_lower:
                    setup.enabled = True
                    setup.side = "LONG"
                    setup.entry = float(df_main["close"].iloc[-2])
                    band_cap = max(0.8*atrv, float(upper.iloc[-2]) - setup.entry)
                    setup.sl = setup.entry - band_cap
                    R = abs(setup.entry - setup.sl)
                    tp_target = max(setup.entry + 1.5*R, float(kmid) if not np.isnan(kmid) else float(ma.iloc[-2]))
                    setup.tp = tp_target
                    setup.note = "MR-FX (lower band)"
        
                return setup
            except Exception:
                return setup
        
        # ---- bắt đầu áp dụng --------------------------------------------------------
        try:
            # Tập hợp dữ liệu cần thiết; nếu thiếu, bỏ qua block nâng cấp
            df2h_for_mr = fetch_candles(symbol, "2h")
            adx2 = adx(df2h_for_mr, 14) if df2h_for_mr is not None else np.nan
        
            # RR MIN động theo ADX (áp cho cả setup trend)
            RR_MIN = _rr_min_by_adx(adx2)
        
            # Watchdog (nếu quá lâu không có lệnh, tạm nới điều kiện 1 vòng)
            wdog = _load_wdog()
            sym_key = f"{name}|{symbol}"
            last_ok_ts = wdog.get(sym_key)  # ISO
            too_long = _minutes_since_iso(last_ok_ts) > RELAX_AFTER_H*60 if last_ok_ts else True
        
            # 1) Nếu chưa có plan/entry vì bị RANGE chặn, thử bật MR cho FX
            mr_used = False
            if (plan == "SIDEWAY" or entry is None or sl is None or tp is None):
                if regime == "RANGE" and (_fx_symbol(name) or _fx_symbol(symbol)):
                    _mr = _mr_fx_range(df2h_for_mr, df_main, atrval if atrval else atr(df_main, 14))
                    if _mr.enabled:
                        # gán lại kế hoạch theo MR
                        plan   = _mr.side
                        entry  = _mr.entry
                        sl     = _mr.sl
                        tp     = _mr.tp
                        mr_used = True
                        # confidence: giảm 8–12% so với trend để phân biệt
                        final_conf = max(0, int(round(final_conf - 10)))
                        block_reason = ""  # MR có tín hiệu thì không kèm lỗi cũ

            # 2) Nới RSI/MACD khi RANGE (đặc biệt khi MR)
            if entry is not None and sl is not None and tp is not None:
                try:
                    rsi_last = rsi(df_main["close"], 14).iloc[-2]
                    macd_h   = macd_hist(df_main["close"])
                    # điều kiện "aligned" nhẹ khi RANGE hoặc MR
                    if regime == "RANGE" or mr_used:
                        rsi_ok = (plan == "LONG" and rsi_last >= 48) or (plan == "SHORT" and rsi_last <= 52)
                        macd_ok = (plan == "LONG" and macd_h >= -0.00001) or (plan == "SHORT" and macd_h <= +0.00001)
                    else:
                        rsi_ok = (plan == "LONG" and rsi_last >= 55) or (plan == "SHORT" and rsi_last <= 45)
                        macd_ok = (plan == "LONG" and macd_h > 0) or (plan == "SHORT" and macd_h < 0)
        
                    if not (rsi_ok and macd_ok):
                        # chỉ bỏ nếu KHÔNG phải MR (MR được phép nới hơn)
                        if not mr_used:
                            plan = "SIDEWAY"; entry = sl = tp = None
                            block_reason = "RSI/MACD not aligned (relaxed)"
                except Exception:
                    pass
        
            # 3) Price deviation guard động theo loại sản phẩm
            if entry is not None and sl is not None and tp is not None and atrval is not None:
                px_now = get_realtime_price(symbol)
                lim = _price_deviation_limit(name if name else symbol, atrval)
                if px_now is not None and abs(px_now - entry) > lim:
                    plan = "SIDEWAY"; entry = sl = tp = None
                    block_reason = f"Price deviated > {lim/atrval:.2f}×ATR"
        
            # 4) RR kiểm tra lại với RR_MIN động
            # === Dynamic RR min + smart TP expand + optional SCALP ===
            MIN_RR_BASE = float(os.getenv("MIN_RR_BASE", "1.8"))
            MIN_RR_TREND_BONUS = float(os.getenv("MIN_RR_TREND_BONUS", "-0.2"))
            MIN_RR_HICONF_BONUS = float(os.getenv("MIN_RR_HICONF_BONUS", "-0.2"))
            MIN_RR_CLAMP_LOW = float(os.getenv("MIN_RR_CLAMP_LOW", "1.4"))
            MIN_RR_CLAMP_HIGH = float(os.getenv("MIN_RR_CLAMP_HIGH", "2.2"))
            SCALP_MODE_ON = os.getenv("SCALP_MODE_ON", "1") == "1"
            SCALP_MIN_RR = float(os.getenv("SCALP_MIN_RR", "1.30"))
            SCALP_RISK_MULT = float(os.getenv("SCALP_RISK_MULT", "0.5"))
            
            def _calc_rr(e, s, t):
                return abs(t - e) / max(1e-9, abs(e - s))
            
            if entry is not None and sl is not None and tp is not None:
                rr_ratio = _calc_rr(entry, sl, tp)
                regime = "TREND" if results.get("4H") in ("LONG", "SHORT") else "RANGE"
                rr_min = MIN_RR_BASE
                if regime == "TREND":
                    rr_min += MIN_RR_TREND_BONUS
                if final_conf >= 85:
                    rr_min += MIN_RR_HICONF_BONUS
                rr_min = float(min(MIN_RR_CLAMP_HIGH, max(MIN_RR_CLAMP_LOW, rr_min)))
            
                # thử nới TP lên nếu RR chỉ thiếu chút
                if rr_ratio < rr_min:
                    try:
                        k_mid, kup, kdn = keltner_mid(df_main, 20, atr_mult=1.0)
                    except Exception:
                        kup = kdn = np.nan
            
                    if final_dir == "LONG":
                        desired_tp = entry + rr_min * abs(entry - sl)
                        caps = [desired_tp]
                        if not np.isnan(kup): caps.append(kup)
                        if not np.isnan(swing_hi): caps.append(swing_hi)
                        tp_new = min(caps)
                        rr_new = _calc_rr(entry, sl, tp_new)
                        if rr_new >= rr_min * 0.98:
                            tp = tp_new
                            rr_ratio = rr_new
            
                    elif final_dir == "SHORT":
                        desired_tp = entry - rr_min * abs(entry - sl)
                        caps = [desired_tp]
                        if not np.isnan(kdn): caps.append(kdn)
                        if not np.isnan(swing_lo): caps.append(swing_lo)
                        tp_new = max(caps)
                        rr_new = _calc_rr(entry, sl, tp_new)
                        if rr_new >= rr_min * 0.98:
                            tp = tp_new
                            rr_ratio = rr_new
            
                if rr_ratio < rr_min:
                    if SCALP_MODE_ON and rr_ratio >= SCALP_MIN_RR:
                        reasons.append(f"SCALP mode (RR {rr_ratio:.2f} < {rr_min:.2f})")
                        rpct = dynamic_risk_pct(final_conf, regime) * SCALP_RISK_MULT
                        lots = compute_lot_size(entry, sl, symbol, name, risk_pct=rpct)
                    else:
                        reasons.append(f"RR too low ({rr_ratio:.2f} < {rr_min:.2f})")
        
            # 5) Bỏ bắt buộc Volume/PA cho FX RANGE (nếu code gốc có chặn)
            #    -> thực hiện bằng cách không nối thêm lý do “No volume confirmation/No supportive PA” khi mr_used hoặc (regime==RANGE & FX)
            #    Nếu đoạn reasons đã tạo trước đó, KHÔNG thêm lỗi mới về volume/PA trong nhánh này.
            #    (Phần này “no-op” nếu bạn đặt block trước khi build 'reasons')
        
            # 6) Watchdog: nếu vừa tạo được lệnh hợp lệ -> cập nhật; nếu không, không đổi.
            if entry is not None and sl is not None and tp is not None:
                wdog[sym_key] = datetime.now(timezone.utc).isoformat()
                _save_wdog(wdog)
        
        except Exception as _e:
            logging.warning(f"[UPGRADE-2025] skipped due to: {_e}")

        # ---- Breakout trigger (sau squeeze) ----
        try:
            m = df_main['close'].rolling(20).mean()
            s = df_main['close'].rolling(20).std()
            upper = m + 2*s; lower = m - 2*s
            e20, e50 = ema(df_main['close'], 20), ema(df_main['close'], 50)
            bw = bb_width(df_main, 20)
        
            brk_long  = (df_main['close'].iloc[-2] > upper.iloc[-2]) and (e20.iloc[-2] > e50.iloc[-2])
            brk_short = (df_main['close'].iloc[-2] < lower.iloc[-2]) and (e20.iloc[-2] < e50.iloc[-2])
        
            if plan == "SIDEWAY":  # chỉ dùng làm công tắc bật lệnh khi mọi thứ khác OK
                if brk_long and final_dir == "LONG" and bw < VOL_BW_HIGH:
                    plan = "LONG"
                elif brk_short and final_dir == "SHORT" and bw < VOL_BW_HIGH:
                    plan = "SHORT"
        except Exception:
            pass

        # ---- Pullback-to-MA trigger ----
        try:
            e20 = ema(df_main['close'], 20)
            e50 = ema(df_main['close'], 50)
            c2, o2 = float(df_main['close'].iloc[-2]), float(df_main['open'].iloc[-2])
            r2 = float(rsi(df_main['close'], 14).iloc[-2])
        
            pulled_long  = (min(o2, c2) <= e20.iloc[-2] <= max(o2, c2)) and (c2 > e20.iloc[-2]) and (e20.iloc[-2] > e50.iloc[-2]) and (r2 >= 50)
            pulled_short = (min(o2, c2) <= e20.iloc[-2] <= max(o2, c2)) and (c2 < e20.iloc[-2]) and (e20.iloc[-2] < e50.iloc[-2]) and (r2 <= 50)
        
            if plan == "SIDEWAY":
                if pulled_long and final_dir == "LONG":
                    plan = "LONG"
                elif pulled_short and final_dir == "SHORT":
                    plan = "SHORT"
        except Exception:
            pass
        # === END UPGRADE MR/TRADING FILTER 2025 =====================================
        # ====== 5 FILTER NÂNG WINRATE ======
        #reasons = []
        # 4H bias phải trùng hướng trade
        # ---- Soft filters: chấm điểm thay vì chặn cứng ----
        score_ok = 0
        why = []
        
        # 4H bias cùng hướng
        bias4 = _norm_dir(results.get("4H", "N/A"))
        if final_dir in ("LONG","SHORT") and bias4 == final_dir:
            score_ok += 1
        else:
            why.append("4H bias mismatch")
        
        # RSI & MACD
        try:
            rsi_last = rsi(df_main["close"], 14).iloc[-2]
            macd_h   = macd_hist(df_main["close"])
            rsi_ok   = (final_dir == "LONG"  and rsi_last >= 50) or (final_dir == "SHORT" and rsi_last <= 50)
            macd_ok  = (final_dir == "LONG"  and macd_h > 0)     or (final_dir == "SHORT" and macd_h < 0)
            if rsi_ok and macd_ok:
                score_ok += 1
            else:
                why.append("RSI/MACD not aligned")
        except Exception:
            pass
        
        # Price Action (engulfing/pin) – chỉ cộng điểm nếu có, KHÔNG chặn cứng
        bull_ok = is_bullish_engulfing(df_main) or is_bullish_pinbar(df_main)
        bear_ok = is_bearish_engulfing(df_main) or is_bearish_pinbar(df_main)
        pa_ok = (final_dir == "LONG" and bull_ok) or (final_dir == "SHORT" and bear_ok)
        if pa_ok:
            score_ok += 1
        else:
            why.append("No supportive PA")
        
        # Volume – bỏ qua nếu không có cột hoặc dữ liệu kém
        vol_ok = has_volume_spike(df_main, n=20, mult=float(os.getenv("VOL_SPIKE_MULT","1.05")))
        if vol_ok:
            score_ok += 1
        else:
            why.append("No volume confirmation")
        
        # RR tối thiểu theo độ mạnh tín hiệu
        rr_min = 1.3 if (final_conf >= 80 and bias4 == final_dir) else (1.5 if final_conf >= 70 else 1.8)
        if entry is not None and sl is not None and tp is not None:
            rr_ratio = abs(tp - entry) / max(1e-9, abs(entry - sl))
            if rr_ratio >= rr_min:
                score_ok += 1
            else:
                why.append(f"RR too low ({rr_ratio:.2f} < {rr_min})")
        
        # Ngưỡng đỗ: yêu cầu ít nhất 2 điều kiện (tuỳ chỉnh bằng env)
        need = int(os.getenv("FILTER_PASS_MIN","2"))
        if score_ok < need:
            plan = "SIDEWAY"; entry = sl = tp = None; lots = 0.0
            block_reason = " | ".join(why) if why else "Filters not passed"

        # ==== GUARDS ====
        if entry is not None and sl is not None and tp is not None:
            px_now = get_realtime_price(symbol)
            if px_now is not None and abs(px_now - entry) > INTRABAR_PRICE_DEVIATION_ATR * atrval:
                plan = "SIDEWAY"; entry = sl = tp = None; block_reason = f"Price deviated > {INTRABAR_PRICE_DEVIATION_ATR}×ATR"

        if entry is not None and sl is not None and tp is not None:
            if not micro_trend_ok(symbol, final_dir):
                plan = "SIDEWAY"; entry = sl = tp = None; block_reason = f"Micro-trend {MICROTREND_TF} disagrees"

        if entry is not None and sl is not None and tp is not None and is_high_vol(symbol):
            px_now = px_now if 'px_now' in locals() and px_now is not None else get_realtime_price(symbol)
            if (not micro_trend_ok(symbol, final_dir)) or (px_now is not None and abs(px_now - entry) > 0.3 * atrval):
                plan = "SIDEWAY"; entry = sl = tp = None; block_reason = "High-volatility guard"

        if entry is not None and sl is not None and tp is not None:
            if bias_invalidation(symbol, final_dir):
                plan = "SIDEWAY"; entry = sl = tp = None; block_reason = "Bias invalidated intrabar"
        # --- Session filter: chỉ trade trong giờ "sống" của từng sản phẩm ---
        if entry is not None and sl is not None and tp is not None:
            if not session_ok(name if name else symbol):
                plan = "SIDEWAY"; entry = sl = tp = None
                block_reason = "Out of trading session"
        
        # === Position sizing ===
        if entry is not None and sl is not None and tp is not None:
            rpct = dynamic_risk_pct(final_conf, regime)
            lots = compute_lot_size(entry, sl, symbol, name, risk_pct=rpct)
            open_risk_money = fetch_open_positions_risk()
            day_cap_money   = account_equity_usd() * DAILY_RISK_CAP_PCT
            if open_risk_money + (account_equity_usd() * rpct) > day_cap_money:
                plan = "SIDEWAY"; entry = sl = tp = None
                block_reason = f"Daily risk cap reached ({int(DAILY_RISK_CAP_PCT*100)}%)"
            else:
                log_signal(name, plan, entry, sl, tp, final_conf, regime, lots)
    # Trả thêm 'final_conf' để in ra Telegram (nếu bạn muốn)
    return results, plan, entry, sl, tp, atrval, True, final_dir, int(round(final_conf)), lots, block_reason
# ================= OFFLINE CANDLE CACHE (no API backtest) =================
CANDLE_CACHE_DIR = os.getenv("CANDLE_CACHE_DIR", "/tmp/candles_cache")
os.makedirs(CANDLE_CACHE_DIR, exist_ok=True)


def _safe_name(x: str) -> str:
    x = x.upper().replace("/", "_")
    # Giữ nguyên 1 dấu gạch dưới nếu trùng
    while "__" in x:
        x = x.replace("__", "_")
    return x

def _cache_file(symbol: str, interval: str) -> str:
    return os.path.join(CANDLE_CACHE_DIR, f"{_safe_name(symbol)}_{interval}.parquet")

def save_candles_to_disk(symbol: str, interval: str, df: pd.DataFrame):
    """Gộp incremental và lưu Parquet; gọi ngay MỖI LẦN fetch_candles trả về df."""
    try:
        if df is None or len(df) == 0:
            return
        p = _cache_file(symbol, interval)
        # Chuẩn hoá kiểu và UTC
        d = df.copy()
        d["datetime"] = pd.to_datetime(d["datetime"], utc=True)
        for c in ["open","high","low","close"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=["datetime","open","high","low","close"])
        if os.path.exists(p):
            old = pd.read_parquet(p)
            old["datetime"] = pd.to_datetime(old["datetime"], utc=True)
            merged = pd.concat([old, d], ignore_index=True)
            merged = merged.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
            merged.to_parquet(p, index=False)
        else:
            d.sort_values("datetime").to_parquet(p, index=False)
        # sau khi save Parquet xong
        msg = upload_to_drive_overwrite(p, os.getenv("GOOGLE_DRIVE_FOLDER_ID"))
        logging.info(f"✅ {msg}")
    except Exception as e:
        logging.warning(f"[CACHE] save failed {symbol}-{interval}: {e}")


GOOGLE_DRIVE_FOLDER_ID = "1dPxMrLoy73et8rJDjpC7TDaOGv7RgEQF?usp=drive_link"  # 👈 đổi thành ID của chị

def _drive_creds_from_env():
    client_id = os.getenv("GOOGLE_OAUTH_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET")
    refresh_token = os.getenv("GOOGLE_OAUTH_REFRESH_TOKEN")
    if not (client_id and client_secret and refresh_token):
        logging.warning("⚠️ Thiếu CLIENT_ID/SECRET/REFRESH_TOKEN → bỏ qua upload")
        return None
    # access_token để trống; Google SDK sẽ tự refresh bằng refresh_token
    creds = Credentials(
        None,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret,
        scopes=["https://www.googleapis.com/auth/drive.file"],
    )
    # chủ động refresh 1 lần cho chắc
    try:
        creds.refresh(Request())
    except Exception as e:
        logging.error(f"❌ Không refresh được token: {e}")
        return None
    return creds

SCOPES = ['https://www.googleapis.com/auth/drive.file']
FOLDER_ID = os.getenv('GOOGLE_DRIVE_FOLDER_ID')  # đã set trên Render
def _drive_service():
    """Tạo Drive service từ service account (ưu tiên môi trường)."""
    sa_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not sa_json:
        raise FileNotFoundError("❌ Thiếu biến GOOGLE_SERVICE_ACCOUNT_JSON trong môi trường!")
    info = json.loads(sa_json)
    creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)

from googleapiclient.http import MediaFileUpload
import os, json, requests
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

def build_drive_with_oauth():
    client_id     = os.getenv("GOOGLE_OAUTH_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET")
    access_token  = os.getenv("GOOGLE_OAUTH_ACCESS_TOKEN")
    refresh_token = os.getenv("GOOGLE_OAUTH_REFRESH_TOKEN")

    if not all([client_id, client_secret, access_token, refresh_token]):
        raise RuntimeError("Thiếu biến OAuth (ID/SECRET/ACCESS/REFRESH).")

    creds = Credentials(
        token=access_token,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret,
        scopes=["https://www.googleapis.com/auth/drive.file"]
    )

    # refresh nếu access_token hết hạn
    if not creds.valid:
        request = Request()
        creds.refresh(request)

    return build("drive", "v3", credentials=creds, cache_discovery=False)
def upload_to_drive_overwrite(local_path: str, drive_folder_id: str):
    service = build_drive_with_oauth()
    file_name = os.path.basename(local_path)

    # 1) tìm file trùng tên trong đúng folder
    q = "name = '{}' and '{}' in parents and trashed = false".format(
        file_name.replace("'", r"\'"), drive_folder_id
    )
    res = service.files().list(q=q, fields="files(id, name)").execute()
    exists = res.get("files", [])
    media = MediaFileUpload(local_path, mimetype="application/octet-stream", resumable=True)

    if exists:
        file_id = exists[0]["id"]
        service.files().update(
            fileId=file_id,
            media_body=media
        ).execute()
        return f"Đã **cập nhật (overwrite)** {file_name} lên Drive."
    else:
        service.files().create(
            body={"name": file_name, "parents": [drive_folder_id]},
            media_body=media,
            fields="id"
        ).execute()
        return f"Đã **tạo mới** {file_name} lên Drive."


def download_from_drive(symbol: str, interval: str) -> str | None:
    """Kéo toàn bộ folder cache từ Drive về /tmp rồi lấy đúng file cần.
       Yêu cầu: Folder trên Drive bật 'Anyone with the link (Viewer)'. """
    try:
        import os, subprocess

        folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")  # ví dụ: 1dPxMrLoy73e...RgEQF
        if not folder_id:
            logging.warning("⚠️ Chưa set GOOGLE_DRIVE_FOLDER_ID trong ENV.")
            return None

        # 1) Sync folder Drive -> local cache dir
        folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
        os.makedirs(CANDLE_CACHE_DIR, exist_ok=True)
        # tải toàn bộ folder (nhanh + đơn giản). Không cần cookies.
        subprocess.run(
            ["gdown", "--fuzzy", folder_url, "-O", CANDLE_CACHE_DIR, "-q"],
            check=False
        )

        # 2) Tìm đúng file parquet theo quy tắc đặt tên local
        fname = f"{_safe_name(symbol)}_{interval}.parquet"
        local_path = os.path.join(CANDLE_CACHE_DIR, fname)

        if os.path.exists(local_path):
            logging.info(f"✅ Found cache from Drive: {local_path}")
            return local_path

        logging.warning(f"⚠️ Không thấy file sau khi sync: {fname}")
        return None

    except Exception as e:
        logging.warning(f"⚠️ Drive sync failed: {e}")
        return None
def load_candles_local(symbol: str, interval: str, min_days: int = 90) -> pd.DataFrame | None:
    """Chỉ đọc file local; KHÔNG gọi API. Trả về df tối thiểu ~min_days (nếu đủ)."""
    try:
        p = _cache_file(symbol, interval)
        if not os.path.exists(p):
            return None
        df = pd.read_parquet(p)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
        # Cắt 90 ngày gần nhất (dưới dạng thời gian)
        cutoff = datetime.now(timezone.utc) - timedelta(days=min_days)
        df = df[df["datetime"] >= cutoff].reset_index(drop=True)
        # ép kiểu số
        for c in ["open","high","low","close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["open","high","low","close"])
        return df if len(df) > 120 else None
    except Exception as e:
        logging.warning(f"[CACHE] load failed {symbol}-{interval}: {e}")
        return None

# Patch fetch_candles để LUÔN lưu cache khi đã gọi API
_ORIG_fetch_candles = fetch_candles
def fetch_candles(symbol, interval, retries=3, use_cache=True):
    df = _ORIG_fetch_candles(symbol, interval, retries=retries, use_cache=use_cache)
    # Mỗi lần có df từ API -> lưu xuống đĩa (để backtest sau này không cần API)
    try:
        if df is not None and len(df) > 0:
            save_candles_to_disk(symbol, interval, df)
    except Exception:
        pass
    return df
# ================= END OFFLINE CANDLE CACHE =================
# ========================= BACKTEST 90D OFFLINE ============================
RUN_BACKTEST_OFFLINE = os.getenv("RUN_BACKTEST_OFFLINE", "0") == "1"
BACKTEST_CSV_PATH    = os.getenv("BACKTEST_CSV_PATH", "/tmp/backtest_90d.csv")

def _first_hit_outcome(side: str, entry: float, sl: float, tp: float, df_future: pd.DataFrame):
    if df_future is None or len(df_future) == 0:
        return "TIMEOUT", 0
    max_ahead = min(40, len(df_future))
    for i in range(max_ahead):
        hi = float(df_future["high"].iloc[i])
        lo = float(df_future["low" ].iloc[i])
        if side == "LONG":
            if lo <= sl: return "SL", i+1
            if hi >= tp: return "TP", i+1
        else:
            if hi >= sl: return "SL", i+1
            if lo <= tp: return "TP", i+1
    return "TIMEOUT", max_ahead

def _bt_sideway_block_offline(df2h: pd.DataFrame, name_or_sym: str) -> bool:
    if df2h is None or len(df2h) < 60:
        return True
    a2  = adx(df2h, 14)
    bw2 = bb_width(df2h, 20)
    if is_crypto(name_or_sym):
        BW_MIN = 0.020
    elif is_commodity(name_or_sym):
        BW_MIN = 0.015
    else:
        BW_MIN = 0.012
    return (np.isnan(a2) or a2 < 20) or (np.isnan(bw2) or bw2 < BW_MIN)

def backtest_90d_offline_for_symbol(name: str, symbol: str, main_tf: str = None):
    df = load_candles_local(symbol, "2h", min_days=95)
    # 🧩 Bắt đầu xử lý cache Google Drive
    local_file = download_from_drive(symbol, "2h")
    
    if local_file and os.path.exists(local_file):
        logging.info(f"✅ Dùng cache local từ Google Drive cho {symbol}")
        df = pd.read_parquet(local_file)
    else:
        logging.info(f"⚠️ Không có cache Google Drive, tải API cho {symbol}")
        df = load_candles_local(symbol, "2h", min_days=95)  # hoặc hàm tải nến gốc của chị
        save_candles_to_disk(symbol, "2h", df)
        upload_to_drive(_cache_file(symbol, "2h"))
    # 🧩 Kết thúc xử lý cache Google Drive
    logging.info(f"[BT-OFF] {symbol} cache: {len(df)} nến | {df['datetime'].min()} -> {df['datetime'].max()}")
    tf = main_tf or os.getenv("MAIN_TF", "2h")
    # CHỈ đọc local, tuyệt đối không gọi API
    df_main = load_candles_local(symbol, tf, min_days=95)
    df_2h   = df_main if tf == "2h" else load_candles_local(symbol, "2h", min_days=95)

    if df_main is None or len(df_main) < 150:
        return {"symbol": name, "trades": 0, "win": 0, "loss": 0, "timeout": 0, "winrate": 0.0, "expR": 0.0}

    rows = []; wins=losses=tout=0; total_R=0.0

    for i in range(80, len(df_main) - 2):
        hist = df_main.iloc[:i+1].copy()
        if len(hist) < 65:
            continue

        bias = strong_trend(hist)
        regime = "TREND" if bias in ("LONG","SHORT") else "RANGE"
        confidence = 1.0
        
        # TÍNH KELTNER TRƯỚC
        try:
            km, kup, kdn = keltner_mid(hist, 20, atr_mult=1.0)
        except Exception:
            km = kup = kdn = np.nan
        
        # RANGE: gán side bằng biên kênh để có SL/TP
        if regime == "RANGE":
            px = float(hist["close"].iloc[-1])
            if not np.isnan(kup) and px >= kup:
                bias = "SHORT"
            elif not np.isnan(kdn) and px <= kdn:
                bias = "LONG"
            else:
                # chưa chạm biên -> bỏ qua nến này
                continue
        
        # ✅ BÂY GIỜ mới kiểm tra side hợp lệ
        if bias not in ("LONG","SHORT"):
            continue
        
        entry = float(hist["close"].iloc[-1])
        atrv = atr(hist, 14)
        swing_hi, swing_lo = swing_levels(hist, 20)
        sl, tp = smart_sl_tp(entry, atrv, swing_hi, swing_lo, kup, kdn, bias, is_fx(symbol) or is_fx(name))
        if sl is None or tp is None or entry is None:
            continue
        R = abs(entry - sl)
        if R <= 0: 
            continue
        rr = abs(tp - entry) / R
        if rr < 1.2:  # nới lỏng để có đủ trade
            continue
        
        fut = df_main.iloc[i+1:]
        if len(fut) == 0:
            break
        fill = float(fut["open"].iloc[0])
        outcome, bars = _first_hit_outcome(bias, entry, sl, tp, fut)
        outR = 0.0
        if outcome == "TP":
            wins += 1; outR = rr
        elif outcome == "SL":
            losses += 1; outR = -1.0
        else:
            tout += 1; outR = 0.0
        total_R += outR
        #regime = "TREND" if str(bias).upper() in ("LONG", "SHORT") else "RANGE"
        rows.append({
            "symbol": name, "tf": tf,
            "signal_time": hist["datetime"].iloc[-1].isoformat(),
            "side": bias, "entry": entry, "fill": fill, "sl": sl, "tp": tp,
            "regime": regime, "confidence": confidence,
            "R": round(outR,3), "bars_to_outcome": bars, "outcome": outcome
        })

    # ghi CSV (append)
    try:
        need_hdr = not os.path.exists(BACKTEST_CSV_PATH)
        with open(BACKTEST_CSV_PATH, "a", encoding="utf-8") as f:
            if need_hdr:
                f.write("symbol,tf,signal_time,side,entry,fill,sl,tp,regime,confidence,R,bars_to_outcome,outcome\n")
            for r in rows:
                f.write("{symbol},{tf},{signal_time},{side},{entry},{fill},{sl},{tp},{regime},{confidence},{R},{bars_to_outcome},{outcome}\n".format(**r))
    except Exception as e:
        logging.warning(f"[BT-OFF] CSV write failed: {e}")
    def _stats(subrows):
        trades = len(subrows)
        wins   = sum(1 for r in subrows if r["outcome"] == "Tp" or r["outcome"] == "TP")
        losses = sum(1 for r in subrows if r["outcome"] == "Sl" or r["outcome"] == "SL")
        tout   = sum(1 for r in subrows if r["outcome"] not in ("TP","Sl","SL","Tp"))
        winrate = (wins / max(1, wins + losses)) * 100.0 if (wins + losses) > 0 else 0.0
        expR = sum((r["R"] if r["outcome"] in ("TP","Tp") else (-1.0 if r["outcome"] in ("SL","Sl") else 0.0)) for r in subrows) / max(1, trades)
        return {"trades": trades, "win": wins, "loss": losses, "timeout": tout, "winrate": round(winrate,1), "expR": round(expR,3)}  
    trend_rows = [r for r in rows if r.get("regime") == "TREND"]
    range_rows = [r for r in rows if r.get("regime") == "RANGE"]
    
    trend_stat = _stats(trend_rows)
    range_stat = _stats(range_rows)
    
    trades  = wins + losses + tout
    winrate = (wins / max(1, wins + losses)) * 100.0 if (wins + losses) > 0 else 0.0
    expR    = total_R / max(1, trades)
    
    res = {
        "symbol": name,
        "trades": trades, "win": wins, "loss": losses, "timeout": tout,
        "winrate": round(winrate, 1), "expR": round(expR, 3),
        "trend": trend_stat,         # ✅ có số liệu TREND
        "range": range_stat,         # ✅ có số liệu RANGE
    }
    
    return res

def backtest_90d_offline():
    MAIN_TF = os.getenv("MAIN_TF", "2h")
    summary = []
    for name, sym in symbols.items():
        try:
            res = backtest_90d_offline_for_symbol(name, sym, MAIN_TF)
        except Exception as e:
            logging.error(f"[BT-OFF] {name} failed: {e}")
            res = {"symbol": name, "trades": 0, "win": 0, "loss": 0, "timeout": 0, "winrate": 0.0, "expR": 0.0}
        summary.append(res)

    lines = [f"📊 Backtest 90d (OFFLINE, {MAIN_TF}, no API):"]
    for r in summary:
        lines.append(
            f"- {r['symbol']}: {r['trades']} trades | W:{r['win']} L:{r['loss']} T:{r['timeout']} | "
            f"Winrate {r['winrate']}% | ExpR {r['expR']}"
        )
        t = r.get("trend")
        if t:
            lines.append(f"   • TREND : {t['trades']} | W:{t['win']} L:{t['loss']} T:{t['timeout']} | Win {t['winrate']}% | ExpR {t['expR']}")
        g = r.get("range")
        if g:
            lines.append(f"   • RANGE : {g['trades']} | W:{g['win']} L:{g['loss']} T:{g['timeout']} | Win {g['winrate']}% | ExpR {g['expR']}")
    msg = "\n".join(lines)
    logging.info(msg)
    try:
        if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
            send_telegram(msg)
    except Exception:
        pass
# ======================= END BACKTEST 90D OFFLINE ==========================

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
    import traceback
    try:    
        # luôn kiểm tra/làm mới cache 1D (chỉ fetch khi tới giờ/đúng ngày)
        daily_cache = maybe_refresh_daily_cache()
        RUN_CACHE.clear()  # đảm bảo nến mới được tải
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
        
        #1 Gửi bản thông minh   
        #had_entry   = any("Entry" in l for l in lines)
        #had_blocked = any(l.startswith("⛔ ") for l in lines)
        #if had_entry or had_blocked:
            #send_telegram("\n".join(lines))
        #else:
            # vẫn gửi bản tóm tắt tối thiểu để biết hệ thống đang chạy
            #send_telegram("\n".join(lines[:10]))
        #2 Nếu tất cả đều N/A/SIDEWAY & không có Entry -> vẫn gửi để biết trạng thái; nếu muốn có thể chặn tại đây
        #msg = "\n".join(lines)
        #send_telegram(msg)
        
        #3 Chỉ gửi nếu có ít nhất 1 symbol có Entry thật (không phải N/A)
        valid_msg = any(
        ("Entry" in l and not any(x in l for x in ["N/A", "None", "NaN"]))
        for l in lines
    )
        if valid_msg:
            msg = "\n".join(lines)
            send_telegram(msg)
        else:
            print("🚫 Tất cả đều N/A, không gửi Telegram")
        # === Chạy backtest offline lúc 12:05 UTC nếu bật =5==
        try:
            if RUN_BACKTEST_OFFLINE:
                now_utc = datetime.now(timezone.utc)
                if now_utc.hour == 5 and 4 <= now_utc.minute <= 15:
                    logging.info("[BT-OFF] Running daily offline backtest (no API)...")
                    try:
                        backtest_90d_offline()
                    except Exception as e:
                        logging.error(f"[BT-OFF] Error running backtest: {e}")   
        except Exception:
            logging.error(traceback.format_exc())
    except Exception:
            logging.error(traceback.format_exc())
if __name__ == "__main__":
    main()
