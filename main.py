import os
import time
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

# ================= LOGGING =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)

# ================= CONFIG =================
API_KEY = os.getenv("TWELVE_DATA_KEY", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

BASE_URL = "https://api.twelvedata.com/time_series"
TIMEZ = os.getenv("TZ", "Asia/Ho_Chi_Minh")

# Request per minute throttle
RPM = int(os.getenv("RPM", 6))

symbols = {
    "Bitcoin": "BTC/USD",
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

# ================= HELPERS =================
def fetch_candles(symbol, interval, retries=3):
    url = f"{BASE_URL}?symbol={symbol}&interval={interval}&apikey={API_KEY}&outputsize=100"
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
    close = df["close"].shift(1).fillna(df["close"])
    tr = np.maximum(high - low, np.maximum(abs(high - close), abs(low - close)))
    atr_vals = pd.Series(tr).rolling(period).mean()
    return atr_vals.iloc[-1]
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
    return dx.rolling(n).mean().iloc[-1]

def strong_trend(df):
    """Lọc nhiễu: EMA(20/50) + vị trí giá + ADX + slope."""
    if df is None or len(df) < 60:
        return "N/A"
    e20 = ema(df['close'], 20)
    e50 = ema(df['close'], 50)
    last = df['close'].iloc[-1]
    adx_val = adx(df, 14)
    # slope % của EMA20 trên 5 nến gần nhất
    if len(e20) < 6 or np.isnan(adx_val):
        return "N/A"
    slope = (e20.iloc[-1] - e20.iloc[-6]) / max(1e-9, e20.iloc[-6]) * 100

    if (e20.iloc[-1] > e50.iloc[-1]) and (last > e20.iloc[-1]) and (adx_val >= 20) and (slope > 0.02):
        return "LONG"
    if (e20.iloc[-1] < e50.iloc[-1]) and (last < e20.iloc[-1]) and (adx_val >= 20) and (slope < -0.02):
        return "SHORT"
    return "SIDEWAY"

def swing_levels(df, lookback=20):
    """Lấy swing gần nhất (đỉnh/đáy trước nến hiện tại)."""
    if df is None or len(df) < lookback + 2:
        return (np.nan, np.nan)
    swing_hi = df['high'].rolling(lookback).max().iloc[-2]
    swing_lo = df['low' ].rolling(lookback).min().iloc[-2]
    return swing_hi, swing_lo

def confluence_score(results_dict):
    """Điểm đồng thuận 0–3: 15–30, 1H–2H, 4H."""
    g15_30 = results_dict.get("15m-30m", "N/A")
    g1_2   = results_dict.get("1H-2H",   "N/A")
    g4     = results_dict.get("4H",      "N/A")

    def norm(x):  # lấy LONG/SHORT nếu Mixed
        if x.startswith("Mixed"):
            return "MIX"
        return x

    a, b, c = norm(g15_30), norm(g1_2), norm(g4)
    score = 0
    if a in ("LONG","SHORT") and b == a: score += 1
    if b in ("LONG","SHORT") and c == b: score += 1
    if a in ("LONG","SHORT") and c == a: score += 1
    return score
def get_trend(df):
    if df is None or len(df) < 20:
        return "N/A"
    short = df["close"].rolling(10).mean().iloc[-1]
    long = df["close"].rolling(30).mean().iloc[-1]
    last_close = df["close"].iloc[-1]
    if short > long and last_close > short:
        return "LONG"
    elif short < long and last_close < short:
        return "SHORT"
    else:
        return "SIDEWAY"


def recent_1h_trend_15_30(symbol):
    """Lấy xu hướng 1 giờ qua từ nến 15m và 30m"""
    def bias(df, need):
        if df is None or len(df) < need:
            return "N/A"
        start = df["close"].iloc[-need]
        end = df["close"].iloc[-1]
        if end > start:
            return "LONG"
        elif end < start:
            return "SHORT"
        return "SIDEWAY"

    df15 = fetch_candles(symbol, "15min")
    b15 = bias(df15, 4)
    time.sleep(60.0/RPM)

    df30 = fetch_candles(symbol, "30min")
    b30 = bias(df30, 2)

    uniq = {b15, b30} - {"N/A"}
    if not uniq:
        summary = "N/A"
    elif len(uniq) == 1:
        summary = uniq.pop()
    else:
        summary = "Mixed"
    return f"{summary} (15m:{b15}, 30m:{b30})"

def analyze_symbol(name, symbol):
    results = {}
    has_data = False

    # 1) Tính trend cho từng khung
    for group, intervals in interval_groups.items():
        trends = []
        for iv in intervals:
            df = fetch_candles(symbol, iv)
            trend = strong_trend(df)
            trends.append(f"{iv}:{trend}")
            time.sleep(60.0 / RPM)  # throttle
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

    # 2) Entry/SL/TP từ 1H (bám swing + ATR)
    df1h = fetch_candles(symbol, "1h")
    plan = "SIDEWAY"
    entry = sl = tp = atrval = None

    if df1h is not None and len(df1h) > 40:
        bias = strong_trend(df1h)
        if bias != "N/A":
            has_data = True  # 1H có dữ liệu hợp lệ

        entry  = df1h["close"].iloc[-1]
        atrval = atr(df1h, 14)
        swing_hi, swing_lo = swing_levels(df1h, 20)

        # hệ số ATR khác nhau
        is_fx = name in ("EUR/USD", "USD/JPY")
        base_mult = 2.5 if is_fx else 1.5
        buf = 0.5 * atrval  # đệm để khỏi chạm đúng swing

        if bias == "LONG":
            plan = "LONG"
            sl_candidates = [
                entry - base_mult * atrval,
                swing_lo - buf if not np.isnan(swing_lo) else entry - base_mult * atrval,
            ]
            sl = min(sl_candidates)  # lấy xa hơn để an toàn
            r  = entry - sl
            tp = entry + 1.4 * r     # TP ≈ 1.4R
        elif bias == "SHORT":
            plan = "SHORT"
            sl_candidates = [
                entry + base_mult * atrval,
                swing_hi + buf if not np.isnan(swing_hi) else entry + base_mult * atrval,
            ]
            sl = max(sl_candidates)
            r  = sl - entry
            tp = entry - 1.4 * r

    # 3) (Tùy chọn) Nếu điểm đồng thuận quá thấp thì bỏ Entry/SL/TP
    try:
        score = confluence_score(results)  # 0..3
        if score < 2:  # chưa đủ đồng thuận -> không khuyến nghị lệnh
            entry = sl = tp = None
    except Exception:
        pass

    return results, plan, entry, sl, tp, atrval, has_data

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

# ================= MAIN =================
#def main():
    #lines = []
    #now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    #lines.append("💵 TRADE GOODS")
    #lines.append(f"⏱ {now}\n")
def main():
    lines = []
    now = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).strftime("%Y-%m-%d %H:%M:%S")
    lines.append("💵 TRADE GOODS")
    lines.append(f"⏱ {now}\n")
    any_symbol_has_data = False  # <--- tổng hợp cờ

    for name, sym in symbols.items():
        results, plan, entry, sl, tp, atrval, has_data = analyze_symbol(name, sym)
        if has_data:
            any_symbol_has_data = True

        lines.append(f"==={name}===")
        for group, trend in results.items():
            lines.append(f"{group}: {trend}")
        # <= THÊM DÒNG NÀY
        lines.append(f"15m-30m (1h qua): {recent_1h_trend_15_30(sym)}")
        if entry and sl and tp:
            lines.append(f"Entry {entry:.2f} | SL {sl:.2f} | TP {tp:.2f}")
        lines.append("")
        time.sleep(10)

    # Nếu TẤT CẢ đều N/A -> KHÔNG gửi
    if not any_symbol_has_data:
        logging.info("Skip Telegram: all symbols/timeframes are N/A")
        return

    msg = "\n".join(lines)
    send_telegram(msg)

if __name__ == "__main__":
    main()
