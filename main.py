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
RPM = int(os.getenv("RPM", 7))

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
    for group, intervals in interval_groups.items():
        trends = []
        for iv in intervals:
            df = fetch_candles(symbol, iv)
            trend = get_trend(df)
            trends.append(f"{iv}:{trend}")
            time.sleep(60.0 / RPM)

        if len(intervals) == 1:
            results[group] = trends[0].split(":")[1]
        else:
            uniq = set(t.split(":")[1] for t in trends)
            if len(uniq) == 1:
                results[group] = uniq.pop()
            else:
                results[group] = "Mixed (" + ", ".join(trends) + ")"

    # Entry/SL/TP từ 1H
    df1h = fetch_candles(symbol, "1h")
    plan = "SIDEWAY"
    entry = sl = tp = atrval = None
    has_data = False

    if df1h is not None and len(df1h) > 20:
        bias = get_trend(df1h)
        # coi 1H có dữ liệu hợp lệ
        if bias != "N/A":
            has_data = True

        entry = df1h["close"].iloc[-1]
        atrval = atr(df1h, 14)

        # Hệ số ATR: Forex 2.5, còn lại 1.5
        forex_set = {"EUR/USD", "USD/JPY"}  # giữ đúng 2 cặp bạn đang dùng
        if (symbol in forex_set) or any(k in name for k in forex_set):
            atr_mult = 2.5
        else:
            atr_mult = 1.5

        if bias == "LONG":
            plan = "LONG"
            sl = entry - atr_mult * atrval
            tp = entry + atr_mult * atrval
        elif bias == "SHORT":
            plan = "SHORT"
            sl = entry + atr_mult * atrval
            tp = entry - atr_mult * atrval

    # TRẢ VỀ 7 GIÁ TRỊ như main đang unpack
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
        if not has_data:
            continue
    
        lines.append(f"==={name}===")
    
        # in các nhóm khung nến hiện tại
        group_lines = []
        for group, trend in results.items():
            group_lines.append(f"{group}: {trend}")
        lines.extend(group_lines)
    
        # CHỈ 1 DÒNG "1h qua" CHO MỖI SYMBOL
        lines.append(f"15m-30m (1h qua): {recent_1h_trend_15_30(sym)}")
    
        if entry is not None and sl is not None and tp is not None:
            lines.append(f"Entry {entry:.2f} | SL {sl:.2f} | TP {tp:.2f}")
        lines.append("")

    # Nếu TẤT CẢ đều N/A -> KHÔNG gửi
    if not any_symbol_has_data:
        logging.info("Skip Telegram: all symbols/timeframes are N/A")
        return

    msg = "\n".join(lines)
    send_telegram(msg)

if __name__ == "__main__":
    main()
