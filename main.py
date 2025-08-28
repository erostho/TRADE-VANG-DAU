
import os
import time
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

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
    "Ethereum": "ETH/USD",
    "XAU/USD (Gold)": "XAU/USD",
    "USD/JPY": "USD/JPY",
    # "WTI Oil": "WTI/USD"   # free plan thường không có -> comment
}

interval_groups = {
    "30m-1H": ["30min", "1h"],
    "2H-4H": ["2h", "4h"],
    "1D": ["1day"]
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

def analyze_symbol(name, symbol):
    results = {}
    for group, intervals in interval_groups.items():
        trends = []
        for iv in intervals:
            df = fetch_candles(symbol, iv)
            trend = get_trend(df)
            trends.append(f"{iv}:{trend}")
            time.sleep(60.0/RPM)
        # Nếu 1 interval -> trực tiếp
        if len(intervals) == 1:
            results[group] = trends[0].split(":")[1]
        else:
            uniq = set([t.split(":")[1] for t in trends])
            if len(uniq) == 1:
                results[group] = uniq.pop()
            else:
                results[group] = "Mixed (" + ", ".join(trends) + ")"
    # Entry/SL/TP từ 1H
    df1h = fetch_candles(symbol, "1h")
    plan = "SIDEWAY"
    entry = sl = tp = atrval = None
    if df1h is not None and len(df1h) > 20:
        bias = get_trend(df1h)
        entry = df1h["close"].iloc[-1]
        atrval = atr(df1h, 14)
        if bias == "LONG":
            plan = "LONG"
            sl = entry - 1.5*atrval
            tp = entry + 1.5*atrval
        elif bias == "SHORT":
            plan = "SHORT"
            sl = entry + 1.5*atrval
            tp = entry - 1.5*atrval
    return results, plan, entry, sl, tp, atrval

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
def main():
    lines = []
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    lines.append("💵 TRADE GOODS")
    lines.append(f"⏱ {now}\\n")
    for name, sym in symbols.items():
        results, plan, entry, sl, tp, atrval = analyze_symbol(name, sym)
        lines.append(f"==={name}===")
        for group, trend in results.items():
            lines.append(f"{group}: {trend}")
        if entry and sl and tp:
            lines.append(f"1H plan: {plan} | Entry {entry:.2f} | SL {sl:.2f} | TP {tp:.2f} (ATR14 {atrval:.3f})")
        lines.append("")
    msg = "\\n".join(lines)
    send_telegram(msg)

if __name__ == "__main__":
    main()
