import os
import time
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ================= LOGGING =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)

# ================= CONFIG =================
API_KEY_TD = os.getenv("TWELVE_DATA_KEY", "")
API_KEY_AV = os.getenv("ALPHA_VANTAGE_KEY", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

BASE_URL_TD = "https://api.twelvedata.com/time_series"
BASE_URL_YF = "https://query1.finance.yahoo.com/v8/finance/chart"
TIMEZ = os.getenv("TZ", "Asia/Ho_Chi_Minh")

RPM = int(os.getenv("RPM", 7))

# ================= SYMBOLS =================
symbols = {
    "Bitcoin": "BTC/USD",
    "Silver (XAG)": "XAG/USD",    # ✅ thay ETH thành BẠC
    "XAU/USD (Gold)": "XAU/USD",
    "WTI Oil": "CL",
    "USD/JPY": "USD/JPY",
}

interval_groups = {
    "15m-30m": ["15min", "30min"],
    "1H-2H": ["1h", "2h"],
    "4H": ["4h"]
}

# ================= HELPERS =================
def fetch_from_twelvedata(symbol, interval):
    url = f"{BASE_URL_TD}?symbol={symbol}&interval={interval}&apikey={API_KEY_TD}&outputsize=100"
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        return None
    data = r.json()
    if "values" not in data:
        return None
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col])
    return df

def fetch_from_yahoo(symbol, interval):
    mapping = {"15min":"15m","30min":"30m","1h":"60m","2h":"120m","4h":"240m"}
    if interval not in mapping:
        return None
    ticker = symbol.replace("/USD","USD=X").replace("/","-")
    url = f"{BASE_URL_YF}/{ticker}?interval={mapping[interval]}&range=5d"
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        return None
    data = r.json().get("chart", {}).get("result")
    if not data:
        return None
    ts = data[0]["timestamp"]
    quotes = data[0]["indicators"]["quote"][0]
    df = pd.DataFrame({
        "datetime": pd.to_datetime(ts, unit="s"),
        "open": quotes["open"],
        "high": quotes["high"],
        "low": quotes["low"],
        "close": quotes["close"]
    })
    return df.dropna()

def fetch_from_alphavantage(symbol, interval):
    if not API_KEY_AV:
        return None
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=60min&apikey={API_KEY_AV}"
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        return None
    data = r.json()
    key = list(data.keys())[-1]
    values = data[key]
    rows = []
    for t,v in values.items():
        rows.append({
            "datetime": pd.to_datetime(t),
            "open": float(v["1. open"]),
            "high": float(v["2. high"]),
            "low": float(v["3. low"]),
            "close": float(v["4. close"])
        })
    return pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)

def fetch_candles(symbol, interval):
    df = fetch_from_twelvedata(symbol, interval)
    if df is not None and len(df) > 0:
        return df
    logging.warning(f"TwelveData NA for {symbol}-{interval}, trying Yahoo")
    df = fetch_from_yahoo(symbol, interval)
    if df is not None and len(df) > 0:
        return df
    logging.warning(f"Yahoo NA for {symbol}-{interval}, trying AlphaVantage")
    df = fetch_from_alphavantage(symbol, interval)
    return df

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
    has_data = False
    for group, intervals in interval_groups.items():
        trends = []
        for iv in intervals:
            df = fetch_candles(symbol, iv)
            trend = get_trend(df)
            trends.append(f"{iv}:{trend}")
            if trend != "N/A":
                has_data = True
            time.sleep(60.0/RPM)
        if len(intervals) == 1:
            results[group] = trends[0].split(":")[1]
        else:
            uniq = set([t.split(":")[1] for t in trends])
            if len(uniq) == 1:
                results[group] = uniq.pop()
            else:
                results[group] = "Mixed (" + ", ".join(trends) + ")"
    df1h = fetch_candles(symbol, "1h")
    plan = "SIDEWAY"
    entry = sl = tp = atrval = None
    if df1h is not None and len(df1h) > 20:
        bias = get_trend(df1h)
        if bias != "N/A":
            has_data = True
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
def main():
    lines = []
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    lines.append("💵 TRADE GOODS")
    lines.append(f"⏱ {now}\n")

    any_symbol_has_data = False
    for name, sym in symbols.items():
        results, plan, entry, sl, tp, atrval, has_data = analyze_symbol(name, sym)
        if has_data:
            any_symbol_has_data = True
        lines.append(f"==={name}===")
        for group, trend in results.items():
            lines.append(f"{group}: {trend}")
        if entry and sl and tp:
            lines.append(f"Entry {entry:.2f} | SL {sl:.2f} | TP {tp:.2f}")
        lines.append("")
    if not any_symbol_has_data:
        logging.info("Skip Telegram: all symbols/timeframes are N/A")
        return
    msg = "\n".join(lines)
    send_telegram(msg)

if __name__ == "__main__":
    main()
