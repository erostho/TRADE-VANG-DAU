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
API_KEY = os.getenv("TWELVE_DATA_KEY", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

TD_BASE_URL = "https://api.twelvedata.com/time_series"
TIMEZ = os.getenv("TZ", "Asia/Ho_Chi_Minh")
RPM = int(os.getenv("RPM", 7))  # requests per minute (throttle)

# Symbols để hiển thị -> mã TwelveData (mặc định)
SYMBOLS_TD = {
    "Bitcoin": "BTC/USD",
    "Ethereum": "ETH/USD",
    "XAU/USD (Gold)": "XAU/USD",
    "WTI Oil": "CL",
    "USD/JPY": "USD/JPY",
    # Silver (Bạc) trên TD (dùng làm fallback)
    "XAG/USD (Silver)": "XAG/USD",
}

# Riêng Bạc dùng Yahoo Finance làm nguồn chính (ngoài TV/TD)
# Yahoo ticker cho Bạc spot: XAGUSD=X
SYMBOLS_YF = {
    "XAG/USD (Silver)": "XAGUSD=X",
}

# Nhóm khung nến (GIỮ NGUYÊN)
INTERVAL_GROUPS = {
    "15m-30m": ["15min", "30min"],
    "1H-2H": ["1h", "2h"],
    "4H": ["4h"]
}

# ================= HELPERS =================
def _finish_df(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa df OHLC (datetime tăng dần, cột float)."""
    if df is None or df.empty:
        return None
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(TIMEZ)
    df = df.sort_values("datetime").reset_index(drop=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df if not df.empty else None

def fetch_candles_td(symbol, interval, retries=3):
    """Lấy nến từ Twelve Data (mặc định cho các symbol khác)."""
    url = f"{TD_BASE_URL}?symbol={symbol}&interval={interval}&apikey={API_KEY}&outputsize=200"
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=12)
            if r.status_code == 429:
                logging.warning(f"TD rate limit {symbol}-{interval}, sleep 65s...")
                time.sleep(65)
                continue
            data = r.json()
            if "values" not in data:
                logging.warning(f"TD no data {symbol}-{interval}: {data}")
                return None
            rows = data["values"]
            df = pd.DataFrame(rows)[["datetime", "open", "high", "low", "close"]]
            return _finish_df(df)
        except Exception as e:
            logging.error(f"TD fetch error {symbol}-{interval}: {e}")
            time.sleep(3)
    return None

# ---------- Yahoo Finance (dành cho Bạc) ----------
# Map interval của mình -> Yahoo interval và range tối thiểu
YF_INTERVAL_MAP = {
    "15min": ("15m", "5d"),
    "30min": ("30m", "5d"),
    "1h":   ("60m", "30d"),
    # Yahoo không có 2h/4h trực tiếp -> dùng 60m rồi resample
    "2h":   ("60m", "60d"),
    "4h":   ("60m", "60d"),
}

def fetch_candles_yf(ticker, interval):
    """
    Lấy dữ liệu từ Yahoo Finance (không cần API key).
    Trả về df OHLC; nếu interval là 2h/4h thì resample từ 60m.
    """
    if interval not in YF_INTERVAL_MAP:
        return None
    yf_iv, yf_range = YF_INTERVAL_MAP[interval]
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval={yf_iv}&range={yf_range}"
    try:
        r = requests.get(url, timeout=12)
        j = r.json()
        result = j.get("chart", {}).get("result", [])
        if not result:
            logging.warning(f"YF no result {ticker}-{interval}: {j}")
            return None
        res = result[0]
        ts = res.get("timestamp")
        quote = res.get("indicators", {}).get("quote", [{}])[0]
        if not ts or not quote:
            logging.warning(f"YF empty series {ticker}-{interval}")
            return None
        df = pd.DataFrame({
            "datetime": pd.to_datetime(ts, unit="s", utc=True),
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
        })
        df = _finish_df(df)
        if df is None:
            return None

        # Resample cho 2h/4h nếu cần
        if interval in ("2h", "4h"):
            rule = "2H" if interval == "2h" else "4H"
            df = (df.set_index("datetime")
                    .resample(rule, label="right", closed="right")
                    .agg({"open":"first", "high":"max", "low":"min", "close":"last"})
                    .dropna()
                    .reset_index())
        return df
    except Exception as e:
        logging.error(f"YF fetch error {ticker}-{interval}: {e}")
        return None

def fetch_candles(symbol_display, interval):
    """
    Router: nếu là Bạc -> dùng Yahoo trước, rồi fallback TwelveData.
    Ngược lại: dùng TwelveData.
    """
    if symbol_display in SYMBOLS_YF:
        # nguồn chính: Yahoo
        df = fetch_candles_yf(SYMBOLS_YF[symbol_display], interval)
        if df is not None and len(df) > 0:
            return df
        # fallback TD
        logging.info(f"Fallback TD for {symbol_display}-{interval}")
        return fetch_candles_td(SYMBOLS_TD[symbol_display], interval)
    else:
        return fetch_candles_td(SYMBOLS_TD[symbol_display], interval)

def atr(df, period=14):
    high = df["high"].values
    low = df["low"].values
    close = df["close"].shift(1).fillna(df["close"])
    tr = np.maximum(high - low, np.maximum(abs(high - close), abs(low - close)))
    atr_vals = pd.Series(tr).rolling(period).mean()
    return float(atr_vals.iloc[-1])

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

def analyze_symbol(name, symbol_display):
    results = {}
    for group, intervals in INTERVAL_GROUPS.items():
        trends = []
        for iv in intervals:
            df = fetch_candles(symbol_display, iv)
            trend = get_trend(df)
            trends.append(f"{iv}:{trend}")
            time.sleep(60.0 / RPM)  # GIỮ NHƯ CŨ để không vượt quota
        if len(intervals) == 1:
            results[group] = trends[0].split(":")[1]
        else:
            uniq = set(t.split(":")[1] for t in trends)
            if len(uniq) == 1:
                results[group] = uniq.pop()
            else:
                results[group] = "Mixed (" + ", ".join(trends) + ")"

    # Entry/SL/TP từ 1H (GIỮ NGUYÊN LOGIC)
    df1h = fetch_candles(symbol_display, "1h")
    plan = "SIDEWAY"
    entry = sl = tp = atrval = None
    if df1h is not None and len(df1h) > 20:
        bias = get_trend(df1h)
        entry = float(df1h["close"].iloc[-1])
        atrval = atr(df1h, 14)
        if bias == "LONG":
            plan = "LONG"
            sl = entry - 1.5 * atrval
            tp = entry + 1.5 * atrval
        elif bias == "SHORT":
            plan = "SHORT"
            sl = entry + 1.5 * atrval
            tp = entry - 1.5 * atrval
    return results, plan, entry, sl, tp, atrval

def send_telegram(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram not configured")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
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

    for name, _td_code in SYMBOLS_TD.items():
        results, plan, entry, sl, tp, atrval = analyze_symbol(name, name)
        lines.append(f"==={name}===")
        for group, trend in results.items():
            lines.append(f"{group}: {trend}")
        if entry and sl and tp:
            lines.append(f"Entry {entry:.2f} | SL {sl:.2f} | TP {tp:.2f}")
        lines.append("")

    msg = "\n".join(lines)
    send_telegram(msg)

if __name__ == "__main__":
    main()
