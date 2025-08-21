# -*- coding: utf-8 -*-
"""
TwelveData-only Scanner (batch + throttle)
Frames: 30m, 1H, 2H, 4H, 1D
Assets: XAU/USD, CL, BTC/USD, EUR/USD, USD/JPY
Signals: LONG / SHORT / SIDEWAY
Telegram format:
===SYMBOL===
30m-1H: ...
2H-4H: ...
1D: ...

ENV REQUIRED:
  TWELVE_DATA_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
Optional:
  LOG_LEVEL=INFO|DEBUG
  INCLUDE_ERRORS_IN_TELEGRAM=1
  TD_CREDITS_PER_MIN=8  (giới hạn/phút tuỳ plan)
  TD_SLEEP_BETWEEN_BATCH=15  (giây)
"""

import os, time, logging, requests
import pandas as pd
import numpy as np

# ----------- ENV / LOG -----------
TD_KEY   = os.getenv("TWELVE_DATA_KEY")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
INC_ERR_TG = os.getenv("INCLUDE_ERRORS_IN_TELEGRAM", "0") == "1"
TD_CREDITS_PER_MIN = int(os.getenv("TD_CREDITS_PER_MIN", "8"))
SLEEP_BETWEEN_BATCH = int(os.getenv("TD_SLEEP_BETWEEN_BATCH", "15"))

if not TD_KEY:
    raise SystemExit("❌ Missing TWELVE_DATA_KEY env.")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ----------- CONFIG -----------
ASSETS = {
    "XAU/USD (Gold)": "XAU/USD",
    "WTI Oil": "CL",
    "Bitcoin": "BTC/USD",
    "EUR/USD": "EUR/USD",
    "USD/JPY": "USD/JPY",
}
INTERVALS = {"30m": "30min", "1H": "1h", "2H": "2h", "4H": "4h", "1D": "1day"}
OUTPUTSIZE = 2000  # đủ dài cho indicator, không quá tốn credit
MIN_BARS_INTRADAY = 60
MIN_BARS_DAILY = 120
ADX_TREND = 20

# ----------- Indicators -----------
TD_SLEEP_BETWEEN_CALL = float(os.getenv("TD_SLEEP_BETWEEN_CALL", "8"))  # giây

def td_single_time_series(symbol: str, interval: str, outputsize=2000, retries=2):
    """interval: '30min','1h','2h','4h','1day'"""
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": symbol, "interval": interval, "outputsize": outputsize, "apikey": TD_KEY}
    last_err = None
    for attempt in range(retries + 1):
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 429:
            wait = 65 if attempt < retries else 0
            logging.warning(f"TD 429 for {symbol} {interval}, backoff {wait}s (attempt {attempt+1}/{retries+1})")
            if wait: time.sleep(wait); continue
        try:
            r.raise_for_status()
            js = r.json()
            if "values" not in js or not js["values"]:
                raise RuntimeError(f"empty payload: {js}")
            df = pd.DataFrame(js["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").set_index("datetime")
            df = df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
            df[["Open","High","Low","Close"]] = df[["Open","High","Low","Close"]].astype(float)
            if "Volume" in df.columns:
                df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
            logging.info(f"TD OK {symbol} {interval}: {len(df)} rows [{df.index[0]}..{df.index[-1]}]")
            # throttle mỗi call để không vượt credit/min
            time.sleep(TD_SLEEP_BETWEEN_CALL)
            return df
        except Exception as e:
            last_err = e
            logging.warning(f"TD fail {symbol} {interval}: {e}")
            time.sleep(TD_SLEEP_BETWEEN_CALL)
    raise RuntimeError(f"TD final fail {symbol} {interval}: {last_err}")
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi(close, n=14):
    d = close.diff()
    gain, loss = d.clip(lower=0), (-d).clip(lower=0)
    ag = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    al = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))
def macd(close, fast=12, slow=26, sig=9):
    m = ema(close, fast) - ema(close, slow); s = ema(m, sig)
    return m, s, m - s
def true_range(df):
    pc = df["Close"].shift(1)
    return pd.concat([(df["High"]-df["Low"]),
                      (df["High"]-pc).abs(),
                      (df["Low"]-pc).abs()], axis=1).max(axis=1)
def adx(df, n=14):
    up, dn = df["High"].diff(), -df["Low"].diff()
    plus  = up.where((up > dn) & (up > 0), 0.0)
    minus = dn.where((dn > up) & (dn > 0), 0.0)
    trn = true_range(df).rolling(n).sum()
    pdi = 100 * (plus.rolling(n).sum() / trn)
    mdi = 100 * (minus.rolling(n).sum() / trn)
    dx  = ((pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)) * 100
    return dx.rolling(n).mean()
def bb_width(close, n=20):
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    return ((ma + 2*sd) - (ma - 2*sd)) / ma
def decorate(df):
    out = df.copy()
    out["EMA20"]  = ema(out["Close"], 20)
    out["EMA50"]  = ema(out["Close"], 50)
    out["EMA200"] = ema(out["Close"], 200)
    out["RSI14"]  = rsi(out["Close"], 14)
    out["MACD"], out["MACD_SIG"], out["MACD_HIST"] = macd(out["Close"])
    out["ADX14"]  = adx(out, 14)
    out["BBWIDTH"] = bb_width(out["Close"], 20)
    return out
def decide(df):
    last = df.iloc[-1]
    trending = (last["ADX14"] > ADX_TREND) and \
               (last["BBWIDTH"] > df["BBWIDTH"].rolling(50).mean().iloc[-1])
    if (last["Close"] > last["EMA20"] > last["EMA50"] > last["EMA200"]
        and last["RSI14"] > 55 and last["MACD"] > last["MACD_SIG"] and trending):
        return "LONG"
    if (last["Close"] < last["EMA20"] < last["EMA50"] < last["EMA200"]
        and last["RSI14"] < 45 and last["MACD"] < last["MACD_SIG"] and trending):
        return "SHORT"
    return "SIDEWAY"
def fetch_all_frames(td_symbol: str):
    IV = {"30m":"30min", "1H":"1h", "2H":"2h", "4H":"4h", "1D":"1day"}
    out = {}
    targets = ["30m","1H","2H","4H","1D"]

    # fallback riêng cho dầu
    candidates = [td_symbol] if td_symbol != "CL" else ["CL", "WTI/USD"]

    for lbl in targets:
        got = None
        for sym in candidates:
            try:
                got = td_single_time_series(sym, IV[lbl])
                break
            except Exception as e:
                logging.warning(f"Try {sym} {IV[lbl]} fail: {e}")
        out[lbl] = got
    return out

def analyze_one(name: str, td_symbol: str) -> str:
    logging.info(f"=== Start {name} ({td_symbol}) ===")
    frames = fetch_all_frames(td_symbol)

    def sig(df, is_daily=False):
        need = MIN_BARS_DAILY if is_daily else MIN_BARS_INTRADAY
        if isinstance(df, pd.DataFrame) and len(df) >= need:
            return decide(decorate(df))
        return "N/A"

    s30 = sig(frames["30m"])
    s1h = sig(frames["1H"])
    s2h = sig(frames["2H"])
    s4h = sig(frames["4H"])
    s1d = sig(frames["1D"], is_daily=True)

    short = s30 if (s30 == s1h and s30 != "N/A") else f"Mixed (30m:{s30}, 1H:{s1h})"
    mid   = s2h if (s2h == s4h and s2h != "N/A") else f"Mixed (2H:{s2h}, 4H:{s4h})"
    msg = f"==={name}===\n30m-1H: {short}\n2H-4H: {mid}\n1D: {s1d}"
    logging.info(msg.replace("\n", " | "))
    return msg
# ----------- Telegram ----------
def send_tele(text):
    if not BOT_TOKEN or not CHAT_ID:
        logging.warning("Missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text},
            timeout=20
        )
    except Exception as e:
        logging.exception(f"Telegram error: {e}")

# ----------- MAIN FLOW -----------
def main():
    # symbols theo TD
    symbols = list(ASSETS.values())

    # Batch theo từng interval để giảm credit/phút
    frames_by_iv = {}
    for lbl, iv in INTERVALS.items():
        try:
            frames_by_iv[lbl] = fetch_all_frames(symbols, iv)
        except Exception as e:
            logging.exception(f"Batch {iv} failed")
            frames_by_iv[lbl] = {s: None for s in symbols}
        # Throttle để không vượt 8 credits/min (mỗi batch ~ len(symbols) credits)
        time.sleep(SLEEP_BETWEEN_BATCH)

    # Tạo báo cáo cho từng asset
    reports = []
    for disp, sym in ASSETS.items():
        notes = []
        def sig(df, is_daily=False):
            if isinstance(df, pd.DataFrame):
                need = MIN_BARS_DAILY if is_daily else MIN_BARS_INTRADAY
                if len(df) >= need:
                    return decide(decorate(df))
            return "N/A"

        s30 = sig(frames_by_iv["30m"].get(sym))
        s1h = sig(frames_by_iv["1H"].get(sym))
        s2h = sig(frames_by_iv["2H"].get(sym))
        s4h = sig(frames_by_iv["4H"].get(sym))
        s1d = sig(frames_by_iv["1D"].get(sym), is_daily=True)

        line_short = s30 if (s30 == s1h and s30 != "N/A") else f"Mixed (30m:{s30}, 1H:{s1h})"
        line_mid   = s2h if (s2h == s4h and s2h != "N/A") else f"Mixed (2H:{s2h}, 4H:{s4h})"
        msg = f"==={disp}===\n30m-1H: {line_short}\n2H-4H: {line_mid}\n1D: {s1d}"

        # notes gọn khi thiếu dữ liệu
        if INC_ERR_TG:
            missing = [k for k,v in [("30m",s30),("1H",s1h),("2H",s2h),("4H",s4h),("1D",s1d)] if v=="N/A"]
            if missing:
                msg += f"\n⚠ thiếu dữ liệu: {', '.join(missing)}"
        logging.info(msg.replace("\n", " | "))
        reports.append(msg)

    send_tele("\n\n".join(reports))

if __name__ == "__main__":
    main()
