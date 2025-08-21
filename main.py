# -*- coding: utf-8 -*-
"""
TwelveData-only Trend Scanner
Frames: 30m, 1H, 2H, 4H, 1D
Assets:
  - XAU/USD (Gold) -> XAU/USD
  - WTI Oil        -> CL   (fallback WTI/USD)
  - Bitcoin        -> BTC/USD
  - EUR/USD        -> EUR/USD
  - USD/JPY        -> USD/JPY

Output Telegram (ví dụ):
===XAU/USD (Gold)===
30m-1H: SIDEWAY
2H-4H: LONG
1D: LONG

ENV (bắt buộc):
  TWELVE_DATA_KEY
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID
ENV (tùy chọn):
  LOG_LEVEL=INFO|DEBUG
  INCLUDE_ERRORS_IN_TELEGRAM=1
  TD_SLEEP_BETWEEN_CALL=8   # giây nghỉ giữa mỗi call để tránh 429
"""

import os, time, logging, requests
import pandas as pd
import numpy as np

# ---------- ENV / LOG ----------
TD_KEY   = os.getenv("TWELVE_DATA_KEY")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
INC_ERR_TG = os.getenv("INCLUDE_ERRORS_IN_TELEGRAM", "0") == "1"
TD_SLEEP_BETWEEN_CALL = float(os.getenv("TD_SLEEP_BETWEEN_CALL", "8"))

if not TD_KEY:
    raise SystemExit("❌ Missing TWELVE_DATA_KEY")
if not BOT_TOKEN or not CHAT_ID:
    print("⚠️ Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID (vẫn chạy nhưng không gửi Telegram)")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------- CONFIG ----------
ASSETS = {
    "XAU/USD (Gold)": "XAU/USD",
    "WTI Oil": "CL",            # fallback "WTI/USD" ở bên dưới
    "Bitcoin": "BTC/USD",
    "EUR/USD": "EUR/USD",
    "USD/JPY": "USD/JPY",
}
TD_INTERVAL = {"30m": "30min", "1H": "1h", "2H": "2h", "4H": "4h", "1D": "1day"}
OUTPUTSIZE = 2000

# Ngưỡng lọc / số nến tối thiểu
MIN_BARS_INTRADAY = 60    # 30m/1H/2H/4H
MIN_BARS_DAILY    = 120   # 1D
ADX_TREND = 20

# ---------- Indicators ----------
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(close, n=14):
    d = close.diff()
    gain, loss = d.clip(lower=0), (-d).clip(lower=0)
    ag = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    al = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(close, fast=12, slow=26, sig=9):
    m = ema(close, fast) - ema(close, slow)
    s = ema(m, sig)
    return m, s, m - s

def true_range(df):
    pc = df["Close"].shift(1)
    return pd.concat([(df["High"]-df["Low"]),
                      (df["High"]-pc).abs(),
                      (df["Low"]-df["Close"].shift(1)).abs()], axis=1).max(axis=1)

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

# ---------- TwelveData: single call (retry + throttle) ----------
def td_single_time_series(symbol: str, interval: str, outputsize=OUTPUTSIZE, retries=2):
    """
    interval: '30min','1h','2h','4h','1day'
    Trả về DataFrame OHLC (index = datetime). Tự nghỉ giữa các call để tránh 429.
    """
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": symbol, "interval": interval, "outputsize": outputsize, "apikey": TD_KEY}
    last_err = None
    for attempt in range(retries + 1):
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 429:
            wait = 65 if attempt < retries else 0
            logging.warning(f"TD 429 for {symbol} {interval}, backoff {wait}s (attempt {attempt+1}/{retries+1})")
            if wait:
                time.sleep(wait)
                continue
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
            time.sleep(TD_SLEEP_BETWEEN_CALL)  # throttle
            return df
        except Exception as e:
            last_err = e
            logging.warning(f"TD fail {symbol} {interval}: {e}")
            time.sleep(TD_SLEEP_BETWEEN_CALL)
    raise RuntimeError(f"TD final fail {symbol} {interval}: {last_err}")

# ---------- Telegram ----------
def send_tele(text: str):
    if not BOT_TOKEN or not CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text},
            timeout=20
        )
    except Exception as e:
        logging.exception(f"Telegram error: {e}")

# ---------- Main ----------
def main():
    reports = []
    any_signal = False  # có ít nhất 1 LONG/SHORT?

    for display, base_symbol in ASSETS.items():
        logging.info(f"=== Start {display} ({base_symbol}) ===")
        notes = []
        sym_candidates = [base_symbol] if base_symbol != "CL" else ["CL", "WTI/USD"]

        frames = {}
        for lbl, iv in TD_INTERVAL.items():
            df = None
            for sym in sym_candidates:
                try:
                    df = td_single_time_series(sym, iv)
                    break
                except Exception as e:
                    logging.warning(f"Try {sym} {iv} fail: {e}")
            frames[lbl] = df
            if df is None:
                notes.append(f"{lbl} ERR")

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

        line_short = s30 if (s30 == s1h and s30 != "N/A") else f"Mixed (30m:{s30}, 1H:{s1h})"
        line_mid   = s2h if (s2h == s4h and s2h != "N/A") else f"Mixed (2H:{s2h}, 4H:{s4h})"
        msg = f"==={display}===\n30m-1H: {line_short}\n2H-4H: {line_mid}\n1D: {s1d}"

        # chỉ add nếu có LONG/SHORT ở bất kỳ khung nào
        if any(x in ("LONG", "SHORT") for x in [s30, s1h, s2h, s4h, s1d]):
            reports.append(msg)
            any_signal = True
        else:
            logging.info(f"{display}: all SIDEWAY/N/A -> skip Telegram")

    if any_signal:
        send_tele("\n\n".join(reports))
    else:
        logging.info("No trade signals (LONG/SHORT). Telegram not sent.")

if __name__ == "__main__":
    main()
