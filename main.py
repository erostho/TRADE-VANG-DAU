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
# ---- Stronger filters / thresholds ----
ADX_MIN = 22            # >20 trước đây
ADX_RISING_BARS = 3     # ADX tăng liên tiếp N bar
RSI_LONG = 55           # RSI ngưỡng LONG
RSI_SHORT = 45          # RSI ngưỡng SHORT
SLOPE_LOOKBACK = 5      # số bar để kiểm tra độ dốc EMA
ATR_MIN_MULT = 0.002    # tối thiểu biến động: ATR / Close > 0.2% (lọc thị trường “đứng hình”)
PERSIST_BARS = 2        # yêu cầu các điều kiện giữ ít nhất N bar (MACD hist, EMA alignment)
CONF_THRESHOLD = 3      # điểm tối thiểu để coi là LONG/SHORT (xem scorer bên dưới)
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
    d = build_indicators(df)
    label, _ = score_signal(d)
    return label
  
def pct_slope(series, lookback=SLOPE_LOOKBACK):
    if len(series) < lookback + 1: return 0.0
    a, b = series.iloc[-1], series.iloc[-1-lookback]
    return float((a-b)/b) if b else 0.0

def atr(df, n=14):
    return true_range(df).rolling(n).mean()

def rising(series, n=3):
    if len(series) < n+1: return False
    x = series.tail(n+1)
    return bool((x.diff().iloc[1:] > 0).all())

def persist(cond_series, n=PERSIST_BARS):
    if len(cond_series) < n: return False
    return bool(cond_series.tail(n).all())

def build_indicators(df):
    d = decorate(df).copy()
    d["ATR14"] = atr(d, 14)
    d["ema_align_long"]  = (d["EMA20"] > d["EMA50"]) & (d["EMA50"] > d["EMA200"])
    d["ema_align_short"] = (d["EMA20"] < d["EMA50"]) & (d["EMA50"] < d["EMA200"])
    d["macd_bull"] = d["MACD"] > d["MACD_SIG"]
    d["macd_bear"] = d["MACD"] < d["MACD_SIG"]
    d["rsi_bull"]  = d["RSI14"] > RSI_LONG
    d["rsi_bear"]  = d["RSI14"] < RSI_SHORT
    d["adx_ok"]    = d["ADX14"] > ADX_MIN
    return d

def score_signal(d):
    last = d.iloc[-1]
    vol_ok = (last["ATR14"] / last["Close"]) > ATR_MIN_MULT
    adx_up = rising(d["ADX14"], ADX_RISING_BARS)

    ema_long_p = persist(d["ema_align_long"])
    ema_short_p= persist(d["ema_align_short"])
    macd_bull_p= persist(d["macd_bull"])
    macd_bear_p= persist(d["macd_bear"])

    slope50_pos = pct_slope(d["EMA50"])  > 0
    slope200_pos= pct_slope(d["EMA200"]) > 0
    slope50_neg = pct_slope(d["EMA50"])  < 0
    slope200_neg= pct_slope(d["EMA200"]) < 0

    long_score = 0
    if ema_long_p: long_score += 1
    if last["rsi_bull"]: long_score += 1
    if macd_bull_p: long_score += 1
    if last["adx_ok"] and adx_up: long_score += 1
    if slope50_pos and slope200_pos: long_score += 1
    if vol_ok: long_score += 1

    short_score = 0
    if ema_short_p: short_score += 1
    if last["rsi_bear"]: short_score += 1
    if macd_bear_p: short_score += 1
    if last["adx_ok"] and adx_up: short_score += 1
    if slope50_neg and slope200_neg: short_score += 1
    if vol_ok: short_score += 1

    if long_score >= CONF_THRESHOLD and long_score > short_score:
        return "LONG", long_score
    if short_score >= CONF_THRESHOLD and short_score > long_score:
        return "SHORT", short_score
    return "SIDEWAY", max(long_score, short_score)
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
        full_message = "💵 TRADE GOODS\n" + text
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": full_message},
            timeout=20
        )
    except Exception as e:
        logging.exception(f"Telegram error: {e}")

# ---------- Main ----------
def main():
    reports = []
    any_signal = False  # chỉ gửi khi có ít nhất 1 LONG/SHORT

    for display, base_symbol in ASSETS.items():
        logging.info(f"=== Start {display} ({base_symbol}) ===")
        notes = []
        # fallback riêng cho dầu
        sym_candidates = [base_symbol] if base_symbol != "CL" else ["CL", "WTI/USD"]

        # 1) Lấy dữ liệu từng khung
        frames = {}
        for lbl, iv in TD_INTERVAL.items():   # TD_INTERVAL: {"30m":"30min", "1H":"1h", "2H":"2h", "4H":"4h", "1D":"1day"}
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

        # 2) Hàm tính tín hiệu cho từng khung
        def sig(df, is_daily=False):
            need = MIN_BARS_DAILY if is_daily else MIN_BARS_INTRADAY
            if isinstance(df, pd.DataFrame) and len(df) >= need:
                return decide(decorate(df))
            return "N/A"

        # 3) Tín hiệu từng khung
        s30 = sig(frames.get("30m"))
        s1h = sig(frames.get("1H"))
        s2h = sig(frames.get("2H"))
        s4h = sig(frames.get("4H"))
        s1d = sig(frames.get("1D"), is_daily=True)

        # 5) Ghép message theo format yêu cầu
        line_short = s30 if (s30 == s1h and s30 != "N/A") else f"Mixed (30m:{s30}, 1H:{s1h})"
        line_mid   = s2h if (s2h == s4h and s2h != "N/A") else f"Mixed (2H:{s2h}, 4H:{s4h})"
        msg = f"==={display}===\n30m-1H: {line_short}\n2H-4H: {line_mid}\n1D: {s1d}"

        if INC_ERR_TG and notes:
            msg += f"\n⚠ thiếu dữ liệu: {', '.join(notes)}"

        logging.info(msg.replace("\n", " | "))

        # 6) Chỉ add khi có LONG/SHORT ở bất kỳ khung nào
        if any(x in ("LONG", "SHORT") for x in [s30, s1h, s2h, s4h, s1d]):
            reports.append(msg)
            any_signal = True
        else:
            logging.info(f"{display}: all SIDEWAY/N/A -> skip")

    # 7) Gửi Telegram nếu có tín hiệu
    if any_signal:
        send_tele("\n\n".join(reports))
    else:
        logging.info("No trade signals (LONG/SHORT). Telegram not sent.")

if __name__ == "__main__":
    main()
