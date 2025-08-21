# -*- coding: utf-8 -*-
"""
Frames: 30m, 1H, 1D
Symbols: GC=F (Gold futures), CL=F (WTI), BTC-USD, EURUSD=X, JPY=X
Logic: EMA20/50/200 + RSI14 + MACD + ADX + BBWidth
Yahoo (yfinance) -> nếu fail vì bị chặn IP/404 -> fallback TwelveData (nếu có TWELVE_DATA_KEY)
ENV:
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
  LOG_LEVEL=INFO/DEBUG (optional)
  INCLUDE_ERRORS_IN_TELEGRAM=1 (optional)
  TWELVE_DATA_KEY=<your_key> (optional)
"""

import os, logging, requests
import numpy as np
import pandas as pd
import yfinance as yf

# ---------- CONFIG ----------
ASSETS = {
    "XAU/USD (Gold)": "GC=F",
    "WTI Oil": "CL=F",
    "Bitcoin": "BTC-USD",
    "EUR/USD": "EURUSD=X",
    "USD/JPY": "JPY=X",
}

INTRADAY_PERIODS = ["60d", "30d", "14d", "7d", "5d", "2d", "1d"]
DAILY_PERIODS    = ["5y", "3y", "1y"]

ADX_TREND = 20
SEND_TELE = True

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
TD_KEY    = os.getenv("TWELVE_DATA_KEY")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
INC_ERR_TG = os.getenv("INCLUDE_ERRORS_IN_TELEGRAM", "0") == "1"

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------- COMMON INDICATORS ----------
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
        and last["RSI14"] > 55
        and last["MACD"] > last["MACD_SIG"]
        and trending):
        return "LONG"
    if (last["Close"] < last["EMA20"] < last["EMA50"] < last["EMA200"]
        and last["RSI14"] < 45
        and last["MACD"] < last["MACD_SIG"]
        and trending):
        return "SHORT"
    return "SIDEWAY"

# ---------- YAHOO (with Session UA) ----------
_YF_SESSION = requests.Session()
_YF_SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0 Safari/537.36"
})

def _yahoo_download(symbol, interval, period):
    logging.debug(f"Yahoo try {symbol} {interval}/{period}")
    df = yf.download(symbol, interval=interval, period=period,
                     auto_adjust=False, progress=False, session=_YF_SESSION)
    if df is None or df.empty or not {"Open","High","Low","Close"}.issubset(df.columns):
        raise RuntimeError(f"No data for {symbol} @ {interval}/{period}")
    df = df.dropna().copy()
    df.index = pd.to_datetime(df.index)
    logging.info(f"Yahoo OK {symbol} {interval}/{period} -> {len(df)} rows "
                 f"[{df.index[0]} .. {df.index[-1]}]")
    return df

def yahoo_intraday(symbol, interval):
    notes = []
    if interval not in ("30m", "1h"):
        raise ValueError("intraday interval must be 30m or 1h")
    # 30m thử rồi fallback 1h
    if interval == "30m":
        for p in INTRADAY_PERIODS:
            try:
                return _yahoo_download(symbol, "30m", p), "30m", p, "; ".join(notes)
            except Exception as e:
                notes.append(f"30m/{p} fail: {e}")
        for p in INTRADAY_PERIODS:
            try:
                notes.append("fallback 30m→1h")
                return _yahoo_download(symbol, "1h", p), "1h", p, "; ".join(notes)
            except Exception as e:
                notes.append(f"1h/{p} fail: {e}")
        raise RuntimeError("; ".join(notes))
    else:
        last_err = None
        for p in INTRADAY_PERIODS:
            try:
                return _yahoo_download(symbol, "1h", p), "1h", p, "; ".join(notes)
            except Exception as e:
                last_err = e; notes.append(f"1h/{p} fail: {e}")
        raise RuntimeError("; ".join(notes))

def yahoo_daily(symbol):
    for p in DAILY_PERIODS:
        try:
            return _yahoo_download(symbol, "1d", p), "1d", p, ""
        except Exception as e:
            logging.warning(f"Yahoo 1d {symbol} {p} fail: {e}")
    raise RuntimeError(f"Yahoo daily fail for {symbol}")

# ---------- TWELVE DATA (fallback) ----------
# Map Yahoo ticker -> TwelveData symbol
TD_MAP = {
    "GC=F": "XAU/USD",   # vàng spot trên TD
    "CL=F": "CL",        # WTI continuous
    "BTC-USD": "BTC/USD",
    "EURUSD=X": "EUR/USD",
    "JPY=X": "USD/JPY",  # TD dùng chiều USD/JPY
}

def _td_series(symbol, interval):
    # intervals hợp lệ: 30min, 1h, 1day
    iv = {"30m":"30min", "1h":"1h", "1d":"1day"}[interval]
    url = "https://api.twelvedata.com/time_series"
    r = requests.get(url, params={
        "symbol": symbol,
        "interval": iv,
        "outputsize": 5000,
        "apikey": TD_KEY
    }, timeout=25)
    r.raise_for_status()
    js = r.json()
    if "values" not in js:
        raise RuntimeError(f"TD no values: {js}")
    df = pd.DataFrame(js["values"])
    # TD trả newest->oldest
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").set_index("datetime")
    df = df.rename(columns={
        "open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"
    })
    df[["Open","High","Low","Close"]] = df[["Open","High","Low","Close"]].astype(float)
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    return df

def td_fetch(yahoo_symbol, interval):
    sym = TD_MAP[yahoo_symbol]
    logging.info(f"TwelveData fetch {sym} {interval}")
    df = _td_series(sym, "1d" if interval=="1d" else interval)
    logging.info(f"TwelveData OK {sym} {interval} -> {len(df)} rows "
                 f"[{df.index[0]} .. {df.index[-1]}]")
    return df, interval, "TD", ""

# ---------- TELEGRAM ----------
def send_tele(text):
    if not SEND_TELE: return
    if not BOT_TOKEN or not CHAT_ID:
        logging.warning("Missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID"); return
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text},
            timeout=20
        )
        if r.status_code != 200:
            logging.warning(f"Telegram send failed: {r.status_code} {r.text}")
    except Exception as e:
        logging.exception(f"Telegram error: {e}")

# ---------- ANALYZE ONE ----------
def analyze_one(name, ticker):
    logging.info(f"=== Start {name} ({ticker}) ===")
    notes = []

    # 30m
    try:
        d30, i30, p30, n30 = yahoo_intraday(ticker, "30m")
        if n30: notes.append(f"YF 30m: {n30}")
    except Exception as e:
        notes.append(f"YF 30m ERR: {e}")
        if TD_KEY and ticker in TD_MAP:
            try:
                d30, i30, p30, n30 = td_fetch(ticker, "30m")
                notes.append("TD 30m OK")
            except Exception as ee:
                notes.append(f"TD 30m ERR: {ee}")
                d30 = None
        else:
            d30 = None
    s30 = "N/A"
    if isinstance(d30, pd.DataFrame) and len(d30) >= 100:
        s30 = decide(decorate(d30))

    # 1H
    try:
        d1h, i1h, p1h, n1h = yahoo_intraday(ticker, "1h")
        if n1h: notes.append(f"YF 1h: {n1h}")
    except Exception as e:
        notes.append(f"YF 1h ERR: {e}")
        if TD_KEY and ticker in TD_MAP:
            try:
                d1h, i1h, p1h, n1h = td_fetch(ticker, "1h")
                notes.append("TD 1h OK")
            except Exception as ee:
                notes.append(f"TD 1h ERR: {ee}")
                d1h = None
        else:
            d1h = None
    s1h = "N/A"
    if isinstance(d1h, pd.DataFrame) and len(d1h) >= 100:
        s1h = decide(decorate(d1h))

    # 1D
    try:
        d1d, i1d, p1d, _ = yahoo_daily(ticker)
    except Exception as e:
        notes.append(f"YF 1d ERR: {e}")
        if TD_KEY and ticker in TD_MAP:
            try:
                d1d, _, _, _ = td_fetch(ticker, "1d")
                notes.append("TD 1d OK")
            except Exception as ee:
                notes.append(f"TD 1d ERR: {ee}")
                d1d = None
    s1d = "N/A"
    if isinstance(d1d, pd.DataFrame) and len(d1d) >= 200:
        s1d = decide(decorate(d1d))

    # Compose
    short_line = s30 if (s30 == s1h and s30 != "N/A") else f"Mixed (30m:{s30}, 1H:{s1h})"
    msg = f"==={name}===\n30m-1H: {short_line}\n1H: {s1h}\n1D: {s1d}"
    if notes and INC_ERR_TG:
        compact = "; ".join(notes)
        if len(compact) > 700: compact = compact[:700] + "..."
        msg += f"\n⚠ {compact}"
    logging.info(msg.replace("\n"," | "))
    return msg

# ---------- MAIN ----------
def main():
    parts = []
    for name, ticker in ASSETS.items():
        try:
            parts.append(analyze_one(name, ticker))
        except Exception as e:
            logging.exception(f"{name} fatal")
            parts.append(f"==={name}===\nLỗi nghiêm trọng: {e}")
    send_tele("\n\n".join(parts))

if __name__ == "__main__":
    main()
