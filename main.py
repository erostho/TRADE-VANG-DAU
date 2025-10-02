#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# ================= LOGGING =================
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ================= CONFIG =================
API_KEY = os.getenv("TWELVE_DATA_KEY", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

BASE_URL = "https://api.twelvedata.com/time_series"
TZ_NAME = os.getenv("TZ", "Asia/Ho_Chi_Minh")  # dùng cho timestamp & lịch daily
TZ = ZoneInfo("Asia/Ho_Chi_Minh") if TZ_NAME == "Asia/Ho_Chi_Minh" else ZoneInfo(TZ_NAME)

# Request per minute throttle
RPM = int(os.getenv("RPM", 6))  # giảm mặc định 6 req/phút để tránh 429

# Lưu cache Daily (1D) ra file để survive restart
DAILY_CACHE_PATH = os.getenv("DAILY_CACHE_PATH", "daily_cache.json")
DAILY_FETCH_HOUR = 0
DAILY_FETCH_MIN = 5  # 00:05 VN: nến day đã đóng

# ================= SYMBOLS & INTERVALS =================
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
    "4H": ["4h"],
}

# ================= HELPERS =================
def now_vn():
    return datetime.now(TZ)

def td_url(symbol, interval, outputsize=100):
    return f"{BASE_URL}?symbol={symbol}&interval={interval}&apikey={API_KEY}&outputsize={outputsize}"

def fetch_candles(symbol, interval, retries=3):
    url = td_url(symbol, interval)
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=12)
            if r.status_code == 429:
                # hit rate limit, chờ sang phút sau
                sleep_s = 65
                logging.warning(f"429 for {symbol}-{interval}, sleep {sleep_s}s...")
                time.sleep(sleep_s)
                continue
            r.raise_for_status()
            data = r.json()
            if "values" not in data:
                logging.warning(f"No data for {symbol}-{interval}: {data}")
                return None
            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").reset_index(drop=True)
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna()
            return df
        except Exception as e:
            logging.error(f"Fetch error {symbol}-{interval} (attempt {attempt+1}): {e}")
            time.sleep(3)
    return None

def atr(df, period=14):
    if df is None or len(df) < period + 2:
        return np.nan
    high = df["high"].values
    low = df["low"].values
    prev_close = df["close"].shift(1).fillna(df["close"])
    tr = np.maximum(high - low, np.maximum(abs(high - prev_close), abs(low - prev_close)))
    atr_vals = pd.Series(tr).rolling(period).mean()
    return float(atr_vals.iloc[-1])

# ===== bộ lọc xu hướng nâng cao =====
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
    val = dx.rolling(n).mean().iloc[-1]
    try:
        return float(val)
    except Exception:
        return np.nan

def strong_trend(df):
    """
    Lọc nhiễu: EMA(20/50) + vị trí giá + ADX + slope EMA20 (5 nến).
    LONG khi: EMA20>EMA50, giá>EMA20, ADX>=20, slope>0.02%/5 nến
    SHORT khi: EMA20<EMA50, giá<EMA20, ADX>=20, slope<-0.02%/5 nến
    """
    if df is None or len(df) < 60:
        return "N/A"
    e20 = ema(df['close'], 20)
    e50 = ema(df['close'], 50)
    if len(e20) < 6:
        return "N/A"
    last = df['close'].iloc[-1]
    adx_val = adx(df, 14)
    if np.isnan(adx_val):
        return "N/A"
    slope = (e20.iloc[-1] - e20.iloc[-6]) / max(1e-9, e20.iloc[-6]) * 100

    if (e20.iloc[-1] > e50.iloc[-1]) and (last > e20.iloc[-1]) and (adx_val >= 20) and (slope > 0.02):
        return "LONG"
    if (e20.iloc[-1] < e50.iloc[-1]) and (last < e20.iloc[-1]) and (adx_val >= 20) and (slope < -0.02):
        return "SHORT"
    return "SIDEWAY"

def swing_levels(df, lookback=20):
    """Swing gần nhất (đỉnh/đáy trước nến hiện tại) để đặt SL thực tế hơn."""
    if df is None or len(df) < lookback + 2:
        return (np.nan, np.nan)
    swing_hi = df['high'].rolling(lookback).max().iloc[-2]
    swing_lo = df['low' ].rolling(lookback).min().iloc[-2]
    return float(swing_hi), float(swing_lo)

# ===== 1D cache (fetch 1 lần/ngày lúc 00:05 VN) =====
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
        logging.warning(f"Save daily cache failed: {e}")

def should_fetch_daily(now: datetime, cache_date: str | None) -> bool:
    """
    Trả True nếu:
    - Sau 00:05 hôm nay (VN) và cache chưa là hôm nay.
    """
    today = now.date()
    cutoff = datetime(today.year, today.month, today.day, DAILY_FETCH_HOUR, DAILY_FETCH_MIN, tzinfo=TZ)
    if now >= cutoff:
        return cache_date != today.isoformat()
    return False

def get_daily_trend(symbol: str, name: str, cache: dict) -> str:
    """
    Lấy trend 1D:
      - Nếu đến giờ 00:05 và chưa có cache hôm nay -> fetch 1D và cập nhật cache.
      - Ngược lại -> dùng cache (nếu có), nếu chưa có thì tạm "N/A".
    """
    now = now_vn()
    cache_date = cache.get("date")
    if should_fetch_daily(now, cache_date):
        # fetch 1D HÔM NAY
        df1d = fetch_candles(symbol, "1day")
        trend = strong_trend(df1d) if df1d is not None else "N/A"
        cache.setdefault("data", {})[symbol] = {"trend": trend}
        cache["date"] = now.date().isoformat()
        save_daily_cache(cache)
        logging.info(f"[1D] fetched {name} -> {trend}")
        # throttle nhẹ đề phòng
        time.sleep(60.0 / RPM)
        return trend
    # dùng cache hiện có
    trend = cache.get("data", {}).get(symbol, {}).get("trend", "N/A")
    return trend

# ========= FORMAT ==========
def format_price(symbol: str, price: float) -> str:
    """Định dạng số thập phân hợp lý theo loại sản phẩm."""
    if price is None or np.isnan(price):
        return "N/A"
    # Forex: thường 5 chữ số, JPY 3
    if symbol in ("EUR/USD",):
        return f"{price:.5f}"
    if symbol in ("USD/JPY",):
        return f"{price:.3f}"
    # Vàng/Dầu/BTC: 2 chữ số là ổn
    return f"{price:.2f}"

# ================= CORE ANALYZE =================
def analyze_symbol(name, symbol, daily_cache):
    results = {}
    has_data = False

    # 1) Tính trend cho các nhóm như cũ nhưng dùng strong_trend()
    for group, intervals in interval_groups.items():
        trends = []
        for iv in intervals:
            df = fetch_candles(symbol, iv)
            trend = strong_trend(df)
            trends.append(f"{iv}:{trend}")
            time.sleep(60.0 / RPM)  # throttle mỗi interval
        if len(intervals) == 1:
            res = trends[0].split(":")[1]
        else:
            uniq = set(t.split(":")[1] for t in trends)
            if len(uniq) == 1:
                res = uniq.pop()
            else:
                res = "Mixed (" + ", ".join(trends) + ")"
        results[group] = res
        if res != "N/A":
            has_data = True

    # 2) Khung chính 1H & 4H + 1D (từ cache)
    df1h = fetch_candles(symbol, "1h")
    time.sleep(60.0 / RPM)
    df4h = fetch_candles(symbol, "4h")

    plan = "SIDEWAY"; entry = sl = tp = atrval = None

    bias_1h = strong_trend(df1h) if df1h is not None else "N/A"
    bias_4h = strong_trend(df4h) if df4h is not None else "N/A"

    daily_trend = get_daily_trend(symbol, name, daily_cache)  # KHÔNG in ra Tele, chỉ dùng lọc
    # 3) Điểm tin cậy (0..4): (1H~4H), (15/30 khớp 1H), (ADX1H ok), (1D khớp 1H)
    score = 0
    if bias_1h in ("LONG","SHORT") and bias_1h == bias_4h:
        score += 1
    # rút trend từ nhóm 15m-30m
    g1530 = results.get("15m-30m", "N/A")
    g1530_norm = "N/A"
    if g1530.startswith("Mixed"):
        g1530_norm = "MIX"
    else:
        g1530_norm = g1530
    if g1530_norm in ("LONG","SHORT") and g1530_norm == bias_1h:
        score += 1
    adx_1h = adx(df1h, 14) if df1h is not None else np.nan
    if not np.isnan(adx_1h) and adx_1h >= 20:
        score += 1
    if daily_trend in ("LONG","SHORT") and daily_trend == bias_1h:
        score += 1

    # 4) Tạo Entry/SL/TP khi score >= 2
    if score >= 2 and bias_1h == bias_4h and bias_1h in ("LONG","SHORT") and df1h is not None and len(df1h) > 40:
        has_data = True
        plan = bias_1h
        entry  = float(df1h["close"].iloc[-1])
        atrval = atr(df1h, 14)
        swing_hi, swing_lo = swing_levels(df1h, 20)

        # hệ số ATR: Forex 2.5, còn lại 1.5
        is_fx = (symbol in {"EUR/USD","USD/JPY"}) or (name in {"EUR/USD","USD/JPY"})
        base_mult = 2.5 if is_fx else 1.5
        buf = 0.5 * atrval if not np.isnan(atrval) else 0.0

        if plan == "LONG":
            sl_candidates = [entry - base_mult * atrval]
            if not np.isnan(swing_lo):
                sl_candidates.append(swing_lo - buf)
            sl = min(sl_candidates)
            r  = entry - sl
            tp = entry + 1.4 * r
        else:  # SHORT
            sl_candidates = [entry + base_mult * atrval]
            if not np.isnan(swing_hi):
                sl_candidates.append(swing_hi + buf)
            sl = max(sl_candidates)
            r  = sl - entry
            tp = entry - 1.4 * r

    return results, plan, entry, sl, tp, atrval, has_data

# ================= TELEGRAM =================
def send_telegram(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram not configured")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
        r.raise_for_status()
        logging.info("Telegram: sent")
    except Exception as e:
        logging.error(f"Telegram send failed: {e}")

# ================= MAIN =================
def main():
    # load daily cache mỗi lần chạy
    daily_cache = load_daily_cache()

    lines = []
    now_str = now_vn().strftime("%Y-%m-%d %H:%M:%S")
    lines.append("💵 TRADE GOODS")
    lines.append(f"⏱ {now_str}\n")

    any_symbol_has_data = False

    for name, sym in symbols.items():
        results, plan, entry, sl, tp, atrval, has_data = analyze_symbol(name, sym, daily_cache)

        lines.append(f"==={name}===")
        for group, trend in results.items():
            lines.append(f"{group}: {trend}")
    
        # >>> thêm dòng 1D trend
        daily_trend = daily_cache.get("data", {}).get(sym, {}).get("trend", "N/A")
        lines.append(f"1D: {daily_trend}")
    
        if entry is not None and sl is not None and tp is not None:
            lines.append(
                f"Entry {format_price(sym, entry)} | SL {format_price(sym, sl)} | TP {format_price(sym, tp)}"
            )
        lines.append("")

        # dàn request để không vượt quota
        time.sleep(10)

    # Nếu tất cả đều N/A/SIDEWAY & không có Entry -> vẫn gửi để biết trạng thái; nếu muốn, có thể chặn ở đây
    msg = "\n".join(lines)
    send_telegram(msg)

if __name__ == "__main__":
    main()
