# -*- coding: utf-8 -*-
"""
Main: Phân tích VÀNG (XAUUSD) & DẦU (XTIUSD, XBRUSD) trên Exness/MT5
- Khung: M15, H1, H4
- Rule chặt (xu hướng H1 & H4 đồng pha; M15 là trigger)
- Chỉ báo: EMA20/50/200, RSI14, MACD(12,26,9), ADX14, ATR14
- SL/TP theo ATR; RR >= 1.5
- Gửi tín hiệu về Telegram (BotFather)
"""

import os
import sys
import time
import math
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytz
from dotenv import load_dotenv

import MetaTrader5 as mt5
import requests

# ============ CONFIG ============
TZ = pytz.timezone("Asia/Ho_Chi_Minh")

SYMBOLS = [
    "XAUUSD",  # Gold
    "XTIUSD",  # WTI
    "XBRUSD",  # Brent
]

TIMEFRAMES = {
    "M15": mt5.TIMEFRAME_M15,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
}

BARS = 400  # số nến tải cho mỗi TF (đủ cho EMA/ADX)

# Tham số ATR/Rule
ATR_PERIOD = 14
ADX_PERIOD = 14
EMA_FAST = 20
EMA_MID  = 50
EMA_SLOW = 200

# RR & SL/TP multipliers
SL_ATR_MULT = 1.5
TP_RR_MIN   = 1.5  # TP tối thiểu theo RR

# ADX ngưỡng trend
ADX_TREND = 20

# RSI filter
RSI_LONG_MIN  = 52
RSI_SHORT_MAX = 48

# MACD tiêu chuẩn
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIG  = 9

# Gửi tele khi có tín hiệu mới (True), hoặc chỉ in log (False)
SEND_TELE = True

# ============ LOGGING ============
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ============ UTILS ============
def now_vn():
    return datetime.now(tz=TZ)

def ema(series: pd.Series, n: int):
    return series.ewm(span=n, adjust=False).mean()

def rsi(series: pd.Series, n: int = 14):
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/n, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(50)

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_ln = ema(macd_line, signal)
    hist = macd_line - signal_ln
    return macd_line, signal_ln, hist

def true_range(df: pd.DataFrame):
    prev_close = df["c"].shift(1)
    tr1 = df["h"] - df["l"]
    tr2 = (df["h"] - prev_close).abs()
    tr3 = (df["l"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr

def atr(df: pd.DataFrame, n=14):
    tr = true_range(df)
    return tr.rolling(n).mean()

def adx(df: pd.DataFrame, n=14):
    # Welles Wilder ADX
    up_move = df["h"].diff()
    down_move = df["l"].diff() * -1

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(df)
    tr_n = tr.rolling(n).sum()
plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(n).sum() / tr_n)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(n).sum() / tr_n)

    dx = ( (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) ) * 100
    adx_val = dx.rolling(n).mean()
    return plus_di.fillna(0), minus_di.fillna(0), adx_val.fillna(0)

def bbands(series: pd.Series, n=20, nb=2.0):
    ma = series.rolling(n).mean()
    sd = series.rolling(n).std(ddof=0)
    upper = ma + nb*sd
    lower = ma - nb*sd
    return upper, ma, lower

def fetch_ohlc(symbol: str, tf_const, bars: int) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, bars)
    if rates is None:
        raise RuntimeError(f"Cannot fetch rates for {symbol} @ {tf_const}: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    # MT5 time is UTC seconds
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(TZ)
    df.rename(columns={"open":"o","high":"h","low":"l","close":"c","tick_volume":"vol"}, inplace=True)
    return df[["time","o","h","l","c","vol"]]

def enrich_indicators(df: pd.DataFrame):
    df = df.copy()
    df["ema20"]  = ema(df["c"], EMA_FAST)
    df["ema50"]  = ema(df["c"], EMA_MID)
    df["ema200"] = ema(df["c"], EMA_SLOW)
    df["rsi14"]  = rsi(df["c"], 14)
    df["macd"], df["macd_sig"], df["macd_hist"] = macd(df["c"], MACD_FAST, MACD_SLOW, MACD_SIG)
    df["atr14"]  = atr(df, ATR_PERIOD)
    df["+di"], df["-di"], df["adx"] = adx(df, ADX_PERIOD)
    df["bb_up"], df["bb_mid"], df["bb_lo"] = bbands(df["c"], 20, 2.0)
    return df

def align_trend(df_h1: pd.DataFrame, df_h4: pd.DataFrame, side: str) -> bool:
    """Xu hướng H1 & H4 đồng pha bằng EMA và ADX"""
    h1 = df_h1.iloc[-1]
    h4 = df_h4.iloc[-1]

    if side == "LONG":
        cond_h1 = (h1["c"] > h1["ema50"] > h1["ema200"]) and (h1["adx"] >= ADX_TREND)
        cond_h4 = (h4["c"] > h4["ema50"] > h4["ema200"]) and (h4["adx"] >= ADX_TREND)
        return bool(cond_h1 and cond_h4)

    if side == "SHORT":
        cond_h1 = (h1["c"] < h1["ema50"] < h1["ema200"]) and (h1["adx"] >= ADX_TREND)
        cond_h4 = (h4["c"] < h4["ema50"] < h4["ema200"]) and (h4["adx"] >= ADX_TREND)
        return bool(cond_h1 and cond_h4)

    return False

def trigger_m15(df_m15: pd.DataFrame, side: str) -> bool:
    """Trigger M15: cross EMA20 + MACD đồng pha + RSI filter + tránh quá mua/quá bán cực đoan"""
    last = df_m15.iloc[-1]
    prev = df_m15.iloc[-2]

    # Cross EMA20
    crossed_up   = (prev["c"] <= prev["ema20"]) and (last["c"] > last["ema20"])
    crossed_down = (prev["c"] >= prev["ema20"]) and (last["c"] < last["ema20"])

    # MACD cùng hướng
    macd_up   = (last["macd"] > last["macd_sig"])
    macd_down = (last["macd"] < last["macd_sig"])

    # RSI filter
    if side == "LONG":
        if not (crossed_up and macd_up and last["rsi14"] >= RSI_LONG_MIN):
            return False
# Tránh mua quá hưng phấn: giá không quá sát BB trên > +1σ
        if pd.notna(last["bb_up"]) and (last["c"] > last["bb_up"]):
            return False
        return True

    if side == "SHORT":
        if not (crossed_down and macd_down and last["rsi14"] <= RSI_SHORT_MAX):
            return False
        # Tránh bán đuổi: giá không thủng mạnh dưới BB dưới < -1σ
        if pd.notna(last["bb_lo"]) and (last["c"] < last["bb_lo"]):
            return False
        return True

    return False

def build_sl_tp(price: float, atr_val: float, side: str, rr_min=TP_RR_MIN):
    if np.isnan(atr_val) or atr_val <= 0:
        return None, None, None
    sl = None
    tp = None
    rr = None
    if side == "LONG":
        sl = price - SL_ATR_MULT * atr_val
        # TP theo RR tối thiểu
        risk = price - sl
        tp = price + rr_min * risk
        rr = (tp - price) / (price - sl) if (price - sl) != 0 else None
    else:
        sl = price + SL_ATR_MULT * atr_val
        risk = sl - price
        tp = price - rr_min * risk
        rr = (price - tp) / (sl - price) if (sl - price) != 0 else None
    return float(sl), float(tp), (float(rr) if rr is not None else None)

def format_signal_msg(sym, side, tf_trigger, price, sl, tp, rr,
                      h1, h4, m15, now_ts):
    def f(x, n=2):
        try: return f"{x:.{n}f}"
        except: return str(x)

    msg = [
        "🔥 TÍN HIỆU HÀNG HÓA (Exness/MT5)",
        f"Mã: {sym} | Hướng: {side}",
        f"Trigger: {tf_trigger} @ {f(price)} | SL: {f(sl)} | TP: {f(tp)} | RR≈{f(rr,2)}",
        "",
        "— H1 Snapshot —",
        f"Price {f(h1['c'])} | EMA20 {f(h1['ema20'])} | EMA50 {f(h1['ema50'])} | EMA200 {f(h1['ema200'])}",
        f"RSI14 {f(h1['rsi14'],1)} | ADX {f(h1['adx'],1)} | +DI {f(h1['+di'],1)} | -DI {f(h1['-di'],1)}",
        "",
        "— H4 Snapshot —",
        f"Price {f(h4['c'])} | EMA50 {f(h4['ema50'])} | EMA200 {f(h4['ema200'])} | ADX {f(h4['adx'],1)}",
        "",
        "— M15 Snapshot —",
        f"RSI14 {f(m15['rsi14'],1)} | MACD {f(m15['macd'],4)} vs Sig {f(m15['macd_sig'],4)} | ATR14 {f(m15['atr14'])}",
        f"Time: {now_ts.strftime('%Y-%m-%d %H:%M:%S %Z')}"
    ]
    return "\n".join(msg)

def send_telegram(bot_token: str, chat_id: str, text: str):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=15)
        if r.status_code != 200:
            logging.warning(f"Telegram send failed [{r.status_code}]: {r.text}")
    except Exception as e:
        logging.exception(f"Telegram error: {e}")

def analyze_symbol(sym: str):
    # Tải OHLC cho M15, H1, H4
    dfs = {}
    for label, tfc in TIMEFRAMES.items():
        df = fetch_ohlc(sym, tfc, BARS)
        df = enrich_indicators(df)
        dfs[label] = df

    m15 = dfs["M15"].iloc[-1]
    h1  = dfs["H1"].iloc[-1]
    h4  = dfs["H4"].iloc[-1]
    # Xác định hướng tiềm năng theo H1/H4
    long_trend_ok  = align_trend(dfs["H1"], dfs["H4"], "LONG")
    short_trend_ok = align_trend(dfs["H1"], dfs["H4"], "SHORT")

    signals = []
    # Trigger M15
    if long_trend_ok and trigger_m15(dfs["M15"], "LONG"):
        price = float(m15["c"])
        sl, tp, rr = build_sl_tp(price, float(m15["atr14"]), "LONG", TP_RR_MIN)
        if rr is not None and rr >= TP_RR_MIN:
            signals.append(("LONG", price, sl, tp, rr))

    if short_trend_ok and trigger_m15(dfs["M15"], "SHORT"):
        price = float(m15["c"])
        sl, tp, rr = build_sl_tp(price, float(m15["atr14"]), "SHORT", TP_RR_MIN)
        if rr is not None and rr >= TP_RR_MIN:
            signals.append(("SHORT", price, sl, tp, rr))

    # Chọn 1 tín hiệu (ưu tiên theo DI/ADX thuận hướng mạnh hơn)
    side_picked = None
    payload = None
    if len(signals) == 1:
        side_picked, price, sl, tp, rr = signals[0]
        payload = (side_picked, price, sl, tp, rr)
    elif len(signals) == 2:
        # Quyết định theo H1: nếu +DI > -DI → ưu tiên LONG, ngược lại ưu tiên SHORT
        side_pref = "LONG" if h1["+di"] > h1["-di"] else "SHORT"
        cand = [s for s in signals if s[0] == side_pref]
        payload = cand[0] if cand else signals[0]

    return dfs, payload  # dfs chứa 3 TF; payload có thể None nếu không có tín hiệu

def ensure_symbol_visible(symbol: str):
    info = mt5.symbol_info(symbol)
    if info is None:
        # thử hiển thị
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Symbol {symbol} không khả dụng trên tài khoản MT5 hiện tại.")
    elif not info.visible:
        mt5.symbol_select(symbol, True)

def main():
    load_dotenv()
    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")

    bot_token = os.getenv("TELE_BOT_TOKEN")
    chat_id   = os.getenv("TELE_CHAT_ID")

    if not (login and password and server):
        logging.error("Thiếu thông tin MT5_LOGIN / MT5_PASSWORD / MT5_SERVER trong .env")
        sys.exit(1)

    login = int(login)

    if not mt5.initialize():
        logging.error(f"MT5 init fail: {mt5.last_error()}")
        sys.exit(1)

    if not mt5.login(login, password=password, server=server):
        logging.error(f"MT5 login fail: {mt5.last_error()}")
        mt5.shutdown()
        sys.exit(1)

    logging.info("Đăng nhập MT5 thành công.")

    for sym in SYMBOLS:
        try:
            ensure_symbol_visible(sym)
            dfs, payload = analyze_symbol(sym)
            m15 = dfs["M15"].iloc[-1]
            h1  = dfs["H1"].iloc[-1]
            h4  = dfs["H4"].iloc[-1]

            if payload:
                side, price, sl, tp, rr = payload
                msg = format_signal_msg(
                    sym, side, "M15", price, sl, tp, rr,
                    h1=h1, h4=h4, m15=m15, now_ts=now_vn()
                )
                logging.info(f"\n{msg}\n")
                if SEND_TELE and bot_token and chat_id:
                    send_telegram(bot_token, chat_id, msg)
            else:
                logging.info(f"{sym}: Không có tín hiệu hợp lệ (rule chặt).")

        except Exception as e:
            logging.exception(f"Lỗi xử lý {sym}: {e}")

    mt5.shutdown()
    logging.info("Hoàn tất.")

if __name__ == "__main__":
    main()