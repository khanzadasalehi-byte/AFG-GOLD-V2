#!/usr/bin/env python3
"""AFG GOLD independent 5-minute Telegram scanner; same XAU/USD rule engine as Streamlit app."""
import os, json, hashlib
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import requests
import pandas as pd
import numpy as np

def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def rsi(series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(df, n=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def adx(df, n=14):
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    a = atr(df, n).replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1/n, adjust=False).mean() / a
    minus_di = 100 * minus_dm.ewm(alpha=1/n, adjust=False).mean() / a
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean()

def add_indicators(df):
    df = df.copy()
    for n in (20, 50, 200):
        df[f"ema{n}"] = ema(df["close"], n)
    df["rsi"] = rsi(df["close"])
    df["atr"] = atr(df)
    df["adx"] = adx(df)
    return df.dropna().copy()

def market_session(now_utc):
    # London / New York local clocks, including daylight-saving changes.
    london = now_utc.astimezone(ZoneInfo("Europe/London"))
    new_york = now_utc.astimezone(ZoneInfo("America/New_York"))
    london_open = 8 <= london.hour < 17
    ny_open = 8 <= new_york.hour < 17
    if london_open and ny_open:
        return "LONDON + NEW YORK", True
    if london_open:
        return "LONDON", True
    if ny_open:
        return "NEW YORK", True
    return "OFF SESSION", False

def build_signal(m15, h1, h4, min_adx, min_score, use_session):
    a = add_indicators(m15)
    b = add_indicators(h1)
    c = add_indicators(h4)
    if min(len(a), len(b), len(c)) < 5:
        raise ValueError("Not enough historical candles for indicator calculations.")

    x, p = a.iloc[-1], a.iloc[-2]
    one, four = b.iloc[-1], c.iloc[-1]
    now = datetime.now(timezone.utc)
    session, in_session = market_session(now)

    long_checks = {
        "1H bullish trend": one.ema20 > one.ema50 and one.close > one.ema200,
        "4H bullish trend": four.ema20 > four.ema50,
        "15M EMA alignment": x.ema20 > x.ema50,
        "ADX trend strength": x.adx >= min_adx,
        "RSI bullish momentum": 50 <= x.rsi <= 72,
        "Bullish candle confirmation": x.close > x.open and x.close > p.high,
        "Price above 15M EMA20": x.close > x.ema20,
    }
    short_checks = {
        "1H bearish trend": one.ema20 < one.ema50 and one.close < one.ema200,
        "4H bearish trend": four.ema20 < four.ema50,
        "15M EMA alignment": x.ema20 < x.ema50,
        "ADX trend strength": x.adx >= min_adx,
        "RSI bearish momentum": 28 <= x.rsi <= 50,
        "Bearish candle confirmation": x.close < x.open and x.close < p.low,
        "Price below 15M EMA20": x.close < x.ema20,
    }

    long_score = sum(bool(v) for v in long_checks.values()) / len(long_checks) * 100
    short_score = sum(bool(v) for v in short_checks.values()) / len(short_checks) * 100
    side = "BUY" if long_score > short_score else "SELL" if short_score > long_score else "WAIT"
    checks = long_checks if side == "BUY" else short_checks if side == "SELL" else {}
    score = int(round(max(long_score, short_score))) if side != "WAIT" else 0
    if use_session and not in_session:
        side = "WAIT"
    # Filters contribute to the score; they are not all mandatory.
    # Signal only when the score is strictly greater than the selected threshold.
    if score <= min_score or not checks:
        side = "WAIT"

    price = float(x.close)
    volatility = float(x.atr)
    if not np.isfinite(volatility) or volatility <= 0:
        raise ValueError("ATR could not be calculated from the returned candles.")

    if side == "BUY":
        swing = float(a["low"].tail(8).min())
        sl = min(swing, price - 1.35 * volatility)
        risk = max(price - sl, 0.65 * volatility)
        sl = price - risk
        entry_low, entry_high = price - 0.12 * volatility, price + 0.12 * volatility
        direction = 1
    elif side == "SELL":
        swing = float(a["high"].tail(8).max())
        sl = max(swing, price + 1.35 * volatility)
        risk = max(sl - price, 0.65 * volatility)
        sl = price + risk
        entry_low, entry_high = price - 0.12 * volatility, price + 0.12 * volatility
        direction = -1
    else:
        sl = risk = entry_low = entry_high = np.nan
        direction = 0

    if side != "WAIT":
        tps = [price + direction * risk * rr for rr in (1.42, 2.24, 3.0)]
        rr_text = f"1:{2.24:.2f}"
        invalidation = sl
        grade = "A" if score >= 90 else "B" if score >= 80 else "C"
        risk_label = "LOW RISK" if score >= 90 else "MODERATE RISK" if score >= 80 else "HIGHER RISK"
        reason = "Trend conditions aligned" if score >= 85 else "Trend setup confirmed"
    else:
        tps = [np.nan, np.nan, np.nan]
        rr_text, invalidation, grade, risk_label = "—", np.nan, "—", "NO TRADE"
        reason = "WAIT FOR A VALID TREND SETUP"

    return {
        "side": side, "score": score, "grade": grade, "risk_label": risk_label,
        "session": session, "price": price, "entry_low": entry_low,
        "entry_high": entry_high, "sl": sl, "tps": tps, "rr": rr_text,
        "invalidation": invalidation, "reason": reason, "checks": checks,
        "rsi": float(x.rsi), "adx": float(x.adx), "atr": volatility,
        "trend_1h": "BULLISH" if one.ema20 > one.ema50 else "BEARISH",
        "trend_4h": "BULLISH" if four.ema20 > four.ema50 else "BEARISH",
        "ema20_15m": float(x.ema20),
        "candle_time": a.index[-1],
    }

def get_candles(api_key, interval, outputsize=500):
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol":"XAU/USD", "interval":interval, "outputsize":outputsize,
              "apikey":api_key, "format":"JSON", "timezone":"UTC"}
    response = requests.get(url, params=params, timeout=25)
    response.raise_for_status()
    payload = response.json()
    if "values" not in payload:
        raise RuntimeError(payload.get("message", payload.get("code", "No candle data returned")))
    df = pd.DataFrame(payload["values"])
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime", "open", "high", "low", "close"])
    return df.sort_values("datetime").set_index("datetime")


def send_telegram(token, chat_id, message):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    response = requests.post(url, data={"chat_id":chat_id, "text":message,
        "disable_web_page_preview":True}, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("ok"):
        raise RuntimeError("Telegram rejected the message")


def fmt(value):
    return "—" if pd.isna(value) else f"{value:,.2f}"


def main():
    api_key = os.environ["TWELVE_DATA_API_KEY"].strip()
    token = os.environ["TELEGRAM_BOT_TOKEN"].strip()
    chat_id = os.environ["TELEGRAM_CHAT_ID"].strip()
    min_score = int(os.getenv("MIN_SCORE", "40"))
    min_adx = float(os.getenv("MIN_ADX", "16"))
    use_session = os.getenv("USE_SESSION_FILTER", "true").lower() == "true"
    state_path = Path(".afg_gold_scanner_state.json")
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    m15 = get_candles(api_key, "15min", 500)
    h1 = get_candles(api_key, "1h", 300)
    h4 = get_candles(api_key, "4h", 300)
    signal = build_signal(m15, h1, h4, min_adx, min_score, use_session)
    if signal["side"] not in ("BUY", "SELL"):
        print(f"No trade. Score={signal['score']}; reason={signal['reason']}")
        return

    candle_time = signal["candle_time"].isoformat()
    fingerprint = f"XAUUSD|{signal['side']}|{candle_time}"
    if state.get("last_fingerprint") == fingerprint:
        print("Same signal/candle already notified; skipping duplicate.")
        return

    side = signal["side"]
    price, sl, tps = signal["price"], signal["sl"], signal["tps"]
    valid_minutes = int(os.getenv("SIGNAL_VALID_MINUTES", "45"))
    common = (f"🇦🇫 AFG GOLD — XAU/USD\nDirection: {side}\n"
              f"Stop Loss: {fmt(sl)}\nTP1: {fmt(tps[0])}\n"
              f"TP2: {fmt(tps[1])}\nTP3: {fmt(tps[2])}\n"
              f"Score: {signal['score']}/100 | Grade: {signal['grade']}\n"
              f"Session: {signal['session']} | R:R {signal['rr']}\n"
              f"Validity: {valid_minutes} minutes\n"
              "Not financial advice. Verify price and risk before trading.")
    send_telegram(token, chat_id, f"🇦🇫 AFG GOLD — MARKET ENTRY\nEntry: {fmt(price)}\n" + common)

    # Conditional pullback idea: EMA20 must be on the retracement side of price.
    limit_entry = signal["ema20_15m"]
    has_limit = limit_entry < price if side == "BUY" else limit_entry > price
    if has_limit:
        risk_distance = abs(price - sl)
        limit_sl = limit_entry - risk_distance if side == "BUY" else limit_entry + risk_distance
        direction = 1 if side == "BUY" else -1
        limit_tps = [limit_entry + direction * risk_distance * rr for rr in (1.42, 2.24, 3.0)]
        limit_msg = (f"🇦🇫 AFG GOLD — LIMIT ENTRY IDEA\nEntry: {fmt(limit_entry)}\n"
                     f"Stop Loss: {fmt(limit_sl)}\nTP1: {fmt(limit_tps[0])}\n"
                     f"TP2: {fmt(limit_tps[1])}\nTP3: {fmt(limit_tps[2])}\n"
                     f"Score: {signal['score']}/100 | Grade: {signal['grade']}\n"
                     f"Session: {signal['session']}\n"
                     "Pullback to 15M EMA20 idea only; no broker order is placed.\n"
                     "Not financial advice. Verify price and risk before trading.")
        send_telegram(token, chat_id, limit_msg)
    else:
        print("No valid EMA20 pullback limit price; market alert sent only.")

    state_path.write_text(json.dumps({"last_fingerprint":fingerprint,
        "last_notified_at":datetime.now(timezone.utc).isoformat()}, indent=2))
    print(f"Alerted {side} on candle {candle_time}; score={signal['score']}.")

if __name__ == "__main__":
    main()
