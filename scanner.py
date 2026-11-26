"""
AFG GOLD standalone scanner for GitHub Actions.
Uses the same 7-filter Rule-Based Trend Following logic as the Streamlit app.
Telegram alerts are sent for XAU/USD only. BTC/USDT is scanned for diagnostics only.
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

API_URL = "https://api.twelvedata.com/time_series"
STATE_FILE = Path(".afg_gold_scanner_state.json")
MIN_SCORE = 40
MIN_ADX = 16
USE_SESSION_FILTER = True
OUTPUT_SIZE = 500
TIMEOUT = 25


def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()


def atr(df, n=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def rsi(series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def adx(df, n=14):
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    average_range = atr(df, n).replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / average_range
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / average_range
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def add_indicators(df):
    df = df.copy()
    for n in (20, 50, 200):
        df[f"ema{n}"] = ema(df["close"], n)
    df["rsi"] = rsi(df["close"])
    df["atr"] = atr(df)
    df["adx"] = adx(df)
    return df.dropna().copy()


def fetch_candles(api_key, symbol, interval):
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": OUTPUT_SIZE,
        "apikey": api_key,
        "format": "JSON",
        "timezone": "UTC",
    }
    response = requests.get(API_URL, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    if "values" not in payload:
        message = payload.get("message") or payload.get("code") or str(payload)
        raise RuntimeError(f"Twelve Data {symbol} {interval}: {message}")
    df = pd.DataFrame(payload["values"])
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime", "open", "high", "low", "close"])
    return df.sort_values("datetime").set_index("datetime")


def market_session(now_utc):
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


def build_signal(m15, h1, h4):
    a, b, c = add_indicators(m15), add_indicators(h1), add_indicators(h4)
    if min(len(a), len(b), len(c)) < 5:
        raise ValueError("Not enough historical candles for indicator calculations.")

    x, previous = a.iloc[-1], a.iloc[-2]
    one, four = b.iloc[-1], c.iloc[-1]
    session, in_session = market_session(datetime.now(timezone.utc))

    long_checks = {
        "1H bullish trend": one.ema20 > one.ema50 and one.close > one.ema200,
        "4H bullish trend": four.ema20 > four.ema50,
        "15M EMA alignment": x.ema20 > x.ema50,
        "ADX trend strength": x.adx >= MIN_ADX,
        "RSI bullish momentum": 50 <= x.rsi <= 72,
        "Bullish candle confirmation": x.close > x.open and x.close > previous.high,
        "Price above 15M EMA20": x.close > x.ema20,
    }
    short_checks = {
        "1H bearish trend": one.ema20 < one.ema50 and one.close < one.ema200,
        "4H bearish trend": four.ema20 < four.ema50,
        "15M EMA alignment": x.ema20 < x.ema50,
        "ADX trend strength": x.adx >= MIN_ADX,
        "RSI bearish momentum": 28 <= x.rsi <= 50,
        "Bearish candle confirmation": x.close < x.open and x.close < previous.low,
        "Price below 15M EMA20": x.close < x.ema20,
    }

    long_score = sum(bool(v) for v in long_checks.values()) / 7 * 100
    short_score = sum(bool(v) for v in short_checks.values()) / 7 * 100
    side = "BUY" if long_score > short_score else "SELL" if short_score > long_score else "WAIT"
    checks = long_checks if side == "BUY" else short_checks if side == "SELL" else {}
    score = int(round(max(long_score, short_score))) if side != "WAIT" else 0

    if USE_SESSION_FILTER and not in_session:
        side = "WAIT"
    if score <= MIN_SCORE or not checks:
        side = "WAIT"

    price = float(x.close)
    volatility = float(x.atr)
    if not np.isfinite(volatility) or volatility <= 0:
        raise ValueError("ATR could not be calculated.")

    if side == "BUY":
        swing = float(a["low"].tail(8).min())
        risk = max(price - min(swing, price - 1.35 * volatility), 0.65 * volatility)
        sl = price - risk
        direction = 1
    elif side == "SELL":
        swing = float(a["high"].tail(8).max())
        risk = max(max(swing, price + 1.35 * volatility) - price, 0.65 * volatility)
        sl = price + risk
        direction = -1
    else:
        return {
            "side": "WAIT", "score": score, "session": session,
            "price": price, "candle_time": a.index[-1].isoformat(),
        }

    tps = [price + direction * risk * rr for rr in (1.42, 2.24, 3.0)]
    grade = "A+" if score >= 95 else "A" if score >= 85 else "B" if score >= 70 else "C"

    # A limit idea is included only when EMA20 is a nearby pullback level.
    ema20 = float(x.ema20)
    distance = abs(price - ema20)
    limit_entry = None
    if distance <= 0.50 * volatility and (
        (side == "BUY" and ema20 < price) or (side == "SELL" and ema20 > price)
    ):
        limit_entry = ema20

    return {
        "side": side,
        "score": score,
        "grade": grade,
        "session": session,
        "price": price,
        "sl": sl,
        "tps": tps,
        "rr": "1:2.24",
        "limit_entry": limit_entry,
        "candle_time": a.index[-1].isoformat(),
        "rsi": float(x.rsi),
        "adx": float(x.adx),
    }


def fmt(value):
    return f"{value:,.2f}"


def build_telegram_message(signal):
    direction = 1 if signal["side"] == "BUY" else -1
    lines = [
        "🇦🇫 AFG GOLD — SIGNAL",
        "Symbol: XAU/USD",
        f"Direction: {signal['side']}",
        "",
        "MARKET ENTRY",
        f"Entry: {fmt(signal['price'])}",
        f"Stop Loss: {fmt(signal['sl'])}",
        f"TP1: {fmt(signal['tps'][0])}",
        f"TP2: {fmt(signal['tps'][1])}",
        f"TP3: {fmt(signal['tps'][2])}",
    ]
    if signal["limit_entry"] is not None:
        limit = signal["limit_entry"]
        risk = abs(limit - signal["sl"])
        limit_tps = [limit + direction * risk * rr for rr in (1.42, 2.24, 3.0)]
        lines += [
            "",
            "LIMIT ENTRY IDEA (pullback)",
            f"Entry: {fmt(limit)}",
            f"Stop Loss: {fmt(signal['sl'])}",
            f"TP1: {fmt(limit_tps[0])}",
            f"TP2: {fmt(limit_tps[1])}",
            f"TP3: {fmt(limit_tps[2])}",
        ]
    else:
        lines += ["", "LIMIT ENTRY: No valid pullback setup"]
    lines += [
        "",
        f"Score: {signal['score']}/100 | Grade: {signal['grade']}",
        f"Session: {signal['session']} | R:R (Market): {signal['rr']}",
        f"RSI: {signal['rsi']:.1f} | ADX: {signal['adx']:.1f}",
        "Decision support only. Verify price and risk before trading.",
    ]
    return "\n".join(lines)


def send_telegram(message):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID GitHub secret.")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    response = requests.post(
        url,
        data={"chat_id": chat_id, "text": message, "disable_web_page_preview": True},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("ok"):
        raise RuntimeError(f"Telegram rejected the message: {payload}")
    return True


def read_state():
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def scan_symbol(api_key, symbol):
    m15 = fetch_candles(api_key, symbol, "15min")
    h1 = fetch_candles(api_key, symbol, "1h")
    h4 = fetch_candles(api_key, symbol, "4h")
    return build_signal(m15, h1, h4)


def main():
    api_key = os.getenv("TWELVE_DATA_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing TWELVE_DATA_API_KEY GitHub secret.")

    # Gold is scanned first so BTC diagnostics cannot prevent the gold alert.
    gold = scan_symbol(api_key, "XAU/USD")
    print(
        f"XAU/USD: side={gold['side']} score={gold['score']} "
        f"grade={gold.get('grade', '-')} candle={gold['candle_time']}"
    )

    # BTC is scanned for diagnostics only. No BTC Telegram alert is ever sent.
    try:
        btc = scan_symbol(api_key, "BTC/USDT")
        print(
            f"BTC/USDT diagnostic: side={btc['side']} "
            f"score={btc['score']} grade={btc.get('grade', '-')}"
        )
    except Exception as exc:
        print(f"BTC/USDT diagnostic scan skipped: {exc}")

    if gold["side"] == "WAIT":
        print("No qualifying XAU/USD signal. No Telegram message sent.")
        return

    state = read_state()
    candle_key = gold["candle_time"]
    if state.get("last_gold_candle") == candle_key:
        print("This XAU/USD candle was already notified. Duplicate blocked.")
        return

    message = build_telegram_message(gold)
    send_telegram(message)
    state["last_gold_candle"] = candle_key
    write_state(state)
    print("XAU/USD Telegram alert sent and state saved.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"SCANNER ERROR: {exc}", file=sys.stderr)
        raise
