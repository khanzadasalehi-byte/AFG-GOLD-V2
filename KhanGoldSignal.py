
# ============================================================
# KHAN GOLD SIGNAL
# ADVANCED ADAPTIVE RULE-BASED MULTI-TIMEFRAME TREND FOLLOWING
# PART 1: CORE / MARKET DATA / INDICATORS / SESSIONS
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import hashlib
import json
import os

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# ============================================================
# APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="KHAN GOLD SIGNAL",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_NAME = "KHAN GOLD SIGNAL"
APP_VERSION = "1.0.0"

UTC = timezone.utc

ASSETS = {
    "GOLD": "XAU/USD",
    "BITCOIN": "BTC/USD"
}

TIMEFRAMES = {
    "1M": "1min",
    "5M": "5min",
    "15M": "15min",
    "1H": "1h",
    "4H": "4h",
    "1D": "1day"
}

TIMEFRAME_MINUTES = {
    "1min": 1,
    "5min": 5,
    "15min": 15,
    "1h": 60,
    "4h": 240,
    "1day": 1440
}

# ============================================================
# SECRET MANAGEMENT
# ============================================================

def get_secret(name, default=""):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return os.getenv(name, default)


TWELVE_DATA_API_KEY = get_secret(
    "TWELVE_DATA_API_KEY"
)

TELEGRAM_BOT_TOKEN = get_secret(
    "TELEGRAM_BOT_TOKEN"
)

TELEGRAM_CHAT_ID = get_secret(
    "TELEGRAM_CHAT_ID"
)

SUPABASE_URL = get_secret(
    "SUPABASE_URL"
)

SUPABASE_KEY = get_secret(
    "SUPABASE_KEY"
)

# ============================================================
# GENERAL HELPERS
# ============================================================

def utc_now():
    return datetime.now(UTC)


def safe_float(value, default=np.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def timeframe_delta(interval):
    minutes = TIMEFRAME_MINUTES.get(interval)

    if minutes is None:
        raise ValueError("Unsupported timeframe")

    return timedelta(minutes=minutes)


# ============================================================
# MARKET DATA ENGINE
# ============================================================

@st.cache_data(ttl=15, show_spinner=False)
def fetch_market_data(symbol, interval, outputsize=500):

    if not TWELVE_DATA_API_KEY:
        return pd.DataFrame(), "Twelve Data API key is missing"

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_DATA_API_KEY,
        "timezone": "UTC",
        "format": "JSON"
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=20
        )

        response.raise_for_status()

        result = response.json()

        if "values" not in result:
            return pd.DataFrame(), result.get(
                "message",
                "Market data unavailable"
            )

        df = pd.DataFrame(result["values"])

        required = [
            "datetime",
            "open",
            "high",
            "low",
            "close"
        ]

        if not all(col in df.columns for col in required):
            return pd.DataFrame(), "Invalid market data"

        df["datetime"] = pd.to_datetime(
            df["datetime"],
            utc=True,
            errors="coerce"
        )

        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(
                df["volume"],
                errors="coerce"
            )
        else:
            df["volume"] = np.nan

        df = df.dropna(
            subset=[
                "datetime",
                "open",
                "high",
                "low",
                "close"
            ]
        )

        df = df.sort_values(
            "datetime"
        ).drop_duplicates(
            subset=["datetime"],
            keep="last"
        ).reset_index(drop=True)

        # Remove incomplete candle.
        now = utc_now()

        candle_end = (
            df["datetime"] +
            timeframe_delta(interval)
        )

        df = df.loc[
            candle_end <= now
        ].reset_index(drop=True)

        return df, None

    except requests.RequestException as e:
        return pd.DataFrame(), str(e)

    except Exception as e:
        return pd.DataFrame(), str(e)


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def add_indicators(df):

    if df.empty:
        return df.copy()

    data = df.copy()

    # EMA
    data["EMA20"] = data["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    data["EMA50"] = data["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    data["EMA200"] = data["close"].ewm(
        span=200,
        adjust=False
    ).mean()

    # RSI
    delta = data["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    data["RSI"] = 100 - (
        100 / (1 + rs)
    )

    # ATR
    previous_close = data["close"].shift(1)

    tr = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    data["ATR"] = tr.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    # ADX
    up_move = data["high"].diff()
    down_move = -data["low"].diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0.0
        ),
        index=data.index
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0.0
        ),
        index=data.index
    )

    atr_safe = data["ATR"].replace(0, np.nan)

    plus_di = (
        100 * plus_dm.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean() / atr_safe
    )

    minus_di = (
        100 * minus_dm.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean() / atr_safe
    )

    dx = (
        100 *
        (plus_di - minus_di).abs() /
        (plus_di + minus_di).replace(0, np.nan)
    )

    data["PLUS_DI"] = plus_di
    data["MINUS_DI"] = minus_di

    data["ADX"] = dx.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    return data


# ============================================================
# MARKET SESSION ENGINE
# ============================================================

def market_session():

    now = utc_now()

    london = now.astimezone(
        ZoneInfo("Europe/London")
    )

    new_york = now.astimezone(
        ZoneInfo("America/New_York")
    )

    asia = now.astimezone(
        ZoneInfo("Asia/Tokyo")
    )

    # Approximate active session windows.
    london_active = (
        london.weekday() < 5
        and 8 <= london.hour < 17
    )

    new_york_active = (
        new_york.weekday() < 5
        and 8 <= new_york.hour < 17
    )

    asia_active = (
        asia.weekday() < 5
        and 9 <= asia.hour < 18
    )

    if london_active and new_york_active:
        session = "LONDON / NY OVERLAP"

    elif london_active:
        session = "LONDON"

    elif new_york_active:
        session = "NEW YORK"

    elif asia_active:
        session = "ASIAN"

    else:
        session = "LOW LIQUIDITY"

    return {
        "session": session,
        "utc_time": now.isoformat(),
        "london_time": london.isoformat(),
        "new_york_time": new_york.isoformat(),
        "asia_time": asia.isoformat()
    }


# ============================================================
# DATA VALIDATION
# ============================================================

def market_data_is_valid(df):

    if df.empty or len(df) < 220:
        return False

    latest = df.iloc[-1]

    required = [
        "close",
        "EMA20",
        "EMA50",
        "EMA200",
        "RSI",
        "ATR",
        "ADX"
    ]

    for column in required:
        if pd.isna(latest[column]):
            return False

    if latest["close"] <= 0:
        return False

    if latest["ATR"] <= 0:
        return False

    return True


# ============================================================
# END OF PART 1
# ============================================================# ============================================================
# PART 2 — TREND, MARKET STRUCTURE & MULTI-TIMEFRAME ANALYSIS
# ============================================================

def _find_column(df, candidates):
    """Find a column without depending on upper/lower-case naming."""
    lookup = {str(c).lower(): c for c in df.columns}

    for name in candidates:
        if name.lower() in lookup:
            return lookup[name.lower()]

    return None


def _last_value(df, candidates, default=np.nan):
    col = _find_column(df, candidates)

    if col is None or df.empty:
        return default

    values = pd.to_numeric(df[col], errors="coerce").dropna()

    if values.empty:
        return default

    return float(values.iloc[-1])


def detect_swings(df, left=3, right=3):
    """
    Detect confirmed swing highs/lows.

    A pivot is recorded on its confirmation candle, not on the
    original pivot candle. This prevents future-data leakage.
    """
    if df is None or len(df) < left + right + 10:
        return pd.DataFrame() if df is None else df.copy()

    result = df.copy()

    high_col = _find_column(result, ["high", "High"])
    low_col = _find_column(result, ["low", "Low"])

    if high_col is None or low_col is None:
        return result

    highs = pd.to_numeric(result[high_col], errors="coerce").to_numpy()
    lows = pd.to_numeric(result[low_col], errors="coerce").to_numpy()

    confirmed_high = np.full(len(result), np.nan)
    confirmed_low = np.full(len(result), np.nan)

    for pivot_idx in range(left, len(result) - right):
        confirm_idx = pivot_idx + right

        left_highs = highs[pivot_idx-left:pivot_idx]
        right_highs = highs[pivot_idx+1:pivot_idx+right+1]

        left_lows = lows[pivot_idx-left:pivot_idx]
        right_lows = lows[pivot_idx+1:pivot_idx+right+1]

        pivot_high = highs[pivot_idx]
        pivot_low = lows[pivot_idx]

        if (
            np.isfinite(pivot_high)
            and np.all(np.isfinite(left_highs))
            and np.all(np.isfinite(right_highs))
            and pivot_high > np.max(left_highs)
            and pivot_high >= np.max(right_highs)
        ):
            confirmed_high[confirm_idx] = pivot_high

        if (
            np.isfinite(pivot_low)
            and np.all(np.isfinite(left_lows))
            and np.all(np.isfinite(right_lows))
            and pivot_low < np.min(left_lows)
            and pivot_low <= np.min(right_lows)
        ):
            confirmed_low[confirm_idx] = pivot_low

    result["Confirmed_Swing_High"] = confirmed_high
    result["Confirmed_Swing_Low"] = confirmed_low

    return result


def get_trend_state(df):
    """Evaluate trend using EMA alignment, ADX, RSI and EMA slope."""

    neutral = {
        "direction": "NEUTRAL",
        "strength": 0,
        "score": 0,
        "ema_alignment": "MIXED",
        "adx": None,
        "rsi": None,
        "price": None,
        "reasons": ["Insufficient or unavailable market data"]
    }

    if df is None or len(df) < 30:
        return neutral

    close_col = _find_column(df, ["close", "Close"])

    if close_col is None:
        return neutral

    close_series = pd.to_numeric(df[close_col], errors="coerce").dropna()

    if len(close_series) < 30:
        return neutral

    price = float(close_series.iloc[-1])

    ema20 = _last_value(df, ["EMA20", "EMA_20", "ema20"])
    ema50 = _last_value(df, ["EMA50", "EMA_50", "ema50"])
    ema200 = _last_value(df, ["EMA200", "EMA_200", "ema200"])

    adx = _last_value(df, ["ADX", "adx"])
    rsi = _last_value(df, ["RSI", "rsi"])

    if not all(np.isfinite(x) for x in [ema20, ema50, ema200]):
        return neutral

    score = 0
    reasons = []

    # EMA alignment
    bullish_alignment = ema20 > ema50 > ema200
    bearish_alignment = ema20 < ema50 < ema200

    if bullish_alignment:
        score += 35
        reasons.append("Bullish EMA alignment")
        alignment = "BULLISH"
    elif bearish_alignment:
        score -= 35
        reasons.append("Bearish EMA alignment")
        alignment = "BEARISH"
    else:
        alignment = "MIXED"
        reasons.append("EMA alignment is mixed")

    # Price relative to EMA20 and EMA50
    if price > ema20 and price > ema50:
        score += 15
        reasons.append("Price above EMA20 and EMA50")
    elif price < ema20 and price < ema50:
        score -= 15
        reasons.append("Price below EMA20 and EMA50")

    # EMA20 slope over recent candles
    ema20_col = _find_column(df, ["EMA20", "EMA_20", "ema20"])

    if ema20_col is not None and len(df) >= 6:
        ema_values = pd.to_numeric(
            df[ema20_col], errors="coerce"
        ).dropna()

        if len(ema_values) >= 6:
            old_ema = float(ema_values.iloc[-6])
            new_ema = float(ema_values.iloc[-1])

            if new_ema > old_ema:
                score += 10
                reasons.append("EMA20 slope is positive")
            elif new_ema < old_ema:
                score -= 10
                reasons.append("EMA20 slope is negative")

    # ADX measures trend strength, not direction
    if np.isfinite(adx):
        if adx >= 25:
            strength = "STRONG"
            reasons.append("ADX confirms stronger trend conditions")
        elif adx >= 18:
            strength = "MODERATE"
            reasons.append("ADX indicates moderate trend conditions")
        else:
            strength = "WEAK"
            reasons.append("ADX indicates weak or ranging conditions")
    else:
        strength = "UNKNOWN"

    # RSI confirmation
    if np.isfinite(rsi):
        if 52 <= rsi <= 70:
            score += 10
            reasons.append("RSI supports bullish momentum")
        elif 30 <= rsi <= 48:
            score -= 10
            reasons.append("RSI supports bearish momentum")
        elif rsi > 75:
            reasons.append("RSI is elevated; bullish entry may be extended")
        elif rsi < 25:
            reasons.append("RSI is very low; bearish entry may be extended")

    score = int(max(-100, min(100, score)))

    if score >= 25:
        direction = "BULLISH"
    elif score <= -25:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"

    return {
        "direction": direction,
        "strength": strength,
        "score": score,
        "ema_alignment": alignment,
        "adx": round(adx, 2) if np.isfinite(adx) else None,
        "rsi": round(rsi, 2) if np.isfinite(rsi) else None,
        "price": price,
        "reasons": reasons
    }


def detect_bos_mss(df):
    """
    Detect close-based breaks of the latest confirmed swing levels.

    BOS: break in the direction of the existing trend.
    MSS: break against the previous trend direction.
    """
    result = {
        "event": "NONE",
        "direction": "NEUTRAL",
        "broken_level": None,
        "message": "No confirmed structure break"
    }

    if df is None or len(df) < 20:
        return result

    working = detect_swings(df)

    if working.empty:
        return result

    close_col = _find_column(working, ["close", "Close"])

    if close_col is None:
        return result

    close = pd.to_numeric(
        working[close_col], errors="coerce"
    ).dropna()

    if close.empty:
        return result

    current_close = float(close.iloc[-1])

    high_col = "Confirmed_Swing_High"
    low_col = "Confirmed_Swing_Low"

    if high_col not in working.columns or low_col not in working.columns:
        return result

    previous_highs = pd.to_numeric(
        working[high_col], errors="coerce"
    ).dropna()

    previous_lows = pd.to_numeric(
        working[low_col], errors="coerce"
    ).dropna()

    # Exclude a level confirmed on the latest candle itself.
    previous_highs = previous_highs.iloc[:-1]
    previous_lows = previous_lows.iloc[:-1]

    swing_high = (
        float(previous_highs.iloc[-1])
        if not previous_highs.empty else None
    )

    swing_low = (
        float(previous_lows.iloc[-1])
        if not previous_lows.empty else None
    )

    trend = get_trend_state(working)
    trend_direction = trend["direction"]

    if swing_high is not None and current_close > swing_high:
        if trend_direction == "BEARISH":
            event = "MSS"
            message = "Bullish market structure shift"
        else:
            event = "BOS"
            message = "Bullish break of structure"

        return {
            "event": event,
            "direction": "BULLISH",
            "broken_level": swing_high,
            "message": message
        }

    if swing_low is not None and current_close < swing_low:
        if trend_direction == "BULLISH":
            event = "MSS"
            message = "Bearish market structure shift"
        else:
            event = "BOS"
            message = "Bearish break of structure"

        return {
            "event": event,
            "direction": "BEARISH",
            "broken_level": swing_low,
            "message": message
        }

    return result


def analyze_market_structure(df):
    """Combine trend, confirmed swings and BOS/MSS information."""

    if df is None or df.empty:
        return {
            "valid": False,
            "trend": get_trend_state(df),
            "structure": detect_bos_mss(df),
            "last_swing_high": None,
            "last_swing_low": None
        }

    working = detect_swings(df)

    swing_highs = (
        pd.to_numeric(
            working.get("Confirmed_Swing_High", pd.Series(dtype=float)),
            errors="coerce"
        ).dropna()
    )

    swing_lows = (
        pd.to_numeric(
            working.get("Confirmed_Swing_Low", pd.Series(dtype=float)),
            errors="coerce"
        ).dropna()
    )

    return {
        "valid": True,
        "trend": get_trend_state(working),
        "structure": detect_bos_mss(working),
        "last_swing_high": (
            float(swing_highs.iloc[-1])
            if not swing_highs.empty else None
        ),
        "last_swing_low": (
            float(swing_lows.iloc[-1])
            if not swing_lows.empty else None
        )
    }


def analyze_multi_timeframe(frames):
    """
    Analyze already-fetched DataFrames.

    Example:
    frames = {
        "5min": df_5m,
        "15min": df_15m,
        "1h": df_1h,
        "4h": df_4h
    }

    This function does not fetch data itself. It uses the data
    supplied by the main application.
    """

    if not isinstance(frames, dict) or not frames:
        return {
            "direction": "NEUTRAL",
            "score": 0,
            "agreement": 0,
            "timeframes": {},
            "reasons": ["No timeframe data supplied"]
        }

    weights = {
        "1min": 1,
        "5min": 1,
        "15min": 2,
        "30min": 2,
        "1h": 3,
        "4h": 4,
        "1d": 5,
        "1day": 5
    }

    results = {}
    weighted_total = 0
    total_weight = 0
    bullish_count = 0
    bearish_count = 0
    usable_count = 0

    for timeframe, frame in frames.items():
        if frame is None or frame.empty:
            continue

        analysis = analyze_market_structure(frame)
        trend = analysis["trend"]

        direction = trend["direction"]
        score = trend["score"]

        weight = weights.get(str(timeframe).lower(), 1)

        results[str(timeframe)] = analysis

        weighted_total += score * weight
        total_weight += weight
        usable_count += 1

        if direction == "BULLISH":
            bullish_count += 1
        elif direction == "BEARISH":
            bearish_count += 1

    if total_weight == 0:
        return {
            "direction": "NEUTRAL",
            "score": 0,
            "agreement": 0,
            "timeframes": results,
            "reasons": ["No valid timeframe data"]
        }

    final_score = int(round(weighted_total / total_weight))

    if final_score >= 25:
        final_direction = "BULLISH"
    elif final_score <= -25:
        final_direction = "BEARISH"
    else:
        final_direction = "NEUTRAL"

    matching = (
        bullish_count if final_direction == "BULLISH"
        else bearish_count if final_direction == "BEARISH"
        else 0
    )

    agreement = round(
        matching / usable_count * 100
    ) if usable_count else 0

    reasons = [
        f"Analyzed {usable_count} available timeframes",
        f"Timeframe directional agreement: {agreement}%"
    ]

    if agreement < 50:
        reasons.append("Timeframes are not sufficiently aligned")

    return {
        "direction": final_direction,
        "score": final_score,
        "agreement": agreement,
        "timeframes": results,
        "reasons": reasons
    }


# END OF PART 2# ============================================================
# PART 3 — ADVANCED PRECISION ENTRY ENGINE
# ============================================================

def entry_config(profile, asset):
    """Return independent entry configuration for each profile."""

    configs = {
        "Scalping": {
            "max_market_distance_atr": 0.35,
            "limit_zone_atr": 0.18,
            "min_pullback_atr": 0.08,
            "max_entry_distance_atr": 1.20,
        },
        "Intraday": {
            "max_market_distance_atr": 0.45,
            "limit_zone_atr": 0.22,
            "min_pullback_atr": 0.10,
            "max_entry_distance_atr": 1.60,
        },
        "Swing": {
            "max_market_distance_atr": 0.55,
            "limit_zone_atr": 0.28,
            "min_pullback_atr": 0.12,
            "max_entry_distance_atr": 2.00,
        }
    }

    config = configs.get(profile, configs["Intraday"]).copy()

    # BTC and Gold have different volatility characteristics.
    if asset == "BTC/USD":
        config["limit_zone_atr"] *= 1.15
        config["max_entry_distance_atr"] *= 1.10

    return config


def calculate_fibonacci_levels(df, direction):
    """Calculate retracement levels from recent confirmed swing points."""

    result = {
        "valid": False,
        "fib_50": None,
        "fib_618": None,
        "swing_high": None,
        "swing_low": None
    }

    if df is None or len(df) < 20:
        return result

    working = detect_swings(df)

    highs = working["Confirmed_Swing_High"].dropna()
    lows = working["Confirmed_Swing_Low"].dropna()

    if highs.empty or lows.empty:
        return result

    swing_high = float(highs.iloc[-1])
    swing_low = float(lows.iloc[-1])

    if swing_high <= swing_low:
        return result

    move = swing_high - swing_low

    if direction == "BULLISH":
        fib_50 = swing_high - move * 0.50
        fib_618 = swing_high - move * 0.618

    elif direction == "BEARISH":
        fib_50 = swing_low + move * 0.50
        fib_618 = swing_low + move * 0.618

    else:
        return result

    return {
        "valid": True,
        "fib_50": float(fib_50),
        "fib_618": float(fib_618),
        "swing_high": swing_high,
        "swing_low": swing_low
    }


def calculate_entry_confluence(df, direction):
    """
    Find a possible pullback zone using EMA, Fibonacci and
    confirmed swing levels.
    """

    result = {
        "valid": False,
        "zone_low": None,
        "zone_high": None,
        "confluence_score": 0,
        "reasons": []
    }

    if df is None or len(df) < 30:
        result["reasons"].append("Insufficient market history")
        return result

    close = _last_value(df, ["close", "Close"])
    ema20 = _last_value(df, ["EMA20", "EMA_20"])
    ema50 = _last_value(df, ["EMA50", "EMA_50"])
    atr = _last_value(df, ["ATR", "atr"])

    if not all(np.isfinite(x) for x in [close, ema20, ema50, atr]):
        result["reasons"].append("Required indicators are unavailable")
        return result

    if atr <= 0:
        result["reasons"].append("Invalid ATR")
        return result

    fib = calculate_fibonacci_levels(df, direction)

    if not fib["valid"]:
        result["reasons"].append("No confirmed Fibonacci swing structure")
        return result

    levels = [
        ("EMA20", ema20),
        ("EMA50", ema50),
        ("Fib 50", fib["fib_50"]),
        ("Fib 61.8", fib["fib_618"])
    ]

    if direction == "BULLISH":
        relevant = [
            (name, value) for name, value in levels
            if value < close and close - value <= atr * 1.60
        ]

    elif direction == "BEARISH":
        relevant = [
            (name, value) for name, value in levels
            if value > close and value - close <= atr * 1.60
        ]

    else:
        result["reasons"].append("Neutral trend")
        return result

    if len(relevant) < 2:
        result["reasons"].append("Insufficient pullback confluence")
        return result

    values = [value for _, value in relevant]

    zone_low = min(values)
    zone_high = max(values)

    # Reject zones that are too wide.
    max_zone_width = atr * 0.85

    if zone_high - zone_low > max_zone_width:
        center = float(np.mean(values))
        half_width = max_zone_width / 2
        zone_low = center - half_width
        zone_high = center + half_width

    score = min(100, len(relevant) * 20)

    result.update({
        "valid": True,
        "zone_low": float(zone_low),
        "zone_high": float(zone_high),
        "confluence_score": score,
        "reasons": [
            f"{name} confluence detected"
            for name, _ in relevant
        ]
    })

    return result


def select_entry_type(df, direction, profile, asset):
    """
    Select MARKET or LIMIT automatically.

    Market entry is considered only when price is close to
    the validated entry area and momentum is confirmed.
    Otherwise a valid pullback may use a Limit zone.
    """

    config = entry_config(profile, asset)

    close = _last_value(df, ["close", "Close"])
    atr = _last_value(df, ["ATR", "atr"])
    rsi = _last_value(df, ["RSI", "rsi"])

    if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
        return {
            "valid": False,
            "entry_type": "WAIT",
            "reason": "Invalid price or ATR data"
        }

    confluence = calculate_entry_confluence(df, direction)

    if not confluence["valid"]:
        return {
            "valid": False,
            "entry_type": "WAIT",
            "reason": "No reliable pullback entry zone",
            "confluence": confluence
        }

    low = confluence["zone_low"]
    high = confluence["zone_high"]

    if direction == "BULLISH":
        distance = max(0, close - high)
        momentum_ok = np.isfinite(rsi) and 50 <= rsi <= 70

    elif direction == "BEARISH":
        distance = max(0, low - close)
        momentum_ok = np.isfinite(rsi) and 30 <= rsi <= 50

    else:
        return {
            "valid": False,
            "entry_type": "WAIT",
            "reason": "Neutral direction"
        }

    distance_atr = distance / atr

    # Market entry: only if price is close to the zone.
    if (
        low <= close <= high
        and momentum_ok
        and distance_atr <= config["max_market_distance_atr"]
    ):
        return {
            "valid": True,
            "entry_type": "MARKET",
            "entry_price": close,
            "zone_low": None,
            "zone_high": None,
            "confluence": confluence,
            "reason": "Price is inside the validated entry zone"
        }

    # Limit entry: zone must be actionable and not too far away.
    if direction == "BULLISH":
        limit_distance = close - high
    else:
        limit_distance = low - close

    if (
        limit_distance >= 0
        and limit_distance <= atr * config["max_entry_distance_atr"]
    ):
        return {
            "valid": True,
            "entry_type": "LIMIT",
            "entry_price": None,
            "zone_low": low,
            "zone_high": high,
            "confluence": confluence,
            "reason": "Validated pullback Limit zone detected"
        }

    return {
        "valid": False,
        "entry_type": "WAIT",
        "reason": "Entry zone is no longer actionable",
        "confluence": confluence
    }


def build_precision_entry(df, direction, profile, asset):
    """Build a complete, fixed entry snapshot."""

    selected = select_entry_type(
        df=df,
        direction=direction,
        profile=profile,
        asset=asset
    )

    if not selected["valid"]:
        return {
            "valid": False,
            "entry_type": "WAIT",
            "reason": selected["reason"]
        }

    result = {
        "valid": True,
        "entry_type": selected["entry_type"],
        "direction": direction,
        "profile": profile,
        "asset": asset,
        "reason": selected["reason"],
        "confluence": selected.get("confluence", {})
    }

    if selected["entry_type"] == "MARKET":
        result["entry_price"] = float(selected["entry_price"])
        result["zone_low"] = None
        result["zone_high"] = None

    else:
        result["entry_price"] = None
        result["zone_low"] = float(selected["zone_low"])
        result["zone_high"] = float(selected["zone_high"])

    return result


# END OF PART 3
# ============================================================
# PART 4 — ADVANCED STOP LOSS & TAKE PROFIT ENGINE
# ============================================================

def risk_config(profile, asset):
    """Independent risk parameters for each asset and profile."""

    configs = {
        "Scalping": {
            "atr_sl_multiplier": 1.25,
            "structure_buffer": 0.12,
            "min_sl_atr": 0.65,
            "max_sl_atr": 2.20,
            "min_rr": 1.00,
            "tp_rr": [1.0, 1.8, 2.6]
        },
        "Intraday": {
            "atr_sl_multiplier": 1.60,
            "structure_buffer": 0.18,
            "min_sl_atr": 0.80,
            "max_sl_atr": 3.00,
            "min_rr": 1.20,
            "tp_rr": [1.0, 2.0, 3.0]
        },
        "Swing": {
            "atr_sl_multiplier": 2.00,
            "structure_buffer": 0.25,
            "min_sl_atr": 1.00,
            "max_sl_atr": 4.00,
            "min_rr": 1.40,
            "tp_rr": [1.2, 2.2, 3.5]
        }
    }

    config = configs.get(profile, configs["Intraday"]).copy()

    if asset == "BTC/USD":
        config["atr_sl_multiplier"] *= 1.10
        config["min_sl_atr"] *= 1.10
        config["max_sl_atr"] *= 1.15

    return config


def get_entry_reference(entry):
    """
    Get a conservative reference price for risk calculation.

    For a Limit zone:
    - Bullish: use the lower edge for conservative risk checks.
    - Bearish: use the upper edge.
    """

    direction = entry.get("direction")
    entry_type = entry.get("entry_type")

    if entry_type == "MARKET":
        price = entry.get("entry_price")

        if price is None:
            return None

        return float(price)

    if entry_type == "LIMIT":
        low = entry.get("zone_low")
        high = entry.get("zone_high")

        if low is None or high is None:
            return None

        if direction == "BULLISH":
            return float(low)

        if direction == "BEARISH":
            return float(high)

    return None


def calculate_structure_stop(df, direction, entry_price, profile, asset):
    """Calculate a structure-aware stop loss with ATR protection."""

    config = risk_config(profile, asset)

    atr = _last_value(df, ["ATR", "atr"])

    if not np.isfinite(atr) or atr <= 0:
        return {
            "valid": False,
            "reason": "ATR is unavailable"
        }

    working = detect_swings(df)

    swing_highs = (
        pd.to_numeric(
            working.get("Confirmed_Swing_High", pd.Series(dtype=float)),
            errors="coerce"
        ).dropna()
    )

    swing_lows = (
        pd.to_numeric(
            working.get("Confirmed_Swing_Low", pd.Series(dtype=float)),
            errors="coerce"
        ).dropna()
    )

    buffer = atr * config["structure_buffer"]

    if direction == "BULLISH":

        candidates = []

        if not swing_lows.empty:
            recent_low = float(swing_lows.iloc[-1])

            if recent_low < entry_price:
                candidates.append(recent_low - buffer)

        atr_stop = entry_price - atr * config["atr_sl_multiplier"]

        candidates.append(atr_stop)

        # Choose the closest valid structural protection,
        # while preserving a minimum ATR distance.
        min_distance = atr * config["min_sl_atr"]
        max_distance = atr * config["max_sl_atr"]

        valid_candidates = [
            x for x in candidates
            if min_distance <= entry_price - x <= max_distance
        ]

        if not valid_candidates:
            return {
                "valid": False,
                "reason": "No valid bullish stop-loss structure"
            }

        stop_loss = max(valid_candidates)

    elif direction == "BEARISH":

        candidates = []

        if not swing_highs.empty:
            recent_high = float(swing_highs.iloc[-1])

            if recent_high > entry_price:
                candidates.append(recent_high + buffer)

        atr_stop = entry_price + atr * config["atr_sl_multiplier"]

        candidates.append(atr_stop)

        min_distance = atr * config["min_sl_atr"]
        max_distance = atr * config["max_sl_atr"]

        valid_candidates = [
            x for x in candidates
            if min_distance <= x - entry_price <= max_distance
        ]

        if not valid_candidates:
            return {
                "valid": False,
                "reason": "No valid bearish stop-loss structure"
            }

        stop_loss = min(valid_candidates)

    else:
        return {
            "valid": False,
            "reason": "Direction must be BULLISH or BEARISH"
        }

    return {
        "valid": True,
        "stop_loss": float(stop_loss),
        "risk_distance": float(abs(entry_price - stop_loss)),
        "atr": float(atr),
        "method": "Structure + ATR"
    }


def find_liquidity_targets(df, direction, entry_price, stop_loss):
    """
    Find possible target levels using confirmed swing points.

    These are candidate liquidity levels, not guaranteed
    institutional liquidity pools.
    """

    working = detect_swings(df)

    risk_distance = abs(entry_price - stop_loss)

    if risk_distance <= 0:
        return []

    if direction == "BULLISH":
        levels = (
            pd.to_numeric(
                working.get(
                    "Confirmed_Swing_High",
                    pd.Series(dtype=float)
                ),
                errors="coerce"
            ).dropna()
        )

        targets = [
            float(x) for x in levels
            if x > entry_price
        ]

    else:
        levels = (
            pd.to_numeric(
                working.get(
                    "Confirmed_Swing_Low",
                    pd.Series(dtype=float)
                ),
                errors="coerce"
            ).dropna()
        )

        targets = [
            float(x) for x in levels
            if x < entry_price
        ]

    # Remove duplicate levels and sort by distance from entry.
    targets = sorted(
        set(targets),
        key=lambda x: abs(x - entry_price)
    )

    return targets


def calculate_smart_take_profits(
    df,
    direction,
    entry_price,
    stop_loss,
    profile,
    asset
):
    """Calculate TP1, TP2 and TP3 using RR and market structure."""

    config = risk_config(profile, asset)

    risk_distance = abs(entry_price - stop_loss)

    if risk_distance <= 0:
        return {
            "valid": False,
            "reason": "Invalid risk distance"
        }

    rr_targets = config["tp_rr"]

    liquidity_targets = find_liquidity_targets(
        df,
        direction,
        entry_price,
        stop_loss
    )

    take_profits = []

    for index, rr in enumerate(rr_targets):

        if direction == "BULLISH":
            target = entry_price + risk_distance * rr

            # Prefer a nearby confirmed swing high when it
            # provides at least the minimum required RR.
            candidates = [
                x for x in liquidity_targets
                if x > entry_price
                and x >= entry_price + risk_distance * config["min_rr"]
            ]

            if candidates:
                suitable = [
                    x for x in candidates
                    if x <= target * 1.0005
                ]

                if suitable:
                    target = min(suitable)

        else:
            target = entry_price - risk_distance * rr

            candidates = [
                x for x in liquidity_targets
                if x < entry_price
                and x <= entry_price - risk_distance * config["min_rr"]
            ]

            if candidates:
                suitable = [
                    x for x in candidates
                    if x >= target * 0.9995
                ]

                if suitable:
                    target = max(suitable)

        take_profits.append(float(target))

    # Ensure targets are correctly ordered and separated.
    if direction == "BULLISH":
        take_profits = sorted(set(take_profits))

    else:
        take_profits = sorted(
            set(take_profits),
            reverse=True
        )

    if not take_profits:
        return {
            "valid": False,
            "reason": "No valid take-profit targets"
        }

    return {
        "valid": True,
        "tp1": take_profits[0],
        "tp2": take_profits[1] if len(take_profits) > 1 else None,
        "tp3": take_profits[2] if len(take_profits) > 2 else None,
        "method": "Risk/Reward + Confirmed Swing Liquidity"
    }


def validate_trade_geometry(entry, stop_loss, take_profits):
    """Check that SL and TP are on the correct side of entry."""

    direction = entry.get("direction")
    entry_type = entry.get("entry_type")

    if entry_type == "LIMIT":
        entry_low = entry.get("zone_low")
        entry_high = entry.get("zone_high")

        if entry_low is None or entry_high is None:
            return False, "Invalid Limit entry zone"

        if entry_low >= entry_high:
            return False, "Limit zone boundaries are reversed"

        if direction == "BULLISH" and stop_loss >= entry_low:
            return False, "Bullish SL must be below the full Limit zone"

        if direction == "BEARISH" and stop_loss <= entry_high:
            return False, "Bearish SL must be above the full Limit zone"

        # Validate TP against the conservative entry reference.
        reference = get_entry_reference(entry)

    else:
        reference = get_entry_reference(entry)

    if reference is None:
        return False, "Entry reference unavailable"

    targets = [
        take_profits.get("tp1"),
        take_profits.get("tp2"),
        take_profits.get("tp3")
    ]

    targets = [x for x in targets if x is not None]

    if not targets:
        return False, "No take-profit target available"

    if direction == "BULLISH":

        if stop_loss >= reference:
            return False, "Bullish SL must be below entry"

        if any(x <= reference for x in targets):
            return False, "Bullish TP must be above entry"

        if targets != sorted(targets):
            return False, "Bullish targets are not correctly ordered"

    elif direction == "BEARISH":

        if stop_loss <= reference:
            return False, "Bearish SL must be above entry"

        if any(x >= reference for x in targets):
            return False, "Bearish TP must be below entry"

        if targets != sorted(targets, reverse=True):
            return False, "Bearish targets are not correctly ordered"

    else:
        return False, "Invalid trade direction"

    return True, "Trade geometry valid"


def build_trade_levels(df, entry, profile, asset):
    """
    Build the complete SL/TP package.

    For Limit zones, use the conservative edge for risk
    calculations. The full zone is validated separately.
    """

    if not entry.get("valid"):
        return {
            "valid": False,
            "reason": "Entry is not valid"
        }

    reference = get_entry_reference(entry)

    if reference is None:
        return {
            "valid": False,
            "reason": "Entry reference is unavailable"
        }

    direction = entry["direction"]

    sl_result = calculate_structure_stop(
        df=df,
        direction=direction,
        entry_price=reference,
        profile=profile,
        asset=asset
    )

    if not sl_result["valid"]:
        return {
            "valid": False,
            "reason": sl_result["reason"]
        }

    tp_result = calculate_smart_take_profits(
        df=df,
        direction=direction,
        entry_price=reference,
        stop_loss=sl_result["stop_loss"],
        profile=profile,
        asset=asset
    )

    if not tp_result["valid"]:
        return {
            "valid": False,
            "reason": tp_result["reason"]
        }

    valid_geometry, geometry_message = validate_trade_geometry(
        entry=entry,
        stop_loss=sl_result["stop_loss"],
        take_profits=tp_result
    )

    if not valid_geometry:
        return {
            "valid": False,
            "reason": geometry_message
        }

    return {
        "valid": True,
        "entry_reference": float(reference),
        "stop_loss": sl_result["stop_loss"],
        "tp1": tp_result["tp1"],
        "tp2": tp_result["tp2"],
        "tp3": tp_result["tp3"],
        "risk_distance": sl_result["risk_distance"],
        "sl_method": sl_result["method"],
        "tp_method": tp_result["method"],
        "geometry_validation": geometry_message
    }


# END OF PART 4# ============================================================
# PART 5 — ADVANCED SIGNAL SCORING ENGINE
# ============================================================

def score_market_structure(df, direction):
    """Score confirmed structure and BOS/MSS."""

    score = 0
    reasons = []

    if df is None or len(df) < 30:
        return 0, ["Insufficient structure data"]

    structure = analyze_market_structure(df)
    trend = structure["trend"]
    event = structure["structure"]

    if trend["direction"] == direction:
        score += 12
        reasons.append("Market trend supports trade direction")

    if event["direction"] == direction:
        if event["event"] == "BOS":
            score += 10
            reasons.append("Directional BOS detected")
        elif event["event"] == "MSS":
            score += 6
            reasons.append("Directional MSS detected")

    if direction == "BULLISH":
        if structure["last_swing_high"] is not None:
            score += 2
            reasons.append("Confirmed swing-high structure available")

    elif direction == "BEARISH":
        if structure["last_swing_low"] is not None:
            score += 2
            reasons.append("Confirmed swing-low structure available")

    return min(30, score), reasons


def score_multi_timeframe(mtf_analysis, direction):
    """Score agreement between timeframe trends."""

    if not mtf_analysis:
        return 0, ["No multi-timeframe analysis"]

    final_direction = mtf_analysis.get("direction")
    agreement = mtf_analysis.get("agreement", 0)

    score = 0
    reasons = []

    if final_direction == direction:
        score += 15
        reasons.append("Multi-timeframe direction aligned")

    if agreement >= 80:
        score += 10
        reasons.append("Strong timeframe agreement")
    elif agreement >= 60:
        score += 7
        reasons.append("Moderate timeframe agreement")
    elif agreement >= 50:
        score += 4
        reasons.append("Partial timeframe agreement")

    return min(25, score), reasons


def score_smc_liquidity(df, direction):
    """
    Evaluate available SMC-style structure evidence.

    Swing levels are used as observable price references.
    They are not proof of institutional orders.
    """

    if df is None or len(df) < 30:
        return 0, ["Insufficient SMC data"]

    working = detect_swings(df)

    high = _find_column(working, ["high", "High"])
    low = _find_column(working, ["low", "Low"])
    close = _find_column(working, ["close", "Close"])

    if not high or not low or not close:
        return 0, ["OHLC data unavailable"]

    current_high = float(working[high].iloc[-1])
    current_low = float(working[low].iloc[-1])
    current_close = float(working[close].iloc[-1])

    swing_highs = working["Confirmed_Swing_High"].dropna()
    swing_lows = working["Confirmed_Swing_Low"].dropna()

    score = 0
    reasons = []

    if not swing_highs.empty and not swing_lows.empty:

        previous_high = float(swing_highs.iloc[-1])
        previous_low = float(swing_lows.iloc[-1])

        # Bullish liquidity sweep:
        # wick below a confirmed low, followed by close above it.
        if direction == "BULLISH":
            if current_low < previous_low and current_close > previous_low:
                score += 10
                reasons.append("Potential sell-side liquidity sweep")

        # Bearish liquidity sweep.
        elif direction == "BEARISH":
            if current_high > previous_high and current_close < previous_high:
                score += 10
                reasons.append("Potential buy-side liquidity sweep")

    # Directional structure break.
    structure = detect_bos_mss(working)

    if structure["direction"] == direction:
        score += 10
        reasons.append("Directional structure break supports setup")

    return min(20, score), reasons


def score_entry_quality(entry):
    """Score the precision and validity of the entry."""

    if not entry or not entry.get("valid"):
        return 0, ["Entry is invalid"]

    score = 0
    reasons = []

    entry_type = entry.get("entry_type")

    if entry_type == "LIMIT":
        score += 8
        reasons.append("Validated pullback Limit entry")

    elif entry_type == "MARKET":
        score += 6
        reasons.append("Market entry selected")

    confluence = entry.get("confluence", {})
    confluence_score = confluence.get("confluence_score", 0)

    if confluence_score >= 60:
        score += 7
        reasons.append("Strong entry confluence")
    elif confluence_score >= 40:
        score += 4
        reasons.append("Moderate entry confluence")

    return min(15, score), reasons


def score_momentum_volatility(df, direction):
    """Evaluate RSI, ADX and ATR conditions."""

    if df is None or len(df) < 30:
        return 0, ["Insufficient momentum data"]

    rsi = _last_value(df, ["RSI", "rsi"])
    adx = _last_value(df, ["ADX", "adx"])
    atr = _last_value(df, ["ATR", "atr"])

    score = 0
    reasons = []

    if np.isfinite(rsi):
        if direction == "BULLISH" and 50 <= rsi <= 70:
            score += 4
            reasons.append("Bullish RSI momentum")

        elif direction == "BEARISH" and 30 <= rsi <= 50:
            score += 4
            reasons.append("Bearish RSI momentum")

    if np.isfinite(adx):
        if adx >= 25:
            score += 4
            reasons.append("Strong trend strength")
        elif adx >= 18:
            score += 2
            reasons.append("Moderate trend strength")

    if np.isfinite(atr) and atr > 0:
        score += 2
        reasons.append("Valid volatility measurement")

    return min(10, score), reasons


def calculate_signal_score(
    df,
    direction,
    mtf_analysis,
    entry
):
    """
    Calculate a transparent score from 0 to 100.

    Weights:
    Structure: 30
    MTF:       25
    SMC:       20
    Entry:     15
    Momentum:  10
    """

    components = {}

    structure_score, structure_reasons = score_market_structure(
        df, direction
    )

    mtf_score, mtf_reasons = score_multi_timeframe(
        mtf_analysis, direction
    )

    smc_score, smc_reasons = score_smc_liquidity(
        df, direction
    )

    entry_score, entry_reasons = score_entry_quality(entry)

    momentum_score, momentum_reasons = score_momentum_volatility(
        df, direction
    )

    components["structure"] = structure_score
    components["multi_timeframe"] = mtf_score
    components["smc_liquidity"] = smc_score
    components["entry_quality"] = entry_score
    components["momentum"] = momentum_score

    total = sum(components.values())
    total = int(max(0, min(100, total)))

    reasons = (
        structure_reasons
        + mtf_reasons
        + smc_reasons
        + entry_reasons
        + momentum_reasons
    )

    if total >= 85:
        grade = "A"
        quality = "HIGH"
    elif total >= 70:
        grade = "B"
        quality = "GOOD"
    elif total >= 55:
        grade = "C"
        quality = "MODERATE"
    else:
        grade = "NO TRADE"
        quality = "LOW"

    return {
        "score": total,
        "grade": grade,
        "quality": quality,
        "components": components,
        "reasons": reasons
    }


def determine_risk_level(score, direction, entry):
    """Return a descriptive risk category, not a position size."""

    if not entry or not entry.get("valid"):
        return "HIGH"

    if direction not in ["BULLISH", "BEARISH"]:
        return "HIGH"

    if score >= 85:
        return "LOW"

    if score >= 70:
        return "MODERATE"

    return "HIGH"


def evaluate_signal_quality(
    df,
    direction,
    mtf_analysis,
    entry
):
    """Final scoring package for the signal engine."""

    result = calculate_signal_score(
        df=df,
        direction=direction,
        mtf_analysis=mtf_analysis,
        entry=entry
    )

    risk_level = determine_risk_level(
        score=result["score"],
        direction=direction,
        entry=entry
    )

    # A score alone cannot override invalid entry.
    approved = (
        entry.get("valid", False)
        and result["score"] >= 70
        and direction in ["BULLISH", "BEARISH"]
    )

    result["risk_level"] = risk_level
    result["approved"] = approved

    if not approved:
        result["decision"] = "WAIT"
    else:
        result["decision"] = "SIGNAL CANDIDATE"

    return result


# END OF PART 5#
# ============================================================
# PART 6 — ADVANCED MARKET & SIGNAL VALIDATION
# ============================================================

def validation_config(profile, asset):
    """
    Validation thresholds.
    These are initial safety thresholds and must be calibrated
    through historical testing for each asset and timeframe.
    """
    asset = str(asset).upper()
    profile = str(profile).title()

    configs = {
        "Scalping": {
            "min_rows": 50,
            "max_entry_atr": 1.50,
            "max_spread_atr": 0.15,
            "min_rr": 1.00,
            "max_candle_atr": 3.00,
            "min_atr_pct": 0.00005,
            "max_atr_pct": 0.05
        },
        "Intraday": {
            "min_rows": 50,
            "max_entry_atr": 2.00,
            "max_spread_atr": 0.20,
            "min_rr": 1.00,
            "max_candle_atr": 3.50,
            "min_atr_pct": 0.00005,
            "max_atr_pct": 0.08
        },
        "Swing": {
            "min_rows": 50,
            "max_entry_atr": 2.50,
            "max_spread_atr": 0.25,
            "min_rr": 1.00,
            "max_candle_atr": 4.00,
            "min_atr_pct": 0.00005,
            "max_atr_pct": 0.12
        }
    }

    config = configs.get(profile, configs["Intraday"]).copy()

    # Asset-specific thresholds can be calibrated later.
    if "BTC" in asset or "BITCOIN" in asset:
        config["max_candle_atr"] *= 1.15
        config["max_spread_atr"] *= 1.25

    return config


def _validation_number(value, default=None):
    try:
        number = float(value)
        if not np.isfinite(number):
            return default
        return number
    except (TypeError, ValueError):
        return default


def _validation_get_price(data, *keys):
    if not isinstance(data, dict):
        return None

    for key in keys:
        value = _validation_number(data.get(key))
        if value is not None:
            return value

    return None


def validate_market_data_quality(df, min_rows=50):
    """
    Returns True only when OHLC data is usable.
    Used by PART 7 during pending-signal review.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return False

    if len(df) < min_rows:
        return False

    required = ["open", "high", "low", "close"]

    if not all(column in df.columns for column in required):
        return False

    recent = df[required].tail(min_rows).copy()

    for column in required:
        recent[column] = pd.to_numeric(
            recent[column], errors="coerce"
        )

    if recent.isnull().any().any():
        return False

    if not np.isfinite(recent.to_numpy(dtype=float)).all():
        return False

    if (recent[["open", "high", "low", "close"]] <= 0).any().any():
        return False

    invalid_candles = (
        (recent["high"] < recent["low"]) |
        (recent["high"] < recent["open"]) |
        (recent["high"] < recent["close"]) |
        (recent["low"] > recent["open"]) |
        (recent["low"] > recent["close"])
    )

    if invalid_candles.any():
        return False

    return True


def detect_market_regime(df):
    """
    Classifies the market as trending, ranging, volatile,
    or uncertain. This is a rule-based estimate, not a prediction.
    """
    result = {
        "regime": "UNCERTAIN",
        "trend_strength": 0.0,
        "atr_pct": None,
        "adx": None,
        "is_trending": False,
        "is_ranging": False,
        "is_volatile": False
    }

    if not validate_market_data_quality(df, min_rows=50):
        return result

    last = df.iloc[-1]
    close = _validation_number(last.get("close"))
    atr = _validation_number(last.get("ATR"))
    adx = _validation_number(last.get("ADX"))

    if close is None or close <= 0 or atr is None or atr <= 0:
        return result

    atr_pct = atr / close

    result["atr_pct"] = atr_pct
    result["adx"] = adx

    if adx is not None:
        result["trend_strength"] = min(100.0, max(0.0, adx))

    # Volatility is assessed relative to price.
    if atr_pct >= 0.025:
        result["regime"] = "HIGH_VOLATILITY"
        result["is_volatile"] = True
        return result

    if adx is not None and adx >= 22:
        result["regime"] = "TRENDING"
        result["is_trending"] = True
        return result

    if adx is not None and adx < 18:
        result["regime"] = "RANGING"
        result["is_ranging"] = True
        return result

    result["regime"] = "TRANSITIONAL"
    return result


def detect_abnormal_candle(df, profile="Intraday", asset="XAU/USD"):
    """
    Returns True if the latest candle is unusually large
    compared with ATR. It does not reject normal volatility.
    """
    if not validate_market_data_quality(df, min_rows=20):
        return True

    config = validation_config(profile, asset)
    last = df.iloc[-1]

    high = _validation_number(last.get("high"))
    low = _validation_number(last.get("low"))
    atr = _validation_number(last.get("ATR"))

    if high is None or low is None or atr is None or atr <= 0:
        return True

    candle_range = high - low

    if candle_range <= 0:
        return True

    return candle_range > atr * config["max_candle_atr"]


def validate_entry_distance(
    df,
    entry,
    direction,
    profile="Intraday",
    asset="XAU/USD"
):
    """
    Checks whether the proposed entry is still reasonably
    close to the current market price.
    """
    result = {
        "valid": False,
        "reason": "",
        "distance": None,
        "distance_atr": None
    }

    if not validate_market_data_quality(df, min_rows=20):
        result["reason"] = "Insufficient or invalid market data"
        return result

    last = df.iloc[-1]
    current_price = _validation_number(last.get("close"))
    atr = _validation_number(last.get("ATR"))

    if current_price is None or atr is None or atr <= 0:
        result["reason"] = "Invalid current price or ATR"
        return result

    if not isinstance(entry, dict):
        result["reason"] = "Invalid entry structure"
        return result

    entry_price = _validation_get_price(
        entry, "entry_price", "entry", "price"
    )

    zone_low = _validation_get_price(entry, "zone_low")
    zone_high = _validation_get_price(entry, "zone_high")

    if zone_low is not None and zone_high is not None:
        if zone_low > zone_high:
            result["reason"] = "Invalid entry zone"
            return result

        reference_price = (
            zone_low if str(direction).upper() in ("BUY", "LONG")
            else zone_high
        )
    else:
        reference_price = entry_price

    if reference_price is None or reference_price <= 0:
        result["reason"] = "Entry price is missing"
        return result

    distance = abs(current_price - reference_price)
    distance_atr = distance / atr

    result["distance"] = distance
    result["distance_atr"] = distance_atr

    config = validation_config(profile, asset)

    if distance_atr > config["max_entry_atr"]:
        result["reason"] = "Entry is too far from current market price"
        return result

    result["valid"] = True
    result["reason"] = "Entry distance is acceptable"
    return result


def calculate_reward_risk(
    entry_price,
    stop_loss,
    take_profit,
    direction
):
    """
    Calculates reward/risk ratio for one target.
    Returns None when geometry is invalid.
    """
    entry_price = _validation_number(entry_price)
    stop_loss = _validation_number(stop_loss)
    take_profit = _validation_number(take_profit)

    if None in (entry_price, stop_loss, take_profit):
        return None

    if entry_price <= 0:
        return None

    direction = str(direction).upper()

    if direction in ("BUY", "LONG"):
        risk = entry_price - stop_loss
        reward = take_profit - entry_price

    elif direction in ("SELL", "SHORT"):
        risk = stop_loss - entry_price
        reward = entry_price - take_profit

    else:
        return None

    if risk <= 0 or reward <= 0:
        return None

    return reward / risk


def validate_reward_risk(
    entry_price,
    stop_loss,
    take_profit,
    direction,
    profile="Intraday",
    asset="XAU/USD"
):
    ratio = calculate_reward_risk(
        entry_price,
        stop_loss,
        take_profit,
        direction
    )

    minimum_rr = validation_config(
        profile, asset
    )["min_rr"]

    return ratio is not None and ratio >= minimum_rr


def validate_mtf_alignment(mtf_analysis, direction):
    """
    Accepts the multi-timeframe analysis dictionary.
    Checks the available timeframe trend directions.
    Missing timeframe data is not treated as confirmation.
    """
    result = {
        "valid": False,
        "aligned_count": 0,
        "checked_count": 0,
        "reason": ""
    }

    if not isinstance(mtf_analysis, dict) or not mtf_analysis:
        result["reason"] = "MTF analysis is missing"
        return result

    wanted = str(direction).upper()

    if wanted in ("BUY", "LONG"):
        accepted = ("BULLISH", "UP", "BUY", "LONG")
    elif wanted in ("SELL", "SHORT"):
        accepted = ("BEARISH", "DOWN", "SELL", "SHORT")
    else:
        result["reason"] = "Invalid trade direction"
        return result

    for timeframe, data in mtf_analysis.items():
        if not isinstance(data, dict):
            continue

        trend = data.get("trend")

        if trend is None:
            trend = data.get("direction")

        if trend is None:
            continue

        result["checked_count"] += 1

        if str(trend).upper() in accepted:
            result["aligned_count"] += 1

    if result["checked_count"] == 0:
        result["reason"] = "No usable timeframe confirmations"
        return result

    # At least half of available timeframes must agree.
    result["valid"] = (
        result["aligned_count"] >=
        max(1, (result["checked_count"] + 1) // 2)
    )

    result["reason"] = (
        "MTF alignment acceptable"
        if result["valid"]
        else "MTF trend conflict"
    )

    return result


def validate_signal_candidate(
    df,
    entry,
    trade_levels,
    score,
    mtf_analysis,
    profile,
    asset,
    direction,
    spread=None
):
    """
    Final validation before a signal can be published.

    IMPORTANT:
    Spread must be supplied from a real source or a clearly
    configured estimate. Missing spread is a validation failure.
    Never assume spread is zero.
    """
    result = {
        "valid": False,
        "reasons": [],
        "checks": {}
    }

    config = validation_config(profile, asset)

    # 1. Market data
    data_ok = validate_market_data_quality(
        df, min_rows=config["min_rows"]
    )
    result["checks"]["market_data"] = data_ok

    if not data_ok:
        result["reasons"].append("Market data quality failed")
        return result

    # 2. Abnormal candle
    abnormal = detect_abnormal_candle(
        df, profile, asset
    )
    result["checks"]["abnormal_candle"] = not abnormal

    if abnormal:
        result["reasons"].append("Abnormal or unreliable candle")

    # 3. Market regime
    regime = detect_market_regime(df)
    result["checks"]["market_regime"] = regime

    if regime["regime"] == "HIGH_VOLATILITY":
        result["reasons"].append("Market volatility is unusually high")

    if regime["regime"] == "UNCERTAIN":
        result["reasons"].append("Market regime cannot be determined")

    # 4. Entry distance
    entry_check = validate_entry_distance(
        df, entry, direction, profile, asset
    )
    result["checks"]["entry_distance"] = entry_check

    if not entry_check["valid"]:
        result["reasons"].append(entry_check["reason"])

    # 5. MTF alignment
    mtf_check = validate_mtf_alignment(
        mtf_analysis, direction
    )
    result["checks"]["mtf_alignment"] = mtf_check

    if not mtf_check["valid"]:
        result["reasons"].append(mtf_check["reason"])

    # 6. Score
    score_value = _validation_number(score)

    score_ok = (
        score_value is not None and
        score_value >= 55 and
        score_value <= 100
    )
    result["checks"]["score"] = score_ok

    if not score_ok:
        result["reasons"].append("Signal score is below minimum")

    # 7. Spread
    spread_value = _validation_number(spread)
    atr = _validation_number(df.iloc[-1].get("ATR"))

    spread_ok = (
        spread_value is not None and
        spread_value > 0 and
        atr is not None and
        atr > 0 and
        spread_value <= atr * config["max_spread_atr"]
    )

    result["checks"]["spread"] = spread_ok

    if not spread_ok:
        result["reasons"].append(
            "Spread is missing, invalid, or too high"
        )

    # 8. Trade levels and reward/risk
    if not isinstance(trade_levels, dict):
        result["reasons"].append("Trade levels are missing")
        result["checks"]["trade_geometry"] = False
        return result

    stop_loss = _validation_get_price(
        trade_levels, "stop_loss", "sl"
    )

    take_profit = _validation_get_price(
        trade_levels, "tp1", "take_profit", "take_profit_1"
    )

    entry_price = _validation_get_price(
        entry, "entry_price", "entry", "price"
    )

    if entry_price is None:
        entry_price = _validation_get_price(
            trade_levels, "entry_price", "entry"
        )

    if (
        entry_price is not None and
        stop_loss is not None and
        take_profit is not None
    ):
        rr_ok = validate_reward_risk(
            entry_price,
            stop_loss,
            take_profit,
            direction,
            profile,
            asset
        )
    else:
        rr_ok = False

    result["checks"]["trade_geometry"] = rr_ok

    if not rr_ok:
        result["reasons"].append(
            "Invalid trade geometry or insufficient reward/risk"
        )

    # Final decision
    result["valid"] = len(result["reasons"]) == 0

    if result["valid"]:
        result["reasons"].append(
            "All required validation checks passed"
        )

    return result


# ================= END OF PART 6 =================
 ============================================================
# PART 7 — SIGNAL LOCK, PERSISTENCE & EXPIRY MANAGEMENT
# ============================================================

from datetime import datetime, timezone, timedelta


SIGNAL_VALIDITY_MINUTES = {
    "Scalping": 30,
    "Intraday": 90,
    "Swing": 360
}

ACTIVE_SIGNAL_STATUSES = [
    "WAITING",
    "PENDING LIMIT",
    "ACTIVE TRADE",
    "TP1 HIT",
    "TP2 HIT"
]

FINAL_SIGNAL_STATUSES = [
    "TP3 HIT",
    "SL HIT",
    "INVALIDATED",
    "CANCELLED",
    "CLOSED"
]


def utc_now():
    """Return current UTC time."""
    return datetime.now(timezone.utc)


def iso_utc(value=None):
    """Convert datetime to UTC ISO format."""
    value = value or utc_now()

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc).isoformat()


def parse_utc(value):
    """Parse stored ISO datetime safely."""

    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        return parsed.astimezone(timezone.utc)

    except (ValueError, TypeError):
        return None


def get_signal_validity(profile):
    """Return configured signal validity in minutes."""

    return SIGNAL_VALIDITY_MINUTES.get(
        profile,
        SIGNAL_VALIDITY_MINUTES["Intraday"]
    )


def get_supabase_config():
    """Read Supabase credentials from Streamlit secrets."""

    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]

        if not url or not key:
            return None

        return {
            "url": str(url).rstrip("/"),
            "key": str(key)
        }

    except Exception:
        return None


def supabase_headers(config, prefer=None):
    headers = {
        "apikey": config["key"],
        "Authorization": f"Bearer {config['key']}",
        "Content-Type": "application/json"
    }

    if prefer:
        headers["Prefer"] = prefer

    return headers


def supabase_request(method, table, params=None, payload=None,
                     prefer=None):
    """Centralized Supabase REST request."""

    config = get_supabase_config()

    if not config:
        return {
            "success": False,
            "data": None,
            "error": "Supabase credentials are unavailable"
        }

    url = f"{config['url']}/rest/v1/{table}"

    try:
        response = requests.request(
            method=method,
            url=url,
            headers=supabase_headers(config, prefer),
            params=params,
            json=payload,
            timeout=15
        )

        if response.status_code not in [200, 201, 204]:
            return {
                "success": False,
                "data": None,
                "error": response.text[:1000],
                "status_code": response.status_code
            }

        if response.status_code == 204 or not response.text:
            data = None
        else:
            data = response.json()

        return {
            "success": True,
            "data": data,
            "error": None
        }

    except requests.RequestException as exc:
        return {
            "success": False,
            "data": None,
            "error": str(exc)
        }


def create_signal_key(signal):
    """
    Generate a deterministic signal key.

    A new signal requires a new closed candle or a genuinely
    different setup. Refreshing the dashboard must not create
    a new key.
    """

    identity = {
        "asset": signal.get("asset"),
        "profile": signal.get("profile"),
        "direction": signal.get("direction"),
        "entry_type": signal.get("entry_type"),
        "candle_time": signal.get("candle_time"),
        "entry_price": signal.get("entry_price"),
        "zone_low": signal.get("zone_low"),
        "zone_high": signal.get("zone_high")
    }

    raw = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":")
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def build_signal_snapshot(
    asset,
    profile,
    direction,
    entry,
    trade_levels,
    score_result,
    session,
    candle_time
):
    """
    Create an immutable snapshot of a candidate signal.

    This function does not publish anything by itself.
    """

    if not entry or not entry.get("valid"):
        return {
            "valid": False,
            "reason": "Invalid entry"
        }

    if not trade_levels or not trade_levels.get("valid"):
        return {
            "valid": False,
            "reason": "Invalid trade levels"
        }

    if direction not in ["BULLISH", "BEARISH"]:
        return {
            "valid": False,
            "reason": "Invalid direction"
        }

    profile_minutes = get_signal_validity(profile)
    created = utc_now()
    expires = created + timedelta(minutes=profile_minutes)

    snapshot = {
        "asset": asset,
        "profile": profile,
        "direction": direction,
        "entry_type": entry["entry_type"],

        "entry_price": entry.get("entry_price"),
        "zone_low": entry.get("zone_low"),
        "zone_high": entry.get("zone_high"),

        "stop_loss": trade_levels["stop_loss"],
        "tp1": trade_levels["tp1"],
        "tp2": trade_levels.get("tp2"),
        "tp3": trade_levels.get("tp3"),

        "score": score_result.get("score", 0),
        "grade": score_result.get("grade", "NO TRADE"),
        "risk_level": score_result.get("risk_level", "HIGH"),

        "session": session,
        "candle_time": str(candle_time),

        "generated_at": iso_utc(created),
        "expires_at": iso_utc(expires),

        "status": (
            "PENDING LIMIT"
            if entry["entry_type"] == "LIMIT"
            else "WAITING"
        ),

        "extension_count": 0,
        "last_review_at": None,
        "last_review_reason": None,

        "trade_opened_at": None,
        "closed_at": None
    }

    snapshot["signal_key"] = create_signal_key(snapshot)

    return {
        "valid": True,
        "snapshot": snapshot
    }


def get_signal_by_key(signal_key):
    """Retrieve a signal using its unique key."""

    result = supabase_request(
        "GET",
        "signals",
        params={
            "select": "*",
            "signal_key": f"eq.{signal_key}",
            "limit": "1"
        }
    )

    if not result["success"]:
        return None

    rows = result["data"]

    if isinstance(rows, list) and rows:
        return rows[0]

    return None


def save_new_signal(snapshot):
    """
    Insert a signal without replacing an existing signal.

    The database must enforce UNIQUE(signal_key).
    """

    if not snapshot or not snapshot.get("signal_key"):
        return {
            "success": False,
            "created": False,
            "error": "Invalid signal snapshot"
        }

    result = supabase_request(
        "POST",
        "signals",
        params={
            "on_conflict": "signal_key"
        },
        payload=snapshot,
        prefer="resolution=ignore-duplicates,return=representation"
    )

    if not result["success"]:
        return {
            "success": False,
            "created": False,
            "error": result["error"]
        }

    inserted = result["data"]

    if isinstance(inserted, list) and inserted:
        return {
            "success": True,
            "created": True,
            "signal": inserted[0]
        }

    # Empty response means an existing duplicate may have been ignored.
    existing = get_signal_by_key(snapshot["signal_key"])

    if existing:
        return {
            "success": True,
            "created": False,
            "signal": existing,
            "reason": "Duplicate signal prevented"
        }

    return {
        "success": False,
        "created": False,
        "error": "Signal insertion could not be confirmed"
    }


def update_signal_status(signal_id, new_status, reason=None):
    """
    Update status only.

    Published Entry, SL and TP fields are deliberately excluded.
    """

    if new_status not in ACTIVE_SIGNAL_STATUSES + FINAL_SIGNAL_STATUSES:
        return {
            "success": False,
            "error": "Unsupported signal status"
        }

    payload = {
        "status": new_status,
        "last_review_at": iso_utc(),
        "last_review_reason": reason
    }

    if new_status in FINAL_SIGNAL_STATUSES:
        payload["closed_at"] = iso_utc()

    result = supabase_request(
        "PATCH",
        "signals",
        params={
            "id": f"eq.{signal_id}",
            "status": f"not.in.({','.join(ACTIVE_SIGNAL_STATUSES)})"
        } if new_status in FINAL_SIGNAL_STATUSES else {
            "id": f"eq.{signal_id}",
            "status": f"in.({','.join(ACTIVE_SIGNAL_STATUSES)})"
        },
        payload=payload,
        prefer="return=representation"
    )

    return result


def extend_signal_validity(signal):
    """
    Extend validity without modifying published trade prices.

    The new expiry is based on the review time, not on the old
    expiry. Every extension is recorded.
    """

    if not signal:
        return {
            "success": False,
            "error": "Signal not found"
        }

    if signal.get("status") not in [
        "WAITING",
        "PENDING LIMIT"
    ]:
        return {
            "success": False,
            "error": "Only pending signals can be extended"
        }

    signal_id = signal.get("id")

    if signal_id is None:
        return {
            "success": False,
            "error": "Signal ID missing"
        }

    profile = signal.get("profile", "Intraday")
    minutes = get_signal_validity(profile)

    new_expiry = utc_now() + timedelta(minutes=minutes)

    payload = {
        "expires_at": iso_utc(new_expiry),
        "extension_count": int(
            signal.get("extension_count") or 0
        ) + 1,
        "last_review_at": iso_utc(),
        "last_review_reason": "Setup remains valid after expiry review"
    }

    result = supabase_request(
        "PATCH",
        "signals",
        params={
            "id": f"eq.{signal_id}",
            "status": "in.(WAITING,PENDING LIMIT)"
        },
        payload=payload,
        prefer="return=representation"
    )

    if result["success"]:
        return {
            "success": True,
            "new_expiry": payload["expires_at"],
            "extension_count": payload["extension_count"],
            "data": result["data"]
        }

    return {
        "success": False,
        "error": result["error"]
    }


def is_fixed_limit_actionable(signal, current_price, atr):
    """
    Check whether the original Limit zone remains actionable.

    Never move the original zone to follow the current market.
    """

    if signal.get("entry_type") != "LIMIT":
        return False, "Not a pending Limit signal"

    low = signal.get("zone_low")
    high = signal.get("zone_high")
    direction = signal.get("direction")

    if low is None or high is None:
        return False, "Original Limit zone is incomplete"

    if not np.isfinite(current_price) or not np.isfinite(atr) or atr <= 0:
        return False, "Current market data is invalid"

    low = float(low)
    high = float(high)

    if direction == "BULLISH":

        if current_price < low - atr * 0.25:
            return False, "Price moved below the original bullish zone"

        if current_price > high + atr * 2.0:
            return False, "Original bullish Limit zone is too far away"

    elif direction == "BEARISH":

        if current_price > high + atr * 0.25:
            return False, "Price moved above the original bearish zone"

        if current_price < low - atr * 2.0:
            return False, "Original bearish Limit zone is too far away"

    else:
        return False, "Invalid signal direction"

    return True, "Original Limit zone remains actionable"


def review_pending_signal(signal, df, current_structure):
    """
    Review an expired pending signal.

    If the setup remains valid, extend its timer.
    If invalidated, update status only.
    Never change original Entry, SL or TP.
    """

    if not signal:
        return {
            "success": False,
            "decision": "ERROR",
            "reason": "Signal not found"
        }

    if signal.get("status") not in [
        "WAITING",
        "PENDING LIMIT"
    ]:
        return {
            "success": False,
            "decision": "SKIP",
            "reason": "Signal is not pending"
        }

    expiry = parse_utc(signal.get("expires_at"))

    if expiry is None:
        return {
            "success": False,
            "decision": "ERROR",
            "reason": "Stored expiry timestamp is invalid"
        }

    if utc_now() < expiry:
        return {
            "success": True,
            "decision": "STILL_VALID",
            "reason": "Signal validity period has not ended"
        }

    if df is None or df.empty:
        return {
            "success": False,
            "decision": "REVIEW_REQUIRED",
            "reason": "Market data unavailable; cannot safely extend"
        }

    data_ok, data_message = validate_market_data_quality(df)

    if not data_ok:
        return {
            "success": False,
            "decision": "REVIEW_REQUIRED",
            "reason": data_message
        }

    direction = signal.get("direction")
    current_price = _last_value(df, ["close", "Close"])
    atr = _last_value(df, ["ATR", "atr"])

    if not np.isfinite(current_price) or not np.isfinite(atr):
        return {
            "success": False,
            "decision": "REVIEW_REQUIRED",
            "reason": "Current price or ATR unavailable"
        }

    trend = current_structure.get("trend", {})
    structure = current_structure.get("structure", {})

    trend_direction = trend.get("direction")
    structure_direction = structure.get("direction")

    # Strong opposing structure invalidates the original setup.
    opposing_structure = (
        structure_direction in ["BULLISH", "BEARISH"]
        and structure_direction != direction
        and structure.get("event") == "MSS"
    )

    if opposing_structure:
        update_signal_status(
            signal["id"],
            "INVALIDATED",
            "Opposing market structure shift detected"
        )

        return {
            "success": True,
            "decision": "INVALIDATED",
            "reason": "Opposing MSS detected"
        }

    if trend_direction not in [direction, "NEUTRAL"]:
        update_signal_status(
            signal["id"],
            "INVALIDATED",
            "Original directional trend has reversed"
        )

        return {
            "success": True,
            "decision": "INVALIDATED",
            "reason": "Trend direction has reversed"
        }

    # Check the original fixed Limit zone.
    if signal.get("entry_type") == "LIMIT":

        actionable, action_reason = is_fixed_limit_actionable(
            signal,
            current_price,
            atr
        )

        if not actionable:
            update_signal_status(
                signal["id"],
                "INVALIDATED",
                action_reason
            )

            return {
                "success": True,
                "decision": "INVALIDATED",
                "reason": action_reason
            }

    # A market-entry signal that was not executed must not
    # silently become a new entry at the current market price.
    if signal.get("entry_type") == "MARKET":

        original_entry = signal.get("entry_price")

        if original_entry is None:
            return {
                "success": False,
                "decision": "REVIEW_REQUIRED",
                "reason": "Original Market entry is missing"
            }

        distance = abs(
            current_price - float(original_entry)
        )

        if distance > atr * 0.50:
            update_signal_status(
                signal["id"],
                "INVALIDATED",
                "Original Market entry is no longer actionable"
            )

            return {
                "success": True,
                "decision": "INVALIDATED",
                "reason": "Original Market entry is no longer actionable"
            }

    # Structure remains acceptable: extend timer only.
    extension = extend_signal_validity(signal)

    if not extension["success"]:
        return {
            "success": False,
            "decision": "ERROR",
            "reason": extension.get("error")
        }

    return {
        "success": True,
        "decision": "EXTENDED",
        "new_expiry": extension["new_expiry"],
        "extension_count": extension["extension_count"],
        "reason": "Original setup remains valid; prices unchanged"
    }


def get_signal_remaining_seconds(signal):
    """Calculate countdown from persisted absolute expiry."""

    expiry = parse_utc(signal.get("expires_at"))

    if expiry is None:
        return None

    remaining = (expiry - utc_now()).total_seconds()

    return max(0, int(remaining))


def format_signal_countdown(signal):
    """Return countdown text."""

    seconds = get_signal_remaining_seconds(signal)

    if seconds is None:
        return "INVALID TIMER"

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"


def update_open_trade_status(signal_id, new_status, reason=None):
    """
    Update an already-open trade independently of signal expiry.

    Expiry review must never call this function automatically.
    """

    allowed = [
        "ACTIVE TRADE",
        "TP1 HIT",
        "TP2 HIT",
        "TP3 HIT",
        "SL HIT",
        "CLOSED"
    ]

    if new_status not in allowed:
        return {
            "success": False,
            "error": "Invalid trade status"
        }

    payload = {
        "status": new_status,
        "last_review_at": iso_utc(),
        "last_review_reason": reason
    }

    if new_status in ["TP3 HIT", "SL HIT", "CLOSED"]:
        payload["closed_at"] = iso_utc()

    return supabase_request(
        "PATCH",
        "signals",
        params={
            "id": f"eq.{signal_id}",
            "status": "in.(ACTIVE TRADE,TP1 HIT,TP2 HIT)"
        },
        payload=payload,
        prefer="return=representation"
    )


# END OF PART 7
# ============================================================
# PART 8 — TELEGRAM ALERTS & DUPLICATE PROTECTION
# ============================================================

import html
from datetime import datetime, timezone


def get_telegram_config():
    """Read Telegram credentials from Streamlit Secrets."""
    try:
        token = str(st.secrets["TELEGRAM_BOT_TOKEN"]).strip()
        chat_id = str(st.secrets["TELEGRAM_CHAT_ID"]).strip()

        if not token or not chat_id:
            return None, None

        return token, chat_id

    except Exception:
        return None, None


def telegram_format_price(value, asset=""):
    """Format a price without changing its value."""
    try:
        number = float(value)

        if "BTC" in str(asset).upper():
            return f"{number:,.2f}"

        return f"{number:,.2f}"

    except (TypeError, ValueError):
        return "N/A"


def telegram_countdown(expires_at):
    """Calculate remaining time from persisted expiry."""
    try:
        expiry = parse_utc(expires_at)

        if expiry is None:
            return "Unknown"

        now = datetime.now(timezone.utc)
        remaining = int((expiry - now).total_seconds())

        if remaining <= 0:
            return "Review required"

        hours, remainder = divmod(remaining, 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours:
            return f"{hours}h {minutes}m"

        return f"{minutes}m {seconds}s"

    except Exception:
        return "Unknown"


def build_telegram_signal_message(signal):
    """Create a Telegram HTML message from a fixed signal snapshot."""

    asset = html.escape(str(signal.get("asset", "UNKNOWN")))
    profile = html.escape(str(signal.get("profile", "UNKNOWN")))

    direction = str(signal.get("direction", "")).upper()

    if direction in ("BULLISH", "BUY", "LONG"):
        side = "BUY 🟢"
    elif direction in ("BEARISH", "SELL", "SHORT"):
        side = "SELL 🔴"
    else:
        side = html.escape(direction or "UNKNOWN")

    entry_type = html.escape(
        str(signal.get("entry_type", "UNKNOWN")).upper()
    )

    entry_price = signal.get("entry_price")
    zone_low = signal.get("zone_low")
    zone_high = signal.get("zone_high")

    if entry_type == "LIMIT":
        entry_text = (
            f"{telegram_format_price(zone_low, asset)}"
            f" - "
            f"{telegram_format_price(zone_high, asset)}"
        )
    else:
        entry_text = telegram_format_price(entry_price, asset)

    sl = telegram_format_price(
        signal.get("stop_loss"), asset
    )

    tp1 = telegram_format_price(signal.get("tp1"), asset)
    tp2 = telegram_format_price(signal.get("tp2"), asset)
    tp3 = telegram_format_price(signal.get("tp3"), asset)

    score = html.escape(str(signal.get("score", "N/A")))
    grade = html.escape(str(signal.get("grade", "N/A")))
    risk = html.escape(str(signal.get("risk_level", "N/A")))
    session = html.escape(str(signal.get("session", "N/A")))

    expiry = telegram_countdown(signal.get("expires_at"))

    signal_id = html.escape(
        str(signal.get("signal_key", "N/A"))[:16]
    )

    return (
        "🚨 <b>KHAN GOLD SIGNAL</b>\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"💹 <b>Asset:</b> {asset}\n"
        f"📊 <b>Profile:</b> {profile}\n"
        f"🎯 <b>Direction:</b> {side}\n"
        f"⚙️ <b>Entry Type:</b> {entry_type}\n"
        f"📍 <b>Entry:</b> {entry_text}\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🛑 <b>Stop Loss:</b> {sl}\n"
        f"🥇 <b>TP1:</b> {tp1}\n"
        f"🥈 <b>TP2:</b> {tp2}\n"
        f"🥉 <b>TP3:</b> {tp3}\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📈 <b>Score:</b> {score}/100\n"
        f"🏆 <b>Grade:</b> {grade}\n"
        f"⚠️ <b>Risk:</b> {risk}\n"
        f"🌍 <b>Session:</b> {session}\n"
        f"⏳ <b>Validity:</b> {expiry}\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🔐 <b>Signal ID:</b> {signal_id}\n"
        "<i>Trade responsibly. No profit is guaranteed.</i>"
    )


def telegram_supabase_headers():
    """Build Supabase REST headers using server-side credentials."""

    url = str(st.secrets["SUPABASE_URL"]).rstrip("/")
    key = str(st.secrets["SUPABASE_KEY"])

    return url, {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def reserve_telegram_alert(signal):
    """
    Reserve a unique alert in Supabase.
    A previously registered signal is never automatically resent.
    """

    signal_key = signal.get("signal_key")

    if not signal_key:
        return False, "Missing signal_key"

    try:
        url, headers = telegram_supabase_headers()

        endpoint = f"{url}/rest/v1/telegram_outbox"

        payload = {
            "signal_key": signal_key,
            "status": "PENDING",
            "message_payload": signal,
        }

        insert_headers = dict(headers)
        insert_headers["Prefer"] = (
            "resolution=ignore-duplicates,return=representation"
        )

        response = requests.post(
            endpoint,
            params={"on_conflict": "signal_key"},
            headers=insert_headers,
            json=payload,
            timeout=15,
        )

        if not response.ok:
            return False, "Supabase reservation failed"

        rows = response.json()

        if isinstance(rows, list) and len(rows) > 0:
            return True, "RESERVED"

        # Existing key: do not send again.
        return False, "DUPLICATE"

    except Exception:
        return False, "Supabase connection error"


def update_telegram_outbox(signal_key, status,
                           message_id=None, error=None):
    """Save Telegram delivery result in Supabase."""

    try:
        url, headers = telegram_supabase_headers()

        endpoint = f"{url}/rest/v1/telegram_outbox"

        payload = {
            "status": status,
            "updated_at": iso_utc(),
        }

        if message_id is not None:
            payload["telegram_message_id"] = message_id

        if status == "SENT":
            payload["sent_at"] = iso_utc()

        if error:
            payload["error_message"] = str(error)[:500]

        response = requests.patch(
            endpoint,
            params={"signal_key": f"eq.{signal_key}"},
            headers={
                **headers,
                "Prefer": "return=representation",
            },
            json=payload,
            timeout=15,
        )

        return response.ok

    except Exception:
        return False


def send_telegram_message(message):
    """Send one HTML message through Telegram Bot API."""

    token, chat_id = get_telegram_config()

    if not token or not chat_id:
        return False, None, "Telegram credentials missing"

    endpoint = (
        f"https://api.telegram.org/bot{token}/sendMessage"
    )

    try:
        response = requests.post(
            endpoint,
            json={
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=20,
        )

        result = response.json()

        if response.ok and result.get("ok"):
            message_id = (
                result.get("result", {}).get("message_id")
            )

            return True, message_id, None

        error = result.get("description", "Telegram API error")

        return False, None, error

    except Exception:
        return False, None, "Telegram network error"


def send_signal_telegram_once(signal):
    """
    Send a signal only after a unique database reservation.

    FAILED and SENDING records are not automatically retried.
    This avoids accidental duplicate Telegram messages.
    """

    if not isinstance(signal, dict):
        return {
            "success": False,
            "status": "INVALID_SIGNAL",
        }

    signal_key = signal.get("signal_key")

    if not signal_key:
        return {
            "success": False,
            "status": "MISSING_SIGNAL_KEY",
        }

    reserved, reason = reserve_telegram_alert(signal)

    if not reserved:
        return {
            "success": False,
            "status": reason,
        }

    # Record the sending state before calling Telegram.
    update_telegram_outbox(
        signal_key,
        "SENDING"
    )

    message = build_telegram_signal_message(signal)

    success, message_id, error = send_telegram_message(
        message
    )

    if success:
        saved = update_telegram_outbox(
            signal_key,
            "SENT",
            message_id=message_id,
        )

        return {
            "success": True,
            "status": "SENT" if saved else "SENT_DB_UPDATE_FAILED",
            "message_id": message_id,
        }

    update_telegram_outbox(
        signal_key,
        "FAILED",
        error=error,
    )

    return {
        "success": False,
        "status": "FAILED",
        "error": error,
    }


# END OF PART 8
