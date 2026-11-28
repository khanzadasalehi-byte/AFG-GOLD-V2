
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import html
import uuid
import time

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# ============================================================
# KHANGOLDSIGNAL — PART 1 OF 3
# CONFIGURATION + MARKET DATA + STRATEGY ENGINE
# ============================================================

st.set_page_config(
    page_title="KhanGoldSignal | GOLD & BTC",
    page_icon="🟡",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "KhanGoldSignal"
UTC = timezone.utc
KABUL = ZoneInfo("Asia/Kabul")

ASSETS = {
    "GOLD": {
        "symbol": "XAU/USD",
        "label": "XAUUSD",
    },
    "BITCOIN": {
        "symbol": "BTC/USD",
        "label": "BTCUSD",
    },
}

PROFILES = {
    "Scalping": {
        "entry_tf": "1min",
        "trend_tf": "5min",
        "confirm_tf": "15min",
        "entry_minutes": 1,
        "trend_minutes": 5,
        "confirm_minutes": 15,
        "min_score": 65,
        "min_adx": 15,
        "atr_sl": 1.5,
        "pullback_atr_min": 0.05,
        "pullback_atr_max": 0.75,
    },
    "Intraday": {
        "entry_tf": "15min",
        "trend_tf": "1h",
        "confirm_tf": "4h",
        "entry_minutes": 15,
        "trend_minutes": 60,
        "confirm_minutes": 240,
        "min_score": 65,
        "min_adx": 16,
        "atr_sl": 1.7,
        "pullback_atr_min": 0.08,
        "pullback_atr_max": 0.90,
    },
    "Swing": {
        "entry_tf": "4h",
        "trend_tf": "1day",
        "confirm_tf": "1day",
        "entry_minutes": 240,
        "trend_minutes": 1440,
        "confirm_minutes": 1440,
        "min_score": 65,
        "min_adx": 17,
        "atr_sl": 2.0,
        "pullback_atr_min": 0.10,
        "pullback_atr_max": 1.10,
    },
}

VALIDITY_MINUTES = 45

# ----------------------- DESIGN -----------------------------

st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background:#090611; color:#e3d9f2; }
[data-testid="stSidebar"] {
    background:#17131f;
    border-right:1px solid #30263b;
}
.block-container {
    padding-top:1.1rem;
    max-width:1400px;
}
h1,h2,h3,p,label {
    font-family:Georgia,'Times New Roman',serif;
}
div[data-testid="stMetric"] {
    background:#100b1b;
    border:1px solid #30233d;
    padding:12px;
    border-radius:12px;
}
.stButton > button {
    width:100%;
    min-height:43px;
    border-radius:10px;
    font-weight:800;
}
.small-muted {
    color:#9888ad;
    letter-spacing:2px;
    font-size:11px;
}
.signal-card {
    background:#100b1b;
    border:1px solid #392b49;
    border-radius:16px;
    padding:18px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# TIME / GENERAL HELPERS
# ============================================================

def utc_now():
    return datetime.now(UTC)


def iso_utc(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return value.astimezone(UTC).isoformat()


def parse_datetime(value):
    if not value:
        return None
    try:
        result = pd.Timestamp(value).to_pydatetime()
        if result.tzinfo is None:
            result = result.replace(tzinfo=UTC)
        return result.astimezone(UTC)
    except Exception:
        return None


def fmt(value):
    try:
        if value is None or not np.isfinite(float(value)):
            return "—"
        return f"{float(value):,.2f}"
    except Exception:
        return "—"


def safe_float(value, default=np.nan):
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except Exception:
        return default


def interval_delta(interval):
    mapping = {
        "1min": timedelta(minutes=1),
        "5min": timedelta(minutes=5),
        "15min": timedelta(minutes=15),
        "30min": timedelta(minutes=30),
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "1day": timedelta(days=1),
    }
    return mapping[interval]


# ============================================================
# INDICATORS
# ============================================================

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def rsi(series, period=14):
    change = series.diff()
    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)

    avg_gain = gains.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = losses.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))

    result = result.mask(
        (avg_loss == 0) & (avg_gain > 0), 100
    )
    result = result.mask(
        (avg_gain == 0) & (avg_loss > 0), 0
    )
    return result


def true_range(df):
    previous_close = df["close"].shift(1)

    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    )

    return ranges.max(axis=1)


def atr(df, period=14):
    return true_range(df).ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


def adx_components(df, period=14):
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0.0,
        ),
        index=df.index,
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0.0,
        ),
        index=df.index,
    )

    smoothed_tr = true_range(df).ewm(
        alpha=1 / period,
        adjust=False
    ).mean().replace(0, np.nan)

    plus_di = (
        100
        * plus_dm.ewm(alpha=1 / period, adjust=False).mean()
        / smoothed_tr
    )

    minus_di = (
        100
        * minus_dm.ewm(alpha=1 / period, adjust=False).mean()
        / smoothed_tr
    )

    denominator = (plus_di + minus_di).replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / denominator

    adx_value = dx.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    return adx_value, plus_di, minus_di


def add_indicators(df):
    result = df.copy()

    for period in (20, 50, 200):
        result[f"ema{period}"] = ema(
            result["close"], period
        )

    result["rsi"] = rsi(result["close"])
    result["atr"] = atr(result)

    (
        result["adx"],
        result["plus_di"],
        result["minus_di"],
    ) = adx_components(result)

    result["candle_range"] = (
        result["high"] - result["low"]
    )

    result["body"] = (
        result["close"] - result["open"]
    ).abs()

    return result.replace(
        [np.inf, -np.inf], np.nan
    ).dropna().copy()


# ============================================================
# MARKET DATA — TWELVE DATA
# ============================================================

@st.cache_data(ttl=45, show_spinner=False)
def get_candles(api_key, symbol, interval, outputsize=500):
    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": api_key,
        "format": "JSON",
        "timezone": "UTC",
    }

    response = requests.get(
        url,
        params=params,
        timeout=25,
    )
    response.raise_for_status()

    payload = response.json()

    if "values" not in payload:
        message = payload.get(
            "message",
            payload.get("code", "No candle data"),
        )
        raise ValueError(
            f"{symbol} {interval}: {message}"
        )

    df = pd.DataFrame(payload["values"])

    for column in ("open", "high", "low", "close"):
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True,
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "datetime",
            "open",
            "high",
            "low",
            "close",
        ]
    )

    df = (
        df.sort_values("datetime")
        .drop_duplicates("datetime")
        .set_index("datetime")
    )

    if len(df) < 220:
        raise ValueError(
            f"Not enough historical candles for {symbol} {interval}."
        )

    # Never use a candle that may still be forming.
    last_allowed_time = utc_now() - interval_delta(interval)

    df = df[df.index <= last_allowed_time]

    if len(df) < 210:
        raise ValueError(
            f"Not enough CLOSED candles for {symbol} {interval}."
        )

    return df.copy()


# ============================================================
# MARKET SESSION — DST AWARE
# ============================================================

def market_session(now=None):
    now = now or utc_now()

    london = now.astimezone(
        ZoneInfo("Europe/London")
    )
    new_york = now.astimezone(
        ZoneInfo("America/New_York")
    )

    london_open = (
        london.weekday() < 5
        and 8 <= london.hour < 17
    )

    ny_open = (
        new_york.weekday() < 5
        and 8 <= new_york.hour < 17
    )

    if london_open and ny_open:
        return "LONDON / NEW YORK OVERLAP"

    if london_open:
        return "LONDON"

    if ny_open:
        return "NEW YORK"

    if now.weekday() < 5:
        return "ASIA / OFF SESSION"

    return "WEEKEND"


# ============================================================
# TREND CLASSIFICATION
# ============================================================

def classify_trend(row):
    if (
        row["ema20"] > row["ema50"]
        and row["close"] > row["ema200"]
        and row["plus_di"] > row["minus_di"]
    ):
        return "BULLISH"

    if (
        row["ema20"] < row["ema50"]
        and row["close"] < row["ema200"]
        and row["minus_di"] > row["plus_di"]
    ):
        return "BEARISH"

    return "SIDEWAYS"


# ============================================================
# STRATEGY ENGINE
# Rule-Based Multi-Timeframe Trend Following
# No forced trades. No invented market prices.
# ============================================================

def evaluate_direction(entry_df, trend_df, confirm_df, profile):
    cfg = PROFILES[profile]

    entry = add_indicators(entry_df)
    trend = add_indicators(trend_df)
    confirm = add_indicators(confirm_df)

    if min(len(entry), len(trend), len(confirm)) < 5:
        raise ValueError(
            f"Insufficient candles for {profile} analysis."
        )

    current = entry.iloc[-1]
    previous = entry.iloc[-2]
    trend_now = trend.iloc[-1]
    confirm_now = confirm.iloc[-1]

    price = safe_float(current["close"])
    volatility = safe_float(current["atr"])

    if (
        not np.isfinite(price)
        or not np.isfinite(volatility)
        or volatility <= 0
    ):
        raise ValueError("Invalid price or ATR data.")

    trend_direction = classify_trend(trend_now)
    confirm_direction = classify_trend(confirm_now)

    entry_direction = classify_trend(current)

    long_checks = {
        "Entry timeframe bullish": (
            entry_direction == "BULLISH"
        ),
        "Main timeframe bullish": (
            trend_direction == "BULLISH"
        ),
        "Higher timeframe confirmation": (
            confirm_direction == "BULLISH"
        ),
        "ADX trend strength": (
            current["adx"] >= cfg["min_adx"]
        ),
        "Directional strength positive": (
            current["plus_di"] > current["minus_di"]
        ),
        "RSI bullish range": (
            50 <= current["rsi"] <= 70
        ),
        "Price above EMA20": (
            price > current["ema20"]
        ),
        "EMA20 above EMA50": (
            current["ema20"] > current["ema50"]
        ),
        "Bullish candle": (
            current["close"] > current["open"]
        ),
        "Candle confirmation": (
            current["close"] > previous["high"]
        ),
    }

    short_checks = {
        "Entry timeframe bearish": (
            entry_direction == "BEARISH"
        ),
        "Main timeframe bearish": (
            trend_direction == "BEARISH"
        ),
        "Higher timeframe confirmation": (
            confirm_direction == "BEARISH"
        ),
        "ADX trend strength": (
            current["adx"] >= cfg["min_adx"]
        ),
        "Directional strength negative": (
            current["minus_di"] > current["plus_di"]
        ),
        "RSI bearish range": (
            30 <= current["rsi"] <= 50
        ),
        "Price below EMA20": (
            price < current["ema20"]
        ),
        "EMA20 below EMA50": (
            current["ema20"] < current["ema50"]
        ),
        "Bearish candle": (
            current["close"] < current["open"]
        ),
        "Candle confirmation": (
            current["close"] < previous["low"]
        ),
    }

    # Weighted scoring: trend alignment is more important
    # than a single candle or momentum reading.
    weights = {
        "Entry timeframe": 15,
        "Main timeframe": 20,
        "Higher timeframe": 15,
        "ADX": 10,
        "DI direction": 10,
        "RSI": 8,
        "EMA20": 7,
        "EMA alignment": 5,
        "Candle": 5,
        "Break confirmation": 5,
    }

    def calculate_score(checks):
        passed = [
            "Entry timeframe",
            "Main timeframe",
            "Higher timeframe",
            "ADX",
            "DI direction",
            "RSI",
            "EMA20",
            "EMA alignment",
            "Candle",
            "Break confirmation",
        ]

        keys = list(checks.keys())

        score = 0
        for key, weight_name in zip(keys, passed):
            if checks[key]:
                score += weights[weight_name]

        return int(round(score))

    long_score = calculate_score(long_checks)
    short_score = calculate_score(short_checks)

    if long_score > short_score:
        side = "BUY"
        score = long_score
        checks = long_checks
    elif short_score > long_score:
        side = "SELL"
        score = short_score
        checks = short_checks
    else:
        side = "WAIT"
        score = max(long_score, short_score)
        checks = {}

    # Conflicting higher-timeframe direction means no trade.
    if side == "BUY" and (
        trend_direction != "BULLISH"
        or confirm_direction != "BULLISH"
    ):
        side = "WAIT"

    if side == "SELL" and (
        trend_direction != "BEARISH"
        or confirm_direction != "BEARISH"
    ):
        side = "WAIT"

    if score < cfg["min_score"]:
        side = "WAIT"

    # A trend score alone is not enough:
    # the price must be in a reasonable pullback area.
    ema_distance = abs(
        price - float(current["ema20"])
    ) / volatility

    pullback_valid = (
        cfg["pullback_atr_min"]
        <= ema_distance
        <= cfg["pullback_atr_max"]
    )

    # Reject abnormally large candles.
    candle_range = safe_float(
        current["high"] - current["low"]
    )

    abnormal_candle = (
        candle_range > 2.8 * volatility
    )

    if not pullback_valid or abnormal_candle:
        side = "WAIT"

    # Entry, SL and targets are calculated once here.
    # Published values must be saved as a permanent snapshot
    # in Part 2 and must never be recalculated on refresh.

    if side == "BUY":
        direction = 1

        recent_swing = float(
            entry["low"].tail(10).min()
        )

        proposed_entry = float(
            current["ema20"]
        )

        if proposed_entry >= price:
            side = "WAIT"

    elif side == "SELL":
        direction = -1

        recent_swing = float(
            entry["high"].tail(10).max()
        )

        proposed_entry = float(
            current["ema20"]
        )

        if proposed_entry <= price:
            side = "WAIT"

    else:
        direction = 0
        recent_swing = np.nan
        proposed_entry = np.nan

    if side in ("BUY", "SELL"):
        # Structure-based stop with an ATR safety distance.
        if side == "BUY":
            sl = min(
                recent_swing,
                proposed_entry - cfg["atr_sl"] * volatility,
            )
            risk_distance = proposed_entry - sl
        else:
            sl = max(
                recent_swing,
                proposed_entry + cfg["atr_sl"] * volatility,
            )
            risk_distance = sl - proposed_entry

        if (
            not np.isfinite(risk_distance)
            or risk_distance <= 0
        ):
            side = "WAIT"

    if side in ("BUY", "SELL"):
        # Targets are risk multiples. They are not claims
        # that market structure guarantees these prices.
        tps = [
            proposed_entry + direction * risk_distance * 1.0,
            proposed_entry + direction * risk_distance * 2.0,
            proposed_entry + direction * risk_distance * 3.0,
        ]

        entry_low = proposed_entry
        entry_high = proposed_entry

        grade = (
            "A" if score >= 85
            else "B" if score >= 70
            else "C"
        )

        risk_label = (
            "LOWER SETUP RISK" if score >= 85
            else "MODERATE SETUP"
            if score >= 70
            else "WEAKER SETUP"
        )

        entry_type = "LIMIT"
        reason = "TREND + PULLBACK CONDITIONS CONFIRMED"

    else:
        score = max(score, 0)
        grade = "—"
        risk_label = "NO TRADE"
        entry_type = "NONE"
        proposed_entry = np.nan
        entry_low = np.nan
        entry_high = np.nan
        sl = np.nan
        tps = [np.nan, np.nan, np.nan]
        risk_distance = np.nan
        reason = "WAIT FOR A VALID TREND AND PULLBACK"

    return {
        "side": side,
        "score": int(score),
        "grade": grade,
        "risk_label": risk_label,
        "entry_type": entry_type,
        "entry": safe_float(proposed_entry),
        "entry_low": safe_float(entry_low),
        "entry_high": safe_float(entry_high),
        "sl": safe_float(sl),
        "tps": [safe_float(x) for x in tps],
        "risk_distance": safe_float(risk_distance),
        "reason": reason,
        "checks": checks,
        "price": price,
        "atr": volatility,
        "rsi": float(current["rsi"]),
        "adx": float(current["adx"]),
        "trend_main": trend_direction,
        "trend_confirm": confirm_direction,
        "trend_entry": entry_direction,
        "ema20": float(current["ema20"]),
        "ema50": float(current["ema50"]),
        "session": market_session(),
        "candle_time": entry.index[-1].to_pydatetime(),
        "pullback_distance_atr": float(ema_distance),
    }


def build_profile_signal(api_key, asset_key, profile):
    asset = ASSETS[asset_key]
    cfg = PROFILES[profile]

    entry_data = get_candles(
        api_key,
        asset["symbol"],
        cfg["entry_tf"],
        500,
    )

    trend_data = get_candles(
        api_key,
        asset["symbol"],
        cfg["trend_tf"],
        400,
    )

    confirm_data = get_candles(
        api_key,
        asset["symbol"],
        cfg["confirm_tf"],
        400,
    )

    signal = evaluate_direction(
        entry_data,
        trend_data,
        confirm_data,
        profile,
    )

    signal.update({
        "asset": asset_key,
        "symbol": asset["symbol"],
        "profile": profile,
        "valid_minutes": VALIDITY_MINUTES,
        "created_at": utc_now(),
        "status": (
            "PENDING_LIMIT"
            if signal["side"] in ("BUY", "SELL")
            else "WAIT"
        ),
        "timeframe": (
            f"{cfg['entry_tf']} / "
            f"{cfg['trend_tf']} / "
            f"{cfg['confirm_tf']}"
        ),
    })

    return signal# ============================================================
# KHANGOLDSIGNAL — PART 2 OF 4
# SUPABASE DATABASE + IMMUTABLE SIGNAL SNAPSHOTS
# ============================================================

import copy

SUPABASE_TIMEOUT = 15
SIGNAL_TABLE = "signals"
OUTBOX_TABLE = "telegram_outbox"


def supabase_config():
    """Read Supabase credentials from Streamlit secrets."""
    try:
        url = str(st.secrets.get("SUPABASE_URL", "")).strip()
        key = str(st.secrets.get("SUPABASE_KEY", "")).strip()
    except Exception:
        url, key = "", ""

    if not url or not key:
        return None, None

    return url.rstrip("/"), key


def db_request(method, table, params=None, payload=None):
    """Supabase REST request."""
    url, key = supabase_config()

    if not url or not key:
        raise RuntimeError("Supabase credentials are missing.")

    endpoint = f"{url}/rest/v1/{table}"

    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    if method.upper() == "POST":
        headers["Prefer"] = "return=representation"
    elif method.upper() == "PATCH":
        headers["Prefer"] = "return=representation"

    response = requests.request(
        method=method.upper(),
        url=endpoint,
        headers=headers,
        params=params,
        json=payload,
        timeout=SUPABASE_TIMEOUT,
    )

    if response.status_code >= 400:
        raise RuntimeError(
            f"Supabase error {response.status_code}: "
            f"{response.text[:500]}"
        )

    if not response.text.strip():
        return []

    return response.json()


def db_get_signals(asset=None, profile=None, statuses=None, limit=100):
    params = {
        "select": "*",
        "order": "published_at.desc",
        "limit": str(limit),
    }

    if asset:
        params["asset"] = f"eq.{asset}"

    if profile:
        params["profile"] = f"eq.{profile}"

    if statuses:
        params["status"] = "in.(" + ",".join(statuses) + ")"

    return db_request("GET", SIGNAL_TABLE, params=params)


def db_get_signal(signal_id):
    rows = db_request(
        "GET",
        SIGNAL_TABLE,
        params={
            "select": "*",
            "signal_id": f"eq.{signal_id}",
            "limit": "1",
        },
    )
    return rows[0] if rows else None


def db_insert_signal(signal):
    rows = db_request(
        "POST",
        SIGNAL_TABLE,
        payload=signal,
    )

    if not rows:
        raise RuntimeError("Signal was not returned after database insert.")

    return rows[0]


def db_update_signal(signal_id, updates):
    updates = dict(updates)
    updates["updated_at"] = iso_utc(utc_now())

    return db_request(
        "PATCH",
        SIGNAL_TABLE,
        params={"signal_id": f"eq.{signal_id}"},
        payload=updates,
    )


def db_get_latest_for_route(asset, profile):
    rows = db_get_signals(
        asset=asset,
        profile=profile,
        limit=30,
    )
    return rows[0] if rows else None


def db_get_active_signals():
    return db_request(
        "GET",
        SIGNAL_TABLE,
        params={
            "select": "*",
            "status": "in.(PENDING_LIMIT,ACTIVE_TRADE,TP1_HIT,TP2_HIT)",
            "order": "published_at.asc",
            "limit": "300",
        },
    )


def db_get_history(limit=200):
    return db_request(
        "GET",
        SIGNAL_TABLE,
        params={
            "select": "*",
            "order": "published_at.desc",
            "limit": str(limit),
        },
    )


def signal_snapshot(candidate):
    """
    Values that must never be recalculated for an already-published signal.
    """
    keys = [
        "asset",
        "symbol",
        "profile",
        "side",
        "score",
        "grade",
        "entry_type",
        "entry",
        "entry_low",
        "entry_high",
        "sl",
        "tp1",
        "tp2",
        "tp3",
        "risk_distance",
        "reason",
        "session",
        "timeframe",
        "candle_time",
    ]

    snapshot = {}

    for key in keys:
        if key in candidate:
            snapshot[key] = candidate[key]

    return snapshot


def make_signal_database_row(candidate):
    now = utc_now()
    published_at = iso_utc(now)
    expires_at = iso_utc(
        now + timedelta(minutes=VALIDITY_MINUTES)
    )

    snapshot = signal_snapshot(candidate)

    row = {
        "signal_id": str(uuid.uuid4()),
        "asset": str(candidate["asset"]),
        "symbol": str(candidate["symbol"]),
        "profile": str(candidate["profile"]),
        "side": str(candidate["side"]),
        "score": int(candidate["score"]),
        "grade": str(candidate.get("grade", "B")),
        "entry_type": str(candidate.get("entry_type", "LIMIT")),
        "entry": float(candidate["entry"]),
        "entry_low": float(candidate.get("entry_low", candidate["entry"])),
        "entry_high": float(candidate.get("entry_high", candidate["entry"])),
        "sl": float(candidate["sl"]),
        "tp1": float(candidate["tp1"]),
        "tp2": float(candidate["tp2"]),
        "tp3": float(candidate["tp3"]),
        "risk_distance": float(candidate.get("risk_distance", 0)),
        "reason": str(candidate.get("reason", "")),
        "session": str(candidate.get("session", "")),
        "timeframe": str(candidate.get("timeframe", "")),
        "candle_time": candidate.get("candle_time"),
        "published_at": published_at,
        "expires_at": expires_at,
        "status": "PENDING_LIMIT",
        "telegram_status": "PENDING",
        "snapshot": snapshot,
        "reviewed_at": None,
        "closed_at": None,
        "outcome": None,
        "created_at": published_at,
        "updated_at": published_at,
    }

    return row


def candidate_is_valid(candidate):
    return (
        isinstance(candidate, dict)
        and candidate.get("side") in ("BUY", "SELL")
        and candidate.get("entry") is not None
        and candidate.get("sl") is not None
        and candidate.get("tp1") is not None
        and candidate.get("tp2") is not None
        and candidate.get("tp3") is not None
    )


def is_same_route_signal(signal, asset, profile):
    return (
        signal
        and signal.get("asset") == asset
        and signal.get("profile") == profile
    )


def is_signal_active(signal):
    return signal and signal.get("status") in (
        "PENDING_LIMIT",
        "ACTIVE_TRADE",
        "TP1_HIT",
        "TP2_HIT",
    )


def get_candidate_price(candidate):
    value = candidate.get("current_price")

    if value is None:
        value = candidate.get("price")

    return safe_float(value, None)


def review_pending_signal(existing, candidate):
    """
    Review a pending signal without changing its published prices.

    At expiry:
    - Keep it if the same direction and setup remain valid.
    - Extend expiry by another 45 minutes.
    - Otherwise invalidate it.
    """
    if not existing or existing.get("status") != "PENDING_LIMIT":
        return existing

    expires_at = parse_datetime(existing.get("expires_at"))

    if not expires_at:
        return db_update_signal(
            existing["signal_id"],
            {"status": "INVALIDATED"},
        )

    now = utc_now()

    if now < expires_at:
        return existing

    if not candidate_is_valid(candidate):
        db_update_signal(
            existing["signal_id"],
            {
                "status": "INVALIDATED",
                "reviewed_at": iso_utc(now),
            },
        )
        return db_get_signal(existing["signal_id"])

    same_direction = (
        candidate.get("side") == existing.get("side")
    )

    old_entry = safe_float(existing.get("entry"), None)
    current_price = get_candidate_price(candidate)

    if old_entry is None or current_price is None:
        actionable = False
    else:
        atr_value = safe_float(candidate.get("atr"), 0)
        tolerance = max(atr_value * 0.50, old_entry * 0.0005)

        if existing.get("side") == "BUY":
            actionable = (
                current_price > old_entry
                and current_price - old_entry <= tolerance
            )
        else:
            actionable = (
                current_price < old_entry
                and old_entry - current_price <= tolerance
            )

    if same_direction and actionable:
        new_expiry = iso_utc(
            now + timedelta(minutes=VALIDITY_MINUTES)
        )

        db_update_signal(
            existing["signal_id"],
            {
                "expires_at": new_expiry,
                "reviewed_at": iso_utc(now),
                "status": "PENDING_LIMIT",
            },
        )
    else:
        db_update_signal(
            existing["signal_id"],
            {
                "status": "INVALIDATED",
                "reviewed_at": iso_utc(now),
            },
        )

    return db_get_signal(existing["signal_id"])


def get_or_create_route_signal(asset, profile, candidate):
    """
    Return the persisted signal for this route.
    Never replace an unresolved active signal silently.
    """
    existing = db_get_latest_for_route(asset, profile)

    if is_signal_active(existing):
        if existing.get("status") == "PENDING_LIMIT":
            return review_pending_signal(existing, candidate)

        # Open trades are managed separately.
        # A new candidate must not overwrite an open trade.
        return existing

    if not candidate_is_valid(candidate):
        return existing

    new_row = make_signal_database_row(candidate)
    return db_insert_signal(new_row)


def load_signal_snapshot(signal):
    """
    Always display the database snapshot, not a fresh calculation.
    """
    if not signal:
        return None

    snapshot = signal.get("snapshot")

    if isinstance(snapshot, str):
        try:
            snapshot = json.loads(snapshot)
        except Exception:
            snapshot = {}

    if not isinstance(snapshot, dict):
        snapshot = {}

    result = copy.deepcopy(snapshot)

    # Database lifecycle fields are allowed to change.
    result["signal_id"] = signal.get("signal_id")
    result["signal_number"] = signal.get("signal_number")
    result["status"] = signal.get("status")
    result["published_at"] = signal.get("published_at")
    result["expires_at"] = signal.get("expires_at")
    result["telegram_status"] = signal.get("telegram_status")
    result["outcome"] = signal.get("outcome")

    return result


def signal_countdown(signal):
    expires_at = parse_datetime(signal.get("expires_at"))

    if not expires_at:
        return 0

    seconds = int((expires_at - utc_now()).total_seconds())
    return max(0, seconds)


def countdown_text(signal):
    seconds = signal_countdown(signal)
    minutes, remaining = divmod(seconds, 60)
    return f"{minutes:02d}:{remaining:02d}"


def format_signal_price(value):
    number = safe_float(value, None)

    if number is None:
        return "N/A"

    return f"{number:,.2f}"


def signal_grade(score):
    score = safe_float(score, 0)

    if score >= 90:
        return "A+"
    if score >= 80:
        return "A"
    if score >= 70:
        return "B"
    if score >= 65:
        return "C"

    return "WAIT"


# ============================================================
# END OF PART 2
# ============================================================# ============================================================
# KHANGOLDSIGNAL — PART 3 OF 4
# TELEGRAM OUTBOX + TRADE MANAGEMENT + RESULTS
# ============================================================

from datetime import timedelta


def telegram_config():
    try:
        token = str(st.secrets.get("TELEGRAM_BOT_TOKEN", "")).strip()
        chat_id = str(st.secrets.get("TELEGRAM_CHAT_ID", "")).strip()
    except Exception:
        token, chat_id = "", ""

    if not token or not chat_id:
        return None, None

    return token, chat_id


def format_telegram_message(signal):
    side = signal.get("side", "N/A")
    symbol = signal.get("symbol", "N/A")
    profile = signal.get("profile", "N/A")

    return (
        "🟡 KHAN GOLD SIGNAL\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"Asset: {symbol}\n"
        f"Strategy: {profile}\n"
        f"Direction: {side}\n"
        f"Score: {signal.get('score', 'N/A')}/100\n"
        f"Grade: {signal.get('grade', 'N/A')}\n"
        f"Entry Type: {signal.get('entry_type', 'LIMIT')}\n"
        f"Entry: {format_signal_price(signal.get('entry'))}\n"
        f"SL: {format_signal_price(signal.get('sl'))}\n"
        f"TP1: {format_signal_price(signal.get('tp1'))}\n"
        f"TP2: {format_signal_price(signal.get('tp2'))}\n"
        f"TP3: {format_signal_price(signal.get('tp3'))}\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"Session: {signal.get('session', 'N/A')}\n"
        f"Validity: {VALIDITY_MINUTES} minutes initially\n"
        "Risk carefully. This is not an order execution service."
    )


def create_telegram_outbox(signal):
    """
    Create one outbox row per signal.
    Database unique constraint prevents duplicate rows.
    """
    signal_id = signal.get("signal_id")

    if not signal_id:
        return None

    message = format_telegram_message(signal)

    payload = {
        "signal_id": signal_id,
        "channel": "telegram",
        "message": message,
        "status": "PENDING",
        "attempts": 0,
        "created_at": iso_utc(utc_now()),
    }

    try:
        rows = db_request(
            "POST",
            OUTBOX_TABLE,
            params={"on_conflict": "signal_id,channel"},
            payload=payload,
        )
        return rows[0] if rows else None

    except Exception:
        # It may already exist due to the unique constraint.
        rows = db_request(
            "GET",
            OUTBOX_TABLE,
            params={
                "select": "*",
                "signal_id": f"eq.{signal_id}",
                "channel": "eq.telegram",
                "limit": "1",
            },
        )
        return rows[0] if rows else None


def send_telegram_message(message):
    token, chat_id = telegram_config()

    if not token or not chat_id:
        raise RuntimeError("Telegram credentials are missing.")

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    response = requests.post(
        url,
        json={
            "chat_id": chat_id,
            "text": message,
            "disable_web_page_preview": True,
        },
        timeout=20,
    )

    response.raise_for_status()
    result = response.json()

    if not result.get("ok"):
        raise RuntimeError("Telegram did not confirm message delivery.")

    return True


def process_telegram_outbox(signal):
    """
    Send only a PENDING outbox item.

    UNKNOWN is never automatically resent because the previous
    request may have reached Telegram despite a network timeout.
    """
    signal_id = signal.get("signal_id")

    if not signal_id:
        return "NO_SIGNAL"

    outbox = create_telegram_outbox(signal)

    if not outbox:
        return "OUTBOX_ERROR"

    status = outbox.get("status")

    if status in ("SENT", "UNKNOWN"):
        return status

    if status != "PENDING":
        return status

    outbox_id = outbox.get("id")

    db_request(
        "PATCH",
        OUTBOX_TABLE,
        params={"id": f"eq.{outbox_id}"},
        payload={
            "status": "SENDING",
            "attempts": int(outbox.get("attempts") or 0) + 1,
        },
    )

    try:
        send_telegram_message(outbox["message"])

        db_request(
            "PATCH",
            OUTBOX_TABLE,
            params={"id": f"eq.{outbox_id}"},
            payload={
                "status": "SENT",
                "sent_at": iso_utc(utc_now()),
                "last_error": None,
            },
        )

        db_update_signal(
            signal_id,
            {"telegram_status": "SENT"},
        )

        return "SENT"

    except requests.Timeout as exc:
        # Delivery is uncertain; do not resend automatically.
        db_request(
            "PATCH",
            OUTBOX_TABLE,
            params={"id": f"eq.{outbox_id}"},
            payload={
                "status": "UNKNOWN",
                "last_error": str(exc)[:500],
            },
        )
        return "UNKNOWN"

    except Exception as exc:
        db_request(
            "PATCH",
            OUTBOX_TABLE,
            params={"id": f"eq.{outbox_id}"},
            payload={
                "status": "FAILED",
                "last_error": str(exc)[:500],
            },
        )
        db_update_signal(
            signal_id,
            {"telegram_status": "FAILED"},
        )
        return "FAILED"


def candle_touched_level(candle, level):
    if candle is None or level is None:
        return False

    low = safe_float(candle.get("low"), None)
    high = safe_float(candle.get("high"), None)
    level = safe_float(level, None)

    if low is None or high is None or level is None:
        return False

    return low <= level <= high


def candle_touched_entry(candle, signal):
    low = safe_float(candle.get("low"), None)
    high = safe_float(candle.get("high"), None)

    entry_low = safe_float(signal.get("entry_low"), None)
    entry_high = safe_float(signal.get("entry_high"), None)

    if None in (low, high, entry_low, entry_high):
        return False

    return high >= entry_low and low <= entry_high


def candle_hits_sl(candle, signal):
    side = signal.get("side")
    sl = safe_float(signal.get("sl"), None)

    if sl is None:
        return False

    low = safe_float(candle.get("low"), None)
    high = safe_float(candle.get("high"), None)

    if low is None or high is None:
        return False

    if side == "BUY":
        return low <= sl

    return high >= sl


def candle_hits_tp(candle, signal, tp_key):
    side = signal.get("side")
    target = safe_float(signal.get(tp_key), None)

    if target is None:
        return False

    low = safe_float(candle.get("low"), None)
    high = safe_float(candle.get("high"), None)

    if low is None or high is None:
        return False

    if side == "BUY":
        return high >= target

    return low <= target


def manage_one_trade(signal):
    """
    Estimate limit fill and manage trade using the latest candle.
    Uses persisted prices only; never recalculates Entry/SL/TP.
    """
    signal_id = signal.get("signal_id")
    asset = signal.get("asset")
    profile = signal.get("profile")

    if not signal_id or not asset or not profile:
        return signal

    symbol = ASSETS.get(asset, {}).get("symbol")

    if not symbol:
        return signal

    timeframe = PROFILES.get(profile, {}).get("entry_tf")

    if not timeframe:
        return signal

    try:
        df = get_candles(
            st.secrets.get("TWELVE_DATA_API_KEY", ""),
            symbol,
            timeframe,
        )
    except Exception:
        return signal

    if df is None or df.empty:
        return signal

    candle = df.iloc[-1].to_dict()
    status = signal.get("status")

    # Pending limit: estimate fill if the candle reaches entry.
    if status == "PENDING_LIMIT":
        published = parse_datetime(signal.get("published_at"))

        candle_time = candle.get("datetime")
        if published and candle_time:
            parsed_candle_time = parse_datetime(candle_time)
            if parsed_candle_time and parsed_candle_time < published:
                return signal

        if candle_touched_entry(candle, signal):
            db_update_signal(
                signal_id,
                {
                    "status": "ACTIVE_TRADE",
                    "filled_at": iso_utc(utc_now()),
                    "fill_type": "ESTIMATED_FROM_CANDLE",
                },
            )
            return db_get_signal(signal_id)

        return signal

    if status not in ("ACTIVE_TRADE", "TP1_HIT", "TP2_HIT"):
        return signal

    # Conservative candle rule: SL is checked before targets.
    if candle_hits_sl(candle, signal):
        db_update_signal(
            signal_id,
            {
                "status": "SL_HIT",
                "closed_at": iso_utc(utc_now()),
                "outcome": "LOSS",
            },
        )
        return db_get_signal(signal_id)

    if status == "ACTIVE_TRADE":
        if candle_hits_tp(candle, signal, "tp1"):
            db_update_signal(
                signal_id,
                {
                    "status": "TP1_HIT",
                    "tp1_hit_at": iso_utc(utc_now()),
                    "realized_allocation": 50,
                },
            )
            return db_get_signal(signal_id)

    if status == "TP1_HIT":
        if candle_hits_tp(candle, signal, "tp2"):
            db_update_signal(
                signal_id,
                {
                    "status": "TP2_HIT",
                    "tp2_hit_at": iso_utc(utc_now()),
                    "realized_allocation": 80,
                },
            )
            return db_get_signal(signal_id)

    if status == "TP2_HIT":
        if candle_hits_tp(candle, signal, "tp3"):
            db_update_signal(
                signal_id,
                {
                    "status": "TP3_HIT",
                    "closed_at": iso_utc(utc_now()),
                    "outcome": "WIN",
                    "realized_allocation": 100,
                },
            )
            return db_get_signal(signal_id)

    return signal


def manage_all_active_trades():
    try:
        active = db_get_active_signals()
    except Exception:
        return []

    updated = []

    for signal in active:
        try:
            updated.append(manage_one_trade(signal))
        except Exception:
            updated.append(signal)

    return updated


def set_manual_outcome(signal_id, outcome):
    """
    Manual result is allowed only after a terminal lifecycle status.
    Prevents editing the same result repeatedly.
    """
    if outcome not in ("WIN", "LOSS", "BREAKEVEN"):
        raise ValueError("Invalid outcome.")

    signal = db_get_signal(signal_id)

    if not signal:
        raise RuntimeError("Signal not found.")

    if signal.get("outcome"):
        raise RuntimeError("This signal already has a recorded outcome.")

    terminal_statuses = (
        "TP3_HIT",
        "SL_HIT",
        "INVALIDATED",
        "CANCELLED",
        "CLOSED",
    )

    if signal.get("status") not in terminal_statuses:
        raise RuntimeError(
            "Close or invalidate the trade before recording its result."
        )

    db_update_signal(
        signal_id,
        {
            "outcome": outcome,
            "closed_at": signal.get("closed_at") or iso_utc(utc_now()),
        },
    )

    return db_get_signal(signal_id)


def get_performance_summary():
    rows = db_get_history(limit=1000)

    completed = [
        row for row in rows
        if row.get("outcome") in ("WIN", "LOSS", "BREAKEVEN")
    ]

    wins = sum(1 for row in completed if row["outcome"] == "WIN")
    losses = sum(1 for row in completed if row["outcome"] == "LOSS")
    breakeven = sum(
        1 for row in completed if row["outcome"] == "BREAKEVEN"
    )

    total = len(completed)

    win_rate = (wins / total * 100) if total else 0

    return {
        "total": total,
        "wins": wins,
        "losses": losses,
        "breakeven": breakeven,
        "win_rate": win_rate,
    }


# ============================================================
# END OF PART 3
# ============================================================# ============================================================
# KHANGOLDSIGNAL — PART 4 OF 4
# COMPLETE DASHBOARD + GOLD/BTC + THREE PROFILES
# ============================================================

try:
    from streamlit_autorefresh import st_autorefresh
    AUTO_REFRESH_AVAILABLE = True
except Exception:
    AUTO_REFRESH_AVAILABLE = False


def get_api_key():
    try:
        return str(st.secrets.get("TWELVE_DATA_API_KEY", "")).strip()
    except Exception:
        return ""


def enrich_candidate(candidate, asset_key, profile):
    """Attach route metadata and current market price."""
    if not isinstance(candidate, dict):
        return candidate

    candidate = dict(candidate)

    asset_info = ASSETS.get(asset_key, {})

    candidate.setdefault("asset", asset_key)
    candidate.setdefault("symbol", asset_info.get("symbol", asset_key))
    candidate.setdefault("profile", profile)
    candidate.setdefault("grade", signal_grade(candidate.get("score", 0)))

    # Fetch current price for pending-signal review.
    try:
        api_key = get_api_key()
        df = get_candles(
            api_key,
            asset_info["symbol"],
            PROFILES[profile]["entry_tf"],
        )

        if df is not None and not df.empty:
            last = df.iloc[-1]
            candidate["current_price"] = safe_float(last.get("close"), None)

            if "atr" in last:
                candidate["atr"] = safe_float(last.get("atr"), 0)

            if "datetime" in last:
                candidate["candle_time"] = str(last["datetime"])

    except Exception:
        pass

    return candidate


def get_route_state(asset_key, profile):
    """Calculate a candidate, then return the persisted route signal."""
    api_key = get_api_key()

    if not api_key:
        return None, "Twelve Data API key is missing."

    try:
        candidate = build_profile_signal(
            api_key,
            asset_key,
            profile,
        )

        candidate = enrich_candidate(
            candidate,
            asset_key,
            profile,
        )

        existing = get_or_create_route_signal(
            asset_key,
            profile,
            candidate,
        )

        if not existing:
            return None, "No valid setup at this time."

        # Publish Telegram message only for a new signal whose
        # outbox item is still pending.
        if existing.get("telegram_status") in ("PENDING", None):
            snapshot = load_signal_snapshot(existing)
            if snapshot:
                try:
                    process_telegram_outbox(snapshot)
                except Exception:
                    pass

        return existing, None

    except Exception as exc:
        return None, str(exc)


def show_signal_card(signal):
    data = load_signal_snapshot(signal)

    if not data:
        st.info("No signal available.")
        return

    status = signal.get("status", "UNKNOWN")
    side = data.get("side", "WAIT")

    with st.container(border=True):
        top1, top2, top3 = st.columns([2, 1, 1])

        with top1:
            st.subheader(
                f"{data.get('symbol', 'Asset')} — "
                f"{data.get('profile', 'Profile')}"
            )

        with top2:
            st.metric("Direction", side)

        with top3:
            st.metric("Score", f"{data.get('score', 0)}/100")

        st.caption(
            f"Signal #{signal.get('signal_number', '—')} "
            f" | Grade: {data.get('grade', '—')} "
            f" | Status: {status}"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Entry",
            format_signal_price(data.get("entry")),
        )
        c2.metric(
            "Stop Loss",
            format_signal_price(data.get("sl")),
        )
        c3.metric(
            "Session",
            data.get("session", "—"),
        )

        p1, p2, p3 = st.columns(3)

        p1.metric("TP1", format_signal_price(data.get("tp1")))
        p2.metric("TP2", format_signal_price(data.get("tp2")))
        p3.metric("TP3", format_signal_price(data.get("tp3")))

        st.caption(
            f"Entry type: {data.get('entry_type', 'LIMIT')} "
            f"| Timeframe: {data.get('timeframe', '—')}"
        )

        if status == "PENDING_LIMIT":
            st.info(
                f"Validity countdown: {countdown_text(signal)}"
            )
            st.caption(
                "At expiry the setup is reviewed. "
                "The published Entry, SL and TP values are not changed."
            )

        elif status in ("ACTIVE_TRADE", "TP1_HIT", "TP2_HIT"):
            st.success(f"Trade lifecycle: {status}")

        elif status in ("INVALIDATED", "CANCELLED"):
            st.warning(f"Signal ended: {status}")

        elif status in ("TP3_HIT", "SL_HIT"):
            st.success(f"Trade completed: {status}")

        if data.get("reason"):
            with st.expander("Analysis details"):
                st.write(data["reason"])

        st.caption(
            f"Published: {signal.get('published_at', '—')}"
        )


def show_route(asset_key, profile):
    st.markdown(f"#### {profile}")

    try:
        signal, error = get_route_state(asset_key, profile)

        if error:
            st.warning(error)
            return

        if signal:
            show_signal_card(signal)
        else:
            st.info("No active signal.")

    except Exception as exc:
        st.error(f"Route error: {exc}")


def show_performance():
    st.subheader("Performance")

    try:
        stats = get_performance_summary()

        a, b, c, d, e = st.columns(5)

        a.metric("Closed", stats["total"])
        b.metric("Wins", stats["wins"])
        c.metric("Losses", stats["losses"])
        d.metric("Breakeven", stats["breakeven"])
        e.metric("Win Rate", f"{stats['win_rate']:.1f}%")

        st.caption(
            "Statistics use only records with a saved WIN, LOSS "
            "or BREAKEVEN outcome. They are not a guarantee of future results."
        )

    except Exception as exc:
        st.warning(f"Could not load performance: {exc}")


def show_history():
    st.subheader("Signal History")

    try:
        rows = db_get_history(limit=200)

        if not rows:
            st.info("No saved signals yet.")
            return

        display_rows = []

        for row in rows:
            display_rows.append({
                "Number": row.get("signal_number"),
                "Asset": row.get("symbol"),
                "Profile": row.get("profile"),
                "Side": row.get("side"),
                "Score": row.get("score"),
                "Entry": row.get("entry"),
                "SL": row.get("sl"),
                "TP1": row.get("tp1"),
                "TP2": row.get("tp2"),
                "TP3": row.get("tp3"),
                "Status": row.get("status"),
                "Outcome": row.get("outcome"),
                "Published": row.get("published_at"),
            })

        st.dataframe(
            display_rows,
            use_container_width=True,
            hide_index=True,
        )

    except Exception as exc:
        st.error(f"History error: {exc}")


def show_trade_management():
    st.subheader("Trade Management")

    try:
        active = db_get_active_signals()

        if not active:
            st.info("No active signals or trades.")
            return

        for signal in active:
            data = load_signal_snapshot(signal)

            with st.expander(
                f"{signal.get('signal_number', '—')} | "
                f"{signal.get('symbol')} | "
                f"{signal.get('profile')} | "
                f"{signal.get('status')}"
            ):
                st.write(
                    f"Direction: {data.get('side')} "
                    f"| Entry: {format_signal_price(data.get('entry'))} "
                    f"| SL: {format_signal_price(data.get('sl'))}"
                )

                st.write(
                    f"TP1: {format_signal_price(data.get('tp1'))} "
                    f"| TP2: {format_signal_price(data.get('tp2'))} "
                    f"| TP3: {format_signal_price(data.get('tp3'))}"
                )

                st.caption(
                    "Trade status is estimated from market candles. "
                    "It is not connected to a broker."
                )

    except Exception as exc:
        st.error(f"Trade management error: {exc}")


def run_dashboard():
    st.set_page_config(
        page_title="KhanGoldSignal",
        page_icon="🟡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🟡 KhanGoldSignal")
    st.caption(
        "Rule-Based Multi-Timeframe Trend Following "
        "| GOLD + BITCOIN"
    )

    with st.sidebar:
        st.header("Control Panel")

        if st.button("🔄 Refresh Market Data", use_container_width=True):
            try:
                st.cache_data.clear()
            except Exception:
                pass
            st.rerun()

        auto_refresh = st.checkbox(
            "Auto refresh",
            value=True,
        )

        st.divider()

        st.caption("Initial signal validity: 45 minutes")
        st.caption("Profiles: Scalping / Intraday / Swing")
        st.caption("Assets: GOLD / BITCOIN")

    if auto_refresh and AUTO_REFRESH_AVAILABLE:
        st_autorefresh(
            interval=15000,
            key="khangold_autorefresh",
        )

    # Update trade lifecycle before showing dashboard.
    try:
        manage_all_active_trades()
    except Exception:
        pass

    tab_gold, tab_btc, tab_trades, tab_history, tab_stats = st.tabs([
        "🥇 GOLD",
        "₿ BITCOIN",
        "Trade Management",
        "History",
        "Performance",
    ])

    with tab_gold:
        st.header("GOLD — XAU/USD")

        for profile in ("Scalping", "Intraday", "Swing"):
            show_route("GOLD", profile)

    with tab_btc:
        st.header("BITCOIN — BTC/USD")

        for profile in ("Scalping", "Intraday", "Swing"):
            show_route("BITCOIN", profile)

    with tab_trades:
        show_trade_management()

    with tab_history:
        show_history()

    with tab_stats:
        show_performance()

    st.divider()
    st.caption(
        "KhanGoldSignal is an analytical signal dashboard. "
        "It does not place orders or guarantee profits. "
        "Always verify execution, spread, slippage and risk independently."
    )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_dashboard()


# ============================================================
# END OF KHANGOLDSIGNAL.PY
# ============================================================
