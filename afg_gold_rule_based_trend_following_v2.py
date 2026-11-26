import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import html

# ============================================================
# AFG GOLD | RULE-BASED TREND FOLLOWING
# One-file Streamlit app. Data: Twelve Data (XAU/USD).
# Signals are decision support, not financial advice.
# ============================================================

st.set_page_config(
    page_title="AFG GOLD | XAUUSD",
    page_icon="🟡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background: #090611; color: #d8c9f5; }
[data-testid="stSidebar"] { background: #17141e; border-right: 1px solid #30283b; }
.block-container { padding-top: 1.2rem; max-width: 1100px; }
h1,h2,h3,p,label { font-family: Georgia, 'Times New Roman', serif; }
div[data-testid="stMetric"] { background:#100b1b; border:1px solid #30233d; padding:12px; border-radius:12px; }
.stButton > button { width:100%; border-radius:10px; min-height:46px; font-weight:800; }
div.stButton > button[kind="primary"] { background:#087a43; border:1px solid #29db83; color:white; }
div.stButton > button[kind="secondary"] { background:#491b28; border:1px solid #d74b65; color:white; }
.small-muted { color:#8d7da8; letter-spacing:2px; font-size:11px; }
</style>
""", unsafe_allow_html=True)

# ----------------------- indicators -------------------------

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

# ----------------------- market data ------------------------

@st.cache_data(ttl=60, show_spinner=False)
def get_candles(api_key, interval, outputsize=500):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": "XAU/USD",
        "interval": interval,
        "outputsize": outputsize,
        "apikey": api_key,
        "format": "JSON",
        "timezone": "UTC",
    }
    response = requests.get(url, params=params, timeout=25)
    response.raise_for_status()
    payload = response.json()
    if "values" not in payload:
        raise ValueError(payload.get("message", payload.get("code", "No candle data returned")))
    df = pd.DataFrame(payload["values"])
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime", "open", "high", "low", "close"])
    return df.sort_values("datetime").set_index("datetime")

# ----------------------- strategy ----------------------------

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
        "candle_time": a.index[-1],
    }

def fmt(v):
    return "—" if pd.isna(v) else f"{v:,.2f}"

# ----------------------- sidebar ----------------------------

st.sidebar.markdown("## 🟡 AFG GOLD")
st.sidebar.caption("MARKET DATA & STRATEGY SETTINGS")
api_key = st.sidebar.text_input("Twelve Data API Key", type="password", help="کلید API خود را از حساب Twelve Data وارد کنید.")
st.sidebar.markdown("### STRATEGY SETTINGS")
min_score = st.sidebar.slider("Minimum signal score", 40, 100, 40, 1)
min_adx = st.sidebar.slider("Minimum ADX trend strength", 10, 35, 16, 1)
use_session = st.sidebar.checkbox("London / New York session filter", value=True)
valid_minutes = st.sidebar.selectbox("Signal validity (minutes)", [15, 30, 45, 60], index=2)
st.sidebar.caption("TP allocation: TP1 = 50%  ·  TP2 = 30%  ·  TP3 = 20%")
st.sidebar.caption("Risk settings are rule-based. No account balance is requested.")
refresh = st.sidebar.button("↻ REFRESH MARKET DATA", use_container_width=True)
if refresh:
    st.cache_data.clear()
    st.rerun()

# ----------------------- header -----------------------------

st.markdown("<div style='font-size:30px;font-weight:900;color:#eee2ff'>🟡 AFG GOLD</div>", unsafe_allow_html=True)
st.markdown("<div class='small-muted'>ORIGINAL RULE-BASED TREND FOLLOWING · XAU/USD</div>", unsafe_allow_html=True)
st.write("")

if not api_key:
    st.info("Enter your Twelve Data API Key in the left settings panel to load live market candles and calculate a signal.")
    st.warning("The signal card below will populate only after real market data is received. No sample price or fabricated signal is used.")
    st.stop()

try:
    with st.spinner("Loading XAU/USD candles and calculating trend conditions..."):
        m15 = get_candles(api_key, "15min", 500)
        h1 = get_candles(api_key, "1h", 300)
        h4 = get_candles(api_key, "4h", 300)
        signal = build_signal(m15, h1, h4, min_adx, min_score, use_session)
except Exception as exc:
    st.error(f"Market data error: {html.escape(str(exc))}")
    st.caption("Check the API key, Twelve Data plan/credits, internet connection, and XAU/USD symbol access.")
    st.stop()

# Persist the validity timer for the same generated signal during reruns.
signature = (signal["side"], signal["candle_time"].isoformat(), signal["price"])
if st.session_state.get("signal_signature") != signature:
    st.session_state["signal_signature"] = signature
    st.session_state["signal_generated_at"] = datetime.now(timezone.utc)
    st.session_state["signal_outcome"] = None

generated = st.session_state["signal_generated_at"]
expires = generated + timedelta(minutes=valid_minutes)
remaining = max(0, int((expires - datetime.now(timezone.utc)).total_seconds()))
if remaining == 0 and signal["side"] != "WAIT":
    signal["side"] = "WAIT"
    signal["reason"] = "SIGNAL EXPIRED — WAIT FOR A NEW SETUP"

side = signal["side"]
is_buy = side == "BUY"
accent = "#22d879" if is_buy else "#ff596d" if side == "SELL" else "#d6a944"
bg = "#041b0d" if is_buy else "#250911" if side == "SELL" else "#211b0b"
symbol = "🟢" if is_buy else "🔴" if side == "SELL" else "🟡"
rr_display = signal["rr"] if side != "WAIT" else "—"
local_generated = generated.astimezone(ZoneInfo("Asia/Kabul")).strftime("%I:%M %p")

# ----------------------- main signal card -------------------

st.markdown(f"""
<div style="border:3px solid {accent};border-radius:22px;overflow:hidden;
background:#0b0713;box-shadow:0 0 18px {accent}55;margin:8px 0 18px 0">
  <div style="background:{bg};padding:24px 16px;text-align:center;border-bottom:1px solid #30243b">
    <div style="font-family:Georgia,serif;font-size:52px;font-weight:900;letter-spacing:5px;color:{accent}">{side}</div>
    <div style="color:#a995c6;letter-spacing:3px;font-family:monospace;margin-top:3px">
      XAUUSD&nbsp; · &nbsp;{signal['session']}&nbsp; · &nbsp;R:R {rr_display}
    </div>
    <div style="display:inline-block;margin-top:16px;padding:10px 20px;border:1px solid {accent}88;
      border-radius:30px;background:#07150e;color:{accent};font-family:monospace;font-weight:bold">
      {symbol} {signal['grade']} — SCORE: {signal['score']}/100 — {signal['risk_label']}
    </div>
    <div style="margin-top:14px;color:#bba7d7;font-family:monospace;letter-spacing:1px">
      📊 TRENDING &nbsp; ⏱ {signal['reason']}
    </div>
    <div style="margin-top:18px;padding:17px 10px;border:1px solid #806b27;border-radius:15px;
      background:#211b0b;color:#e2b84e;font-weight:bold;letter-spacing:2px;font-family:monospace">
      ⏳ AI SAYS: {'WAIT THIS TRADE' if side == 'WAIT' else 'WAIT FOR ENTRY ZONE' if remaining < valid_minutes*60-60 else 'SETUP CONFIRMED — MANAGE WITH PLAN'}
    </div>
  </div>
  <div style="display:flex;justify-content:space-between;gap:10px;align-items:center;padding:18px 22px;background:#100b1b;border-bottom:1px solid #30243b">
    <div><div style="font-size:28px;color:{accent};font-family:monospace">{remaining//60:02d}:{remaining%60:02d}</div>
      <div class="small-muted">VALID FOR</div></div>
    <div style="text-align:right"><div class="small-muted">GENERATED</div>
      <div style="font-size:24px;color:#c6a9ec;font-family:monospace">{local_generated}</div>
      <div class="small-muted">EXPIRES IN {valid_minutes} MIN</div></div>
  </div>
  <div style="padding:18px;text-align:center;background:#100b1b;border-bottom:1px solid #30243b">
    <div class="small-muted">ENTRY ZONE</div>
    <div style="font-size:23px;color:#c9b1ee;font-family:monospace;margin-top:8px">{fmt(signal['entry_low'])} — {fmt(signal['entry_high'])}</div>
  </div>
  <div style="padding:18px;text-align:center;background:#120914;border-bottom:1px solid #30243b">
    <div class="small-muted">STOP LOSS</div>
    <div style="font-size:24px;color:#ff6677;font-family:monospace;margin-top:8px">{fmt(signal['sl'])}</div>
  </div>
  <div style="padding:18px 14px;background:#0b0713">
    <div class="small-muted" style="text-align:center;margin-bottom:15px">🎯 3-TARGET PROFIT SYSTEM</div>
""", unsafe_allow_html=True)

tp_alloc = [50, 30, 20]
for i, (tp, alloc, rr) in enumerate(zip(signal["tps"], tp_alloc, [1.42, 2.24, 3.00]), start=1):
    if side == "WAIT":
        hit_label, tp_value = "HIT-RATE: NOT YET MEASURED", "—"
    else:
        tp_value = fmt(tp)
        hit_label = "HISTORICAL HIT-RATE: NOT ENOUGH DATA"
    st.markdown(f"""
    <div style="background:#10101a;border:1px solid #26313b;border-left:4px solid #26d987;
      border-radius:14px;padding:13px 15px;margin:10px 0">
      <div style="display:flex;justify-content:space-between;gap:8px;align-items:center;flex-wrap:wrap">
        <span style="color:#67e6a5;font-family:monospace;font-weight:bold">TP{i} &nbsp; <span style="border:1px solid #299c68;border-radius:20px;padding:3px 8px">{alloc}%</span></span>
        <span style="color:#ad9acb;font-family:monospace">R:R 1:{rr:.2f}</span>
        <span style="color:#65e6a4;font-family:monospace;font-size:19px">{tp_value}</span>
      </div>
      <div style="height:5px;background:#24172f;border-radius:8px;margin-top:13px"></div>
      <div style="text-align:right;color:#8e83a4;font-size:10px;margin-top:5px">{hit_label}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
  </div>
  <div style="padding:17px;background:#100b1b;border-top:1px solid #30243b">
    <div class="small-muted">INVALIDATION</div>
    <div style="font-size:20px;color:#ff7787;font-family:monospace;margin-top:8px">{fmt(signal['invalidation'])}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Outcome buttons log only user-confirmed outcomes. No invented hit-rate.
st.markdown("### TRADE RESULT")
c1, c2 = st.columns(2)
with c1:
    if st.button("✓  MARK WIN", type="primary", disabled=side == "WAIT"):
        st.session_state["trade_log"] = st.session_state.get("trade_log", []) + [{
            "time": datetime.now(ZoneInfo("Asia/Kabul")).strftime("%Y-%m-%d %H:%M"),
            "side": side, "result": "WIN", "score": signal["score"]
        }]
        st.session_state["signal_outcome"] = "WIN"
with c2:
    if st.button("✕  MARK LOSS", type="secondary", disabled=side == "WAIT"):
        st.session_state["trade_log"] = st.session_state.get("trade_log", []) + [{
            "time": datetime.now(ZoneInfo("Asia/Kabul")).strftime("%Y-%m-%d %H:%M"),
            "side": side, "result": "LOSS", "score": signal["score"]
        }]
        st.session_state["signal_outcome"] = "LOSS"

if st.session_state.get("signal_outcome"):
    st.info(f"Current signal marked as {st.session_state['signal_outcome']}.")

# ----------------------- analysis / audit --------------------

st.markdown("### MARKET FILTERS")
m1, m2, m3, m4 = st.columns(4)
m1.metric("XAU/USD", fmt(signal["price"]))
m2.metric("15M ADX", f"{signal['adx']:.1f}")
m3.metric("15M RSI", f"{signal['rsi']:.1f}")
m4.metric("15M ATR", fmt(signal["atr"]))
t1, t2 = st.columns(2)
t1.metric("1H TREND", signal["trend_1h"])
t2.metric("4H TREND", signal["trend_4h"])

with st.expander("Show rule-by-rule signal checklist"):
    if signal["checks"]:
        for name, passed in signal["checks"].items():
            st.write(("✅ " if passed else "❌ ") + name)
    else:
        st.write("No trade: at least one required trend-following condition is not satisfied.")
    st.caption("The selected direction is scored from the trend filters. A signal is allowed only when the score is strictly above the selected minimum; 1H/4H trend alignment and the optional session filter remain important context.")

with st.expander("Recent XAU/USD candles"):
    st.line_chart(m15.tail(120)["close"], height=220)

log = st.session_state.get("trade_log", [])
st.markdown("### REAL USER-RECORDED RESULTS")
if log:
    wins = sum(x["result"] == "WIN" for x in log)
    losses = sum(x["result"] == "LOSS" for x in log)
    total = wins + losses
    hit_rate = wins / total * 100 if total else 0
    q1, q2, q3 = st.columns(3)
    q1.metric("RECORDED TRADES", total)
    q2.metric("WINS / LOSSES", f"{wins} / {losses}")
    q3.metric("RECORDED WIN RATE", f"{hit_rate:.1f}%")
    st.dataframe(pd.DataFrame(log), use_container_width=True, hide_index=True)
else:
    st.caption("No trade outcomes have been recorded yet. Hit-rate will appear only after you mark actual results.")

st.caption("Educational decision-support tool. Backtest and forward-test before risking capital. Market data availability and API limits depend on your Twelve Data account.")
