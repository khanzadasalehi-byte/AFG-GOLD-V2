import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="AFG GOLD + BTC PRO SCANNER", page_icon="🟡", layout="wide")

st.markdown("""
<style>
.stApp {background:#090611;color:#e8def8}
[data-testid="stSidebar"]{background:#17121f}
.block-container{max-width:1200px;padding-top:1rem}
div[data-testid="stMetric"]{background:#130d1d;border:1px solid #392b49;padding:12px;border-radius:12px}
.buy{color:#27df89}.sell{color:#ff6377}.wait{color:#e7bd55}
</style>
""", unsafe_allow_html=True)

# API key is read from Streamlit Cloud > App > Settings > Secrets.
# Required secret: TWELVE_DATA_API_KEY = "your_key"
try:
    API_KEY = st.secrets.get("TWELVE_DATA_API_KEY", "")
except Exception:
    API_KEY = ""

if not API_KEY:
    st.error("API Key در Streamlit Secrets پیدا نشد. در Settings > Secrets این مقدار را ذخیره کن: TWELVE_DATA_API_KEY = \"YOUR_KEY\"")
    st.stop()


# Telegram notifications are optional until both Streamlit secrets are configured.
try:
    TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", "")
except Exception:
    TELEGRAM_BOT_TOKEN = ""
    TELEGRAM_CHAT_ID = ""

def send_telegram_signal(result):
    """Send one signal message; return (success, explanation)."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False, "تنظیمات تلگرام در Secrets وارد نشده است."
    side = result["side"]
    if side not in ("BUY", "SELL") or result["score"] <= 40:
        return False, "سیگنال معتبر برای ارسال وجود ندارد."
    market = (
        f"📍 {side} MARKET\nEntry: {price(result['price'])}\n"
        f"SL: {price(result['sl'])}\n"
        f"TP1: {price(result['tps'][0])}\n"
        f"TP2: {price(result['tps'][1])}\n"
        f"TP3: {price(result['tps'][2])}"
    )
    limit_side = "BUY LIMIT" if side == "BUY" else "SELL LIMIT"
    limit = (
        f"\n\n🎯 {limit_side}\nEntry: {price(result['limit_entry'])}\n"
        f"SL: {price(result['limit_sl'])}\n"
        f"TP1: {price(result['limit_tps'][0])}\n"
        f"TP2: {price(result['limit_tps'][1])}\n"
        f"TP3: {price(result['limit_tps'][2])}"
    )
    stamp = result["candle_time"].strftime("%Y-%m-%d %H:%M UTC")
    message = (
        f"🚨 AFG GOLD + BTC PRO SCANNER\n"
        f"Market: {result['symbol']}\nSignal: {side} | Grade {result['grade']} | Score {result['score']}/100\n"
        f"Price: {price(result['price'])}\n"
        f"RSI: {result['rsi']:.1f} | ADX: {result['adx']:.1f}\n"
        f"1H: {result['trend1h']} | 4H: {result['trend4h']}\n"
        f"Candle: {stamp}\n\n{market}{limit}\n\n"
        "یادآوری: این سیگنال توصیه مالی نیست؛ Limit فقط قیمت پیشنهادی است و سفارش خودکار اجرا نمی‌شود."
    )
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    response = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=15)
    payload = response.json()
    if response.ok and payload.get("ok"):
        return True, "اعلان تلگرام فرستاده شد."
    return False, payload.get("description", f"Telegram HTTP {response.status_code}")

AUTO_REFRESH_MS = 5 * 60 * 1000
st_autorefresh(interval=AUTO_REFRESH_MS, key="market_scanner_refresh")

@st.cache_data(ttl=240, show_spinner=False)
def get_candles(symbol, interval, outputsize=500):
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": symbol, "interval": interval, "outputsize": outputsize,
              "apikey": API_KEY, "format": "JSON", "timezone": "UTC"}
    response = requests.get(url, params=params, timeout=25)
    response.raise_for_status()
    payload = response.json()
    if "values" not in payload:
        raise ValueError(payload.get("message", payload.get("code", "No candle data returned")))
    df = pd.DataFrame(payload["values"])
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    return df.dropna(subset=["datetime", "open", "high", "low", "close"]).sort_values("datetime").set_index("datetime")

def indicators(df):
    x = df.copy()
    for n in (20, 50, 200):
        x[f"ema{n}"] = x.close.ewm(span=n, adjust=False).mean()
    delta = x.close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi"] = 100 - 100/(1+rs)
    prev = x.close.shift()
    tr = pd.concat([(x.high-x.low), (x.high-prev).abs(), (x.low-prev).abs()], axis=1).max(axis=1)
    x["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    up, down = x.high.diff(), -x.low.diff()
    pdm = pd.Series(np.where((up > down) & (up > 0), up, 0), index=x.index)
    mdm = pd.Series(np.where((down > up) & (down > 0), down, 0), index=x.index)
    plus = 100*pdm.ewm(alpha=1/14, adjust=False).mean()/x.atr.replace(0,np.nan)
    minus = 100*mdm.ewm(alpha=1/14, adjust=False).mean()/x.atr.replace(0,np.nan)
    dx = 100*(plus-minus).abs()/(plus+minus).replace(0,np.nan)
    x["adx"] = dx.ewm(alpha=1/14, adjust=False).mean()
    return x.dropna()

def analyze(symbol):
    m15 = indicators(get_candles(symbol, "15min", 500))
    h1 = indicators(get_candles(symbol, "1h", 300))
    h4 = indicators(get_candles(symbol, "4h", 300))
    if min(len(m15), len(h1), len(h4)) < 5:
        raise ValueError("تعداد کندل‌های دریافتی کافی نیست.")
    x, prev, one, four = m15.iloc[-1], m15.iloc[-2], h1.iloc[-1], h4.iloc[-1]
    buy_rules = [
        one.ema20 > one.ema50 and one.close > one.ema200,
        four.ema20 > four.ema50,
        x.ema20 > x.ema50,
        x.adx >= 16,
        50 <= x.rsi <= 72,
        x.close > x.open and x.close > prev.high,
        x.close > x.ema20,
    ]
    sell_rules = [
        one.ema20 < one.ema50 and one.close < one.ema200,
        four.ema20 < four.ema50,
        x.ema20 < x.ema50,
        x.adx >= 16,
        28 <= x.rsi <= 50,
        x.close < x.open and x.close < prev.low,
        x.close < x.ema20,
    ]
    bs, ss = round(sum(buy_rules)/7*100), round(sum(sell_rules)/7*100)
    side = "BUY" if bs > ss else "SELL" if ss > bs else "WAIT"
    score = max(bs, ss) if side != "WAIT" else 0
    now = datetime.now(timezone.utc)
    london = now.astimezone(ZoneInfo("Europe/London"))
    ny = now.astimezone(ZoneInfo("America/New_York"))
    in_session = (8 <= london.hour < 17) or (8 <= ny.hour < 17)
    if not in_session:
        # BTC trades 24/7; Gold session filter is intentionally not enforced here.
        pass
    if score <= 40:
        side = "WAIT"
    grade = "A+" if score >= 95 else "A" if score >= 85 else "B" if score >= 70 else "C" if score > 40 else "—"
    price, vol = float(x.close), float(x.atr)
    if side == "BUY":
        swing = float(m15.low.tail(8).min())
        risk = max(price - min(swing, price-1.35*vol), .65*vol)
        sl = price-risk
        tps = [price+risk*r for r in (1.42,2.24,3.0)]
        # Pullback limit level near the 15m EMA20, constrained below current price.
        limit_entry = min(float(x.ema20), price-0.15*vol)
        limit_entry = max(limit_entry, price-0.85*vol)
        limit_sl = limit_entry-risk
        limit_tps = [limit_entry+risk*r for r in (1.42,2.24,3.0)]
    elif side == "SELL":
        swing = float(m15.high.tail(8).max())
        risk = max(max(swing,price+1.35*vol)-price,.65*vol)
        sl = price+risk
        tps = [price-risk*r for r in (1.42,2.24,3.0)]
        limit_entry = max(float(x.ema20), price+0.15*vol)
        limit_entry = min(limit_entry, price+0.85*vol)
        limit_sl = limit_entry+risk
        limit_tps = [limit_entry-risk*r for r in (1.42,2.24,3.0)]
    else:
        sl, tps = np.nan, [np.nan]*3
        limit_entry = limit_sl = np.nan
        limit_tps = [np.nan]*3
    return {"symbol":symbol,"side":side,"score":score,"grade":grade,"price":price,
            "sl":sl,"tps":tps,"limit_entry":limit_entry,"limit_sl":limit_sl,
            "limit_tps":limit_tps,"rsi":float(x.rsi),"adx":float(x.adx),
            "atr":vol,"trend1h":"BULLISH" if one.ema20>one.ema50 else "BEARISH",
            "trend4h":"BULLISH" if four.ema20>four.ema50 else "BEARISH",
            "candle_time":m15.index[-1],"buy_score":bs,"sell_score":ss}

def price(v):
    return "—" if pd.isna(v) else f"{v:,.2f}"

def render_asset(result, label):
    st.subheader(label)
    side = result["side"]
    css = "buy" if side == "BUY" else "sell" if side == "SELL" else "wait"
    st.markdown(f"## <span class='{css}'>{side} · {result['grade']} · {result['score']}/100</span>", unsafe_allow_html=True)
    a,b,c,d = st.columns(4)
    a.metric("Current price", price(result["price"]))
    b.metric("RSI 15M", f"{result['rsi']:.1f}")
    c.metric("ADX 15M", f"{result['adx']:.1f}")
    d.metric("ATR 15M", price(result["atr"]))
    st.caption(f"1H: {result['trend1h']}  |  4H: {result['trend4h']}  |  آخرین کندل: {result['candle_time'].strftime('%Y-%m-%d %H:%M UTC')}")
    market, limit = st.columns(2)
    with market:
        st.markdown("### MARKET ENTRY")
        if side == "WAIT":
            st.warning("فعلاً سیگنال معتبر وجود ندارد.")
        else:
            st.success(f"{side} MARKET")
            st.write(f"Entry: **{price(result['price'])}**")
            st.write(f"Stop Loss: **{price(result['sl'])}**")
            for i,tp in enumerate(result["tps"],1):
                st.write(f"TP{i}: **{price(tp)}**")
    with limit:
        st.markdown("### LIMIT ENTRY")
        if side == "WAIT":
            st.warning("برای Limit نیز ابتدا جهت معتبر لازم است.")
        else:
            limit_type = "BUY LIMIT" if side == "BUY" else "SELL LIMIT"
            st.info(limit_type)
            st.write(f"Limit Entry: **{price(result['limit_entry'])}**")
            st.write(f"Stop Loss: **{price(result['limit_sl'])}**")
            for i,tp in enumerate(result["limit_tps"],1):
                st.write(f"TP{i}: **{price(tp)}**")
            st.caption("Limit قیمت پیشنهادیِ پولبک به EMA20 است؛ تاچ‌شدن یا اجرای سفارش تضمین نیست.")
    with st.expander("جزئیات امتیازدهی"):
        st.write(f"امتیاز BUY: {result['buy_score']}/100")
        st.write(f"امتیاز SELL: {result['sell_score']}/100")
        st.caption("درجه‌ها بر اساس امتیاز فیلترها هستند، نه درصد احتمال برد.")

st.title("🟡 AFG GOLD + BTC PRO SCANNER")
st.caption("Rule-Based Trend Following · اسکن خودکار هر ۵ دقیقه · داده Twelve Data")
last_scan = datetime.now(ZoneInfo("Asia/Kabul")).strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"آخرین اجرای اسکن صفحه: {last_scan} (وقت افغانستان)")

tab_gold, tab_btc, tab_scan = st.tabs(["🟡 GOLD XAU/USD","₿ BITCOIN BTC/USD","🔎 MARKET SCANNER"])
with tab_gold:
    try:
        with st.spinner("در حال بررسی طلا..."):
            gold = analyze("XAU/USD")
        render_asset(gold, "XAU/USD — GOLD")
    except Exception as e:
        st.error(f"خطا در دریافت/تحلیل طلا: {e}")
with tab_btc:
    try:
        with st.spinner("در حال بررسی بیت‌کوین..."):
            btc = analyze("BTC/USD")
        render_asset(btc, "BTC/USD — BITCOIN")
    except Exception as e:
        st.error(f"خطا در دریافت/تحلیل بیت‌کوین: {e}")
with tab_scan:
    st.subheader("اسکنر هم‌زمان بازار")
    scan_results=[]
    for sym in ("XAU/USD","BTC/USD"):
        try:
            scan_results.append(analyze(sym))
        except Exception as e:
            st.warning(f"{sym}: {e}")
    # Notify once per symbol, direction and 15m candle. Streamlit reruns won't spam duplicates.
    if "telegram_sent_signal_keys" not in st.session_state:
        st.session_state.telegram_sent_signal_keys = set()
    sent_count = 0
    telegram_errors = []
    for result in scan_results:
        if result["side"] not in ("BUY", "SELL") or result["score"] <= 40:
            continue
        signal_key = (result["symbol"], result["side"], result["candle_time"].isoformat())
        if signal_key in st.session_state.telegram_sent_signal_keys:
            continue
        ok, detail = send_telegram_signal(result)
        if ok:
            st.session_state.telegram_sent_signal_keys.add(signal_key)
            sent_count += 1
        elif TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            telegram_errors.append(f"{result['symbol']}: {detail}")
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        if sent_count:
            st.success(f"{sent_count} اعلان جدید به تلگرام ارسال شد.")
        elif telegram_errors:
            for err in telegram_errors:
                st.warning(f"خطای ارسال تلگرام — {err}")
        else:
            st.caption("تلگرام فعال است؛ در این اسکن سیگنال تازه‌ای برای ارسال نبود.")
    else:
        st.warning("اعلان تلگرام هنوز فعال نیست؛ TELEGRAM_BOT_TOKEN و TELEGRAM_CHAT_ID را در Streamlit Secrets اضافه کن.")

    if scan_results:
        st.dataframe(pd.DataFrame([{"Market":r["symbol"],"Signal":r["side"],"Grade":r["grade"],
                                    "Score":r["score"],"Price":r["price"],
                                    "BUY score":r["buy_score"],"SELL score":r["sell_score"]}
                                   for r in scan_results]), use_container_width=True, hide_index=True)
    st.info("اسکن خودکار هر ۵ دقیقه انجام می‌شود؛ برنامه باید باز باشد. برای جلوگیری از مصرف زیاد اعتبار API، داده‌ها تا ۴ دقیقه cache می‌شوند.")

st.divider()
st.caption("هشدار: سیگنال‌ها توصیه مالی نیستند. امتیاز و Grade تضمین‌کننده سود یا Win Rate نیستند. پیش از معامله واقعی، بک‌تست و فورواردتست انجام بده.")
