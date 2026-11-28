import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import uuid
import hashlib
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

st.set_page_config(page_title="KhanGoldSignal | GOLD & BTC", page_icon="🟡", layout="wide", initial_sidebar_state="expanded")

APP_NAME = "KhanGoldSignal"
UTC = timezone.utc
ASSETS = {"GOLD": {"symbol": "XAU/USD", "label": "XAUUSD"}, "BITCOIN": {"symbol": "BTC/USD", "label": "BTCUSD"}}
PROFILES = {
    "Scalping": {"entry_tf": "1min", "trend_tf": "5min", "confirm_tf": "15min", "min_score": 65, "min_adx": 15, "atr_sl": 1.5, "pullback_min": .05, "pullback_max": .75},
    "Intraday": {"entry_tf": "15min", "trend_tf": "1h", "confirm_tf": "4h", "min_score": 65, "min_adx": 16, "atr_sl": 1.7, "pullback_min": .08, "pullback_max": .90},
    "Swing": {"entry_tf": "4h", "trend_tf": "1day", "confirm_tf": "1day", "min_score": 65, "min_adx": 17, "atr_sl": 2.0, "pullback_min": .10, "pullback_max": 1.10},
}
VALIDITY_MINUTES = 45
SIGNAL_TABLE = "signals"
OUTBOX_TABLE = "telegram_outbox"

st.markdown("""<style>.stApp{background:#090611;color:#e3d9f2}.block-container{max-width:1400px;padding-top:1rem}[data-testid=stSidebar]{background:#17131f}div[data-testid=stMetric]{background:#100b1b;border:1px solid #30233d;padding:10px;border-radius:12px}</style>""", unsafe_allow_html=True)

# --------------------------- General helpers ---------------------------
def utc_now(): return datetime.now(UTC)
def iso_utc(v):
    if v is None: return None
    if isinstance(v, str): return v
    return v.astimezone(UTC).isoformat()
def parse_dt(v):
    if not v: return None
    try:
        d = pd.Timestamp(v).to_pydatetime()
        if d.tzinfo is None: d = d.replace(tzinfo=UTC)
        return d.astimezone(UTC)
    except Exception: return None
def num(v, default=None):
    try:
        x=float(v)
        return x if np.isfinite(x) else default
    except Exception: return default
def price(v):
    x=num(v)
    return f"{x:,.2f}" if x is not None else "—"
def secret(name):
    try: return str(st.secrets.get(name, "")).strip()
    except Exception: return ""
def interval_delta(tf):
    return {"1min":timedelta(minutes=1),"5min":timedelta(minutes=5),"15min":timedelta(minutes=15),"30min":timedelta(minutes=30),"1h":timedelta(hours=1),"4h":timedelta(hours=4),"1day":timedelta(days=1)}[tf]
def market_session():
    now=utc_now(); lon=now.astimezone(ZoneInfo("Europe/London")); ny=now.astimezone(ZoneInfo("America/New_York"))
    lo=lon.weekday()<5 and 8<=lon.hour<17; no=ny.weekday()<5 and 8<=ny.hour<17
    return "LONDON / NEW YORK OVERLAP" if lo and no else "LONDON" if lo else "NEW YORK" if no else "ASIA / OFF SESSION" if now.weekday()<5 else "WEEKEND"

# --------------------------- Market data ---------------------------
@st.cache_data(ttl=45, show_spinner=False)
def get_candles(api_key, symbol, interval, outputsize=500):
    r=requests.get("https://api.twelvedata.com/time_series",params={"symbol":symbol,"interval":interval,"outputsize":outputsize,"apikey":api_key,"format":"JSON","timezone":"UTC"},timeout=25)
    r.raise_for_status(); p=r.json()
    if "values" not in p: raise RuntimeError(f"Twelve Data {symbol} {interval}: {p.get('message',p.get('code','No data'))}")
    d=pd.DataFrame(p["values"])
    for c in ("open","high","low","close"): d[c]=pd.to_numeric(d[c],errors="coerce")
    d["datetime"]=pd.to_datetime(d["datetime"],utc=True,errors="coerce")
    d=d.dropna(subset=["datetime","open","high","low","close"]).sort_values("datetime").drop_duplicates("datetime").set_index("datetime")
    d=d[d.index<=utc_now()-interval_delta(interval)]
    if len(d)<220: raise RuntimeError(f"Not enough closed candles for {symbol} {interval}.")
    return d

def add_indicators(d):
    x=d.copy(); c=x.close
    for n in (20,50,200): x[f"ema{n}"]=c.ewm(span=n,adjust=False).mean()
    delta=c.diff(); gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean(); loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    rs=gain/loss.replace(0,np.nan); x["rsi"]=(100-100/(1+rs)).mask((loss==0)&(gain>0),100).mask((gain==0)&(loss>0),0)
    prev=c.shift(1); tr=pd.concat([x.high-x.low,(x.high-prev).abs(),(x.low-prev).abs()],axis=1).max(axis=1)
    x["atr"]=tr.ewm(alpha=1/14,adjust=False).mean()
    up=x.high.diff(); down=-x.low.diff(); plus=pd.Series(np.where((up>down)&(up>0),up,0),index=x.index); minus=pd.Series(np.where((down>up)&(down>0),down,0),index=x.index)
    smooth=tr.ewm(alpha=1/14,adjust=False).mean().replace(0,np.nan); pdi=100*plus.ewm(alpha=1/14,adjust=False).mean()/smooth; mdi=100*minus.ewm(alpha=1/14,adjust=False).mean()/smooth
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan); x["adx"]=dx.ewm(alpha=1/14,adjust=False).mean(); x["plus_di"]=pdi; x["minus_di"]=mdi
    return x.replace([np.inf,-np.inf],np.nan).dropna()
def trend(row):
    if row.ema20>row.ema50 and row.close>row.ema200 and row.plus_di>row.minus_di: return "BULLISH"
    if row.ema20<row.ema50 and row.close<row.ema200 and row.minus_di>row.plus_di: return "BEARISH"
    return "SIDEWAYS"

# --------------------------- Strategy ---------------------------
def evaluate(entry_raw, trend_raw, confirm_raw, profile):
    cfg=PROFILES[profile]; e=add_indicators(entry_raw); t=add_indicators(trend_raw); h=add_indicators(confirm_raw)
    if min(len(e),len(t),len(h))<5: raise RuntimeError("Insufficient indicator history")
    cur=e.iloc[-1]; prev=e.iloc[-2]; main=trend(t.iloc[-1]); higher=trend(h.iloc[-1]); ent=trend(cur); px=float(cur.close); vol=float(cur.atr)
    long=[ent=="BULLISH",main=="BULLISH",higher=="BULLISH",cur.adx>=cfg["min_adx"],cur.plus_di>cur.minus_di,50<=cur.rsi<=70,px>cur.ema20,cur.ema20>cur.ema50,cur.close>cur.open,cur.close>prev.high]
    short=[ent=="BEARISH",main=="BEARISH",higher=="BEARISH",cur.adx>=cfg["min_adx"],cur.minus_di>cur.plus_di,30<=cur.rsi<=50,px<cur.ema20,cur.ema20<cur.ema50,cur.close<cur.open,cur.close<prev.low]
    weights=[15,20,15,10,10,8,7,5,5,5]; ls=sum(w for w,v in zip(weights,long) if v); ss=sum(w for w,v in zip(weights,short) if v)
    side,score=("BUY",ls) if ls>ss else ("SELL",ss) if ss>ls else ("WAIT",max(ls,ss))
    dist=abs(px-float(cur.ema20))/vol if vol>0 else 999
    if main!= ("BULLISH" if side=="BUY" else "BEARISH") or higher!= ("BULLISH" if side=="BUY" else "BEARISH"): side="WAIT"
    if score<cfg["min_score"] or not cfg["pullback_min"]<=dist<=cfg["pullback_max"] or float(cur.high-cur.low)>2.8*vol: side="WAIT"
    if side=="BUY": entry=float(cur.ema20); swing=float(e.low.tail(10).min()); sl=min(swing,entry-cfg["atr_sl"]*vol); risk=entry-sl
    elif side=="SELL": entry=float(cur.ema20); swing=float(e.high.tail(10).max()); sl=max(swing,entry+cfg["atr_sl"]*vol); risk=sl-entry
    else: entry=sl=risk=np.nan
    if side in ("BUY","SELL") and risk>0 and np.isfinite(risk):
        sign=1 if side=="BUY" else -1; tps=[entry+sign*risk*k for k in (1,2,3)]; grade="A" if score>=85 else "B" if score>=70 else "C"; risk_label="LOWER SETUP RISK" if score>=85 else "MODERATE SETUP" if score>=70 else "WEAKER SETUP"; status="PENDING_LIMIT"
    else:
        side="WAIT"; entry=sl=risk=np.nan; tps=[np.nan]*3; grade="—"; risk_label="NO TRADE"; status="WAIT"
    return {"asset":"","profile":profile,"direction":side,"entry_type":"LIMIT" if side!="WAIT" else "NONE","entry_price":num(entry),"zone_low":num(entry),"zone_high":num(entry),"stop_loss":num(sl),"tp1":num(tps[0]),"tp2":num(tps[1]),"tp3":num(tps[2]),"score":int(score),"grade":grade,"risk_level":risk_label,"session":market_session(),"candle_time":iso_utc(e.index[-1].to_pydatetime()),"current_price":px,"atr":vol,"status":status}

def build_candidate(api_key, asset, profile):
    a=ASSETS[asset]; c=PROFILES[profile]
    e=get_candles(api_key,a["symbol"],c["entry_tf"],500); t=get_candles(api_key,a["symbol"],c["trend_tf"],400); h=get_candles(api_key,a["symbol"],c["confirm_tf"],400)
    x=evaluate(e,t,h,profile); x["asset"]=asset; x["current_price"]=float(e.close.iloc[-1]); x["atr"]=float(add_indicators(e).atr.iloc[-1]); x["setup_key"]=hashlib.sha256(f"{asset}|{profile}|{x['direction']}|{x['candle_time']}".encode()).hexdigest()[:32]
    return x

# --------------------------- Supabase: exact existing schema ---------------------------
def supabase_config():
    url=secret("SUPABASE_URL"); key=secret("SUPABASE_KEY") or secret("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key: return None,None
    return url.rstrip("/"),key
def db_request(method,table,params=None,payload=None,prefer=None):
    url,key=supabase_config()
    if not url: raise RuntimeError("Supabase URL/KEY missing in Streamlit secrets")
    headers={"apikey":key,"Authorization":f"Bearer {key}","Content-Type":"application/json","Accept":"application/json"}
    if prefer: headers["Prefer"]=prefer
    r=requests.request(method,f"{url}/rest/v1/{table}",headers=headers,params=params,json=payload,timeout=20)
    if r.status_code>=400: raise RuntimeError(f"Supabase {r.status_code}: {r.text[:400]}")
    return r.json() if r.text.strip() else []
def get_signal_by_key(key):
    rows=db_request("GET",SIGNAL_TABLE,{"select":"*","signal_key":f"eq.{key}","limit":"1"})
    return rows[0] if rows else None
def get_route_latest(asset,profile):
    rows=db_request("GET",SIGNAL_TABLE,{"select":"*","asset":f"eq.{asset}","profile":f"eq.{profile}","order":"generated_at.desc","limit":"1"})
    return rows[0] if rows else None
def update_signal(signal_key,updates):
    return db_request("PATCH",SIGNAL_TABLE,{"signal_key":f"eq.{signal_key}"},{**updates},"return=representation")
def make_signal_row(c):
    now=utc_now(); key=c["setup_key"]
    # Only columns confirmed to exist in the user's current signals table are inserted.
    return {"signal_key":key,"asset":c["asset"],"profile":c["profile"],"direction":c["direction"],"entry_type":c["entry_type"],"entry_price":c["entry_price"],"zone_low":c["zone_low"],"zone_high":c["zone_high"],"stop_loss":c["stop_loss"],"tp1":c["tp1"],"tp2":c["tp2"],"tp3":c["tp3"],"score":c["score"],"grade":c["grade"],"risk_level":c["risk_level"],"session":c["session"],"candle_time":c["candle_time"],"generated_at":iso_utc(now),"expires_at":iso_utc(now+timedelta(minutes=VALIDITY_MINUTES)),"status":"PENDING_LIMIT","extension_count":0}
def is_active(s): return s and s.get("status") in ("PENDING_LIMIT","ACTIVE_TRADE","TP1_HIT","TP2_HIT")
def review_pending(existing,c):
    if existing.get("status")!="PENDING_LIMIT": return existing
    exp=parse_dt(existing.get("expires_at")); now=utc_now()
    if exp and now<exp: return existing
    old=num(existing.get("entry_price")); cur=num(c.get("current_price")); atrv=num(c.get("atr"),0)
    same=c.get("direction")==existing.get("direction") and c.get("direction") in ("BUY","SELL")
    tol=max((atrv or 0)*.5,(old or 0)*.0005)
    actionable=old is not None and cur is not None and ((cur>old and cur-old<=tol) if existing.get("direction")=="BUY" else (cur<old and old-cur<=tol))
    if same and actionable:
        update_signal(existing["signal_key"],{"expires_at":iso_utc(now+timedelta(minutes=VALIDITY_MINUTES)),"extension_count":int(existing.get("extension_count") or 0)+1,"last_review_at":iso_utc(now),"last_review_reason":"Setup still valid; original prices preserved"})
    else:
        update_signal(existing["signal_key"],{"status":"INVALIDATED","last_review_at":iso_utc(now),"last_review_reason":"Setup invalid or original entry no longer actionable"})
    return get_signal_by_key(existing["signal_key"])
def get_or_create_signal(c):
    existing=get_route_latest(c["asset"],c["profile"])
    if is_active(existing):
        return review_pending(existing,c) if existing.get("status")=="PENDING_LIMIT" else existing
    if c.get("direction") not in ("BUY","SELL"): return existing
    prior=get_signal_by_key(c["setup_key"])
    if prior: return prior
    rows=db_request("POST",SIGNAL_TABLE,payload=make_signal_row(c),prefer="return=representation")
    if not rows: raise RuntimeError("Supabase did not return inserted signal")
    return rows[0]
def get_history(limit=200): return db_request("GET",SIGNAL_TABLE,{"select":"*","order":"generated_at.desc","limit":str(limit)})
def get_active(): return db_request("GET",SIGNAL_TABLE,{"select":"*","status":"in.(PENDING_LIMIT,ACTIVE_TRADE,TP1_HIT,TP2_HIT)","order":"generated_at.asc","limit":"300"})
def snapshot_from_row(row):
    # Snapshot fields are read from persisted DB columns; no fresh strategy values overwrite them.
    return {"asset":row.get("asset"),"profile":row.get("profile"),"direction":row.get("direction"),"entry_type":row.get("entry_type"),"entry_price":row.get("entry_price"),"zone_low":row.get("zone_low"),"zone_high":row.get("zone_high"),"stop_loss":row.get("stop_loss"),"tp1":row.get("tp1"),"tp2":row.get("tp2"),"tp3":row.get("tp3"),"score":row.get("score"),"grade":row.get("grade"),"risk_level":row.get("risk_level"),"session":row.get("session"),"candle_time":row.get("candle_time")}

# --------------------------- Telegram outbox: exact existing schema ---------------------------
def telegram_config():
    token=secret("TELEGRAM_BOT_TOKEN"); chat=secret("TELEGRAM_CHAT_ID")
    return (token,chat) if token and chat else (None,None)
def telegram_text(row):
    s=snapshot_from_row(row); sym=ASSETS.get(s["asset"],{}).get("symbol",s["asset"])
    return (f"🟡 KHAN GOLD SIGNAL\nAsset: {sym}\nProfile: {s['profile']}\nDirection: {s['direction']}\nScore: {s['score']}/100 | Grade: {s['grade']}\nEntry: {price(s['entry_price'])}\nSL: {price(s['stop_loss'])}\nTP1: {price(s['tp1'])}\nTP2: {price(s['tp2'])}\nTP3: {price(s['tp3'])}\nSession: {s['session']}\nInitial validity: {VALIDITY_MINUTES} minutes\nAnalytical signal only; not an order execution service.")
def send_telegram(row):
    key=row.get("signal_key")
    existing=db_request("GET",OUTBOX_TABLE,{"select":"*","signal_key":f"eq.{key}","limit":"1"})
    if existing:
        item=existing[0]
        if item.get("status") in ("SENT","UNKNOWN","SENDING"): return item.get("status")
        if item.get("status")!="PENDING": return item.get("status")
    else:
        msg=telegram_text(row)
        created=db_request("POST",OUTBOX_TABLE,payload={"signal_key":key,"status":"PENDING","message_payload":json.dumps({"text":msg},ensure_ascii=False),"created_at":iso_utc(utc_now()),"updated_at":iso_utc(utc_now())},prefer="return=representation")
        if not created: return "OUTBOX_ERROR"
        item=created[0]
    item_id=item.get("id"); db_request("PATCH",OUTBOX_TABLE,{"id":f"eq.{item_id}"},{"status":"SENDING","updated_at":iso_utc(utc_now())},"return=minimal")
    token,chat=telegram_config()
    if not token: 
        db_request("PATCH",OUTBOX_TABLE,{"id":f"eq.{item_id}"},{"status":"FAILED","error_message":"Telegram credentials missing","updated_at":iso_utc(utc_now())},"return=minimal"); return "FAILED"
    payload=item.get("message_payload") or "{}"
    if isinstance(payload,str):
        try: payload=json.loads(payload)
        except Exception: payload={"text":payload}
    try:
        r=requests.post(f"https://api.telegram.org/bot{token}/sendMessage",json={"chat_id":chat,"text":payload.get("text","")},timeout=20); r.raise_for_status(); data=r.json()
        if not data.get("ok"): raise RuntimeError("Telegram did not confirm delivery")
        db_request("PATCH",OUTBOX_TABLE,{"id":f"eq.{item_id}"},{"status":"SENT","telegram_message_id":str(data["result"]["message_id"]),"sent_at":iso_utc(utc_now()),"updated_at":iso_utc(utc_now()),"error_message":None},"return=minimal"); return "SENT"
    except requests.Timeout as e:
        db_request("PATCH",OUTBOX_TABLE,{"id":f"eq.{item_id}"},{"status":"UNKNOWN","error_message":str(e)[:400],"updated_at":iso_utc(utc_now())},"return=minimal"); return "UNKNOWN"
    except Exception as e:
        db_request("PATCH",OUTBOX_TABLE,{"id":f"eq.{item_id}"},{"status":"FAILED","error_message":str(e)[:400],"updated_at":iso_utc(utc_now())},"return=minimal"); return "FAILED"

# --------------------------- Estimated candle trade lifecycle ---------------------------
def manage_trade(row):
    if row.get("status") not in ("PENDING_LIMIT","ACTIVE_TRADE","TP1_HIT","TP2_HIT"): return row
    asset=row.get("asset"); profile=row.get("profile"); key=row.get("signal_key")
    try: df=get_candles(secret("TWELVE_DATA_API_KEY"),ASSETS[asset]["symbol"],PROFILES[profile]["entry_tf"],500)
    except Exception: return row
    if df.empty: return row
    candle=df.iloc[-1]; candle_time=iso_utc(df.index[-1].to_pydatetime()); generated=parse_dt(row.get("generated_at")); ct=parse_dt(candle_time)
    if generated and ct and ct<=generated: return row
    lo=float(candle.low); hi=float(candle.high); direction=row.get("direction"); entry=num(row.get("entry_price")); sl=num(row.get("stop_loss"))
    updates=None
    if row.get("status")=="PENDING_LIMIT" and entry is not None and lo<=entry<=hi:
        updates={"status":"ACTIVE_TRADE","trade_opened_at":candle_time,"last_review_at":iso_utc(utc_now()),"last_review_reason":"Estimated limit fill from closed candle"}
    elif row.get("status") in ("ACTIVE_TRADE","TP1_HIT","TP2_HIT"):
        sl_hit=sl is not None and (lo<=sl if direction=="BUY" else hi>=sl)
        targets=[num(row.get(f"tp{i}")) for i in (1,2,3)]
        hit=[t is not None and (hi>=t if direction=="BUY" else lo<=t) for t in targets]
        if sl_hit: updates={"status":"SL_HIT","closed_at":candle_time,"last_review_at":iso_utc(utc_now()),"last_review_reason":"Conservative rule: stop checked before targets"}
        elif row.get("status")=="ACTIVE_TRADE" and hit[0]: updates={"status":"TP1_HIT","last_review_at":iso_utc(utc_now()),"last_review_reason":"TP1 touched; estimated partial allocation 50%"}
        elif row.get("status")=="TP1_HIT" and hit[1]: updates={"status":"TP2_HIT","last_review_at":iso_utc(utc_now()),"last_review_reason":"TP2 touched; cumulative estimated allocation 80%"}
        elif row.get("status")=="TP2_HIT" and hit[2]: updates={"status":"TP3_HIT","closed_at":candle_time,"last_review_at":iso_utc(utc_now()),"last_review_reason":"TP3 touched; estimated trade completion"}
    if updates:
        update_signal(key,updates); return get_signal_by_key(key)
    return row
def manage_all():
    try: rows=get_active()
    except Exception: return
    for row in rows:
        try: manage_trade(row)
        except Exception: pass

# --------------------------- Dashboard ---------------------------
def show_signal(row):
    s=snapshot_from_row(row); status=row.get("status","UNKNOWN"); sym=ASSETS.get(s["asset"],{}).get("symbol",s["asset"])
    with st.container(border=True):
        a,b,c=st.columns([2,1,1]); a.subheader(f"{sym} — {s['profile']}"); b.metric("Direction",s["direction"]); c.metric("Score",f"{s['score']}/100")
        st.caption(f"Signal key: {row.get('signal_key','—')} | Grade: {s['grade']} | Status: {status}")
        x,y,z=st.columns(3); x.metric("Entry",price(s["entry_price"])); y.metric("Stop Loss",price(s["stop_loss"])); z.metric("Session",s["session"])
        x,y,z=st.columns(3); x.metric("TP1",price(s["tp1"])); y.metric("TP2",price(s["tp2"])); z.metric("TP3",price(s["tp3"]))
        if status=="PENDING_LIMIT":
            exp=parse_dt(row.get("expires_at")); sec=max(0,int((exp-utc_now()).total_seconds())) if exp else 0; st.info(f"Validity: {sec//60:02d}:{sec%60:02d} | Expiry is reviewed; published prices stay unchanged.")
        elif status in ("ACTIVE_TRADE","TP1_HIT","TP2_HIT"): st.success(f"Estimated trade lifecycle: {status}")
        elif status in ("INVALIDATED","SL_HIT","TP3_HIT","CANCELLED"): st.warning(f"Signal status: {status}")
        st.caption(f"Generated: {row.get('generated_at','—')} | Candle: {s['candle_time']}")
def route(asset,profile):
    try:
        api=secret("TWELVE_DATA_API_KEY")
        if not api: st.warning("TWELVE_DATA_API_KEY is missing in Streamlit secrets."); return
        c=build_candidate(api,asset,profile); row=get_or_create_signal(c)
        if not row: st.info("No valid setup at this time."); return
        if row.get("status")=="PENDING_LIMIT":
            try: send_telegram(row)
            except Exception: pass
        show_signal(row)
    except Exception as e: st.warning(f"{asset} / {profile}: {e}")
def dashboard():
    st.title("🟡 KhanGoldSignal")
    st.caption("Rule-Based Multi-Timeframe Trend Following | GOLD + BITCOIN")
    with st.sidebar:
        st.header("Control Panel")
        if st.button("Refresh Market Data",use_container_width=True): st.cache_data.clear(); st.rerun()
        auto=st.checkbox("Auto refresh",True)
        st.caption("Initial validity: 45 minutes | Scalping / Intraday / Swing")
    if auto:
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=15000,key="khangold_refresh")
        except Exception: pass
    manage_all()
    tg,bg,tm,hi,pe=st.tabs(["🥇 GOLD","₿ BITCOIN","Trade Management","History","Performance"])
    with tg:
        st.header("GOLD — XAU/USD")
        for p in PROFILES: route("GOLD",p)
    with bg:
        st.header("BITCOIN — BTC/USD")
        for p in PROFILES: route("BITCOIN",p)
    with tm:
        try:
            rows=get_active()
            if not rows: st.info("No active signals or trades.")
            for r in rows: show_signal(r)
            st.caption("Trade fill and target events are estimated from closed candles, not broker-confirmed.")
        except Exception as e: st.error(str(e))
    with hi:
        try:
            rows=get_history(200)
            if rows: st.dataframe([{k:r.get(k) for k in ("signal_key","asset","profile","direction","score","entry_price","stop_loss","tp1","tp2","tp3","status","generated_at","closed_at")} for r in rows],use_container_width=True,hide_index=True)
            else: st.info("No saved signals yet.")
        except Exception as e: st.error(str(e))
    with pe:
        try:
            rows=get_history(1000); done=[r for r in rows if r.get("status") in ("TP3_HIT","SL_HIT")]; wins=sum(r.get("status")=="TP3_HIT" for r in done); losses=sum(r.get("status")=="SL_HIT" for r in done); total=len(done)
            a,b,c=st.columns(3); a.metric("Completed",total); b.metric("TP3",wins); c.metric("SL",losses); st.metric("TP3 rate",f"{wins/total*100:.1f}%" if total else "—")
            st.caption("Based only on persisted estimated lifecycle statuses; not a guarantee of future results.")
        except Exception as e: st.error(str(e))
    st.divider(); st.caption("Analytical signals only. No broker orders are placed. Verify spread, slippage and risk independently.")

if __name__=="__main__": dashboard()
