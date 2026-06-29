"""
═══════════════════════════════════════════════════════════════════════════════
 DWARKESH CAPITAL · EMA TOUCH CANDLE SCANNER  (Streamlit)
═══════════════════════════════════════════════════════════════════════════════
 The candle  : lower wick taps the EMA10/EMA20 band and the candle closes back
               above BOTH EMAs (bullish reclaim) — same logic as your script.
 Per stock   : YES / NO for  RSI>60 · MACD>Signal · MACD>0 · Volume>20-day avg

 RUN (Anaconda Prompt on your Windows PC):
       streamlit run dwarkesh_candle_scanner_app.py
 (yfinance / NSE are blocked in Claude's sandbox — run this on your machine.)

 Notes
   • Click "Run scan" to download prices. After that, changing the touch
     settings, date, or filters re-filters INSTANTLY (no re-download).
   • Change the universe selection -> click "Run scan" again to refetch.
═══════════════════════════════════════════════════════════════════════════════
"""

import io
import time
import datetime as dt

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
#  FIXED INDICATOR SETTINGS  (match TradingView / your script)
# ─────────────────────────────────────────────────────────────────────────────
EMA_FAST, EMA_MID, EMA_SLOW = 10, 20, 50
RSI_LEN = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
VOL_AVG_LEN = 20
LOOKBACK_DAYS = 400
BATCH_SIZE = 120

NSE_INDEX_CSVS = {
    "Nifty 500":    "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    "Midcap 150":   "https://archives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
    "Smallcap 250": "https://archives.nseindia.com/content/indices/ind_niftysmallcap250list.csv",
}

# ═════════════════════════════════════════════════════════════════════════════
#  SCANNER CORE  (identical maths to the validated standalone script)
# ═════════════════════════════════════════════════════════════════════════════
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi_wilder(close, length=14):
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_l = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    out   = 100 - (100 / (1 + rs))
    out   = out.mask((avg_l == 0) & (avg_g > 0), 100.0)
    return out

def macd(close, fast=12, slow=26, signal=9):
    line = ema(close, fast) - ema(close, slow)
    sig  = line.ewm(span=signal, adjust=False).mean()
    return line, sig

NEEDED = ["Open", "High", "Low", "Close", "Volume"]

def extract_symbol(raw, sym):
    yf_sym = sym + ".NS"
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            lvl0 = raw.columns.get_level_values(0)
            lvl1 = raw.columns.get_level_values(1)
            if yf_sym in lvl0:
                df = raw[yf_sym].copy()
            elif yf_sym in lvl1:
                df = raw.xs(yf_sym, axis=1, level=1).copy()
            else:
                return None
        else:
            df = raw.copy()
        if not all(c in df.columns for c in NEEDED):
            return None
        df = df[NEEDED].apply(pd.to_numeric, errors="coerce").dropna(how="all")
        return df if len(df) > 60 else None
    except Exception:
        return None

def add_indicators(df):
    c = df["Close"]
    df["EMA_F"] = ema(c, EMA_FAST)
    df["EMA_M"] = ema(c, EMA_MID)
    df["EMA_S"] = ema(c, EMA_SLOW)
    df["RSI"]   = rsi_wilder(c, RSI_LEN)
    df["MACD"], df["SIG"] = macd(c, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    return df

def _nse_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
        "Accept-Language": "en-US,en;q=0.9", "Accept": "text/csv,*/*",
    })
    try:
        s.get("https://www.nseindia.com", timeout=10)
    except Exception:
        pass
    return s

@st.cache_data(show_spinner=False, ttl=6 * 3600)
def load_universe(selected):
    syms, sess, report = set(), _nse_session(), {}
    for name in selected:
        url = NSE_INDEX_CSVS[name]
        try:
            r = sess.get(url, timeout=15); r.raise_for_status()
            d = pd.read_csv(io.StringIO(r.text))
            col = "Symbol" if "Symbol" in d.columns else d.columns[2]
            got = d[col].dropna().astype(str).str.strip().tolist()
            syms.update(got); report[name] = len(got)
        except Exception as e:
            report[name] = f"FAILED ({e})"
    return sorted(s for s in syms if s and s.isascii()), report

@st.cache_data(show_spinner=False, ttl=3600)
def download_batch_cached(symbols_tuple, lookback):
    tickers = [s + ".NS" for s in symbols_tuple]
    return yf.download(tickers, period=f"{lookback}d", interval="1d",
                       group_by="ticker", auto_adjust=False, threads=True, progress=False)

def fetch_prices(symbols, progress_cb=None):
    """Download + enrich every symbol. Returns (dict sym->df_with_indicators, failed list)."""
    prices, failed = {}, []
    batches = [symbols[i:i + BATCH_SIZE] for i in range(0, len(symbols), BATCH_SIZE)]
    done = 0
    for batch in batches:
        try:
            raw = download_batch_cached(tuple(batch), LOOKBACK_DAYS)
        except Exception:
            failed.extend(batch); raw = None
        if raw is not None:
            for sym in batch:
                df = extract_symbol(raw, sym)
                if df is None:
                    failed.append(sym)
                else:
                    prices[sym] = add_indicators(df)
        done += len(batch)
        if progress_cb:
            progress_cb(done, len(symbols))
    # one retry pass for misses
    if failed:
        retry, failed = failed, []
        rb = [retry[i:i + BATCH_SIZE] for i in range(0, len(retry), BATCH_SIZE)]
        for batch in rb:
            try:
                raw = download_batch_cached(tuple(batch), LOOKBACK_DAYS)
            except Exception:
                failed.extend(batch); continue
            for sym in batch:
                df = extract_symbol(raw, sym)
                if df is None:
                    failed.append(sym)
                else:
                    prices[sym] = add_indicators(df)
    return prices, failed

def target_position(index, target_date):
    if target_date is None:
        return len(index) - 1
    tgt = pd.to_datetime(target_date).normalize()
    pos = np.where(index.normalize() <= tgt)[0]
    return int(pos[-1]) if len(pos) else None

def evaluate(sym, df, target_date, touch_mode, tol, require_bullish):
    pos = target_position(df.index, target_date)
    if pos is None or pos < max(VOL_AVG_LEN, EMA_SLOW):
        return None
    row = df.iloc[pos]
    if row[["EMA_F", "EMA_M", "RSI", "MACD", "SIG"]].isna().any():
        return None

    close = float(row["Close"]); openp = float(row["Open"]); low = float(row["Low"])
    e_f = float(row["EMA_F"]);   e_m = float(row["EMA_M"]);  e_s = float(row["EMA_S"])
    band_hi, band_lo = max(e_f, e_m), min(e_f, e_m)

    touched = (low <= band_lo * (1 + tol)) if touch_mode == "both" \
              else (low <= band_hi * (1 + tol))
    closes_above = close > e_f and close > e_m
    bullish = close > openp
    if not (touched and closes_above and (bullish or not require_bullish)):
        return None

    rsi_v = float(row["RSI"]); macd_v = float(row["MACD"]); sig_v = float(row["SIG"])
    vol_v = float(row["Volume"])
    vol_avg = float(df["Volume"].iloc[pos - VOL_AVG_LEN:pos].mean())

    c1, c2, c3, c4 = rsi_v > 60, macd_v > sig_v, macd_v > 0, vol_v > vol_avg
    return {
        "Symbol": sym, "Date": df.index[pos].strftime("%Y-%m-%d"),
        "Close": round(close, 2), "EMA10": round(e_f, 2), "EMA20": round(e_m, 2),
        "EMA50": round(e_s, 2), "RSI": round(rsi_v, 1),
        "MACD": round(macd_v, 3), "Signal": round(sig_v, 3),
        "Vol": int(vol_v), "VolAvg20": int(vol_avg),
        "VolRatio": round(vol_v / vol_avg, 2) if vol_avg else 0.0,
        "C1": c1, "C2": c2, "C3": c3, "C4": c4,
        "Met": int(c1) + int(c2) + int(c3) + int(c4),
        "AllPass": c1 and c2 and c3 and c4,
    }

def run_evaluation(prices, target_date, touch_mode, tol, require_bullish):
    out = []
    for sym, df in prices.items():
        try:
            r = evaluate(sym, df, target_date, touch_mode, tol, require_bullish)
            if r:
                out.append(r)
        except Exception:
            pass
    return out

# ═════════════════════════════════════════════════════════════════════════════
#  UI
# ═════════════════════════════════════════════════════════════════════════════
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root{
  --bg:#0A0E13; --panel:#121A23; --panel2:#0E151D; --line:#1F2A35; --line-soft:#18222C;
  --ink:#E8EEF4; --dim:#9AA7B4; --faint:#5E6C79;
  --cyan:#22D3EE; --gold:#E4B95B; --green:#34D399; --rose:#F2768D; --off:#37434F;
}
.stApp{ background:radial-gradient(1100px 600px at 80% -10%, #11202b 0%, var(--bg) 55%) fixed; color:var(--ink); }
#MainMenu, header[data-testid="stHeader"], footer{ visibility:hidden; height:0; }
.block-container{ padding-top:1.2rem; padding-bottom:3rem; max-width:1180px; }
*{ font-family:'Inter',sans-serif; }
.mono,.num{ font-family:'JetBrains Mono',monospace; font-variant-numeric:tabular-nums; }

/* ── Header ─────────────────────────────────────────── */
.hero{ display:flex; align-items:center; gap:16px; padding:6px 2px 18px; border-bottom:1px solid var(--line-soft); margin-bottom:22px; }
.mark{ width:46px;height:46px;border:1px solid var(--line); border-radius:13px; background:linear-gradient(160deg,#13202b,#0d141c);
       display:flex;align-items:center;justify-content:center; box-shadow:0 0 0 1px rgba(34,211,238,.06), inset 0 0 22px rgba(34,211,238,.05); }
.brand{ font-size:12px; letter-spacing:.34em; text-transform:uppercase; color:var(--gold); font-weight:600; }
.title{ font-size:25px; font-weight:800; letter-spacing:-.02em; margin:1px 0 2px; }
.subtitle{ font-size:13px; color:var(--dim); }

/* ── Metric cards ───────────────────────────────────── */
.metrics{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:4px 0 22px; }
.mcard{ background:linear-gradient(180deg,var(--panel),var(--panel2)); border:1px solid var(--line); border-radius:14px; padding:15px 16px; }
.mcard .k{ font-size:11px; letter-spacing:.13em; text-transform:uppercase; color:var(--faint); }
.mcard .v{ font-size:27px; font-weight:700; margin-top:6px; }
.mcard.green .v{ color:var(--green); }  .mcard.cyan .v{ color:var(--cyan); }  .mcard.gold .v{ color:var(--gold); }

/* ── Stock rows (the signature device) ──────────────── */
.rows{ display:flex; flex-direction:column; gap:9px; }
.row{ position:relative; display:grid; grid-template-columns:18px 200px 1fr 64px; align-items:center; gap:14px;
      background:var(--panel2); border:1px solid var(--line-soft); border-radius:13px; padding:13px 16px 13px 0; overflow:hidden;
      transition:transform .12s ease, border-color .12s ease; }
.row:hover{ transform:translateY(-1px); border-color:#2b3a47; }
.bar{ width:5px; height:100%; position:absolute; left:0; top:0; }
.m0 .bar,.m1 .bar{ background:var(--off); }  .m2 .bar{ background:var(--gold); }
.m3 .bar{ background:var(--cyan); }
.m4{ border-color:rgba(52,211,153,.45); box-shadow:0 0 0 1px rgba(52,211,153,.18), 0 6px 24px -12px rgba(52,211,153,.5); }
.m4 .bar{ background:var(--green); }
.sym{ font-weight:700; font-size:15px; letter-spacing:-.01em; padding-left:16px; }
.sub{ font-size:12px; color:var(--dim); padding-left:16px; margin-top:2px; }
.sub b{ color:var(--ink); font-weight:600; }
.pills{ display:flex; gap:7px; flex-wrap:wrap; }
.pill{ font-size:11.5px; font-weight:600; padding:5px 10px; border-radius:8px; border:1px solid var(--line);
       color:var(--faint); background:rgba(255,255,255,.015); white-space:nowrap; display:inline-flex; gap:6px; align-items:center; }
.pill .dot{ width:6px;height:6px;border-radius:50%; background:var(--off); }
.pill.on{ color:#7EE7C0; border-color:rgba(52,211,153,.32); background:rgba(52,211,153,.10); }
.pill.on .dot{ background:var(--green); box-shadow:0 0 7px rgba(52,211,153,.8); }
.met{ text-align:right; font-family:'JetBrains Mono',monospace; font-weight:700; font-size:15px; padding-right:4px; }
.met span{ color:var(--faint); font-size:12px; }

/* ── Sidebar ────────────────────────────────────────── */
section[data-testid="stSidebar"]{ background:#0C131A; border-right:1px solid var(--line-soft); }
section[data-testid="stSidebar"] .block-container{ padding-top:1.4rem; }
.sb-h{ font-size:11px; letter-spacing:.16em; text-transform:uppercase; color:var(--faint); margin:18px 0 6px; font-weight:600; }
.stButton>button{ width:100%; background:linear-gradient(180deg,#1ec8e4,#129fbb); color:#04161b; font-weight:700;
                  border:0; border-radius:11px; padding:.6rem 1rem; letter-spacing:.01em; }
.stButton>button:hover{ filter:brightness(1.06); color:#04161b; }
.stDownloadButton>button{ width:100%; background:var(--panel); color:var(--ink); border:1px solid var(--line); border-radius:10px; }

/* ── Misc ───────────────────────────────────────────── */
.legend{ font-size:12.5px; color:var(--dim); background:var(--panel2); border:1px solid var(--line-soft);
         border-radius:12px; padding:13px 16px; margin-bottom:18px; line-height:1.55; }
.legend b{ color:var(--ink); }
.empty{ text-align:center; padding:60px 20px; color:var(--dim); border:1px dashed var(--line); border-radius:16px; background:var(--panel2); }
.empty .big{ font-size:18px; color:var(--ink); font-weight:600; margin-bottom:6px; }
@media (prefers-reduced-motion:reduce){ .row{ transition:none; } }
"""

CANDLE_SVG = ("<svg width='22' height='26' viewBox='0 0 22 26' fill='none'>"
    "<line x1='6' y1='2' x2='6' y2='24' stroke='#5E6C79' stroke-width='1.4'/>"
    "<rect x='2.5' y='8' width='7' height='10' rx='1.4' fill='#22D3EE'/>"
    "<line x1='16' y1='5' x2='16' y2='22' stroke='#5E6C79' stroke-width='1.4'/>"
    "<rect x='12.5' y='10' width='7' height='8' rx='1.4' fill='#F2768D'/></svg>")


def header():
    st.markdown(
        f"<div class='hero'><div class='mark'>{CANDLE_SVG}</div>"
        "<div><div class='brand'>Dwarkesh Capital</div>"
        "<div class='title'>EMA Touch Candle Scanner</div>"
        "<div class='subtitle'>Daily NSE setups where price taps the EMA10/20 band and reclaims it.</div>"
        "</div></div>", unsafe_allow_html=True)


def metric_cards(scanned, matched, allpass, failed):
    st.markdown(
        "<div class='metrics'>"
        f"<div class='mcard'><div class='k'>Scanned</div><div class='v num'>{scanned}</div></div>"
        f"<div class='mcard cyan'><div class='k'>Candle matches</div><div class='v num'>{matched}</div></div>"
        f"<div class='mcard green'><div class='k'>Pass all 4</div><div class='v num'>{allpass}</div></div>"
        f"<div class='mcard gold'><div class='k'>Couldn't fetch</div><div class='v num'>{failed}</div></div>"
        "</div>", unsafe_allow_html=True)


def stock_rows_html(rows):
    labels = [("C1", "RSI &gt; 60"), ("C2", "MACD &gt; Signal"),
              ("C3", "MACD &gt; 0"), ("C4", "Vol &gt; Avg20")]
    html = "<div class='rows'>"
    for r in rows:
        pills = "".join(
            f"<span class='pill {'on' if r[k] else ''}'><span class='dot'></span>{lab}</span>"
            for k, lab in labels)
        html += (
            f"<div class='row m{r['Met']}'><div class='bar'></div>"
            f"<div></div>"
            f"<div><div class='sym'>{r['Symbol']}</div>"
            f"<div class='sub'>&#8377;<b>{r['Close']:,.2f}</b> &middot; RSI {r['RSI']:.0f} "
            f"&middot; Vol <b>{r['VolRatio']:.1f}&times;</b></div></div>"
            f"<div class='pills'>{pills}</div>"
            f"<div class='met'>{r['Met']}<span>/4</span></div></div>")
    return html + "</div>"


def to_excel_bytes(df):
    buf = io.BytesIO()
    out = df.copy()
    for c in ["C1", "C2", "C3", "C4", "AllPass"]:
        out[c] = out[c].map({True: "YES", False: "NO"})
    out = out.rename(columns={"C1": "RSI>60", "C2": "MACD>Signal", "C3": "MACD>0",
                              "C4": "Vol>Avg20", "AllPass": "All_4_Pass"})
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        out.to_excel(w, index=False, sheet_name="Candle Scan")
    return buf.getvalue()



def main():
    st.set_page_config(page_title="Dwarkesh · Candle Scanner", page_icon="📈", layout="wide")
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
    header()

    if "prices" not in st.session_state:
        st.session_state.prices = None
        st.session_state.failed = []

    # ── Sidebar (desktop) — mirrors the main-page controls ───────────────────
    with st.sidebar:
        st.markdown("<div class='sb-h'>Universe</div>", unsafe_allow_html=True)
        sel_sb = [name for name in NSE_INDEX_CSVS
                  if st.checkbox(name, value=True, key=f"sb_{name}")]
        st.markdown("<div class='sb-h'>Scan day</div>", unsafe_allow_html=True)
        scan_date_sb = st.date_input("Candle date", value=dt.date.today(),
                                     label_visibility="collapsed", key="sb_date")
        st.markdown("<div class='sb-h'>Candle definition</div>", unsafe_allow_html=True)
        touch_label_sb = st.radio("Wick must reach",
                                  ["Top of EMA band", "Below both EMAs"], key="sb_touch")
        tol_sb = st.slider("Touch tolerance %", 0.0, 1.0, 0.20, 0.05, key="sb_tol") / 100.0
        require_bullish_sb = st.checkbox("Candle must be green", value=True, key="sb_bull")
        st.markdown("<div class='sb-h'>&nbsp;</div>", unsafe_allow_html=True)
        run_sb = st.button("⬇  Run scan", type="primary", key="run_sb")

    # ── Main-page controls (always visible, works on mobile) ─────────────────
    st.markdown("<div class='sb-h'>Settings</div>", unsafe_allow_html=True)

    # Universe checkboxes in one row
    ucols = st.columns(3)
    sel_main = []
    for i, name in enumerate(NSE_INDEX_CSVS):
        with ucols[i]:
            if st.checkbox(name, value=True, key=f"mp_{name}"):
                sel_main.append(name)

    # Date + candle settings in two columns
    mc1, mc2 = st.columns(2)
    with mc1:
        scan_date_main = st.date_input("Scan date", value=dt.date.today(), key="mp_date")
        touch_label_main = st.radio("Wick must reach",
                                    ["Top of EMA band", "Below both EMAs"], key="mp_touch")
    with mc2:
        tol_main = st.slider("Touch tolerance %", 0.0, 1.0, 0.20, 0.05, key="mp_tol") / 100.0
        require_bullish_main = st.checkbox("Candle must be green (Close > Open)",
                                           value=True, key="mp_bull")

    # Big prominent Run Scan button on main page
    run_main = st.button("⬇  Run Scan", type="primary", use_container_width=True, key="run_main")

    st.markdown("<hr style='border-color:#1F2A35;margin:16px 0 20px'>", unsafe_allow_html=True)

    # Merge: sidebar takes priority if used, else main-page values
    run           = run_sb or run_main
    sel           = sel_sb if run_sb else sel_main
    scan_date     = scan_date_sb if run_sb else scan_date_main
    touch_label   = touch_label_sb if run_sb else touch_label_main
    touch_mode    = "both" if touch_label == "Below both EMAs" else "band_top"
    tol           = tol_sb if run_sb else tol_main
    require_bullish = require_bullish_sb if run_sb else require_bullish_main

    # ── Run scan ──────────────────────────────────────────────────────────────
    if run:
        if not sel:
            st.warning("Pick at least one index.")
        else:
            with st.status("Loading NSE universe…", expanded=False) as status:
                universe, report = load_universe(tuple(sel))
                for k, v in report.items():
                    status.write(f"{k}: {v}")
                if not universe:
                    status.update(label="Universe unreachable", state="error")
                    st.error("Couldn't load symbols from NSE. Check your internet connection.")
                    st.stop()
                status.update(label=f"Universe ready — {len(universe)} symbols", state="complete")

            bar = st.progress(0.0, text="Downloading prices…")
            def cb(done, total):
                bar.progress(min(done / total, 1.0), text=f"Downloading prices… {done}/{total}")
            prices, failed = fetch_prices(universe, cb)
            bar.empty()
            st.session_state.prices = prices
            st.session_state.failed = failed
            st.session_state.scanned = len(universe)

    # ── Legend ────────────────────────────────────────────────────────────────
    st.markdown(
        "<div class='legend'><b>The candle:</b> lower wick taps the EMA10/EMA20 band and "
        "closes above <b>both</b> EMAs (green reclaim). &nbsp;|&nbsp; "
        "<b>Four checks</b> (info only, don't filter): "
        "RSI&nbsp;&gt;&nbsp;60 &middot; MACD&nbsp;&gt;&nbsp;Signal &middot; "
        "MACD&nbsp;&gt;&nbsp;0 &middot; Volume&nbsp;&gt;&nbsp;20-day avg.</div>",
        unsafe_allow_html=True)

    # ── Results ───────────────────────────────────────────────────────────────
    prices = st.session_state.prices
    if prices is None:
        st.markdown("<div class='empty'><div class='big'>No scan yet</div>"
                    "Set your universe and date above, then tap <b>Run Scan</b>.</div>",
                    unsafe_allow_html=True)
        return
    if not prices:
        st.markdown("<div class='empty'><div class='big'>Couldn't fetch any prices</div>"
                    "Check your internet connection.</div>", unsafe_allow_html=True)
        return

    td = None if scan_date >= dt.date.today() else scan_date
    results = run_evaluation(prices, td, touch_mode, tol, require_bullish)

    scanned = st.session_state.get("scanned", len(prices))
    n_all = sum(1 for r in results if r["AllPass"])
    metric_cards(scanned, len(results), n_all, len(st.session_state.failed))

    if not results:
        st.markdown("<div class='empty'><div class='big'>No stocks printed this candle "
                    "on the chosen day</div>Try loosening the touch tolerance.</div>",
                    unsafe_allow_html=True)
        return

    # filter + sort + download
    fc1, fc2, fc3 = st.columns([2, 2, 1.4])
    with fc1:
        view = st.radio("Show", ["All matches", "Pass all 4 only"],
                        horizontal=True, label_visibility="collapsed")
    with fc2:
        sort_by = st.selectbox("Sort by",
                               ["Conditions met", "RSI", "Volume ratio", "Symbol A-Z"],
                               label_visibility="collapsed")
    df = pd.DataFrame(results)
    if view == "Pass all 4 only":
        df = df[df["AllPass"]]
    sort_map = {"Conditions met": (["Met", "RSI"], [False, False]),
                "RSI": (["RSI"], [False]),
                "Volume ratio": (["VolRatio"], [False]),
                "Symbol A-Z": (["Symbol"], [True])}
    cols_s, asc = sort_map[sort_by]
    df = df.sort_values(cols_s, ascending=asc).reset_index(drop=True)
    with fc3:
        st.download_button("Download Excel", to_excel_bytes(df),
                           file_name=f"candle_scan_{results[0]['Date']}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.markdown(stock_rows_html(df.to_dict("records")), unsafe_allow_html=True)

    with st.expander("Raw data table"):
        show = df.rename(columns={"C1": "RSI>60", "C2": "MACD>Signal",
                                  "C3": "MACD>0", "C4": "Vol>Avg20", "AllPass": "All 4"})
        st.dataframe(show, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
