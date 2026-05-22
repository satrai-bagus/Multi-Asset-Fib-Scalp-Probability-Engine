"""
Streamlit app: Fib TP/SL Predictor — ETHUSDT 1h

Production-style UI with strategy presets, disclaimers, and backtest validation.
"""
from datetime import datetime
from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd
import streamlit as st

from backtest_engine import (prepare_features, run_backtest, walk_forward_chunks,
                              run_backtest_multi_tp, walk_forward_chunks_multi_tp,
                              portfolio_backtest, detect_confluence,
                              predict_probs)


HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
MODEL_DIR = HERE / "models"


# ---------------------------------------------------------------------------
# Auto-discover available tickers from model files
# ---------------------------------------------------------------------------
def discover_tickers():
    tickers = []
    for pkl in MODEL_DIR.glob("xgb_*.pkl"):
        sym = pkl.stem.replace("xgb_", "")
        ds = DATA_DIR / f"dataset_{sym}_18m.xlsx"
        if ds.exists():
            tickers.append(sym)
    return sorted(tickers)


AVAILABLE_TICKERS = discover_tickers()
if not AVAILABLE_TICKERS:
    import streamlit as st
    st.error("Tidak ada model ditemukan di models/. Run `python build_pipeline.py SYMBOL` dulu.")
    st.stop()


# ---------------------------------------------------------------------------
# Strategy presets (chosen via _compare_strategies.py)
# ---------------------------------------------------------------------------
STRATEGIES = {
    "Conservative (54% winrate)": {
        "tp": 2.5, "sl": 2.5, "min_conf": 0.65, "min_body_pct": 0.008,
        "description": "1:1 R:R. Higher winrate, lower trade frequency. "
                       "Cocok untuk subscriber baru yang butuh psychological comfort.",
    },
    "Balanced (39% winrate, mid R:R)": {
        "tp": 3.6, "sl": 2.5, "min_conf": 0.60, "min_body_pct": 0.005,
        "description": "1.73:1 R:R. Tradeoff antara winrate dan trade frequency. "
                       "Subscriber butuh disciplined execution.",
    },
    "Aggressive (31% winrate, 4.26:1 R:R)": {
        "tp": 3.6, "sl": 1.61, "min_conf": 0.70, "min_body_pct": 0.005,
        "description": "High R:R catch-the-big-move strategy. Banyak losing streaks. "
                       "Hanya untuk experienced traders dengan capital management ketat. "
                       "JANGAN PAKAI LEVERAGE >2x.",
    },
}


# ---------------------------------------------------------------------------
# Per-ticker auto-tuned configs (output dari optimize_per_ticker.py)
# ---------------------------------------------------------------------------
def _load_json(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


OPTIMAL_PARAMS = _load_json(MODEL_DIR / "optimal_params.json")
SCALP_PARAMS = _load_json(MODEL_DIR / "scalp_params.json")

OPTIMAL_KEY = "🎯 Optimal (Per-Ticker Auto-Tuned, Spot)"
SCALP_KEY = "🔥 Scalp Futures (Many Signals, Tight Risk)"

if OPTIMAL_PARAMS:
    STRATEGIES[OPTIMAL_KEY] = {
        "tp": None, "sl": None, "min_conf": None, "min_body_pct": None,
        "is_per_ticker": True, "mode": "balanced",
        "description": "Spot-tuned per-ticker config. Calmar-optimized "
                       "(return/drawdown). Lebih jarang signal tapi setup lebih kuat.",
    }
if SCALP_PARAMS:
    STRATEGIES[SCALP_KEY] = {
        "tp": None, "sl": None, "min_conf": None, "min_body_pct": None,
        "is_per_ticker": True, "mode": "scalp",
        "description": "**FUTURES TUNED** — high signal frequency, tight TP/SL "
                       "(Fib 1.27-1.61), perp fees 0.08% RT. ~10-18 trades/day per ticker. "
                       "**Leverage 5-10x maks**. JANGAN over-leverage.",
    }


def get_strat_for_ticker(ticker, selected_strat_name):
    """Resolve strategy params per-ticker for Optimal/Scalp modes."""
    base = STRATEGIES.get(selected_strat_name, {})
    if base.get("is_per_ticker"):
        params_src = SCALP_PARAMS if base.get("mode") == "scalp" else OPTIMAL_PARAMS
        cfg = params_src.get(ticker)
        if cfg and "metrics" in cfg:
            return {
                "tp": cfg["tp2"],
                "tp1": cfg["tp1"],
                "tp2": cfg["tp2"],
                "sl": cfg["sl"],
                "min_conf": cfg["min_conf"],
                "min_body_pct": cfg["min_body_pct"],
                "is_per_ticker": True,
                "mode": base.get("mode", "balanced"),
                "metrics": cfg["metrics"],
            }
        # No config for this ticker -> fallback
        return {**STRATEGIES["Conservative (54% winrate)"], "tp1": 2.5, "tp2": 3.6}
    return {**base, "tp1": 2.5, "tp2": 3.6}


st.set_page_config(page_title="Fib TP/SL Predictor", page_icon="📈", layout="wide")


@st.cache_resource
def load_resources(ticker):
    df = pd.read_excel(DATA_DIR / f"dataset_{ticker}_18m.xlsx")
    df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"])
    bundle = joblib.load(MODEL_DIR / f"xgb_{ticker}.pkl")
    return df, bundle


# ---------------------------------------------------------------------------
# Shared feature-building & signal-computing helpers
# ---------------------------------------------------------------------------
def make_feature_dict(row):
    """Build the 32-feature dict from a single dataset row."""
    atr_safe = row["ATR 14"] if pd.notna(row["ATR 14"]) and row["ATR 14"] > 0 else 1
    return {
        "Bar Color Code": {"Green": 1, "Red": -1, "Doji": 0}.get(row["Bar Color"], 0),
        "Fib Zone": row["Fib Zone"] if pd.notna(row["Fib Zone"]) else 0,
        "Fib Position": row["Fib Position"] if pd.notna(row["Fib Position"]) else 0,
        "Body %": row["Body %"],
        "Upper Wick %": row["Upper Wick %"],
        "Lower Wick %": row["Lower Wick %"],
        "Range": row["Range"],
        "Range/ATR": row["Range/ATR"] if pd.notna(row["Range/ATR"]) else 0,
        "Body% MA5": row["Body% MA5"] if pd.notna(row["Body% MA5"]) else 0,
        "Body% Pct100": row["Body% Pct100"] if pd.notna(row["Body% Pct100"]) else 0.5,
        "ATR Pct100": row["ATR Pct100"] if pd.notna(row["ATR Pct100"]) else 0.5,
        "Streak": row["Streak"] if pd.notna(row["Streak"]) else 0,
        "Prev Body%": row["Prev Body%"] if pd.notna(row["Prev Body%"]) else 0,
        "Prev Bar Color": row["Prev Bar Color"] if pd.notna(row["Prev Bar Color"]) else 0,
        "Prev Range/ATR": row["Prev Range/ATR"] if pd.notna(row["Prev Range/ATR"]) else 0,
        "Close Ret 5": row["Close Ret 5"] if pd.notna(row["Close Ret 5"]) else 0,
        "SQZMOM Delta3": row["SQZMOM Delta3"] if pd.notna(row["SQZMOM Delta3"]) else 0,
        "Vol/MA20": row["Vol/MA20"] if pd.notna(row["Vol/MA20"]) else 1,
        "Vol Pct100": row["Vol Pct100"] if pd.notna(row["Vol Pct100"]) else 0.5,
        "Hour Sin": row["Hour Sin"], "Hour Cos": row["Hour Cos"],
        "DoW Sin": row["DoW Sin"], "DoW Cos": row["DoW Cos"],
        "BTC Ret 1h":     row.get("BTC Ret 1h",     0) if pd.notna(row.get("BTC Ret 1h", 0)) else 0,
        "BTC Ret 4h":     row.get("BTC Ret 4h",     0) if pd.notna(row.get("BTC Ret 4h", 0)) else 0,
        "BTC Ret 24h":    row.get("BTC Ret 24h",    0) if pd.notna(row.get("BTC Ret 24h", 0)) else 0,
        "BTC Body Dir":   row.get("BTC Body Dir",   0) if pd.notna(row.get("BTC Body Dir", 0)) else 0,
        "BTC Range/ATR":  row.get("BTC Range/ATR",  1) if pd.notna(row.get("BTC Range/ATR", 1)) else 1,
        "BTC ATR Pct100": row.get("BTC ATR Pct100", 0.5) if pd.notna(row.get("BTC ATR Pct100", 0.5)) else 0.5,
        "SQZMOM Value": row["SQZMOM Value"],
        "Mom Color Code": {"lime": 2, "green": 1, "maroon": -1, "red": -2}.get(row["Momentum Color"], 0),
        "Squeeze Code": {"Squeeze ON (black)": 2, "Squeeze OFF (gray)": 1,
                         "No Squeeze (blue)": 0}.get(row["Squeeze Status"], 0),
        "RSI 14": row["RSI 14"] if pd.notna(row["RSI 14"]) else 50,
        "ADX 14": row["ADX 14"] if pd.notna(row["ADX 14"]) else 20,
        "MACD Hist": (row["MACD"] - row["MACD Signal"]) if pd.notna(row["MACD"]) and pd.notna(row["MACD Signal"]) else 0,
        "HTF 4H Trend": row["HTF 4H Trend"],
        "Posisi Code": {"LONG": 1, "SHORT": -1, "NO TRADE": 0}.get(row["Raw Posisi"], 0),
        "Last TR / ATR": row["Last TR"] / atr_safe if pd.notna(row["Last TR"]) else 0,
    }


def compute_signal(row, bundle, strat):
    """Returns dict with TRADE verdict, direction, prices, confidence."""
    body_top = max(row["Open"], row["Close"])
    body_bot = min(row["Open"], row["Close"])
    body_len = abs(row["Close"] - row["Open"])
    body_pct_of_price = body_len / row["Close"] if row["Close"] > 0 else 0

    fib = {
        "1.27 Up":   body_top + 0.27 * body_len,
        "1.27 Down": body_bot - 0.27 * body_len,
        "1.61 Up":   body_top + 0.61 * body_len,
        "1.61 Down": body_bot - 0.61 * body_len,
        "2.5 Up":    body_top + 1.5  * body_len,
        "2.5 Down":  body_bot - 1.5  * body_len,
        "3.6 Up":    body_top + 2.6  * body_len,
        "3.6 Down":  body_bot - 2.6  * body_len,
    }

    feat = make_feature_dict(row)
    X_one = np.array([[feat[c] for c in bundle["feature_cols"]]])
    probs = predict_probs(bundle, X_one)[0]
    pd_ = dict(zip(bundle["target_cols"], probs))

    p_up = max(pd_["Fib 1.61 Up"], pd_["Fib 2.5 Up"])
    p_dn = max(pd_["Fib 1.61 Down"], pd_["Fib 2.5 Down"])
    conf = max(p_up, p_dn)
    direction = "LONG" if p_up > p_dn else "SHORT"
    tradable = (conf >= strat["min_conf"]) and (body_pct_of_price >= strat["min_body_pct"])

    if direction == "LONG":
        side = "Up"; opp = "Down"
    else:
        side = "Down"; opp = "Up"

    # Per-ticker TP1/TP2 levels (Optimal/Scalp mode use ticker-specific levels)
    tp1_lvl = strat.get("tp1", 2.5)
    tp2_lvl = strat.get("tp2", 3.6)
    single_tp = strat.get("tp", tp2_lvl)

    return {
        "tradable": tradable,
        "direction": direction,
        "confidence": conf,
        "body_pct_of_price": body_pct_of_price,
        "entry": row["Close"],
        "tp_price": fib[f"{single_tp} {side}"],
        "sl_price": fib[f"{strat['sl']} {opp}"],
        "tp1_price": fib[f"{tp1_lvl} {side}"],
        "tp2_price": fib[f"{tp2_lvl} {side}"],
        "tp1_level": tp1_lvl,
        "tp2_level": tp2_lvl,
        "fib_prices": fib,
        "probs_dict": pd_,
        "reward_pct": (single_tp - 1) * body_pct_of_price * 100,
        "risk_pct": (strat["sl"] - 1) * body_pct_of_price * 100,
    }


# ---------------------------------------------------------------------------
# Sidebar: ticker selector + strategy + model info + DISCLAIMER
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🪙 Pilih Crypto")
    default_idx = AVAILABLE_TICKERS.index("ETHUSDT") if "ETHUSDT" in AVAILABLE_TICKERS else 0
    ticker = st.selectbox(
        "Ticker",
        AVAILABLE_TICKERS,
        index=default_idx,
        help="Tiap ticker punya model + dataset sendiri (trained per-ticker).",
    )

    df_full, bundle = load_resources(ticker)
    FEATURES = bundle["feature_cols"]
    TARGETS = bundle["target_cols"]

    st.divider()
    st.header("⚙️ Strategy")
    # Default to Scalp if available, else Optimal, else Conservative
    strat_keys = list(STRATEGIES.keys())
    if SCALP_KEY in strat_keys:
        default_strat_idx = strat_keys.index(SCALP_KEY)
    elif OPTIMAL_KEY in strat_keys:
        default_strat_idx = strat_keys.index(OPTIMAL_KEY)
    else:
        default_strat_idx = 0
    strategy_name = st.selectbox(
        "Pilih risk profile",
        strat_keys,
        index=default_strat_idx,
    )
    # Resolve per-ticker config for Optimal/Scalp modes
    STRAT = get_strat_for_ticker(ticker, strategy_name)
    st.caption(STRATEGIES[strategy_name]["description"])

    if STRAT.get("is_per_ticker"):
        mode_label = "🔥 Scalp" if STRAT.get("mode") == "scalp" else "🎯 Optimal"
        st.success(f"{mode_label} config untuk **{ticker}**:")
        st.write(f"- TP1: Fib {STRAT['tp1']}  ·  TP2: Fib {STRAT['tp2']}  ·  SL: Fib {STRAT['sl']}")
        st.write(f"- Min Conf: {STRAT['min_conf']*100:.0f}%  ·  Min Body: {STRAT['min_body_pct']*100:.2f}%")
        m = STRAT.get("metrics", {})
        if m:
            st.write(f"- Backtest: **{m.get('trades', '?')} trades, WR {m.get('winrate_pct', 0):.1f}%, "
                     f"Ret {m.get('total_return_pct', 0):+.1f}%, DD {m.get('max_drawdown_pct', 0):.1f}%, "
                     f"Calmar {m.get('calmar', 0):.1f}**")
    else:
        st.write(f"**TP**: Fib {STRAT['tp']}  ·  **SL**: Fib {STRAT['sl']}")
        st.write(f"**Min confidence**: {STRAT['min_conf']*100:.0f}%")
    st.write(f"**Min body**: {STRAT['min_body_pct']*100:.2f}% of price")

    st.divider()
    st.header("📊 Model Info")
    st.caption(f"Type: {bundle.get('model_type', 'unknown')}")
    st.caption(f"Train period start: {bundle['test_period'][0][:10]}")
    st.caption(f"Test period: {bundle['test_period'][0]}  →  {bundle['test_period'][1]}")
    st.caption(f"Train: {bundle['n_train']} · Calib: {bundle.get('n_calib','?')} · Test: {bundle['n_test']}")

    with st.expander("AUC per target"):
        for k, v in bundle["auc_scores"].items():
            st.write(f"`{k:<16}` **{v:.3f}**")

    with st.expander("Feature importance"):
        fi = sorted(bundle["feature_importance"].items(), key=lambda x: -x[1])
        fi_df = pd.DataFrame(fi[:15], columns=["Feature", "Importance"])
        st.bar_chart(fi_df.set_index("Feature"))

    st.divider()
    st.error(
        "⚠️ **DISCLAIMER**\n\n"
        "Bukan financial advice. Hanya alat analisa.\n\n"
        "Past performance tidak menjamin future returns.\n\n"
        "Crypto trading sangat berisiko. **Leverage dapat menghabiskan modal Anda lebih cepat.**\n\n"
        "Risk per trade max 1-2% equity. Stop pakai strategy ini kalau realized winrate < backtest -10%."
    )


st.title(f"📈 Fib TP/SL Predictor — {ticker} 1h")
st.caption(f"Ticker aktif: **{ticker}**. Ganti di sidebar. "
           "ML-driven Fibonacci TP/SL dari Squeeze Momentum + 32 fitur bar/indicator.")

tab_scan, tab_portfolio, tab_pred, tab_bt, tab_about = st.tabs(
    ["📡 Today's Scanner", "💼 Portfolio", "🎯 Live Signal", "📊 Backtest", "ℹ️ About"]
)


# ===========================================================================
# TAB 0: TODAY'S SCANNER — scan all tickers at once
# ===========================================================================
with tab_scan:
    st.subheader(f"Scanner — semua {len(AVAILABLE_TICKERS)} ticker")
    st.caption(f"Strategi aktif: **{strategy_name}**  ·  "
               f"TP Fib {STRAT['tp']} / SL Fib {STRAT['sl']}  ·  "
               f"Conf ≥ {STRAT['min_conf']*100:.0f}%  ·  Body ≥ {STRAT['min_body_pct']*100:.2f}%")

    col_a, col_b = st.columns([3, 1])
    with col_a:
        # Bar selector — default ke latest bar (rule: ambil bar terakhir yg ada di semua ticker)
        # Pakai range yang sama untuk semua
        scan_dt = st.date_input("Tanggal scan (UTC)", value=None,
                                key="scan_date", help="Kosongkan untuk latest bar per ticker.")
    with col_b:
        scan_hour = st.selectbox("Jam (UTC)", ["Latest"] + [f"{h:02d}:00" for h in range(24)],
                                 index=0, key="scan_hour")

    # Loop all tickers, compute signal
    rows = []
    for tk in AVAILABLE_TICKERS:
        try:
            df_tk, bundle_tk = load_resources(tk)
        except Exception as e:
            rows.append({"Ticker": tk, "Status": "ERR", "Note": str(e)[:50]})
            continue

        # Pick the bar
        if scan_dt is not None and scan_hour != "Latest":
            hh = int(scan_hour.split(":")[0])
            target = pd.Timestamp(datetime.combine(scan_dt, datetime.min.time()).replace(hour=hh))
            sel = df_tk[df_tk["Datetime (UTC)"] == target]
            if sel.empty:
                rows.append({"Ticker": tk, "Status": "—", "Note": "bar tidak ada"})
                continue
            row_tk = sel.iloc[0]
        else:
            row_tk = df_tk.iloc[-1]

        try:
            # For Optimal/Scalp modes, each ticker uses its own auto-tuned config
            strat_for_tk = get_strat_for_ticker(tk, strategy_name)
            sig = compute_signal(row_tk, bundle_tk, strat_for_tk)
        except Exception as e:
            rows.append({"Ticker": tk, "Status": "ERR", "Note": str(e)[:50]})
            continue

        verdict = f"✅ {sig['direction']}" if sig["tradable"] else "⛔ NO"
        rows.append({
            "Ticker": tk,
            "Bar (UTC)": str(row_tk["Datetime (UTC)"]),
            "Verdict": verdict,
            "Entry": f"${sig['entry']:,.4f}" if sig["entry"] < 10 else f"${sig['entry']:,.2f}",
            "TP1 (2.5)": f"${sig['tp1_price']:,.4f}" if sig["tp1_price"] < 10 else f"${sig['tp1_price']:,.2f}",
            "TP2 (3.6)": f"${sig['tp2_price']:,.4f}" if sig["tp2_price"] < 10 else f"${sig['tp2_price']:,.2f}",
            "SL": f"${sig['sl_price']:,.4f}" if sig["sl_price"] < 10 else f"${sig['sl_price']:,.2f}",
            "Conf %": f"{sig['confidence']*100:.1f}",
            "Body %": f"{sig['body_pct_of_price']*100:.2f}",
        })

    scan_df = pd.DataFrame(rows)

    # Split into TRADE vs NO TRADE
    trades = scan_df[scan_df["Verdict"].str.startswith("✅")] if "Verdict" in scan_df.columns else pd.DataFrame()
    no_trades = scan_df[scan_df["Verdict"].str.startswith("⛔")] if "Verdict" in scan_df.columns else pd.DataFrame()
    errs = scan_df[~scan_df["Verdict"].str.startswith(("✅", "⛔"))] if "Verdict" in scan_df.columns else pd.DataFrame()

    n_tr = len(trades)

    # ---- Confluence detection on current bar ----
    if n_tr >= 3:
        long_n = trades[trades["Verdict"].str.contains("LONG")].shape[0]
        short_n = trades[trades["Verdict"].str.contains("SHORT")].shape[0]
        if long_n >= 3:
            longs = trades[trades["Verdict"].str.contains("LONG")]
            st.error(
                f"🔥 **CONFLUENCE DETECTED — {long_n} ticker LONG bersamaan!**  "
                f"({', '.join(longs['Ticker'].tolist())})  "
                f"Avg conf: {longs['Conf %'].astype(float).mean():.0f}%. "
                f"Possible alt-season / risk-on momentum. **HIGH CONVICTION signal.**"
            )
        if short_n >= 3:
            shorts = trades[trades["Verdict"].str.contains("SHORT")]
            st.error(
                f"🔥 **CONFLUENCE DETECTED — {short_n} ticker SHORT bersamaan!**  "
                f"({', '.join(shorts['Ticker'].tolist())})  "
                f"Avg conf: {shorts['Conf %'].astype(float).mean():.0f}%. "
                f"Possible risk-off / dump momentum. **HIGH CONVICTION signal.**"
            )

    st.markdown(f"### Active Signals: **{n_tr}** of {len(scan_df)} tickers")

    if n_tr > 0:
        st.dataframe(trades, hide_index=True, use_container_width=True)

        st.markdown("#### 📋 Bulk Publish Format")
        st.caption("Copy-paste ke channel subscriber Anda untuk publish multiple signals sekaligus.")
        bulk_lines = [f"📡 SIGNAL UPDATE — {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}\n"]
        for _, r in trades.iterrows():
            dir_emoji = "🟢" if "LONG" in r["Verdict"] else "🔴"
            bulk_lines.append(
                f"{dir_emoji} {r['Ticker']}: Entry {r['Entry']} → "
                f"TP1 {r['TP1 (2.5)']} / TP2 {r['TP2 (3.6)']} | "
                f"SL {r['SL']} | Conf {r['Conf %']}%"
            )
        bulk_lines.append("\n⚠️ Not financial advice. Max 2x leverage, 1-2% equity per trade.")
        st.code("\n".join(bulk_lines), language="text")
    else:
        st.info("Tidak ada signal TRADE saat ini. Tunggu jam berikutnya atau coba turunkan threshold di sidebar.")

    if len(no_trades) > 0:
        with st.expander(f"⛔ NO TRADE — {len(no_trades)} ticker (untuk monitoring)"):
            display_nt = no_trades.drop(columns=["TP1 (2.5)", "TP2 (3.6)", "SL"], errors="ignore")
            st.dataframe(display_nt, hide_index=True, use_container_width=True)

    if len(errs) > 0:
        with st.expander(f"⚠️ Errors — {len(errs)}"):
            st.dataframe(errs, hide_index=True, use_container_width=True)

    # ---- Historical Confluence Events (test set) ----
    with st.expander("🔥 Historical Confluence Events (test set, all tickers)"):
        st.caption(
            "Cari momen di mana 3+ ticker trigger TRADE arah sama dalam jam yang sama. "
            "Indikator alt-season / risk-off momentum. Berdasarkan test set ~82 hari terakhir."
        )

        @st.cache_data(show_spinner=False)
        def _historical_confluence(tickers, strat_dict):
            ticker_signals = {}
            for tk in tickers:
                df_tk = pd.read_excel(DATA_DIR / f"dataset_{tk}_18m.xlsx")
                df_tk["Datetime (UTC)"] = pd.to_datetime(df_tk["Datetime (UTC)"])
                df_tk = df_tk[(df_tk["Lookahead Bars"] == 48) & df_tk["SQZMOM Value"].notna()].reset_index(drop=True)
                df_tk = df_tk.dropna(subset=["Body% MA5", "Vol/MA20", "ATR Pct100"]).reset_index(drop=True)
                bundle_tk = joblib.load(MODEL_DIR / f"xgb_{tk}.pkl")
                i2 = int(len(df_tk) * 0.85)
                df_test = prepare_features(df_tk.iloc[i2:].reset_index(drop=True))

                X = df_test[bundle_tk["feature_cols"]].fillna(0).values
                probs = predict_probs(bundle_tk, X)
                p_up_max = np.maximum(probs[:, 0], probs[:, 2])
                p_dn_max = np.maximum(probs[:, 1], probs[:, 3])
                conf = np.maximum(p_up_max, p_dn_max)
                direction = np.where(p_up_max > p_dn_max, "LONG", "SHORT")

                tradable = (conf >= strat_dict["min_conf"]) & (df_test["body_abs_pct"] >= strat_dict["min_body_pct"])
                rows = []
                for j in range(len(df_test)):
                    if tradable[j]:
                        rows.append({
                            "Datetime (UTC)": df_test["Datetime (UTC)"].iloc[j],
                            "Direction": direction[j],
                            "Confidence": float(conf[j]),
                            "tradable": True,
                        })
                ticker_signals[tk] = pd.DataFrame(rows)
            return ticker_signals

        with st.spinner("Mencari confluence events di test set..."):
            signals_by_ticker = _historical_confluence(tuple(AVAILABLE_TICKERS), STRAT)
            conf_events = detect_confluence(signals_by_ticker, min_count=3)

        if len(conf_events) == 0:
            st.info("Tidak ada confluence event (3+ ticker same direction) di test set dengan filter ini. "
                    "Coba turunkan strategy threshold di sidebar.")
        else:
            st.write(f"**Total confluence events:** {len(conf_events)}")
            display = conf_events.head(30).copy()
            display["Avg Confidence %"] = display["Avg Confidence %"].round(1)
            display["Min Confidence %"] = display["Min Confidence %"].round(1)
            st.dataframe(display, hide_index=True, use_container_width=True)

            # Counts by direction
            longs = (conf_events["Direction"] == "LONG").sum()
            shorts = (conf_events["Direction"] == "SHORT").sum()
            cc1, cc2 = st.columns(2)
            cc1.metric("🟢 LONG confluences", longs)
            cc2.metric("🔴 SHORT confluences", shorts)

    st.caption(
        "💡 **Workflow admin**: Refresh tab ini setiap jam ke-1 menit (00:01, 01:01, dst). "
        "Copy Bulk Publish Format → paste ke channel subscriber. "
        "Ganti Strategy di sidebar untuk lihat hasil dengan risk profile berbeda."
    )


# ===========================================================================
# TAB 0b: PORTFOLIO DASHBOARD — diversification analysis across multiple tickers
# ===========================================================================
with tab_portfolio:
    st.subheader("💼 Portfolio Diversification")
    st.caption(
        "Backtest gabungan kalau subscriber follow MULTIPLE ticker sekaligus dengan equal-weight allocation. "
        "Diversifikasi biasanya bikin drawdown lebih smooth karena tickers gak correlate 100%."
    )

    # Ticker multi-select — default top performers
    default_top = [t for t in ["LINKUSDT", "SUIUSDT", "AAVEUSDT", "ARBUSDT", "SOLUSDT"] if t in AVAILABLE_TICKERS]
    selected_tickers = st.multiselect(
        "Pilih ticker yang masuk portfolio",
        AVAILABLE_TICKERS,
        default=default_top,
        help="Equal-weight: tiap trade pakai 1/N modal. N = jumlah ticker.",
    )

    if not selected_tickers:
        st.info("Pilih minimal 1 ticker.")
        st.stop()

    p1, p2 = st.columns(2)
    with p1:
        port_mode = st.radio(
            "Mode backtest",
            ["Multi-TP (default)", "Single-TP"],
            index=0,
            horizontal=True,
            key="port_mode",
        )
    with p2:
        st.write("Strategy aktif dari sidebar:")
        st.write(f"**{strategy_name}**  ·  TP {STRAT['tp']} / SL {STRAT['sl']}  ·  "
                 f"Conf {STRAT['min_conf']*100:.0f}%, Body {STRAT['min_body_pct']*100:.2f}%")

    use_multi_port = "Multi-TP" in port_mode

    # Build per-ticker trades
    @st.cache_data(show_spinner=False)
    def _portfolio_run(tickers, strat_dict, use_multi):
        per_ticker_trades = {}
        per_ticker_equity = {}
        per_ticker_stats = {}
        for tk in tickers:
            df_tk = pd.read_excel(DATA_DIR / f"dataset_{tk}_18m.xlsx")
            df_tk["Datetime (UTC)"] = pd.to_datetime(df_tk["Datetime (UTC)"])
            df_tk = df_tk[(df_tk["Lookahead Bars"] == 48) & df_tk["SQZMOM Value"].notna()].reset_index(drop=True)
            df_tk = df_tk.dropna(subset=["Body% MA5", "Vol/MA20", "ATR Pct100"]).reset_index(drop=True)
            bundle_tk = joblib.load(MODEL_DIR / f"xgb_{tk}.pkl")
            i2 = int(len(df_tk) * 0.85)
            df_test_tk = prepare_features(df_tk.iloc[i2:].reset_index(drop=True))

            if use_multi:
                trades_tk, eq_tk, stats_tk = run_backtest_multi_tp(
                    df_test_tk, bundle_tk,
                    tp1_level=2.5, tp2_level=3.6, sl_level=strat_dict["sl"],
                    tp1_partial=0.5,
                    min_confidence=strat_dict["min_conf"],
                    min_body_pct=strat_dict["min_body_pct"],
                )
            else:
                trades_tk, eq_tk, stats_tk = run_backtest(
                    df_test_tk, bundle_tk,
                    tp_level=strat_dict["tp"], sl_level=strat_dict["sl"],
                    min_confidence=strat_dict["min_conf"],
                    min_body_pct=strat_dict["min_body_pct"],
                )
            per_ticker_trades[tk] = trades_tk
            per_ticker_equity[tk] = eq_tk
            per_ticker_stats[tk] = stats_tk
        return per_ticker_trades, per_ticker_equity, per_ticker_stats

    with st.spinner(f"Running backtest untuk {len(selected_tickers)} ticker..."):
        per_trades, per_eq, per_stats = _portfolio_run(
            tuple(selected_tickers), STRAT, use_multi_port,
        )

    combined_trades, port_equity, port_stats = portfolio_backtest(per_trades, equal_weight=True)

    if not port_stats or port_stats["trades"] == 0:
        st.warning("Tidak ada trade dari ticker terpilih dengan filter ini.")
    else:
        # Headline metrics
        st.markdown("### Portfolio Performance")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Tickers", port_stats["n_tickers"])
        c2.metric("Total Trades", port_stats["trades"])
        c3.metric("Winrate", f"{port_stats['winrate_pct']:.1f}%",
                  delta=f"W{port_stats['wins']} L{port_stats['losses']}", delta_color="off")
        c4.metric("Total Return", f"{port_stats['total_return_pct']:+.2f}%")
        c5.metric("Max Drawdown", f"{port_stats['max_drawdown_pct']:+.2f}%")

        # Equity curve: portfolio + per-ticker for comparison
        st.markdown("### Equity Curves")
        st.caption("Garis tebal = portfolio. Garis tipis = per-ticker (kalau dijalankan sendiri-sendiri).")

        eq_df = pd.DataFrame({"💼 Portfolio (equal-weight)": port_equity})
        # Add per-ticker equity (resampled to common timeline)
        for tk in selected_tickers:
            if tk in per_eq and len(per_eq[tk]) > 0:
                eq_df[tk] = per_eq[tk]
        # Forward-fill so chart looks clean
        eq_df = eq_df.sort_index().ffill()
        st.line_chart(eq_df)

        # Diversification benefit
        st.markdown("### 📉 Diversification Benefit")
        single_returns = []
        single_dds = []
        for tk in selected_tickers:
            if tk in per_stats and per_stats[tk]:
                single_returns.append(per_stats[tk]["total_return_pct"])
                single_dds.append(per_stats[tk]["max_drawdown_pct"])

        if single_returns:
            avg_single_ret = sum(single_returns) / len(single_returns)
            avg_single_dd = sum(single_dds) / len(single_dds)
            d1, d2, d3 = st.columns(3)
            d1.metric(
                "Avg Single-Ticker Return",
                f"{avg_single_ret:+.2f}%",
                help="Rata-rata return masing-masing ticker kalau dijalankan independen",
            )
            d2.metric(
                "Avg Single-Ticker MaxDD",
                f"{avg_single_dd:+.2f}%",
                help="Rata-rata drawdown ticker individual",
            )
            d3.metric(
                "Portfolio MaxDD",
                f"{port_stats['max_drawdown_pct']:+.2f}%",
                delta=f"{port_stats['max_drawdown_pct']-avg_single_dd:+.2f}% vs avg single",
                help="Drawdown portfolio. Negatif delta = portfolio LEBIH BAIK (drawdown lebih kecil dari rata-rata individual).",
            )

            dd_improvement = port_stats['max_drawdown_pct'] - avg_single_dd
            if dd_improvement > 5:
                st.success(f"✅ **Diversifikasi BERHASIL**: MaxDD portfolio {dd_improvement:.1f}% lebih kecil dari rata-rata single ticker. Tickers kurang correlated → drawdown smoother.")
            elif dd_improvement < -5:
                st.warning(f"⚠️ Portfolio MaxDD {abs(dd_improvement):.1f}% lebih BESAR dari rata-rata single. Tickers high correlated atau bad chunks overlap.")
            else:
                st.info("ℹ️ Diversifikasi efek netral. Pilih ticker yang lebih beragam sectoral.")

        # Per-ticker contribution table
        st.markdown("### Per-Ticker Contribution")
        contrib_rows = []
        for tk in selected_tickers:
            s = per_stats.get(tk)
            pt = port_stats["per_ticker"].get(tk, {})
            if s:
                contrib_rows.append({
                    "Ticker": tk,
                    "Trades": pt.get("trades", 0),
                    "Winrate %": f"{pt.get('winrate', 0):.1f}",
                    "Avg P/L %": f"{pt.get('avg_pl', 0):+.3f}",
                    "Solo Total Ret %": f"{s.get('total_return_pct', 0):+.2f}",
                    "Solo MaxDD %": f"{s.get('max_drawdown_pct', 0):+.2f}",
                })
        contrib_df = pd.DataFrame(contrib_rows)
        st.dataframe(contrib_df, hide_index=True, use_container_width=True)

        st.markdown("### Marketing pitch")
        st.code(
            f"""📊 PORTFOLIO BACKTEST — {len(selected_tickers)} ticker, {'Multi-TP' if use_multi_port else 'Single-TP'} strategy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Periode test  : {bundle['test_period'][0]}  ->  {bundle['test_period'][1]}
Total trades  : {port_stats['trades']}
Winrate       : {port_stats['winrate_pct']:.1f}%
Total return  : {port_stats['total_return_pct']:+.2f}%   (in 82 days)
Max drawdown  : {port_stats['max_drawdown_pct']:+.2f}%

Tickers: {', '.join(selected_tickers)}

⚠️ Backtest only. Realistic live performance: ~30-60% dari paper.
   Subtract slippage + execution lag. Not financial advice.""",
            language="text",
        )


# ===========================================================================
# TAB 1: LIVE SIGNAL
# ===========================================================================
with tab_pred:
    dt_min = df_full["Datetime (UTC)"].min().to_pydatetime()
    dt_max = df_full["Datetime (UTC)"].max().to_pydatetime()

    st.subheader("1. Pilih bar")
    c1, c2 = st.columns([2, 1])
    with c1:
        date_pick = st.date_input("Tanggal (UTC)", value=dt_max.date(),
                                  min_value=dt_min.date(), max_value=dt_max.date())
    with c2:
        hour_pick = st.selectbox("Jam (UTC)", list(range(24)),
                                 index=min(dt_max.hour, 23))

    target_dt = pd.Timestamp(datetime.combine(date_pick, datetime.min.time()).replace(hour=hour_pick))
    row_df = df_full[df_full["Datetime (UTC)"] == target_dt]
    if row_df.empty:
        st.error(f"Bar {target_dt} tidak ada di dataset. Range: {dt_min} sampai {dt_max}.")
        st.stop()
    row = row_df.iloc[0]

    # ---- Compute fib levels + features + prediction ----
    body_top = max(row["Open"], row["Close"])
    body_bot = min(row["Open"], row["Close"])
    body_len = abs(row["Close"] - row["Open"])
    body_pct_of_price = body_len / row["Close"] if row["Close"] > 0 else 0

    fib_prices = {
        "1.61 Up":   body_top + 0.61 * body_len,
        "1.61 Down": body_bot - 0.61 * body_len,
        "2.5 Up":    body_top + 1.5  * body_len,
        "2.5 Down":  body_bot - 1.5  * body_len,
        "3.6 Up":    body_top + 2.6  * body_len,
        "3.6 Down":  body_bot - 2.6  * body_len,
    }

    # Build feature vector — must include all features the model needs
    atr_safe = row["ATR 14"] if pd.notna(row["ATR 14"]) and row["ATR 14"] > 0 else 1
    feat = {
        # Bar geometry
        "Bar Color Code": {"Green": 1, "Red": -1, "Doji": 0}.get(row["Bar Color"], 0),
        "Fib Zone": row["Fib Zone"] if pd.notna(row["Fib Zone"]) else 0,
        "Fib Position": row["Fib Position"] if pd.notna(row["Fib Position"]) else 0,
        "Body %": row["Body %"],
        "Upper Wick %": row["Upper Wick %"],
        "Lower Wick %": row["Lower Wick %"],
        "Range": row["Range"],
        "Range/ATR": row["Range/ATR"] if pd.notna(row["Range/ATR"]) else 0,
        # Multi-bar context
        "Body% MA5": row["Body% MA5"] if pd.notna(row["Body% MA5"]) else 0,
        "Body% Pct100": row["Body% Pct100"] if pd.notna(row["Body% Pct100"]) else 0.5,
        "ATR Pct100": row["ATR Pct100"] if pd.notna(row["ATR Pct100"]) else 0.5,
        "Streak": row["Streak"] if pd.notna(row["Streak"]) else 0,
        "Prev Body%": row["Prev Body%"] if pd.notna(row["Prev Body%"]) else 0,
        "Prev Bar Color": row["Prev Bar Color"] if pd.notna(row["Prev Bar Color"]) else 0,
        "Prev Range/ATR": row["Prev Range/ATR"] if pd.notna(row["Prev Range/ATR"]) else 0,
        "Close Ret 5": row["Close Ret 5"] if pd.notna(row["Close Ret 5"]) else 0,
        "SQZMOM Delta3": row["SQZMOM Delta3"] if pd.notna(row["SQZMOM Delta3"]) else 0,
        # Volume
        "Vol/MA20": row["Vol/MA20"] if pd.notna(row["Vol/MA20"]) else 1,
        "Vol Pct100": row["Vol Pct100"] if pd.notna(row["Vol Pct100"]) else 0.5,
        # Time-of-day
        "Hour Sin": row["Hour Sin"], "Hour Cos": row["Hour Cos"],
        "DoW Sin": row["DoW Sin"], "DoW Cos": row["DoW Cos"],
        # SQZMOM
        "SQZMOM Value": row["SQZMOM Value"],
        "Mom Color Code": {"lime": 2, "green": 1, "maroon": -1, "red": -2}.get(row["Momentum Color"], 0),
        "Squeeze Code": {"Squeeze ON (black)": 2, "Squeeze OFF (gray)": 1,
                         "No Squeeze (blue)": 0}.get(row["Squeeze Status"], 0),
        # Indicators
        "RSI 14": row["RSI 14"] if pd.notna(row["RSI 14"]) else 50,
        "ADX 14": row["ADX 14"] if pd.notna(row["ADX 14"]) else 20,
        "MACD Hist": (row["MACD"] - row["MACD Signal"]) if pd.notna(row["MACD"]) and pd.notna(row["MACD Signal"]) else 0,
        "HTF 4H Trend": row["HTF 4H Trend"],
        "Posisi Code": {"LONG": 1, "SHORT": -1, "NO TRADE": 0}.get(row["Raw Posisi"], 0),
        "Last TR / ATR": row["Last TR"] / atr_safe if pd.notna(row["Last TR"]) else 0,
    }
    X_one = np.array([[feat[c] for c in FEATURES]])
    probs = predict_probs(bundle, X_one)[0]
    probs_dict = dict(zip(TARGETS, probs))

    p_up_max = max(probs_dict["Fib 1.61 Up"], probs_dict["Fib 2.5 Up"])
    p_dn_max = max(probs_dict["Fib 1.61 Down"], probs_dict["Fib 2.5 Down"])
    confidence = max(p_up_max, p_dn_max)
    direction = "LONG" if p_up_max > p_dn_max else "SHORT"

    # ---- TRADABLE verdict ----
    st.subheader("2. Verdict — Layak Trade?")
    tradable = (confidence >= STRAT["min_conf"]) and (body_pct_of_price >= STRAT["min_body_pct"])

    if tradable:
        side = "Up" if direction == "LONG" else "Down"
        opp = "Down" if direction == "LONG" else "Up"
        tp_price = fib_prices[f"{STRAT['tp']} {side}"]
        sl_price = fib_prices[f"{STRAT['sl']} {opp}"]
        # ALSO compute TP1+TP2 for multi-target publishing (use 2.5 as TP1, 3.6 as TP2)
        tp1_price = fib_prices[f"2.5 {side}"]
        tp2_price = fib_prices[f"3.6 {side}"]
        entry = row["Close"]
        reward_pct = (STRAT["tp"] - 1) * body_pct_of_price * 100
        risk_pct = (STRAT["sl"] - 1) * body_pct_of_price * 100
        tp1_pct = 1.5 * body_pct_of_price * 100
        tp2_pct = 2.6 * body_pct_of_price * 100

        st.success(f"### ✅ TRADE — {direction}   ({strategy_name})")
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Entry", f"${entry:.2f}")
        v2.metric(f"TP @ Fib {STRAT['tp']} {side}", f"${tp_price:.2f}",
                  delta=f"+{reward_pct:.2f}%" if direction == "LONG" else f"-{reward_pct:.2f}%",
                  delta_color="normal" if direction == "LONG" else "inverse")
        v3.metric(f"SL @ Fib {STRAT['sl']} {opp}", f"${sl_price:.2f}",
                  delta=f"-{risk_pct:.2f}%" if direction == "LONG" else f"+{risk_pct:.2f}%",
                  delta_color="inverse" if direction == "LONG" else "normal")
        v4.metric("R : R", f"{(STRAT['tp']-1)/(STRAT['sl']-1):.2f} : 1",
                  delta=f"Conf {confidence*100:.0f}%")

        st.caption(
            f"**Holding period max**: 48 jam. **Position sizing**: max 1-2% equity per trade.\n\n"
            f"**Suggested leverage**: 1-3x (Conservative), 1-2x (Aggressive — NEVER 5x+)."
        )

        # ============= ADMIN PUBLISH SECTION =============
        st.markdown("---")
        st.markdown("### 📋 Quick Publish (untuk subscriber)")
        st.caption("Copy-paste teks di bawah ke Telegram / Discord / web subscriber. "
                   "TP1+TP2 multi-target supaya subscriber bisa partial exit + trail SL.")

        emoji_dir = "🟢 LONG" if direction == "LONG" else "🔴 SHORT"
        is_scalp_mode = STRAT.get("mode") == "scalp"
        # Per-mode tuning recommendations
        if is_scalp_mode:
            leverage_rec = "5-10x"  # tighter SL allows higher leverage safely
            hold_max = "max 24 jam (scalp = quick in/out)"
            mode_tag = "Futures Scalp"
            sl_caution = "(SL ketat ~Fib 1.27, hindari over-leverage)"
        else:
            leverage_rec = "2-3x (max 5x untuk pro)"
            hold_max = "max 48 jam"
            mode_tag = "Multi-TP"
            sl_caution = ""

        # Multi-TP exit plan: close 50% at TP1, move SL to entry (BE), ride 50% to TP2
        publish_text = (
            f"{emoji_dir}  {ticker}  ·  1h  ·  {mode_tag}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📍 Entry  : ${entry:,.4f}\n"
            f"🎯 TP1    : ${tp1_price:,.4f}   ({'+' if direction=='LONG' else '-'}{tp1_pct:.2f}%)  → close 50%, move SL to BE\n"
            f"🎯 TP2    : ${tp2_price:,.4f}   ({'+' if direction=='LONG' else '-'}{tp2_pct:.2f}%)  → close remaining 50%\n"
            f"🛑 SL     : ${sl_price:,.4f}   ({'-' if direction=='LONG' else '+'}{risk_pct:.2f}%) {sl_caution}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Confidence : {confidence*100:.0f}%  (model)\n"
            f"⚖️  R:R        : {(STRAT.get('tp2', STRAT.get('tp', 2.5))-1)/(STRAT['sl']-1):.2f} : 1\n"
            f"💪 Leverage   : {leverage_rec}\n"
            f"⏱️  Hold       : {hold_max}\n"
            f"🎚️  Strategy   : {strategy_name.split(' (')[0]}\n"
            f"\n"
            f"📋 EXIT PLAN:\n"
            f"  1. Close 50% di TP1 (lock in profit)\n"
            f"  2. Move SL ke entry (free trade)\n"
            f"  3. Ride sisa 50% ke TP2\n"
            f"  4. Kalau price balik ke entry sebelum TP2 → exit BE\n"
            f"\n"
            f"⚠️ FUTURES RISK: Leverage amplify gain DAN loss. Liquidation kalau over-leverage.\n"
            f"   Max risk per trade: 1-2% equity. Posisi sizing pakai Kelly atau fixed%.\n"
            f"   Not financial advice."
        )
        st.code(publish_text, language="text")

        # Also a one-liner for quick Telegram
        oneliner = (
            f"{emoji_dir} {ticker} 1h | Entry ${entry:,.0f} | "
            f"TP1 ${tp1_price:,.0f} | TP2 ${tp2_price:,.0f} | "
            f"SL ${sl_price:,.0f} | Conf {confidence*100:.0f}%"
        )
        st.text_input("One-liner (untuk header):", oneliner, disabled=False, key="oneliner")

        with st.expander("📖 Cara baca signal ini sebagai admin"):
            st.markdown(f"""
            **Urutan prioritas info untuk Anda:**

            1. **Verdict** — TRADE atau NO TRADE. Kalau NO TRADE → skip bar ini, tunggu jam berikutnya.
            2. **Direction** — LONG / SHORT. Wajib jelas di pesan ke subscriber.
            3. **3 Harga**: Entry, TP, SL. Itu yang subscriber butuhkan untuk eksekusi.
            4. **Confidence** — untuk subscriber decide ukuran posisi (semakin tinggi conf, makin agresif sizing).
            5. *(Optional)* Body %, Range/ATR, SQZMOM — untuk konten edukatif "kenapa signal ini muncul".

            **Workflow admin per jam:**
            1. Buka app jam 00:01, 01:01, 02:01... (1 menit setelah candle 1h close)
            2. Kalau muncul ✅ TRADE → copy publish text di atas → paste ke channel
            3. Insert ke MySQL (manual atau auto via signal_generator.py kalau sudah jadi)
            4. Tab Backtest untuk validasi performance setiap minggu
            5. Update statistik live ke subscriber: "Bulan ini X% accuracy, Y trades"

            **Yang TIDAK perlu dipublish ke subscriber:**
            - Detail indicator (RSI, ADX, SQZMOM) — bingungin newbie
            - Probability 6 fib level — terlalu teknis
            - Walk-forward chunks — internal validation Anda

            **Yang HARUS dipublish:**
            - Direction, Entry, TP, SL, Suggested leverage, Holding period
            - Disclaimer: not financial advice + leverage warning
            """)
    else:
        reasons = []
        if confidence < STRAT["min_conf"]:
            reasons.append(f"Confidence {confidence*100:.1f}% di bawah threshold {STRAT['min_conf']*100:.0f}%")
        if body_pct_of_price < STRAT["min_body_pct"]:
            reasons.append(f"Body {body_pct_of_price*100:.2f}% di bawah threshold {STRAT['min_body_pct']*100:.2f}% (terlalu kecil — fees akan makan)")
        st.warning(f"### ⛔ NO TRADE  ({strategy_name})")
        for r in reasons:
            st.write(f"   • {r}")
        st.caption("Tunggu setup yang match preset Anda. Patience > overtrading.")

    st.divider()

    # ---- Bar info ----
    st.subheader(f"3. Detail Bar {target_dt} UTC")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Open",  f"${row['Open']:.2f}")
    m2.metric("High",  f"${row['High']:.2f}")
    m3.metric("Low",   f"${row['Low']:.2f}")
    m4.metric("Close", f"${row['Close']:.2f}")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Bar Color", str(row["Bar Color"]))
    m6.metric("Body %",    f"{row['Body %']*100:.1f}%",
              delta=f"{body_pct_of_price*100:.2f}% of price")
    m7.metric("Range/ATR", f"{row['Range/ATR']:.2f}")
    m8.metric("Streak",    f"{int(row['Streak']):+d}")

    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("Score", int(row["Score"]), delta=row["Raw Posisi"], delta_color="off")
    a2.metric("ADX 14", f"{row['ADX 14']:.1f}")
    a3.metric("RSI 14", f"{row['RSI 14']:.1f}")
    a4.metric("HTF 4H", {1: "UP", -1: "DOWN", 0: "FLAT"}.get(int(row["HTF 4H Trend"]), "—"))
    a5.metric("Vol/MA20", f"{row['Vol/MA20']:.2f}",
              delta=f"pct {int(row['Vol Pct100']*100)}%", delta_color="off")

    # ---- All 6 probability bars ----
    with st.expander("📊 Lihat semua 6 probability fib level"):
        for tag in ["1.61", "2.5", "3.6"]:
            c1, c2 = st.columns(2)
            p_u = probs_dict[f"Fib {tag} Up"]
            p_d = probs_dict[f"Fib {tag} Down"]
            with c1:
                st.write(f"**Fib {tag} Up** @ `${fib_prices[f'{tag} Up']:.2f}`")
                st.progress(float(p_u))
                st.caption(f"{p_u*100:.1f}% probability hit")
            with c2:
                st.write(f"**Fib {tag} Down** @ `${fib_prices[f'{tag} Down']:.2f}`")
                st.progress(float(p_d))
                st.caption(f"{p_d*100:.1f}% probability hit")

    # ---- Actual outcome (verification) ----
    with st.expander("📋 Actual outcome (48h setelah bar ini)"):
        if row["Lookahead Bars"] < 48:
            st.warning(f"Lookahead belum lengkap ({int(row['Lookahead Bars'])}/48 jam).")
        else:
            actual_rows = []
            for tag in ["1.61", "2.5", "3.6"]:
                u_o = int(row[f"Fib {tag} Up Order"])
                d_o = int(row[f"Fib {tag} Down Order"])
                actual_rows.append({
                    "Level": f"Fib {tag}",
                    "Up": f"order {u_o}" if u_o > 0 else "—",
                    "Down": f"order {d_o}" if d_o > 0 else "—",
                })
            st.table(pd.DataFrame(actual_rows))


# ===========================================================================
# TAB 2: BACKTEST
# ===========================================================================
with tab_bt:
    st.subheader("Strategy Backtest")
    st.caption(f"Backtest pada test set (15% akhir dari 18 bulan): "
               f"{bundle['test_period'][0]} → {bundle['test_period'][1]}. "
               f"Fee 0.2% round-trip included. Tidak ada slippage modeling.")

    # Show current preset
    st.info(f"**Strategi aktif**: {strategy_name}")

    # Backtest mode selector
    bt_mode = st.radio(
        "Mode backtest",
        ["Single-TP (close 100% di TP)", "Multi-TP (50% di TP1, ride 50% ke TP2, BE setelah TP1) ⭐"],
        index=1,  # default multi-TP
        horizontal=False,
        help="Multi-TP biasanya kasih winrate +2-4% lebih tinggi (psikologi subscriber), "
             "tapi return absolut bisa lebih kecil. Single-TP lebih agresif profit-maxing."
    )
    use_multi_tp = "Multi-TP" in bt_mode

    # Manual override sliders
    with st.expander("🔧 Tweak parameter manual"):
        if use_multi_tp:
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                tp1_lvl = st.selectbox("TP1 Level", [1.61, 2.5, 3.6],
                                       index=[1.61, 2.5, 3.6].index(2.5))
            with c2:
                tp2_lvl = st.selectbox("TP2 Level", [1.61, 2.5, 3.6],
                                       index=[1.61, 2.5, 3.6].index(3.6))
            with c3:
                sl_lvl = st.selectbox("SL Level", [1.61, 2.5, 3.6],
                                      index=[1.61, 2.5, 3.6].index(STRAT["sl"]))
            with c4:
                min_conf = st.slider("Min Confidence", 0.0, 0.90, STRAT["min_conf"], 0.05)
            with c5:
                min_body_pct = st.slider("Min Body %", 0.0, 0.020,
                                         STRAT["min_body_pct"], 0.001, format="%.3f")
            tp1_partial = st.slider(
                "TP1 partial close (0.5 = 50% close di TP1)",
                0.2, 0.8, 0.5, 0.1,
                help="Berapa % posisi yang di-close saat TP1 hit. "
                     "Lower (0.3) = lock less, ride more ke TP2. "
                     "Higher (0.7) = lock more, less upside."
            )
            tp_lvl = tp1_lvl  # for backward-compat var name
        else:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                tp_lvl = st.selectbox("TP Level", [1.61, 2.5, 3.6],
                                      index=[1.61, 2.5, 3.6].index(STRAT["tp"]))
            with c2:
                sl_lvl = st.selectbox("SL Level", [1.61, 2.5, 3.6],
                                      index=[1.61, 2.5, 3.6].index(STRAT["sl"]))
            with c3:
                min_conf = st.slider("Min Confidence", 0.0, 0.90, STRAT["min_conf"], 0.05)
            with c4:
                min_body_pct = st.slider("Min Body % of price", 0.0, 0.020,
                                         STRAT["min_body_pct"], 0.001, format="%.3f")
            tp1_lvl, tp2_lvl, tp1_partial = 2.5, 3.6, 0.5  # placeholder

    # Run backtest
    df_clean = df_full[(df_full["Lookahead Bars"] == 48) & df_full["SQZMOM Value"].notna()].reset_index(drop=True)
    df_clean = df_clean.dropna(subset=["Body% MA5", "Vol/MA20", "ATR Pct100"]).reset_index(drop=True)
    i2 = int(len(df_clean) * 0.85)
    df_test = df_clean.iloc[i2:].reset_index(drop=True)
    df_test_feat = prepare_features(df_test)

    if use_multi_tp:
        trades, equity, stats = run_backtest_multi_tp(
            df_test_feat, bundle,
            tp1_level=tp1_lvl, tp2_level=tp2_lvl, sl_level=sl_lvl,
            tp1_partial=tp1_partial,
            min_confidence=min_conf, min_body_pct=min_body_pct,
        )
    else:
        trades, equity, stats = run_backtest(
            df_test_feat, bundle,
            tp_level=tp_lvl, sl_level=sl_lvl,
            min_confidence=min_conf, min_body_pct=min_body_pct,
        )

    if stats:
        st.markdown("#### Hasil")
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Trades", stats["trades"])
        if use_multi_tp:
            s2.metric("Winrate", f"{stats['winrate_pct']:.1f}%",
                      delta=f"W{stats['wins']} L{stats['losses']} BE{stats['breakevens']}",
                      delta_color="off")
        else:
            s2.metric("Winrate", f"{stats['winrate_pct']:.1f}%",
                      delta=f"W{stats['wins']} L{stats['losses']} N{stats.get('neutrals',0)}",
                      delta_color="off")
        s3.metric("Avg / trade", f"{stats['avg_pl_pct']:+.3f}%")
        s4.metric("Total Return", f"{stats['total_return_pct']:+.2f}%",
                  delta=f"MaxDD {stats['max_drawdown_pct']:.1f}%", delta_color="off")
        s5.metric("Trade Rate", f"{stats['trade_rate']:.1f}%",
                  delta="of bars", delta_color="off")

        # Outcome breakdown (multi-TP only)
        if use_multi_tp and "outcomes" in stats:
            st.markdown("#### Outcome breakdown (multi-TP)")
            oc = stats["outcomes"]
            o_cols = st.columns(len(oc))
            for col, (k, v) in zip(o_cols, sorted(oc.items(), key=lambda x: -x[1])):
                col.metric(k, v, delta=f"{v/stats['trades']*100:.0f}%", delta_color="off")

        st.markdown("#### Equity Curve")
        eq_df = pd.DataFrame({"Equity (x1.0 = start)": equity})
        st.line_chart(eq_df)

        st.markdown("#### Walk-Forward (test set di-split jadi 4 chunk)")
        st.caption("Cek apakah strategi konsisten. Kalau ada chunk negatif → strategy fragile pada regime tertentu.")
        if use_multi_tp:
            wf = walk_forward_chunks_multi_tp(
                df_test_feat, bundle, n_chunks=4,
                tp1_level=tp1_lvl, tp2_level=tp2_lvl, sl_level=sl_lvl,
                tp1_partial=tp1_partial,
                min_confidence=min_conf, min_body_pct=min_body_pct,
            )
        else:
            wf = walk_forward_chunks(
                df_test_feat, bundle, n_chunks=4,
                tp_level=tp_lvl, sl_level=sl_lvl,
                min_confidence=min_conf, min_body_pct=min_body_pct,
            )
        st.dataframe(wf, use_container_width=True, hide_index=True)

        st.markdown("#### Trade Log")
        display_trades = trades.copy()
        display_trades["Confidence"] = (display_trades["Confidence"] * 100).round(1)
        display_trades["Body %"] = (display_trades["Body %"] * 100).round(3)
        display_trades["P/L %"] = display_trades["P/L %"].round(3)
        display_trades["Entry"] = display_trades["Entry"].round(2)
        st.dataframe(display_trades, use_container_width=True, hide_index=True)
    else:
        st.warning("Tidak ada trade dengan filter ini. Turunkan threshold.")


# ===========================================================================
# TAB 3: ABOUT
# ===========================================================================
with tab_about:
    st.subheader("Tentang Model")
    st.markdown("""
    **Pipeline:**
    1. Fetch ETHUSDT 1h spot dari Binance (mirror `data-api.binance.vision`)
    2. Compute 45+ features: bar geometry, fib classification, SQZMOM, EMA/MACD/RSI/ADX,
       volume, multi-bar context, time-of-day cyclic encoding, volatility regime
    3. Track outcome 48h ke depan untuk 6 fib levels (1.61/2.5/3.6 × Up/Down)
    4. Train 6 XGBoost classifiers (satu per target) + isotonic calibration
    5. Time-based split: 70% train / 15% calibration / 15% test (NO shuffling, no future leak)

    **Model:**
    - XGBoost (n_estimators=400, max_depth=5, lr=0.05)
    - Isotonic regression calibration → "65% confidence" = 65% empirical winrate (kira-kira)
    - 32 input features, 6 binary outputs

    **Yang TIDAK dimodelkan (caveats):**
    - **Slippage** (real ~0.05-0.15% per trade)
    - **Execution lag** (entry di next bar's open, bukan current close)
    - **Funding rate** (futures only — kalau pakai perp)
    - **News-driven volatility spikes** (unpredictable)
    - **Liquidity issues di altcoin** (kalau extend ke ETH/USDT, ini OK)

    **Cara menggunakan (untuk operator signal service):**
    1. Pick strategy preset → Conservative direkomendasikan untuk subscriber baru
    2. Tiap jam (bar close), check apakah "TRADE" muncul
    3. Kalau TRADE: kirim signal ke subscriber dengan entry, TP, SL price
    4. Track realized vs backtest performance — jika drift > 10%, pause & re-investigate
    5. **Be radically honest** soal performance live vs backtest. Subscriber yang stay = subscriber yang trust Anda.

    **Realistic profit expectation (annualized):**
    - Conservative: +50-120% / yr (winrate 50-55%)
    - Balanced: +20-60% / yr (winrate 38-45%)
    - Aggressive: +80-200% / yr tapi dengan deep drawdown periods

    Annualization SELALU optimistic. Real-world subtract 30-50% dari backtest expected.

    **Untuk subscriber Anda:**
    > "Strategy ini punya backtest winrate X%. Saya operate live dengan disiplin sama,
    > tapi 1 bulan lalu bukan jaminan 1 bulan depan. Trade hanya kalau Anda OK dengan
    > kemungkinan kehilangan 100% modal slot trade ini."

    Itu pitch yang sustainable. Hindari "guaranteed profit", "auto-rich", dll.
    """)
