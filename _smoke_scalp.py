"""Smoke test Scalp mode — verify signal count across 18 tickers (latest bar)."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backtest_engine import predict_probs

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
MODEL_DIR = HERE / "models"

scalp = json.loads((MODEL_DIR / "scalp_params.json").read_text())
print(f"Loaded scalp_params: {len(scalp)} tickers\n")


def make_feat(row):
    atr_safe = row["ATR 14"] if pd.notna(row["ATR 14"]) and row["ATR 14"] > 0 else 1
    return {
        "Bar Color Code": {"Green": 1, "Red": -1, "Doji": 0}.get(row["Bar Color"], 0),
        "Fib Zone": row["Fib Zone"] if pd.notna(row["Fib Zone"]) else 0,
        "Fib Position": row["Fib Position"] if pd.notna(row["Fib Position"]) else 0,
        "Body %": row["Body %"], "Upper Wick %": row["Upper Wick %"], "Lower Wick %": row["Lower Wick %"],
        "Range": row["Range"], "Range/ATR": row["Range/ATR"] if pd.notna(row["Range/ATR"]) else 0,
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
        "BTC Ret 1h": row.get("BTC Ret 1h", 0) if pd.notna(row.get("BTC Ret 1h", 0)) else 0,
        "BTC Ret 4h": row.get("BTC Ret 4h", 0) if pd.notna(row.get("BTC Ret 4h", 0)) else 0,
        "BTC Ret 24h": row.get("BTC Ret 24h", 0) if pd.notna(row.get("BTC Ret 24h", 0)) else 0,
        "BTC Body Dir": row.get("BTC Body Dir", 0) if pd.notna(row.get("BTC Body Dir", 0)) else 0,
        "BTC Range/ATR": row.get("BTC Range/ATR", 1) if pd.notna(row.get("BTC Range/ATR", 1)) else 1,
        "BTC ATR Pct100": row.get("BTC ATR Pct100", 0.5) if pd.notna(row.get("BTC ATR Pct100", 0.5)) else 0.5,
        "SQZMOM Value": row["SQZMOM Value"],
        "Mom Color Code": {"lime": 2, "green": 1, "maroon": -1, "red": -2}.get(row["Momentum Color"], 0),
        "Squeeze Code": {"Squeeze ON (black)": 2, "Squeeze OFF (gray)": 1, "No Squeeze (blue)": 0}.get(row["Squeeze Status"], 0),
        "RSI 14": row["RSI 14"] if pd.notna(row["RSI 14"]) else 50,
        "ADX 14": row["ADX 14"] if pd.notna(row["ADX 14"]) else 20,
        "MACD Hist": (row["MACD"] - row["MACD Signal"]) if pd.notna(row["MACD"]) and pd.notna(row["MACD Signal"]) else 0,
        "HTF 4H Trend": row["HTF 4H Trend"],
        "Posisi Code": {"LONG": 1, "SHORT": -1, "NO TRADE": 0}.get(row["Raw Posisi"], 0),
        "Last TR / ATR": row["Last TR"] / atr_safe if pd.notna(row["Last TR"]) else 0,
    }


print(f"{'Ticker':<10} {'TP1':<5} {'TP2':<5} {'SL':<5} {'MinConf':<8} {'MinBody':<8}  {'Verdict':<14} {'Conf%':<7}")
print("-" * 85)
n_trade = 0
for tk in sorted(scalp.keys()):
    cfg = scalp[tk]
    df = pd.read_excel(DATA_DIR / f"dataset_{tk}_18m.xlsx")
    df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"])
    bundle = joblib.load(MODEL_DIR / f"xgb_{tk}.pkl")
    row = df.iloc[-1]

    o, c = row["Open"], row["Close"]
    body_pct = abs(c - o) / c
    feat = make_feat(row)
    X = np.array([[feat[col] for col in bundle["feature_cols"]]])
    probs = predict_probs(bundle, X)[0]
    pd_ = dict(zip(bundle["target_cols"], probs))
    p_up = max(pd_["Fib 1.61 Up"], pd_["Fib 2.5 Up"])
    p_dn = max(pd_["Fib 1.61 Down"], pd_["Fib 2.5 Down"])
    conf = max(p_up, p_dn)
    direction = "LONG" if p_up > p_dn else "SHORT"
    tradable = (conf >= cfg["min_conf"]) and (body_pct >= cfg["min_body_pct"])
    if tradable:
        n_trade += 1
    verdict = f"TRADE {direction}" if tradable else "NO"
    print(f"{tk:<10} {cfg['tp1']:<5} {cfg['tp2']:<5} {cfg['sl']:<5} "
          f"{cfg['min_conf']*100:>4.0f}%   {cfg['min_body_pct']*100:>5.2f}%   "
          f"{verdict:<14} {conf*100:>5.1f}%")

print(f"\nTotal TRADE: {n_trade}/{len(scalp)} tickers")
print(f"Avg trades/ticker/day (backtest): {sum(scalp[t]['metrics']['trades'] for t in scalp)/len(scalp)/82:.1f}")
total_daily = sum(scalp[t]['metrics']['trades'] for t in scalp) / 82
print(f"Total daily signal frequency across all tickers: {total_daily:.0f}")
print("\nSmoke test Scalp PASSED.")
