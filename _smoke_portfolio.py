"""End-to-end smoke test: portfolio aggregation + confluence detection."""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from backtest_engine import (prepare_features, run_backtest_multi_tp,
                              portfolio_backtest, detect_confluence,
                              predict_probs)

HERE = Path(__file__).parent
TICKERS = ["LINKUSDT", "SUIUSDT", "AAVEUSDT", "ARBUSDT", "SOLUSDT"]
STRAT = {"tp": 2.5, "sl": 2.5, "min_conf": 0.65, "min_body_pct": 0.008}


def load_test(ticker):
    df = pd.read_excel(HERE / "data" / f"dataset_{ticker}_18m.xlsx")
    df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"])
    df = df[(df["Lookahead Bars"] == 48) & df["SQZMOM Value"].notna()].reset_index(drop=True)
    df = df.dropna(subset=["Body% MA5", "Vol/MA20", "ATR Pct100"]).reset_index(drop=True)
    bundle = joblib.load(HERE / "models" / f"xgb_{ticker}.pkl")
    i2 = int(len(df) * 0.85)
    return prepare_features(df.iloc[i2:].reset_index(drop=True)), bundle


# Portfolio backtest
print("=== PORTFOLIO BACKTEST (5 tickers, equal-weight, multi-TP) ===")
per_trades = {}
per_stats = {}
for tk in TICKERS:
    df_test, bundle = load_test(tk)
    trades, eq, stats = run_backtest_multi_tp(
        df_test, bundle, tp1_level=2.5, tp2_level=3.6, sl_level=2.5,
        tp1_partial=0.5, min_confidence=STRAT["min_conf"], min_body_pct=STRAT["min_body_pct"],
    )
    per_trades[tk] = trades
    per_stats[tk] = stats
    print(f"  {tk}: {stats['trades']} trades, WR {stats['winrate_pct']:.1f}%, "
          f"Ret {stats['total_return_pct']:+.1f}%, DD {stats['max_drawdown_pct']:.1f}%")

combined, port_eq, port_stats = portfolio_backtest(per_trades, equal_weight=True)
print(f"\nPortfolio result:")
print(f"  Trades: {port_stats['trades']}")
print(f"  Winrate: {port_stats['winrate_pct']:.1f}%")
print(f"  Total Return: {port_stats['total_return_pct']:+.2f}%")
print(f"  Max DD: {port_stats['max_drawdown_pct']:+.2f}%")

# Diversification benefit
avg_solo_ret = sum(per_stats[t]["total_return_pct"] for t in TICKERS) / len(TICKERS)
avg_solo_dd = sum(per_stats[t]["max_drawdown_pct"] for t in TICKERS) / len(TICKERS)
print(f"\nDiversification benefit:")
print(f"  Avg solo return: {avg_solo_ret:+.2f}%")
print(f"  Avg solo DD: {avg_solo_dd:+.2f}%")
print(f"  Portfolio DD vs avg solo DD: {port_stats['max_drawdown_pct'] - avg_solo_dd:+.2f}%")


# Historical confluence detection
print("\n=== HISTORICAL CONFLUENCE EVENTS (test set, all 5 tickers) ===")
ticker_signals = {}
for tk in TICKERS:
    df_test, bundle = load_test(tk)
    X = df_test[bundle["feature_cols"]].fillna(0).values
    probs = predict_probs(bundle, X)
    p_up = np.maximum(probs[:, 0], probs[:, 2])
    p_dn = np.maximum(probs[:, 1], probs[:, 3])
    conf = np.maximum(p_up, p_dn)
    direction = np.where(p_up > p_dn, "LONG", "SHORT")
    tradable = (conf >= STRAT["min_conf"]) & (df_test["body_abs_pct"] >= STRAT["min_body_pct"])
    rows = []
    for j in range(len(df_test)):
        if tradable.iloc[j]:
            rows.append({
                "Datetime (UTC)": df_test["Datetime (UTC)"].iloc[j],
                "Direction": direction[j],
                "Confidence": float(conf[j]),
                "tradable": True,
            })
    ticker_signals[tk] = pd.DataFrame(rows)
    print(f"  {tk}: {len(rows)} tradable signals")

events = detect_confluence(ticker_signals, min_count=3)
print(f"\nConfluence events (3+ same direction same hour): {len(events)}")
if len(events) > 0:
    print(events.head(10).to_string(index=False))
    print(f"\nLONG confluences: {(events['Direction']=='LONG').sum()}")
    print(f"SHORT confluences: {(events['Direction']=='SHORT').sum()}")

print("\nSmoke test PASSED.")
