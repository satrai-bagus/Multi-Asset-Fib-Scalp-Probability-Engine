"""
Comprehensive Scalp Mode backtest report.

For each ticker, runs Multi-TP backtest dengan scalp_params.json config.
Reports:
  1. Per-ticker stats (trades, WR, return, DD, outcomes breakdown)
  2. Walk-forward 4 chunks (consistency check)
  3. Aggregate portfolio stats
"""
import json
from pathlib import Path

import joblib
import pandas as pd

from backtest_engine import prepare_features, run_backtest_multi_tp, walk_forward_chunks_multi_tp

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
MODEL_DIR = HERE / "models"

SCALP_PARAMS = json.loads((MODEL_DIR / "scalp_params.json").read_text())
SCALP_FEE = 0.0008  # futures perp 0.04% × 2


def load_test_df(ticker):
    df = pd.read_excel(DATA_DIR / f"dataset_{ticker}_18m.xlsx")
    df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"])
    df = df[(df["Lookahead Bars"] == 48) & df["SQZMOM Value"].notna()].reset_index(drop=True)
    df = df.dropna(subset=["Body% MA5", "Vol/MA20", "ATR Pct100"]).reset_index(drop=True)
    bundle = joblib.load(MODEL_DIR / f"xgb_{ticker}.pkl")
    i2 = int(len(df) * 0.85)
    return prepare_features(df.iloc[i2:].reset_index(drop=True)), bundle


# ===========================================================================
# 1. Per-ticker backtest
# ===========================================================================
print("=" * 105)
print("  SCALP MODE BACKTEST — Per-Ticker Stats (test set 82 days, futures fee 0.08% RT)")
print("=" * 105)
print(f"{'Ticker':<10} {'TP1':<5} {'TP2':<5} {'SL':<5} {'Conf':<5} {'Body':<5}  "
      f"{'Trades':<7} {'WR%':<5} {'Avg%':<7} {'TotalRet%':<10} {'MaxDD%':<8} {'Calmar':<6} {'Sharpe':<6}")
print("-" * 105)

per_ticker = []
total_trades = 0
total_return_sum = 0
test_period = None

for tk in sorted(SCALP_PARAMS.keys()):
    cfg = SCALP_PARAMS[tk]
    df_test, bundle = load_test_df(tk)
    if test_period is None:
        test_period = (df_test["Datetime (UTC)"].iloc[0], df_test["Datetime (UTC)"].iloc[-1])

    trades, equity, stats = run_backtest_multi_tp(
        df_test, bundle,
        tp1_level=cfg["tp1"], tp2_level=cfg["tp2"], sl_level=cfg["sl"],
        tp1_partial=0.5,
        min_confidence=cfg["min_conf"],
        min_body_pct=cfg["min_body_pct"],
        fee_pct_roundtrip=SCALP_FEE,
    )

    if not stats:
        print(f"{tk:<10} NO TRADES")
        continue

    calmar = stats["total_return_pct"] / max(abs(stats["max_drawdown_pct"]), 1)
    per_ticker.append({
        "ticker": tk, "stats": stats, "trades_df": trades,
        "tp1": cfg["tp1"], "tp2": cfg["tp2"], "sl": cfg["sl"],
        "conf": cfg["min_conf"], "body": cfg["min_body_pct"],
        "calmar": calmar,
    })
    total_trades += stats["trades"]
    total_return_sum += stats["total_return_pct"]

    print(f"{tk:<10} {cfg['tp1']:<5} {cfg['tp2']:<5} {cfg['sl']:<5} "
          f"{cfg['min_conf']*100:>3.0f}% {cfg['min_body_pct']*100:>4.2f}%  "
          f"{stats['trades']:<7} {stats['winrate_pct']:>4.1f} {stats['avg_pl_pct']:>+6.3f} "
          f"{stats['total_return_pct']:>+9.2f} {stats['max_drawdown_pct']:>+7.2f} "
          f"{calmar:>6.1f} {stats['sharpe_like']:>5.1f}")


# ===========================================================================
# 2. Outcome breakdown (top 5 tickers)
# ===========================================================================
print(f"\n{'='*105}")
print("  OUTCOME BREAKDOWN — Top 5 tickers by Calmar")
print(f"{'='*105}")
per_ticker.sort(key=lambda x: -x["calmar"])
top5 = per_ticker[:5]

for p in top5:
    s = p["stats"]
    oc = s.get("outcomes", {})
    total = s["trades"]
    print(f"\n  {p['ticker']}  (Calmar {p['calmar']:.1f}, Ret {s['total_return_pct']:+.0f}%, DD {s['max_drawdown_pct']:.1f}%)")
    sorted_oc = sorted(oc.items(), key=lambda x: -x[1])
    for outcome, n in sorted_oc:
        pct = n / total * 100
        bar = "#" * int(pct / 2)
        print(f"    {outcome:<22} {n:>5} ({pct:>5.1f}%)  {bar}")


# ===========================================================================
# 3. Walk-forward 4 chunks (consistency check on top 5)
# ===========================================================================
print(f"\n{'='*105}")
print("  WALK-FORWARD 4 CHUNKS — Top 5 tickers (consistency check)")
print(f"{'='*105}")
print(f"{'Ticker':<10} {'Chunk':<6} {'From':<18} {'To':<18} {'Trades':<7} {'WR%':<5} {'Ret%':<8} {'DD%':<7}")
print("-" * 90)

for p in top5:
    tk = p["ticker"]
    df_test, bundle = load_test_df(tk)
    cfg = SCALP_PARAMS[tk]
    wf = walk_forward_chunks_multi_tp(
        df_test, bundle, n_chunks=4,
        tp1_level=cfg["tp1"], tp2_level=cfg["tp2"], sl_level=cfg["sl"],
        tp1_partial=0.5,
        min_confidence=cfg["min_conf"],
        min_body_pct=cfg["min_body_pct"],
        fee_pct_roundtrip=SCALP_FEE,
    )
    for _, row in wf.iterrows():
        print(f"{tk:<10} {int(row['Chunk']):<6} {row['From'][:16]:<18} {row['To'][:16]:<18} "
              f"{int(row['Trades']):<7} {row['Winrate %']:>4.1f} {row['Total Return %']:>+7.2f} {row['Max DD %']:>+6.2f}")
    print()


# ===========================================================================
# 4. Aggregate portfolio stats
# ===========================================================================
print(f"{'='*105}")
print(f"  AGGREGATE PORTFOLIO (18 tickers, equal-weight, futures Scalp mode)")
print(f"{'='*105}")
print(f"  Test period: {test_period[0]}  to  {test_period[1]}")
print(f"  Total trades across all tickers: {total_trades:,}")
print(f"  Avg trades/ticker:               {total_trades/len(per_ticker):.0f}")
test_days = (test_period[1] - test_period[0]).total_seconds() / 86400
print(f"  Test period:                     {test_days:.0f} days")
print(f"  Estimated signals/day total:     {total_trades/test_days:.0f}")
print(f"  Estimated signals/day/ticker:    {total_trades/test_days/len(per_ticker):.1f}")
print()

# Aggregate winrate (trade-weighted)
total_wins = sum(p["stats"]["wins"] for p in per_ticker)
total_decided = sum(p["stats"]["wins"] + p["stats"]["losses"] for p in per_ticker)
agg_wr = total_wins / total_decided * 100 if total_decided else 0
print(f"  Trade-weighted winrate:          {agg_wr:.1f}%")

# Sum + median return
print(f"  Avg per-ticker return:           {total_return_sum/len(per_ticker):+.1f}%")
print(f"  Median per-ticker return:        {sorted([p['stats']['total_return_pct'] for p in per_ticker])[len(per_ticker)//2]:+.1f}%")

# DD distribution
dds = [p["stats"]["max_drawdown_pct"] for p in per_ticker]
print(f"  Best DD:   {max(dds):+.1f}%   ({per_ticker[dds.index(max(dds))]['ticker']})")
print(f"  Worst DD:  {min(dds):+.1f}%   ({per_ticker[dds.index(min(dds))]['ticker']})")
print(f"  Median DD: {sorted(dds)[len(dds)//2]:+.1f}%")

# Top 5 + Bottom 3 by Calmar
print(f"\n  TOP 5 by Calmar:")
for p in sorted(per_ticker, key=lambda x: -x['calmar'])[:5]:
    s = p["stats"]
    print(f"    {p['ticker']:<10} Calmar {p['calmar']:>5.1f}   Ret {s['total_return_pct']:>+7.1f}%   "
          f"DD {s['max_drawdown_pct']:>+5.1f}%   {s['trades']} trades")

print(f"\n  BOTTOM 3 by Calmar:")
for p in sorted(per_ticker, key=lambda x: x['calmar'])[:3]:
    s = p["stats"]
    print(f"    {p['ticker']:<10} Calmar {p['calmar']:>5.1f}   Ret {s['total_return_pct']:>+7.1f}%   "
          f"DD {s['max_drawdown_pct']:>+5.1f}%   {s['trades']} trades")

print()
