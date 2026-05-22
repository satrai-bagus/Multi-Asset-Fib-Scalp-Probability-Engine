"""
check_performance.py — Quick health summary for production monitoring.

Prints:
  1. Dataset freshness (last bar timestamp per ticker)
  2. Model AUC per ticker
  3. Backtest stats per ticker (latest test set)
  4. Active vs configured tickers (Optimal + Scalp)
  5. Sanity alerts (stale data, broken model, etc.)

Run weekly or before publishing signals to subscribers.

Usage:
  python check_performance.py
  python check_performance.py --mode scalp   # focus on scalp config
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
MODEL_DIR = HERE / "models"


def days_old(dt_str):
    """Days between dt_str and now (UTC)."""
    if pd.isna(dt_str):
        return None
    dt = pd.to_datetime(dt_str)
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")
    return (datetime.now(timezone.utc) - dt.to_pydatetime()).total_seconds() / 86400


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["balanced", "scalp", "both"], default="both")
    args = parser.parse_args()

    tickers = sorted([p.stem.replace("xgb_", "") for p in MODEL_DIR.glob("xgb_*USDT.pkl")])
    print(f"\n{'='*85}")
    print(f"  CHECK PERFORMANCE — {len(tickers)} active tickers")
    print(f"  Run at: {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S UTC}")
    print(f"{'='*85}\n")

    optimal = {}
    scalp = {}
    if (MODEL_DIR / "optimal_params.json").exists():
        optimal = json.loads((MODEL_DIR / "optimal_params.json").read_text())
    if (MODEL_DIR / "scalp_params.json").exists():
        scalp = json.loads((MODEL_DIR / "scalp_params.json").read_text())

    alerts = []
    rows = []

    for tk in tickers:
        ds_path = DATA_DIR / f"dataset_{tk}_18m.xlsx"
        pkl_path = MODEL_DIR / f"xgb_{tk}.pkl"

        # Dataset freshness
        if ds_path.exists():
            df = pd.read_excel(ds_path)
            df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"])
            last_bar = df["Datetime (UTC)"].max()
            stale_days = days_old(last_bar)
        else:
            last_bar = None
            stale_days = None
            alerts.append(f"  [{tk}] DATASET MISSING")

        # Model + AUC
        try:
            bundle = joblib.load(pkl_path)
            aucs = bundle.get("auc_scores", {})
            avg_auc = sum(aucs.values()) / len(aucs) if aucs else 0
            n_features = len(bundle["feature_cols"])
            has_btc = "BTC Ret 1h" in bundle["feature_cols"]
            test_period = bundle.get("test_period", ("?", "?"))
        except Exception as e:
            avg_auc = 0
            n_features = 0
            has_btc = False
            test_period = ("?", "?")
            alerts.append(f"  [{tk}] MODEL LOAD FAILED: {e}")

        # Per-ticker config availability
        opt = optimal.get(tk, {})
        sca = scalp.get(tk, {})

        opt_ret = opt.get("metrics", {}).get("total_return_pct") if opt and "metrics" in opt else None
        sca_ret = sca.get("metrics", {}).get("total_return_pct") if sca and "metrics" in sca else None
        sca_trades = sca.get("metrics", {}).get("trades") if sca and "metrics" in sca else None

        rows.append({
            "Ticker": tk,
            "Data Last": str(last_bar)[:16] if last_bar is not None else "MISSING",
            "Stale (d)": f"{stale_days:.1f}" if stale_days is not None else "?",
            "Features": f"{n_features}" + (" +BTC" if has_btc else " noBTC"),
            "Avg AUC": f"{avg_auc:.3f}" if avg_auc else "?",
            "Opt Ret%": f"{opt_ret:+.0f}" if opt_ret is not None else "-",
            "Scalp Ret%": f"{sca_ret:+.0f}" if sca_ret is not None else "-",
            "Scalp Trades": str(sca_trades) if sca_trades is not None else "-",
        })

        # Alerts
        if stale_days is not None and stale_days > 7:
            alerts.append(f"  [{tk}] DATA STALE ({stale_days:.0f} days old) — run refresh_all.py")
        if avg_auc and avg_auc < 0.6:
            alerts.append(f"  [{tk}] LOW AUC ({avg_auc:.3f}) — model may be broken")

    df_summary = pd.DataFrame(rows)
    print(df_summary.to_string(index=False))

    # Aggregate stats
    print(f"\n{'='*85}")
    print(f"  AGGREGATE")
    print(f"{'='*85}")
    aucs = [float(r["Avg AUC"]) for r in rows if r["Avg AUC"] != "?"]
    if aucs:
        print(f"  Avg AUC across all tickers: {sum(aucs)/len(aucs):.3f}")
    stale = [float(r["Stale (d)"]) for r in rows if r["Stale (d)"] != "?"]
    if stale:
        print(f"  Avg data age: {sum(stale)/len(stale):.1f} days")
        print(f"  Oldest data: {max(stale):.1f} days ({rows[stale.index(max(stale))]['Ticker']})")

    scalp_trades = [int(r["Scalp Trades"]) for r in rows if r["Scalp Trades"] != "-"]
    if scalp_trades:
        print(f"\n  [SCALP MODE (backtest test set)]")
        print(f"     Total trades across {len(scalp_trades)} tickers: {sum(scalp_trades):,}")
        print(f"     Avg per ticker: {sum(scalp_trades)/len(scalp_trades):.0f}")
        print(f"     Per day estimate (82-day test): {sum(scalp_trades)/82:.0f} signals/day")

    opt_returns = [float(r["Opt Ret%"]) for r in rows if r["Opt Ret%"] != "-"]
    if opt_returns:
        print(f"\n  [OPTIMAL MODE returns]")
        print(f"     Best: {max(opt_returns):+.0f}% ({rows[opt_returns.index(max(opt_returns))]['Ticker']})")
        print(f"     Median: {sorted(opt_returns)[len(opt_returns)//2]:+.0f}%")
        print(f"     Worst: {min(opt_returns):+.0f}% ({rows[opt_returns.index(min(opt_returns))]['Ticker']})")

    # Alerts
    print(f"\n{'='*85}")
    print(f"  ALERTS ({len(alerts)})")
    print(f"{'='*85}")
    if alerts:
        for a in alerts:
            print(a)
    else:
        print("  [OK] No issues detected")

    # Recommendation
    print(f"\n{'='*85}")
    print(f"  RECOMMENDED ACTIONS")
    print(f"{'='*85}")
    if stale and max(stale) > 1:
        print(f"  - Run: python refresh_all.py --data-only   (refresh datasets, ~15 min)")
    if stale and max(stale) > 7:
        print(f"  - Run: python refresh_all.py              (full retrain, ~30 min)")
    if stale and max(stale) > 30:
        print(f"  - Run: python refresh_all.py --optimize   (full + re-optimize, ~50 min)")
    if not alerts and stale and max(stale) <= 1:
        print(f"  [OK] System healthy. No action needed.")

    print()


if __name__ == "__main__":
    main()
