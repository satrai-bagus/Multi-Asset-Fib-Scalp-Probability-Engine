"""
refresh_all.py — Full pipeline refresh for production.

What it does (in order):
  1. Refresh datasets untuk semua active ticker (re-fetch 18mo dari Binance)
  2. Retrain semua model (XGBoost + isotonic calibration)
  3. Re-run optimize_per_ticker untuk Balanced + Scalp modes
  4. Print summary table dengan AUC + signal count change

Schedule recommendation:
  - Daily run dataset refresh ONLY: --data-only       (~15 min)
  - Weekly run full pipeline:        no flags         (~30-45 min)
  - Monthly run with optimization:   --optimize       (~50 min)

Usage:
  python refresh_all.py                  # full: data + train (default weekly)
  python refresh_all.py --data-only      # daily: just refresh data
  python refresh_all.py --optimize       # monthly: + re-optimize params
  python refresh_all.py --skip TICKER1 TICKER2   # exclude specific tickers
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
MODEL_DIR = HERE / "models"
PARENT = HERE.parent
SQZMOM = PARENT / "sqzmom_export.py"


def discover_tickers():
    return sorted([p.stem.replace("xgb_", "") for p in MODEL_DIR.glob("xgb_*USDT.pkl")])


def fetch_data(tickers, end_date):
    """Re-fetch dataset for each ticker (18 months back from end_date)."""
    print(f"\n{'='*65}\n  STEP 1: DATASET REFRESH ({len(tickers)} tickers)\n{'='*65}")
    start_date = (datetime.fromisoformat(end_date) - __import__("datetime").timedelta(days=550)).strftime("%Y-%m-%d")
    results = {}
    for i, tk in enumerate(tickers, 1):
        t0 = time.time()
        print(f"[{i}/{len(tickers)}] {tk}...", end=" ", flush=True)
        ret = subprocess.run([
            sys.executable, str(SQZMOM),
            "--symbol", tk,
            "--start", start_date,
            "--end", end_date,
            "--out", str(HERE / "data" / f"dataset_{tk}_18m.xlsx"),
        ], cwd=str(PARENT), capture_output=True, text=True, encoding="utf-8", errors="replace")
        dt = time.time() - t0
        results[tk] = ("OK", dt) if ret.returncode == 0 else ("FAIL", dt)
        print(f"{'OK' if ret.returncode == 0 else 'FAIL'} ({dt:.0f}s)")
    return results


def retrain_models(tickers, no_btc_set):
    """Retrain each ticker model."""
    print(f"\n{'='*65}\n  STEP 2: MODEL RETRAIN ({len(tickers)} tickers)\n{'='*65}")
    results = {}
    for i, tk in enumerate(tickers, 1):
        t0 = time.time()
        print(f"[{i}/{len(tickers)}] {tk}...", end=" ", flush=True)
        args = [sys.executable, str(HERE / "train_model.py"), "--symbol", tk]
        if tk in no_btc_set:
            args.append("--no-btc")
        ret = subprocess.run(args, cwd=str(HERE), capture_output=True, text=True, encoding="utf-8", errors="replace")
        dt = time.time() - t0

        # Extract AUC summary
        auc = "?"
        if ret.returncode == 0:
            for line in ret.stdout.split("\n"):
                if "Fib 1.61 Up" in line and "AUC=" in line:
                    auc = line.split("AUC=")[1][:5]
                    break
        results[tk] = (auc, dt) if ret.returncode == 0 else ("FAIL", dt)
        print(f"AUC={auc} ({dt:.0f}s)" if ret.returncode == 0 else f"FAIL ({dt:.0f}s)")
    return results


def optimize_modes():
    """Re-run optimize_per_ticker for both modes."""
    print(f"\n{'='*65}\n  STEP 3: RE-OPTIMIZE per-ticker params\n{'='*65}")
    for mode in ["balanced", "scalp"]:
        print(f"\n[{mode.upper()}]...")
        t0 = time.time()
        ret = subprocess.run(
            [sys.executable, str(HERE / "optimize_per_ticker.py"), "--mode", mode],
            cwd=str(HERE), capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        dt = time.time() - t0
        if ret.returncode == 0:
            print(f"  OK ({dt:.0f}s)")
        else:
            print(f"  FAIL ({dt:.0f}s)\n  {ret.stderr[-300:]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-only", action="store_true", help="Only refresh datasets, skip training")
    parser.add_argument("--optimize", action="store_true", help="Also re-run optimize_per_ticker (monthly)")
    parser.add_argument("--skip", nargs="*", default=[], help="Tickers to skip")
    parser.add_argument("--end-date", default=datetime.now().strftime("%Y-%m-%d"),
                        help="Dataset end date (default: today)")
    args = parser.parse_args()

    # Discover active tickers (drop --skip)
    tickers = [t for t in discover_tickers() if t not in args.skip]
    print(f"refresh_all.py — {len(tickers)} active tickers")
    print(f"End date: {args.end_date}")
    print(f"Skip: {args.skip if args.skip else 'none'}")
    print(f"Mode: {'data-only' if args.data_only else 'full retrain'}{' + optimize' if args.optimize else ''}")

    # Identify no-BTC tickers (their bundles store fewer features)
    no_btc_set = set()
    import joblib
    for tk in tickers:
        try:
            b = joblib.load(MODEL_DIR / f"xgb_{tk}.pkl")
            if "BTC Ret 1h" not in b["feature_cols"]:
                no_btc_set.add(tk)
        except Exception:
            pass
    if no_btc_set:
        print(f"No-BTC config (preserve): {sorted(no_btc_set)}")

    t_start = time.time()

    # Step 1: Always refresh data
    data_results = fetch_data(tickers, args.end_date)
    failed_data = [t for t, (s, _) in data_results.items() if s != "OK"]

    if args.data_only:
        print(f"\n{'='*65}\nDATA-ONLY refresh complete in {time.time()-t_start:.0f}s")
        if failed_data:
            print(f"FAILED: {failed_data}")
        return

    # Step 2: Retrain (skip failed-data tickers)
    train_tickers = [t for t in tickers if t not in failed_data]
    train_results = retrain_models(train_tickers, no_btc_set)

    # Step 3: Optimize (if --optimize)
    if args.optimize:
        optimize_modes()

    # Summary
    print(f"\n{'='*65}\n  SUMMARY  ({time.time()-t_start:.0f}s total)\n{'='*65}")
    print(f"{'Ticker':<10} {'Data':<8} {'Model AUC':<12}")
    for tk in tickers:
        d = data_results.get(tk, ("?", 0))[0]
        m = train_results.get(tk, ("-", 0))[0] if not args.data_only else "-"
        print(f"  {tk:<10} {d:<8} {m:<12}")

    failed = [t for t, (s, _) in train_results.items() if s == "FAIL"]
    if failed:
        print(f"\n⚠️  FAILED training: {failed}")
    else:
        print(f"\n✓ All {len(train_tickers)} models trained successfully")

    print(f"\nNext step: streamlit run app.py  (signals auto-reflect new models)")


if __name__ == "__main__":
    main()
