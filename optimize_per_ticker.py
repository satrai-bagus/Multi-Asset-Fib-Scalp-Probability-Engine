"""
Per-Ticker Optimal Parameter Auto-Finder.

Two modes:
  --mode balanced (default) — Calmar-optimized, swing-style, lower frequency
  --mode scalp             — futures-tuned, MANY signals, tight TP/SL, low fees

Output:
  balanced -> models/optimal_params.json
  scalp    -> models/scalp_params.json

Run once after model retraining. Recompute monthly.
"""
import argparse
import json
import math
import time
from pathlib import Path

import joblib
import pandas as pd

from backtest_engine import prepare_features, run_backtest_multi_tp

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
MODEL_DIR = HERE / "models"


# ---------------------------------------------------------------------------
# Mode configurations
# ---------------------------------------------------------------------------
MODES = {
    "balanced": {
        "output_file": "optimal_params.json",
        "fee_pct_roundtrip": 0.002,         # spot trading (0.1% × 2)
        "grid": {
            "tp1":       [1.61, 2.5],
            "tp2":       [2.5, 3.6],
            "sl":        [1.61, 2.5, 3.6],
            "min_conf":  [0.55, 0.60, 0.65, 0.70, 0.75],
            "min_body":  [0.003, 0.005, 0.008, 0.012, 0.016],
        },
        "min_trades": 30,
        "max_dd_tolerance": -60,
        "min_winrate_pct": 0,
        "ranking_metric": "calmar",          # return / |drawdown|
    },
    "scalp": {
        # FUTURES-TUNED: high frequency, tight risk, perp fees
        "output_file": "scalp_params.json",
        "fee_pct_roundtrip": 0.0008,         # Binance futures perp ~= 0.04% × 2
        "grid": {
            "tp1":       [1.27, 1.61],         # closer targets, quicker hits
            "tp2":       [2.5, 3.6],
            "sl":        [1.27, 1.61, 2.5],    # tight stop OK with low fees
            "min_conf":  [0.50, 0.55, 0.60],   # permissive — MORE signals
            "min_body":  [0.001, 0.002, 0.003, 0.005],  # any meaningful bar
        },
        "min_trades": 200,                   # require HIGH signal frequency
        "max_dd_tolerance": -45,             # tighter — futures leverage kills
        "min_winrate_pct": 48,               # "often correct" requirement
        "ranking_metric": "freq_score",      # total_return × log10(trades)
    },
}


def calmar_like(stats):
    dd = abs(stats["max_drawdown_pct"])
    return stats["total_return_pct"] / max(dd, 1)


def freq_score(stats):
    """Reward both high trade count AND positive return."""
    if stats["total_return_pct"] <= 0:
        return -1e9
    return stats["total_return_pct"] * math.log10(max(stats["trades"], 10))


SCORERS = {
    "calmar":     calmar_like,
    "freq_score": freq_score,
}


def optimize_ticker(ticker, mode_cfg):
    """Grid search for one ticker. Returns dict with best config + metrics."""
    df_path = DATA_DIR / f"dataset_{ticker}_18m.xlsx"
    pkl_path = MODEL_DIR / f"xgb_{ticker}.pkl"
    if not df_path.exists() or not pkl_path.exists():
        return None

    df = pd.read_excel(df_path)
    df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"])
    df = df[(df["Lookahead Bars"] == 48) & df["SQZMOM Value"].notna()].reset_index(drop=True)
    df = df.dropna(subset=["Body% MA5", "Vol/MA20", "ATR Pct100"]).reset_index(drop=True)
    bundle = joblib.load(pkl_path)
    i2 = int(len(df) * 0.85)
    df_test = prepare_features(df.iloc[i2:].reset_index(drop=True))

    grid = mode_cfg["grid"]
    fee = mode_cfg["fee_pct_roundtrip"]
    min_trades = mode_cfg["min_trades"]
    max_dd_tolerance = mode_cfg["max_dd_tolerance"]
    min_winrate = mode_cfg["min_winrate_pct"]
    scorer = SCORERS[mode_cfg["ranking_metric"]]

    results = []
    n_combos = 0
    for tp1 in grid["tp1"]:
        for tp2 in grid["tp2"]:
            if tp2 <= tp1:
                continue
            for sl in grid["sl"]:
                for min_conf in grid["min_conf"]:
                    for min_body in grid["min_body"]:
                        n_combos += 1
                        _, _, s = run_backtest_multi_tp(
                            df_test, bundle,
                            tp1_level=tp1, tp2_level=tp2, sl_level=sl,
                            tp1_partial=0.5,
                            min_confidence=min_conf, min_body_pct=min_body,
                            fee_pct_roundtrip=fee,
                        )
                        if not s or s["trades"] < min_trades:
                            continue
                        if s["max_drawdown_pct"] < max_dd_tolerance:
                            continue
                        if s["winrate_pct"] < min_winrate:
                            continue
                        results.append({
                            "tp1": tp1, "tp2": tp2, "sl": sl,
                            "min_conf": min_conf, "min_body_pct": min_body,
                            "trades": s["trades"], "winrate_pct": s["winrate_pct"],
                            "total_return_pct": s["total_return_pct"],
                            "max_drawdown_pct": s["max_drawdown_pct"],
                            "avg_pl_pct": s["avg_pl_pct"],
                            "calmar": calmar_like(s),
                            "score": scorer(s),
                        })

    if not results:
        return {"error": "no profitable config found", "n_combos_tested": n_combos}

    results.sort(key=lambda r: -r["score"])
    best = results[0]
    runner_up = results[1] if len(results) > 1 else None

    return {
        "tp1": best["tp1"], "tp2": best["tp2"], "sl": best["sl"],
        "min_conf": best["min_conf"], "min_body_pct": best["min_body_pct"],
        "metrics": {
            "trades": best["trades"],
            "winrate_pct": round(best["winrate_pct"], 1),
            "total_return_pct": round(best["total_return_pct"], 2),
            "max_drawdown_pct": round(best["max_drawdown_pct"], 2),
            "avg_pl_pct": round(best["avg_pl_pct"], 3),
            "calmar": round(best["calmar"], 2),
            "score": round(best["score"], 2),
        },
        "runner_up": {
            "tp1": runner_up["tp1"], "tp2": runner_up["tp2"], "sl": runner_up["sl"],
            "min_conf": runner_up["min_conf"], "min_body_pct": runner_up["min_body_pct"],
            "total_return_pct": round(runner_up["total_return_pct"], 2),
            "score": round(runner_up["score"], 2),
        } if runner_up else None,
        "n_combos_tested": n_combos,
        "n_valid_configs": len(results),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=list(MODES.keys()), default="balanced",
                        help="Optimization mode. 'balanced' = swing/spot. 'scalp' = futures/high-freq.")
    args = parser.parse_args()

    mode_cfg = MODES[args.mode]
    output_path = MODEL_DIR / mode_cfg["output_file"]
    print(f"=== MODE: {args.mode.upper()} ===")
    print(f"Fee: {mode_cfg['fee_pct_roundtrip']*100:.3f}% round-trip, "
          f"Min trades: {mode_cfg['min_trades']}, "
          f"Min winrate: {mode_cfg['min_winrate_pct']}%, "
          f"Max DD: {mode_cfg['max_dd_tolerance']}%, "
          f"Scorer: {mode_cfg['ranking_metric']}")
    print(f"Output: {output_path.name}\n")

    tickers = sorted([p.stem.replace("xgb_", "") for p in MODEL_DIR.glob("xgb_*USDT.pkl")])
    print(f"Optimizing {len(tickers)} tickers...")

    all_results = {}
    t_start = time.time()
    for i, tk in enumerate(tickers, 1):
        t0 = time.time()
        print(f"\n[{i}/{len(tickers)}] {tk}...", end=" ", flush=True)
        result = optimize_ticker(tk, mode_cfg)
        dt = time.time() - t0
        all_results[tk] = result

        if not result:
            print(f"FAILED (no model)")
            continue
        if "error" in result:
            print(f"NO CONFIG ({result['error']})")
            continue

        m = result["metrics"]
        print(f"done in {dt:.1f}s")
        print(f"  Best: TP1={result['tp1']}, TP2={result['tp2']}, SL={result['sl']}, "
              f"conf>={result['min_conf']*100:.0f}%, body>={result['min_body_pct']*100:.2f}%")
        print(f"  -> {m['trades']} trades, WR {m['winrate_pct']:.1f}%, "
              f"Ret {m['total_return_pct']:+.2f}%, DD {m['max_drawdown_pct']:+.2f}%, "
              f"Calmar {m['calmar']:.1f}, Score {m['score']:.1f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"DONE in {time.time()-t_start:.0f}s. Saved to {output_path}")
    print(f"\nSummary table:")
    print(f"{'Ticker':<10} {'TP1':<5} {'TP2':<5} {'SL':<5} {'Conf':<6} {'Body':<6}  {'Trades':<7} {'WR':<7} {'Ret%':<10} {'DD%':<8} {'Calmar':<7}")
    for tk, r in all_results.items():
        if r and "metrics" in r:
            m = r["metrics"]
            print(f"{tk:<10} {r['tp1']:<5} {r['tp2']:<5} {r['sl']:<5} {r['min_conf']*100:<5.0f}% {r['min_body_pct']*100:<5.2f}%  "
                  f"{m['trades']:<7} {m['winrate_pct']:<6.1f}% {m['total_return_pct']:<+9.2f}% {m['max_drawdown_pct']:<+7.2f}% {m['calmar']:<6.2f}")


if __name__ == "__main__":
    main()
