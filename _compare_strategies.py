"""Compare 3 strategies (Conservative / Balanced / Aggressive) across all tickers."""
import joblib
import pandas as pd
from pathlib import Path

from backtest_engine import prepare_features, run_backtest

HERE = Path(__file__).parent
TICKERS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
           "AVAXUSDT", "LINKUSDT", "POLUSDT",
           "NEARUSDT", "SUIUSDT", "ARBUSDT", "APTUSDT", "AAVEUSDT", "FETUSDT"]

STRATEGIES = [
    ("Conservative",  2.5,  2.5,  0.65, 0.008),
    ("Balanced",      3.6,  2.5,  0.60, 0.005),
    ("Aggressive",    3.6,  1.61, 0.70, 0.005),
]

print(f"{'Ticker':<10} {'Strategy':<13} {'Trades':>6} {'WinRt%':>7} {'AvgPnL%':>8} {'TotalRet%':>10} {'MaxDD%':>8}")
print("-" * 80)

for ticker in TICKERS:
    df_path = HERE / "data" / f"dataset_{ticker}_18m.xlsx"
    pkl_path = HERE / "models" / f"xgb_{ticker}.pkl"
    if not df_path.exists() or not pkl_path.exists():
        print(f"{ticker} — files missing, skipping")
        continue

    df = pd.read_excel(df_path)
    df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"])
    df = df[(df["Lookahead Bars"] == 48) & df["SQZMOM Value"].notna()].reset_index(drop=True)
    df = df.dropna(subset=["Body% MA5", "Vol/MA20", "ATR Pct100"]).reset_index(drop=True)
    bundle = joblib.load(pkl_path)
    i2 = int(len(df) * 0.85)
    df_test = prepare_features(df.iloc[i2:].reset_index(drop=True))

    for name, tp, sl, conf, body in STRATEGIES:
        _, _, stats = run_backtest(df_test, bundle,
                                    tp_level=tp, sl_level=sl,
                                    min_confidence=conf, min_body_pct=body)
        if not stats:
            print(f"{ticker:<10} {name:<13} (no trades)")
            continue
        print(f"{ticker:<10} {name:<13} {stats['trades']:>6} "
              f"{stats['winrate_pct']:>6.1f}% {stats['avg_pl_pct']:>+7.3f}% "
              f"{stats['total_return_pct']:>+9.2f}% {stats['max_drawdown_pct']:>+7.2f}%")
    print()
