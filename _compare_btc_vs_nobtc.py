"""Compare with-BTC vs no-BTC models for ARB / FET / DOGE."""
import joblib
import pandas as pd
from pathlib import Path

from backtest_engine import prepare_features, run_backtest

HERE = Path(__file__).parent
TICKERS = ["ARBUSDT", "FETUSDT", "DOGEUSDT"]

STRATEGIES = [
    ("Conservative",  2.5,  2.5,  0.65, 0.008),
    ("Balanced",      3.6,  2.5,  0.60, 0.005),
    ("Aggressive",    3.6,  1.61, 0.70, 0.005),
]

print(f"{'Ticker':<10} {'Variant':<10} {'Strategy':<13} {'Trades':>6} {'WinRt%':>7} {'TotalRet%':>10} {'MaxDD%':>8}")
print("-" * 80)

for ticker in TICKERS:
    df_path = HERE / "data" / f"dataset_{ticker}_18m.xlsx"
    df = pd.read_excel(df_path)
    df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"])
    df = df[(df["Lookahead Bars"] == 48) & df["SQZMOM Value"].notna()].reset_index(drop=True)
    df = df.dropna(subset=["Body% MA5", "Vol/MA20", "ATR Pct100"]).reset_index(drop=True)

    for variant, suffix in [("with-BTC", ""), ("no-BTC", "_nobtc")]:
        pkl_path = HERE / "models" / f"xgb_{ticker}{suffix}.pkl"
        if not pkl_path.exists():
            print(f"{ticker:<10} {variant:<10} (model missing)")
            continue
        bundle = joblib.load(pkl_path)
        i2 = int(len(df) * 0.85)
        df_test = prepare_features(df.iloc[i2:].reset_index(drop=True))

        for name, tp, sl, conf, body in STRATEGIES:
            _, _, stats = run_backtest(df_test, bundle,
                                        tp_level=tp, sl_level=sl,
                                        min_confidence=conf, min_body_pct=body)
            if not stats:
                print(f"{ticker:<10} {variant:<10} {name:<13}  no trades")
                continue
            print(f"{ticker:<10} {variant:<10} {name:<13} {stats['trades']:>6} "
                  f"{stats['winrate_pct']:>6.1f}% {stats['total_return_pct']:>+9.2f}% "
                  f"{stats['max_drawdown_pct']:>+7.2f}%")
    print()
