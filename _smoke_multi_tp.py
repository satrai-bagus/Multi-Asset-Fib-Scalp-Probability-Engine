"""End-to-end smoke test: multi-TP backtest works for top tickers."""
import joblib
import pandas as pd
from pathlib import Path

from backtest_engine import prepare_features, run_backtest_multi_tp, walk_forward_chunks_multi_tp

HERE = Path(__file__).parent

for ticker in ["LINKUSDT", "SUIUSDT", "SOLUSDT"]:
    print(f"\n=== {ticker} ===")
    df = pd.read_excel(HERE / "data" / f"dataset_{ticker}_18m.xlsx")
    df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"])
    df = df[(df["Lookahead Bars"] == 48) & df["SQZMOM Value"].notna()].reset_index(drop=True)
    df = df.dropna(subset=["Body% MA5", "Vol/MA20", "ATR Pct100"]).reset_index(drop=True)
    bundle = joblib.load(HERE / "models" / f"xgb_{ticker}.pkl")
    i2 = int(len(df) * 0.85)
    df_test = prepare_features(df.iloc[i2:].reset_index(drop=True))

    trades, equity, stats = run_backtest_multi_tp(
        df_test, bundle,
        tp1_level=2.5, tp2_level=3.6, sl_level=2.5,
        tp1_partial=0.5,
        min_confidence=0.65, min_body_pct=0.008,
    )

    print(f"  Trades: {stats['trades']}, Winrate: {stats['winrate_pct']:.1f}%, "
          f"Avg: {stats['avg_pl_pct']:+.3f}%, Total: {stats['total_return_pct']:+.2f}%, "
          f"DD: {stats['max_drawdown_pct']:+.2f}%")
    print(f"  Outcomes: {stats['outcomes']}")

    print(f"  Walk-forward chunks:")
    wf = walk_forward_chunks_multi_tp(df_test, bundle, n_chunks=4,
                                       tp1_level=2.5, tp2_level=3.6, sl_level=2.5,
                                       tp1_partial=0.5,
                                       min_confidence=0.65, min_body_pct=0.008)
    print(wf.to_string(index=False))

print("\nMulti-TP smoke test PASSED.")
