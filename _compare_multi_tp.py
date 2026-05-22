"""Compare Single-TP vs Multi-TP for top tickers."""
import joblib
import pandas as pd
from pathlib import Path

from backtest_engine import prepare_features, run_backtest, run_backtest_multi_tp

HERE = Path(__file__).parent
# Focus on top performers
TICKERS = ["LINKUSDT", "SUIUSDT", "AAVEUSDT", "ARBUSDT", "SOLUSDT", "ETHUSDT",
           "DOGEUSDT", "FETUSDT", "APTUSDT"]

print(f"{'Ticker':<10} {'Variant':<12} {'Trades':>6} {'WinRt%':>7} {'AvgPnL%':>8} {'TotalRet%':>10} {'MaxDD%':>8}")
print("-" * 78)

for ticker in TICKERS:
    df = pd.read_excel(HERE / "data" / f"dataset_{ticker}_18m.xlsx")
    df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"])
    df = df[(df["Lookahead Bars"] == 48) & df["SQZMOM Value"].notna()].reset_index(drop=True)
    df = df.dropna(subset=["Body% MA5", "Vol/MA20", "ATR Pct100"]).reset_index(drop=True)
    bundle = joblib.load(HERE / "models" / f"xgb_{ticker}.pkl")
    i2 = int(len(df) * 0.85)
    df_test = prepare_features(df.iloc[i2:].reset_index(drop=True))

    # Single-TP Conservative
    _, _, s1 = run_backtest(df_test, bundle,
                             tp_level=2.5, sl_level=2.5,
                             min_confidence=0.65, min_body_pct=0.008)
    if s1:
        print(f"{ticker:<10} {'Single-TP':<12} {s1['trades']:>6} "
              f"{s1['winrate_pct']:>6.1f}% {s1['avg_pl_pct']:>+7.3f}% "
              f"{s1['total_return_pct']:>+9.2f}% {s1['max_drawdown_pct']:>+7.2f}%")

    # Multi-TP (TP1=2.5 50%, TP2=3.6 50%, SL=2.5 opp, BE after TP1)
    _, _, s2 = run_backtest_multi_tp(df_test, bundle,
                                      tp1_level=2.5, tp2_level=3.6, sl_level=2.5,
                                      tp1_partial=0.5,
                                      min_confidence=0.65, min_body_pct=0.008)
    if s2:
        print(f"{ticker:<10} {'Multi-TP':<12} {s2['trades']:>6} "
              f"{s2['winrate_pct']:>6.1f}% {s2['avg_pl_pct']:>+7.3f}% "
              f"{s2['total_return_pct']:>+9.2f}% {s2['max_drawdown_pct']:>+7.2f}%")
        # Outcome breakdown
        print(f"           outcomes: {s2['outcomes']}")
    print()
