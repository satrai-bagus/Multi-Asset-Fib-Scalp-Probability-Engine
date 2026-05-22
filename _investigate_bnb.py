"""Deep-dive investigation: why is BNB model underperforming?
Compare market characteristics + target balance across all tickers."""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
TICKERS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"]

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)


# =============================================================================
# Step 1: Market profile per ticker
# =============================================================================
print("=" * 100)
print("STEP 1: Market profile per ticker")
print("=" * 100)
profiles = []
for t in TICKERS:
    df = pd.read_excel(HERE / "data" / f"dataset_{t}_18m.xlsx")
    df = df[(df["Lookahead Bars"] == 48) & df["SQZMOM Value"].notna()].reset_index(drop=True)
    df = df.dropna(subset=["Body% MA5", "Vol/MA20"]).reset_index(drop=True)
    body_abs_pct = (df["Close"] - df["Open"]).abs() / df["Close"]
    range_pct = (df["High"] - df["Low"]) / df["Close"]

    profiles.append({
        "Ticker": t,
        "Bars": len(df),
        "Median Body% of price": body_abs_pct.median() * 100,
        "Median Range% of price": range_pct.median() * 100,
        "Mean Body%": body_abs_pct.mean() * 100,
        "Stdev Body%": body_abs_pct.std() * 100,
        "% Doji-like (body < 0.05%)": (body_abs_pct < 0.0005).mean() * 100,
        "% Green bars": (df["Bar Color"] == "Green").mean() * 100,
    })
prof_df = pd.DataFrame(profiles)
print(prof_df.to_string(index=False))


# =============================================================================
# Step 2: Target class balance (does BNB have too few "hits"?)
# =============================================================================
print("\n" + "=" * 100)
print("STEP 2: Target hit rate per ticker (% of bars where each fib level hit in 48h)")
print("=" * 100)
hit_rows = []
for t in TICKERS:
    df = pd.read_excel(HERE / "data" / f"dataset_{t}_18m.xlsx")
    df = df[(df["Lookahead Bars"] == 48) & df["SQZMOM Value"].notna()].reset_index(drop=True)
    row = {"Ticker": t}
    for col in ["Fib 1.61 Up", "Fib 1.61 Down", "Fib 2.5 Up", "Fib 2.5 Down", "Fib 3.6 Up", "Fib 3.6 Down"]:
        row[col] = df[col].mean() * 100
    hit_rows.append(row)
hit_df = pd.DataFrame(hit_rows)
print(hit_df.to_string(index=False))


# =============================================================================
# Step 3: Per-ticker test set AUC (model quality per coin)
# =============================================================================
print("\n" + "=" * 100)
print("STEP 3: Test-set AUC + sample sizes per ticker")
print("=" * 100)
auc_rows = []
for t in TICKERS:
    pkl = HERE / "models" / f"xgb_{t}.pkl"
    if not pkl.exists():
        continue
    b = joblib.load(pkl)
    row = {"Ticker": t, "n_train": b["n_train"], "n_test": b["n_test"]}
    for k, v in b["auc_scores"].items():
        row[k] = v
    auc_rows.append(row)
auc_df = pd.DataFrame(auc_rows)
print(auc_df.to_string(index=False))


# =============================================================================
# Step 4: BNB walk-forward — is it bad period or bad model?
# =============================================================================
print("\n" + "=" * 100)
print("STEP 4: BNB walk-forward (split test into 6 chunks) to see if issue is regime-specific")
print("=" * 100)
from backtest_engine import prepare_features, run_backtest

df = pd.read_excel(HERE / "data" / "dataset_BNBUSDT_18m.xlsx")
df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"])
df = df[(df["Lookahead Bars"] == 48) & df["SQZMOM Value"].notna()].reset_index(drop=True)
df = df.dropna(subset=["Body% MA5", "Vol/MA20", "ATR Pct100"]).reset_index(drop=True)
bundle = joblib.load(HERE / "models" / "xgb_BNBUSDT.pkl")
i2 = int(len(df) * 0.85)
df_test = prepare_features(df.iloc[i2:].reset_index(drop=True))

n_chunks = 6
chunk_size = len(df_test) // n_chunks
print(f"{'Chunk':<6} {'From':<20} {'To':<20} {'Trades':>6} {'WinRt%':>7} {'TotalRet%':>10}")
for k in range(n_chunks):
    start = k * chunk_size
    end = (k + 1) * chunk_size if k < n_chunks - 1 else len(df_test)
    chunk = df_test.iloc[start:end].reset_index(drop=True)
    if len(chunk) < 5:
        continue
    _, _, stats = run_backtest(chunk, bundle, tp_level=2.5, sl_level=2.5,
                                min_confidence=0.65, min_body_pct=0.005)
    if not stats:
        print(f"{k+1:<6} {str(chunk['Datetime (UTC)'].iloc[0]):<20} {str(chunk['Datetime (UTC)'].iloc[-1]):<20}  (no trades)")
        continue
    print(f"{k+1:<6} {str(chunk['Datetime (UTC)'].iloc[0]):<20} {str(chunk['Datetime (UTC)'].iloc[-1]):<20} "
          f"{stats['trades']:>6} {stats['winrate_pct']:>6.1f}% {stats['total_return_pct']:>+9.2f}%")


# =============================================================================
# Step 5: BNB strategy grid search — maybe optimal params differ
# =============================================================================
print("\n" + "=" * 100)
print("STEP 5: BNB strategy grid search (find ANY profitable config)")
print("=" * 100)
results = []
for tp in [1.61, 2.5, 3.6]:
    for sl in [1.61, 2.5, 3.6]:
        for conf in [0.55, 0.60, 0.65, 0.70, 0.75]:
            for body in [0.003, 0.005, 0.008, 0.012]:
                _, _, stats = run_backtest(df_test, bundle, tp_level=tp, sl_level=sl,
                                            min_confidence=conf, min_body_pct=body)
                if stats and stats["trades"] >= 25:
                    results.append({
                        "TP": tp, "SL": sl, "Conf": conf, "MinBody": body,
                        "Trades": stats["trades"], "Win%": stats["winrate_pct"],
                        "TotalRet%": stats["total_return_pct"],
                        "MaxDD%": stats["max_drawdown_pct"],
                    })
res_df = pd.DataFrame(results).sort_values("TotalRet%", ascending=False)
print(f"Best 8 BNB configs (>=25 trades):")
print(res_df.head(8).to_string(index=False))
print(f"\nWorst 3:")
print(res_df.tail(3).to_string(index=False))
print(f"\nProfitable configs (TotalRet > 0%): {(res_df['TotalRet%'] > 0).sum()} / {len(res_df)}")
