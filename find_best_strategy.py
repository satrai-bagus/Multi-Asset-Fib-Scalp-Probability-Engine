"""
EV optimizer: for every (TP, SL) combination at every confidence threshold,
compute Expected Value per trade in body units. Used to answer:
    "Given the model we have, what's the most profitable TP/SL strategy?"

Math note:
    Reward per win = (TP_level - 1) body lengths
    Risk per loss  = (SL_level - 1) body lengths
    EV (body)      = P(win) * Reward - P(loss) * Risk
                     (Neutral = no profit/loss in this framing, but still ties up capital.)

Binance spot fee ≈ 0.1% per side -> 0.2% round trip.
For ETH 1h, body% of price is typically 0.15-0.5% -> fee = 0.4-1.3 body units.
That's why low-reward strategies (TP 1.61 = 0.61 body) struggle after fees.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


HERE = Path(__file__).parent
DATASET = HERE / "data" / "dataset_18m.xlsx"
MODEL = HERE / "models" / "xgb_calibrated.pkl"

# ---------------------------------------------------------------------------
# Load + rebuild features (same as train_model.py)
# ---------------------------------------------------------------------------
df = pd.read_excel(DATASET)
df = df[(df["Lookahead Bars"] == 48) & df["SQZMOM Value"].notna()].reset_index(drop=True)
df = df.dropna(subset=["Body% MA5", "Vol/MA20", "ATR Pct100"]).reset_index(drop=True)
bundle = joblib.load(MODEL)
FEATURES = bundle["feature_cols"]
TARGETS = bundle["target_cols"]

df["Bar Color Code"] = df["Bar Color"].map({"Green": 1, "Red": -1, "Doji": 0})
df["Mom Color Code"] = df["Momentum Color"].map({"lime": 2, "green": 1, "maroon": -1, "red": -2}).fillna(0)
df["Squeeze Code"]   = df["Squeeze Status"].map({"Squeeze ON (black)": 2, "Squeeze OFF (gray)": 1, "No Squeeze (blue)": 0}).fillna(0)
df["Posisi Code"]    = df["Raw Posisi"].map({"LONG": 1, "SHORT": -1, "NO TRADE": 0}).fillna(0)
df["Last TR / ATR"]  = (df["Last TR"] / df["ATR 14"]).replace([np.inf, -np.inf], 0).fillna(0)
df["MACD Hist"]      = (df["MACD"] - df["MACD Signal"]).fillna(0)

# Same time-based split as train_model.py: last 15% = test
n = len(df)
i2 = int(n * 0.85)
df_test = df.iloc[i2:].reset_index(drop=True)
X_test = df_test[FEATURES].fillna(0).values

# Body% of price (rough): |close - open| / close
df_test["body_abs_pct"] = (df_test["Close"] - df_test["Open"]).abs() / df_test["Close"]
median_body_pct = df_test["body_abs_pct"].median()

# Predict probabilities (new bundle structure)
probs = np.column_stack([
    bundle["models"][t].predict_proba(X_test)[:, 1] for t in TARGETS
])
p_up_max = np.maximum(probs[:, 0], probs[:, 2])
p_dn_max = np.maximum(probs[:, 1], probs[:, 3])
pred_dir = np.where(p_up_max > p_dn_max, "Up", "Down")
confidence = np.maximum(p_up_max, p_dn_max)

orders = {
    "1.61 Up":   df_test["Fib 1.61 Up Order"].values,
    "2.5 Up":    df_test["Fib 2.5 Up Order"].values,
    "3.6 Up":    df_test["Fib 3.6 Up Order"].values,
    "1.61 Down": df_test["Fib 1.61 Down Order"].values,
    "2.5 Down":  df_test["Fib 2.5 Down Order"].values,
    "3.6 Down":  df_test["Fib 3.6 Down Order"].values,
}


def evaluate(tp_lvl, sl_lvl, conf_threshold=0.0, min_body_pct=0.0,
             fee_pct_roundtrip=0.002):
    """Evaluate (TP, SL) strategy on test set.
    Returns dict of stats including EV in body units AND realistic % return.

    min_body_pct: only trade bars with body% of price above this threshold.
    """
    wins = losses = neutrals = 0
    profit_pct_sum = 0.0
    n_traded = 0

    reward_body = tp_lvl - 1
    risk_body = sl_lvl - 1
    fee_body_unit = fee_pct_roundtrip / max(median_body_pct, 1e-6)  # fee in body units

    for i in range(len(df_test)):
        if confidence[i] < conf_threshold:
            continue
        body_pct = df_test["body_abs_pct"].iloc[i]
        if body_pct < min_body_pct:
            continue
        n_traded += 1

        if pred_dir[i] == "Up":
            tp_o = orders[f"{tp_lvl} Up"][i]
            sl_o = orders[f"{sl_lvl} Down"][i]
        else:
            tp_o = orders[f"{tp_lvl} Down"][i]
            sl_o = orders[f"{sl_lvl} Up"][i]

        if tp_o > 0 and (sl_o == 0 or tp_o < sl_o):
            wins += 1
            profit_pct_sum += reward_body * body_pct - fee_pct_roundtrip
        elif sl_o > 0 and (tp_o == 0 or sl_o < tp_o):
            losses += 1
            profit_pct_sum += -risk_body * body_pct - fee_pct_roundtrip
        else:
            neutrals += 1
            profit_pct_sum += 0 - fee_pct_roundtrip  # close at break-even but still pay fees

    total = wins + losses + neutrals
    if total == 0:
        return None

    p_win = wins / total
    p_loss = losses / total
    ev_body = p_win * reward_body - p_loss * risk_body
    ev_body_after_fees = ev_body - fee_body_unit  # rough approximation

    return {
        "tp": tp_lvl, "sl": sl_lvl,
        "rr": reward_body / risk_body if risk_body > 0 else float("inf"),
        "trades": total, "wins": wins, "losses": losses, "neutrals": neutrals,
        "winrate_all": p_win * 100,
        "ev_body_gross": ev_body,
        "ev_body_net": ev_body_after_fees,
        "avg_profit_pct_per_trade": profit_pct_sum / total * 100,
        "total_return_pct": profit_pct_sum * 100,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
print(f"Median body% of price (ETH 1h, test set): {median_body_pct*100:.3f}%")
print(f"Round-trip fee (Binance spot 0.1% × 2):    0.2%")
print(f"Fee in body units (median):                {0.002/median_body_pct:.2f} body\n")

print("=" * 90)
print("ALL (TP, SL) COMBINATIONS — no confidence filter (sorted by EV body net)")
print("=" * 90)
print(f"{'TP':>4} {'SL':>4} {'R:R':>5}  {'Trades':>7} {'WinRt%':>7}  "
      f"{'EV (body)':>10} {'EV net':>9}  {'Avg %/trade':>12} {'Total Ret%':>11}")

all_results = []
for tp in [1.61, 2.5, 3.6]:
    for sl in [1.61, 2.5, 3.6]:
        r = evaluate(tp, sl, conf_threshold=0)
        if r:
            all_results.append(r)
            print(f"{tp:>4} {sl:>4}  {r['rr']:>4.2f}   {r['trades']:>7} "
                  f"{r['winrate_all']:>6.1f}%   {r['ev_body_gross']:>+8.3f}  {r['ev_body_net']:>+8.3f}  "
                  f"{r['avg_profit_pct_per_trade']:>+11.4f}%  {r['total_return_pct']:>+10.2f}%")

print("\n" + "=" * 90)
print("BEST 5 COMBOS ACROSS ALL CONFIDENCE THRESHOLDS  (min 30 trades)")
print("=" * 90)

best_combos = []
for threshold in [0.0, 0.55, 0.60, 0.65, 0.70, 0.75]:
    for min_body in [0.0, 0.003, 0.005, 0.008, 0.012]:  # 0%, 0.3%, 0.5%, 0.8%, 1.2%
        for tp in [1.61, 2.5, 3.6]:
            for sl in [1.61, 2.5, 3.6]:
                r = evaluate(tp, sl, conf_threshold=threshold, min_body_pct=min_body)
                if r and r["trades"] >= 30:
                    r["threshold"] = threshold
                    r["min_body"] = min_body
                    best_combos.append(r)

# Rank by total_return_pct (more realistic than EV body alone since it accounts for actual body sizes)
best_combos.sort(key=lambda x: -x["total_return_pct"])
print(f"{'TP':>4} {'SL':>4} {'Conf>=':>6} {'MinBody':>7}  {'Trades':>7} {'WinRt%':>7}  "
      f"{'Avg %/trade':>12} {'Total Ret%':>11}")
for r in best_combos[:10]:
    print(f"{r['tp']:>4} {r['sl']:>4}  {r['threshold']*100:>4.0f}%  {r['min_body']*100:>5.2f}%   "
          f"{r['trades']:>7} {r['winrate_all']:>6.1f}%  "
          f"{r['avg_profit_pct_per_trade']:>+11.4f}%  {r['total_return_pct']:>+10.2f}%")

print("\n" + "=" * 90)
print("KEY INSIGHTS")
print("=" * 90)
best = best_combos[0]
test_days = (pd.to_datetime(df_test["Datetime (UTC)"].iloc[-1]) -
             pd.to_datetime(df_test["Datetime (UTC)"].iloc[0])).total_seconds() / 86400

print(f"\nBest strategy on test set ({df_test['Datetime (UTC)'].iloc[0]} -> {df_test['Datetime (UTC)'].iloc[-1]}):")
print(f"   TP = {best['tp']},  SL = {best['sl']}")
print(f"   Confidence >= {best['threshold']*100:.0f}%, Body >= {best['min_body']*100:.2f}% of price")
print(f"   {best['trades']} trades in {test_days:.0f} days  ({best['trades']/test_days:.1f} per day)")
print(f"   Winrate: {best['winrate_all']:.1f}%")
print(f"   Avg %/trade (after 0.2% fees): {best['avg_profit_pct_per_trade']:+.4f}%")
print(f"   Total return: {best['total_return_pct']:+.2f}%")
annualized = best['total_return_pct'] * 365 / test_days if test_days > 0 else 0
print(f"   Annualized (extrapolation):  {annualized:+.1f}% / year")

# Show second and third best for sensitivity
if len(best_combos) > 1:
    print(f"\nFor sensitivity, top 3:")
    for r in best_combos[:3]:
        print(f"   TP={r['tp']}, SL={r['sl']}, Conf>={r['threshold']*100:.0f}%, "
              f"BodyMin={r['min_body']*100:.2f}%  -> "
              f"{r['trades']} trades, {r['total_return_pct']:+.2f}% total")
