"""
Train calibrated XGBoost multi-output classifier on enriched 18-month dataset.

Improvements over v1:
  - Dataset: 18 months (vs 6) -> more training data
  - Features: +volume, +multi-bar context, +time-of-day, +volatility regime
  - Model: XGBoost (usually +3-5% AUC over RF on tabular)
  - Calibration: isotonic regression -> "65% confidence" actually means 65% empirical winrate
  - Validation: time-based 70/15/15 split (train / calib / test)
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss
from xgboost import XGBClassifier


HERE = Path(__file__).parent

parser = argparse.ArgumentParser(description="Train XGBoost on dataset.")
parser.add_argument("--symbol", default="ETHUSDT", help="Symbol tag for file naming")
parser.add_argument("--dataset", default=None, help="Override dataset path")
parser.add_argument("--model-out", default=None, help="Override model output path")
parser.add_argument("--no-btc", action="store_true",
                    help="Train without BTC cross-asset features (useful for ARB/FET/DOGE which suffer from BTC noise)")
args = parser.parse_args()

DATASET = Path(args.dataset) if args.dataset else HERE / "data" / f"dataset_{args.symbol}_18m.xlsx"
suffix = "_nobtc" if args.no_btc else ""
MODEL_OUT = Path(args.model_out) if args.model_out else HERE / "models" / f"xgb_{args.symbol}{suffix}.pkl"
print(f"Symbol: {args.symbol}")
print(f"Dataset: {DATASET}")
print(f"Model out: {MODEL_OUT}")


# ---------------------------------------------------------------------------
# 1) Load & filter
# ---------------------------------------------------------------------------
df = pd.read_excel(DATASET)
print(f"Loaded {len(df)} rows from {DATASET.name}")

# Drop bars without full 48h lookahead, SQZMOM warmup, multi-bar context warmup
df = df[(df["Lookahead Bars"] == 48) & df["SQZMOM Value"].notna()].reset_index(drop=True)
dropna_cols = ["Body% MA5", "Vol/MA20", "ATR Pct100"]
if not args.no_btc:
    dropna_cols += ["BTC Ret 24h", "BTC ATR Pct100"]
df = df.dropna(subset=dropna_cols).reset_index(drop=True)
print(f"After filtering (lookahead==48, indicators warm): {len(df)} rows")


# ---------------------------------------------------------------------------
# 2) Feature engineering — encode categoricals + derive features
# ---------------------------------------------------------------------------
df["Bar Color Code"] = df["Bar Color"].map({"Green": 1, "Red": -1, "Doji": 0})
df["Mom Color Code"] = df["Momentum Color"].map(
    {"lime": 2, "green": 1, "maroon": -1, "red": -2}
).fillna(0)
df["Squeeze Code"] = df["Squeeze Status"].map(
    {"Squeeze ON (black)": 2, "Squeeze OFF (gray)": 1, "No Squeeze (blue)": 0}
).fillna(0)
df["Posisi Code"] = df["Raw Posisi"].map(
    {"LONG": 1, "SHORT": -1, "NO TRADE": 0}
).fillna(0)
df["Last TR / ATR"] = (df["Last TR"] / df["ATR 14"]).replace([np.inf, -np.inf], 0).fillna(0)
df["MACD Hist"] = (df["MACD"] - df["MACD Signal"]).fillna(0)


BTC_FEATURES = ["BTC Ret 1h", "BTC Ret 4h", "BTC Ret 24h",
                "BTC Body Dir", "BTC Range/ATR", "BTC ATR Pct100"]

BASE_FEATURES = [
    # Bar geometry
    "Bar Color Code", "Fib Zone", "Fib Position",
    "Body %", "Upper Wick %", "Lower Wick %", "Range", "Range/ATR",
    # Multi-bar context
    "Body% MA5", "Body% Pct100", "ATR Pct100",
    "Streak", "Prev Body%", "Prev Bar Color", "Prev Range/ATR",
    "Close Ret 5", "SQZMOM Delta3",
    # Volume
    "Vol/MA20", "Vol Pct100",
    # Time-of-day
    "Hour Sin", "Hour Cos", "DoW Sin", "DoW Cos",
    # SQZMOM
    "SQZMOM Value", "Mom Color Code", "Squeeze Code",
    # Analisa-style indicators
    "RSI 14", "ADX 14", "MACD Hist",
    "HTF 4H Trend", "Posisi Code", "Last TR / ATR",
]

FEATURES = BASE_FEATURES if args.no_btc else BASE_FEATURES + BTC_FEATURES
print(f"Feature set: {'BASE (no BTC)' if args.no_btc else 'BASE + BTC'}  ({len(FEATURES)} features)")
TARGETS = [
    "Fib 1.61 Up", "Fib 1.61 Down",
    "Fib 2.5 Up",  "Fib 2.5 Down",
    "Fib 3.6 Up",  "Fib 3.6 Down",
]

X = df[FEATURES].fillna(0).values
y = df[TARGETS].astype(int).values
print(f"Features: {len(FEATURES)}, Targets: {len(TARGETS)}")


# ---------------------------------------------------------------------------
# 3) Time-based 70 / 15 / 15 split
# ---------------------------------------------------------------------------
n = len(df)
i1 = int(n * 0.70)
i2 = int(n * 0.85)
X_train, X_calib, X_test = X[:i1], X[i1:i2], X[i2:]
y_train, y_calib, y_test = y[:i1], y[i1:i2], y[i2:]
df_test = df.iloc[i2:].reset_index(drop=True)

print(f"Split — Train: {len(X_train)}  Calib: {len(X_calib)}  Test: {len(X_test)}")
print(f"  Train period: {df['Datetime (UTC)'].iloc[0]} -> {df['Datetime (UTC)'].iloc[i1-1]}")
print(f"  Calib period: {df['Datetime (UTC)'].iloc[i1]} -> {df['Datetime (UTC)'].iloc[i2-1]}")
print(f"  Test  period: {df['Datetime (UTC)'].iloc[i2]} -> {df['Datetime (UTC)'].iloc[-1]}")


# ---------------------------------------------------------------------------
# 4) Train one XGBoost per target, calibrate on hold-out
# ---------------------------------------------------------------------------
xgb_params = dict(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=10,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
)

models = {}
probs_test = np.zeros_like(y_test, dtype=float)
auc_scores = {}
brier_scores = {}

print("\nTraining XGBoost + isotonic calibration per target...")
for i, target in enumerate(TARGETS):
    base = XGBClassifier(**xgb_params)
    base.fit(X_train, y_train[:, i])
    # Wrap with calibration trained on hold-out calibration set
    cal = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
    cal.fit(X_calib, y_calib[:, i])
    models[target] = cal
    p_te = cal.predict_proba(X_test)[:, 1]
    probs_test[:, i] = p_te
    if len(np.unique(y_test[:, i])) > 1:
        auc_scores[target] = roc_auc_score(y_test[:, i], p_te)
        brier_scores[target] = brier_score_loss(y_test[:, i], p_te)
    print(f"  {target:18}  AUC={auc_scores.get(target, float('nan')):.3f}  "
          f"Brier={brier_scores.get(target, float('nan')):.3f}")


# ---------------------------------------------------------------------------
# 5) Winrate evaluation on test set
# ---------------------------------------------------------------------------
p_up_max = np.maximum(probs_test[:, 0], probs_test[:, 2])
p_dn_max = np.maximum(probs_test[:, 1], probs_test[:, 3])
pred_dir = np.where(p_up_max > p_dn_max, "Up", "Down")

o_1u = df_test["Fib 1.61 Up Order"].values
o_2u = df_test["Fib 2.5 Up Order"].values
o_3u = df_test["Fib 3.6 Up Order"].values
o_1d = df_test["Fib 1.61 Down Order"].values
o_2d = df_test["Fib 2.5 Down Order"].values
o_3d = df_test["Fib 3.6 Down Order"].values


def outcome(tp_order, sl_order):
    if tp_order > 0 and (sl_order == 0 or tp_order < sl_order):
        return "win"
    if sl_order > 0 and (tp_order == 0 or sl_order < tp_order):
        return "loss"
    return "neutral"


tp_levels = ["1.61", "2.5", "3.6"]
results = {tp: {"win": 0, "loss": 0, "neutral": 0} for tp in tp_levels}

for i in range(len(df_test)):
    if pred_dir[i] == "Up":
        sl_o = o_3d[i]
        tp_orders = {"1.61": o_1u[i], "2.5": o_2u[i], "3.6": o_3u[i]}
    else:
        sl_o = o_3u[i]
        tp_orders = {"1.61": o_1d[i], "2.5": o_2d[i], "3.6": o_3d[i]}
    for tp_name, tp_o in tp_orders.items():
        results[tp_name][outcome(int(tp_o), int(sl_o))] += 1


def winrate_decided(r):
    decided = r["win"] + r["loss"]
    return (r["win"] / decided * 100) if decided > 0 else float("nan")


print("\n=== Winrate (full test set, no filter) ===")
print(f"  {'TP':<6} {'Win':>5} {'Loss':>5} {'Neutral':>7} {'Win% (decided)':>17}")
for tp in tp_levels:
    r = results[tp]
    print(f"  {tp:<6} {r['win']:>5} {r['loss']:>5} {r['neutral']:>7}  {winrate_decided(r):>15.1f}%")


# ---------------------------------------------------------------------------
# 6) Feature importance (averaged over 6 base XGBoost models)
# ---------------------------------------------------------------------------
fi_total = np.zeros(len(FEATURES))
for target, cal_model in models.items():
    # CalibratedClassifierCV stores the underlying estimator in calibrated_classifiers_[0].estimator
    base_est = cal_model.calibrated_classifiers_[0].estimator
    fi_total += base_est.feature_importances_
fi_avg = fi_total / len(models)

fi = sorted(zip(FEATURES, fi_avg), key=lambda x: -x[1])
print("\n=== Feature importance (top 15, averaged across 6 models) ===")
for name, imp in fi[:15]:
    bar = "*" * int(imp * 200)
    print(f"  {name:20} {imp:.4f}  {bar}")


# ---------------------------------------------------------------------------
# 7) Save bundle (compatible with backtest_engine + app)
# ---------------------------------------------------------------------------
MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
joblib.dump({
    "symbol": args.symbol,
    "models": models,
    "feature_cols": FEATURES,
    "target_cols": TARGETS,
    "auc_scores": auc_scores,
    "brier_scores": brier_scores,
    "winrate_results": results,
    "test_period": (str(df_test["Datetime (UTC)"].iloc[0]),
                    str(df_test["Datetime (UTC)"].iloc[-1])),
    "n_train": len(X_train), "n_calib": len(X_calib), "n_test": len(X_test),
    "feature_importance": dict(zip(FEATURES, fi_avg.tolist())),
    "model_type": "xgboost_isotonic_calibrated",
}, MODEL_OUT)
print(f"\nSaved to {MODEL_OUT}")
