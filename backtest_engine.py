"""
Backtest engine: simulates trading the model on the test set.

For each bar in test set:
  - Compute model prediction (direction + confidence)
  - Check filters (body% threshold, confidence threshold)
  - If tradable: simulate TP-vs-SL outcome using actual fib-hit data
  - Track equity curve, per-trade log, walk-forward chunks
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


HERE = Path(__file__).parent


def prepare_features(df):
    df = df.copy()
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
    df["Last TR / ATR"] = (df["Last TR"] / df["ATR 14"]).replace(
        [np.inf, -np.inf], 0
    ).fillna(0)
    df["MACD Hist"] = (df["MACD"] - df["MACD Signal"]).fillna(0)
    df["body_abs_pct"] = (df["Close"] - df["Open"]).abs() / df["Close"]
    return df


def predict_probs(bundle, X):
    """Get (n, 6) probability matrix from bundle, supporting both old and new bundles."""
    TARGETS = bundle["target_cols"]
    if "models" in bundle:  # new: dict of calibrated models per target
        return np.column_stack([
            bundle["models"][t].predict_proba(X)[:, 1] for t in TARGETS
        ])
    elif "model" in bundle:  # old: MultiOutputClassifier
        model = bundle["model"]
        return np.array([est.predict_proba(X)[:, 1] for est in model.estimators_]).T
    raise ValueError("Bundle has neither 'models' nor 'model' key")


def run_backtest_multi_tp(df, bundle,
                          tp1_level=2.5, tp2_level=3.6, sl_level=2.5,
                          min_confidence=0.65, min_body_pct=0.008,
                          tp1_partial=0.5, fee_pct_roundtrip=0.002,
                          lookahead_bars=48):
    """Multi-TP backtest with bar-by-bar simulation.

    Logic per trade:
      1. Enter at close
      2. Close `tp1_partial` of position when TP1 hits
      3. Move SL to entry (breakeven) for remaining position
      4. Close remaining at TP2 hit, or at BE if reversal
      5. If 48h timeout: close remaining at last close

    SL/TP priority within a single bar: SL checked first (conservative for risk).
    """
    FEATURES = bundle["feature_cols"]
    X = df[FEATURES].fillna(0).values
    probs = predict_probs(bundle, X)
    p_up_max = np.maximum(probs[:, 0], probs[:, 2])
    p_dn_max = np.maximum(probs[:, 1], probs[:, 3])

    o_arr = df["Open"].values
    c_arr = df["Close"].values
    h_arr = df["High"].values
    l_arr = df["Low"].values
    look_arr = df["Lookahead Bars"].values
    body_pct_arr = df["body_abs_pct"].values
    dt_arr = df["Datetime (UTC)"].values

    trades = []
    n = len(df)
    for i in range(n):
        if body_pct_arr[i] < min_body_pct:
            continue
        conf = max(p_up_max[i], p_dn_max[i])
        if conf < min_confidence:
            continue
        if look_arr[i] < lookahead_bars:
            continue

        direction = "LONG" if p_up_max[i] > p_dn_max[i] else "SHORT"
        o_i = o_arr[i]; c_i = c_arr[i]
        body_top = max(o_i, c_i); body_bot = min(o_i, c_i)
        body_len = abs(c_i - o_i)
        body_pct = body_pct_arr[i]
        entry = c_i

        if direction == "LONG":
            tp1 = body_top + (tp1_level - 1) * body_len
            tp2 = body_top + (tp2_level - 1) * body_len
            sl_orig = body_bot - (sl_level - 1) * body_len
        else:
            tp1 = body_bot - (tp1_level - 1) * body_len
            tp2 = body_bot - (tp2_level - 1) * body_len
            sl_orig = body_top + (sl_level - 1) * body_len

        half_closed = False
        sl_active = sl_orig
        outcome = None
        pnl_pct = 0.0

        end = min(i + 1 + lookahead_bars, n)
        for j in range(i + 1, end):
            h_j = h_arr[j]; l_j = l_arr[j]

            # Check SL first (conservative)
            if direction == "LONG":
                sl_hit = l_j <= sl_active
                tp1_hit = h_j >= tp1
                tp2_hit = h_j >= tp2
            else:
                sl_hit = h_j >= sl_active
                tp1_hit = l_j <= tp1
                tp2_hit = l_j <= tp2

            if sl_hit:
                if half_closed:
                    # SL at entry (BE) on remaining 50%: half profit already locked
                    pnl_pct = tp1_partial * (tp1_level - 1) * body_pct
                    outcome = "TP1+BE"
                else:
                    # Full SL hit
                    pnl_pct = -(sl_level - 1) * body_pct
                    outcome = "SL"
                break

            if tp2_hit:
                if half_closed:
                    pnl_pct = (tp1_partial * (tp1_level - 1) + (1 - tp1_partial) * (tp2_level - 1)) * body_pct
                    outcome = "TP1+TP2"
                else:
                    # TP2 hit before TP1 — single-bar big move spanning both. Treat as TP1+TP2.
                    pnl_pct = (tp1_partial * (tp1_level - 1) + (1 - tp1_partial) * (tp2_level - 1)) * body_pct
                    outcome = "TP1+TP2 same bar"
                break

            if tp1_hit and not half_closed:
                half_closed = True
                sl_active = entry  # Move SL to BE
                # Continue scanning for TP2 or BE-stop

        if outcome is None:
            # Timeout
            last_c = c_arr[end - 1]
            if direction == "LONG":
                tail_ret = (last_c - entry) / entry
            else:
                tail_ret = (entry - last_c) / entry
            if half_closed:
                pnl_pct = tp1_partial * (tp1_level - 1) * body_pct + (1 - tp1_partial) * tail_ret
                outcome = "TP1+timeout"
            else:
                pnl_pct = tail_ret
                outcome = "timeout"

        pnl_pct -= fee_pct_roundtrip  # round-trip fee

        trades.append({
            "Datetime (UTC)": dt_arr[i],
            "Direction": direction,
            "Confidence": conf,
            "Body %": body_pct,
            "Entry": entry,
            "TP1": tp1, "TP2": tp2, "SL": sl_orig,
            "Outcome": outcome,
            "P/L %": pnl_pct * 100,
        })

    if not trades:
        return pd.DataFrame(), pd.Series(dtype=float), {}

    trades_df = pd.DataFrame(trades)
    equity = (1.0 + trades_df["P/L %"] / 100).cumprod()
    equity.index = pd.to_datetime(trades_df["Datetime (UTC)"])

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd_pct = drawdown.min() * 100

    n_trades = len(trades_df)
    wins = (trades_df["P/L %"] > 0).sum()
    losses = (trades_df["P/L %"] < 0).sum()
    breakevens = (trades_df["P/L %"] == 0).sum()

    # Outcome breakdown
    outcomes = trades_df["Outcome"].value_counts().to_dict()

    avg_pl = trades_df["P/L %"].mean()
    std_pl = trades_df["P/L %"].std()
    sharpe_like = (avg_pl / std_pl * np.sqrt(252 * 24)) if std_pl > 0 else 0
    total_ret_pct = (equity.iloc[-1] - 1) * 100

    stats = {
        "trades": n_trades,
        "wins": int(wins),
        "losses": int(losses),
        "breakevens": int(breakevens),
        "winrate_pct": wins / n_trades * 100,
        "outcomes": outcomes,
        "avg_pl_pct": avg_pl,
        "total_return_pct": total_ret_pct,
        "max_drawdown_pct": max_dd_pct,
        "sharpe_like": sharpe_like,
        "n_bars_evaluated": len(df),
        "trade_rate": n_trades / len(df) * 100,
    }
    return trades_df, equity, stats


def walk_forward_chunks_multi_tp(df, bundle, n_chunks=4, **strategy_params):
    """Walk-forward chunks for multi-TP backtest."""
    chunk_size = len(df) // n_chunks
    rows = []
    for k in range(n_chunks):
        start = k * chunk_size
        end = (k + 1) * chunk_size if k < n_chunks - 1 else len(df)
        chunk = df.iloc[start:end].reset_index(drop=True)
        if len(chunk) < 10:
            continue
        _, _, stats = run_backtest_multi_tp(chunk, bundle, **strategy_params)
        if not stats:
            rows.append({
                "Chunk": k + 1,
                "From": str(chunk["Datetime (UTC)"].iloc[0]),
                "To": str(chunk["Datetime (UTC)"].iloc[-1]),
                "Trades": 0, "Winrate %": float("nan"),
                "Total Return %": 0.0, "Max DD %": 0.0,
            })
            continue
        rows.append({
            "Chunk": k + 1,
            "From": str(chunk["Datetime (UTC)"].iloc[0]),
            "To": str(chunk["Datetime (UTC)"].iloc[-1]),
            "Trades": stats["trades"],
            "Winrate %": stats["winrate_pct"],
            "Total Return %": stats["total_return_pct"],
            "Max DD %": stats["max_drawdown_pct"],
        })
    return pd.DataFrame(rows)


def run_backtest(df, bundle, tp_level=3.6, sl_level=2.5,
                 min_confidence=0.55, min_body_pct=0.005,
                 fee_pct_roundtrip=0.002):
    """Simulate trading the strategy on `df` (already feature-prepared).
    Returns (trades_df, equity_series, stats_dict).
    """
    FEATURES = bundle["feature_cols"]
    X = df[FEATURES].fillna(0).values
    probs = predict_probs(bundle, X)
    p_up_max = np.maximum(probs[:, 0], probs[:, 2])
    p_dn_max = np.maximum(probs[:, 1], probs[:, 3])

    reward_body = tp_level - 1
    risk_body = sl_level - 1

    trades = []
    for i in range(len(df)):
        body_pct = df["body_abs_pct"].iloc[i]
        conf = max(p_up_max[i], p_dn_max[i])
        if body_pct < min_body_pct or conf < min_confidence:
            continue
        # Need full 48h lookahead for honest outcome
        if df["Lookahead Bars"].iloc[i] < 48:
            continue

        direction = "Up" if p_up_max[i] > p_dn_max[i] else "Down"
        if direction == "Up":
            tp_o = df[f"Fib {tp_level} Up Order"].iloc[i]
            sl_o = df[f"Fib {sl_level} Down Order"].iloc[i]
        else:
            tp_o = df[f"Fib {tp_level} Down Order"].iloc[i]
            sl_o = df[f"Fib {sl_level} Up Order"].iloc[i]

        if tp_o > 0 and (sl_o == 0 or tp_o < sl_o):
            outcome, pnl_pct = "WIN", reward_body * body_pct - fee_pct_roundtrip
        elif sl_o > 0 and (tp_o == 0 or sl_o < tp_o):
            outcome, pnl_pct = "LOSS", -risk_body * body_pct - fee_pct_roundtrip
        else:
            outcome, pnl_pct = "NEUTRAL", -fee_pct_roundtrip

        trades.append({
            "Datetime (UTC)": df["Datetime (UTC)"].iloc[i],
            "Direction": "LONG" if direction == "Up" else "SHORT",
            "Confidence": conf,
            "Body %": body_pct,
            "Entry": df["Close"].iloc[i],
            "TP order": int(tp_o), "SL order": int(sl_o),
            "Outcome": outcome,
            "P/L %": pnl_pct * 100,
        })

    if not trades:
        return pd.DataFrame(), pd.Series(dtype=float), {}

    trades_df = pd.DataFrame(trades)
    equity = (1.0 + trades_df["P/L %"] / 100).cumprod()
    equity.index = pd.to_datetime(trades_df["Datetime (UTC)"])

    # Running max drawdown
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd_pct = drawdown.min() * 100

    n = len(trades_df)
    wins = (trades_df["Outcome"] == "WIN").sum()
    losses = (trades_df["Outcome"] == "LOSS").sum()
    neutrals = (trades_df["Outcome"] == "NEUTRAL").sum()
    total_ret_pct = (equity.iloc[-1] - 1) * 100
    avg_pl = trades_df["P/L %"].mean()
    std_pl = trades_df["P/L %"].std()
    sharpe_like = (avg_pl / std_pl * np.sqrt(252 * 24)) if std_pl > 0 else 0  # 24 1h bars/day, 252 trading days

    stats = {
        "trades": n,
        "wins": int(wins),
        "losses": int(losses),
        "neutrals": int(neutrals),
        "winrate_pct": wins / n * 100,
        "avg_pl_pct": avg_pl,
        "total_return_pct": total_ret_pct,
        "max_drawdown_pct": max_dd_pct,
        "sharpe_like": sharpe_like,
        "n_bars_evaluated": len(df),
        "trade_rate": n / len(df) * 100,
    }

    return trades_df, equity, stats


def walk_forward_chunks(df, bundle, n_chunks=4, **strategy_params):
    """Split df into n equal time-chunks, run backtest on each. Returns DataFrame summary."""
    chunk_size = len(df) // n_chunks
    rows = []
    for k in range(n_chunks):
        start = k * chunk_size
        end = (k + 1) * chunk_size if k < n_chunks - 1 else len(df)
        chunk = df.iloc[start:end].reset_index(drop=True)
        if len(chunk) < 10:
            continue
        _, _, stats = run_backtest(chunk, bundle, **strategy_params)
        if not stats:
            rows.append({
                "Chunk": k + 1,
                "From": str(chunk["Datetime (UTC)"].iloc[0]),
                "To": str(chunk["Datetime (UTC)"].iloc[-1]),
                "Trades": 0, "Winrate %": float("nan"),
                "Total Return %": 0.0, "Max DD %": 0.0,
            })
            continue
        rows.append({
            "Chunk": k + 1,
            "From": str(chunk["Datetime (UTC)"].iloc[0]),
            "To": str(chunk["Datetime (UTC)"].iloc[-1]),
            "Trades": stats["trades"],
            "Winrate %": stats["winrate_pct"],
            "Total Return %": stats["total_return_pct"],
            "Max DD %": stats["max_drawdown_pct"],
        })
    return pd.DataFrame(rows)


# ===========================================================================
# PORTFOLIO AGGREGATION
# ===========================================================================
def portfolio_backtest(per_ticker_trades, equal_weight=True):
    """Aggregate per-ticker trade logs into a single portfolio.

    per_ticker_trades : dict[ticker_name -> trades_df from run_backtest_multi_tp]

    Strategy: each trade uses 1/N of total equity (equal-weight allocation).
    Trades are processed chronologically; equity compounds.

    Returns (combined_trades_df, equity_series, stats).
    """
    if not per_ticker_trades:
        return pd.DataFrame(), pd.Series(dtype=float), {}

    n_tickers = len(per_ticker_trades)
    weight = 1.0 / n_tickers if equal_weight else 1.0

    all_trades = []
    for ticker, td in per_ticker_trades.items():
        if td is None or len(td) == 0:
            continue
        td = td.copy()
        td["Ticker"] = ticker
        all_trades.append(td)
    if not all_trades:
        return pd.DataFrame(), pd.Series(dtype=float), {}

    combined = pd.concat(all_trades, ignore_index=True)
    combined["Datetime (UTC)"] = pd.to_datetime(combined["Datetime (UTC)"])
    combined = combined.sort_values("Datetime (UTC)").reset_index(drop=True)

    # Sequential compound: each trade uses `weight` fraction of current equity
    equity = 1.0
    eq_history = []
    for _, t in combined.iterrows():
        trade_pnl_frac = (t["P/L %"] / 100.0) * weight
        equity *= (1 + trade_pnl_frac)
        eq_history.append(equity)
    combined["Portfolio Equity"] = eq_history

    eq_series = pd.Series(eq_history, index=combined["Datetime (UTC)"])

    # Running max & drawdown
    running_max = eq_series.cummax()
    drawdown = (eq_series - running_max) / running_max
    max_dd_pct = drawdown.min() * 100

    total_ret_pct = (eq_series.iloc[-1] - 1) * 100
    n_trades = len(combined)
    wins = (combined["P/L %"] > 0).sum()
    losses = (combined["P/L %"] < 0).sum()

    # Per-ticker contribution to final equity (sum of weighted log returns -> approx)
    per_ticker = {}
    for tk, td in per_ticker_trades.items():
        if td is None or len(td) == 0:
            per_ticker[tk] = {"trades": 0, "contribution_pct": 0.0, "avg_pl": 0.0}
            continue
        # Approximate contribution: weight * sum of pnl% (in pct of total portfolio)
        per_ticker[tk] = {
            "trades": len(td),
            "avg_pl": td["P/L %"].mean(),
            "contribution_pct": (td["P/L %"].sum() / 100.0) * weight * 100,  # rough
            "winrate": (td["P/L %"] > 0).mean() * 100,
        }

    stats = {
        "n_tickers": n_tickers,
        "weight_per_trade": weight,
        "trades": n_trades,
        "wins": int(wins),
        "losses": int(losses),
        "winrate_pct": wins / n_trades * 100 if n_trades > 0 else 0,
        "total_return_pct": total_ret_pct,
        "max_drawdown_pct": max_dd_pct,
        "per_ticker": per_ticker,
    }
    return combined, eq_series, stats


# ===========================================================================
# CONFLUENCE DETECTOR
# ===========================================================================
def detect_confluence(ticker_signal_data, min_count=3, window_hours=1):
    """
    Detect "confluence" events: multiple tickers signalling same direction in close time window.

    ticker_signal_data : dict[ticker -> DataFrame] each with columns:
        - 'Datetime (UTC)' (datetime)
        - 'Direction' (LONG/SHORT)
        - 'tradable' (bool, optional — if missing, assumed True)
        - 'Confidence' (float, optional)

    Returns DataFrame of confluence events:
        Datetime | Direction | Count | Tickers | Avg Confidence
    """
    rows = []
    for ticker, df in ticker_signal_data.items():
        if df is None or len(df) == 0:
            continue
        for _, r in df.iterrows():
            if "tradable" in r and not r["tradable"]:
                continue
            rows.append({
                "Datetime": pd.Timestamp(r["Datetime (UTC)"]),
                "Ticker": ticker,
                "Direction": r["Direction"],
                "Confidence": r.get("Confidence", float("nan")),
            })
    if not rows:
        return pd.DataFrame()

    all_signals = pd.DataFrame(rows).sort_values("Datetime").reset_index(drop=True)

    # Group by exact hour timestamp (1h candles already align)
    events = []
    for (dt, direction), group in all_signals.groupby(["Datetime", "Direction"]):
        if len(group) >= min_count:
            events.append({
                "Datetime (UTC)": dt,
                "Direction": direction,
                "Count": len(group),
                "Tickers": ", ".join(sorted(group["Ticker"].tolist())),
                "Avg Confidence %": group["Confidence"].mean() * 100,
                "Min Confidence %": group["Confidence"].min() * 100,
            })
    if not events:
        return pd.DataFrame()
    return pd.DataFrame(events).sort_values("Datetime (UTC)", ascending=False).reset_index(drop=True)
