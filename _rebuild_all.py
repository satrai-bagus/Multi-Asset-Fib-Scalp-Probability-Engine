"""Batch-rebuild all active tickers with new BTC features."""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
TICKERS = ["BTCUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
           "AVAXUSDT", "LINKUSDT", "POLUSDT",
           "SUIUSDT", "ARBUSDT", "APTUSDT", "AAVEUSDT", "FETUSDT"]
# ETHUSDT already done — skipped

results = {}
for i, t in enumerate(TICKERS, 1):
    print(f"\n{'#'*70}\n# [{i}/{len(TICKERS)}] {t}\n{'#'*70}")
    t0 = time.time()
    ret = subprocess.run(
        [sys.executable, str(HERE / "build_pipeline.py"), t],
        cwd=str(HERE),
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    dt = time.time() - t0
    if ret.returncode != 0:
        results[t] = f"FAILED ({dt:.0f}s)"
        print(f"FAILED: {ret.stderr[-500:]}")
    else:
        # extract AUC summary from output
        auc_lines = [l for l in ret.stdout.split("\n") if "AUC=" in l]
        results[t] = f"OK ({dt:.0f}s) | " + " | ".join(l.strip() for l in auc_lines[:2])
        print(f"OK ({dt:.0f}s)")

print(f"\n\n{'='*70}\n  REBUILD SUMMARY\n{'='*70}")
for t, r in results.items():
    print(f"  {t:<10} {r}")
