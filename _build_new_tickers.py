"""Batch-build 5 new tickers for futures-scalp expansion."""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
NEW_TICKERS = ["NEARUSDT", "ATOMUSDT", "OPUSDT", "INJUSDT", "SEIUSDT"]

results = {}
t_start = time.time()
for i, tk in enumerate(NEW_TICKERS, 1):
    print(f"\n{'#'*60}\n# [{i}/{len(NEW_TICKERS)}] {tk}\n{'#'*60}")
    t0 = time.time()
    ret = subprocess.run(
        [sys.executable, str(HERE / "build_pipeline.py"), tk],
        cwd=str(HERE),
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    dt = time.time() - t0
    if ret.returncode != 0:
        results[tk] = f"FAILED ({dt:.0f}s)"
        print(f"FAILED: {ret.stderr[-300:]}")
    else:
        auc_lines = [l for l in ret.stdout.split("\n") if "AUC=" in l]
        results[tk] = f"OK ({dt:.0f}s) | " + " | ".join(l.strip() for l in auc_lines[:2])
        print(f"OK ({dt:.0f}s)")

print(f"\n{'='*60}\n  SUMMARY  ({time.time()-t_start:.0f}s total)\n{'='*60}")
for t, r in results.items():
    print(f"  {t:<12} {r}")
