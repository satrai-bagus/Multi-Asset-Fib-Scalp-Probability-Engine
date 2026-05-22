"""
Master pipeline: build dataset + train model for one ticker.
Usage:
    python build_pipeline.py BTCUSDT
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
PARENT = HERE.parent
SQZMOM = PARENT / "sqzmom_export.py"

if len(sys.argv) < 2:
    print("Usage: python build_pipeline.py SYMBOL")
    sys.exit(1)

symbol = sys.argv[1].upper()
dataset = HERE / "data" / f"dataset_{symbol}_18m.xlsx"

print(f"\n{'='*60}\n  STEP 1: Fetch + build features for {symbol}\n{'='*60}")
ret = subprocess.run([
    sys.executable, str(SQZMOM),
    "--symbol", symbol,
    "--start", "2024-11-20",
    "--end", "2026-05-20",
    "--out", str(dataset),
], cwd=str(PARENT))
if ret.returncode != 0:
    print(f"FAILED to build dataset for {symbol}")
    sys.exit(1)

print(f"\n{'='*60}\n  STEP 2: Train model for {symbol}\n{'='*60}")
ret = subprocess.run([
    sys.executable, str(HERE / "train_model.py"),
    "--symbol", symbol,
], cwd=str(HERE))
if ret.returncode != 0:
    print(f"FAILED to train model for {symbol}")
    sys.exit(1)

print(f"\n[OK] Pipeline complete for {symbol}")
