"""
Candle-stream simulator.

Reads a CSV of candles once, then replays it ONE CANDLE AT A TIME, oldest first,
through a black-box model:

        candle t  ->  model.learn(candle t)  ->  model.predict(candle t)  ->  output t
        candle t+1 -> ...

Guarantees
----------
- The model never receives the DataFrame, only one Candle object at a time.
  When it handles candle t it has seen candles 0..t and nothing after.
  If it needs history (indicators, swings, ...) it must keep that itself,
  exactly as it will have to when it runs live.
- Candles are immutable (frozen dataclass), so the model cannot change the data.
- Candles arrive in strictly increasing time order; broken rows are skipped
  and counted (missing/non-positive prices, high < low, duplicate times).
- Every candle carries ALL columns of its CSV row (candle["rsi_14"], candle.data).
  Columns with FUTURE information (ichimoku_chikou = close 26 candles ahead)
  are listed in the JSON as a warning: a live system can never have them.
- The model is pluggable: any subclass of RegimeModel (see regime_model.py),
  chosen on the command line with --model file:ClassName.

Going live later means replacing `candle_stream(read_candles(...))` with another generator
that yields `Candle` objects from the broker; `run_stream` and the model stay the same.

Usage
-----
    python stream_sim.py datapoints/XAU/1h_tmp.csv                                   # NullModel
    python stream_sim.py datapoints/XAU/1h_tmp.csv --model linreg_model:LinearRegressionModel
    python stream_sim.py datapoints/XAU/1h_tmp.csv --delay 0.05                     # slow replay
"""

import argparse
import importlib
import json
import math
import os
import time
from collections import Counter
from datetime import datetime, timezone
from types import MappingProxyType

import pandas as pd

from regime_model import Candle, RegimeModel

REQUIRED_COLUMNS = ["time", "open", "high", "low", "close", "volume"]
FUTURE_COLUMNS = {"ichimoku_chikou"}        # computed from future candles -> never allowed

# ----------------------------------------------------------------------------
#  Source: CSV file -> stream of Candle objects
# ----------------------------------------------------------------------------
def read_candles(path, stats=None):
    stats = stats if stats is not None else Counter()
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} has no column(s) {missing}")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    stats["rows_in_file"] = len(df)
    if not df["time"].is_monotonic_increasing:
        stats["file_was_unsorted"] = 1
        df = df.sort_values("time", kind="stable").reset_index(drop=True)
    return df


def candle_stream(df, stats=None):
    stats = stats if stats is not None else Counter()
    columns = list(df.columns)
    has_volume = "volume" in df.columns
    index, last_time = 0, None
 
    for values in df.itertuples(index=False, name=None):
        row = dict(zip(columns, values))
        t, o, h, l, c, v = (row[k] for k in REQUIRED_COLUMNS)
        if any(pd.isna(v) for v in (t, o, h, l, c)):
            stats["skipped_missing_value"] += 1
            continue
        if min(o, h, l, c) <= 0:
            stats["skipped_non_positive_price"] += 1
            continue
        if h < l or h < max(o, c) or l > min(o, c):
            stats["skipped_inconsistent_ohlc"] += 1
            continue
        if last_time is not None and t <= last_time:
            stats["skipped_duplicate_time"] += 1
            continue
        yield Candle(index, t, float(o), float(h), float(l), float(c), float(v), MappingProxyType(row))
        last_time = t
        index += 1

def load_model(spec):
    """
    "linreg_model:LinearRegressionModel" -> imports linreg_model.py and
    returns a new LinearRegressionModel(). The class must subclass RegimeModel.
    """
    module_name, _, class_name = spec.partition(":")
    if not class_name:
        raise ValueError(f"--model must look like file:ClassName, got '{spec}'")
    cls = getattr(importlib.import_module(module_name), class_name)
    if not (isinstance(cls, type) and issubclass(cls, RegimeModel)):
        raise TypeError(f"{spec} is not a subclass of RegimeModel (regime_model.py)")
    return cls()

def run_stream(source, model, on_output=None, delay=0.0, progress_every=100_000):
    count = 0
    for candle in source:
        model.learn(candle)                  # 1. candle goes in, model learns
        regime = model.predict(candle)       # 2. model gives its output for this candle
        if on_output is not None:
            on_output(candle, regime)        # 3. candle goes out
        count += 1
        if progress_every and count % progress_every == 0:
            print(f"  ... {count:,} candles  (now at {candle.time})")
        if delay:
            time.sleep(delay)
    return count

def simulate(csv_path, model_spec="regime_model:NullModel", delay=0.0, save_outputs=True, mode="train"):
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    stats = Counter()
    outputs = []
    regimes = Counter()
    first_last = {}
 
    def collect(candle, regime):
        regimes[str(regime)] += 1
        first_last.setdefault("first", candle.time)
        first_last["last"] = candle.time
        if save_outputs:
            outputs.append({"time": candle.time.isoformat(), "close": candle.close, "regime": regime})
 
    # Step 1: read the whole file
    t0 = time.perf_counter()
    df = read_candles(csv_path, stats)
    read_seconds = time.perf_counter() - t0
    future_cols = sorted(FUTURE_COLUMNS.intersection(df.columns))
 
    # Step 2: stream it through the model
    t0 = time.perf_counter()
    source = candle_stream(df, stats=stats)
    model = load_model(model_spec)
    print(f"[{stem}] model: {model_spec}")
    n = run_stream(source, model, on_output=collect, delay=delay)
    seconds = time.perf_counter() - t0
 
    print(f"[{stem}] candles streamed: {n:,} in {seconds:.1f} s ({n / max(seconds, 1e-9):,.0f} candles/s)")
    skipped = {k: v for k, v in stats.items() if k.startswith("skipped")}
    if skipped:
        print(f"[{stem}] skipped rows: {skipped}")
 
    results = {
        "file": csv_path,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {"delay_seconds": delay},
        "future_columns_warning": future_cols,
        "mode": mode,
        "model": model_spec,
        "stream": {
            "rows_in_file": stats["rows_in_file"],
            "candles_streamed": n,
            "skipped": skipped,
            "file_was_unsorted": bool(stats["file_was_unsorted"]),
            "read_seconds": round(read_seconds, 3),
            "stream_seconds": round(seconds, 3),
        },
        "outputs_summary": dict(regimes),
    }
    if save_outputs:
        results["outputs"] = outputs
 
    out_path = os.path.join(os.path.dirname(os.path.abspath(csv_path)), f"{stem}_stream_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{stem}] results saved to {out_path}\n")
    return results

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Replay a candle CSV one candle at a time through a model.")
    ap.add_argument("csv", nargs="+", help="one or more CSV files from datapoints/")
    ap.add_argument("--mode", type=str, help="train or test", default="train")
    ap.add_argument("--model", default="regime_model:NullModel", help="model to run, as file:ClassName (default: regime_model:NullModel)")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to wait between candles")
    ap.add_argument("--no-outputs", action="store_true", help="do not store the per-candle outputs in the JSON (smaller file)")
    a = ap.parse_args()
    for path in a.csv:
        simulate(path, a.model, a.delay, save_outputs=not a.no_outputs, mode=a.mode)
