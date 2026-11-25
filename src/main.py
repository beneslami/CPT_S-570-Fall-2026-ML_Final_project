import argparse
import pathlib
import sys

import pandas as pd

SRC_DIR = pathlib.Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backtest import run_bracket_backtest
from models.XGBoost import XGBoostModel, load_data

parent_dir = SRC_DIR.parent
dataset_dir = parent_dir / "datapoints"


def main():
    parser = argparse.ArgumentParser(description="Train and forward-backtest the XGBoost bracket classifier")
    parser.add_argument("--symbol", type=str, default="BTC", help="Trading symbol (e.g., BTC)")
    parser.add_argument("--timeframe", type=str, default="1h", help="Candle timeframe (e.g., 1h)")
    parser.add_argument(
        "--test-end",
        type=str,
        default=None,
        help="Optional UTC cutoff for train/test data; candles after this are used for forward backtesting",
    )
    parser.add_argument("--commission", type=float, default=0.001, help="Per-side commission rate (e.g., 0.001 = 0.1)")
    args = parser.parse_args()

    csv_path = dataset_dir / args.symbol / f"{args.symbol}_{args.timeframe}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    # 1. Load completed candles strictly up to current UTC (and optional --test-end)
    source_data = load_data(csv_path)
    current_utc = pd.Timestamp.now(tz="UTC")
    candle_length = pd.Timedelta(args.timeframe)
    source_data = source_data.loc[source_data.index + candle_length <= current_utc].copy()

    if args.test_end:
        test_end = pd.to_datetime(args.test_end, utc=True, errors="raise")
        dev_source = source_data.loc[source_data.index + candle_length <= test_end].copy()
        fwd_source = source_data.loc[source_data.index >= test_end].copy()
    else:
        split_idx = int(len(source_data) * 0.85)
        dev_source = source_data.iloc[:split_idx].copy()
        fwd_source = source_data.iloc[split_idx:].copy()

    if len(dev_source) < 100:
        raise ValueError(f"Not enough candles to train XGBoost model ({len(dev_source)} rows)")

    # 2. Chronological train/test split inside dev_source
    split_at = int(len(dev_source) * 0.80)
    train_data = dev_source.iloc[:split_at].copy()
    test_data = dev_source.iloc[split_at:].copy()

    # 3. Initialize, engineer features, train, test, and save accuracy report
    backend = XGBoostModel(
        symbol=args.symbol,
        timeframe=args.timeframe,
    )
    backend.feature_engineering(train_data, test_data)
    backend.train()
    backend.test()
    backend.evaluate()

    # 4. Warm indicators with prior candles, but trade only the forward slice.
    if fwd_source.empty:
        print("No completed forward candles after the cutoff; forward backtest skipped.")
        return
    run_bracket_backtest(
        symbol=args.symbol,
        timeframe=args.timeframe,
        ohlcv_df=source_data,
        backend=backend,
        start_time=fwd_source.index[0],
        commission=args.commission,
    )

if __name__ == "__main__":
    main()
