import argparse
import pathlib
import sys
from datetime import datetime

import pandas as pd

SRC_DIR = pathlib.Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.XGBoost import XGBoostModel, compute_features, load_data
from backtest import prepare_multitimeframe_proxy_signals, run_bracket_backtest, plot_backtest_positions

parent_dir = SRC_DIR.parent
dataset_dir = parent_dir / "datapoints"


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a market prediction model")
    parser.add_argument("--symbol", type=str, default="BTC", help="Trading symbol (e.g., BTC)")
    parser.add_argument("--timeframe", type=str, default="1h", help="Candle timeframe (e.g., 1h)")
    parser.add_argument("--test-end", type=str, default=None, help="Optional UTC cutoff for completed training and test candles",)
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
    else:
        dev_source = source_data

    if len(dev_source) < 100:
        raise ValueError(f"Not enough candles to train XGBoost model ({len(dev_source)} rows)")

    # 2. Chronological train/test split inside dev_source
    split_at = int(len(dev_source) * 0.80)
    train_data = dev_source.iloc[:split_at].copy()
    test_data = dev_source.iloc[split_at:].copy()

    # 3. Initialize, engineer features, train, test, and save accuracy report
    backend = XGBoostModel(symbol=args.symbol, timeframe=args.timeframe,)
    backend.feature_engineering(train_data, test_data)
    backend.train()
    backend.test()
    backend.evaluate()

    # The model is fitted only on training candles. The 15-minute test starts
    # at the original higher-timeframe holdout, with no later candles included.
    execution_timeframe = '15m'
    if candle_length <= pd.Timedelta(execution_timeframe) or candle_length % pd.Timedelta(execution_timeframe):
        print("15-minute backtest skipped: model timeframe must be a larger whole multiple of 15 minutes.")
        return
    execution_path = dataset_dir / args.symbol / f"{args.symbol}_{execution_timeframe}.csv"
    if not execution_path.exists():
        print(f"15-minute backtest skipped: data file not found: {execution_path}")
        return
    low = load_data(execution_path)
    cutoff = dev_source.index[-1] + candle_length
    low = low.loc[(low.index >= test_data.index[0]) & (low.index + pd.Timedelta(execution_timeframe) <= cutoff)].copy()
    signals = prepare_multitimeframe_proxy_signals(dev_source, low, backend, args.timeframe, execution_timeframe)
    print(f"Backtest: {args.timeframe} closed-candle signals; {execution_timeframe} execution; "
          "2 ATR stop, 2:1 target-to-stop, 1% equity risk including estimated stop fees, 0.1% per side")
    result = run_bracket_backtest(args.symbol, execution_timeframe, low, backend, test_data.index[0],
                                 prepared_signals=signals, reward_risk=2.5, proxy_distance_mode=True,
                                 risk_fraction=0.04, stop_atr_multiplier=8.5)
    stamp = datetime.now().strftime('%Y%m%dT%H%M%S%f')
    for suffix, days in (('full', None), ('recent_30_days', 30)):
        path = parent_dir / 'results' / f'backtest_{args.symbol}_{args.timeframe}_{execution_timeframe}_{suffix}_{stamp}.png'
        plot_backtest_positions(signals, result, path, execution_timeframe, recent_days=days)
        print(f"Backtest chart saved: {path}")

if __name__ == "__main__":
    main()
