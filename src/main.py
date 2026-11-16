import argparse
import json
import pandas as pd
from data import Data
import pathlib
from ML_interface import Model, ML
from models.logistic_regression import LogisticRegressionModel

parent_dir = pathlib.Path(__file__).parent.parent
dataset_dir = parent_dir / "datapoints"
symbols = [p.name for p in pathlib.Path(dataset_dir).glob("*") if p.is_dir()]
timeframes = ["1m", "5m", "15m", "30m", "1h"]

TRANSFORMER_ABLATION_STAGES = (
    ("1h_only", ()),
    ("1h_plus_4h", ("4h",)),
    ("1h_plus_4h_15m", ("4h", "15m")),
    ("1h_plus_4h_15m_5m", ("4h", "15m", "5m")),
    ("1h_plus_4h_15m_5m_1m", ("4h", "15m", "5m", "1m")),
)
AUXILIARY_COLUMNS = [
    "time", "open", "high", "low", "close", "volume", "volume_delta",
    "atr_14", "macd_histogram", "cci",
]


def load_auxiliary_frame(symbol, auxiliary_timeframe, primary_timeframe):
    """Load only the auxiliary columns used by the selected feature list.

    For a faster-than-primary source, only its final completed candle in each
    primary interval can affect a primary close.  Reducing chunks to those rows
    preserves the causal as-of join while preventing BTC_1m.csv from occupying
    several gigabytes of memory.
    """
    path = dataset_dir / symbol / f"{symbol}_{auxiliary_timeframe}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing multi-timeframe source file: {path}")

    auxiliary_duration = pd.Timedelta(auxiliary_timeframe)
    primary_duration = pd.Timedelta(primary_timeframe)
    if auxiliary_duration >= primary_duration:
        return (
            pd.read_csv(path, usecols=AUXILIARY_COLUMNS)
            .dropna(subset=AUXILIARY_COLUMNS)
            .sort_values("time")
            .reset_index(drop=True)
        )

    reduced_chunks = []
    rolling_context = None
    for chunk in pd.read_csv(path, usecols=AUXILIARY_COLUMNS, chunksize=250_000):
        chunk = chunk.dropna(subset=AUXILIARY_COLUMNS)
        chunk["time"] = pd.to_datetime(chunk["time"], utc=True)
        # A 20-candle rolling median must be calculated at the original
        # auxiliary frequency, before reducing 1m/5m rows to hourly snapshots.
        context_length = 0 if rolling_context is None else len(rolling_context)
        working = (
            chunk
            if rolling_context is None
            else pd.concat([rolling_context, chunk], ignore_index=True)
        )
        volume = pd.to_numeric(working["volume"], errors="coerce")
        working["relative_volume"] = volume / volume.rolling(
            window=20, min_periods=20
        ).median().replace(0, pd.NA)
        current_rows = working.iloc[context_length:].copy()
        rolling_context = working[AUXILIARY_COLUMNS].tail(19).copy()
        current_rows["_primary_close_bucket"] = (
            current_rows["time"] + auxiliary_duration
        ).dt.floor(primary_timeframe)
        reduced_chunks.append(
            current_rows.sort_values("time")
            .groupby("_primary_close_bucket", as_index=False)
            .tail(1)
        )

    compact = pd.concat(reduced_chunks, ignore_index=True)
    compact = (
        compact.sort_values("time")
        .groupby("_primary_close_bucket", as_index=False)
        .tail(1)
        .drop(columns="_primary_close_bucket")
        .sort_values("time")
        .reset_index(drop=True)
    )
    if compact.empty:
        raise ValueError(f"No usable rows found in auxiliary source: {path}")
    return compact


def run_transformer_timeframe_ablation(args, train_data, test_data):
    """Run the cumulative timeframe experiment with test data held out per stage."""
    from models.transformerModel import TransformerModel
    if args.timeframe != "1h":
        raise ValueError("--timeframe-ablation currently targets a 1h prediction horizon")

    results_dir = parent_dir / "results"
    summary_path = results_dir / f"transformer_{args.symbol}_{args.timeframe}_timeframe_ablation.json"
    auxiliary_cache = {}
    summary = []

    for stage_name, auxiliary_timeframes in TRANSFORMER_ABLATION_STAGES:
        for auxiliary_timeframe in auxiliary_timeframes:
            if auxiliary_timeframe not in auxiliary_cache:
                print(f"Loading {auxiliary_timeframe} data for {stage_name}...")
                auxiliary_cache[auxiliary_timeframe] = load_auxiliary_frame(
                    args.symbol, auxiliary_timeframe, args.timeframe
                )

        backend = TransformerModel(
            args.symbol,
            timeframe=args.timeframe,
            lookback="auto",
            target_threshold="auto",
            multi_timeframes=auxiliary_timeframes,
            search_strategy="progressive",
            screening_trials=args.screening_trials,
            finalist_trials=args.finalist_trials,
            screening_epochs=args.screening_epochs,
            results_path=results_dir / f"transformer_{args.symbol}_{args.timeframe}_{stage_name}.json",
        )
        auxiliary_data = {
            timeframe: auxiliary_cache[timeframe] for timeframe in auxiliary_timeframes
        }
        backend.feature_engineering(train_data, test_data, auxiliary_data=auxiliary_data)
        backend.train()
        backend.test()
        result = backend.evaluate()
        summary.append(
            {
                "stage": stage_name,
                "multi_timeframes": list(auxiliary_timeframes),
                "lookback": result["lookback"],
                "target_threshold_atr_fraction": result["target_threshold_atr_fraction"],
                "transformer_config": result["transformer_config"],
                "validation_balanced_acc": next(
                    trial["validation_balanced_acc"]
                    for trial in result["hyperparameter_trials"]
                    if trial["search_round"] == "finalist"
                    and trial["lookback"] == result["lookback"]
                    and trial["target_setting"] == backend._current_target_setting()
                    and trial["transformer_config"] == result["transformer_config"]
                ),
                "test_acc": result["test_acc"],
                "test_balanced_acc": result["test_balanced_acc"],
                "majority_baseline_acc": result["majority_baseline_acc"],
                "results_path": str(backend.results_path),
            }
        )
        print(
            f"{stage_name}: test accuracy={result['test_acc']:.2%}, "
            f"balanced accuracy={result['test_balanced_acc']:.2%}"
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as file:
        json.dump(
            {
                "objective": (
                    "Predict the next completed 1h candle direction from information "
                    "available at the current 1h close."
                ),
                "selection_rule": (
                    "Progressive validation search: screen candidate configurations on "
                    "chronological validation, retrain finalists, then test once per stage."
                ),
                "hyperparameter_candidates": {
                    "lookback": list(TransformerModel.DEFAULT_LOOKBACK_CANDIDATES),
                    "target_threshold_atr_fraction": list(TransformerModel.DEFAULT_TARGET_THRESHOLD_CANDIDATES),
                    "embedding_dimension": list(TransformerModel.DEFAULT_EMBEDDING_DIMENSIONS),
                    "attention_heads": list(TransformerModel.DEFAULT_ATTENTION_HEADS),
                    "encoder_blocks": list(TransformerModel.DEFAULT_ENCODER_BLOCKS),
                    "dropout": list(TransformerModel.DEFAULT_DROPOUTS),
                },
                "stages": summary,
            },
            file,
            indent=2,
        )
    print(f"Timeframe-ablation summary written to {summary_path}")
    return summary


def main():

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", type=str, help="BTC, XAU, etc.")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe: (1m, 5m, 15m, 30m, 1h, 4h)") 
    parser.add_argument("--test-end", default=None, help="For XGBoost: latest candle close allowed in training/test data (UTC); later candles are reserved for forward backtesting")
    parser.add_argument("--lookback", default="auto", help="Sequence window length, or 'auto' to select from validation performance")
    parser.add_argument("--target-threshold", default="auto", help="ATR fraction for the target, or 'auto' to select from validation performance")
    parser.add_argument("--lstm-units", default="auto", help="LSTM memory units, or 'auto' to select from validation performance")
    parser.add_argument("--confidence-threshold", default="auto", help="Signal-confidence cutoff, or 'auto' to select on validation data")
    parser.add_argument("--model", choices=["logistic", "gru", "lstm", "transformer", "logistic-baseline", "xgboost"], default="logistic")
    # Target for lstm / transformer / logistic-baseline
    parser.add_argument("--target", choices=["direction", "triple_barrier"], default="direction", help="direction: next candle body; triple_barrier: does a long trade hit TP before SL")
    parser.add_argument("--sl-atr", default="1,1.5,2", help="Triple barrier: SL distances in ATRs to search, comma separated")
    parser.add_argument("--alpha", default="1,1.5,2", help="Triple barrier: reward:risk ratios (TP / SL) to search, comma separated")
    parser.add_argument("--horizon", type=int, default=30, help="Triple barrier: close the trade after this many candles")
    parser.add_argument("--cost-pct", type=float, default=0.001, help="Round-trip cost as a fraction of price (0.001 = 0.1%%, an assumption: check your broker)")
    parser.add_argument("--aux-timeframes", default="15m,4h", help="Extra timeframes merged as features, comma separated; '' for none")
    parser.add_argument("--timeframe-ablation", action="store_true", help="Run the cumulative 1h/4h/15m/5m/1m Transformer experiment")
    parser.add_argument("--screening-trials", type=int, default=32, help="Progressive Transformer candidates screened per timeframe stage")
    parser.add_argument("--finalist-trials", type=int, default=6, help="Screening finalists retrained per timeframe stage")
    parser.add_argument("--screening-epochs", type=int, default=8, help="Epoch budget for each progressive screening trial")
    args = parser.parse_args()
    if args.test_end and args.model != "xgboost":
        parser.error("--test-end is currently supported only with --model xgboost")
    symbol = args.symbol
    timeframe = args.timeframe

    ## 1- Dataset preparation
    csv_path = dataset_dir / args.symbol / f"{args.symbol}_{args.timeframe}.csv"
    if args.model == "xgboost":
        # Data drops rows with missing values in any precomputed indicator;
        # the source XGBoost method starts with raw OHLCV instead.
        from models.XGBoost import XGBoostModel, load_data as load_xgboost_data
        source_data = load_xgboost_data(csv_path)
        # Timestamps identify candle opens; exclude a still-forming candle.
        current_utc = pd.Timestamp.now(tz="UTC")
        candle_length = pd.Timedelta(timeframe)
        source_data = source_data.loc[source_data.index + candle_length <= current_utc].copy()
        cutoff = pd.Timestamp("2025-07-21", tz="UTC")
        all_data = source_data.loc[source_data.index >= cutoff].copy()
        if args.test_end:
            test_end = pd.to_datetime(args.test_end, utc=True, errors="raise")
            all_data = all_data.loc[all_data.index + candle_length <= test_end].copy()
        if all_data.empty:
            raise ValueError(f"No completed XGBoost candles in the selected training/test period in {csv_path}")
        print(f"XGBoost data from 2025-05-01 UTC: {len(all_data)} candles; first={all_data.index[0]}, last={all_data.index[-1]}")
    else:
        dataObj = Data(args.symbol, str(csv_path), str(csv_path), str(csv_path), args.timeframe)
        all_data = dataObj.get_train_data().sort_values("time").reset_index(drop=True)

    split_at = int(len(all_data) * 0.80)
    train_data = all_data.iloc[:split_at].copy()
    test_data = all_data.iloc[split_at:].copy()

    if args.timeframe_ablation:
        if args.model != "transformer":
            raise ValueError("--timeframe-ablation requires --model transformer")
        run_transformer_timeframe_ablation(args, train_data, test_data)
        return
    
    ## 2- Model initialization
    if args.model == "gru":
        from models.RNNGRU import RNNGRU
        backend: Model = RNNGRU(args.symbol, timeframe=args.timeframe, random_state=42)
    elif args.model == "logistic":
        backend = LogisticRegressionModel(args.symbol, timeframe=args.timeframe, max_iter=100000, random_state=42)
    elif args.model == "xgboost":
        backend = XGBoostModel(symbol, timeframe=timeframe)
    else:
        lookback = args.lookback if args.lookback == "auto" else int(args.lookback)
        target_threshold = (args.target_threshold if args.target_threshold == "auto" else float(args.target_threshold))
        confidence_threshold = (args.confidence_threshold if args.confidence_threshold == "auto" else float(args.confidence_threshold))
        # Settings shared by the three sequence models, so they all see the same labels
        target_options = dict(
            target=args.target,
            sl_atr_candidates=[float(value) for value in args.sl_atr.split(",")],
            alpha_candidates=[float(value) for value in args.alpha.split(",")],
            horizon=args.horizon,
            cost_pct=args.cost_pct,
            multi_timeframes=tuple(value for value in args.aux_timeframes.split(",") if value),
        )
        if args.model == "lstm":
            from models.LSTMModel import LSTMModel
            lstm_units = args.lstm_units if args.lstm_units == "auto" else int(args.lstm_units)
            backend = LSTMModel(symbol, timeframe=timeframe, lookback=lookback, target_threshold=target_threshold, lstm_units=lstm_units, confidence_threshold=confidence_threshold, **target_options)
        elif args.model == "logistic-baseline":
            from models.logistic_baseline import LogisticBaseline
            backend = LogisticBaseline(symbol, timeframe=timeframe, lookback=lookback, target_threshold=target_threshold, confidence_threshold=confidence_threshold, **target_options)
        else:
            from models.transformerModel import TransformerModel
            backend = TransformerModel(
                symbol,
                timeframe=timeframe,
                lookback=lookback,
                target_threshold=target_threshold,
                confidence_threshold=confidence_threshold,
                search_strategy="progressive",
                screening_trials=args.screening_trials,
                finalist_trials=args.finalist_trials,
                screening_epochs=args.screening_epochs,
                **target_options,
            )

    ml = ML(backend)  # front end: only knows the Model interface

    ## 3- feature engineering
    if args.model in {"lstm", "transformer", "logistic-baseline"}:
        auxiliary_data = {}
        for auxiliary_timeframe in backend.multi_timeframes:
            auxiliary_data[auxiliary_timeframe] = load_auxiliary_frame(
                args.symbol, auxiliary_timeframe, args.timeframe
            )
        ml.feature_engineering(train_data, test_data, auxiliary_data=auxiliary_data)
    else:
        ml.feature_engineering(train_data, test_data)
    
    ## 4- Model training, evaluation, and testing
    ml.train()

    ## 5- Model testing
    ml.test()

    ## 6- Model evaluation
    ml.evaluate()

    ## 7- Backtest the trained XGBoost model on the selected candle data.
    if args.model == "xgboost":
        # Use the most recent candle in the frontend's chronological test
        # partition, not the last row of XGBoost's randomly shuffled X_test.
        last_test_time = test_data.index.max()
        backtest_data = source_data.loc[source_data.index > last_test_time].copy()
        if backtest_data.empty:
            print(f"No completed candles after the last test candle ({last_test_time}) through {current_utc}; forward backtest skipped.")
            print("Update the dataset or reserve a later period with --test-end (UTC).")
        else:
            from backtest import run_backtest

            print(f"Forward backtest: {len(backtest_data)} candles; first={backtest_data.index[0]}, last={backtest_data.index[-1]} (UTC)")
            print("Backtrader indicators require a warm-up before the first signal.")
            result = run_backtest(symbol, timeframe, backend.model, data=backtest_data)
            if result is not None:
                print(f"Initial value: ${result['initial_value']:.2f}")
                print(f"Final value: ${result['final_value']:.2f}")
                print(f"Total return: {result['total_return']:.2f}%")
    

if __name__ == "__main__":
    main()
