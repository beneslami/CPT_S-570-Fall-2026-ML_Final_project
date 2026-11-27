import argparse
import os
import pathlib
import sys
from datetime import datetime

# Prevent OpenMP/BLAS thread-pool collisions on macOS (xgboost + torch/keras + sklearn).
for _env_name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_env_name, "1")

import pandas as pd

SRC_DIR = pathlib.Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.XGBoost import XGBoostBracketClassifier, XGBoostModel, load_data
from backtest import (
    plot_backtest_positions,
    prepare_multitimeframe_proxy_signals,
    run_bracket_backtest,
    run_consensus_ev_backtest,
)

parent_dir = SRC_DIR.parent
dataset_dir = parent_dir / "datapoints"


def _load_auxiliary_data(
    symbol: str,
    primary_timeframe: str,
    cutoff_utc: pd.Timestamp,
    data_dir: pathlib.Path = dataset_dir,
    candidate_tfs: tuple[str, ...] = ("5m", "15m", "1h", "4h"),
) -> dict:
    """Load available auxiliary timeframes strictly up to cutoff_utc."""
    aux = {}
    for tf in candidate_tfs:
        if tf == primary_timeframe:
            continue
        aux_path = data_dir / symbol / f"{symbol}_{tf}.csv"
        if not aux_path.exists():
            continue
        df = load_data(aux_path)
        td = pd.Timedelta(tf)
        df = df.loc[df.index + td <= cutoff_utc].copy()
        if not df.empty:
            aux[tf] = df
    return aux


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate directional market prediction models")
    parser.add_argument(
        "--model",
        type=str,
        default="consensus",
        choices=["consensus", "xgboost_bracket", "multi_rr_bracket", "direct_r_bracket", "lstm", "logistic", "xgboost_price"],
        help="Model backend to run",
    )
    parser.add_argument("--symbol", type=str, default="BTC", help="Trading symbol (e.g., BTC)")
    parser.add_argument("--timeframe", type=str, default="1h", help="Candle timeframe (e.g., 1h)")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="datapoints",
        help="Dataset directory relative to project root (e.g., datapoints or datapoints_new)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Prediction horizon in candles for directional target: Close[t+H] > Close[t]",
    )
    parser.add_argument(
        "--min-move-atr",
        type=float,
        default=0.0,
        help="Optional deadzone in ATR fractions (0.0 keeps all non-flat candles)",
    )
    parser.add_argument(
        "--xgb-fvg-ob",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include the 14 sparse FVG/Order-Block features in XGBoost (default: pruned)",
    )
    parser.add_argument(
        "--decision-mode",
        type=str,
        default="static",
        choices=["static", "rolling_center"],
        help="Decision adjustment mode for consensus (static thresholds vs causal rolling logit centering)",
    )
    parser.add_argument(
        "--center-window-bars",
        type=int,
        default=720,
        help="Trailing window in bars for rolling_center mode (default: 720 bars)",
    )
    parser.add_argument(
        "--center-min-periods",
        type=int,
        default=100,
        help="Minimum prior bars required before rolling median activates (default: 100)",
    )
    parser.add_argument(
        "--test-end",
        type=str,
        default=None,
        help="Optional UTC cutoff for completed training and test candles",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run expanding-window walk-forward evaluation for consensus model",
    )
    parser.add_argument(
        "--wf-splits",
        type=int,
        default=3,
        help="Number of walk-forward folds when --walk-forward is enabled",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run EV-gated bracket backtest (consensus) or 15m execution backtest (standalone xgboost)",
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=0.0001,
        help="Broker commission rate per side (default: 0.0001 = 0.01%% per side)",
    )
    parser.add_argument(
        "--min-ev-r",
        type=float,
        default=0.0,
        help="Minimum fee-adjusted expected value in R-multiples required to enter a trade (default: 0.0)",
    )
    parser.add_argument(
        "--max-bars",
        type=int,
        default=32,
        help="Max holding bars for intraday multi_rr_bracket model (default: 32 bars = 8h on 15m)",
    )
    parser.add_argument(
        "--sl-quantile",
        type=float,
        default=0.70,
        help="MAE quantile for Stage-1 dynamic SL model (default: 0.70)",
    )
    parser.add_argument(
        "--refit-days",
        type=int,
        default=180,
        help="Walk-forward periodic refitting interval in days for multi_rr_bracket (default: 180; 0 disables)",
    )
    parser.add_argument(
        "--xgb-reg-profile",
        type=str,
        default="strong",
        choices=["baseline", "strong"],
        help="XGBoost regularization search profile (baseline vs stronger regularization)",
    )
    parser.add_argument(
        "--multi-timeframes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use strictly causal auxiliary timeframe features (e.g., 15m and 4h) when available",
    )
    parser.add_argument(
        "--lstm-ensemble",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use 3-model multi-lookback sequence ensemble (LSTM lb=12, LSTM lb=24, GRU lb=48)",
    )
    parser.add_argument(
        "--ev-mode",
        type=str,
        default="conservative",
        choices=["conservative", "factorized_3class"],
        help="EV gate formula for multi_rr_bracket: conservative lower-bound vs factorized 3-outcome",
    )
    parser.add_argument(
        "--sl-mode",
        type=str,
        default="learned",
        choices=["learned", "fixed_atr"],
        help="Stage-1 stop-loss mode for multi_rr_bracket: learned ATR-MAE quantile vs fixed ATR multiple",
    )
    parser.add_argument(
        "--fixed-sl-atr",
        type=float,
        default=1.5,
        help="Fixed ATR multiple when --sl-mode fixed_atr",
    )
    parser.add_argument(
        "--rr-ratios",
        type=str,
        default="2.0,2.5,3.0,3.5,4.0",
        help="Comma-separated target R:R ratios for multi_rr_bracket (e.g. '2.5' for single-RR baseline)",
    )
    parser.add_argument(
        "--min-ev-floor",
        type=float,
        default=0.12,
        help="Minimum validation EV threshold floor in R for multi_rr_bracket",
    )
    parser.add_argument(
        "--max-comm-r",
        type=float,
        default=0.05,
        help="Maximum allowed round-trip commission drag in R (2*c / sl_pct <= max_comm_r)",
    )
    parser.add_argument(
        "--prob-shrinkage",
        type=float,
        default=0.10,
        help="Winner's-curse probability shrinkage toward prior in [0, 0.5]",
    )
    parser.add_argument(
        "--entry-session",
        type=str,
        default="all",
        choices=["all", "weekdays", "asia", "london", "ny", "london_ny", "ny_rth"],
        help="Entry-and-training session mask applied on continuous real candles (default: all)",
    )

    args = parser.parse_args()

    active_data_dir = pathlib.Path(args.data_dir)
    if not active_data_dir.is_absolute():
        active_data_dir = parent_dir / active_data_dir
    csv_path = active_data_dir / args.symbol / f"{args.symbol}_{args.timeframe}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    source_data = load_data(csv_path)
    current_utc = pd.Timestamp.now(tz="UTC")
    candle_length = pd.Timedelta(args.timeframe)
    source_data = source_data.loc[source_data.index + candle_length <= current_utc].copy()

    if args.test_end:
        test_end = pd.to_datetime(args.test_end, utc=True, errors="raise")
        dev_source = source_data.loc[source_data.index + candle_length <= test_end].copy()
    else:
        dev_source = source_data

    if len(dev_source) < 200:
        raise ValueError(f"Not enough candles to train model ({len(dev_source)} rows)")

    split_at = int(len(dev_source) * 0.80)
    train_data = dev_source.iloc[:split_at].copy()
    test_data = dev_source.iloc[split_at:].copy()
    cutoff_utc = dev_source.index[-1] + candle_length
    aux_tfs = ("1h", "4h") if args.model in ("multi_rr_bracket", "direct_r_bracket") else ("5m", "15m", "4h")
    auxiliary_data = (
        _load_auxiliary_data(
            args.symbol,
            args.timeframe,
            cutoff_utc,
            data_dir=active_data_dir,
            candidate_tfs=aux_tfs,
        )
        if args.multi_timeframes
        else {}
    )

    if args.model == "consensus":
        from models.XGBoostLSTMConsensus import XGBoostLSTMConsensusModel

        backend = XGBoostLSTMConsensusModel(
            symbol=args.symbol,
            timeframe=args.timeframe,
            horizon=args.horizon,
            target_mode="direction",
            min_move_atr=args.min_move_atr,
            xgb_include_fvg_ob=args.xgb_fvg_ob,
            xgb_reg_profile=args.xgb_reg_profile,
            decision_mode=args.decision_mode,
            center_window_bars=args.center_window_bars,
            center_min_periods=args.center_min_periods,
            lstm_ensemble=args.lstm_ensemble,
        )
    elif args.model == "xgboost_bracket":
        backend = XGBoostBracketClassifier(
            symbol=args.symbol,
            timeframe=args.timeframe,
            horizon=args.horizon,
            target_mode="direction",
            min_move_atr=args.min_move_atr,
            include_fvg_ob=args.xgb_fvg_ob,
            reg_profile=args.xgb_reg_profile,
        )
    elif args.model == "lstm":
        from models.LSTMModel import LSTMModel

        lstm_aux_tfs = tuple(
            tf for tf in ("5m", "15m", "4h")
            if tf in auxiliary_data and (
                pd.Timedelta(tf) > candle_length
                or (pd.Timedelta(tf) < candle_length and candle_length % pd.Timedelta(tf) == pd.Timedelta(0))
            )
        )
        backend = LSTMModel(
            symbol=args.symbol,
            timeframe=args.timeframe,
            lookback=24,
            lstm_units=32,
            epochs=40,
            batch_size=64,
            target="direction",
            horizon=args.horizon,
            target_threshold=args.min_move_atr,
            multi_timeframes=lstm_aux_tfs,
            use_sample_weights=False,
        )
    elif args.model == "multi_rr_bracket":
        from models.multi_rr_bracket import MultiRRBracketModel

        parsed_rrs = tuple(float(x.strip()) for x in args.rr_ratios.split(",") if x.strip())
        backend = MultiRRBracketModel(
            symbol=args.symbol,
            timeframe=args.timeframe,
            rr_ratios=parsed_rrs,
            max_bars=args.max_bars,
            ev_mode=args.ev_mode,
            sl_mode=args.sl_mode,
            fixed_sl_atr=args.fixed_sl_atr,
            sl_quantile=args.sl_quantile,
            commission=args.commission,
            refit_days=args.refit_days,
            min_ev_floor=args.min_ev_floor,
            max_comm_r=args.max_comm_r,
            prob_shrinkage=args.prob_shrinkage,
            entry_session=args.entry_session,
        )
    elif args.model == "direct_r_bracket":
        from models.direct_r_bracket import DirectNetRBracketModel

        parsed_rrs = tuple(float(x.strip()) for x in args.rr_ratios.split(",") if x.strip())
        backend = DirectNetRBracketModel(
            symbol=args.symbol,
            timeframe=args.timeframe,
            rr_ratios=parsed_rrs,
            max_bars=args.max_bars,
            sl_mode=args.sl_mode,
            fixed_sl_atr=args.fixed_sl_atr,
            sl_quantile=args.sl_quantile,
            commission=args.commission,
            refit_days=args.refit_days,
            min_ev_floor=args.min_ev_floor,
            max_comm_r=args.max_comm_r,
        )
    elif args.model == "logistic":
        from models.logistic_regression import LogisticRegressionModel

        backend = LogisticRegressionModel(
            symbol=args.symbol,
            timeframe=args.timeframe,
            horizon=args.horizon,
            target_mode="direction",
            min_move_atr=args.min_move_atr,
            use_sample_weights=False,
        )
    else:
        backend = XGBoostModel(symbol=args.symbol, timeframe=args.timeframe)

    if args.model == "lstm":
        from models.XGBoostLSTMConsensus import _to_lstm_frame

        lstm_aux = {}
        for tf in backend.multi_timeframes:
            df_aux = auxiliary_data[tf]
            if pd.Timedelta(tf) > candle_length:
                lstm_aux[tf] = _to_lstm_frame(df_aux)
            else:
                lstm_aux[tf] = pd.DataFrame(
                    {
                        "time": df_aux.index,
                        "open": df_aux["Open"].to_numpy(dtype=float),
                        "high": df_aux["High"].to_numpy(dtype=float),
                        "low": df_aux["Low"].to_numpy(dtype=float),
                        "close": df_aux["Close"].to_numpy(dtype=float),
                        "volume": df_aux["Volume"].to_numpy(dtype=float),
                    }
                )
        backend.feature_engineering(_to_lstm_frame(train_data), _to_lstm_frame(test_data), auxiliary_data=lstm_aux)
    else:
        backend.feature_engineering(train_data, test_data, auxiliary_data=auxiliary_data)
    
    backend.train()
    backend.test()

    if args.model == "consensus":
        if args.walk_forward:
            backend.walk_forward_evaluation(n_splits=args.wf_splits)
        if args.backtest:
            bt_result, bt_signals = run_consensus_ev_backtest(
                symbol=args.symbol,
                timeframe=args.timeframe,
                train_ohlcv=train_data,
                test_ohlcv=test_data,
                val_preds_df=backend.val_predictions_df,
                test_preds_df=backend.test_predictions_df,
                commission=args.commission,
                min_ev_r=args.min_ev_r,
                prob_col="p_meta",
            )
            stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
            chart_path = (
                parent_dir
                / "results"
                / f"backtest_consensus_ev_{args.symbol}_{args.timeframe}_{stamp}.png"
            )
            plot_backtest_positions(bt_signals, bt_result, chart_path, args.timeframe)
            print(f"Consensus EV backtest chart saved: {chart_path}")
            backend.backtest_results = {
                k: v for k, v in bt_result.items()
                if k not in ("trades", "open_position")
            }
            backend.backtest_results["start"] = str(bt_result["start"])
            backend.backtest_results["end"] = str(bt_result["end"])
            backend.backtest_results["chart_path"] = str(chart_path)
        backend.evaluate()
        return

    if args.model in ("lstm", "logistic", "multi_rr_bracket"):
        backend.evaluate()
        return

    backend.evaluate()

    if not args.backtest:
        return

    execution_timeframe = "15m"
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
    result = run_bracket_backtest(
        args.symbol,
        execution_timeframe,
        low,
        backend,
        test_data.index[0],
        prepared_signals=signals,
        reward_risk=2.0,
        proxy_distance_mode=True,
        risk_fraction=0.04,
        stop_atr_multiplier=5,
    )
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    for suffix, days in (("full", None), ("recent_30_days", 30)):
        path = (
            parent_dir
            / "results"
            / f"backtest_{args.model}_{args.symbol}_{args.timeframe}_{execution_timeframe}_{suffix}_{stamp}.png"
        )
        plot_backtest_positions(
            signals,
            result,
            path,
            execution_timeframe,
            recent_days=days,
            prediction_timeframe=args.timeframe,
        )
        print(f"Backtest chart saved: {path}")


if __name__ == "__main__":
    main()
