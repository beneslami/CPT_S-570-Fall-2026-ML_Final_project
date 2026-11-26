"""Parallel directional consensus combining XGBoost, LSTM, and Logistic Regression."""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ML_interface import Model
from models.LSTMModel import LSTMModel
from models.logistic_regression import LogisticRegressionModel
from models.XGBoost import (
    DEFAULT_MAX_BARS,
    DEFAULT_SL_ATR,
    DEFAULT_TP_ATR,
    HORIZON,
    PROJECT_DIR,
    XGBoostBracketClassifier,
    _normalize_ohlcv,
    compute_direction_sample_weights,
    compute_directional_targets,
    compute_features,
    compute_resolved_bracket_targets,
)


def _to_lstm_frame(ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """Convert a capitalized OHLCV DatetimeIndex frame into a causal LSTMModel frame."""
    df = _normalize_ohlcv(ohlcv_df)
    close = df["Close"].astype(float)
    open_px = df["Open"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)
    eps = 1e-10

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr_14 = tr.rolling(14, min_periods=14).mean()

    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0.0)).rolling(14, min_periods=14).mean()
    rsi_14 = 100.0 - (100.0 / (1.0 + gain / (loss + eps)))

    typical_price = (high + low + close) / 3.0
    tp_sma = typical_price.rolling(20, min_periods=20).mean()
    tp_mad = typical_price.rolling(20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))),
        raw=True,
    )
    cci = (typical_price - tp_sma) / (0.015 * tp_mad + eps)

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_histogram = macd - macd.ewm(span=9, adjust=False).mean()

    bar_range = (high - low).replace(0.0, np.nan)
    if "volume_delta" in df.columns:
        volume_delta = pd.to_numeric(df["volume_delta"], errors="coerce").fillna(0.0)
    else:
        volume_delta = (volume * ((2.0 * close - high - low) / bar_range)).fillna(0.0)

    out = pd.DataFrame(
        {
            "time": df.index,
            "open": open_px.to_numpy(dtype=float),
            "high": high.to_numpy(dtype=float),
            "low": low.to_numpy(dtype=float),
            "close": close.to_numpy(dtype=float),
            "volume": volume.to_numpy(dtype=float),
            "volume_delta": volume_delta.to_numpy(dtype=float),
            "rsi_14": rsi_14.to_numpy(dtype=float),
            "cci": cci.to_numpy(dtype=float),
            "atr_14": atr_14.to_numpy(dtype=float),
            "macd_histogram": macd_histogram.to_numpy(dtype=float),
        }
    )
    for extra_col in ("taker_buy_ratio", "spot_taker_buy_ratio", "premium_close", "funding_rate"):
        if extra_col in df.columns:
            out[extra_col] = pd.to_numeric(df[extra_col], errors="coerce").to_numpy(dtype=float)
    return out.reset_index(drop=True)


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype=float), -30.0, 30.0)))


def _causal_rolling_logit_center(
    target_probs: pd.Series,
    window_bars: int = 720,
    min_periods: int = 100,
    seed_probs: pd.Series | None = None,
) -> pd.Series:
    """Center probabilities in logit space using strictly past scores (.shift(1))."""
    if target_probs.empty:
        return target_probs.copy()

    eff_window = max(1, int(window_bars))
    eff_min = max(1, min(int(min_periods), eff_window))

    target_valid = target_probs.dropna()
    if target_valid.empty:
        return target_probs.copy()

    target_logits = pd.Series(_logit(target_valid.to_numpy(dtype=float)), index=target_valid.index, dtype=float)

    fallback_offset = 0.0
    if seed_probs is not None:
        seed_valid = seed_probs.dropna()
        if not seed_valid.empty:
            if seed_valid.index[-1] >= target_valid.index[0]:
                raise ValueError("Seed probabilities must strictly precede target probabilities")
            seed_logits = pd.Series(_logit(seed_valid.to_numpy(dtype=float)), index=seed_valid.index, dtype=float)
            fallback_offset = float(seed_logits.median())
            combined_logits = pd.concat([seed_logits, target_logits])
        else:
            combined_logits = target_logits
    else:
        combined_logits = target_logits

    rolling_med = combined_logits.rolling(window=eff_window, min_periods=eff_min).median().shift(1)
    offsets = rolling_med.reindex(target_logits.index).fillna(fallback_offset)
    centered_probs = pd.Series(
        _sigmoid((target_logits - offsets).to_numpy(dtype=float)),
        index=target_logits.index,
        dtype=float,
    )
    return centered_probs.reindex(target_probs.index)


def _calibration_summary(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """Compute fixed-bin reliability table and Expected Calibration Error (ECE)."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    edges = (0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 1.000001)
    bins = []
    ece = 0.0
    n = len(y)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            bins.append(
                {
                    "range": f"[{lo:.2f}, {min(hi, 1.0):.2f})",
                    "samples": 0,
                    "mean_pred_prob": None,
                    "empirical_pos_rate": None,
                    "abs_gap": None,
                }
            )
            continue
        mean_p = float(p[mask].mean())
        emp_y = float(y[mask].mean())
        gap = abs(mean_p - emp_y)
        ece += (cnt / max(1, n)) * gap
        bins.append(
            {
                "range": f"[{lo:.2f}, {min(hi, 1.0):.2f})",
                "samples": cnt,
                "mean_pred_prob": mean_p,
                "empirical_pos_rate": emp_y,
                "abs_gap": float(gap),
            }
        )
    return {"ece": float(ece), "bins": bins}


def _meta_features(px: np.ndarray, pl: np.ndarray, pg: np.ndarray) -> np.ndarray:
    lx, ll, lg = _logit(px), _logit(pl), _logit(pg)
    return np.column_stack([lx, ll, lg, px - pl, px - pg, pl - pg])


def _majority_vote(xgb_label, lstm_label, logistic_label):
    xgb, lstm, logistic = (np.asarray(v, dtype=int) for v in (xgb_label, lstm_label, logistic_label))
    return ((xgb + lstm + logistic) >= 2).astype(int)


def _block_bootstrap_ci(correct_a, correct_b=None, block_size=24, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    a = np.asarray(correct_a, dtype=float)
    b = None if correct_b is None else np.asarray(correct_b, dtype=float)
    n = len(a)
    if n == 0:
        return None
    eff_block = max(1, min(int(block_size), n))
    n_blocks = int(np.ceil(n / eff_block))
    max_start = n - eff_block
    stats = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + eff_block) for s in starts])[:n]
        stats[k] = a[idx].mean() if b is None else (a[idx].mean() - b[idx].mean())
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {
        "samples": int(n),
        "block_size": int(eff_block),
        "point_estimate": float(a.mean() if b is None else (a.mean() - b.mean())),
        "boot_mean": float(stats.mean()),
        "ci_95": [float(lo), float(hi)],
        "p_gt_0_5" if b is None else "p_gt_zero": float(np.mean(stats > (0.5 if b is None else 0.0))),
    }


def _non_overlapping_mask(candle_positions: np.ndarray, min_gap: int) -> np.ndarray:
    keep = np.zeros(len(candle_positions), dtype=bool)
    last_pos = -10**9
    for idx, pos in enumerate(candle_positions):
        if pos - last_pos >= min_gap:
            keep[idx] = True
            last_pos = pos
    return keep


class DirectionalLSTMModel(LSTMModel):
    """LSTM branch aligned with the exact directional (or bracket) target used by XGBoost."""

    def _make_sequences(self, data, fit_scaler):
        features = self._numeric_features(data)
        candles = _normalize_ohlcv(data)
        target_mode = getattr(self, "branch_target_mode", "direction")
        if target_mode == "direction":
            labels = compute_directional_targets(
                candles,
                horizon=self.horizon,
                min_move_atr=getattr(self, "branch_min_move_atr", 0.0),
            )
            weights = compute_direction_sample_weights(candles, horizon=self.horizon).to_numpy(dtype=np.float32)
            purge = self.horizon
        else:
            labels = compute_resolved_bracket_targets(
                candles,
                sl_atr=self.bracket_sl,
                tp_atr=self.bracket_tp,
                max_bars=self.bracket_horizon,
            )
            weights = np.ones(len(data), dtype=np.float32)
            purge = self.bracket_horizon

        if purge > 0 and len(labels) > purge:
            labels.iloc[-purge:] = np.nan

        if fit_scaler:
            rows = features.notna().all(axis=1)
            if rows.sum() < self.lookback:
                raise ValueError("Not enough complete LSTM rows to fit the scaler")
            self.scaler.fit(features.loc[rows])
        scaled = np.clip(self.scaler.transform(features), -5.0, 5.0)
        windows = np.lib.stride_tricks.sliding_window_view(scaled, self.lookback, axis=0)
        windows = np.moveaxis(windows, -1, 1)
        ends = np.arange(self.lookback - 1, len(data))
        y = labels.to_numpy(dtype=float)[ends]
        w = weights[ends]
        valid = np.isfinite(windows).all(axis=(1, 2)) & np.isfinite(y)
        if not valid.any():
            raise ValueError("No complete LSTM sequences")
        times = pd.DatetimeIndex(pd.to_datetime(data["time"], utc=True), name="feature_end_time")
        return (
            windows[valid].astype(np.float32),
            y[valid].astype(np.float32),
            times[ends[valid]],
            float(valid.sum() / max(1, len(ends))),
            w[valid] if (target_mode == "direction" and self.use_sample_weights) else None,
        )


BracketLSTMModel = DirectionalLSTMModel


class XGBoostLSTMConsensusModel(Model):
    """Directional ensemble combining XGBoost, LSTM/GRU sequence branch, and Logistic Regression."""

    MODEL_NAME = "XGBoostLSTMLogisticDirectionalEnsemble"
    CONFIDENCE_LEVELS = (0.55, 0.60, 0.65)

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        results_path=None,
        horizon: int = HORIZON,
        cv_splits: int = 5,
        confidence_threshold: float = 0.55,
        sl_atr: float = DEFAULT_SL_ATR,
        tp_atr: float = DEFAULT_TP_ATR,
        max_bars: int = DEFAULT_MAX_BARS,
        calibration_method: str = "auto",
        target_mode: str = "direction",
        min_move_atr: float = 0.0,
        xgb_include_fvg_ob: bool = False,
        xgb_reg_profile: str = "strong",
        use_sample_weights: bool = False,
        decision_mode: str = "static",
        center_window_bars: int = 720,
        center_min_periods: int = 100,
        lstm_lookback: int | str = 24,
        lstm_units: int | str = 32,
        lstm_target_threshold: float | str = 0.0,
        lstm_epochs: int = 40,
        lstm_batch_size: int = 64,
        lstm_confidence_threshold: float | str = 0.50,
        lstm_ensemble: bool = False,
        lstm_ensemble_specs=None,
        random_state: int = 42,
    ):
        if decision_mode not in ("static", "rolling_center"):
            raise ValueError("decision_mode must be 'static' or 'rolling_center'")
        if xgb_reg_profile not in ("baseline", "strong"):
            raise ValueError("xgb_reg_profile must be 'baseline' or 'strong'")
        if center_window_bars < 2 or center_min_periods < 1:
            raise ValueError("center_window_bars must be >= 2 and center_min_periods >= 1")

        self.symbol = symbol
        self.timeframe = timeframe
        self.horizon = int(horizon)
        self.cv_splits = int(cv_splits)
        self.max_bars = int(max_bars)
        self.sl_atr = float(sl_atr)
        self.tp_atr = float(tp_atr)
        self.target_mode = target_mode
        self.min_move_atr = float(min_move_atr)
        self.xgb_include_fvg_ob = bool(xgb_include_fvg_ob)
        self.xgb_reg_profile = xgb_reg_profile
        self.use_sample_weights = bool(use_sample_weights)
        self.decision_mode = decision_mode
        self.center_window_bars = int(center_window_bars)
        self.center_min_periods = int(center_min_periods)
        self.purge_gap = self.horizon if self.target_mode == "direction" else self.max_bars
        self.confidence_threshold = float(confidence_threshold)
        self.calibration_method = calibration_method
        self.lstm_lookback = lstm_lookback
        self.lstm_units = lstm_units
        self.lstm_target_threshold = lstm_target_threshold
        self.lstm_epochs = lstm_epochs
        self.lstm_batch_size = lstm_batch_size
        self.lstm_confidence_threshold = lstm_confidence_threshold
        self.lstm_ensemble = bool(lstm_ensemble)
        self.random_state = int(random_state)
        self.results_path = (
            Path(results_path)
            if results_path
            else PROJECT_DIR / "results" / f"xgb_lstm_consensus_{symbol}_{timeframe}.json"
        )

        self.xgb_branch = XGBoostBracketClassifier(
            symbol=symbol,
            timeframe=timeframe,
            horizon=self.horizon,
            cv_splits=cv_splits,
            confidence_threshold=confidence_threshold,
            sl_atr=sl_atr,
            tp_atr=tp_atr,
            max_bars=max_bars,
            calibration_method=calibration_method,
            target_mode=self.target_mode,
            min_move_atr=self.min_move_atr,
            use_sample_weights=self.use_sample_weights,
            include_fvg_ob=self.xgb_include_fvg_ob,
            reg_profile=self.xgb_reg_profile,
            random_state=random_state,
        )

        if self.lstm_ensemble:
            default_units = lstm_units if isinstance(lstm_units, int) else 32
            self.lstm_ensemble_specs = tuple(
                lstm_ensemble_specs
                or (
                    ("lstm", 12, default_units),
                    ("lstm", 24, default_units),
                    ("gru", 48, default_units),
                )
            )
        else:
            self.lstm_ensemble_specs = (("lstm", lstm_lookback, lstm_units),)

        self.lstm_branches = []
        for idx, (rnn_type, lb, units) in enumerate(self.lstm_ensemble_specs):
            branch = DirectionalLSTMModel(
                symbol=symbol,
                timeframe=timeframe,
                lookback=lb,
                target_threshold=lstm_target_threshold,
                lstm_units=units,
                epochs=lstm_epochs,
                batch_size=lstm_batch_size,
                confidence_threshold=lstm_confidence_threshold,
                random_seed=self.random_state + idx,
                multi_timeframes=(),
                target="direction",
                horizon=self.horizon,
                rnn_type=rnn_type,
                use_sample_weights=self.use_sample_weights,
            )
            branch.branch_target_mode = self.target_mode
            branch.branch_min_move_atr = self.min_move_atr
            branch.bracket_sl = self.sl_atr
            branch.bracket_tp = self.tp_atr
            branch.bracket_horizon = self.max_bars
            branch.training_verbose = 0
            self.lstm_branches.append(branch)

        self.lstm_branch = self.lstm_branches[0]
        self.lstm_decision_threshold = 0.50
        self.lstm_training_accuracy = None

        self.logistic_branch = LogisticRegressionModel(
            symbol,
            timeframe,
            random_state=random_state,
            sl_atr=self.sl_atr,
            tp_atr=self.tp_atr,
            max_bars=self.max_bars,
            horizon=self.horizon,
            target_mode=self.target_mode,
            min_move_atr=self.min_move_atr,
            use_sample_weights=self.use_sample_weights,
        )

        self._train_ohlcv = None
        self._test_ohlcv = None
        self._auxiliary_data = None
        self._lstm_aux_frames = {}
        self._lstm_multi_timeframes = ()
        self._selection_raw_probs = {}
        self.train_time = 0.0
        self.metrics = None
        self.soft_weights = (1 / 3, 1 / 3, 1 / 3)
        self.soft_threshold = 0.50
        self.meta_model = None
        self.meta_threshold = 0.50
        self.selected_policy = "soft_weighted"
        self.selection_summary = {}
        self.val_predictions_df = None
        self.test_predictions_df = None
        self.backtest_results = {}
        self.walk_forward_results = None

    def feature_engineering(self, train_data, test_data, auxiliary_data=None, **kwargs):
        train_df = _normalize_ohlcv(train_data)
        test_df = _normalize_ohlcv(test_data)
        if train_df.equals(test_df):
            split_at = int(len(train_df) * 0.8)
            train_df, test_df = train_df.iloc[:split_at], train_df.iloc[split_at:]
        if train_df.empty or test_df.empty or train_df.index[-1] >= test_df.index[0]:
            raise ValueError("Training candles must end before test candles begin")

        self._train_ohlcv = train_df.copy()
        self._test_ohlcv = test_df.copy()
        self._auxiliary_data = auxiliary_data
        self._lstm_aux_frames = {}
        lstm_tfs = []
        if auxiliary_data:
            primary_td = pd.Timedelta(self.timeframe)
            for tf in ("5m", "15m", "4h"):
                if tf == self.timeframe or tf not in auxiliary_data:
                    continue
                aux_td = pd.Timedelta(tf)
                if aux_td > primary_td:
                    self._lstm_aux_frames[tf] = _to_lstm_frame(auxiliary_data[tf])
                    lstm_tfs.append(tf)
                elif aux_td < primary_td and primary_td % aux_td == pd.Timedelta(0):
                    raw_sub = _normalize_ohlcv(auxiliary_data[tf])
                    self._lstm_aux_frames[tf] = pd.DataFrame(
                        {
                            "time": raw_sub.index,
                            "open": raw_sub["Open"].to_numpy(dtype=float),
                            "high": raw_sub["High"].to_numpy(dtype=float),
                            "low": raw_sub["Low"].to_numpy(dtype=float),
                            "close": raw_sub["Close"].to_numpy(dtype=float),
                            "volume": raw_sub["Volume"].to_numpy(dtype=float),
                        }
                    )
                    lstm_tfs.append(tf)
        self._lstm_multi_timeframes = tuple(lstm_tfs)
        for member in self.lstm_branches:
            member.multi_timeframes = self._lstm_multi_timeframes

        # Reserve the final 20% of train_df (purged) for fitting soft weights & meta-learner
        selection_start = int(len(train_df) * 0.80)
        branch_end = selection_start - self.purge_gap
        if branch_end <= 100:
            raise ValueError("Not enough training history for purged ensemble selection")
        branch_train = train_df.iloc[:branch_end]
        self._selection_ohlcv = train_df.iloc[selection_start:].copy()
        self._branch_train_ohlcv = branch_train.copy()

        print(
            f"Preparing branches (target_mode={self.target_mode}, horizon={self.horizon}, "
            f"xgb_fvg_ob={self.xgb_include_fvg_ob}, xgb_reg={self.xgb_reg_profile}, "
            f"weights={self.use_sample_weights}, decision_mode={self.decision_mode}, "
            f"lstm_ensemble={self.lstm_ensemble}, lstm_mtf={list(self._lstm_multi_timeframes)}):"
        )
        self.xgb_branch.feature_engineering(
            branch_train,
            self._selection_ohlcv,
            auxiliary_data=self._auxiliary_data,
            order_block_history=train_df,
        )
        lstm_train_frame = _to_lstm_frame(branch_train)
        lstm_sel_frame = _to_lstm_frame(self._selection_ohlcv)
        for member in self.lstm_branches:
            member.feature_engineering(lstm_train_frame, lstm_sel_frame, auxiliary_data=self._lstm_aux_frames)
        self.logistic_branch.feature_engineering(
            branch_train,
            self._selection_ohlcv,
            auxiliary_data=self._auxiliary_data,
        )

    def train(self):
        if self._train_ohlcv is None:
            raise RuntimeError("Run feature_engineering() before train()")

        started = time.perf_counter()
        print("\n--- Training XGBoost Branch ---")
        xgb_train_acc = self.xgb_branch.train()

        print("\n--- Training LSTM Branch ---")
        member_train_accs = []
        val_prob_series = []
        val_label_series = None
        for idx, member in enumerate(self.lstm_branches):
            acc = member.train()
            member_train_accs.append(float(acc))
            print(
                f"  Seq[{idx}] {member.rnn_type.upper()}(lb={member.lookback}, units={member.lstm_units}, "
                f"seed={member.random_seed}) -> threshold={member.decision_threshold:.2f} | "
                f"Train Acc: {acc * 100:.2f}% | Val Acc: {member.validation_accuracy * 100:.2f}%"
            )
            vp_raw = member.model.predict(member.X_validation, batch_size=member.batch_size, verbose=0).reshape(-1)
            vp = member.calibrate_proba(vp_raw)
            val_prob_series.append(pd.Series(vp, index=member.validation_times, dtype=float))
            if val_label_series is None:
                val_label_series = pd.Series(member.y_validation.astype(int), index=member.validation_times)

        self.lstm_training_accuracy = float(np.mean(member_train_accs))
        if len(self.lstm_branches) == 1:
            self.lstm_decision_threshold = float(self.lstm_branches[0].decision_threshold)
        else:
            val_df = pd.concat(val_prob_series, axis=1).dropna()
            aligned_y = val_label_series.reindex(val_df.index).to_numpy(dtype=int)
            ens_val_prob = val_df.mean(axis=1).to_numpy(dtype=float)
            self.lstm_decision_threshold = float(
                max(
                    np.linspace(0.45, 0.55, 21),
                    key=lambda t: (balanced_accuracy_score(aligned_y, ens_val_prob >= t), -abs(t - 0.5)),
                )
            )
            ens_val_acc = float(accuracy_score(aligned_y, ens_val_prob >= self.lstm_decision_threshold))
            print(
                f"  Seq Ensemble ({len(self.lstm_branches)} models) -> "
                f"threshold={self.lstm_decision_threshold:.2f} | Val Acc: {ens_val_acc * 100:.2f}%"
            )
        lstm_train_acc = self.lstm_training_accuracy

        print("\n--- Training Logistic Branch ---")
        logistic_train_acc = self.logistic_branch.train()
        print(
            f"Logistic -> C={self.logistic_branch.best_C}, features={len(self.logistic_branch.features)}, "
            f"threshold={self.logistic_branch.decision_threshold:.2f} | "
            f"Train Acc: {logistic_train_acc * 100:.2f}% | "
            f"Val Acc: {(self.logistic_branch.validation_acc or 0.0) * 100:.2f}%"
        )

        self._fit_ensemble_combiners()

        # Phase 2: Refit all 3 branches on 100% of train_df (no test data touched)
        print("\n--- Refitting All 3 Branches on Full Training Period ---")
        self.xgb_branch.feature_engineering(
            self._train_ohlcv,
            self._test_ohlcv,
            auxiliary_data=self._auxiliary_data,
            order_block_history=pd.concat([self._train_ohlcv, self._test_ohlcv]),
        )
        xgb_train_acc = self.xgb_branch.train()

        full_lstm_train = _to_lstm_frame(self._train_ohlcv)
        full_lstm_test = _to_lstm_frame(self._test_ohlcv)
        member_train_accs = []
        val_prob_series = []
        val_label_series = None
        for idx, member in enumerate(self.lstm_branches):
            member.feature_engineering(full_lstm_train, full_lstm_test, auxiliary_data=self._lstm_aux_frames)
            acc = member.train()
            member_train_accs.append(float(acc))
            print(
                f"  Refit Seq[{idx}] {member.rnn_type.upper()}(lb={member.lookback}, units={member.lstm_units}) "
                f"-> threshold={member.decision_threshold:.2f} | Train Acc: {acc * 100:.2f}% | "
                f"Val Acc: {member.validation_accuracy * 100:.2f}%"
            )
            vp_raw = member.model.predict(member.X_validation, batch_size=member.batch_size, verbose=0).reshape(-1)
            vp = member.calibrate_proba(vp_raw)
            val_prob_series.append(pd.Series(vp, index=member.validation_times, dtype=float))
            if val_label_series is None:
                val_label_series = pd.Series(member.y_validation.astype(int), index=member.validation_times)
        self.lstm_training_accuracy = float(np.mean(member_train_accs))
        if len(self.lstm_branches) == 1:
            self.lstm_decision_threshold = float(self.lstm_branches[0].decision_threshold)
        else:
            val_df = pd.concat(val_prob_series, axis=1).dropna()
            aligned_y = val_label_series.reindex(val_df.index).to_numpy(dtype=int)
            ens_val_prob = val_df.mean(axis=1).to_numpy(dtype=float)
            self.lstm_decision_threshold = float(
                max(
                    np.linspace(0.45, 0.55, 21),
                    key=lambda t: (balanced_accuracy_score(aligned_y, ens_val_prob >= t), -abs(t - 0.5)),
                )
            )
        lstm_train_acc = self.lstm_training_accuracy

        self.logistic_branch.feature_engineering(
            self._train_ohlcv,
            self._test_ohlcv,
            auxiliary_data=self._auxiliary_data,
        )
        logistic_train_acc = self.logistic_branch.train()
        print(
            f"  Refit Logistic -> C={self.logistic_branch.best_C}, "
            f"threshold={self.logistic_branch.decision_threshold:.2f} | "
            f"Train Acc: {logistic_train_acc * 100:.2f}% | "
            f"Val Acc: {(self.logistic_branch.validation_acc or 0.0) * 100:.2f}%"
        )

        self.train_time = time.perf_counter() - started
        return {
            "xgb_train_accuracy": xgb_train_acc,
            "lstm_train_accuracy": lstm_train_acc,
            "logistic_train_accuracy": logistic_train_acc,
        }

    def _fit_ensemble_combiners(self):
        """Fit soft-voting weights and stacking meta-learner on the purged selection slice."""
        index = self.xgb_branch.X_test.index
        px_s = pd.Series(self.xgb_branch.predict_calibrated_proba(self.xgb_branch.X_test), index=index, dtype=float)
        pl_s = self._predict_lstm_probabilities(self._train_ohlcv, predict_from=index[0]).reindex(index)
        pg_s = self.logistic_branch.predict_probabilities(self._train_ohlcv).reindex(index)

        valid = px_s.notna() & pl_s.notna() & pg_s.notna()
        sel_index = index[valid.to_numpy()]
        y = self.xgb_branch.y_test.loc[sel_index].to_numpy(dtype=int)

        # Store strictly out-of-sample selection scores to seed test-fold rolling centering
        self._selection_raw_probs = {
            "xgb": px_s.loc[sel_index].copy(),
            "lstm": pl_s.loc[sel_index].copy(),
            "logistic": pg_s.loc[sel_index].copy(),
        }

        if self.decision_mode == "rolling_center":
            px = _causal_rolling_logit_center(
                self._selection_raw_probs["xgb"],
                window_bars=self.center_window_bars,
                min_periods=self.center_min_periods,
            ).to_numpy(dtype=float)
            pl = _causal_rolling_logit_center(
                self._selection_raw_probs["lstm"],
                window_bars=self.center_window_bars,
                min_periods=self.center_min_periods,
            ).to_numpy(dtype=float)
            pg = _causal_rolling_logit_center(
                self._selection_raw_probs["logistic"],
                window_bars=self.center_window_bars,
                min_periods=self.center_min_periods,
            ).to_numpy(dtype=float)
        else:
            px = self._selection_raw_probs["xgb"].to_numpy(dtype=float)
            pl = self._selection_raw_probs["lstm"].to_numpy(dtype=float)
            pg = self._selection_raw_probs["logistic"].to_numpy(dtype=float)

        if len(y) < 100 or len(np.unique(y)) < 2:
            raise ValueError("Not enough aligned selection samples to fit ensemble combiners")

        # Purged 60/40 split inside selection period: fit combiners on first 60%, compare on last 40%
        split = int(len(y) * 0.60)
        fit_end = max(50, split - self.purge_gap)
        y_fit, y_val = y[:fit_end], y[split:]
        px_f, pl_f, pg_f = px[:fit_end], pl[:fit_end], pg[:fit_end]
        px_v, pl_v, pg_v = px[split:], pl[split:], pg[split:]

        # 1. Grid search simplex weights for soft probability blend (step = 0.05)
        best_ll = None
        best_w = (1 / 3, 1 / 3, 1 / 3)
        grid = np.linspace(0.0, 1.0, 21)
        for wx in grid:
            for wl in grid:
                wg = round(1.0 - wx - wl, 6)
                if wg < -1e-9:
                    continue
                wg = max(0.0, wg)
                p_blend = wx * px_f + wl * pl_f + wg * pg_f
                ll = float(log_loss(y_fit, p_blend, labels=[0, 1]))
                if best_ll is None or ll < best_ll:
                    best_ll = ll
                    best_w = (float(wx), float(wl), float(wg))
        self.soft_weights = best_w
        p_soft_fit = best_w[0] * px_f + best_w[1] * pl_f + best_w[2] * pg_f

        if self.decision_mode == "rolling_center":
            self.soft_threshold = 0.50
        else:
            self.soft_threshold = float(
                max(
                    np.linspace(0.45, 0.55, 21),
                    key=lambda t: (balanced_accuracy_score(y_fit, p_soft_fit >= t), -abs(t - 0.5)),
                )
            )

        # 2. Fit regularized stacking meta-learner
        X_meta_fit = _meta_features(px_f, pl_f, pg_f)
        X_meta_val = _meta_features(px_v, pl_v, pg_v)
        best_meta_ll = None
        best_meta = None
        for c_val in (0.01, 0.05, 0.2, 1.0):
            clf = LogisticRegression(C=c_val, max_iter=1000, random_state=self.random_state)
            clf.fit(X_meta_fit, y_fit)
            val_p = clf.predict_proba(X_meta_val)[:, 1]
            ll = float(log_loss(y_val, val_p, labels=[0, 1]))
            if best_meta_ll is None or ll < best_meta_ll:
                best_meta_ll = ll
                best_meta = clf
        self.meta_model = best_meta
        p_meta_fit = self.meta_model.predict_proba(X_meta_fit)[:, 1]

        if self.decision_mode == "rolling_center":
            self.meta_threshold = 0.50
        else:
            self.meta_threshold = float(
                max(
                    np.linspace(0.45, 0.55, 21),
                    key=lambda t: (balanced_accuracy_score(y_fit, p_meta_fit >= t), -abs(t - 0.5)),
                )
            )

        # 3. Compare candidate ensemble policies on sel_val
        if self.decision_mode == "rolling_center":
            xgb_thr = lstm_thr = log_thr = 0.50
        else:
            xgb_thr = self.xgb_branch.decision_threshold
            lstm_thr = self.lstm_decision_threshold
            log_thr = self.logistic_branch.decision_threshold

        dx_v = (px_v >= xgb_thr).astype(int)
        dl_v = (pl_v >= lstm_thr).astype(int)
        dg_v = (pg_v >= log_thr).astype(int)
        votes_v = dx_v + dl_v + dg_v
        pred_maj_v = _majority_vote(dx_v, dl_v, dg_v)
        p_soft_v = best_w[0] * px_v + best_w[1] * pl_v + best_w[2] * pg_v
        pred_soft_v = (p_soft_v >= self.soft_threshold).astype(int)
        p_meta_v = self.meta_model.predict_proba(X_meta_val)[:, 1]
        pred_meta_v = (p_meta_v >= self.meta_threshold).astype(int)

        self.val_predictions_df = pd.DataFrame(
            {
                "p_xgb": px_v,
                "p_lstm": pl_v,
                "p_logistic": pg_v,
                "p_soft": p_soft_v,
                "p_meta": p_meta_v,
                "unanimous": ((votes_v == 0) | (votes_v == 3)).astype(bool),
                "majority_side": np.where(pred_maj_v == 1, 1, -1).astype(int),
            },
            index=sel_index[split:],
        )

        candidates = {}
        for name, pred in (
            ("three_majority", pred_maj_v),
            ("soft_weighted", pred_soft_v),
            ("meta_stacking", pred_meta_v),
        ):
            candidates[name] = {
                "val_accuracy": float(accuracy_score(y_val, pred)),
                "val_balanced_accuracy": float(balanced_accuracy_score(y_val, pred)),
            }
        self.selected_policy = max(
            candidates, key=lambda k: (candidates[k]["val_accuracy"], candidates[k]["val_balanced_accuracy"])
        )
        self.selection_summary = {
            "decision_mode": self.decision_mode,
            "center_window_bars": self.center_window_bars if self.decision_mode == "rolling_center" else None,
            "center_min_periods": self.center_min_periods if self.decision_mode == "rolling_center" else None,
            "selection_samples": int(len(y)),
            "fit_samples": int(len(y_fit)),
            "val_samples": int(len(y_val)),
            "soft_weights": {"xgb": best_w[0], "lstm": best_w[1], "logistic": best_w[2]},
            "soft_threshold": self.soft_threshold,
            "meta_threshold": self.meta_threshold,
            "val_candidates": candidates,
            "selected_policy": self.selected_policy,
        }
        print(
            f"\nEnsemble combiners ({self.decision_mode}) -> soft_weights(XGB/LSTM/Log)={best_w}, "
            f"selected_policy={self.selected_policy} | val_candidates={candidates}"
        )

    def _prepare_final_xgb_test(self):
        frame = self._test_ohlcv
        combined = pd.concat([self._train_ohlcv, frame])
        feats = compute_features(
            combined,
            include_fvg_ob=self.xgb_branch.include_fvg_ob,
            auxiliary_data=self._auxiliary_data,
            timeframe=self.timeframe,
        ).loc[frame.index]
        if self.target_mode == "direction":
            labels = compute_directional_targets(frame, horizon=self.horizon, min_move_atr=self.min_move_atr)
        else:
            labels = compute_resolved_bracket_targets(
                frame, sl_atr=self.sl_atr, tp_atr=self.tp_atr, max_bars=self.max_bars
            )
        valid = feats.notna().all(axis=1) & labels.notna() & frame["Close"].notna()
        if self.purge_gap > 0 and len(valid) > self.purge_gap:
            valid.iloc[-self.purge_gap :] = False
        if not valid.any():
            raise ValueError("No valid samples in the final test period")
        self.xgb_branch.X_test = feats.loc[valid].copy()
        self.xgb_branch.y_test = labels.loc[valid].astype(int).copy()
        self.xgb_branch.curr_test = frame.loc[valid, "Close"].copy()
        self.xgb_branch.test_trend_pred = (
            (combined["Close"] > combined["Close"].shift(1)).astype(int).reindex(self.xgb_branch.X_test.index)
        )

    @staticmethod
    def _predict_single_lstm_member(
        member: DirectionalLSTMModel,
        frame: pd.DataFrame,
        times: pd.DatetimeIndex,
        batch_size: int = 512,
        predict_from=None,
    ) -> pd.Series:
        probs = pd.Series(np.nan, index=times, dtype=float)
        lookback = member.lookback
        if len(frame) < lookback:
            return probs

        features = member._numeric_features(frame)
        scaled = np.clip(member.scaler.transform(features), -5.0, 5.0).astype(np.float32)
        windows = np.lib.stride_tricks.sliding_window_view(scaled, lookback, axis=0)
        windows = np.moveaxis(windows, -1, 1)

        valid = np.isfinite(windows).all(axis=(1, 2))
        if predict_from is not None:
            valid &= times[lookback - 1 :] >= pd.Timestamp(predict_from)
        valid_indices = np.flatnonzero(valid)
        if not len(valid_indices):
            return probs

        out = np.full(len(frame), np.nan, dtype=float)
        for start in range(0, len(valid_indices), batch_size):
            idx = valid_indices[start : start + batch_size]
            pred_raw = np.asarray(
                member.model.predict(windows[idx], batch_size=batch_size, verbose=0)
            ).reshape(-1)
            out[idx + lookback - 1] = member.calibrate_proba(pred_raw)
        return pd.Series(out, index=times, dtype=float)

    def _predict_lstm_probabilities(self, ohlcv_df: pd.DataFrame, batch_size: int = 512, predict_from=None) -> pd.Series:
        frame = _to_lstm_frame(ohlcv_df)
        if self._lstm_multi_timeframes and self._lstm_aux_frames:
            frame = self.lstm_branches[0]._align_auxiliary_features(frame, self._lstm_aux_frames)
        times = pd.DatetimeIndex(pd.to_datetime(frame["time"], utc=True))
        member_series = [
            self._predict_single_lstm_member(member, frame, times, batch_size=batch_size, predict_from=predict_from)
            for member in self.lstm_branches
        ]
        if len(member_series) == 1:
            return member_series[0]
        combined = pd.concat(member_series, axis=1)
        valid_all = combined.notna().all(axis=1)
        result = pd.Series(np.nan, index=times, dtype=float)
        if valid_all.any():
            result.loc[valid_all] = combined.loc[valid_all].mean(axis=1)
        return result

    def test(self):
        if (
            self.xgb_branch.model is None
            or any(member.model is None for member in self.lstm_branches)
            or self.logistic_branch.training_acc is None
        ):
            raise RuntimeError("Run train() before test()")

        self._prepare_final_xgb_test()
        print("\n--- Testing XGBoost Branch ---")
        xgb_res = self.xgb_branch.test()

        combined = pd.concat([self._train_ohlcv, self._test_ohlcv])
        lstm_probs_all = self._predict_lstm_probabilities(combined, predict_from=self._test_ohlcv.index[0])

        test_index = self.xgb_branch.X_test.index
        xgb_prob_raw = pd.Series(self.xgb_branch.test_prob, index=test_index, dtype=float)
        lstm_prob_raw = lstm_probs_all.reindex(test_index)
        logistic_prob_raw = self.logistic_branch.predict_probabilities(combined).reindex(test_index)

        valid = xgb_prob_raw.notna() & lstm_prob_raw.notna() & logistic_prob_raw.notna()
        if not valid.any():
            raise ValueError("No overlapping test timestamps between XGBoost, LSTM, and Logistic")

        aligned_index = test_index[valid.to_numpy()]
        y_true = self.xgb_branch.y_test.loc[aligned_index].to_numpy(dtype=int)

        px_test_s = xgb_prob_raw.loc[aligned_index]
        pl_test_s = lstm_prob_raw.loc[aligned_index]
        pg_test_s = logistic_prob_raw.loc[aligned_index]

        if self.decision_mode == "rolling_center":
            p_xgb = _causal_rolling_logit_center(
                px_test_s,
                window_bars=self.center_window_bars,
                min_periods=self.center_min_periods,
                seed_probs=self._selection_raw_probs.get("xgb"),
            ).to_numpy(dtype=float)
            p_lstm = _causal_rolling_logit_center(
                pl_test_s,
                window_bars=self.center_window_bars,
                min_periods=self.center_min_periods,
                seed_probs=self._selection_raw_probs.get("lstm"),
            ).to_numpy(dtype=float)
            p_logistic = _causal_rolling_logit_center(
                pg_test_s,
                window_bars=self.center_window_bars,
                min_periods=self.center_min_periods,
                seed_probs=self._selection_raw_probs.get("logistic"),
            ).to_numpy(dtype=float)
            xgb_thr = lstm_thr = log_thr = 0.50
        else:
            p_xgb = px_test_s.to_numpy(dtype=float)
            p_lstm = pl_test_s.to_numpy(dtype=float)
            p_logistic = pg_test_s.to_numpy(dtype=float)
            xgb_thr = self.xgb_branch.decision_threshold
            lstm_thr = self.lstm_decision_threshold
            log_thr = self.logistic_branch.decision_threshold

        pred_xgb = (p_xgb >= xgb_thr).astype(int)
        pred_lstm = (p_lstm >= lstm_thr).astype(int)
        pred_logistic = (p_logistic >= log_thr).astype(int)

        votes = pred_xgb + pred_lstm + pred_logistic
        pred_majority = _majority_vote(pred_xgb, pred_lstm, pred_logistic)
        unanimous_mask = (votes == 0) | (votes == 3)

        wx, wl, wg = self.soft_weights
        p_soft = wx * p_xgb + wl * p_lstm + wg * p_logistic
        pred_soft = (p_soft >= self.soft_threshold).astype(int)

        p_meta = self.meta_model.predict_proba(_meta_features(p_xgb, p_lstm, p_logistic))[:, 1]
        pred_meta = (p_meta >= self.meta_threshold).astype(int)

        self.test_predictions_df = pd.DataFrame(
            {
                "p_xgb": p_xgb,
                "p_lstm": p_lstm,
                "p_logistic": p_logistic,
                "p_soft": p_soft,
                "p_meta": p_meta,
                "unanimous": unanimous_mask.astype(bool),
                "majority_side": np.where(pred_majority == 1, 1, -1).astype(int),
            },
            index=aligned_index,
        )

        def _score_branch(train_acc, pred, prob):
            p_clip = np.clip(prob, 1e-6, 1.0 - 1e-6)
            bull = int((pred == 1).sum())
            bear = int((pred == 0).sum())
            return {
                "train_accuracy": float(train_acc) if train_acc is not None else None,
                "test_accuracy": float(accuracy_score(y_true, pred)),
                "test_balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
                "test_roc_auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else float("nan"),
                "test_log_loss": float(log_loss(y_true, p_clip, labels=[0, 1])),
                "test_brier_score": float(brier_score_loss(y_true, p_clip)),
                "bullish_predictions": bull,
                "bearish_predictions": bear,
                "bullish_rate": float(bull / max(1, bull + bear)),
            }

        branches = {
            "xgb": _score_branch(self.xgb_branch.train_acc, pred_xgb, p_xgb),
            "lstm": _score_branch(self.lstm_training_accuracy, pred_lstm, p_lstm),
            "logistic": _score_branch(self.logistic_branch.training_acc, pred_logistic, p_logistic),
        }

        p_soft_clip = np.clip(p_soft, 1e-6, 1.0 - 1e-6)
        p_meta_clip = np.clip(p_meta, 1e-6, 1.0 - 1e-6)

        ensembles = {
            "three_majority": {
                "test_accuracy": float(accuracy_score(y_true, pred_majority)),
                "test_balanced_accuracy": float(balanced_accuracy_score(y_true, pred_majority)),
                "bullish_rate": float((pred_majority == 1).mean()),
                "coverage": 1.0,
                "samples": int(len(y_true)),
            },
            "three_unanimous": {
                "test_accuracy": float(accuracy_score(y_true[unanimous_mask], pred_majority[unanimous_mask]))
                if unanimous_mask.any()
                else None,
                "test_balanced_accuracy": float(
                    balanced_accuracy_score(y_true[unanimous_mask], pred_majority[unanimous_mask])
                )
                if unanimous_mask.any()
                else None,
                "bullish_rate": float((pred_majority[unanimous_mask] == 1).mean()) if unanimous_mask.any() else None,
                "coverage": float(unanimous_mask.mean()),
                "samples": int(unanimous_mask.sum()),
            },
            "soft_weighted": {
                "test_accuracy": float(accuracy_score(y_true, pred_soft)),
                "test_balanced_accuracy": float(balanced_accuracy_score(y_true, pred_soft)),
                "test_roc_auc": float(roc_auc_score(y_true, p_soft)) if len(np.unique(y_true)) > 1 else float("nan"),
                "test_log_loss": float(log_loss(y_true, p_soft_clip, labels=[0, 1])),
                "test_brier_score": float(brier_score_loss(y_true, p_soft_clip)),
                "bullish_rate": float((pred_soft == 1).mean()),
                "coverage": 1.0,
                "samples": int(len(y_true)),
            },
            "meta_stacking": {
                "test_accuracy": float(accuracy_score(y_true, pred_meta)),
                "test_balanced_accuracy": float(balanced_accuracy_score(y_true, pred_meta)),
                "test_roc_auc": float(roc_auc_score(y_true, p_meta)) if len(np.unique(y_true)) > 1 else float("nan"),
                "test_log_loss": float(log_loss(y_true, p_meta_clip, labels=[0, 1])),
                "test_brier_score": float(brier_score_loss(y_true, p_meta_clip)),
                "bullish_rate": float((pred_meta == 1).mean()),
                "coverage": 1.0,
                "samples": int(len(y_true)),
            },
        }

        # High-confidence selective accuracy & empirical calibration check on soft-weighted ensemble
        soft_conf = np.where(pred_soft == 1, p_soft, 1.0 - p_soft)
        confidence_tiers = {}
        for level in self.CONFIDENCE_LEVELS:
            mask = soft_conf >= level
            if mask.any():
                acc = float(accuracy_score(y_true[mask], pred_soft[mask]))
                mean_conf = float(soft_conf[mask].mean())
                cal_gap = float(acc - mean_conf)
            else:
                acc = mean_conf = cal_gap = None
            confidence_tiers[f"ge_{level:.2f}"] = {
                "samples": int(mask.sum()),
                "coverage": float(mask.mean()),
                "accuracy": acc,
                "mean_predicted_confidence": mean_conf,
                "calibration_gap": cal_gap,
            }

        calibration = {
            "soft_weighted": _calibration_summary(y_true, p_soft_clip),
            "meta_stacking": _calibration_summary(y_true, p_meta_clip),
        }

        # Non-overlapping evaluation (min_gap = purge_gap; at H=1 every candle is non-overlapping)
        test_positions = self._test_ohlcv.index.get_indexer(aligned_index)
        no_mask = _non_overlapping_mask(test_positions, self.purge_gap)
        non_overlapping = {
            "min_gap_candles": self.purge_gap,
            "samples": int(no_mask.sum()),
            "solo_xgb_accuracy": float(accuracy_score(y_true[no_mask], pred_xgb[no_mask])) if no_mask.any() else None,
            "solo_lstm_accuracy": float(accuracy_score(y_true[no_mask], pred_lstm[no_mask])) if no_mask.any() else None,
            "solo_logistic_accuracy": float(accuracy_score(y_true[no_mask], pred_logistic[no_mask]))
            if no_mask.any()
            else None,
            "three_majority_accuracy": float(accuracy_score(y_true[no_mask], pred_majority[no_mask]))
            if no_mask.any()
            else None,
            "soft_weighted_accuracy": float(accuracy_score(y_true[no_mask], pred_soft[no_mask]))
            if no_mask.any()
            else None,
            "meta_stacking_accuracy": float(accuracy_score(y_true[no_mask], pred_meta[no_mask]))
            if no_mask.any()
            else None,
        }

        block_size = max(1, self.purge_gap)
        bootstrap_cis = {
            "soft_weighted_accuracy": _block_bootstrap_ci(
                pred_soft == y_true, block_size=block_size, seed=self.random_state
            ),
            "meta_stacking_accuracy": _block_bootstrap_ci(
                pred_meta == y_true, block_size=block_size, seed=self.random_state
            ),
            "soft_minus_xgb": _block_bootstrap_ci(
                pred_soft == y_true, pred_xgb == y_true, block_size=block_size, seed=self.random_state
            ),
        }

        self.metrics = {
            "target_mode": self.target_mode,
            "horizon": self.horizon,
            "min_move_atr": self.min_move_atr,
            "xgb_include_fvg_ob": self.xgb_include_fvg_ob,
            "xgb_reg_profile": self.xgb_reg_profile,
            "xgb_params": self.xgb_branch.best_params_,
            "decision_mode": self.decision_mode,
            "center_window_bars": self.center_window_bars if self.decision_mode == "rolling_center" else None,
            "center_min_periods": self.center_min_periods if self.decision_mode == "rolling_center" else None,
            "lstm_ensemble": self.lstm_ensemble,
            "lstm_ensemble_specs": [list(spec) for spec in self.lstm_ensemble_specs],
            "aligned_test_samples": int(len(y_true)),
            "majority_baseline_accuracy": xgb_res["majority_baseline_accuracy"],
            "branches": branches,
            "ensembles": ensembles,
            "confidence_tiers_soft": confidence_tiers,
            "calibration": calibration,
            "non_overlapping": non_overlapping,
            "block_bootstrap_95ci": bootstrap_cis,
            "selection": self.selection_summary,
        }

        print(
            f"\n=== Directional Accuracy Summary "
            f"(mode={self.target_mode}, H={self.horizon}, decision={self.decision_mode}) ==="
        )
        print(f"Aligned test samples : {len(y_true)} | Majority baseline: {xgb_res['majority_baseline_accuracy']:.2%}")
        for name, info in branches.items():
            print(
                f"  Solo {name.upper():8s} : Test={info['test_accuracy']:.2%} | "
                f"BalAcc={info['test_balanced_accuracy']:.2%} | AUC={info['test_roc_auc']:.4f} | "
                f"LL={info['test_log_loss']:.4f} | Brier={info['test_brier_score']:.4f} | "
                f"Bull={info['bullish_rate']:.1%}"
            )
        print("--- Ensembles ---")
        for name, info in ensembles.items():
            acc_str = f"{info['test_accuracy']:.2%}" if info["test_accuracy"] is not None else "N/A"
            bal_str = f"{info['test_balanced_accuracy']:.2%}" if info["test_balanced_accuracy"] is not None else "N/A"
            extra = (
                f" | AUC={info['test_roc_auc']:.4f} | LL={info['test_log_loss']:.4f} | "
                f"Brier={info['test_brier_score']:.4f}"
                if "test_roc_auc" in info
                else ""
            )
            print(f"  {name:16s} : Test={acc_str} | BalAcc={bal_str}{extra} | Cov={info['coverage']:.1%}")
        print(
            f"Calibration ECE -> soft_weighted={calibration['soft_weighted']['ece']:.4f}, "
            f"meta_stacking={calibration['meta_stacking']['ece']:.4f}"
        )
        print("--- High-Confidence Soft Ensemble ---")
        for tier, info in confidence_tiers.items():
            acc_str = f"{info['accuracy']:.2%}" if info["accuracy"] is not None else "N/A"
            conf_str = (
                f"{info['mean_predicted_confidence']:.2%}" if info["mean_predicted_confidence"] is not None else "N/A"
            )
            print(
                f"  Conf {tier:8s} : Acc={acc_str} (mean_conf={conf_str}) | "
                f"Cov={info['coverage']:.1%} (n={info['samples']})"
            )

        return self.metrics

    def walk_forward_evaluation(self, n_splits: int = 3):
        """Run expanding-window walk-forward directional accuracy evaluation."""
        if self._train_ohlcv is None or self._test_ohlcv is None:
            raise RuntimeError("Run feature_engineering() before walk_forward_evaluation()")

        candles = pd.concat([self._train_ohlcv, self._test_ohlcv])
        report = {
            "method": "expanding chronological windows; independently refit all 3 branches and ensemble combiners",
            "target_mode": self.target_mode,
            "horizon": self.horizon,
            "xgb_include_fvg_ob": self.xgb_include_fvg_ob,
            "xgb_reg_profile": self.xgb_reg_profile,
            "decision_mode": self.decision_mode,
            "center_window_bars": self.center_window_bars if self.decision_mode == "rolling_center" else None,
            "center_min_periods": self.center_min_periods if self.decision_mode == "rolling_center" else None,
            "gap_candles": self.purge_gap,
            "lstm_ensemble": self.lstm_ensemble,
            "lstm_ensemble_specs": [list(spec) for spec in self.lstm_ensemble_specs],
            "folds": [],
        }
        try:
            splits = list(TimeSeriesSplit(n_splits=n_splits, gap=self.purge_gap).split(candles))
        except ValueError as exc:
            report["error"] = str(exc)
            self.walk_forward_results = report
            return report

        print(
            f"\n=== Walk-Forward Directional Evaluation "
            f"({n_splits} expanding folds, H={self.horizon}, decision={self.decision_mode}) ==="
        )
        for fold_num, (train_idx, test_idx) in enumerate(splits, 1):
            train_slice, test_slice = candles.iloc[train_idx], candles.iloc[test_idx]
            fold_info = {
                "fold": fold_num,
                "train_period": [str(train_slice.index[0]), str(train_slice.index[-1])],
                "test_period": [str(test_slice.index[0]), str(test_slice.index[-1])],
                "train_candles": len(train_slice),
                "test_candles": len(test_slice),
            }
            print(
                f"\n--- Walk-Forward Fold {fold_num}/{n_splits}: "
                f"{fold_info['test_period'][0]} -> {fold_info['test_period'][1]} ---"
            )
            fold_model = XGBoostLSTMConsensusModel(
                symbol=self.symbol,
                timeframe=self.timeframe,
                horizon=self.horizon,
                cv_splits=self.cv_splits,
                confidence_threshold=self.confidence_threshold,
                sl_atr=self.sl_atr,
                tp_atr=self.tp_atr,
                max_bars=self.max_bars,
                calibration_method=self.calibration_method,
                target_mode=self.target_mode,
                min_move_atr=self.min_move_atr,
                xgb_include_fvg_ob=self.xgb_include_fvg_ob,
                xgb_reg_profile=self.xgb_reg_profile,
                use_sample_weights=self.use_sample_weights,
                decision_mode=self.decision_mode,
                center_window_bars=self.center_window_bars,
                center_min_periods=self.center_min_periods,
                lstm_lookback=self.lstm_lookback,
                lstm_units=self.lstm_units,
                lstm_target_threshold=self.lstm_target_threshold,
                lstm_epochs=self.lstm_epochs,
                lstm_batch_size=self.lstm_batch_size,
                lstm_confidence_threshold=self.lstm_confidence_threshold,
                lstm_ensemble=self.lstm_ensemble,
                lstm_ensemble_specs=self.lstm_ensemble_specs,
                random_state=self.random_state,
            )
            try:
                fold_model.feature_engineering(train_slice, test_slice, auxiliary_data=self._auxiliary_data)
                fold_model.train()
                metrics = fold_model.test()
                fold_info.update(status="ok", metrics=metrics)
            except ValueError as exc:
                fold_info.update(status="skipped", reason=str(exc))
                print(f"Fold {fold_num} skipped: {exc}")
            report["folds"].append(fold_info)

        self.walk_forward_results = report
        return report

    def predict_next_candle(self, ohlcv_data) -> pd.Series:
        """Return direction-implied proxy prices from the soft-weighted directional ensemble."""
        if (
            self.xgb_branch.model is None
            or any(member.model is None for member in self.lstm_branches)
            or self.logistic_branch.training_acc is None
        ):
            raise RuntimeError("Run train() before predicting")

        df = _normalize_ohlcv(ohlcv_data)
        feats = compute_features(
            df,
            include_fvg_ob=self.xgb_branch.include_fvg_ob,
            auxiliary_data=self._auxiliary_data,
            timeframe=self.timeframe,
        )[self.xgb_branch.X_train.columns]
        close = df["Close"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        prev_close = close.shift(1).fillna(close)
        tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        lstm_prob = self._predict_lstm_probabilities(df).reindex(df.index)
        logistic_prob = self.logistic_branch.predict_probabilities(df).reindex(df.index)
        valid = feats.notna().all(axis=1) & close.notna() & atr.notna() & lstm_prob.notna() & logistic_prob.notna()

        pred_price = pd.Series(np.nan, index=df.index, name="predicted_price", dtype=float)
        if not valid.any():
            return pred_price

        valid_idx = df.index[valid.to_numpy()]
        px_s = pd.Series(self.xgb_branch.predict_calibrated_proba(feats.loc[valid_idx]), index=valid_idx, dtype=float)
        pl_s = lstm_prob.loc[valid_idx]
        pg_s = logistic_prob.loc[valid_idx]

        if self.decision_mode == "rolling_center":
            p_xgb = _causal_rolling_logit_center(
                px_s, window_bars=self.center_window_bars, min_periods=self.center_min_periods
            ).to_numpy(dtype=float)
            p_lstm = _causal_rolling_logit_center(
                pl_s, window_bars=self.center_window_bars, min_periods=self.center_min_periods
            ).to_numpy(dtype=float)
            p_logistic = _causal_rolling_logit_center(
                pg_s, window_bars=self.center_window_bars, min_periods=self.center_min_periods
            ).to_numpy(dtype=float)
        else:
            p_xgb = px_s.to_numpy(dtype=float)
            p_lstm = pl_s.to_numpy(dtype=float)
            p_logistic = pg_s.to_numpy(dtype=float)

        wx, wl, wg = self.soft_weights
        p_soft = wx * p_xgb + wl * p_lstm + wg * p_logistic
        signed_dir = np.where(p_soft >= self.soft_threshold, 1.0, -1.0)
        edge = np.clip(np.abs(p_soft - self.soft_threshold) * 4.0, 0.25, 1.0)
        close_vals = close.loc[valid_idx].to_numpy(dtype=float)
        atr_vals = atr.loc[valid_idx].to_numpy(dtype=float)
        step_pct = np.clip(signed_dir * edge * (atr_vals / np.maximum(close_vals, 1e-10)), -0.05, 0.05)
        pred_price.loc[valid_idx] = close_vals * (1.0 + step_pct)
        return pred_price

    def evaluate(self):
        if self.metrics is None:
            raise RuntimeError("Run test() before evaluate()")

        record = {
            "symbol": self.symbol,
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": self.MODEL_NAME,
            "timeframe": self.timeframe,
            "target_mode": self.target_mode,
            "horizon": self.horizon,
            "min_move_atr": self.min_move_atr,
            "xgb_include_fvg_ob": self.xgb_include_fvg_ob,
            "xgb_reg_profile": self.xgb_reg_profile,
            "decision_mode": self.decision_mode,
            "center_window_bars": self.center_window_bars if self.decision_mode == "rolling_center" else None,
            "center_min_periods": self.center_min_periods if self.decision_mode == "rolling_center" else None,
            "xgb_n_features": int(self.xgb_branch.X_train.shape[1]),
            "logistic_n_features": len(self.logistic_branch.features),
            "train_time": float(self.train_time),
            "xgb_params": self.xgb_branch.best_params_,
            "xgb_decision_threshold": self.xgb_branch.decision_threshold,
            "xgb_calibration_method": self.xgb_branch.calibration_method,
            "lstm_ensemble": self.lstm_ensemble,
            "lstm_ensemble_specs": [list(spec) for spec in self.lstm_ensemble_specs],
            "lstm_multi_timeframes": list(self._lstm_multi_timeframes),
            "lstm_lookback": self.lstm_branch.lookback,
            "lstm_units": self.lstm_branch.lstm_units,
            "lstm_decision_threshold": self.lstm_decision_threshold,
            "logistic_C": self.logistic_branch.best_C,
            "logistic_decision_threshold": self.logistic_branch.decision_threshold,
            "soft_weights": self.selection_summary.get("soft_weights"),
            "selected_policy": self.selected_policy,
            "metrics": self.metrics,
            "walk_forward": self.walk_forward_results,
        }
        results = []
        if self.results_path.exists():
            with self.results_path.open() as f:
                results = json.load(f)
        results.append(record)
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with self.results_path.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"Consensus results saved to {self.results_path}")
        return record
