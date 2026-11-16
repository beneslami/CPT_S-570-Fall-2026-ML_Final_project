"""LSTM classifier for the direction of the next completed candle."""

import json
import importlib.util
import os
import pathlib
import time
from datetime import datetime
from typing import Optional
import matplotlib

# This module only writes plots to disk; a non-interactive backend is safer for
# command-line training and must be selected before pyplot or Keras is loaded.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Keras 3 needs one backend selected before Keras is imported.
if "KERAS_BACKEND" not in os.environ:
    for backend_name in ("tensorflow", "jax", "torch"):
        if importlib.util.find_spec(backend_name) is not None:
            os.environ["KERAS_BACKEND"] = backend_name
            break

try:
    import keras
    from keras import Input, Sequential
    from keras.callbacks import EarlyStopping
    from keras.layers import Dense, Dropout, LSTM as KerasLSTM
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "Keras 3 needs a backend framework. Install TensorFlow, JAX, or PyTorch "
        "before importing models.LSTM."
    ) from error
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.preprocessing import StandardScaler

from ML_interface import Model
from strategy import Strategy


class LSTMModel(Model):
    DEFAULT_LOOKBACK_CANDIDATES = (10, 25, 50, 72, 100, 200)
    DEFAULT_TARGET_THRESHOLD_CANDIDATES = (0.01, 0.02, 0.05)
    DEFAULT_CONFIDENCE_THRESHOLD_CANDIDATES = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
    DEFAULT_LSTM_UNIT_CANDIDATES = (4, 8, 16, 50)
    # Triple-barrier target: SL distance in ATRs and alpha = TP / SL (see strategy.py)
    DEFAULT_SL_ATR_CANDIDATES = (1.0, 1.5, 2.0)
    DEFAULT_ALPHA_CANDIDATES = (1.0, 1.5, 2.0)
    MODEL_NAME = "LSTM"

    def __init__(self, symbol: str, timeframe: str, results_path=None, lookback: int | str = "auto", target_threshold: float | str = "auto", lstm_units: int | str = "auto", epochs: int = 50, batch_size: int = 64, confidence_threshold: float | str = "auto", model_path: Optional[str] = None, random_seed: int = 42, lookback_candidates=None, target_threshold_candidates=None, lstm_unit_candidates=None, confidence_threshold_candidates=None, min_confident_coverage: float = 0.10, multi_timeframes=("15m", "4h"), target: str = "direction", sl_atr_candidates=None, alpha_candidates=None, horizon: int = 30, cost_pct: float = 0.001, min_trades: int = 50):
        if target not in {"direction", "triple_barrier"}:
            raise ValueError("target must be 'direction' or 'triple_barrier'")
        if horizon < 1 or cost_pct < 0 or min_trades < 1:
            raise ValueError("horizon and min_trades must be positive and cost_pct non-negative")
        if lookback != "auto" and (not isinstance(lookback, int) or lookback < 2):
            raise ValueError("lookback must be 'auto' or an integer of at least 2 candles")
        if target_threshold != "auto" and (not isinstance(target_threshold, (int, float)) or target_threshold < 0):
            raise ValueError("target_threshold must be 'auto' or a non-negative ATR fraction")
        if lstm_units != "auto" and (not isinstance(lstm_units, int) or lstm_units < 1):
            raise ValueError("lstm_units must be 'auto' or a positive integer")
        if epochs < 1 or batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if confidence_threshold != "auto" and (not isinstance(confidence_threshold, (int, float)) or not 0.5 <= confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be 'auto' or between 0.5 and 1.0")
        if not 0 < min_confident_coverage <= 1:
            raise ValueError("min_confident_coverage must be in (0, 1]")

        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback_mode = lookback
        self.lookback_candidates = tuple(lookback_candidates or self.DEFAULT_LOOKBACK_CANDIDATES)
        if not self.lookback_candidates or any(not isinstance(candidate, int) or candidate < 2 for candidate in self.lookback_candidates):
            raise ValueError("lookback_candidates must contain integers of at least 2")
        self.lookback = None if lookback == "auto" else lookback
        self.target_threshold_mode = target_threshold
        self.target_threshold_candidates = tuple(target_threshold_candidates or self.DEFAULT_TARGET_THRESHOLD_CANDIDATES)
        if not self.target_threshold_candidates or any(not isinstance(candidate, (int, float)) or candidate < 0 for candidate in self.target_threshold_candidates):
            raise ValueError("target_threshold_candidates must contain non-negative ATR fractions")
        self.target_threshold = (None if target_threshold == "auto" else float(target_threshold))
        # target = "direction": label from the next candle's body (the original target).
        # target = "triple_barrier": label from a long trade opened at the candle close,
        #   SL = sl_atr x ATR, TP = alpha x SL, closed after `horizon` candles at the latest.
        self.target = target
        self.sl_atr_candidates = tuple(float(value) for value in (sl_atr_candidates or self.DEFAULT_SL_ATR_CANDIDATES))
        self.alpha_candidates = tuple(float(value) for value in (alpha_candidates or self.DEFAULT_ALPHA_CANDIDATES))
        if any(value <= 0 for value in self.sl_atr_candidates + self.alpha_candidates):
            raise ValueError("sl_atr_candidates and alpha_candidates must be positive")
        self.sl_atr = None
        self.alpha = None
        self.horizon = horizon
        self.cost_pct = cost_pct            # round-trip cost as a fraction of price
        self.min_trades = min_trades        # fewest validation trades for a setting to be selectable
        self.p0_train = None                # no-model take-profit rate on training candles
        # Per-sequence trade outcomes (triple barrier only): result in R before costs, and costs in R
        self.validation_r = self.validation_cost_r = None
        self.test_r = self.test_cost_r = None
        self.validation_trading = None      # trading statistics of the selected setting on validation
        self.lstm_units_mode = lstm_units
        self.lstm_unit_candidates = tuple(lstm_unit_candidates or self.DEFAULT_LSTM_UNIT_CANDIDATES)
        if not self.lstm_unit_candidates or any(not isinstance(candidate, int) or candidate < 1 for candidate in self.lstm_unit_candidates):
            raise ValueError("lstm_unit_candidates must contain positive integers")
        self.lstm_units = None if lstm_units == "auto" else lstm_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.confidence_threshold_mode = confidence_threshold
        self.confidence_threshold_candidates = tuple(confidence_threshold_candidates or self.DEFAULT_CONFIDENCE_THRESHOLD_CANDIDATES)
        if not self.confidence_threshold_candidates or any(not isinstance(candidate, (int, float)) or not 0.5 <= candidate <= 1.0 for candidate in self.confidence_threshold_candidates):
            raise ValueError("confidence_threshold_candidates must be between 0.5 and 1.0")
        self.confidence_threshold = (None if confidence_threshold == "auto" else float(confidence_threshold))
        self.min_confident_coverage = min_confident_coverage
        self.decision_threshold = 0.5
        self.model_path = pathlib.Path(model_path) if model_path else None
        self.random_seed = random_seed
        self.feature_names = ["body_pct", "cci_scaled"]#, "range_pct", "upper_wick_pct", "lower_wick_pct", "return_1", "return_3", "return_6", "return_12", "atr_pct", "macd_hist_pct", "rsi_centered", "relative_volume_10", "volume_delta_ratio",]
        self.primary_feature_names = list(self.feature_names)
        self.multi_timeframes = tuple(multi_timeframes)

        project_dir = pathlib.Path(__file__).resolve().parents[2]
        self.results_path = (pathlib.Path(results_path) if results_path else project_dir / "results" / f"lstm_{symbol}_{timeframe}.json")

        self.scaler = StandardScaler()
        self.model = None
        self.history = None
        self.split_method = ""
        self.train_times = self.validation_times = self.test_times = None
        self.X_train = self.X_validation = self.X_test = None
        self.y_train = self.y_validation = self.y_test = None
        self.y_probability = self.y_prediction = None
        self.train_time = 0.0
        self.training_accuracy = None
        self.validation_accuracy = None
        self.test_accuracy = None
        self.train_target_coverage = None
        self.validation_target_coverage = None
        self.test_target_coverage = None
        self.typical_abs_return = None
        self.test_current_close = None
        self.test_actual_next_close = None
        self.test_predicted_next_close = None
        self.lookback_trials = []
        self.target_threshold_trials = []
        self._train_frame = self._validation_frame = self._test_frame = None
        np.random.seed(self.random_seed)

    def _validate_frame(self, data: pd.DataFrame, name: str):
        required = {"time", "open", "high", "low", "close", "volume", "volume_delta", "rsi_14", "cci", "atr_14", "macd_histogram"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
        if data.empty:
            raise ValueError(f"{name} is empty")
        times = pd.to_datetime(data["time"], utc=True, errors="raise")
        if not times.is_monotonic_increasing or times.duplicated().any():
            raise ValueError(f"{name} must have unique, increasing timestamps")

    def _base_numeric_features(self, data):
        open_price = pd.to_numeric(data["open"], errors="coerce")
        high = pd.to_numeric(data["high"], errors="coerce")
        low = pd.to_numeric(data["low"], errors="coerce")
        close = pd.to_numeric(data["close"], errors="coerce")
        volume = pd.to_numeric(data["volume"], errors="coerce")
        volume_delta = pd.to_numeric(data["volume_delta"], errors="coerce")
        atr = pd.to_numeric(data["atr_14"], errors="coerce")
        macd_histogram = pd.to_numeric(data["macd_histogram"], errors="coerce")

        features = pd.DataFrame(index=data.index)

        features["body_pct"] = (close - open_price) / open_price
        features["range_pct"] = (high - low) / open_price
        candle_range = (high - low).replace(0, np.nan)
        upper_wick = high - np.maximum(open_price, close)
        lower_wick = np.minimum(open_price, close) - low
        features["wick_imbalance"] = (lower_wick - upper_wick) / candle_range
        features["atr_pct"] = atr / close.replace(0, np.nan)
        if "relative_volume" in data.columns:
            features["relative_volume"] = pd.to_numeric(data["relative_volume"], errors="coerce")
        else:
            volume_median_20 = volume.rolling(window=20, min_periods=20).median()
            features["relative_volume"] = volume / volume_median_20.replace(0, np.nan)
        features["relative_volume"] = features["relative_volume"].clip(lower=0, upper=10)
        features["volume_delta_ratio"] = volume_delta / volume.replace(0, np.nan)
        features["macd_hist_atr"] = macd_histogram / atr.replace(0, np.nan)
        features["cci_scaled"] = (pd.to_numeric(data["cci"], errors="coerce").clip(-300, 300) / 100)
        """features["range_pct"] = (high - low) / open_price
        features["upper_wick_pct"] = (high - np.maximum(open_price, close)) / open_price
        features["lower_wick_pct"] = (np.minimum(open_price, close) - low) / open_price

        features["return_1"] = close.pct_change(1)
        features["return_3"] = close.pct_change(3)
        features["return_6"] = close.pct_change(6)
        features["return_12"] = close.pct_change(12)

        features["atr_pct"] = atr / close
        features["macd_hist_pct"] = (pd.to_numeric(data["macd_histogram"], errors="coerce") / close)
        features["rsi_centered"] = (pd.to_numeric(data["rsi_14"], errors="coerce") - 50) / 50
        """
        features = features.replace([np.inf, -np.inf], np.nan)
        return features

    def _numeric_features(self, data):
        """Return the selected primary and causally aligned auxiliary features."""
        primary_features = self._base_numeric_features(data)
        features = pd.DataFrame(index=data.index)
        missing_features = []
        for feature_name in self.feature_names:
            if feature_name in primary_features:
                features[feature_name] = primary_features[feature_name]
            elif feature_name in data.columns:
                features[feature_name] = pd.to_numeric(data[feature_name], errors="coerce")
            else:
                missing_features.append(feature_name)
        if missing_features:
            raise ValueError(f"Selected features are not computed or aligned: {sorted(missing_features)}")
        return features.replace([np.inf, -np.inf], np.nan)

    def _align_auxiliary_features(self, primary_data, auxiliary_data):
        """Attach only auxiliary candles completed by the primary candle close.

        Dataset timestamps mark candle *opens*.  A 4h row stamped 00:00 is not
        known at 01:00, 02:00, or 03:00; it becomes usable only at 04:00.  The
        as-of join therefore aligns auxiliary close times to primary close times.
        """
        aligned = primary_data.copy()
        aligned["time"] = pd.to_datetime(aligned["time"], utc=True)
        primary_duration = pd.Timedelta(self.timeframe)
        aligned["primary_close_time"] = aligned["time"] + primary_duration
        aligned = aligned.sort_values("time").reset_index(drop=True)
        self.feature_names = list(self.primary_feature_names)
        for timeframe in self.multi_timeframes:
            if timeframe not in auxiliary_data:
                raise ValueError(f"Missing auxiliary data for timeframe {timeframe}")
            auxiliary = auxiliary_data[timeframe].copy()
            # Auxiliary frames only contribute the features configured in
            # ``primary_feature_names``.  The current feature set needs open,
            # close, and CCI, so do not require unrelated indicator columns.
            # This permits memory-safe loading of the very large 1-minute file.
            self._validate_auxiliary_frame(auxiliary, f"auxiliary_data[{timeframe}]")
            auxiliary["time"] = pd.to_datetime(auxiliary["time"], utc=True)
            auxiliary_close_time = auxiliary["time"] + pd.Timedelta(timeframe)
            auxiliary_features = self._base_numeric_features(auxiliary)
            missing = set(self.primary_feature_names).difference(auxiliary_features.columns)
            if missing:
                raise ValueError(f"Auxiliary timeframe {timeframe} cannot compute: {sorted(missing)}")
            prefix = f"{timeframe}_"
            right = pd.DataFrame({"auxiliary_close_time": auxiliary_close_time})
            for feature_name in self.primary_feature_names:
                column_name = f"{prefix}{feature_name}"
                right[column_name] = auxiliary_features[feature_name].to_numpy()
                self.feature_names.append(column_name)
            aligned = pd.merge_asof(aligned, right.sort_values("auxiliary_close_time"), left_on="primary_close_time", right_on="auxiliary_close_time", direction="backward", allow_exact_matches=True,).drop(columns="auxiliary_close_time")
        return aligned.drop(columns="primary_close_time")

    @staticmethod
    def _validate_auxiliary_frame(data: pd.DataFrame, name: str):
        required = {
            "time", "open", "high", "low", "close", "volume", "volume_delta",
            "atr_14", "macd_histogram", "cci",
        }
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
        if data.empty:
            raise ValueError(f"{name} is empty")
        times = pd.to_datetime(data["time"], utc=True, errors="raise")
        if not times.is_monotonic_increasing or times.duplicated().any():
            raise ValueError(f"{name} must have unique, increasing timestamps")

    # ------------------------------------------------------------------
    # Target settings: "direction" uses target_threshold, "triple_barrier" uses (sl_atr, alpha)
    # ------------------------------------------------------------------
    def _target_candidates(self):
        """All target settings the search should try, as dictionaries."""
        if self.target == "direction":
            thresholds = self.target_threshold_candidates if self.target_threshold_mode == "auto" else (self.target_threshold,)
            return [{"target_threshold": float(t)} for t in thresholds]
        return [{"sl_atr": s, "alpha": a} for s in self.sl_atr_candidates for a in self.alpha_candidates]

    def _apply_target(self, setting):
        for name, value in setting.items():
            setattr(self, name, value)

    def _current_target_setting(self):
        if self.target == "direction":
            return {"target_threshold": self.target_threshold}
        return {"sl_atr": self.sl_atr, "alpha": self.alpha}

    def _target_tag(self):
        """Short text used in plot file names."""
        if self.target == "direction":
            return f"atr{self.target_threshold * 100:g}pct".replace(".", "p")
        return f"sl{self.sl_atr:g}_a{self.alpha:g}_h{self.horizon}".replace(".", "p")

    def _target_description(self):
        if self.target == "direction":
            return (f"next candle body > {self.target_threshold:.0%} of current ATR: bullish; "
                    f"next candle body < -{self.target_threshold:.0%} of current ATR: bearish; smaller moves omitted")
        return (f"long trade opened at the candle close with SL = {self.sl_atr:g} ATR and TP = {self.alpha:g} x SL; "
                f"1 if TP is hit first, 0 if SL is hit first or neither is hit within {self.horizon} candles "
                "(SL and TP inside the same candle counts as SL)")

    def _triple_barrier(self, data: pd.DataFrame):
        """Simulate a long trade at every candle close.

        Returns three arrays (NaN where fewer than `horizon` future candles exist):
            win    : 1 if TP is hit before SL, otherwise 0
            r      : trade result in R before costs (+alpha, -1, or the open result at the time limit)
            cost_r : round-trip cost in R = cost_pct x entry / SL distance
        """
        close = pd.to_numeric(data["close"], errors="coerce").to_numpy(dtype=float)
        high = pd.to_numeric(data["high"], errors="coerce").to_numpy(dtype=float)
        low = pd.to_numeric(data["low"], errors="coerce").to_numpy(dtype=float)
        atr = pd.to_numeric(data["atr_14"], errors="coerce").to_numpy(dtype=float)
        n = len(close)
        win = np.full(n, np.nan)
        r = np.full(n, np.nan)
        cost_r = np.full(n, np.nan)
        for t in range(n - self.horizon):
            entry = close[t]
            sl_distance = self.sl_atr * atr[t]
            if not np.isfinite(entry) or not np.isfinite(sl_distance) or sl_distance <= 0:
                continue
            stop = entry - sl_distance
            target = entry + self.alpha * sl_distance
            win[t], r[t] = 0, (close[t + self.horizon] - entry) / sl_distance    # timeout unless a barrier is hit
            for j in range(t + 1, t + self.horizon + 1):
                if low[j] <= stop:            # checked first: a candle touching both counts as a loss
                    r[t] = -1.0
                    break
                if high[j] >= target:
                    win[t], r[t] = 1, self.alpha
                    break
            cost_r[t] = self.cost_pct * entry / sl_distance
        return win, r, cost_r

    def _make_sequences(self, data: pd.DataFrame, fit_scaler: bool):
        """End each sequence at candle t and label it from what happens after t.

        Returns X, y, end times, label coverage, and (triple barrier only) the R
        outcome and the cost in R of a trade opened at each sequence end.
        """
        features = self._numeric_features(data)
        outcome_r = outcome_cost_r = None
        if self.target == "direction":
            next_body = (pd.to_numeric(data["close"], errors="coerce").shift(-1) - pd.to_numeric(data["open"], errors="coerce").shift(-1))
            current_atr = pd.to_numeric(data["atr_14"], errors="coerce")
            minimum_move = self.target_threshold * current_atr
            labels = pd.Series(np.nan, index=data.index)
            labels.loc[next_body > minimum_move] = 1
            labels.loc[next_body < -minimum_move] = 0
        else:
            win, outcome_r, outcome_cost_r = self._triple_barrier(data)
            labels = pd.Series(win, index=data.index)

        if fit_scaler:
            scale_rows = features.notna().all(axis=1)
            if scale_rows.sum() < self.lookback:
                raise ValueError("Not enough valid training rows to fit the scaler")
            self.scaler.fit(features.loc[scale_rows])
        scaled = pd.DataFrame(self.scaler.transform(features), columns=self.feature_names, index=data.index,)

        sequences = []
        targets = []
        sequence_times = []
        trade_r = []
        trade_cost_r = []
        parsed_times = pd.to_datetime(data["time"], utc=True)
        for end in range(self.lookback - 1, len(data) - 1):
            start = end - self.lookback + 1
            window = scaled.iloc[start : end + 1]
            if window.isna().any().any() or pd.isna(labels.iloc[end]):
                continue
            sequences.append(window.to_numpy(dtype=np.float32))
            targets.append(int(labels.iloc[end]))
            sequence_times.append(parsed_times.iloc[end])
            if outcome_r is not None:
                trade_r.append(outcome_r[end])
                trade_cost_r.append(outcome_cost_r[end])

        if not sequences:
            raise ValueError("No complete LSTM sequences could be created")
        possible_sequences = max(1, len(data) - self.lookback)
        target_coverage = len(sequences) / possible_sequences
        outcomes = None
        if outcome_r is not None:
            outcomes = (np.asarray(trade_r, dtype=float), np.asarray(trade_cost_r, dtype=float))
        return (np.asarray(sequences, dtype=np.float32), np.asarray(targets, dtype=np.float32), pd.DatetimeIndex(sequence_times, name="feature_end_time"), target_coverage, outcomes)

    @staticmethod
    def _aligned_closes(data: pd.DataFrame, sequence_times):
        prices = pd.DataFrame({"time": pd.to_datetime(data["time"], utc=True), "close": pd.to_numeric(data["close"], errors="coerce"),}).set_index("time")
        prices["next_close"] = prices["close"].shift(-1)
        aligned = prices.reindex(sequence_times)
        return (aligned["close"].to_numpy(dtype=float), aligned["next_close"].to_numpy(dtype=float),)

    def _build_model(self):
        keras.utils.set_random_seed(self.random_seed)
        model = Sequential([Input(shape=(self.lookback, len(self.feature_names))), KerasLSTM(self.lstm_units), Dropout(0.10), Dense(1, activation="sigmoid"), ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"],)
        return model

    def _prepare_sequences(self):
        """Build one candidate's sequences after ``self.lookback`` is set."""
        self.scaler = StandardScaler()
        (self.X_train, self.y_train, self.train_times, self.train_target_coverage, _) = self._make_sequences(self._train_frame, fit_scaler=True)
        (self.X_validation, self.y_validation, self.validation_times, self.validation_target_coverage, validation_outcomes) = self._make_sequences(self._validation_frame, fit_scaler=False)
        (self.X_test, self.y_test, self.test_times, self.test_target_coverage, test_outcomes) = self._make_sequences(self._test_frame, fit_scaler=False)
        if self.target == "triple_barrier":
            # p0 is measured on training candles only: how often the TP is hit with no model at all
            self.p0_train = float(np.mean(self.y_train))
            self.validation_r, self.validation_cost_r = validation_outcomes
            self.test_r, self.test_cost_r = test_outcomes

        train_current_close, train_actual_next_close = self._aligned_closes(self._train_frame, self.train_times,)
        train_abs_return = np.abs(train_actual_next_close / train_current_close - 1.0)
        self.typical_abs_return = float(np.nanmedian(train_abs_return))
        self.test_current_close, self.test_actual_next_close = (self._aligned_closes(self._test_frame, self.test_times))

        if min(len(self.X_train), len(self.X_validation), len(self.X_test)) < 20:
            raise ValueError("Need at least 20 complete sequences in each split")
        if len(np.unique(self.y_train)) != 2 or len(np.unique(self.y_validation)) != 2:
            raise ValueError("Training and validation sequences must each contain bullish and bearish labels")
        self.model = self._build_model()

    def feature_engineering(self, train_data, test_data, auxiliary_data=None):
        self._validate_frame(train_data, "train_data")
        self._validate_frame(test_data, "test_data")
        auxiliary_data = auxiliary_data or {}
        if self.multi_timeframes:
            train_data = self._align_auxiliary_features(train_data, auxiliary_data)
            test_data = self._align_auxiliary_features(test_data, auxiliary_data)
        else:
            self.feature_names = list(self.primary_feature_names)

        if train_data.equals(test_data):
            test_start = int(len(train_data) * 0.8)
            train_and_validation = train_data.iloc[:test_start].copy()
            test_frame = test_data.iloc[test_start:].copy()
            self.split_method = "same_csv_chronological_64_16_20"
        else:
            train_end = pd.to_datetime(train_data["time"].iloc[-1], utc=True)
            test_start_time = pd.to_datetime(test_data["time"].iloc[0], utc=True)
            if train_end >= test_start_time:
                raise ValueError("Training data must end before test data starts")
            train_and_validation = train_data.copy()
            test_frame = test_data.copy()
            self.split_method = "separate_chronological_files_80_20_train_validation"

        validation_start = int(len(train_and_validation) * 0.8)
        train_frame = train_and_validation.iloc[:validation_start].copy()
        validation_frame = train_and_validation.iloc[validation_start:].copy()

        self._train_frame = train_frame
        self._validation_frame = validation_frame
        self._test_frame = test_frame
        # Fixed hyperparameters preserve the former one-model behaviour. In auto
        # mode, train() builds candidates but does not score test data to select one.
        if (self.target == "direction" and self.lookback is not None and self.target_threshold is not None and self.lstm_units is not None):
            self._prepare_sequences()

    def _trading_stats(self, q, y, r, cost_r):
        """Trade every candle where the model's q clears the Strategy threshold, and measure the result.

        q      : model probability that the TP is hit first, per candle
        y      : 1 if the TP was actually hit first, otherwise 0
        r      : actual trade result in R before costs (+alpha, -1, or the open result at the time limit)
        cost_r : cost of the trade in R (c / SL)
        Threshold per candle (Strategy.threshold): q > p0_train + cost_r / (1 + alpha).
        """
        strategy = Strategy(alpha=self.alpha, sl_atr=self.sl_atr, cost_pct=self.cost_pct)
        q = np.asarray(q, dtype=float)
        y = np.asarray(y, dtype=float)
        threshold =self.p0_train + cost_r / (1.0 + self.alpha)
        take = q > threshold
        n_trades = int(take.sum())
        p0_split = float(np.mean(y))            # no-model TP rate on this split
        stats = {
            "n_candles": int(len(q)),
            "n_trades": n_trades,
            "trade_share": n_trades / len(q),
            "p0_train": self.p0_train,
            "p0": p0_split,
            "p0_theory": strategy.break_even_win_rate(),
            "mean_threshold": float(np.mean(threshold)),
            "mean_cost_r": float(np.mean(cost_r)),
            "no_model_expectancy_net_r": float(np.mean(r - cost_r)),   # trading every candle
            "p": None, "edge": None, "timeout_share": None, "mean_gross_r": None,
            "expectancy_net_r": None, "expectancy_net_r_95pct": None, "expectancy_net_formula_r": None,
        }
        if n_trades > 0:
            p = float(np.mean(y[take]))
            net_r = r[take] - cost_r[take]
            half_width = 1.96 * float(np.std(net_r)) / np.sqrt(n_trades)
            stats.update({
                "p": p,
                "edge": p - p0_split,
                "timeout_share": float(np.mean((r[take] != self.alpha) & (r[take] != -1.0))),
                "mean_gross_r": float(np.mean(r[take])),
                "mean_trade_cost_r": float(np.mean(cost_r[take])),
                # Measured: the average real result of the trades taken (timeouts at their real R)
                "expectancy_net_r": float(np.mean(net_r)),
                "expectancy_net_r_95pct": [float(np.mean(net_r)) - half_width, float(np.mean(net_r)) + half_width],
                # Formula (1 + alpha)(p - p0) - c/SL; equals the measured value only when every non-win loses 1R
                "expectancy_net_formula_r": strategy.expectancy(p, p0_split, float(np.mean(cost_r[take]))),
            })
        # Calibration: when the model says q, how often is the TP really hit?
        bins = [0.0, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 1.0]
        calibration = []
        for low, high in zip(bins[:-1], bins[1:]):
            in_bin = (q >= low) & (q < high)
            if in_bin.any():
                calibration.append({"q_range": [low, high], "count": int(in_bin.sum()),
                                    "mean_q": float(np.mean(q[in_bin])), "actual_tp_rate": float(np.mean(y[in_bin]))})
        stats["calibration"] = calibration
        return stats

    def _selection_score(self, trial):
        """Smaller is better. Direction: the original accuracy score. Triple barrier: validation E_net."""
        if self.target == "direction":
            return (-trial["coverage_adjusted_validation_balanced_acc"], trial["validation_log_loss"])
        stats = trial["validation_trading"]
        if stats["n_trades"] < self.min_trades:
            return (1, -stats["n_trades"], trial["validation_log_loss"])   # too few trades: ranked after every valid setting
        return (0, -stats["expectancy_net_r"], trial["validation_log_loss"])

    def _train_current_lookback(self):
        early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True,)
        started = time.time()
        self.history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_validation, self.y_validation), epochs=self.epochs, batch_size=self.batch_size, shuffle=False, callbacks=[early_stopping], verbose=getattr(self, "training_verbose", 1),)
        self.train_time = time.time() - started
        train_probability = self.model.predict(self.X_train, batch_size=self.batch_size, verbose=0).reshape(-1)
        validation_probability = self.model.predict(self.X_validation, batch_size=self.batch_size, verbose=0).reshape(-1)

        thresholds = np.arange(0.40, 0.61, 0.01)
        threshold_scores = [balanced_accuracy_score(self.y_validation, validation_probability >= threshold,) for threshold in thresholds]
        self.decision_threshold = float(thresholds[np.argmax(threshold_scores)])

        self.training_accuracy = float(accuracy_score(self.y_train, train_probability >= self.decision_threshold,))
        self.validation_accuracy = float(accuracy_score(self.y_validation, validation_probability >= self.decision_threshold,))
        validation_loss = float(log_loss(self.y_validation, validation_probability, labels=[0, 1]))
        validation_balanced_accuracy = float(balanced_accuracy_score(self.y_validation, validation_probability >= self.decision_threshold,))
        validation_prediction = (validation_probability >= self.decision_threshold).astype(int)
        validation_confidence = np.where(validation_prediction == 1, validation_probability, 1.0 - validation_probability,)
        candidates = (self.confidence_threshold_candidates if self.confidence_threshold_mode == "auto" else (self.confidence_threshold,))
        best_confidence_score = None
        for threshold in candidates:
            confident = validation_confidence >= threshold
            coverage = float(confident.mean())
            if coverage < self.min_confident_coverage:
                continue
            selective_accuracy = float(accuracy_score(self.y_validation[confident], validation_prediction[confident]))
            score = (-selective_accuracy, -coverage)
            if best_confidence_score is None or score < best_confidence_score:
                best_confidence_score = score
                self.confidence_threshold = float(threshold)
                selected_coverage = coverage
                selected_accuracy = selective_accuracy
        if best_confidence_score is None:
            raise ValueError("No confidence threshold meets min_confident_coverage")
        if self.target == "triple_barrier":
            self.validation_trading = self._trading_stats(validation_probability, self.y_validation, self.validation_r, self.validation_cost_r)
        return (validation_loss, validation_balanced_accuracy, selected_coverage, selected_accuracy,)

    def train(self):
        if self._train_frame is None:
            raise RuntimeError("Run feature_engineering before train")

        lookback_candidates = (self.lookback_candidates if self.lookback_mode == "auto" else (self.lookback,) )
        lstm_unit_candidates = (self.lstm_unit_candidates if self.lstm_units_mode == "auto" else (self.lstm_units,))
        self.lookback_trials = []
        self.target_threshold_trials = self.lookback_trials
        best_state = None
        best_score = None
        for target_setting in self._target_candidates():
            for lookback in lookback_candidates:
                for lstm_units in lstm_unit_candidates:
                    self._apply_target(target_setting)
                    self.lookback = lookback
                    self.lstm_units = lstm_units
                    self._prepare_sequences()
                    (validation_loss, validation_balanced_accuracy, validation_confident_coverage, validation_confident_accuracy) = self._train_current_lookback()
                    coverage_adjusted_balanced_accuracy = (validation_balanced_accuracy * self.validation_target_coverage)
                    trial = {
                        "target_setting": dict(target_setting),
                        "target_threshold_atr_fraction": self.target_threshold,
                        "lookback": lookback,
                        "lstm_units": self.lstm_units,
                        "n_train": len(self.X_train),
                        "n_validation": len(self.X_validation),
                        "train_target_coverage": self.train_target_coverage,
                        "validation_target_coverage": self.validation_target_coverage,
                        "training_acc": self.training_accuracy,
                        "training_error": 1.0 - self.training_accuracy,
                        "validation_acc": self.validation_accuracy,
                        "validation_error": 1.0 - self.validation_accuracy,
                        "validation_balanced_acc": validation_balanced_accuracy,
                        "coverage_adjusted_validation_balanced_acc": coverage_adjusted_balanced_accuracy,
                        "validation_log_loss": validation_loss,
                        "confidence_threshold": self.confidence_threshold,
                        "validation_confident_coverage": validation_confident_coverage,
                        "validation_confident_accuracy": validation_confident_accuracy,
                        "epochs_trained": len(self.history.history["loss"]),
                        "train_time": self.train_time,
                        "validation_trading": self.validation_trading if self.target == "triple_barrier" else None,
                    }
                    self.lookback_trials.append(trial)
                    # Select the configuration by validation only; test data remain untouched until final evaluation.
                    score = self._selection_score(trial)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_state = {
                            "model": self.model,
                            "scaler": self.scaler,
                            "history": self.history,
                            "lookback": self.lookback,
                            "target_threshold": self.target_threshold,
                            "lstm_units": self.lstm_units,
                            "decision_threshold": self.decision_threshold,
                            "confidence_threshold": self.confidence_threshold,
                            "training_accuracy": self.training_accuracy,
                            "validation_accuracy": self.validation_accuracy,
                            "train_time": self.train_time,
                            "X_train": self.X_train,
                            "y_train": self.y_train,
                            "train_times": self.train_times,
                            "train_target_coverage": self.train_target_coverage,
                            "X_validation": self.X_validation,
                            "y_validation": self.y_validation,
                            "validation_times": self.validation_times,
                            "validation_target_coverage": self.validation_target_coverage,
                            "X_test": self.X_test,
                            "y_test": self.y_test,
                            "test_times": self.test_times,
                            "test_target_coverage": self.test_target_coverage,
                            "typical_abs_return": self.typical_abs_return,
                            "test_current_close": self.test_current_close,
                            "test_actual_next_close": self.test_actual_next_close,
                            "sl_atr": self.sl_atr,
                            "alpha": self.alpha,
                            "p0_train": self.p0_train,
                            "validation_r": self.validation_r,
                            "validation_cost_r": self.validation_cost_r,
                            "test_r": self.test_r,
                            "test_cost_r": self.test_cost_r,
                            "validation_trading": self.validation_trading,
                        }

        for name, value in best_state.items():
            setattr(self, name, value)
        if self.model_path is not None:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.model_path)
        return self.training_accuracy

    def test(self):
        if self.model is None or self.history is None:
            raise RuntimeError("Run train before test")
        self.y_probability = self.model.predict(self.X_test, batch_size=self.batch_size, verbose=0).reshape(-1)
        self.y_prediction = (self.y_probability >= self.decision_threshold).astype(int)
        self.test_accuracy = float(accuracy_score(self.y_test, self.y_prediction))
        return self.test_accuracy

    def predict_next(self, recent_data: pd.DataFrame, auxiliary_data=None) -> dict:
        """Predict the candle immediately after the final completed input candle."""
        if self.model is None or self.history is None:
            raise RuntimeError("Train the model before predicting")
        self._validate_frame(recent_data, "recent_data")
        if self.multi_timeframes:
            recent_data = self._align_auxiliary_features(recent_data, auxiliary_data or {})
        features = self._numeric_features(recent_data).tail(self.lookback)
        if len(features) != self.lookback or features.isna().any().any():
            raise ValueError(f"Need {self.lookback} complete recent candles for prediction")
        scaled = self.scaler.transform(features).astype(np.float32)
        probability = float(self.model.predict(scaled[None, :, :], verbose=0)[0, 0])
        direction = ("bullish" if probability >= self.decision_threshold else "bearish")
        confidence = (probability if direction == "bullish" else 1.0 - probability)
        return {
            "as_of_time": str(recent_data["time"].iloc[-1]),
            "prediction_for": "next_candle",
            "direction": direction,
            "bullish_probability": probability,
            "bearish_probability": 1.0 - probability,
            "confidence": confidence,
            "confident_signal": (direction if confidence >= self.confidence_threshold else "abstain"),
        }

    def plot_prediction_comparison(self, max_points: int = 500) -> pathlib.Path:
        """Save actual and direction-implied next-close prices for the test set.

        The LSTM is a direction classifier, not a price regression model.  Its
        implied price therefore uses the predicted direction together with the
        median absolute one-candle return measured on training data only.
        """
        if self.y_prediction is None:
            raise RuntimeError("Run test before plotting predictions")
        if max_points < 1:
            raise ValueError("max_points must be positive")
        if self.typical_abs_return is None or not np.isfinite(self.typical_abs_return):
            raise RuntimeError("Training price movement is unavailable")

        predicted_direction = np.where(self.y_prediction == 1, 1.0, -1.0)
        self.test_predicted_next_close = self.test_current_close * (1.0 + predicted_direction * self.typical_abs_return)

        feature_tag = "-".join(self.feature_names)
        plot_path = self.results_path.with_name(f"{self.results_path.stem}_selected_lb{self.lookback}_{self._target_tag()}_features-{feature_tag}_prediction_comparison.png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)

        # A shorter recent segment remains readable when the test set contains
        # thousands of candles.  The complete test set is still used in every
        # reported evaluation metric.
        first = max(0, len(self.test_times) - max_points)
        plot_times = self.test_times[first:]
        actual_prices = self.test_actual_next_close[first:]
        predicted_prices = self.test_predicted_next_close[first:]

        figure, axis = plt.subplots(figsize=(14, 7))
        axis.plot(plot_times, actual_prices, label="Actual next-candle close", linewidth=1.8, )
        axis.plot(plot_times, predicted_prices, label="Direction-implied next-candle close", linewidth=1.3, alpha=0.85,)
        axis.set_title(f"{self.symbol} {self.timeframe}: actual vs predicted direction-implied price")
        axis.set_xlabel("Feature window end time")
        axis.set_ylabel("Price")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.autofmt_xdate()
        figure.tight_layout()
        figure.savefig(plot_path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return plot_path

    def plot_model_complexity_curve(self) -> pathlib.Path | None:
        """Plot training and validation error against the selected model capacity.

        Test error is intentionally excluded: the curve is for choosing an LSTM
        size, and that choice must be made using training/validation data only.
        """
        matching_trials = [trial for trial in self.lookback_trials  if (trial["target_setting"] == self._current_target_setting() and trial["lookback"] == self.lookback)]
        if not matching_trials:
            return None

        trials_by_units = {trial["lstm_units"]: trial for trial in matching_trials}
        units = sorted(trials_by_units)
        training_errors = [trials_by_units[unit]["training_error"] for unit in units]
        validation_errors = [trials_by_units[unit]["validation_error"] for unit in units]

        plot_path = self.results_path.with_name(f"{self.results_path.stem}_selected_lb{self.lookback}_{self._target_tag()}_model_complexity_curve.png")
        figure, axis = plt.subplots(figsize=(10, 6))
        axis.plot(units, training_errors, marker="o", color="#2e7d32", label="Training error")
        axis.plot(units, validation_errors, marker="o", color="#2468d4", label="Validation error")
        axis.axvline(self.lstm_units, color="#d62728", linestyle="--", label=f"Selected capacity: {self.lstm_units} units",)
        axis.set_title(f"{self.symbol} {self.timeframe}: LSTM capacity selection")
        axis.set_xlabel("LSTM units (model complexity)")
        axis.set_ylabel("Classification error")
        axis.set_xticks(units)
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(plot_path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return plot_path

    def _trading_result(self):
        """Trading part of the JSON report (triple barrier only)."""
        if self.target != "triple_barrier":
            return {"trading": None}
        return {
            "sl_atr": self.sl_atr,
            "alpha": self.alpha,
            "horizon": self.horizon,
            "cost_pct": self.cost_pct,
            "sl_atr_candidates": list(self.sl_atr_candidates),
            "alpha_candidates": list(self.alpha_candidates),
            "min_trades": self.min_trades,
            "trading": {
                "rule": "long when q > p0_train + (c/SL) / (1 + alpha); exit at TP, SL or after horizon candles",
                "note": ("trades can overlap in time, so the 95% range is optimistic; "
                         "cost_pct is an assumed round-trip cost, not a broker quote"),
                "validation": self.validation_trading,
                "test": self._trading_stats(self.y_probability, self.y_test, self.test_r, self.test_cost_r),
            },
        }

    def evaluate(self):
        if self.y_prediction is None:
            raise RuntimeError("Run test before evaluate")
        prediction_plot_path = self.plot_prediction_comparison()
        complexity_plot_path = self.plot_model_complexity_curve()
        confidence = np.where(self.y_prediction == 1, self.y_probability, 1.0 - self.y_probability,)
        confident = confidence >= self.confidence_threshold
        result = {
            "symbol": self.symbol,
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": self.MODEL_NAME,
            "timeframe": self.timeframe,
            "target_type": self.target,
            "target": self._target_description(),
            "target_threshold_atr_fraction": self.target_threshold,
            "target_threshold_mode": self.target_threshold_mode,
            "target_threshold_candidates": list(self.target_threshold_candidates),
            "multi_timeframes": list(self.multi_timeframes),
            "split_method": self.split_method,
            "lookback": self.lookback,
            "lookback_mode": self.lookback_mode,
            "lstm_units": self.lstm_units,
            "lstm_units_mode": self.lstm_units_mode,
            "hyperparameter_selection_metric": (
                "maximum coverage-adjusted validation balanced accuracy; validation log loss breaks ties"
                if self.target == "direction" else
                f"maximum validation expectancy_net_r among settings with at least {self.min_trades} validation trades; validation log loss breaks ties"
            ),
            "lookback_candidates": list(self.lookback_candidates),
            "lookback_candidates_evaluated": sorted({trial["lookback"] for trial in self.lookback_trials}),
            "target_threshold_candidates_evaluated": sorted(
                {trial["target_threshold_atr_fraction"] for trial in self.lookback_trials}
            ),
            "lstm_unit_candidates": list(self.lstm_unit_candidates),
            "lstm_unit_candidates_evaluated": sorted(
                {trial["lstm_units"] for trial in self.lookback_trials}
            ),
            "hyperparameter_trials": self.lookback_trials,
            "features": self.feature_names,
            "n_features": len(self.feature_names),
            "train_period": [str(self.train_times[0]), str(self.train_times[-1])],
            "validation_period": [str(self.validation_times[0]), str(self.validation_times[-1]),],
            "test_period": [str(self.test_times[0]), str(self.test_times[-1])],
            "n_train": len(self.X_train),
            "n_validation": len(self.X_validation),
            "n_test": len(self.X_test),
            "train_target_coverage": self.train_target_coverage,
            "validation_target_coverage": self.validation_target_coverage,
            "test_target_coverage": self.test_target_coverage,
            "epochs_requested": self.epochs,
            "epochs_trained": len(self.history.history["loss"]),
            "batch_size": self.batch_size,
            "train_time": self.train_time,
            "training_acc": self.training_accuracy,
            "training_error": 1.0 - self.training_accuracy,
            "validation_acc": self.validation_accuracy,
            "validation_error": 1.0 - self.validation_accuracy,
            "test_acc": self.test_accuracy,
            "test_error": 1.0 - self.test_accuracy,
            "majority_baseline_acc": float(max(np.mean(self.y_test == 0), np.mean(self.y_test == 1))),
            "test_balanced_acc": float(balanced_accuracy_score(self.y_test, self.y_prediction)),
            "test_macro_f1": float(f1_score(self.y_test, self.y_prediction, average="macro")),
            "test_log_loss": float(log_loss(self.y_test, self.y_probability, labels=[0, 1])),
            "prediction_plot": str(prediction_plot_path),
            "prediction_plot_points": min(500, len(self.test_times)),
            "model_complexity_curve": (
                str(complexity_plot_path) if complexity_plot_path is not None else None
            ),
            "model_complexity_curve_metric": "training and validation classification error; no test error",
            "predicted_price_method": (
                "current close adjusted by the predicted direction and the "
                "median absolute one-candle return learned from training data"
            ),
            "training_median_absolute_return": self.typical_abs_return,
            "labels": ["bearish", "bullish"],
            "confusion_matrix": confusion_matrix(self.y_test, self.y_prediction, labels=[0, 1]).tolist(),
            "classification_report": classification_report(self.y_test, self.y_prediction, labels=[0, 1], target_names=["bearish", "bullish"], output_dict=True, zero_division=0,),
            "decision_threshold": self.decision_threshold,
            "confidence_threshold_mode": self.confidence_threshold_mode,
            "confidence_threshold": self.confidence_threshold,
            "confidence_threshold_candidates": list(self.confidence_threshold_candidates),
            "min_confident_coverage": self.min_confident_coverage,
            "n_confident": int(confident.sum()),
            "confident_coverage": float(confident.mean()),
            "confident_accuracy": (float(accuracy_score(self.y_test[confident], self.y_prediction[confident])) if confident.any() else None),
        }
        result.update(self._trading_result())

        results = []
        if self.results_path.exists():
            with self.results_path.open() as file:
                results = json.load(file)
            if not isinstance(results, list):
                raise ValueError(f"Expected a list of runs in {self.results_path}")
        results.append(result)
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with self.results_path.open("w") as file:
            json.dump(results, file, indent=2)
        print(f"result appended to {self.results_path} ({len(results)} runs in total) for symbol {self.symbol}")
        return result
