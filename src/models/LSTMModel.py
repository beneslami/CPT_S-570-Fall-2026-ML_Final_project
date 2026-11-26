"""LSTM classifier for fixed-horizon directional prediction."""

import json
import hashlib
import importlib.util
import os
import pathlib
import time
from datetime import datetime
from typing import Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if "KERAS_BACKEND" not in os.environ:
    for backend_name in ("tensorflow", "jax", "torch"):
        if importlib.util.find_spec(backend_name) is not None:
            os.environ["KERAS_BACKEND"] = backend_name
            break

try:
    import keras
    from keras import Input, Sequential
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from keras.layers import Dense, Dropout, GRU as KerasGRU, LayerNormalization, LSTM as KerasLSTM
    from keras.losses import BinaryCrossentropy
    from keras.optimizers import Adam, AdamW
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "Keras 3 needs a backend framework. Install TensorFlow, JAX, or PyTorch "
        "before importing models.LSTM."
    ) from error
from sklearn.linear_model import LogisticRegression
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
    DEFAULT_LOOKBACK_CANDIDATES = (12, 24, 48, 72)
    DEFAULT_TARGET_THRESHOLD_CANDIDATES = (0.0,)
    DEFAULT_CONFIDENCE_THRESHOLD_CANDIDATES = (0.50, 0.55, 0.60, 0.65, 0.70)
    DEFAULT_LSTM_UNIT_CANDIDATES = (16, 32, 64)
    DEFAULT_SL_ATR_CANDIDATES = (1.0, 1.5, 2.0)
    DEFAULT_ALPHA_CANDIDATES = (1.0, 1.5, 2.0)
    AUXILIARY_HTF_FEATURES = ("close_pos", "relative_volume", "cci_scaled")
    MODEL_NAME = "LSTM"

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        results_path=None,
        lookback: int | str = "auto",
        target_threshold: float | str = 0.0,
        lstm_units: int | str = "auto",
        epochs: int = 50,
        batch_size: int = 64,
        confidence_threshold: float | str = "auto",
        model_path: Optional[str] = None,
        random_seed: int = 42,
        lookback_candidates=None,
        target_threshold_candidates=None,
        lstm_unit_candidates=None,
        confidence_threshold_candidates=None,
        min_confident_coverage: float = 0.10,
        multi_timeframes=("15m", "4h"),
        target: str = "direction",
        sl_atr_candidates=None,
        alpha_candidates=None,
        horizon: int = 1,
        cost_pct: float = 0.001,
        min_trades: int = 50,
        rnn_type: str = "lstm",
        use_sample_weights: bool = False,
    ):
        if target not in {"direction", "triple_barrier"}:
            raise ValueError("target must be 'direction' or 'triple_barrier'")
        if rnn_type not in {"lstm", "gru"}:
            raise ValueError("rnn_type must be 'lstm' or 'gru'")
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
        if confidence_threshold != "auto" and (
            not isinstance(confidence_threshold, (int, float)) or not 0.5 <= confidence_threshold <= 1.0
        ):
            raise ValueError("confidence_threshold must be 'auto' or between 0.5 and 1.0")
        if not 0 < min_confident_coverage <= 1:
            raise ValueError("min_confident_coverage must be in (0, 1]")

        self.symbol = symbol
        self.timeframe = timeframe
        self.rnn_type = rnn_type
        self.use_sample_weights = bool(use_sample_weights)
        self.lookback_mode = lookback
        self.lookback_candidates = tuple(lookback_candidates or self.DEFAULT_LOOKBACK_CANDIDATES)
        self.lookback = None if lookback == "auto" else lookback
        self.target_threshold_mode = target_threshold
        self.target_threshold_candidates = tuple(
            target_threshold_candidates or self.DEFAULT_TARGET_THRESHOLD_CANDIDATES
        )
        self.target_threshold = None if target_threshold == "auto" else float(target_threshold)
        self.target = target
        self.sl_atr_candidates = tuple(float(value) for value in (sl_atr_candidates or self.DEFAULT_SL_ATR_CANDIDATES))
        self.alpha_candidates = tuple(float(value) for value in (alpha_candidates or self.DEFAULT_ALPHA_CANDIDATES))
        self.sl_atr = None
        self.alpha = None
        self.horizon = int(horizon)
        self.cost_pct = cost_pct
        self.min_trades = min_trades
        self.p0_train = None
        self.validation_r = self.validation_cost_r = None
        self.test_r = self.test_cost_r = None
        self.validation_trading = None
        self.lstm_units_mode = lstm_units
        self.lstm_unit_candidates = tuple(lstm_unit_candidates or self.DEFAULT_LSTM_UNIT_CANDIDATES)
        self.lstm_units = None if lstm_units == "auto" else lstm_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.confidence_threshold_mode = confidence_threshold
        self.confidence_threshold_candidates = tuple(
            confidence_threshold_candidates or self.DEFAULT_CONFIDENCE_THRESHOLD_CANDIDATES
        )
        self.confidence_threshold = None if confidence_threshold == "auto" else float(confidence_threshold)
        self.min_confident_coverage = min_confident_coverage
        self.decision_threshold = 0.5
        self.model_path = pathlib.Path(model_path) if model_path else None
        self.random_seed = random_seed

        # Lean, high-signal 100% stationary core feature set (pruned via permutation importance)
        self.feature_names = [
            "close_pos",
            "htf4_close_pos",
            "htf24_close_pos",
            "relative_volume",
            "volume_delta_ratio",
            "rsi_centered",
            "macd_hist_atr",
            "return_3",
            "engulfing",
            "fvg_imbalance",
            "fvg_type",
            "hour_sin",
            "hour_cos",
        ]
        self.primary_feature_names = list(self.feature_names)
        self._derivative_feature_names = []
        self.multi_timeframes = tuple(multi_timeframes)

        project_dir = pathlib.Path(__file__).resolve().parents[2]
        self.results_path = (
            pathlib.Path(results_path)
            if results_path
            else project_dir / "results" / f"lstm_{symbol}_{timeframe}.json"
        )

        self.scaler = StandardScaler()
        self.calibrator = None
        self.model = None
        self.history = None
        self.split_method = ""
        self.train_times = self.validation_times = self.test_times = None
        self.X_train = self.X_validation = self.X_test = None
        self.y_train = self.y_validation = self.y_test = None
        self.w_train = None
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
        required = {
            "time", "open", "high", "low", "close", "volume",
            "volume_delta", "rsi_14", "cci", "atr_14", "macd_histogram",
        }
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

        features["body_pct"] = (close - open_price) / open_price.replace(0, np.nan)
        features["range_pct"] = (high - low) / open_price.replace(0, np.nan)
        candle_range = (high - low).replace(0, np.nan)
        upper_wick = high - np.maximum(open_price, close)
        lower_wick = np.minimum(open_price, close) - low
        features["wick_imbalance"] = ((lower_wick - upper_wick) / candle_range).fillna(0.0)
        features["close_pos"] = ((close - low) / candle_range).fillna(0.5)
        features["body_to_range"] = ((close - open_price) / candle_range).fillna(0.0).clip(-1.0, 1.0)
        atr_safe = atr.replace(0, np.nan)
        features["gap_atr"] = ((open_price - close.shift(1)) / atr_safe).clip(-5.0, 5.0)
        features["body_atr"] = ((close - open_price) / atr_safe).clip(-5.0, 5.0)
        features["range_atr"] = ((high - low) / atr_safe).clip(0.0, 10.0)
        features["atr_pct"] = atr / close.replace(0, np.nan)

        htf4_o = open_price.shift(3)
        htf4_h = high.rolling(4).max()
        htf4_l = low.rolling(4).min()
        htf4_rng = (htf4_h - htf4_l).replace(0, np.nan)
        features["htf4_close_pos"] = ((close - htf4_l) / htf4_rng).fillna(0.5).clip(0.0, 1.0)
        features["htf4_body_to_range"] = ((close - htf4_o) / htf4_rng).fillna(0.0).clip(-1.0, 1.0)

        for k in (12, 24):
            htf_h = high.rolling(k).max()
            htf_l = low.rolling(k).min()
            htf_rng = (htf_h - htf_l).replace(0, np.nan)
            features[f"htf{k}_close_pos"] = ((close - htf_l) / htf_rng).fillna(0.5).clip(0.0, 1.0)

        if "relative_volume" in data.columns:
            features["relative_volume"] = pd.to_numeric(data["relative_volume"], errors="coerce")
        else:
            volume_median_20 = volume.rolling(window=20, min_periods=20).median()
            features["relative_volume"] = volume / volume_median_20.replace(0, np.nan)
        features["relative_volume"] = features["relative_volume"].clip(lower=0, upper=10)
        features["volume_delta_ratio"] = (volume_delta / volume.replace(0, np.nan)).fillna(0.0).clip(-1.0, 1.0)
        features["macd_hist_atr"] = (macd_histogram / atr.replace(0, np.nan)).clip(-5.0, 5.0)
        features["cci_scaled"] = pd.to_numeric(data["cci"], errors="coerce").clip(-300, 300) / 100.0

        bull_impulse = close.shift(1) > open_price.shift(1)
        bear_impulse = close.shift(1) < open_price.shift(1)
        bull_gap = np.where(bull_impulse, np.maximum(0.0, low - high.shift(2)), 0.0)
        bear_gap = np.where(bear_impulse, np.maximum(0.0, low.shift(2) - high), 0.0)
        denominator = close + 1e-10
        features["fvg_bull_pct"] = bull_gap / denominator
        features["fvg_bear_pct"] = bear_gap / denominator
        features["fvg_imbalance"] = (bull_gap - bear_gap) / denominator
        features["fvg_type"] = np.where(bull_gap > 0, 1, np.where(bear_gap > 0, -1, 0))

        prev_open = open_price.shift(1)
        prev_close = close.shift(1)
        bull_engulfing = (prev_close < prev_open) & (close > open_price) & (open_price <= prev_close) & (close >= prev_open)
        bear_engulfing = (prev_close > prev_open) & (close < open_price) & (open_price >= prev_close) & (close <= prev_open)
        features["engulfing"] = np.where(bull_engulfing, 1.0, np.where(bear_engulfing, -1.0, 0.0))
        features["upper_wick_pct"] = upper_wick / open_price.replace(0, np.nan)
        features["lower_wick_pct"] = lower_wick / open_price.replace(0, np.nan)

        features["return_1"] = np.log(close / close.shift(1))
        features["return_3"] = np.log(close / close.shift(3))
        features["return_6"] = np.log(close / close.shift(6))
        features["return_12"] = np.log(close / close.shift(12))

        features["macd_hist_pct"] = macd_histogram / close.replace(0, np.nan)
        features["rsi_centered"] = (pd.to_numeric(data["rsi_14"], errors="coerce") - 50.0) / 50.0

        times = pd.to_datetime(data["time"], utc=True)
        hour = times.dt.hour + times.dt.minute / 60.0
        features["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        features["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        dt_hours = times.diff().dt.total_seconds().div(3600.0).fillna(1.0)
        features["log_dt_hours"] = np.log1p(dt_hours.clip(lower=0.0, upper=168.0))

        # Strictly causal Binance derivatives sequence features (when present in datapoints_new/)
        eps = 1e-10
        if "taker_buy_ratio" in data.columns:
            tbr = pd.to_numeric(data["taker_buy_ratio"], errors="coerce") - 0.5
            features["perp_tbr"] = tbr
            if "spot_taker_buy_ratio" in data.columns:
                stbr = pd.to_numeric(data["spot_taker_buy_ratio"], errors="coerce").fillna(0.5) - 0.5
                features["spot_tbr"] = stbr
                features["perp_minus_spot_tbr"] = tbr - stbr
        if "premium_close" in data.columns:
            prem = pd.to_numeric(data["premium_close"], errors="coerce").ffill().fillna(0.0)
            features["prem_z24"] = ((prem - prem.rolling(24).mean()) / (prem.rolling(24).std() + eps)).clip(-5.0, 5.0)
        if "funding_rate" in data.columns:
            fr = pd.to_numeric(data["funding_rate"], errors="coerce").fillna(0.0)
            features["funding_z72"] = ((fr - fr.rolling(72).mean()) / (fr.rolling(72).std() + eps)).clip(-5.0, 5.0)

        features = features.replace([np.inf, -np.inf], np.nan)
        return features

    @staticmethod
    def _detect_derivative_features(data: pd.DataFrame) -> list[str]:
        """Select non-redundant derivatives features (volume_delta_ratio already captures perp_tbr)."""
        cols = []
        if "spot_taker_buy_ratio" in data.columns:
            cols.append("spot_tbr")
        if "premium_close" in data.columns:
            cols.append("prem_z24")
        if "funding_rate" in data.columns:
            cols.append("funding_z72")
        return cols

    def _numeric_features(self, data):
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
        aligned = primary_data.copy()
        aligned["time"] = pd.to_datetime(aligned["time"], utc=True)
        primary_duration = pd.Timedelta(self.timeframe)
        aligned["primary_close_time"] = aligned["time"] + primary_duration
        aligned = aligned.sort_values("time").reset_index(drop=True)
        self.feature_names = list(self.primary_feature_names) + list(self._derivative_feature_names)
        eps = 1e-10
        for timeframe in self.multi_timeframes:
            if timeframe not in auxiliary_data:
                raise ValueError(f"Missing auxiliary data for timeframe {timeframe}")
            auxiliary = auxiliary_data[timeframe].copy()
            aux_duration = pd.Timedelta(timeframe)
            prefix = f"{timeframe}_"

            if aux_duration < primary_duration and (primary_duration % aux_duration == pd.Timedelta(0)):
                # Strictly causal intra-bar features: all sub-bars inside [t, t + primary_duration) close <= t + primary_duration.
                auxiliary["time"] = pd.to_datetime(auxiliary["time"], utc=True)
                auxiliary = auxiliary.sort_values("time")
                o = pd.to_numeric(auxiliary["open"], errors="coerce")
                h = pd.to_numeric(auxiliary["high"], errors="coerce")
                l = pd.to_numeric(auxiliary["low"], errors="coerce")
                c = pd.to_numeric(auxiliary["close"], errors="coerce")
                v = pd.to_numeric(auxiliary["volume"], errors="coerce")
                bar_rng = (h - l).replace(0.0, np.nan)
                g = pd.DataFrame(
                    {
                        "_bucket": auxiliary["time"].dt.floor(self.timeframe),
                        "_ret": np.log(c / (c.shift(1) + eps)),
                        "_body": (c - o) / (o + eps),
                        "_cp": ((c - l) / bar_rng).fillna(0.5).clip(0.0, 1.0),
                        "_clv": ((2.0 * c - h - l) / bar_rng).fillna(0.0).clip(-1.0, 1.0),
                        "_vol": v,
                    }
                )
                grouped = g.groupby("_bucket", sort=True)
                last_sub = grouped.last()
                first_sub = grouped.first()
                vol_sum = grouped["_vol"].sum()
                sub_df = pd.DataFrame(index=last_sub.index)
                sub_df[f"{prefix}last_ret"] = last_sub["_ret"]
                sub_df[f"{prefix}last_body"] = last_sub["_body"]
                sub_df[f"{prefix}last_cp"] = last_sub["_cp"]
                sub_df[f"{prefix}last_clv"] = last_sub["_clv"]
                sub_df[f"{prefix}late_minus_early"] = last_sub["_body"] - first_sub["_body"]
                sub_df[f"{prefix}up_frac"] = (g["_body"] > 0.0).groupby(g["_bucket"]).mean()
                sub_df[f"{prefix}last_vol_share"] = last_sub["_vol"] / (vol_sum + eps)
                sub_df[f"{prefix}last2_ret"] = g.groupby("_bucket").tail(2).groupby("_bucket")["_ret"].sum()
                sub_df[f"{prefix}rvol"] = grouped["_ret"].std().fillna(0.0)
                sub_df = sub_df.reindex(aligned["time"]).reset_index(drop=True)
                for col in sub_df.columns:
                    aligned[col] = sub_df[col].to_numpy(dtype=float)
                    self.feature_names.append(col)
                continue

            self._validate_auxiliary_frame(auxiliary, f"auxiliary_data[{timeframe}]")
            auxiliary["time"] = pd.to_datetime(auxiliary["time"], utc=True)
            auxiliary_close_time = auxiliary["time"] + aux_duration
            auxiliary_features = self._base_numeric_features(auxiliary)
            htf_cols = [c for c in self.AUXILIARY_HTF_FEATURES if c in auxiliary_features.columns]
            if not htf_cols:
                raise ValueError(f"Auxiliary timeframe {timeframe} cannot compute any HTF features")
            right = pd.DataFrame({"auxiliary_close_time": auxiliary_close_time})
            for feature_name in htf_cols:
                column_name = f"{prefix}{feature_name}"
                right[column_name] = auxiliary_features[feature_name].to_numpy()
                self.feature_names.append(column_name)
            aligned = pd.merge_asof(
                aligned,
                right.sort_values("auxiliary_close_time"),
                left_on="primary_close_time",
                right_on="auxiliary_close_time",
                direction="backward",
                allow_exact_matches=True,
            ).drop(columns="auxiliary_close_time")
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

    def _target_candidates(self):
        if self.target == "direction":
            thresholds = (
                self.target_threshold_candidates
                if self.target_threshold_mode == "auto"
                else (self.target_threshold,)
            )
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
        if self.target == "direction":
            return f"h{self.horizon}_atr{self.target_threshold * 100:g}pct".replace(".", "p")
        return f"sl{self.sl_atr:g}_a{self.alpha:g}_h{self.horizon}".replace(".", "p")

    def _target_description(self):
        if self.target == "direction":
            return (
                f"close[t + {self.horizon}] - close[t] > {self.target_threshold:.0%} of ATR: bullish; "
                f"< -{self.target_threshold:.0%} of ATR: bearish"
            )
        return (
            f"long trade opened at close with SL = {self.sl_atr:g} ATR and TP = {self.alpha:g} x SL; "
            f"1 if TP hit first within {self.horizon} candles"
        )

    def _triple_barrier(self, data: pd.DataFrame):
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
            win[t], r[t] = 0, (close[t + self.horizon] - entry) / sl_distance
            for j in range(t + 1, t + self.horizon + 1):
                if low[j] <= stop:
                    r[t] = -1.0
                    break
                if high[j] >= target:
                    win[t], r[t] = 1, self.alpha
                    break
            cost_r[t] = self.cost_pct * entry / sl_distance
        return win, r, cost_r

    def _make_sequences(self, data: pd.DataFrame, fit_scaler: bool):
        features = self._numeric_features(data)
        outcome_r = outcome_cost_r = None
        weights = np.ones(len(data), dtype=np.float32)

        if self.target == "direction":
            close = pd.to_numeric(data["close"], errors="coerce")
            move = close.shift(-self.horizon) - close
            current_atr = pd.to_numeric(data["atr_14"], errors="coerce")
            minimum_move = self.target_threshold * current_atr
            labels = pd.Series(np.nan, index=data.index)
            if self.target_threshold > 0:
                labels.loc[move > minimum_move] = 1
                labels.loc[move < -minimum_move] = 0
            else:
                labels.loc[move > 0] = 1
                labels.loc[move < 0] = 0
            norm_move = move.abs() / (current_atr * np.sqrt(max(1, self.horizon)) + 1e-10)
            weights = norm_move.clip(0.25, 3.0).fillna(1.0).to_numpy(dtype=np.float32)
        else:
            win, outcome_r, outcome_cost_r = self._triple_barrier(data)
            labels = pd.Series(win, index=data.index)

        if fit_scaler:
            scale_rows = features.notna().all(axis=1)
            if scale_rows.sum() < self.lookback:
                raise ValueError("Not enough valid training rows to fit the scaler")
            self.scaler.fit(features.loc[scale_rows])
        scaled = np.clip(self.scaler.transform(features), -5.0, 5.0)
        windows = np.lib.stride_tricks.sliding_window_view(scaled, self.lookback, axis=0)
        windows = np.moveaxis(windows, -1, 1)
        ends = np.arange(self.lookback - 1, len(data))
        y = labels.to_numpy(dtype=float)[ends]
        w = weights[ends]
        valid = np.isfinite(windows).all(axis=(1, 2)) & np.isfinite(y)
        if not valid.any():
            raise ValueError("No complete LSTM sequences could be created")

        times = pd.DatetimeIndex(pd.to_datetime(data["time"], utc=True), name="feature_end_time")
        target_coverage = float(valid.sum() / max(1, len(ends)))
        outcomes = None
        if outcome_r is not None:
            outcomes = (outcome_r[ends[valid]].astype(float), outcome_cost_r[ends[valid]].astype(float))
        elif self.target == "direction" and self.use_sample_weights:
            outcomes = w[valid]
        return (
            windows[valid].astype(np.float32),
            y[valid].astype(np.float32),
            times[ends[valid]],
            target_coverage,
            outcomes,
        )

    @staticmethod
    def _aligned_closes(data: pd.DataFrame, sequence_times):
        prices = pd.DataFrame(
            {
                "time": pd.to_datetime(data["time"], utc=True),
                "close": pd.to_numeric(data["close"], errors="coerce"),
            }
        ).set_index("time")
        prices["next_close"] = prices["close"].shift(-1)
        aligned = prices.reindex(sequence_times)
        return (aligned["close"].to_numpy(dtype=float), aligned["next_close"].to_numpy(dtype=float))

    def _build_model(self):
        keras.utils.set_random_seed(self.random_seed)
        hidden = max(8, self.lstm_units // 2)
        rnn_layer = (
            KerasGRU(self.lstm_units, dropout=0.30)
            if self.rnn_type == "gru"
            else KerasLSTM(self.lstm_units, dropout=0.30)
        )
        model = Sequential(
            [
                Input(shape=(self.lookback, len(self.feature_names))),
                rnn_layer,
                LayerNormalization(),
                Dense(hidden, activation="gelu"),
                Dropout(0.30),
                Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer=AdamW(learning_rate=3e-4, weight_decay=1e-3, clipnorm=1.0),
            loss=BinaryCrossentropy(label_smoothing=0.02),
            metrics=["accuracy"],
        )
        return model

    def _prepare_sequences(self):
        self.scaler = StandardScaler()
        (self.X_train, self.y_train, self.train_times, self.train_target_coverage, train_out) = (
            self._make_sequences(self._train_frame, fit_scaler=True)
        )
        (self.X_validation, self.y_validation, self.validation_times, self.validation_target_coverage, val_out) = (
            self._make_sequences(self._validation_frame, fit_scaler=False)
        )
        (self.X_test, self.y_test, self.test_times, self.test_target_coverage, test_out) = (
            self._make_sequences(self._test_frame, fit_scaler=False)
        )

        if self.target == "triple_barrier":
            self.p0_train = float(np.mean(self.y_train))
            self.validation_r, self.validation_cost_r = val_out
            self.test_r, self.test_cost_r = test_out
            self.w_train = None
        else:
            self.w_train = train_out if isinstance(train_out, np.ndarray) else None

        train_current_close, train_actual_next_close = self._aligned_closes(self._train_frame, self.train_times)
        train_abs_return = np.abs(train_actual_next_close / train_current_close - 1.0)
        self.typical_abs_return = float(np.nanmedian(train_abs_return))
        self.test_current_close, self.test_actual_next_close = self._aligned_closes(self._test_frame, self.test_times)

        if min(len(self.X_train), len(self.X_validation), len(self.X_test)) < 20:
            raise ValueError("Need at least 20 complete sequences in each split")
        if len(np.unique(self.y_train)) != 2 or len(np.unique(self.y_validation)) != 2:
            raise ValueError("Training and validation sequences must each contain bullish and bearish labels")
        self.model = self._build_model()

    def feature_engineering(self, train_data, test_data, auxiliary_data=None):
        self._validate_frame(train_data, "train_data")
        self._validate_frame(test_data, "test_data")
        same_data = train_data.equals(test_data)
        auxiliary_data = auxiliary_data or {}
        self._derivative_feature_names = self._detect_derivative_features(train_data)
        if self.multi_timeframes:
            train_data = self._align_auxiliary_features(train_data, auxiliary_data)
            test_data = self._align_auxiliary_features(test_data, auxiliary_data)
        else:
            self.feature_names = list(self.primary_feature_names) + list(self._derivative_feature_names)

        if same_data:
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
        purge = max(1, self.horizon)
        train_frame = train_and_validation.iloc[: max(1, validation_start - purge)].copy()
        validation_frame = train_and_validation.iloc[validation_start:].copy()

        self._train_frame = train_frame
        self._validation_frame = validation_frame
        self._test_frame = test_frame
        if (
            self.target == "direction"
            and self.lookback is not None
            and self.target_threshold is not None
            and self.lstm_units is not None
        ):
            self._prepare_sequences()

    def _trading_stats(self, q, y, r, cost_r):
        strategy = Strategy(alpha=self.alpha, sl_atr=self.sl_atr, cost_pct=self.cost_pct)
        q = np.asarray(q, dtype=float)
        y = np.asarray(y, dtype=float)
        threshold = self.p0_train + cost_r / (1.0 + self.alpha)
        take = q > threshold
        n_trades = int(take.sum())
        p0_split = float(np.mean(y))
        stats = {
            "n_candles": int(len(q)),
            "n_trades": n_trades,
            "trade_share": n_trades / len(q),
            "p0_train": self.p0_train,
            "p0": p0_split,
            "p0_theory": strategy.break_even_win_rate(),
            "mean_threshold": float(np.mean(threshold)),
            "mean_cost_r": float(np.mean(cost_r)),
            "no_model_expectancy_net_r": float(np.mean(r - cost_r)),
            "p": None, "edge": None, "timeout_share": None, "mean_gross_r": None,
            "expectancy_net_r": None, "expectancy_net_r_95pct": None, "expectancy_net_formula_r": None,
        }
        if n_trades > 0:
            p = float(np.mean(y[take]))
            net_r = r[take] - cost_r[take]
            half_width = 1.96 * float(np.std(net_r)) / np.sqrt(n_trades)
            stats.update(
                {
                    "p": p,
                    "edge": p - p0_split,
                    "timeout_share": float(np.mean((r[take] != self.alpha) & (r[take] != -1.0))),
                    "mean_gross_r": float(np.mean(r[take])),
                    "mean_trade_cost_r": float(np.mean(cost_r[take])),
                    "expectancy_net_r": float(np.mean(net_r)),
                    "expectancy_net_r_95pct": [float(np.mean(net_r)) - half_width, float(np.mean(net_r)) + half_width],
                    "expectancy_net_formula_r": strategy.expectancy(p, p0_split, float(np.mean(cost_r[take]))),
                }
            )
        bins = [0.0, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 1.0]
        calibration = []
        for low, high in zip(bins[:-1], bins[1:]):
            in_bin = (q >= low) & (q < high)
            if in_bin.any():
                calibration.append(
                    {
                        "q_range": [low, high],
                        "count": int(in_bin.sum()),
                        "mean_q": float(np.mean(q[in_bin])),
                        "actual_tp_rate": float(np.mean(y[in_bin])),
                    }
                )
        stats["calibration"] = calibration
        return stats

    def _selection_score(self, trial):
        if self.target == "direction":
            return (-trial["coverage_adjusted_validation_balanced_acc"], trial["validation_log_loss"])
        stats = trial["validation_trading"]
        if stats["n_trades"] < self.min_trades:
            return (1, -stats["n_trades"], trial["validation_log_loss"])
        return (0, -stats["expectancy_net_r"], trial["validation_log_loss"])

    @staticmethod
    def _to_logits(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        p = np.clip(np.asarray(probs, dtype=float), eps, 1.0 - eps)
        return np.log(p / (1.0 - p)).reshape(-1, 1)

    def _fit_platt_calibrator(self, raw_val_prob: np.ndarray, y_val: np.ndarray):
        self.calibrator = None
        if len(y_val) < 20 or len(np.unique(y_val)) < 2:
            return
        clf = LogisticRegression(C=1.0, solver="lbfgs")
        clf.fit(self._to_logits(raw_val_prob), y_val.astype(int))
        if float(clf.coef_[0, 0]) > 0.0:
            self.calibrator = clf

    def calibrate_proba(self, raw_probs: np.ndarray) -> np.ndarray:
        arr = np.clip(np.asarray(raw_probs, dtype=float), 0.0, 1.0)
        if self.calibrator is None:
            return arr
        out = arr.copy()
        valid = np.isfinite(out)
        if valid.any():
            out[valid] = self.calibrator.predict_proba(self._to_logits(out[valid]))[:, 1]
        return np.clip(out, 0.0, 1.0)

    def _train_current_lookback(self):
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=0),
        ]
        started = time.time()
        fit_kwargs = {
            "validation_data": (self.X_validation, self.y_validation),
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "shuffle": False,
            "callbacks": callbacks,
            "verbose": getattr(self, "training_verbose", 1),
        }
        if self.w_train is not None:
            fit_kwargs["sample_weight"] = self.w_train
        self.history = self.model.fit(self.X_train, self.y_train, **fit_kwargs)
        self.train_time = time.time() - started

        raw_train_prob = self.model.predict(self.X_train, batch_size=self.batch_size, verbose=0).reshape(-1)
        raw_val_prob = self.model.predict(
            self.X_validation, batch_size=self.batch_size, verbose=0
        ).reshape(-1)

        self._fit_platt_calibrator(raw_val_prob, self.y_validation)
        train_probability = self.calibrate_proba(raw_train_prob)
        validation_probability = self.calibrate_proba(raw_val_prob)

        thresholds = np.linspace(0.45, 0.55, 21)
        threshold_scores = [
            (float(t), float(balanced_accuracy_score(self.y_validation, validation_probability >= t)))
            for t in thresholds
        ]
        self.decision_threshold, validation_balanced_accuracy = max(
            threshold_scores, key=lambda item: (item[1], -abs(item[0] - 0.5))
        )

        self.training_accuracy = float(accuracy_score(self.y_train, train_probability >= self.decision_threshold))
        self.validation_accuracy = float(
            accuracy_score(self.y_validation, validation_probability >= self.decision_threshold)
        )
        validation_loss = float(log_loss(self.y_validation, validation_probability, labels=[0, 1]))
        validation_prediction = (validation_probability >= self.decision_threshold).astype(int)
        validation_confidence = np.where(
            validation_prediction == 1, validation_probability, 1.0 - validation_probability
        )
        candidates = (
            self.confidence_threshold_candidates
            if self.confidence_threshold_mode == "auto"
            else (self.confidence_threshold,)
        )
        best_confidence_score = None
        selected_coverage = 1.0
        selected_accuracy = self.validation_accuracy
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
            self.confidence_threshold = 0.50
        if self.target == "triple_barrier":
            self.validation_trading = self._trading_stats(
                validation_probability, self.y_validation, self.validation_r, self.validation_cost_r
            )
        return (validation_loss, validation_balanced_accuracy, selected_coverage, selected_accuracy)

    def train(self):
        if self._train_frame is None:
            raise RuntimeError("Run feature_engineering before train")

        lookback_candidates = self.lookback_candidates if self.lookback_mode == "auto" else (self.lookback,)
        lstm_unit_candidates = self.lstm_unit_candidates if self.lstm_units_mode == "auto" else (self.lstm_units,)
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
                    (
                        validation_loss,
                        validation_balanced_accuracy,
                        validation_confident_coverage,
                        validation_confident_accuracy,
                    ) = self._train_current_lookback()
                    coverage_adjusted_balanced_accuracy = (
                        validation_balanced_accuracy * self.validation_target_coverage
                    )
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
                    score = self._selection_score(trial)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_state = {
                            "model": self.model,
                            "scaler": self.scaler,
                            "calibrator": self.calibrator,
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
                            "w_train": self.w_train,
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
        raw_prob = self.model.predict(self.X_test, batch_size=self.batch_size, verbose=0).reshape(-1)
        self.y_probability = self.calibrate_proba(raw_prob)
        self.y_prediction = (self.y_probability >= self.decision_threshold).astype(int)
        self.test_accuracy = float(accuracy_score(self.y_test, self.y_prediction))
        return self.test_accuracy

    def predict_next(self, recent_data: pd.DataFrame, auxiliary_data=None) -> dict:
        if self.model is None or self.history is None:
            raise RuntimeError("Train the model before predicting")
        self._validate_frame(recent_data, "recent_data")
        if self.multi_timeframes:
            recent_data = self._align_auxiliary_features(recent_data, auxiliary_data or {})
        features = self._numeric_features(recent_data).tail(self.lookback)
        if len(features) != self.lookback or features.isna().any().any():
            raise ValueError(f"Need {self.lookback} complete recent candles for prediction")
        scaled = np.clip(self.scaler.transform(features), -5.0, 5.0).astype(np.float32)
        raw_p = float(self.model.predict(scaled[None, :, :], verbose=0)[0, 0])
        probability = float(self.calibrate_proba(np.array([raw_p]))[0])
        direction = "bullish" if probability >= self.decision_threshold else "bearish"
        confidence = probability if direction == "bullish" else 1.0 - probability
        return {
            "as_of_time": str(recent_data["time"].iloc[-1]),
            "prediction_for": f"close[t+{self.horizon}]",
            "direction": direction,
            "bullish_probability": probability,
            "bearish_probability": 1.0 - probability,
            "confidence": confidence,
            "confident_signal": direction if confidence >= self.confidence_threshold else "abstain",
        }

    def plot_prediction_comparison(self, max_points: int = 500) -> pathlib.Path:
        if self.y_prediction is None:
            raise RuntimeError("Run test before plotting predictions")
        predicted_direction = np.where(self.y_prediction == 1, 1.0, -1.0)
        self.test_predicted_next_close = self.test_current_close * (1.0 + predicted_direction * self.typical_abs_return)
        feature_digest = hashlib.sha256(json.dumps(self.feature_names, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
        filename = f"{self.results_path.stem}_selected_lb{self.lookback}_{self._target_tag()}_features-{feature_digest}_prediction_comparison.png"
        plot_path = self.results_path.with_name(filename)
        plot_path.parent.mkdir(parents=True, exist_ok=True)

        first = max(0, len(self.test_times) - max_points)
        figure, axis = plt.subplots(figsize=(14, 7))
        axis.plot(self.test_times[first:], self.test_actual_next_close[first:], label="Actual next-candle close", linewidth=1.8)
        axis.plot(self.test_times[first:], self.test_predicted_next_close[first:], label="Direction-implied close", linewidth=1.3, alpha=0.85)
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
        matching_trials = [
            trial
            for trial in self.lookback_trials
            if (trial["target_setting"] == self._current_target_setting() and trial["lookback"] == self.lookback)
        ]
        if not matching_trials:
            return None
        trials_by_units = {trial["lstm_units"]: trial for trial in matching_trials}
        units = sorted(trials_by_units)
        plot_path = self.results_path.with_name(
            f"{self.results_path.stem}_selected_lb{self.lookback}_{self._target_tag()}_model_complexity_curve.png"
        )
        figure, axis = plt.subplots(figsize=(10, 6))
        axis.plot(units, [trials_by_units[u]["training_error"] for u in units], marker="o", color="#2e7d32", label="Training error")
        axis.plot(units, [trials_by_units[u]["validation_error"] for u in units], marker="o", color="#2468d4", label="Validation error")
        axis.axvline(self.lstm_units, color="#d62728", linestyle="--", label=f"Selected: {self.lstm_units} units")
        axis.set_title(f"{self.symbol} {self.timeframe}: LSTM capacity selection")
        axis.set_xlabel("LSTM units")
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
        if self.target != "triple_barrier":
            return {"trading": None}
        return {
            "sl_atr": self.sl_atr,
            "alpha": self.alpha,
            "horizon": self.horizon,
            "cost_pct": self.cost_pct,
            "trading": {
                "validation": self.validation_trading,
                "test": self._trading_stats(self.y_probability, self.y_test, self.test_r, self.test_cost_r),
            },
        }

    def evaluate(self):
        if self.y_prediction is None:
            raise RuntimeError("Run test before evaluate")
        prediction_plot_path = self.plot_prediction_comparison()
        complexity_plot_path = self.plot_model_complexity_curve()
        confidence = np.where(self.y_prediction == 1, self.y_probability, 1.0 - self.y_probability)
        confident = confidence >= self.confidence_threshold
        result = {
            "symbol": self.symbol,
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": self.MODEL_NAME,
            "timeframe": self.timeframe,
            "target_type": self.target,
            "horizon": self.horizon,
            "target": self._target_description(),
            "split_method": self.split_method,
            "lookback": self.lookback,
            "lstm_units": self.lstm_units,
            "features": self.feature_names,
            "n_features": len(self.feature_names),
            "n_train": len(self.X_train),
            "n_validation": len(self.X_validation),
            "n_test": len(self.X_test),
            "epochs_trained": len(self.history.history["loss"]),
            "train_time": self.train_time,
            "training_acc": self.training_accuracy,
            "validation_acc": self.validation_accuracy,
            "test_acc": self.test_accuracy,
            "majority_baseline_acc": float(max(np.mean(self.y_test == 0), np.mean(self.y_test == 1))),
            "test_balanced_acc": float(balanced_accuracy_score(self.y_test, self.y_prediction)),
            "test_macro_f1": float(f1_score(self.y_test, self.y_prediction, average="macro")),
            "test_log_loss": float(log_loss(self.y_test, self.y_probability, labels=[0, 1])),
            "prediction_plot": str(prediction_plot_path),
            "model_complexity_curve": str(complexity_plot_path) if complexity_plot_path is not None else None,
            "decision_threshold": self.decision_threshold,
            "confidence_threshold": self.confidence_threshold,
            "n_confident": int(confident.sum()),
            "confident_coverage": float(confident.mean()),
            "confident_accuracy": (
                float(accuracy_score(self.y_test[confident], self.y_prediction[confident])) if confident.any() else None
            ),
            "confusion_matrix": confusion_matrix(self.y_test, self.y_prediction, labels=[0, 1]).tolist(),
            "classification_report": classification_report(
                self.y_test, self.y_prediction, labels=[0, 1], target_names=["bearish", "bullish"],
                output_dict=True, zero_division=0,
            ),
        }
        result.update(self._trading_result())

        results = []
        if self.results_path.exists():
            with self.results_path.open() as file:
                results = json.load(file)
        results.append(result)
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with self.results_path.open("w") as file:
            json.dump(results, file, indent=2)
        print(f"result appended to {self.results_path} ({len(results)} runs in total) for symbol {self.symbol}")
        return result
