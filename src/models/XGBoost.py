import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ML_interface import Model

PROJECT_DIR = Path(__file__).resolve().parents[2]
HORIZON = 3
DEFAULT_SL_ATR = 3.0
DEFAULT_TP_ATR = 7.0
DEFAULT_MAX_BARS = 24


class BracketProbabilityCalibrator:
    """Calibrate probabilities on a separate chronological calibration period."""

    def __init__(self, method: str = "logistic"):
        if method not in ("none", "logistic", "isotonic"):
            raise ValueError("method must be 'none', 'logistic' or 'isotonic'")
        self.method = method
        self._model = None

    @staticmethod
    def _to_logits(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        p = np.clip(np.asarray(probs, dtype=float), eps, 1.0 - eps)
        return np.log(p / (1.0 - p)).reshape(-1, 1)

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray):
        if self.method == "none":
            self._model = None
            return self
        mask = np.isfinite(raw_probs) & np.isfinite(y_true)
        p = np.asarray(raw_probs, dtype=float)[mask]
        y = np.asarray(y_true, dtype=int)[mask]
        if len(np.unique(y)) < 2 or len(y) < 10:
            self._model = None
            return self

        if self.method == "logistic":
            self._model = LogisticRegression(C=1.0, solver="lbfgs")
            self._model.fit(self._to_logits(p), y)
        else:
            self._model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self._model.fit(p, y)
        return self

    def predict_proba(self, raw_probs: np.ndarray) -> np.ndarray:
        arr = np.asarray(raw_probs, dtype=float)
        if self._model is None:
            return np.clip(arr, 0.0, 1.0)
        out = arr.copy()
        valid = np.isfinite(out)
        if not valid.any():
            return out
        if self.method == "logistic":
            out[valid] = self._model.predict_proba(self._to_logits(out[valid]))[:, 1]
        else:
            out[valid] = self._model.transform(out[valid])
        return np.clip(out, 0.0, 1.0)


def select_probability_calibrator(raw_fit, y_fit, raw_validation, y_validation, methods):
    """Select a non-decreasing calibration using a later two-class period."""
    raw_validation = np.asarray(raw_validation, dtype=float)
    y_validation = np.asarray(y_validation, dtype=int)
    if len(y_validation) == 0 or len(np.unique(y_validation)) < 2:
        reason = "calibration-selection period is empty or contains only one class"
        score = (float(log_loss(y_validation, raw_validation, labels=[0, 1]))
                 if len(y_validation) else None)
        return BracketProbabilityCalibrator('none'), [{
            'method': 'none', 'validation_log_loss': score, 'validation_roc_auc': None,
            'eligible': True, 'fallback_reason': reason,
        }]
    fitted = []
    for method in methods:
        calibrator = BracketProbabilityCalibrator(method).fit(raw_fit, y_fit)
        if method != 'none' and calibrator._model is None:
            fitted.append((calibrator, {'method': method, 'eligible': False,
                                        'rejection_reason': 'insufficient two-class calibration data'}))
            continue
        probability = calibrator.predict_proba(raw_validation)
        score = float(log_loss(y_validation, probability, labels=[0, 1]))
        auc = float(roc_auc_score(y_validation, probability))
        details = {'method': method, 'validation_log_loss': score, 'validation_roc_auc': auc,
                   'eligible': True}
        if method == 'logistic':
            details['logistic_slope'] = float(calibrator._model.coef_[0, 0])
            if details['logistic_slope'] <= 0:
                details['eligible'] = False
                details['rejection_reason'] = 'logistic calibration does not preserve ranking'
        fitted.append((calibrator, details))
    eligible = [item for item in fitted if item[1]['eligible']]
    if not eligible:
        fallback = BracketProbabilityCalibrator('none')
        details = {'method': 'none', 'eligible': True,
                   'validation_log_loss': float(log_loss(y_validation, raw_validation, labels=[0, 1])),
                   'validation_roc_auc': float(roc_auc_score(y_validation, raw_validation)),
                   'fallback_reason': 'no requested calibration method preserved ranking and had sufficient data'}
        fitted.append((fallback, details))
        eligible = [(fallback, details)]
    selected = min(eligible, key=lambda item: item[1]['validation_log_loss'])[0]
    return selected, [item[1] for item in fitted]


def load_data(path):
    """Load OHLCV data with unique, chronological UTC timestamps."""
    data = pd.read_csv(path, usecols=["time", "open", "high", "low", "close", "volume"])
    data["time"] = pd.to_datetime(data["time"], utc=True, errors="raise")
    if not data["time"].is_monotonic_increasing or data["time"].duplicated().any():
        raise ValueError(f"{path}: timestamps must be unique and chronological")
    data = data.set_index("time").rename(columns=str.capitalize)
    return data[["Close", "High", "Low", "Open", "Volume"]]


def _normalize_ohlcv(frame):
    """Ensure a DatetimeIndex (UTC), chronological order, and OHLCV columns."""
    df = frame.copy()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="raise")
        df = df.set_index("time")
    df = df.rename(columns=str.capitalize)
    df.index = pd.to_datetime(df.index, utc=True, errors="raise")
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError("XGBoost requires unique, chronological input timestamps")
    return df[["Close", "High", "Low", "Open", "Volume"]]


def compute_features(data):
    """Compute strictly causal, stationary OHLCV, indicator, and FVG features."""
    close = data["Close"].astype(float)
    open_px = data["Open"].astype(float)
    high = data["High"].astype(float)
    low = data["Low"].astype(float)
    volume = data["Volume"].astype(float)
    eps = 1e-10

    feats = pd.DataFrame(index=data.index)

    # 1. Multi-horizon log returns
    for k in (1, 2, 3, 5, 10, 20, 50):
        feats[f"ret_{k}"] = np.log(close / close.shift(k))

    # 2. Moving-average distances & crossovers
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    feats["dist_sma20"] = (close - sma_20) / (sma_20 + eps)
    #feats["dist_sma50"] = (close - sma_50) / (sma_50 + eps)
    feats["sma20_50_ratio"] = (sma_20 - sma_50) / (sma_50 + eps)
    feats["ema12_26_ratio"] = (ema_12 - ema_26) / (close + eps)

    # 3. ATR, realized volatility & Bollinger Bands
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1,).max(axis=1)
    atr_14 = tr.rolling(14).mean()
    feats["atr_pct"] = atr_14 / (close + eps)

    vol_10 = feats["ret_1"].rolling(10).std()
    vol_20 = feats["ret_1"].rolling(20).std()
    #feats["vol_10"] = vol_10
    #feats["vol_20"] = vol_20
    #feats["vol_ratio"] = vol_10 / (vol_20 + eps)

    #bb_std = close.rolling(20).std()
    #bb_upper = sma_20 + 2.0 * bb_std
    #bb_lower = sma_20 - 2.0 * bb_std
    #feats["bb_pos"] = (close - bb_lower) / (bb_upper - bb_lower + eps)
    #feats["bb_width"] = (bb_upper - bb_lower) / (sma_20 + eps)

    # 4. RSI & slope
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(14).mean()
    loss = (-delta.clip(upper=0.0)).rolling(14).mean()
    rsi = 100.0 - (100.0 / (1.0 + gain / (loss + eps)))
    feats["rsi_14"] = rsi
    feats["rsi_slope_3"] = rsi - rsi.shift(3)

    # 5. Candle microstructure
    bar_range = (high - low).replace(0.0, np.nan)
    feats["body_pct"] = (close - open_px) / (open_px + eps)
    feats["range_pct"] = (high - low) / (open_px + eps)
    feats["upper_wick_ratio"] = ((high - np.maximum(open_px, close)) / bar_range).fillna(0.0)
    feats["lower_wick_ratio"] = ((np.minimum(open_px, close) - low) / bar_range).fillna(0.0)
    feats["close_pos"] = ((close - low) / bar_range).fillna(0.5)

    # 6. Volume features
    vol_med_20 = volume.rolling(20).median()
    feats["rel_vol_20"] = (volume / (vol_med_20 + eps)).clip(0.0, 10.0)
    feats["vol_change_1"] = np.log1p(volume.clip(lower=0.0)).diff(1)
    feats["signed_vol"] = np.sign(close - open_px) * feats["rel_vol_20"]

    # 7. Causal 3-candle FVG features with middle-candle impulse check
    bull_impulse = close.shift(1) > open_px.shift(1)
    bear_impulse = close.shift(1) < open_px.shift(1)
    raw_bull_gap = np.maximum(0.0, low - high.shift(2))
    raw_bear_gap = np.maximum(0.0, low.shift(2) - high)
    bull_gap = np.where(bull_impulse, raw_bull_gap, 0.0)
    bear_gap = np.where(bear_impulse, raw_bear_gap, 0.0)

    feats["fvg_bull_pct"] = bull_gap / (close + eps)
    feats["fvg_bear_pct"] = bear_gap / (close + eps)
    feats["fvg_imbalance"] = (bull_gap - bear_gap) / (close + eps)
    feats["fvg_type"] = np.where(bull_gap > 0.0, 1, np.where(bear_gap > 0.0, -1, 0))

    return feats.replace([np.inf, -np.inf], np.nan)


def compute_resolved_bracket_targets(
    df: pd.DataFrame,
    sl_atr: float = DEFAULT_SL_ATR,
    tp_atr: float = DEFAULT_TP_ATR,
    max_bars: int = DEFAULT_MAX_BARS,
) -> pd.Series:
    """
    Compute binary resolved bracket target (1 = bullish bracket resolves to TP before SL,
    0 = bearish/opposite resolution) over up to `max_bars` future candles.
    """
    close = df["Close"].astype(float).to_numpy()
    high = df["High"].astype(float).to_numpy()
    low = df["Low"].astype(float).to_numpy()
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    atr = pd.Series(tr, index=df.index).rolling(14).mean().to_numpy()

    n = len(df)
    targets = np.full(n, np.nan, dtype=float)

    for i in range(n - 1):
        if not np.isfinite(atr[i]) or atr[i] <= 0.0 or not np.isfinite(close[i]) or close[i] <= 0.0:
            continue
        end_idx = min(n, i + 1 + max_bars)
        if end_idx <= i + 1:
            continue

        tp_long = close[i] + tp_atr * atr[i]
        sl_long = close[i] - sl_atr * atr[i]
        tp_short = close[i] - tp_atr * atr[i]
        sl_short = close[i] + sl_atr * atr[i]

        y_long = np.nan
        for j in range(i + 1, end_idx):
            hit_sl = low[j] <= sl_long
            hit_tp = high[j] >= tp_long
            if hit_sl and hit_tp:
                y_long = 0.0
                break
            if hit_tp:
                y_long = 1.0
                break
            if hit_sl:
                y_long = 0.0
                break

        y_short = np.nan
        for j in range(i + 1, end_idx):
            hit_sl = high[j] >= sl_short
            hit_tp = low[j] <= tp_short
            if hit_sl and hit_tp:
                y_short = 0.0
                break
            if hit_tp:
                y_short = 1.0
                break
            if hit_sl:
                y_short = 0.0
                break

        if np.isfinite(y_long) and np.isfinite(y_short) and y_long != y_short:
            targets[i] = y_long

    return pd.Series(targets, index=df.index, name="target")


class XGBoostModel(Model):
    """Bracket classifier focused on maximizing out-of-sample directional/bracket accuracy."""

    def __init__(
        self,
        symbol,
        timeframe,
        results_path=None,
        horizon: int = HORIZON,
        cv_splits: int = 5,
        confidence_threshold: float = 0.52,
        multi_timeframes=(),
        sl_atr: float = DEFAULT_SL_ATR,
        tp_atr: float = DEFAULT_TP_ATR,
        max_bars: int = DEFAULT_MAX_BARS,
        calibration_method: str = "auto",
        random_state: int = 42,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.horizon = int(horizon)
        self.cv_splits = int(cv_splits)
        self.confidence_threshold = float(confidence_threshold)
        if multi_timeframes:
            raise ValueError("XGBoost multi-timeframe features are not implemented")
        if max_bars < 1 or sl_atr <= 0 or tp_atr <= 0 or cv_splits < 2:
            raise ValueError("max_bars, SL and TP must be positive and cv_splits must be at least 2")
        if not 0.5 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.5 and 1.0")
        self.multi_timeframes = ()
        self.sl_atr = float(sl_atr)
        self.tp_atr = float(tp_atr)
        self.max_bars = int(max_bars)
        if calibration_method not in ('auto', 'none', 'logistic', 'isotonic'):
            raise ValueError("calibration_method must be 'auto', 'none', 'logistic' or 'isotonic'")
        self.calibration_method_requested = calibration_method
        self.calibration_method = None
        self.random_state = int(random_state)
        self.results_path = (Path(results_path) if results_path else PROJECT_DIR / "results" / f"xgboost_{symbol}_{timeframe}.json")

        self.model = None
        self.search = None
        self.calibrator = None
        self.decision_threshold = 0.50
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.curr_train = self.curr_test = None
        self.train_prob = self.test_prob = None
        self.train_pred = self.test_pred = None
        self.train_acc = None
        self.test_acc = None
        self.test_balanced_acc = None
        self.test_roc_auc = None
        self.confident_test_acc = None
        self.train_time = None
        self.sample_summary = {}

    def feature_engineering(self, train_data, test_data, auxiliary_data=None, **kwargs):
        train_df = _normalize_ohlcv(train_data)
        test_df = _normalize_ohlcv(test_data)

        if train_df.equals(test_df):
            split_at = int(len(train_df) * 0.8)
            train_df, test_df = train_df.iloc[:split_at], train_df.iloc[split_at:]
        if train_df.empty or test_df.empty or train_df.index[-1] >= test_df.index[0]:
            raise ValueError("Training candles must end before test candles begin")
        self._walk_forward_candles = pd.concat([train_df, test_df])

        # Later chronological periods fit the calibrator, select the method,
        # and tune the threshold; bracket target windows cannot cross boundaries.
        n = len(train_df)
        validation_start = int(n * 0.8)
        selection_start = validation_start + (n - validation_start) // 2
        threshold_start = selection_start + (n - selection_start) // 2
        if (validation_start <= self.max_bars or selection_start - validation_start <= self.max_bars
                or threshold_start - selection_start <= self.max_bars):
            raise ValueError("Not enough training candles for purged calibration and threshold periods")

        def _make_split(df, purge_tail: bool = True):
            feats = compute_features(df)
            target = compute_resolved_bracket_targets(
                df,
                sl_atr=self.sl_atr,
                tp_atr=self.tp_atr,
                max_bars=self.max_bars,
            )
            curr_close = df["Close"].astype(float)
            valid = feats.notna().all(axis=1) & target.notna() & curr_close.notna()
            if purge_tail and self.max_bars > 0 and len(valid) > self.max_bars:
                valid.iloc[-self.max_bars :] = False
            return (
                feats.loc[valid].copy(),
                target.loc[valid].astype(int).copy(),
                curr_close.loc[valid].copy(),
            )

        # Compute features on the chronological training stream before slicing,
        # so validation features retain past indicator history without lookahead.
        all_X, all_y, all_close = _make_split(train_df, purge_tail=True)
        positions = pd.Series(np.arange(n), index=train_df.index).reindex(all_X.index)
        fit_mask = positions < validation_start - self.max_bars
        calibration_mask = (positions >= validation_start) & (positions < selection_start - self.max_bars)
        method_mask = (positions >= selection_start) & (positions < threshold_start - self.max_bars)
        threshold_mask = positions >= threshold_start
        self.X_train, self.y_train, self.curr_train = all_X.loc[fit_mask], all_y.loc[fit_mask], all_close.loc[fit_mask]
        self.X_calibration, self.y_calibration = all_X.loc[calibration_mask], all_y.loc[calibration_mask]
        self.X_calibration_validation = all_X.loc[method_mask]
        self.y_calibration_validation = all_y.loc[method_mask]
        self.X_validation, self.y_validation = all_X.loc[threshold_mask], all_y.loc[threshold_mask]
        self.train_positions = positions.loc[fit_mask].to_numpy(dtype=int)
        self.X_test, self.y_test, self.curr_test = _make_split(test_df, purge_tail=True)
        self._test_ohlcv = test_df.copy()
        self.test_trend_pred = (test_df["Close"] > test_df["Close"].shift(1)).astype(int).reindex(self.X_test.index)

        for name, features, labels in (("training", self.X_train, self.y_train),
                                       ("threshold validation", self.X_validation, self.y_validation),
                                       ("test", self.X_test, self.y_test)):
            if features.empty or labels.nunique() < 2:
                raise ValueError(f"Not enough resolved samples of both classes in {name} period")
        self.sample_summary = {
            "supplied_train_candles": len(train_df), "supplied_test_candles": len(test_df),
            "fit_samples": len(self.X_train), "calibration_samples": len(self.X_calibration),
            "calibration_selection_samples": len(self.X_calibration_validation),
            "threshold_samples": len(self.X_validation), "test_samples": len(self.X_test),
            "test_resolved_fraction_of_candles": len(self.X_test) / len(test_df),
            "train_classes": {str(k): int(v) for k, v in self.y_train.value_counts().sort_index().items()},
            "test_classes": {str(k): int(v) for k, v in self.y_test.value_counts().sort_index().items()},
            "fit_period": [str(self.X_train.index[0]), str(self.X_train.index[-1])],
            "calibration_period": ([str(self.X_calibration.index[0]), str(self.X_calibration.index[-1])] if len(self.X_calibration) else None),
            "calibration_selection_period": ([str(self.X_calibration_validation.index[0]), str(self.X_calibration_validation.index[-1])] if len(self.X_calibration_validation) else None),
            "threshold_period": [str(self.X_validation.index[0]), str(self.X_validation.index[-1])],
            "test_period": [str(self.X_test.index[0]), str(self.X_test.index[-1])],
        }
        print(f"Resolved test samples: {len(self.X_test)}/{len(test_df)} candles "
              f"({100 * self.sample_summary['test_resolved_fraction_of_candles']:.1f}%)")

    def train(self):
        if self.X_train is None:
            raise RuntimeError("Run feature_engineering() before train()")

        param_grid = {
            "n_estimators": [150],
            "max_depth": [2, 3],
            "learning_rate": [0.01, 0.02],
            "min_child_weight": [15, 25],
            "reg_alpha": [1.0],
            "reg_lambda": [10.0],
            "subsample": [0.8],
            "colsample_bytree": [0.7, 0.8],
            "scale_pos_weight": [1.0],
        }
        splits = min(self.cv_splits, max(2, len(self.X_train) // 50))
        tscv = TimeSeriesSplit(n_splits=splits)
        folds = []
        for train_idx, val_idx in tscv.split(self.X_train):
            # Sample rows are filtered; the gap must be in original candle
            # positions rather than in the surviving sample count.
            train_idx = train_idx[self.train_positions[train_idx] + self.max_bars < self.train_positions[val_idx[0]]]
            if len(train_idx) == 0 or self.y_train.iloc[train_idx].nunique() < 2 or self.y_train.iloc[val_idx].nunique() < 2:
                raise ValueError("A purged time-series fold has too few samples or only one class")
            folds.append((train_idx, val_idx))
        self.search = GridSearchCV(
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=self.random_state,
                n_jobs=1,
            ),
            param_grid=param_grid,
            cv=folds,
            scoring="roc_auc",
            n_jobs=1,
            verbose=0,
        )

        start = time.perf_counter()
        self.search.fit(self.X_train, self.y_train)
        self.train_time = time.perf_counter() - start
        self.model = self.search.best_estimator_

        raw_calibration = (self.model.predict_proba(self.X_calibration)[:, 1]
                           if len(self.X_calibration) else np.array([]))
        raw_method_validation = (self.model.predict_proba(self.X_calibration_validation)[:, 1]
                                 if len(self.X_calibration_validation) else np.array([]))
        methods = ('none', 'logistic', 'isotonic') if self.calibration_method_requested == 'auto' else (self.calibration_method_requested,)
        self.calibrator, self.calibration_scores = select_probability_calibrator(
            raw_calibration, self.y_calibration.to_numpy(dtype=int), raw_method_validation,
            self.y_calibration_validation.to_numpy(dtype=int), methods)
        self.calibration_method = self.calibrator.method
        selected_details = next(x for x in self.calibration_scores if x['method'] == self.calibration_method and x['eligible'])
        selected_loss = selected_details['validation_log_loss']
        print(f"Selected calibration: {self.calibration_method}; validation log loss: "
              f"{selected_loss:.4f}" if selected_loss is not None else
              f"Selected calibration: {self.calibration_method}; validation log loss: unavailable")
        if 'fallback_reason' in selected_details:
            print(f"Calibration fallback: {selected_details['fallback_reason']}")
        validation_probability = self.calibrator.predict_proba(self.model.predict_proba(self.X_validation)[:, 1])
        scores = [(float(t), float(accuracy_score(self.y_validation, validation_probability >= t)))
                  for t in np.linspace(0.45, 0.55, 21)]
        self.decision_threshold, self.validation_accuracy = max(scores, key=lambda item: (item[1], -abs(item[0] - 0.5)))
        self.threshold_scores = [{"threshold": t, "accuracy": score} for t, score in scores]
        y_tr = self.y_train.to_numpy(dtype=int)

        raw_train_prob = self.model.predict_proba(self.X_train)[:, 1]
        self.train_prob = self.calibrator.predict_proba(raw_train_prob)
        self.train_pred = (self.train_prob >= self.decision_threshold).astype(int)
        self.train_acc = float(accuracy_score(y_tr, self.train_pred))

        print(
            f"Best params: {self.search.best_params_} | decision_threshold={self.decision_threshold:.2f} "
            f"\nTrain Acc: {self.train_acc * 100:.2f}%"
        )
        return self.train_acc

    def test(self):
        if self.model is None:
            raise RuntimeError("Run train() before test()")

        raw_test_prob = self.model.predict_proba(self.X_test)[:, 1]
        self.test_prob = self.calibrator.predict_proba(raw_test_prob)
        self.test_pred = (self.test_prob >= self.decision_threshold).astype(int)

        y_te = self.y_test.to_numpy(dtype=int)
        self.test_acc = float(accuracy_score(y_te, self.test_pred))
        self.test_balanced_acc = float(balanced_accuracy_score(y_te, self.test_pred))
        try:
            self.raw_test_roc_auc = float(roc_auc_score(y_te, raw_test_prob))
            self.test_roc_auc = float(roc_auc_score(y_te, self.test_prob))
        except ValueError:
            self.raw_test_roc_auc = float("nan")
            self.test_roc_auc = float("nan")
        self.test_prediction_counts = {
            'bearish': int((self.test_pred == 0).sum()),
            'bullish': int((self.test_pred == 1).sum()),
        }
        self.test_probability_range = [float(np.min(self.test_prob)), float(np.max(self.test_prob))]

        predicted_class_confidence = np.where(self.test_pred == 1, self.test_prob, 1.0 - self.test_prob)
        conf_mask = predicted_class_confidence >= self.confidence_threshold
        self.confident_coverage = float(conf_mask.mean())
        majority_class = int(self.y_train.mean() > 0.5)
        self.majority_baseline_acc = float(accuracy_score(y_te, np.full(len(y_te), majority_class)))
        self.trend_baseline_acc = float(accuracy_score(y_te, self.test_trend_pred))
        print(f"Baselines: training-majority {self.majority_baseline_acc:.2%}, recent candle direction {self.trend_baseline_acc:.2%}")
        print(f"Test predictions: {self.test_prediction_counts['bullish']} bullish, "
              f"{self.test_prediction_counts['bearish']} bearish; raw ROC-AUC: {self.raw_test_roc_auc:.4f}")
        if not self.test_prediction_counts['bullish'] or not self.test_prediction_counts['bearish']:
            print("WARNING: The model predicts only one class across the resolved test samples.")
        if np.any(conf_mask):
            self.confident_test_acc = float(accuracy_score(y_te[conf_mask], self.test_pred[conf_mask]))
            cov = float(np.mean(conf_mask)) * 100.0
            print(
                f"Test Acc: {self.test_acc * 100:.2f}% | ROC-AUC: {self.test_roc_auc:.4f} | "
                f"Confident (>={self.confidence_threshold:.2f}) Acc: {self.confident_test_acc * 100:.2f}% "
                f"(cov={cov:.1f}%)"
            )
        else:
            self.confident_test_acc = None
            print(
                f"Test Acc: {self.test_acc * 100:.2f}% | ROC-AUC: {self.test_roc_auc:.4f} | "
                f"No confident test signals"
            )

        return {
            "train_accuracy": self.train_acc,
            "test_accuracy": self.test_acc,
            "test_balanced_accuracy": self.test_balanced_acc,
            "test_roc_auc": self.test_roc_auc,
            "raw_test_roc_auc": self.raw_test_roc_auc,
            "test_prediction_counts": self.test_prediction_counts,
            "confident_test_accuracy": self.confident_test_acc,
            "confident_coverage": self.confident_coverage,
            "majority_baseline_accuracy": self.majority_baseline_acc,
            "recent_trend_baseline_accuracy": self.trend_baseline_acc,
        }

    def bracket_backtest(self, position_value=100.0, commission=0.001, initial_cash=10000.0):
        """Simulate one bracket at a time on all test candles with known features.

        Signals use a completed close and execute at the next open. Stops and
        targets use ATR from the signal candle and include entry-bar highs/lows.
        A candle touching both levels is treated as a stop. Gap-through exits
        fill at the open. Unresolved positions exit at the time limit or dataset
        end. This is a simple notional simulation, not the separate Backtrader
        displacement strategy used by main.py.
        """
        if self.model is None:
            raise RuntimeError("Run train() before backtesting")
        if position_value <= 0 or initial_cash < position_value or not 0 <= commission < 1:
            raise ValueError("Invalid position value, starting cash, or commission")
        df = self._test_ohlcv
        features = compute_features(df)[self.X_train.columns]
        valid = features.notna().all(axis=1)
        probabilities = np.full(len(df), np.nan)
        probabilities[valid.to_numpy()] = self.calibrator.predict_proba(self.model.predict_proba(features.loc[valid])[:, 1])
        close = df["Close"].to_numpy(dtype=float)
        high = df["High"].to_numpy(dtype=float)
        low = df["Low"].to_numpy(dtype=float)
        opens = df["Open"].to_numpy(dtype=float)
        previous = np.roll(close, 1)
        previous[0] = close[0]
        tr = np.maximum.reduce([high - low, np.abs(high - previous), np.abs(low - previous)])
        atr = pd.Series(tr).rolling(14).mean().to_numpy()
        equity = peak = initial_cash
        max_drawdown = fees = 0.0
        wins = losses = trades = 0
        i = 0
        while i < len(df) - 1:
            probability = probabilities[i]
            if not np.isfinite(probability) or not np.isfinite(atr[i]) or atr[i] <= 0:
                i += 1
                continue
            direction = 1 if probability >= self.decision_threshold else -1
            confidence = probability if direction == 1 else 1.0 - probability
            entry = opens[i + 1]
            if confidence < self.confidence_threshold or not np.isfinite(entry) or entry <= 0 or equity < position_value:
                i += 1
                continue
            stop = entry - direction * self.sl_atr * atr[i]
            target = entry + direction * self.tp_atr * atr[i]
            if stop <= 0 or target <= 0:
                i += 1
                continue
            end = min(len(df) - 1, i + self.max_bars)
            exit_index, exit_price = end, close[end]
            for j in range(i + 1, end + 1):
                if direction == 1:
                    if opens[j] <= stop or opens[j] >= target:
                        exit_index, exit_price = j, opens[j]
                        break
                    stop_hit, target_hit = low[j] <= stop, high[j] >= target
                else:
                    if opens[j] >= stop or opens[j] <= target:
                        exit_index, exit_price = j, opens[j]
                        break
                    stop_hit, target_hit = high[j] >= stop, low[j] <= target
                if stop_hit or target_hit:
                    exit_index, exit_price = j, stop if stop_hit else target
                    break
            quantity = position_value / entry
            fee = commission * quantity * (entry + exit_price)
            net = direction * quantity * (exit_price - entry) - fee
            equity += net
            fees += fee
            trades += 1
            wins += net > 0
            losses += net < 0
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, (peak - equity) / peak)
            i = exit_index
        return {
            "method": "one position at a time; next-open entry; SL first on ambiguous candles; close at timeout or dataset end",
            "position_value": float(position_value), "commission_per_side": float(commission),
            "initial_cash": float(initial_cash), "final_value": float(equity),
            "total_return_pct": float((equity / initial_cash - 1) * 100),
            "max_closed_equity_drawdown_pct": float(max_drawdown * 100),
            "closed_trades": trades, "winning_trades": int(wins), "losing_trades": int(losses),
            "win_rate_pct": float(100 * wins / trades) if trades else None,
            "total_commission": float(fees),
        }

    def predict_next_candle(self, ohlcv_data) -> pd.Series:
        """
        Return directional price targets scaled by calibrated bracket probability
        so downstream backtest code stays compatible while reflecting bracket conviction.
        """
        if self.model is None or self.X_train is None:
            raise RuntimeError("Run train() before predicting")

        df = _normalize_ohlcv(ohlcv_data)
        feats = compute_features(df)[self.X_train.columns]
        close = df["Close"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        prev_close = close.shift(1).fillna(close)
        tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        valid = feats.notna().all(axis=1) & close.notna() & atr.notna()
        pred_price = pd.Series(np.nan, index=df.index, name="predicted_price", dtype=float)
        if valid.any():
            raw_p = self.model.predict_proba(feats.loc[valid])[:, 1]
            cal_p = self.calibrator.predict_proba(raw_p)
            signed_edge = np.clip((cal_p - self.decision_threshold) * 4.0, -1.0, 1.0)
            step_pct = np.clip(signed_edge * (atr.loc[valid].to_numpy(dtype=float) / np.maximum(close.loc[valid].to_numpy(dtype=float), 1e-10)), -0.05, 0.05)
            pred_price.loc[valid] = close.loc[valid].to_numpy(dtype=float) * (1.0 + step_pct)
        return pred_price

    def _plot_walk_forward_prices(self, model, history, test, number):
        """Plot the existing direction-implied price proxy at the next candle."""
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        # predict_next_candle uses only data up to the signal candle. Shift its
        # output once to align each signal with the next available candle.
        proxy = model.predict_next_candle(history).shift(1).reindex(test.index)
        if len(proxy):
            proxy.iloc[0] = np.nan  # Do not plot a signal from before the test period.
        fig = Figure(figsize=(13, 5), layout="constrained")
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        ax.plot(test.index, test["Close"], label="Actual close", color="black", linewidth=1.1)
        ax.plot(test.index, proxy, label="Direction-implied price proxy", color="tab:blue", alpha=.8, linewidth=.8)
        ax.set_title(f"{self.symbol} {self.timeframe} — walk-forward period {number}\n"
                     "Proxy is ATR-scaled direction, not a trained price forecast")
        ax.set_xlabel("Candle time (UTC)")
        ax.set_ylabel("Price")
        ax.grid(alpha=.2)
        ax.legend()
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
        safe_symbol = "".join(c for c in self.symbol if c.isalnum() or c in "-_")
        safe_timeframe = "".join(c for c in self.timeframe if c.isalnum() or c in "-_")
        path = self.results_path.parent / f"xgboost_{safe_symbol}_{safe_timeframe}_period{number}_{stamp}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=140)
        return str(path)

    def _walk_forward_evaluation(self, n_splits=3):
        """Retrain on expanding historical windows and evaluate later candles."""
        candles = self._walk_forward_candles
        report = {
            "method": "expanding chronological windows; independently refit model, calibration and threshold",
            "gap_candles": self.max_bars,
            "accuracy_scope": "resolved bracket labels only; not next-candle direction accuracy",
            "all_candle_counts_scope": "feature-ready candles, regardless of whether their future brackets resolve",
            "folds": [],
        }
        try:
            splits = list(TimeSeriesSplit(n_splits=n_splits, gap=self.max_bars).split(candles))
        except ValueError as exc:
            report["error"] = str(exc)
            return report

        print("\nWalk-forward evaluation (resolved bracket outcomes; three historical periods):")
        for number, (train_idx, test_idx) in enumerate(splits, 1):
            train, test = candles.iloc[train_idx], candles.iloc[test_idx]
            fold = {
                "fold": number,
                "train_period": [str(train.index[0]), str(train.index[-1])],
                "test_period": [str(test.index[0]), str(test.index[-1])],
                "train_candles": len(train),
                "test_candles": len(test),
            }
            print(f"\nPeriod {number}/{n_splits}: {fold['test_period'][0]} to {fold['test_period'][1]}")
            model = XGBoostModel(
                self.symbol, self.timeframe, horizon=self.horizon, cv_splits=self.cv_splits,
                confidence_threshold=self.confidence_threshold, sl_atr=self.sl_atr,
                tp_atr=self.tp_atr, max_bars=self.max_bars,
                calibration_method=self.calibration_method_requested, random_state=self.random_state,
            )
            try:
                model.feature_engineering(train, test)
                model.train()
                model.test()
            except ValueError as exc:
                fold.update(status="skipped", reason=str(exc))
                report["folds"].append(fold)
                print(f"Skipped: {exc}")
                continue

            actual = model.y_test.to_numpy(dtype=int)
            predicted = np.asarray(model.test_pred, dtype=int)
            direction = {}
            for label, value in (("bearish", 0), ("bullish", 1)):
                actual_count = int((actual == value).sum())
                predicted_count = int((predicted == value).sum())
                correct = int(((actual == value) & (predicted == value)).sum())
                direction[label] = {
                    "actual": actual_count, "predicted": predicted_count, "correct": correct,
                    "recall": correct / actual_count if actual_count else None,
                    "precision": correct / predicted_count if predicted_count else None,
                }

            # Include indicator history before this period, but never future candles.
            features = compute_features(candles.iloc[:test_idx[-1] + 1]).loc[test.index]
            valid = features.notna().all(axis=1)
            if valid.any():
                raw = model.model.predict_proba(features.loc[valid])[:, 1]
                probs = model.calibrator.predict_proba(raw)
                all_predictions = (probs >= model.decision_threshold).astype(int)
                all_counts = {"bearish": int((all_predictions == 0).sum()),
                              "bullish": int((all_predictions == 1).sum())}
            else:
                all_counts = {"bearish": 0, "bullish": 0}
            chart_path = self._plot_walk_forward_prices(
                model, candles.iloc[:test_idx[-1] + 1], test, number)
            fold.update({
                "status": "ok", "resolved_samples": len(actual),
                "resolved_fraction_of_candles": len(actual) / len(test),
                "feature_ready_candles": int(valid.sum()),
                "all_candle_prediction_counts": all_counts,
                "resolved_class_metrics": direction,
                "accuracy": model.test_acc, "balanced_accuracy": model.test_balanced_acc,
                "roc_auc": model.test_roc_auc, "raw_roc_auc": model.raw_test_roc_auc,
                "majority_baseline_accuracy": model.majority_baseline_acc,
                "recent_trend_baseline_accuracy": model.trend_baseline_acc,
                "confident_accuracy": model.confident_test_acc,
                "confident_coverage": model.confident_coverage,
                "decision_threshold": model.decision_threshold,
                "calibration_method": model.calibration_method,
                "calibration_scores": model.calibration_scores,
                "sample_summary": model.sample_summary,
                "price_chart": chart_path,
                "price_chart_note": "ATR-scaled direction-implied proxy aligned to the next available candle, not a trained price forecast",
            })
            report["folds"].append(fold)
            print(f"All feature-ready candles: {all_counts['bullish']} bullish, "
                  f"{all_counts['bearish']} bearish ({valid.sum()}/{len(test)})")
            print(f"Price chart saved: {chart_path}")
            print(f"Resolved accuracy: {model.test_acc:.2%}; coverage: {len(actual) / len(test):.2%}; "
                  f"bullish recall: {direction['bullish']['recall']:.2%}; "
                  f"bearish recall: {direction['bearish']['recall']:.2%}")
            for label in ("bullish", "bearish"):
                precision = direction[label]["precision"]
                print(f"  {label.capitalize()} precision: "
                      f"{precision:.2%}" if precision is not None else
                      f"  {label.capitalize()} precision: unavailable (no predictions)")
        return report

    def evaluate(self):
        if self.test_acc is None:
            raise RuntimeError("Run test() before evaluate()")

        record = {
            "symbol": self.symbol,
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": "XGBoostBracketClassifier",
            "timeframe": self.timeframe,
            "sl_atr": self.sl_atr,
            "tp_atr": self.tp_atr,
            "max_bars": self.max_bars,
            "train_time": float(self.train_time),
            "params": self.search.best_params_ if self.search is not None else {},
            "decision_threshold": self.decision_threshold,
            "confidence_threshold": self.confidence_threshold,
            "calibration_method": self.calibration_method,
            "calibration_method_requested": self.calibration_method_requested,
            "calibration_selection_metric": "minimum log loss among ranking-preserving methods on a separate chronological calibration-selection period; use no calibration when selection is unavailable",
            "calibration_scores": self.calibration_scores,
            "calibration_source": "separate chronological calibration period",
            "threshold_source": "separate chronological validation period",
            "validation_accuracy": self.validation_accuracy,
            "threshold_scores": self.threshold_scores,
            "sample_summary": self.sample_summary,
            "train_accuracy": self.train_acc,
            "test_accuracy": self.test_acc,
            "test_balanced_accuracy": self.test_balanced_acc,
            "test_roc_auc": self.test_roc_auc,
            "raw_test_roc_auc": self.raw_test_roc_auc,
            "test_prediction_counts": self.test_prediction_counts,
            "test_probability_range": self.test_probability_range,
            "confident_test_accuracy": self.confident_test_acc,
            "confident_coverage": self.confident_coverage,
            "majority_baseline_accuracy": self.majority_baseline_acc,
            "recent_trend_baseline_accuracy": self.trend_baseline_acc,
        }
        """
        record["walk_forward"] = self._walk_forward_evaluation()
        results = []
        if self.results_path.exists():
            with self.results_path.open() as f:
                results = json.load(f)
        results.append(record)
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with self.results_path.open("w") as f:
            json.dump(results, f, indent=2)"""
        return record
