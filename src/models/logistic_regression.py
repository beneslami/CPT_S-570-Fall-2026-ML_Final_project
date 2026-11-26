"""Regularized stationary-feature logistic classifier for directional prediction."""

import json
import pathlib
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import RobustScaler

from ML_interface import Model
from models.XGBoost import (
    DEFAULT_MAX_BARS,
    DEFAULT_SL_ATR,
    DEFAULT_TP_ATR,
    HORIZON,
    _normalize_ohlcv,
    compute_direction_sample_weights,
    compute_directional_targets,
    compute_features,
    compute_recency_weights,
    compute_resolved_bracket_targets,
)


LEAN_LOGISTIC_FEATURES = (
    # Core momentum, mean-reversion & microstructure
    "rsi_7",
    "rsi_centered",
    "ret_streak",
    "close_pos",
    "wick_imbalance",
    "lower_wick_ratio",
    "volume_delta_ratio",
    "dist_sma20",
    "dist_sma50",
    "rvol_ratio_6_24",
    "bb_width_20",
    "signed_vol_sum_6",
    "up_ratio_6",
    "ret1_atr_sq_signed",
    "hour_sin",
    "hour_cos",
    # Rolling multi-timeframe structure
    "htf4_close_pos",
    "htf12_wick_imb",
    "htf24_close_pos",
    "htf24_body_to_range",
    # Candlestick patterns
    "engulfing",
    "fvg_imbalance",
    "fvg_type",
    # Auxiliary timeframe summaries (included when available)
    "tf4h_close_pos",
    "tf4h_body_atr",
    "tf15m_last_cp",
    "tf5m_last_cp",
    # Binance derivatives & order-flow (included when present in datapoints_new/)
    "perp_tbr_lag1",
    "spot_tbr",
    "spot_tbr_ma6",
    "prem_z24",
    "sum_taker_long_short_vol_ratio_z24",
)


def compute_logistic_features(ohlcv, auxiliary_data=None, timeframe: str | None = None):
    """Lean, non-redundant stationary feature set (~25-30 features) plus candlestick patterns."""
    data = _normalize_ohlcv(ohlcv)
    feats = compute_features(data, include_fvg_ob=False, auxiliary_data=auxiliary_data, timeframe=timeframe)
    feats["rsi_centered"] = (feats["rsi_14"] - 50.0) / 50.0
    selected = [col for col in LEAN_LOGISTIC_FEATURES if col in feats.columns]
    return feats.loc[:, selected].replace([np.inf, -np.inf], np.nan)


class LogisticRegressionModel(Model):
    CANDIDATE_GRID = (
        (0.0, 0.0003),
        (0.0, 0.001),
        (0.0, 0.003),
        (0.0, 0.01),
        (0.5, 0.005),
        (0.5, 0.01),
        (0.5, 0.02),
        (1.0, 0.005),
        (1.0, 0.01),
        (1.0, 0.02),
    )

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        max_iter: int = 1000,
        random_state: int = 42,
        results_path=None,
        sl_atr=DEFAULT_SL_ATR,
        tp_atr=DEFAULT_TP_ATR,
        max_bars=DEFAULT_MAX_BARS,
        horizon: int = HORIZON,
        target_mode: str = "direction",
        min_move_atr: float = 0.0,
        use_sample_weights: bool = False,
        recency_half_life_days: float | None = 730.0,
    ):
        if target_mode not in ("direction", "bracket"):
            raise ValueError("target_mode must be 'direction' or 'bracket'")
        self.symbol, self.timeframe = symbol, timeframe
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self.model = LogisticRegression(max_iter=self.max_iter, random_state=self.random_state, solver="lbfgs")
        self.best_C = 1.0
        self.best_l1_ratio = 0.0
        self.n_nonzero_features = None
        self.scaler = RobustScaler()
        self.sl_atr, self.tp_atr, self.max_bars = float(sl_atr), float(tp_atr), int(max_bars)
        self.horizon = int(horizon)
        self.target_mode = target_mode
        self.min_move_atr = float(min_move_atr)
        self.use_sample_weights = bool(use_sample_weights)
        self.recency_half_life_days = float(recency_half_life_days) if recency_half_life_days is not None else None
        self.purge_gap = self.horizon if self.target_mode == "direction" else self.max_bars
        if self.sl_atr <= 0 or self.tp_atr <= 0 or self.max_bars < 1 or self.horizon < 1:
            raise ValueError("horizon, bracket distances, and max_bars must be positive")
        project_dir = pathlib.Path(__file__).resolve().parents[2]
        self.results_path = (
            pathlib.Path(results_path)
            if results_path
            else project_dir / "results" / f"logistic_regression_{symbol}_{timeframe}.json"
        )
        self.features = None
        self.decision_threshold = 0.5
        self.validation_acc = None
        self.validation_balanced_acc = None
        self.train_time = 0.0
        self.training_acc = self.test_acc = self.test_balanced_acc = self.test_roc_auc = None
        self.X_train = self.X_validation = self.X_test = None
        self.y_train = self.y_validation = self.y_test = self.y_pred = None
        self.w_train = None
        self._auxiliary_data = None

    def _compute_targets(self, frame):
        if self.target_mode == "direction":
            return compute_directional_targets(frame, horizon=self.horizon, min_move_atr=self.min_move_atr)
        return compute_resolved_bracket_targets(
            frame, sl_atr=self.sl_atr, tp_atr=self.tp_atr, max_bars=self.max_bars
        )

    def _scale(self, X, fit: bool = False):
        arr = self.scaler.fit_transform(X) if fit else self.scaler.transform(X)
        return np.clip(arr, -5.0, 5.0)

    def feature_engineering(self, train_data, test_data, auxiliary_data=None, **kwargs):
        train, test = _normalize_ohlcv(train_data), _normalize_ohlcv(test_data)
        self._auxiliary_data = auxiliary_data
        if train.empty or test.empty or train.index[-1] >= test.index[0]:
            raise ValueError("Training candles must end before test candles begin")
        self._train_close = train["Close"].astype(float).copy()
        self._test_close = test["Close"].astype(float).copy()
        all_features = compute_logistic_features(
            pd.concat([train, test]),
            auxiliary_data=self._auxiliary_data,
            timeframe=self.timeframe,
        )
        self.features = all_features.columns.tolist()
        gap = self.purge_gap

        def partition(frame):
            labels = self._compute_targets(frame)
            weights = (
                compute_direction_sample_weights(frame, horizon=self.horizon)
                if (self.target_mode == "direction" and self.use_sample_weights)
                else pd.Series(1.0, index=frame.index, dtype=float)
            )
            valid = labels.notna() & all_features.loc[frame.index].notna().all(axis=1)
            if gap > 0 and len(valid) > gap:
                valid.iloc[-gap:] = False
            return (
                all_features.loc[frame.index].loc[valid, self.features].copy(),
                labels.loc[valid].astype(int),
                weights.loc[valid].astype(float),
            )

        full_X_train, full_y_train, full_w_train = partition(train)
        self.X_test, self.y_test, _ = partition(test)

        # Purged chronological 80/20 split inside train for C & threshold tuning
        n = len(train)
        val_start = int(n * 0.80)
        positions = pd.Series(np.arange(n), index=train.index).reindex(full_X_train.index)
        fit_mask = positions < (val_start - gap)
        val_mask = positions >= val_start

        if (
            fit_mask.sum() >= 50
            and val_mask.sum() >= 50
            and full_y_train.loc[fit_mask].nunique() == 2
            and full_y_train.loc[val_mask].nunique() == 2
        ):
            self.X_train = full_X_train.loc[fit_mask].copy()
            self.y_train = full_y_train.loc[fit_mask].copy()
            self.w_train = full_w_train.loc[fit_mask].copy()
            self.X_validation = full_X_train.loc[val_mask].copy()
            self.y_validation = full_y_train.loc[val_mask].copy()
        else:
            self.X_train, self.y_train, self.w_train = full_X_train, full_y_train, full_w_train
            self.X_validation, self.y_validation = None, None

        if self.X_train.empty or self.y_train.nunique() < 2:
            raise ValueError("Logistic training requires both target classes")
        if self.X_test.empty:
            raise ValueError("Logistic test requires valid samples")
        self.n_train, self.n_test = len(self.X_train), len(self.X_test)

    def train(self):
        if self.X_train is None:
            raise RuntimeError("Run feature_engineering() before train()")

        start = time.perf_counter()
        x_train = self._scale(self.X_train, fit=True)
        if self.use_sample_weights:
            w_train = self.w_train.to_numpy(dtype=float)
        elif self.recency_half_life_days is not None and self.recency_half_life_days > 0:
            w_train = compute_recency_weights(self.X_train.index, self.recency_half_life_days)
        else:
            w_train = np.ones(len(self.X_train), dtype=float)

        if self.X_validation is not None and len(self.X_validation) > 0:
            x_val = self._scale(self.X_validation, fit=False)
            best_score = None
            best_model = None
            best_c = 0.01
            best_l1 = 0.0
            for l1_ratio, c_val in self.CANDIDATE_GRID:
                if l1_ratio > 0.0:
                    clf = LogisticRegression(
                        C=c_val,
                        l1_ratio=l1_ratio,
                        solver="saga",
                        max_iter=500,
                        random_state=self.random_state,
                    )
                else:
                    clf = LogisticRegression(
                        C=c_val,
                        max_iter=self.max_iter,
                        random_state=self.random_state,
                        solver="lbfgs",
                    )
                clf.fit(x_train, self.y_train, sample_weight=w_train)
                val_prob = clf.predict_proba(x_val)[:, 1]
                ll = float(log_loss(self.y_validation, val_prob, labels=[0, 1]))
                auc = float(roc_auc_score(self.y_validation, val_prob)) if len(np.unique(self.y_validation)) > 1 else 0.5
                score = (round(ll, 4), -auc)
                if best_score is None or score < best_score:
                    best_score = score
                    best_model = clf
                    best_c = c_val
                    best_l1 = l1_ratio
            self.model = best_model
            self.best_C = float(best_c)
            self.best_l1_ratio = float(best_l1)
            self.n_nonzero_features = int((np.abs(self.model.coef_[0]) > 1e-5).sum())

            val_prob = self.model.predict_proba(x_val)[:, 1]
            scores = [
                (
                    float(t),
                    float(balanced_accuracy_score(self.y_validation, val_prob >= t)),
                    float(accuracy_score(self.y_validation, val_prob >= t)),
                )
                for t in np.linspace(0.45, 0.55, 21)
            ]
            self.decision_threshold, self.validation_balanced_acc, self.validation_acc = max(
                scores, key=lambda item: (item[1], -abs(item[0] - 0.5))
            )
        else:
            self.best_C = 0.01
            self.best_l1_ratio = 0.5
            self.model = LogisticRegression(
                C=self.best_C, l1_ratio=self.best_l1_ratio, solver="saga", max_iter=500, random_state=self.random_state
            )
            self.model.fit(x_train, self.y_train, sample_weight=w_train)
            self.n_nonzero_features = int((np.abs(self.model.coef_[0]) > 1e-5).sum())
            self.decision_threshold = 0.5
            self.validation_acc = self.validation_balanced_acc = None

        self.train_time = time.perf_counter() - start
        train_prob = self.model.predict_proba(x_train)[:, 1]
        self.training_acc = float(accuracy_score(self.y_train, (train_prob >= self.decision_threshold).astype(int)))
        return self.training_acc

    def predict_probabilities(self, data):
        if self.training_acc is None:
            raise RuntimeError("Run train() before predicting")
        frame = compute_logistic_features(
            data,
            auxiliary_data=self._auxiliary_data,
            timeframe=self.timeframe,
        ).loc[:, self.features]
        valid = frame.notna().all(axis=1)
        result = pd.Series(np.nan, index=frame.index, name="logistic_probability", dtype=float)
        if valid.any():
            result.loc[valid] = self.model.predict_proba(self._scale(frame.loc[valid], fit=False))[:, 1]
        return result

    def test(self):
        if self.training_acc is None:
            raise RuntimeError("Run train() before test()")
        test_prob = self.model.predict_proba(self._scale(self.X_test, fit=False))[:, 1]
        self.y_pred = (test_prob >= self.decision_threshold).astype(int)
        self.test_acc = float(accuracy_score(self.y_test, self.y_pred))
        self.test_balanced_acc = float(balanced_accuracy_score(self.y_test, self.y_pred))
        self.test_roc_auc = (
            float(roc_auc_score(self.y_test, test_prob)) if len(np.unique(self.y_test)) > 1 else float("nan")
        )
        return self.test_acc

    def plot_prediction_comparison(self, max_points: int = 500) -> Path:
        if self.y_pred is None or self.X_test is None:
            raise RuntimeError("Run test() before plotting predictions")
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        train_returns = (
            (self._train_close.shift(-self.horizon) - self._train_close).abs()
            / np.maximum(self._train_close, 1e-10)
        ).dropna()
        typical_abs_return = float(train_returns.median()) if not train_returns.empty else 0.005

        test_times = self.X_test.index
        current_close = self._test_close.reindex(test_times).to_numpy(dtype=float)
        actual_next_close = self._test_close.shift(-self.horizon).reindex(test_times).to_numpy(dtype=float)
        predicted_direction = np.where(np.asarray(self.y_pred) == 1, 1.0, -1.0)
        predicted_next_close = current_close * (1.0 + predicted_direction * typical_abs_return)

        valid = np.isfinite(actual_next_close) & np.isfinite(predicted_next_close)
        times_plot = test_times[valid]
        actual_plot = actual_next_close[valid]
        pred_plot = predicted_next_close[valid]

        first = max(0, len(times_plot) - int(max_points))
        plot_path = self.results_path.with_name(
            f"{self.results_path.stem}_h{self.horizon}_{self.target_mode}_prediction_comparison.png"
        )
        plot_path.parent.mkdir(parents=True, exist_ok=True)

        figure = Figure(figsize=(14, 7))
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        axis.plot(times_plot[first:], actual_plot[first:], label=f"Actual close[t+{self.horizon}]", linewidth=1.8)
        axis.plot(
            times_plot[first:],
            pred_plot[first:],
            label="Direction-implied close",
            linewidth=1.3,
            alpha=0.85,
        )
        axis.set_title(
            f"{self.symbol} {self.timeframe} (Logistic Regression): actual vs predicted direction-implied price"
        )
        axis.set_xlabel("Candle time (UTC)")
        axis.set_ylabel("Price")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.autofmt_xdate()
        figure.tight_layout()
        figure.savefig(plot_path, dpi=160, bbox_inches="tight")
        return plot_path

    def evaluate(self):
        if self.test_acc is None:
            raise RuntimeError("Run test() before evaluate()")
        plot_path = self.plot_prediction_comparison()
        print(f"Prediction comparison plot saved: {plot_path}")
        labels = sorted(set(self.y_test))
        result = {
            "symbol": self.symbol,
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": "StationaryDirectionalLogisticRegression",
            "timeframe": self.timeframe,
            "target_mode": self.target_mode,
            "horizon": self.horizon,
            "params": {
                "C": self.best_C,
                "l1_ratio": self.best_l1_ratio,
                "max_iter": self.max_iter,
                "random_state": self.random_state,
            },
            "decision_threshold": self.decision_threshold,
            "validation_acc": self.validation_acc,
            "validation_balanced_acc": self.validation_balanced_acc,
            "features": self.features,
            "n_features": len(self.features),
            "n_nonzero_features": self.n_nonzero_features,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "train_time": self.train_time,
            "training_acc": self.training_acc,
            "test_acc": self.test_acc,
            "test_balanced_acc": self.test_balanced_acc,
            "test_roc_auc": self.test_roc_auc,
            "prediction_plot": str(plot_path),
            "labels": [str(label) for label in labels],
            "confusion_matrix": confusion_matrix(self.y_test, self.y_pred, labels=labels).tolist(),
            "classification_report": classification_report(
                self.y_test, self.y_pred, labels=labels, output_dict=True, zero_division=0
            ),
        }
        results = []
        if self.results_path.exists():
            with self.results_path.open() as f:
                results = json.load(f)
        results.append(result)
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with self.results_path.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"result appended to {self.results_path} ({len(results)} runs in total) for symbol {self.symbol}")
        return result
