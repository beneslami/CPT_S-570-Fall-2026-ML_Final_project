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
from xgboost import XGBClassifier, XGBRegressor

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ML_interface import Model

PROJECT_DIR = Path(__file__).resolve().parents[2]
HORIZON = 1
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
        score = (
            float(log_loss(y_validation, raw_validation, labels=[0, 1]))
            if len(y_validation)
            else None
        )
        return BracketProbabilityCalibrator("none"), [
            {
                "method": "none",
                "validation_log_loss": score,
                "validation_roc_auc": None,
                "eligible": True,
                "fallback_reason": reason,
            }
        ]
    fitted = []
    for method in methods:
        calibrator = BracketProbabilityCalibrator(method).fit(raw_fit, y_fit)
        if method != "none" and calibrator._model is None:
            fitted.append(
                (
                    calibrator,
                    {
                        "method": method,
                        "eligible": False,
                        "rejection_reason": "insufficient two-class calibration data",
                    },
                )
            )
            continue
        probability = calibrator.predict_proba(raw_validation)
        score = float(log_loss(y_validation, probability, labels=[0, 1]))
        auc = float(roc_auc_score(y_validation, probability))
        details = {
            "method": method,
            "validation_log_loss": score,
            "validation_roc_auc": auc,
            "eligible": True,
        }
        if method == "logistic":
            details["logistic_slope"] = float(calibrator._model.coef_[0, 0])
            if details["logistic_slope"] <= 0:
                details["eligible"] = False
                details["rejection_reason"] = "logistic calibration does not preserve ranking"
        fitted.append((calibrator, details))
    eligible = [item for item in fitted if item[1]["eligible"]]
    if not eligible:
        fallback = BracketProbabilityCalibrator("none")
        details = {
            "method": "none",
            "eligible": True,
            "validation_log_loss": float(log_loss(y_validation, raw_validation, labels=[0, 1])),
            "validation_roc_auc": float(roc_auc_score(y_validation, raw_validation)),
            "fallback_reason": "no requested calibration method preserved ranking and had sufficient data",
        }
        fitted.append((fallback, details))
        eligible = [(fallback, details)]
    selected = min(eligible, key=lambda item: item[1]["validation_log_loss"])[0]
    return selected, [item[1] for item in fitted]


DERIVATIVE_RAW_COLUMNS = (
    "count",
    "volume_delta",
    "taker_buy_ratio",
    "spot_taker_buy_ratio",
    "perp_spot_basis",
    "perp_to_spot_vol_ratio",
    "premium_close",
    "funding_rate",
    "hours_since_funding",
    "sum_open_interest",
    "sum_taker_long_short_vol_ratio",
    "sum_toptrader_long_short_ratio",
)


def load_data(path):
    """Load OHLCV (plus any optional derivatives/order-flow columns) with unique, chronological UTC timestamps."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    base_cols = ["time", "open", "high", "low", "close", "volume"]
    extra_cols = [c for c in DERIVATIVE_RAW_COLUMNS if c in header]
    data = pd.read_csv(path, usecols=base_cols + extra_cols)
    data["time"] = pd.to_datetime(data["time"], utc=True, errors="raise")
    if not data["time"].is_monotonic_increasing or data["time"].duplicated().any():
        raise ValueError(f"{path}: timestamps must be unique and chronological")
    data = data.set_index("time")
    rename_map = {c: c.capitalize() for c in ("open", "high", "low", "close", "volume") if c in data.columns}
    data = data.rename(columns=rename_map)
    return data[["Close", "High", "Low", "Open", "Volume"] + extra_cols]


def _normalize_ohlcv(frame):
    """Ensure a DatetimeIndex (UTC), chronological order, OHLCV columns, and any derivatives columns."""
    df = frame.copy()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="raise")
        df = df.set_index("time")
    rename_map = {c: c.capitalize() for c in ("open", "high", "low", "close", "volume") if c in df.columns}
    df = df.rename(columns=rename_map)
    df.index = pd.to_datetime(df.index, utc=True, errors="raise")
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError("XGBoost requires unique, chronological input timestamps")
    extra_cols = [c for c in DERIVATIVE_RAW_COLUMNS if c in df.columns]
    return df[["Close", "High", "Low", "Open", "Volume"] + extra_cols]


FVG_FEATURES = (
    "fvg_bull_pct",
    "fvg_bear_pct",
    "fvg_imbalance",
    "fvg_type",
)

ORDER_BLOCK_FEATURES = (
    "bull_ob_active",
    "bear_ob_active",
    "bull_ob_distance_pct",
    "bear_ob_distance_pct",
    "bull_ob_width_pct",
    "bear_ob_width_pct",
    "bull_ob_touch",
    "bear_ob_touch",
    "bull_ob_age",
    "bear_ob_age",
)

SPARSE_STRUCTURE_FEATURES = FVG_FEATURES + ORDER_BLOCK_FEATURES

MICROSTRUCTURE_LAG_COLS = (
    "ret_1",
    "body_pct",
    "close_pos",
    "wick_imbalance",
    "volume_delta_ratio",
    "rsi_14",
    "macd_hist_atr",
)
MICROSTRUCTURE_LAGS = (1, 2, 3, 6)


def _compute_signed_streak(values: np.ndarray, clip_val: float = 10.0) -> np.ndarray:
    """Strictly causal count of consecutive positive (>0) or negative (<0) bars."""
    n = len(values)
    streak = np.zeros(n, dtype=float)
    cur = 0.0
    for i in range(n):
        v = values[i]
        if not np.isfinite(v) or v == 0.0:
            cur = 0.0
        elif v > 0.0:
            cur = cur + 1.0 if cur > 0.0 else 1.0
        else:
            cur = cur - 1.0 if cur < 0.0 else -1.0
        streak[i] = cur
    return np.clip(streak, -clip_val, clip_val)


def compute_order_block_features(data, bull_gap, bear_gap):
    """Track the latest wick-bounded opposing candle confirmed by an FVG."""
    bull_gap = np.asarray(bull_gap, dtype=float)
    bear_gap = np.asarray(bear_gap, dtype=float)

    opening = data["Open"].to_numpy(dtype=float)
    close = data["Close"].to_numpy(dtype=float)
    high = data["High"].to_numpy(dtype=float)
    low = data["Low"].to_numpy(dtype=float)
    n = len(data)
    columns = ORDER_BLOCK_FEATURES
    values = np.zeros((n, len(columns)), dtype=float)
    last_down = last_up = None
    bull_zone = bear_zone = None
    for i in range(n):
        candidate = i - 2
        if candidate >= 0 and np.isfinite(high[candidate]) and np.isfinite(low[candidate]):
            if close[candidate] < opening[candidate]:
                last_down = candidate
            elif close[candidate] > opening[candidate]:
                last_up = candidate

        if bull_zone is not None and low[i] <= bull_zone[0]:
            bull_zone = None
        if bear_zone is not None and high[i] >= bear_zone[1]:
            bear_zone = None
        if bull_gap[i] > 0 and last_down is not None and low[i] > low[last_down]:
            bull_zone = (low[last_down], high[last_down], i)
        if bear_gap[i] > 0 and last_up is not None and high[i] < high[last_up]:
            bear_zone = (low[last_up], high[last_up], i)

        if not np.isfinite(close[i]) or close[i] <= 0:
            continue
        for zone, side in ((bull_zone, "bull"), (bear_zone, "bear")):
            if zone is None:
                continue
            bottom, top, confirmed = zone
            ix = 0 if side == "bull" else 1
            values[i, ix] = 1
            values[i, 2 + ix] = ((close[i] - top) if side == "bull" else (bottom - close[i])) / close[i]
            values[i, 4 + ix] = (top - bottom) / close[i]
            values[i, 6 + ix] = float(i > confirmed and low[i] <= top and high[i] >= bottom)
            values[i, 8 + ix] = np.log1p(i - confirmed)
    return pd.DataFrame(values, index=data.index, columns=columns)


def _compute_auxiliary_timeframe_features(
    data: pd.DataFrame,
    auxiliary_data: dict | None,
    timeframe: str | None,
) -> pd.DataFrame | None:
    """Compute strictly causal lower-timeframe intra-bar and higher-timeframe completed-bar features."""
    if not auxiliary_data or not timeframe or not isinstance(data.index, pd.DatetimeIndex):
        return None

    primary_td = pd.Timedelta(timeframe)
    eps = 1e-10
    parts = []

    for aux_tf, aux_raw in auxiliary_data.items():
        if aux_raw is None or len(aux_raw) == 0:
            continue
        aux_td = pd.Timedelta(aux_tf)
        if aux_td == primary_td:
            continue
        aux = _normalize_ohlcv(aux_raw)
        prefix = f"tf{aux_tf}"

        if aux_td < primary_td and (primary_td % aux_td == pd.Timedelta(0)):
            # Strictly causal intra-bar features: every sub-bar in [t, t + primary_td) closes at <= t + primary_td.
            g = aux.copy()
            g["_bucket"] = g.index.floor(timeframe)
            g["_ret"] = np.log(g["Close"] / (g["Close"].shift(1) + eps))
            g["_body"] = (g["Close"] - g["Open"]) / (g["Open"] + eps)
            bar_rng = (g["High"] - g["Low"]).replace(0.0, np.nan)
            g["_cp"] = ((g["Close"] - g["Low"]) / bar_rng).fillna(0.5).clip(0.0, 1.0)
            g["_clv"] = ((2.0 * g["Close"] - g["High"] - g["Low"]) / bar_rng).fillna(0.0).clip(-1.0, 1.0)

            grouped = g.groupby("_bucket", sort=True)
            last_sub = grouped.last()
            first_sub = grouped.first()
            vol_sum = grouped["Volume"].sum()

            sub_df = pd.DataFrame(index=last_sub.index)
            sub_df[f"{prefix}_last_ret"] = last_sub["_ret"]
            sub_df[f"{prefix}_last_body"] = last_sub["_body"]
            sub_df[f"{prefix}_last_cp"] = last_sub["_cp"]
            sub_df[f"{prefix}_last_clv"] = last_sub["_clv"]
            sub_df[f"{prefix}_late_minus_early"] = last_sub["_body"] - first_sub["_body"]
            sub_df[f"{prefix}_up_frac"] = (g["_body"] > 0.0).groupby(g["_bucket"]).mean()
            sub_df[f"{prefix}_last_vol_share"] = last_sub["Volume"] / (vol_sum + eps)
            sub_df[f"{prefix}_last2_ret"] = g.groupby("_bucket").tail(2).groupby("_bucket")["_ret"].sum()
            sub_df[f"{prefix}_rvol"] = grouped["_ret"].std().fillna(0.0)
            parts.append(sub_df.reindex(data.index))

        elif aux_td > primary_td:
            # Strictly causal higher-timeframe completed-candle features aligned by close_time <= primary_close_time.
            c = aux["Close"].astype(float)
            o = aux["Open"].astype(float)
            h = aux["High"].astype(float)
            l = aux["Low"].astype(float)
            rng = (h - l).replace(0.0, np.nan)
            prev_c = c.shift(1)
            tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
            atr = tr.rolling(14, min_periods=14).mean()
            delta = c.diff()
            gain = delta.clip(lower=0.0).rolling(14, min_periods=14).mean()
            loss = (-delta.clip(upper=0.0)).rolling(14, min_periods=14).mean()
            rsi = 100.0 - 100.0 / (1.0 + gain / (loss + eps))

            htf_df = pd.DataFrame(
                {
                    "_close_time": aux.index + aux_td,
                    f"{prefix}_ret_1": np.log(c / (prev_c + eps)),
                    f"{prefix}_ret_3": np.log(c / (c.shift(3) + eps)),
                    f"{prefix}_body_atr": (c - o) / (atr + eps),
                    f"{prefix}_close_pos": ((c - l) / rng).fillna(0.5).clip(0.0, 1.0),
                    f"{prefix}_rsi_centered": (rsi - 50.0) / 50.0,
                    f"{prefix}_atr_pct": atr / (c + eps),
                }
            ).sort_values("_close_time")

            left = pd.DataFrame({"_time": data.index, "_close_time": data.index + primary_td})
            merged = (
                pd.merge_asof(left, htf_df, on="_close_time", direction="backward", allow_exact_matches=True)
                .set_index("_time")
                .drop(columns=["_close_time"])
            )
            parts.append(merged)

    if not parts:
        return None
    return pd.concat(parts, axis=1)


def compute_features(
    data,
    include_fvg_ob: bool = True,
    auxiliary_data: dict | None = None,
    timeframe: str | None = None,
):
    """Compute strictly causal, stationary OHLCV, microstructure-lag, streak, HTF, and time features."""
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
    feats["dist_sma50"] = (close - sma_50) / (sma_50 + eps)
    feats["sma20_50_ratio"] = (sma_20 - sma_50) / (sma_50 + eps)
    feats["ema12_26_ratio"] = (ema_12 - ema_26) / (close + eps)

    # 3. ATR & volatility-adjusted returns
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean()
    feats["atr_pct"] = atr_14 / (close + eps)
    for k in (1, 3, 6, 12):
        feats[f"ret_{k}_atr"] = (close - close.shift(k)) / (atr_14 + eps)

    # 4. Realized volatility & regime ratio
    log_ret_1 = feats["ret_1"]
    rvol_6 = log_ret_1.rolling(6).std()
    rvol_24 = log_ret_1.rolling(24).std()
    feats["rvol_6"] = rvol_6
    feats["rvol_24"] = rvol_24
    feats["rvol_ratio_6_24"] = rvol_6 / (rvol_24 + eps)

    # 5. Bollinger Bands (20)
    bb_std = close.rolling(20).std()
    feats["bb_pos_20"] = (close - sma_20) / (2.0 * bb_std + eps)
    feats["bb_width_20"] = (4.0 * bb_std) / (sma_20 + eps)

    # 6. MACD normalized by ATR
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    feats["macd_atr"] = macd / (atr_14 + eps)
    feats["macd_hist_atr"] = (macd - macd_signal) / (atr_14 + eps)

    # 7. RSI & slope
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(14).mean()
    loss = (-delta.clip(upper=0.0)).rolling(14).mean()
    rsi = 100.0 - (100.0 / (1.0 + gain / (loss + eps)))
    feats["rsi_14"] = rsi
    feats["rsi_slope_3"] = rsi - rsi.shift(3)

    # 8. Multi-window high/low channel position (causal HTF structure proxy)
    for win in (12, 24, 72):
        roll_hi = high.rolling(win).max()
        roll_lo = low.rolling(win).min()
        feats[f"channel_pos_{win}"] = (close - roll_lo) / (roll_hi - roll_lo + eps)

    # 9. Candle microstructure (aligned with LSTM features)
    bar_range = (high - low).replace(0.0, np.nan)
    feats["body_pct"] = (close - open_px) / (open_px + eps)
    feats["range_pct"] = (high - low) / (open_px + eps)
    feats["upper_wick_ratio"] = ((high - np.maximum(open_px, close)) / bar_range).fillna(0.0)
    feats["lower_wick_ratio"] = ((np.minimum(open_px, close) - low) / bar_range).fillna(0.0)
    feats["wick_imbalance"] = feats["lower_wick_ratio"] - feats["upper_wick_ratio"]
    feats["close_pos"] = ((close - low) / bar_range).fillna(0.5)

    # 10. Volume features (including LSTM's volume_delta_ratio)
    vol_med_20 = volume.rolling(20).median()
    log_vol = np.log1p(volume.clip(lower=0.0))
    feats["rel_vol_20"] = (volume / (vol_med_20 + eps)).clip(0.0, 10.0)
    feats["vol_zscore_24"] = (log_vol - log_vol.rolling(24).mean()) / (log_vol.rolling(24).std() + eps)
    feats["vol_change_1"] = log_vol.diff(1)
    raw_clv = ((2.0 * close - high - low) / bar_range).fillna(0.0).clip(-1.0, 1.0)
    feats["volume_delta_ratio"] = np.where(volume > 0.0, raw_clv, 0.0)
    feats["signed_vol"] = np.sign(close - open_px) * feats["rel_vol_20"]
    feats["signed_vol_sum_6"] = feats["signed_vol"].rolling(6).sum()
    feats["signed_vol_sum_24"] = feats["signed_vol"].rolling(24).sum()

    # 11. Explicit sequential microstructure lags (t-1, t-2, t-3, t-6)
    seq_cols = {}
    for col in MICROSTRUCTURE_LAG_COLS:
        series = feats[col]
        for lag in MICROSTRUCTURE_LAGS:
            seq_cols[f"{col}_lag{lag}"] = series.shift(lag)

    # 12. Return autocorrelation, sign persistence & consecutive up-down bar streaks
    ret_lag1 = log_ret_1.shift(1)
    for win in (12, 24):
        ac = log_ret_1.rolling(win, min_periods=win).corr(ret_lag1)
        valid_window = log_ret_1.rolling(win, min_periods=win).count() >= win
        seq_cols[f"ret_autocorr_1_{win}"] = ac.where(~(valid_window & ac.isna()), 0.0).clip(-1.0, 1.0)

    sign_prod = np.sign(log_ret_1) * np.sign(ret_lag1)
    seq_cols["ret_sign_persist_6"] = sign_prod.rolling(6, min_periods=6).mean()
    seq_cols["ret_sign_persist_12"] = sign_prod.rolling(12, min_periods=12).mean()

    seq_cols["ret_streak"] = _compute_signed_streak(log_ret_1.to_numpy(dtype=float), clip_val=10.0)
    seq_cols["body_streak"] = _compute_signed_streak((close - open_px).to_numpy(dtype=float), clip_val=10.0)
    up_bar = (log_ret_1 > 0.0).astype(float)
    seq_cols["up_ratio_6"] = up_bar.rolling(6, min_periods=6).mean()
    seq_cols["up_ratio_12"] = up_bar.rolling(12, min_periods=12).mean()

    feats = pd.concat([feats, pd.DataFrame(seq_cols, index=data.index)], axis=1)

    # 13. Cyclical UTC session features & inter-bar gap / ATR-normalized candle shape / rolling HTF context
    extra_cols = {}
    if isinstance(data.index, pd.DatetimeIndex):
        hour = data.index.hour + data.index.minute / 60.0
        dow = data.index.dayofweek
        extra_cols["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        extra_cols["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        extra_cols["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
        extra_cols["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
        dt_hours = data.index.to_series().diff().dt.total_seconds().div(3600.0).fillna(1.0)
        extra_cols["log_dt_hours"] = np.log1p(dt_hours.clip(lower=0.0, upper=168.0))

    extra_cols["gap_pct"] = np.log(open_px / (prev_close + eps))
    extra_cols["gap_atr"] = (open_px - prev_close) / (atr_14 + eps)
    extra_cols["body_to_range"] = ((close - open_px) / bar_range).fillna(0.0).clip(-1.0, 1.0)
    extra_cols["body_atr"] = (close - open_px) / (atr_14 + eps)
    extra_cols["range_atr"] = (high - low) / (atr_14 + eps)
    extra_cols["upper_wick_atr"] = (high - np.maximum(open_px, close)) / (atr_14 + eps)
    extra_cols["lower_wick_atr"] = (np.minimum(open_px, close) - low) / (atr_14 + eps)

    cp_centered = feats["close_pos"] - 0.5
    log_rel_vol = np.log1p(feats["rel_vol_20"].clip(lower=0.0))
    extra_cols["cp_x_range_atr"] = cp_centered * extra_cols["range_atr"]
    extra_cols["cp_x_logvol"] = cp_centered * log_rel_vol
    extra_cols["ret1_atr_sq_signed"] = np.sign(feats["ret_1_atr"]) * np.minimum(feats["ret_1_atr"] ** 2, 25.0)
    extra_cols["body_atr_x_vol"] = extra_cols["body_atr"] * log_rel_vol

    gain_7 = delta.clip(lower=0.0).rolling(7).mean()
    loss_7 = (-delta.clip(upper=0.0)).rolling(7).mean()
    extra_cols["rsi_7"] = 100.0 - (100.0 / (1.0 + gain_7 / (loss_7 + eps)))

    ema_8 = close.ewm(span=8, adjust=False).mean()
    ema_21 = close.ewm(span=21, adjust=False).mean()
    extra_cols["ema8_21_atr"] = (ema_8 - ema_21) / (atr_14 + eps)
    extra_cols["ema8_slope_atr"] = (ema_8 - ema_8.shift(3)) / (atr_14 + eps)

    prev_open = open_px.shift(1)
    bull_eng = (prev_close < prev_open) & (close > open_px) & (open_px <= prev_close) & (close >= prev_open)
    bear_eng = (prev_close > prev_open) & (close < open_px) & (open_px >= prev_close) & (close <= prev_open)
    extra_cols["engulfing"] = np.where(bull_eng, 1.0, np.where(bear_eng, -1.0, 0.0))
    if not include_fvg_ob:
        bull_imp = prev_close > prev_open
        bear_imp = prev_close < prev_open
        b_gap = np.where(bull_imp, np.maximum(0.0, low - high.shift(2)), 0.0)
        s_gap = np.where(bear_imp, np.maximum(0.0, low.shift(2) - high), 0.0)
        extra_cols["fvg_imbalance"] = (b_gap - s_gap) / (close + eps)
        extra_cols["fvg_type"] = np.where(b_gap > 0.0, 1.0, np.where(s_gap > 0.0, -1.0, 0.0))

    for k in (4, 12, 24):
        h_o = open_px.shift(k - 1)
        h_h = high.rolling(k).max()
        h_l = low.rolling(k).min()
        h_rng = (h_h - h_l).replace(0.0, np.nan)
        extra_cols[f"htf{k}_body_to_range"] = ((close - h_o) / h_rng).fillna(0.0).clip(-1.0, 1.0)
        extra_cols[f"htf{k}_close_pos"] = ((close - h_l) / h_rng).fillna(0.5).clip(0.0, 1.0)
        h_uw = h_h - np.maximum(h_o, close)
        h_lw = np.minimum(h_o, close) - h_l
        extra_cols[f"htf{k}_wick_imb"] = ((h_lw - h_uw) / h_rng).fillna(0.0).clip(-1.0, 1.0)
        extra_cols[f"htf{k}_range_atr"] = (h_h - h_l) / (atr_14 * np.sqrt(k) + eps)

    feats = pd.concat([feats, pd.DataFrame(extra_cols, index=data.index)], axis=1)

    aux_feats = _compute_auxiliary_timeframe_features(data, auxiliary_data, timeframe)
    if aux_feats is not None:
        feats = pd.concat([feats, aux_feats], axis=1)

    # 13b. Strictly causal derivatives & exact order-flow features (when present in datapoints_new/)
    deriv_cols = {}
    if "taker_buy_ratio" in data.columns:
        tbr = pd.to_numeric(data["taker_buy_ratio"], errors="coerce") - 0.5
        deriv_cols["perp_tbr"] = tbr
        deriv_cols["perp_tbr_lag1"] = tbr.shift(1)
        deriv_cols["perp_tbr_ma6"] = tbr.rolling(6).mean()
        deriv_cols["perp_tbr_ma24"] = tbr.rolling(24).mean()
        if "spot_taker_buy_ratio" in data.columns:
            stbr = pd.to_numeric(data["spot_taker_buy_ratio"], errors="coerce").fillna(0.5) - 0.5
            deriv_cols["spot_tbr"] = stbr
            deriv_cols["spot_tbr_ma6"] = stbr.rolling(6).mean()
            diff_tbr = tbr - stbr
            deriv_cols["perp_minus_spot_tbr"] = diff_tbr
            deriv_cols["perp_minus_spot_tbr_ma6"] = diff_tbr.rolling(6).mean()
    if "premium_close" in data.columns:
        prem = pd.to_numeric(data["premium_close"], errors="coerce").ffill().fillna(0.0)
        deriv_cols["prem_z24"] = ((prem - prem.rolling(24).mean()) / (prem.rolling(24).std() + eps)).clip(-5.0, 5.0)
    if "funding_rate" in data.columns:
        fr = pd.to_numeric(data["funding_rate"], errors="coerce").fillna(0.0)
        deriv_cols["funding_rate"] = fr
        deriv_cols["funding_z72"] = ((fr - fr.rolling(72).mean()) / (fr.rolling(72).std() + eps)).clip(-5.0, 5.0)
    if "count" in data.columns:
        log_cnt = np.log1p(pd.to_numeric(data["count"], errors="coerce").clip(lower=0.0))
        deriv_cols["log_count_z24"] = ((log_cnt - log_cnt.rolling(24).mean()) / (log_cnt.rolling(24).std() + eps)).clip(-5.0, 5.0)
    for ls_col in ("sum_taker_long_short_vol_ratio", "sum_toptrader_long_short_ratio"):
        if ls_col in data.columns:
            s = np.log(pd.to_numeric(data[ls_col], errors="coerce").clip(lower=1e-4))
            deriv_cols[f"{ls_col}_z24"] = ((s - s.rolling(24).mean()) / (s.rolling(24).std() + eps)).fillna(0.0).clip(-5.0, 5.0)
    if deriv_cols:
        feats = pd.concat([feats, pd.DataFrame(deriv_cols, index=data.index)], axis=1)

    # 14. Optional sparse 3-candle FVG & Order-Block features (14 columns)
    if include_fvg_ob:
        bull_impulse = close.shift(1) > open_px.shift(1)
        bear_impulse = close.shift(1) < open_px.shift(1)
        raw_bull_gap = np.maximum(0.0, low - high.shift(2))
        raw_bear_gap = np.maximum(0.0, low.shift(2) - high)
        bull_gap = np.where(bull_impulse, raw_bull_gap, 0.0)
        bear_gap = np.where(bear_impulse, raw_bear_gap, 0.0)

        fvg_df = pd.DataFrame(
            {
                "fvg_bull_pct": bull_gap / (close + eps),
                "fvg_bear_pct": bear_gap / (close + eps),
                "fvg_imbalance": (bull_gap - bear_gap) / (close + eps),
                "fvg_type": np.where(bull_gap > 0.0, 1, np.where(bear_gap > 0.0, -1, 0)),
            },
            index=data.index,
        )
        ob_df = compute_order_block_features(data, bull_gap, bear_gap)
        feats = pd.concat([feats, fvg_df, ob_df], axis=1)

    return feats.replace([np.inf, -np.inf], np.nan)


def compute_directional_targets(
    df: pd.DataFrame,
    horizon: int = HORIZON,
    min_move_atr: float = 0.0,
) -> pd.Series:
    """Pure fixed-horizon directional target: 1 if Close[t + horizon] > Close[t], 0 if lower."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    close = df["Close"].astype(float)
    move = close.shift(-horizon) - close
    targets = pd.Series(np.nan, index=df.index, name="target", dtype=float)

    if min_move_atr > 0.0:
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=14).mean()
        threshold = float(min_move_atr) * atr
        targets.loc[move > threshold] = 1.0
        targets.loc[move < -threshold] = 0.0
    else:
        targets.loc[move > 0.0] = 1.0
        targets.loc[move < 0.0] = 0.0

    return targets


def compute_recency_weights(index: pd.Index, half_life_days: float = 1095.0) -> np.ndarray:
    """Strictly causal exponential time-decay weights within a training split."""
    if len(index) == 0:
        return np.array([], dtype=float)
    if isinstance(index, pd.DatetimeIndex) and half_life_days > 0:
        age_days = (index[-1] - index).total_seconds().to_numpy(dtype=float) / 86400.0
        age_days = np.maximum(age_days, 0.0)
        weights = np.power(0.5, age_days / float(half_life_days))
        return weights / max(float(np.mean(weights)), 1e-10)
    return np.ones(len(index), dtype=float)


def compute_direction_sample_weights(df: pd.DataFrame, horizon: int = HORIZON) -> pd.Series:
    """Volatility-normalized move magnitude weights to downweight flat-bar noise."""
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=14).mean()
    abs_move = (close.shift(-horizon) - close).abs()
    norm = abs_move / (atr * np.sqrt(max(1, int(horizon))) + 1e-10)
    weights = norm.clip(lower=0.25, upper=3.0).fillna(1.0)
    return weights.rename("sample_weight")


def compute_resolved_bracket_targets(
    df: pd.DataFrame,
    sl_atr: float = DEFAULT_SL_ATR,
    tp_atr: float = DEFAULT_TP_ATR,
    max_bars: int = DEFAULT_MAX_BARS,
) -> pd.Series:
    """Legacy resolved bracket target kept for optional bracket-mode comparisons."""
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


class XGBoostBracketClassifier(Model):
    """Directional (or optional bracket) XGBoost classifier focused on maximizing accuracy."""

    def __init__(
        self,
        symbol,
        timeframe,
        results_path=None,
        horizon: int = HORIZON,
        cv_splits: int = 5,
        confidence_threshold: float = 0.55,
        multi_timeframes=(),
        sl_atr: float = DEFAULT_SL_ATR,
        tp_atr: float = DEFAULT_TP_ATR,
        max_bars: int = DEFAULT_MAX_BARS,
        calibration_method: str = "auto",
        target_mode: str = "direction",
        min_move_atr: float = 0.0,
        use_sample_weights: bool = False,
        include_fvg_ob: bool = False,
        reg_profile: str = "strong",
        recency_half_life_days: float | None = 1095.0,
        n_bagging_seeds: int = 5,
        max_features: int | None = 55,
        random_state: int = 42,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.horizon = int(horizon)
        self.cv_splits = int(cv_splits)
        self.confidence_threshold = float(confidence_threshold)
        if target_mode not in ("direction", "bracket"):
            raise ValueError("target_mode must be 'direction' or 'bracket'")
        if reg_profile not in ("baseline", "strong"):
            raise ValueError("reg_profile must be 'baseline' or 'strong'")
        self.target_mode = target_mode
        self.min_move_atr = float(min_move_atr)
        self.use_sample_weights = bool(use_sample_weights)
        self.include_fvg_ob = bool(include_fvg_ob)
        self.reg_profile = reg_profile
        self.recency_half_life_days = float(recency_half_life_days) if recency_half_life_days is not None else None
        self.n_bagging_seeds = max(1, int(n_bagging_seeds))
        self.max_features = int(max_features) if max_features is not None else None
        self.selected_features_ = None
        if multi_timeframes:
            raise ValueError("XGBoost multi-timeframe features are not implemented")
        if max_bars < 1 or sl_atr <= 0 or tp_atr <= 0 or cv_splits < 2 or self.horizon < 1:
            raise ValueError("horizon, max_bars, SL and TP must be positive and cv_splits >= 2")
        if not 0.5 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.5 and 1.0")
        self.multi_timeframes = ()
        self.sl_atr = float(sl_atr)
        self.tp_atr = float(tp_atr)
        self.max_bars = int(max_bars)
        self.purge_gap = self.horizon if self.target_mode == "direction" else self.max_bars
        if calibration_method not in ("auto", "none", "logistic", "isotonic"):
            raise ValueError("calibration_method must be 'auto', 'none', 'logistic' or 'isotonic'")
        self.calibration_method_requested = calibration_method
        self.calibration_method = None
        self.random_state = int(random_state)
        self.results_path = (
            Path(results_path)
            if results_path
            else PROJECT_DIR / "results" / f"xgboost_{symbol}_{timeframe}.json"
        )

        self.model = None
        self.models = []
        self.search = None
        self.best_params_ = {}
        self.calibrator = None
        self.decision_threshold = 0.50
        self.validation_accuracy = None
        self.validation_balanced_accuracy = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.w_train = None
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
        self._auxiliary_data = None

    def _compute_targets(self, df: pd.DataFrame) -> pd.Series:
        if self.target_mode == "direction":
            return compute_directional_targets(df, horizon=self.horizon, min_move_atr=self.min_move_atr)
        return compute_resolved_bracket_targets(
            df, sl_atr=self.sl_atr, tp_atr=self.tp_atr, max_bars=self.max_bars
        )

    def feature_engineering(self, train_data, test_data, auxiliary_data=None, order_block_history=None, **kwargs):
        train_df = _normalize_ohlcv(train_data)
        test_df = _normalize_ohlcv(test_data)
        self._auxiliary_data = auxiliary_data

        if train_df.equals(test_df):
            split_at = int(len(train_df) * 0.8)
            train_df, test_df = train_df.iloc[:split_at], train_df.iloc[split_at:]
        if train_df.empty or test_df.empty or train_df.index[-1] >= test_df.index[0]:
            raise ValueError("Training candles must end before test candles begin")
        self._walk_forward_candles = pd.concat([train_df, test_df])
        self._train_close = train_df["Close"].astype(float).copy()
        self._test_close = test_df["Close"].astype(float).copy()

        gap = self.purge_gap
        n = len(train_df)
        validation_start = int(n * 0.80)
        selection_start = validation_start + (n - validation_start) // 2
        threshold_start = selection_start + (n - selection_start) // 2
        if (
            validation_start <= gap
            or selection_start - validation_start <= gap
            or threshold_start - selection_start <= gap
        ):
            raise ValueError("Not enough training candles for purged calibration and threshold periods")

        def _make_split(df, purge_tail: bool = True, causal_features: pd.DataFrame | None = None):
            feats = (
                causal_features.loc[df.index].copy()
                if causal_features is not None
                else compute_features(
                    df,
                    include_fvg_ob=self.include_fvg_ob,
                    auxiliary_data=self._auxiliary_data,
                    timeframe=self.timeframe,
                )
            )
            target = self._compute_targets(df)
            weights = (
                compute_direction_sample_weights(df, horizon=self.horizon)
                if (self.target_mode == "direction" and self.use_sample_weights)
                else pd.Series(1.0, index=df.index, dtype=float)
            )
            curr_close = df["Close"].astype(float)
            valid = feats.notna().all(axis=1) & target.notna() & curr_close.notna()
            if purge_tail and gap > 0 and len(valid) > gap:
                valid.iloc[-gap:] = False
            return (
                feats.loc[valid].copy(),
                target.loc[valid].astype(int).copy(),
                weights.loc[valid].astype(float).copy(),
                curr_close.loc[valid].copy(),
            )

        all_X, all_y, all_w, all_close = _make_split(train_df, purge_tail=True)
        positions = pd.Series(np.arange(n), index=train_df.index).reindex(all_X.index)
        fit_mask = positions < validation_start - gap
        calibration_mask = (positions >= validation_start) & (positions < selection_start - gap)
        method_mask = (positions >= selection_start) & (positions < threshold_start - gap)
        threshold_mask = positions >= threshold_start

        self.X_train, self.y_train = all_X.loc[fit_mask], all_y.loc[fit_mask]
        self.w_train, self.curr_train = all_w.loc[fit_mask], all_close.loc[fit_mask]
        self.X_calibration, self.y_calibration = all_X.loc[calibration_mask], all_y.loc[calibration_mask]
        self.X_calibration_validation = all_X.loc[method_mask]
        self.y_calibration_validation = all_y.loc[method_mask]
        self.X_validation, self.y_validation = all_X.loc[threshold_mask], all_y.loc[threshold_mask]
        self.train_positions = positions.loc[fit_mask].to_numpy(dtype=int)

        history = (
            _normalize_ohlcv(order_block_history)
            if order_block_history is not None
            else pd.concat([train_df, test_df])
        )
        if not test_df.index.isin(history.index).all():
            raise ValueError("Feature history must cover all test candles")
        history_up_to_test = history.loc[: test_df.index[-1]]
        history_features = compute_features(
            history_up_to_test,
            include_fvg_ob=self.include_fvg_ob,
            auxiliary_data=self._auxiliary_data,
            timeframe=self.timeframe,
        )
        self.X_test, self.y_test, _, self.curr_test = _make_split(
            test_df, purge_tail=True, causal_features=history_features
        )
        self._test_ohlcv = test_df.copy()
        self._history_ohlcv = history_up_to_test.copy()
        self.test_trend_pred = (
            (history_up_to_test["Close"] > history_up_to_test["Close"].shift(1))
            .astype(int)
            .reindex(self.X_test.index)
        )

        for name, features, labels in (
            ("training", self.X_train, self.y_train),
            ("threshold validation", self.X_validation, self.y_validation),
            ("test", self.X_test, self.y_test),
        ):
            if features.empty or labels.nunique() < 2:
                raise ValueError(f"Not enough samples of both classes in {name} period")

        self.sample_summary = {
            "target_mode": self.target_mode,
            "horizon": self.horizon,
            "min_move_atr": self.min_move_atr,
            "include_fvg_ob": self.include_fvg_ob,
            "reg_profile": self.reg_profile,
            "n_features": int(self.X_train.shape[1]),
            "supplied_train_candles": len(train_df),
            "supplied_test_candles": len(test_df),
            "fit_samples": len(self.X_train),
            "calibration_samples": len(self.X_calibration),
            "calibration_selection_samples": len(self.X_calibration_validation),
            "threshold_samples": len(self.X_validation),
            "test_samples": len(self.X_test),
            "test_resolved_fraction_of_candles": len(self.X_test) / len(test_df),
            "train_classes": {str(k): int(v) for k, v in self.y_train.value_counts().sort_index().items()},
            "test_classes": {str(k): int(v) for k, v in self.y_test.value_counts().sort_index().items()},
            "fit_period": [str(self.X_train.index[0]), str(self.X_train.index[-1])],
            "threshold_period": [str(self.X_validation.index[0]), str(self.X_validation.index[-1])],
            "test_period": [str(self.X_test.index[0]), str(self.X_test.index[-1])],
        }
        print(
            f"XGBoost ({self.target_mode}, H={self.horizon}, fvg_ob={self.include_fvg_ob}, "
            f"reg={self.reg_profile}): fit={len(self.X_train)}, test={len(self.X_test)}/{len(test_df)} candles "
            f"({100 * self.sample_summary['test_resolved_fraction_of_candles']:.1f}%), "
            f"features={self.X_train.shape[1]}"
        )

    def train(self):
        if self.X_train is None:
            raise RuntimeError("Run feature_engineering() before train()")

        start = time.perf_counter()

        # Purged internal 85/15 split inside fit for hyperparameter selection + early stopping
        n_fit = len(self.X_train)
        es_start = int(n_fit * 0.85)
        inner_mask = self.train_positions < (self.train_positions[es_start] - self.purge_gap)
        es_mask = np.arange(n_fit) >= es_start

        X_inner = self.X_train.iloc[inner_mask]
        y_inner = self.y_train.iloc[inner_mask]
        if self.use_sample_weights:
            w_inner = self.w_train.iloc[inner_mask].to_numpy(dtype=float)
            w_full = self.w_train.to_numpy(dtype=float)
        elif self.recency_half_life_days is not None and self.recency_half_life_days > 0:
            w_inner = compute_recency_weights(X_inner.index, self.recency_half_life_days)
            w_full = compute_recency_weights(self.X_train.index, self.recency_half_life_days)
        else:
            w_inner = np.ones(len(X_inner), dtype=float)
            w_full = np.ones(len(self.X_train), dtype=float)
        X_es = self.X_train.iloc[es_mask]
        y_es = self.y_train.iloc[es_mask]

        # Strictly causal gain-based feature pruning on X_inner only (reduces noise & speeds up grid search ~2x)
        if (
            self.max_features is not None
            and self.X_train.shape[1] > self.max_features
            and len(X_inner) >= 100
            and y_inner.nunique() == 2
        ):
            selector = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.02,
                min_child_weight=60,
                subsample=0.75,
                colsample_bytree=0.4,
                reg_lambda=25.0,
                reg_alpha=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=self.random_state,
                n_jobs=1,
            )
            selector.fit(X_inner, y_inner, sample_weight=w_inner)
            gain_scores = (
                pd.Series(selector.get_booster().get_score(importance_type="gain"))
                .reindex(self.X_train.columns)
                .fillna(0.0)
                .sort_values(ascending=False)
            )
            self.selected_features_ = gain_scores.index[: self.max_features].tolist()
            self.X_train = self.X_train.loc[:, self.selected_features_]
            self.X_calibration = self.X_calibration.loc[:, self.selected_features_]
            self.X_calibration_validation = self.X_calibration_validation.loc[:, self.selected_features_]
            self.X_validation = self.X_validation.loc[:, self.selected_features_]
            self.X_test = self.X_test.loc[:, self.selected_features_]
            X_inner = X_inner.loc[:, self.selected_features_]
            X_es = X_es.loc[:, self.selected_features_]
        else:
            self.selected_features_ = list(self.X_train.columns)

        if self.reg_profile == "strong":
            candidates = [
                {"max_depth": d, "min_child_weight": mcw, "colsample_bytree": cs, "reg_lambda": lam}
                for d in (2, 3, 4)
                for mcw in (30, 60, 100)
                for cs in (0.35, 0.5)
                for lam in (10.0, 25.0, 50.0)
            ]
            fixed_params = {
                "learning_rate": 0.02,
                "subsample": 0.75,
                "reg_alpha": 1.0,
            }
            fallback_cfg = {
                "max_depth": 3,
                "min_child_weight": 60,
                "colsample_bytree": 0.5,
                "reg_lambda": 25.0,
            }
        else:
            candidates = [
                {"max_depth": d, "min_child_weight": mcw, "colsample_bytree": cs, "reg_lambda": lam}
                for d in (3, 4, 5)
                for mcw in (10, 25)
                for cs in (0.6, 0.8)
                for lam in (5.0, 15.0)
            ]
            fixed_params = {
                "learning_rate": 0.02,
                "subsample": 0.8,
                "reg_alpha": 0.5,
            }
            fallback_cfg = {
                "max_depth": 3,
                "min_child_weight": 15,
                "colsample_bytree": 0.7,
                "reg_lambda": 10.0,
            }

        best_score = None
        best_cfg = None
        best_n_estimators = 150

        if len(X_inner) >= 100 and len(X_es) >= 50 and y_inner.nunique() == 2 and y_es.nunique() == 2:
            for cfg in candidates:
                clf = XGBClassifier(
                    n_estimators=600,
                    early_stopping_rounds=30,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=self.random_state,
                    n_jobs=1,
                    **fixed_params,
                    **cfg,
                )
                clf.fit(
                    X_inner,
                    y_inner,
                    sample_weight=w_inner,
                    eval_set=[(X_es, y_es)],
                    verbose=False,
                )
                es_prob = clf.predict_proba(X_es)[:, 1]
                ll = float(log_loss(y_es, es_prob, labels=[0, 1]))
                auc = float(roc_auc_score(y_es, es_prob))
                score = (ll, -auc)
                if best_score is None or score < best_score:
                    best_score = score
                    best_cfg = dict(cfg)
                    best_iter = getattr(clf, "best_iteration", None)
                    best_n_estimators = max(50, int(best_iter) + 1) if best_iter is not None else 200
        else:
            best_cfg = fallback_cfg
            best_n_estimators = 200

        self.best_params_ = {
            **best_cfg,
            "n_estimators": best_n_estimators,
            **fixed_params,
        }
        self.models = []
        for bag_idx in range(self.n_bagging_seeds):
            clf_bag = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=self.random_state + bag_idx * 101,
                n_jobs=1,
                **self.best_params_,
            )
            clf_bag.fit(
                self.X_train,
                self.y_train,
                sample_weight=w_full,
            )
            self.models.append(clf_bag)
        self.model = self.models[0]
        self.train_time = time.perf_counter() - start

        raw_calibration = (
            self._predict_raw_proba(self.X_calibration) if len(self.X_calibration) else np.array([])
        )
        raw_method_validation = (
            self._predict_raw_proba(self.X_calibration_validation)
            if len(self.X_calibration_validation)
            else np.array([])
        )
        methods = (
            ("none", "logistic")
            if self.calibration_method_requested == "auto"
            else (self.calibration_method_requested,)
        )
        self.calibrator, self.calibration_scores = select_probability_calibrator(
            raw_calibration,
            self.y_calibration.to_numpy(dtype=int),
            raw_method_validation,
            self.y_calibration_validation.to_numpy(dtype=int),
            methods,
        )
        self.calibration_method = self.calibrator.method

        validation_probability = self.predict_calibrated_proba(self.X_validation)
        scores = [
            (
                float(t),
                float(balanced_accuracy_score(self.y_validation, validation_probability >= t)),
                float(accuracy_score(self.y_validation, validation_probability >= t)),
            )
            for t in np.linspace(0.45, 0.55, 21)
        ]
        self.decision_threshold, self.validation_balanced_accuracy, self.validation_accuracy = max(
            scores, key=lambda item: (item[1], -abs(item[0] - 0.5))
        )
        self.threshold_scores = [
            {"threshold": t, "balanced_accuracy": bal, "accuracy": acc} for t, bal, acc in scores
        ]

        y_tr = self.y_train.to_numpy(dtype=int)
        raw_train_prob = self._predict_raw_proba(self.X_train)
        self.train_prob = self.calibrator.predict_proba(raw_train_prob)
        self.train_pred = (self.train_prob >= self.decision_threshold).astype(int)
        self.train_acc = float(accuracy_score(y_tr, self.train_pred))

        print(
            f"Best params: {self.best_params_} | bags={len(self.models)} | cal={self.calibration_method} | "
            f"threshold={self.decision_threshold:.2f} (val_acc={self.validation_accuracy * 100:.2f}%, "
            f"val_bal_acc={self.validation_balanced_accuracy * 100:.2f}%)"
            f"\nTrain Acc: {self.train_acc * 100:.2f}%"
        )
        return self.train_acc

    def _predict_raw_proba(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame) and self.selected_features_ is not None:
            X = X.loc[:, self.selected_features_]
        if not self.models:
            if self.model is None:
                raise RuntimeError("Run train() before predicting")
            return self.model.predict_proba(X)[:, 1]
        probs = [m.predict_proba(X)[:, 1] for m in self.models]
        return np.mean(probs, axis=0)

    def predict_calibrated_proba(self, X) -> np.ndarray:
        raw = self._predict_raw_proba(X)
        return self.calibrator.predict_proba(raw) if self.calibrator is not None else raw

    def test(self):
        if self.model is None:
            raise RuntimeError("Run train() before test()")

        raw_test_prob = self._predict_raw_proba(self.X_test)
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
            "bearish": int((self.test_pred == 0).sum()),
            "bullish": int((self.test_pred == 1).sum()),
        }
        self.test_probability_range = [float(np.min(self.test_prob)), float(np.max(self.test_prob))]

        predicted_class_confidence = np.where(self.test_pred == 1, self.test_prob, 1.0 - self.test_prob)
        conf_mask = predicted_class_confidence >= self.confidence_threshold
        self.confident_coverage = float(conf_mask.mean())
        majority_class = int(self.y_train.mean() > 0.5)
        self.majority_baseline_acc = float(accuracy_score(y_te, np.full(len(y_te), majority_class)))
        self.trend_baseline_acc = float(accuracy_score(y_te, self.test_trend_pred))
        print(
            f"Baselines: training-majority {self.majority_baseline_acc:.2%}, "
            f"recent candle direction {self.trend_baseline_acc:.2%}"
        )
        print(
            f"Test predictions: {self.test_prediction_counts['bullish']} bullish, "
            f"{self.test_prediction_counts['bearish']} bearish; "
            f"balanced acc: {self.test_balanced_acc:.2%}; raw ROC-AUC: {self.raw_test_roc_auc:.4f}"
        )
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

    def predict_next_candle(self, ohlcv_data) -> pd.Series:
        """Return directional price targets scaled by calibrated directional probability."""
        if self.model is None or self.X_train is None:
            raise RuntimeError("Run train() before predicting")

        df = _normalize_ohlcv(ohlcv_data)
        feats = compute_features(
            df,
            include_fvg_ob=self.include_fvg_ob,
            auxiliary_data=self._auxiliary_data,
            timeframe=self.timeframe,
        )[self.X_train.columns]
        close = df["Close"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        prev_close = close.shift(1).fillna(close)
        tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        valid = feats.notna().all(axis=1) & close.notna() & atr.notna()
        pred_price = pd.Series(np.nan, index=df.index, name="predicted_price", dtype=float)
        if valid.any():
            cal_p = self.predict_calibrated_proba(feats.loc[valid])
            signed_edge = np.clip((cal_p - self.decision_threshold) * 4.0, -1.0, 1.0)
            step_pct = np.clip(
                signed_edge
                * (atr.loc[valid].to_numpy(dtype=float) / np.maximum(close.loc[valid].to_numpy(dtype=float), 1e-10)),
                -0.05,
                0.05,
            )
            pred_price.loc[valid] = close.loc[valid].to_numpy(dtype=float) * (1.0 + step_pct)
        return pred_price

    def plot_prediction_comparison(self, max_points: int = 500) -> Path:
        if self.test_pred is None or self.X_test is None:
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
        predicted_direction = np.where(np.asarray(self.test_pred) == 1, 1.0, -1.0)
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
        axis.set_title(f"{self.symbol} {self.timeframe} (XGBoost): actual vs predicted direction-implied price")
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

        record = {
            "symbol": self.symbol,
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": "XGBoostDirectionalClassifier" if self.target_mode == "direction" else "XGBoostBracketClassifier",
            "timeframe": self.timeframe,
            "target_mode": self.target_mode,
            "horizon": self.horizon,
            "min_move_atr": self.min_move_atr,
            "include_fvg_ob": self.include_fvg_ob,
            "reg_profile": self.reg_profile,
            "n_features": int(self.X_train.shape[1]),
            "train_time": float(self.train_time),
            "params": self.best_params_,
            "decision_threshold": self.decision_threshold,
            "confidence_threshold": self.confidence_threshold,
            "calibration_method": self.calibration_method,
            "calibration_scores": self.calibration_scores,
            "validation_accuracy": self.validation_accuracy,
            "validation_balanced_accuracy": self.validation_balanced_accuracy,
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
            "prediction_plot": str(plot_path),
        }
        results = []
        if self.results_path.exists():
            with self.results_path.open() as f:
                results = json.load(f)
            if not isinstance(results, list):
                raise ValueError(f"Expected a list of runs in {self.results_path}")
        results.append(record)
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with self.results_path.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"result appended to {self.results_path} ({len(results)} runs in total) for symbol {self.symbol}")
        return record


class XGBoostModel(Model):
    """Forecast the next higher-timeframe close via its log return."""

    def __init__(self, symbol, timeframe, results_path=None, max_bars=DEFAULT_MAX_BARS):
        self.symbol, self.timeframe = symbol, timeframe
        self.results_path = (
            Path(results_path)
            if results_path
            else PROJECT_DIR / "results" / f"xgboost_price_{symbol}_{timeframe}.json"
        )
        self.max_bars = max_bars
        self.model = self.search = None
        self.X_train = self.X_test = None
        self.metrics = None

    def feature_engineering(self, train_data, test_data, **kwargs):
        train = _normalize_ohlcv(train_data)
        test = _normalize_ohlcv(test_data)
        if train.empty or test.empty or train.index[-1] >= test.index[0]:
            raise ValueError("Training candles must end before test candles begin")
        data = pd.concat([train, test])
        features = compute_features(data)
        self.feature_names = features.columns.tolist()
        interval = pd.Timedelta(self.timeframe)

        def partition(frame):
            close = frame["Close"].astype(float)
            next_close = close.shift(-1)
            next_time = pd.Series(frame.index, index=frame.index).shift(-1)
            adjacent = next_time == frame.index + interval
            y = np.log(next_close / close)
            valid = features.loc[frame.index].notna().all(axis=1) & y.notna() & np.isfinite(y) & adjacent
            return (
                features.loc[frame.index].loc[valid].copy(),
                y.loc[valid].copy(),
                close.loc[valid].copy(),
                next_close.loc[valid].copy(),
            )

        self.X_train, self.y_train, self.train_close, self.train_next = partition(train)
        self.X_test, self.y_test, self.test_close, self.test_next = partition(test)
        if len(self.X_train) < 100 or self.X_test.empty:
            raise ValueError("Not enough consecutive candles with valid features to train and test")
        print(
            f"Price prediction: {len(self.X_train)} training and {len(self.X_test)} held-out samples; "
            f"next {self.timeframe} close"
        )

    def train(self):
        if self.X_train is None:
            raise RuntimeError("Run feature_engineering() before train()")
        self.search = GridSearchCV(
            XGBRegressor(objective="reg:squarederror", n_jobs=1, random_state=42),
            {
                "n_estimators": [200],
                "max_depth": [2, 3],
                "learning_rate": [0.02, 0.04],
                "min_child_weight": [15],
                "reg_lambda": [10.0],
                "subsample": [0.8],
                "colsample_bytree": [0.8],
            },
            cv=TimeSeriesSplit(n_splits=5, gap=1),
            scoring="neg_mean_absolute_error",
            n_jobs=1,
        )
        self.search.fit(self.X_train, self.y_train)
        self.model = self.search.best_estimator_
        print(
            f"Best price parameters: {self.search.best_params_}; "
            f"chronological validation return MAE: {-self.search.best_score_:.6f}"
        )
        return self.search.best_params_

    def test(self):
        if self.model is None:
            raise RuntimeError("Run train() before test()")
        predicted_return = self.model.predict(self.X_test)
        predicted_price = self.test_close.to_numpy() * np.exp(predicted_return)
        actual = self.test_next.to_numpy()
        current = self.test_close.to_numpy()
        self.metrics = {
            "test_samples": len(actual),
            "test_period": [str(self.X_test.index[0]), str(self.X_test.index[-1])],
            "price_mae": float(np.mean(np.abs(predicted_price - actual))),
            "last_close_baseline_mae": float(np.mean(np.abs(current - actual))),
            "direction_accuracy": float(np.mean(np.sign(predicted_price - current) == np.sign(actual - current))),
            "predicted_up": int(np.sum(predicted_price > current)),
            "predicted_down": int(np.sum(predicted_price < current)),
        }
        print(
            f"Held-out next-close MAE: {self.metrics['price_mae']:.4f}; "
            f"last-close baseline MAE: {self.metrics['last_close_baseline_mae']:.4f}; "
            f"direction accuracy: {self.metrics['direction_accuracy']:.2%}"
        )
        return self.metrics

    def evaluate(self):
        if self.metrics is None:
            raise RuntimeError("Run test() before evaluate()")
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "model": "XGBoostNextCloseRegressor",
            "target": "next consecutive candle close",
            "metrics": self.metrics,
            "parameters": self.search.best_params_,
        }

    def predict_next_candle(self, ohlcv_data):
        if self.model is None:
            raise RuntimeError("Run train() before predicting")
        data = _normalize_ohlcv(ohlcv_data)
        features = compute_features(data).loc[:, self.feature_names]
        valid = features.notna().all(axis=1) & data["Close"].gt(0)
        prediction = pd.Series(np.nan, index=data.index, name="predicted_price", dtype=float)
        if valid.any():
            predicted_return = self.model.predict(features.loc[valid])
            prediction.loc[valid] = data.loc[valid, "Close"] * np.exp(predicted_return)
        return prediction
