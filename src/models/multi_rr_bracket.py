from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score
import xgboost as xgb

try:
    from ML_interface import Model
    from models.XGBoost import _normalize_ohlcv, compute_features
    from strategy import compute_trade_ev, compute_trade_ev_3outcome
except ImportError:
    from src.ML_interface import Model
    from src.models.XGBoost import _normalize_ohlcv, compute_features
    from src.strategy import compute_trade_ev, compute_trade_ev_3outcome


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RR_RATIOS = (2.0, 2.5, 3.0, 3.5, 4.0)
VALID_ENTRY_SESSIONS = ("all", "weekdays", "asia", "london", "ny", "london_ny", "ny_rth")


def _compute_entry_mask(index: pd.DatetimeIndex, entry_session: str) -> np.ndarray:
    """Compute a boolean entry-and-training mask on continuous UTC candles without deleting rows.

    Keeping all continuous candles prevents artificial session-boundary gap fills and indicator
    corruption while restricting model training, calibration, and new entries to the target session.
    """
    n = len(index)
    if entry_session == "all":
        return np.ones(n, dtype=bool)
    ts_utc = pd.DatetimeIndex(index)
    if ts_utc.tz is None:
        ts_utc = ts_utc.tz_localize("UTC")
    else:
        ts_utc = ts_utc.tz_convert("UTC")

    if entry_session == "ny_rth":
        ts_ny = ts_utc.tz_convert("America/New_York")
        mins = ts_ny.hour * 60 + ts_ny.minute
        return np.asarray((ts_ny.day_of_week < 5) & (mins >= 9 * 60 + 30) & (mins < 16 * 60), dtype=bool)

    is_wd = ts_utc.day_of_week < 5
    h = ts_utc.hour
    if entry_session == "weekdays":
        return np.asarray(is_wd, dtype=bool)
    if entry_session == "asia":
        return np.asarray(is_wd & (h >= 0) & (h < 9), dtype=bool)
    if entry_session == "london":
        return np.asarray(is_wd & (h >= 7) & (h < 16), dtype=bool)
    if entry_session == "ny":
        return np.asarray(is_wd & (h >= 12) & (h < 21), dtype=bool)
    if entry_session == "london_ny":
        return np.asarray(is_wd & (h >= 7) & (h < 21), dtype=bool)
    raise ValueError(f"Unsupported entry_session={entry_session!r}; expected one of {VALID_ENTRY_SESSIONS}")


def _get_git_revision() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_DIR,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _simulate_bracket_3class(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    sl_pct: np.ndarray,
    side: int,
    rr_ratio: float,
    max_bars: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized causal 3-outcome bracket labeler matching the exact backtest simulator.

    Classes:
      0 = SL hit first (or conservative gap/intrabar tie)
      1 = TP hit strictly before SL within max_bars
      2 = TIMEOUT (neither barrier touched within max_bars; closed at closes[i + max_bars])

    Also returns `timeout_gross_r` (realized gross R on timeout bars, NaN otherwise).
    """
    n = len(opens)
    valid_n = max(0, n - max_bars - 1)
    outcomes = np.full(n, np.nan, dtype=float)
    timeout_r = np.full(n, np.nan, dtype=float)
    if valid_n <= 0:
        return outcomes, timeout_r

    entry = opens[1 : valid_n + 1]
    sl_d = sl_pct[:valid_n]
    ok = np.isfinite(entry) & (entry > 0.0) & np.isfinite(sl_d) & (sl_d > 0.0)

    stop = entry * (1.0 - side * sl_d)
    target = entry * (1.0 + side * rr_ratio * sl_d)
    ok &= (stop > 0.0) & (target > 0.0)

    status = np.full(valid_n, -1, dtype=np.int8)
    status[~ok] = -2

    for k in range(1, max_bars + 1):
        still_open = status == -1
        if not np.any(still_open):
            break
        o_k = opens[k : valid_n + k]
        h_k = highs[k : valid_n + k]
        l_k = lows[k : valid_n + k]

        if side == 1:
            gap_sl = still_open & (o_k <= stop)
            status[gap_sl] = 0
            still_open = status == -1

            gap_tp = still_open & (o_k >= target)
            status[gap_tp] = 1
            still_open = status == -1

            hit_sl = still_open & (l_k <= stop)
            status[hit_sl] = 0
            still_open = status == -1

            hit_tp = still_open & (h_k >= target)
            status[hit_tp] = 1
        else:
            gap_sl = still_open & (o_k >= stop)
            status[gap_sl] = 0
            still_open = status == -1

            gap_tp = still_open & (o_k <= target)
            status[gap_tp] = 1
            still_open = status == -1

            hit_sl = still_open & (h_k >= stop)
            status[hit_sl] = 0
            still_open = status == -1

            hit_tp = still_open & (l_k <= target)
            status[hit_tp] = 1

    is_to = status == -1
    status[is_to] = 2

    out = np.where(status == -2, np.nan, status.astype(float))
    outcomes[:valid_n] = out

    close_to = closes[max_bars : valid_n + max_bars]
    risk = np.maximum(entry * sl_d, 1e-10)
    gross_to = side * (close_to - entry) / risk
    gross_to = np.clip(gross_to, -1.0, rr_ratio)
    timeout_r[:valid_n] = np.where(is_to & np.isfinite(gross_to), gross_to, np.nan)
    return outcomes, timeout_r


def _compute_mae_targets(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    horizon_bars: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal Maximum Adverse Excursion over the next `horizon_bars` candles from next-bar open."""
    n = len(opens)
    valid_n = max(0, n - horizon_bars - 1)
    mae_long = np.full(n, np.nan, dtype=float)
    mae_short = np.full(n, np.nan, dtype=float)
    if valid_n <= 0:
        return mae_long, mae_short

    entry = opens[1 : valid_n + 1]
    min_low = np.full(valid_n, np.inf, dtype=float)
    max_high = np.full(valid_n, -np.inf, dtype=float)
    for k in range(1, horizon_bars + 1):
        min_low = np.minimum(min_low, lows[k : valid_n + k])
        max_high = np.maximum(max_high, highs[k : valid_n + k])

    ok = np.isfinite(entry) & (entry > 0.0) & np.isfinite(min_low) & np.isfinite(max_high)
    mae_long[:valid_n] = np.where(ok, np.maximum(0.0, (entry - min_low) / np.maximum(entry, 1e-10)), np.nan)
    mae_short[:valid_n] = np.where(ok, np.maximum(0.0, (max_high - entry) / np.maximum(entry, 1e-10)), np.nan)
    return mae_long, mae_short


def _logit(p: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    pc = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(pc / (1.0 - pc))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))


def _causal_rolling_prior_shift_binary(
    raw_cal_probs: np.ndarray,
    realized_y: np.ndarray,
    max_bars: int,
    window_bars: int = 5760,
    min_periods: int = 400,
    mask_active: np.ndarray | None = None,
    seed_probs: np.ndarray | None = None,
    seed_y: np.ndarray | None = None,
    seed_active: np.ndarray | None = None,
) -> np.ndarray:
    """Strictly causal binary prior-shift logit recalibration over resolved (and optionally active) bars."""
    n = len(raw_cal_probs)
    act = np.ones(n, dtype=bool) if mask_active is None else np.asarray(mask_active, dtype=bool)
    if seed_probs is not None and seed_y is not None and len(seed_probs) > 0:
        tail = min(len(seed_probs), window_bars + max_bars + 1)
        p_all = np.concatenate([seed_probs[-tail:], raw_cal_probs])
        y_all = np.concatenate([seed_y[-tail:], realized_y])
        s_act = np.ones(tail, dtype=bool) if seed_active is None else np.asarray(seed_active[-tail:], dtype=bool)
        a_all = np.concatenate([s_act, act])
        offset = tail
    else:
        p_all = np.asarray(raw_cal_probs, dtype=float)
        y_all = np.asarray(realized_y, dtype=float)
        a_all = act
        offset = 0

    lag = max_bars + 1
    p_s = pd.Series(p_all, dtype=float).shift(lag)
    y_s = pd.Series(y_all, dtype=float).shift(lag)
    a_s = pd.Series(a_all, dtype=bool).shift(lag, fill_value=False)

    valid = p_s.notna() & y_s.notna() & a_s
    p_mean = p_s.where(valid).rolling(window_bars, min_periods=min_periods).mean().to_numpy(dtype=float)
    y_mean = y_s.where(valid).rolling(window_bars, min_periods=min_periods).mean().to_numpy(dtype=float)

    delta = _logit(np.clip(y_mean, 0.01, 0.99)) - _logit(np.clip(p_mean, 0.01, 0.99))
    delta = np.where(np.isfinite(delta), np.clip(delta, -1.5, 1.5), 0.0)
    adj_all = _sigmoid(_logit(p_all) + delta)
    return adj_all[offset : offset + n]


def _enforce_monotone_probs_3class(prob_tensor: np.ndarray) -> np.ndarray:
    """Enforce physically coherent probabilities along increasing R:R columns.

    Input shape: (N, n_rr, 3) with channels [p_sl, p_tp, p_to].
    For fixed SL and increasing RR: p_tp[:, j] <= p_tp[:, j-1].
    Any clipped excess in p_tp shifts into p_to, preserving simplex sum = 1.
    """
    out = np.asarray(prob_tensor, dtype=float).copy()
    out = np.clip(out, 1e-5, 1.0)
    out = out / np.maximum(out.sum(axis=-1, keepdims=True), 1e-12)
    for j in range(1, out.shape[1]):
        excess = np.maximum(0.0, out[:, j, 1] - out[:, j - 1, 1])
        out[:, j, 1] -= excess
        out[:, j, 2] += excess
    out = np.clip(out, 1e-5, 1.0)
    out = out / np.maximum(out.sum(axis=-1, keepdims=True), 1e-12)
    return out


class MultiRRBracketModel(Model):
    """V2.1 Leakage-Free Factorized 3-Outcome Multi-R:R EV-Maximizing Bracket Model.

    Upgrades directly addressing empirical & theoretical critique:
      1. Factorized Exact 3-Outcome Payoff Model (Zero Timeout Mismatch & Zero Volatility Pollution):
         Factorizes P(SL, TP, TIMEOUT | x_t) by exact probability chain rule:
           - Timeout head: p_to(x_t) = P(TIMEOUT | x_t)
           - Directional head: q_TP(x_t) = P(TP | TP or SL, x_t) — trained & early-stopped PURELY
             on directional barrier hits so volatility/timeout predictability never causes overfitting!
           - Conditional timeout payoff regressor: mu_to(x_t) = E[R_to | TIMEOUT, x_t] — prevents
             constant-timeout survivorship hallucination on flat/low-volatility bars!
         Reconstructs p_TP = (1 - p_to) * q_TP, p_SL = (1 - p_to) * (1 - q_TP), p_to = p_to,
         calibrated specifically on tradable (`Fee_R <= max_comm_r`) bars.
      2. Fresh Chronological Rolling-OOF Stage-1 Stops (Zero 4-Year Staleness):
         Trains 4 chronological rolling-2y Stage-1 ATR-MAE quantile folds inside training and
         refits a fresh trailing-2y Stage-1 stop model immediately before each test chunk.
         Also supports `sl_mode="fixed_atr"` for clean apples-to-apples baseline evaluation.
      3. Complete Diagnostics & Reproducibility:
         Selected-trade calibration, always-0 majority baseline, balanced accuracy, separate
         stand-down/commission/EV rejection counts, 30d bull/bear regime breakdown, git hash, config.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str = "15m",
        rr_ratios: tuple[float, ...] = DEFAULT_RR_RATIOS,
        max_bars: int = 32,
        ev_mode: str = "conservative",  # "conservative" (lower-bound P(TP before SL)) or "factorized_3class"
        sl_mode: str = "learned",  # "learned" or "fixed_atr"
        fixed_sl_atr: float = 1.5,
        sl_mae_bars: int = 8,
        sl_quantile: float = 0.70,
        sl_rolling_years: float = 2.0,
        sl_oof_folds: int = 4,
        min_sl_atr: float = 0.80,
        max_sl_atr: float = 3.50,
        min_sl_pct: float = 0.0005,
        max_sl_pct: float = 0.030,
        commission: float = 0.0001,
        risk_fraction: float = 0.01,
        max_leverage: float = 3.0,
        initial_cash: float = 10000.0,
        rolling_cal_bars: int = 5760,
        refit_days: int = 180,
        recency_half_life_days: float = 730.0,
        min_ev_floor: float = 0.12,
        max_comm_r: float = 0.05,
        stand_down_on_neg_val: bool = True,
        prob_shrinkage: float = 0.15,
        entry_session: str = "all",
        results_path: str | Path | None = None,
        random_state: int = 42,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.rr_ratios = tuple(sorted(float(r) for r in rr_ratios))
        self.max_bars = int(max_bars)
        if ev_mode not in ("conservative", "factorized_3class"):
            raise ValueError(f"Unsupported ev_mode={ev_mode!r}")
        self.ev_mode = ev_mode
        if sl_mode not in ("learned", "fixed_atr"):
            raise ValueError(f"Unsupported sl_mode={sl_mode!r}; expected 'learned' or 'fixed_atr'")
        self.sl_mode = sl_mode
        self.fixed_sl_atr = float(fixed_sl_atr)
        self.sl_mae_bars = int(sl_mae_bars)
        self.sl_quantile = float(sl_quantile)
        self.sl_rolling_years = float(sl_rolling_years)
        self.sl_oof_folds = int(sl_oof_folds)
        self.min_sl_atr = float(min_sl_atr)
        self.max_sl_atr = float(max_sl_atr)
        self.commission = float(commission)
        self.min_sl_pct = max(float(min_sl_pct), 5.0 * self.commission)
        self.max_sl_pct = float(max_sl_pct)
        self.risk_fraction = float(risk_fraction)
        self.max_leverage = float(max_leverage)
        self.initial_cash = float(initial_cash)
        self.rolling_cal_bars = int(rolling_cal_bars)
        self.refit_days = int(refit_days)
        self.recency_half_life_days = float(recency_half_life_days)
        self.min_ev_floor = float(min_ev_floor)
        self.max_comm_r = float(max_comm_r)
        self.stand_down_on_neg_val = bool(stand_down_on_neg_val)
        self.prob_shrinkage = float(np.clip(prob_shrinkage, 0.0, 0.5))
        if entry_session not in VALID_ENTRY_SESSIONS:
            raise ValueError(f"Unsupported entry_session={entry_session!r}; expected one of {VALID_ENTRY_SESSIONS}")
        self.entry_session = str(entry_session)
        self.random_state = int(random_state)
        self.refit_summary: list[dict] = []
        self.results_path = (
            Path(results_path)
            if results_path
            else PROJECT_DIR / "results" / f"multi_rr_bracket_v2_{symbol}_{timeframe}.json"
        )

        self._train_df: pd.DataFrame | None = None
        self._test_df: pd.DataFrame | None = None
        self._X_train_full: pd.DataFrame | None = None
        self._X_test_full: pd.DataFrame | None = None
        self._atr_pct_train: np.ndarray | None = None
        self._atr_pct_test: np.ndarray | None = None
        self.feature_names: list[str] = []

        self.sl_models: dict[str, xgb.XGBRegressor] = {}
        self._last_s1_window: tuple[str, str] = ("fixed_atr", "fixed_atr")

        # Factorized Stage-2 components per (side, rr)
        self.dir_heads: dict[tuple[str, float], xgb.XGBClassifier] = {}
        self.dir_cals: dict[tuple[str, float], LogisticRegression] = {}
        self.dir_priors: dict[tuple[str, float], float] = {}

        self.to_heads: dict[tuple[str, float], xgb.XGBClassifier] = {}
        self.to_cals: dict[tuple[str, float], LogisticRegression] = {}
        self.to_priors: dict[tuple[str, float], float] = {}

        self.to_payoff_regs: dict[tuple[str, float], xgb.XGBRegressor | None] = {}
        self.to_payoff_cal_shift: dict[tuple[str, float], float] = {}
        self.to_payoff_stats: dict[tuple[str, float], tuple[float, float]] = {}

        # Strictly causal seeds for 60d rolling prior-shift recalibration
        self._seed_q_p: dict[tuple[str, float], np.ndarray] = {}
        self._seed_q_y: dict[tuple[str, float], np.ndarray] = {}
        self._seed_q_act: dict[tuple[str, float], np.ndarray] = {}
        self._seed_to_p: dict[tuple[str, float], np.ndarray] = {}
        self._seed_to_y: dict[tuple[str, float], np.ndarray] = {}
        self._seed_to_act: dict[tuple[str, float], np.ndarray] = {}

        self.head_diagnostics: dict[str, dict] = {}
        self.selected_min_ev_r: float = 0.0
        self.val_ev_grid: list[dict] = []
        self.test_backtest: dict | None = None
        self.test_signals_df: pd.DataFrame | None = None
        self.train_time: float = 0.0

    def feature_engineering(self, train_data, test_data, auxiliary_data=None, **kwargs):
        train_df = _normalize_ohlcv(train_data)
        test_df = _normalize_ohlcv(test_data)
        if train_df.empty or test_df.empty or train_df.index[-1] >= test_df.index[0]:
            raise ValueError("Training candles must end before test candles begin")

        self._train_df = train_df
        self._test_df = test_df

        full_df = pd.concat([train_df, test_df])
        feats = compute_features(
            full_df,
            include_fvg_ob=False,
            auxiliary_data=auxiliary_data,
            timeframe=self.timeframe,
        )
        train_feats = feats.loc[train_df.index]
        keep_cols = [c for c in train_feats.columns if train_feats[c].notna().mean() > 0.5]
        feats = feats[keep_cols]
        self.feature_names = keep_cols

        self._X_train_full = feats.loc[train_df.index].copy()
        self._X_test_full = feats.loc[test_df.index].copy()

        prev_c = full_df["Close"].astype(float).shift(1)
        tr = pd.concat(
            [
                (full_df["High"].astype(float) - full_df["Low"].astype(float)),
                (full_df["High"].astype(float) - prev_c).abs(),
                (full_df["Low"].astype(float) - prev_c).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_pct = tr.rolling(14, min_periods=14).mean() / np.maximum(full_df["Close"].astype(float), 1e-10)
        self._atr_pct_train = atr_pct.loc[train_df.index].to_numpy(dtype=float)
        self._atr_pct_test = atr_pct.loc[test_df.index].to_numpy(dtype=float)

        print(
            f"MultiRRBracket-V2.1 ({self.symbol} {self.timeframe}, sl_mode={self.sl_mode}, "
            f"max_bars={self.max_bars}, R:R={self.rr_ratios}): "
            f"train={len(train_df)}, test={len(test_df)}, features={len(keep_cols)} | "
            f"median ATR%={np.nanmedian(self._atr_pct_train)*100:.3f}%"
        )

    def _fit_stage1_pair(
        self,
        X: pd.DataFrame,
        atr_pct: np.ndarray,
        feat_ok: np.ndarray,
        mae_long: np.ndarray,
        mae_short: np.ndarray,
        timestamps: pd.DatetimeIndex,
        start_idx: int,
        end_idx: int,
        entry_mask: np.ndarray | None = None,
    ) -> dict[str, xgb.XGBRegressor]:
        sl_slice = slice(start_idx, end_idx)
        atr_s = atr_pct[sl_slice]
        age_days = (timestamps[end_idx - 1] - timestamps[sl_slice]).total_seconds().to_numpy(dtype=float) / 86400.0
        w_base = np.exp(-np.log(2.0) * np.maximum(age_days, 0.0) / 365.0)
        sess_s = entry_mask[sl_slice] if entry_mask is not None else np.ones(end_idx - start_idx, dtype=bool)
        models = {}
        for side_name, mae_arr in (("long", mae_long), ("short", mae_short)):
            mask = feat_ok[sl_slice] & sess_s & np.isfinite(mae_arr[sl_slice]) & np.isfinite(atr_s) & (atr_s > 0.0)
            if int(mask.sum()) < 200:
                mask = feat_ok[sl_slice] & np.isfinite(mae_arr[sl_slice]) & np.isfinite(atr_s) & (atr_s > 0.0)
            X_s = X.iloc[sl_slice].loc[mask]
            y_s = np.clip(mae_arr[sl_slice][mask] / atr_s[mask], self.min_sl_atr, self.max_sl_atr)
            w_s = w_base[mask]
            reg = xgb.XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=self.sl_quantile,
                n_estimators=120,
                max_depth=4,
                learning_rate=0.05,
                min_child_weight=60,
                subsample=0.75,
                colsample_bytree=0.55,
                reg_lambda=20.0,
                reg_alpha=1.0,
                tree_method="hist",
                random_state=self.random_state,
                n_jobs=4,
            )
            reg.fit(X_s, y_s, sample_weight=w_s)
            models[side_name] = reg
        return models

    def _predict_sl_with_models(
        self,
        X: pd.DataFrame,
        atr_pct: np.ndarray,
        sl_models: dict[str, xgb.XGBRegressor] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        atr_safe = np.where(np.isfinite(atr_pct) & (atr_pct > 0.0), atr_pct, np.nan)
        if self.sl_mode == "fixed_atr":
            sl_fixed = np.clip(self.fixed_sl_atr * atr_safe, self.min_sl_pct, self.max_sl_pct).astype(float)
            return sl_fixed, sl_fixed.copy()
        models = sl_models if sl_models is not None else self.sl_models
        mult_long = np.clip(models["long"].predict(X), self.min_sl_atr, self.max_sl_atr)
        mult_short = np.clip(models["short"].predict(X), self.min_sl_atr, self.max_sl_atr)
        sl_long = np.clip(mult_long * atr_safe, self.min_sl_pct, self.max_sl_pct)
        sl_short = np.clip(mult_short * atr_safe, self.min_sl_pct, self.max_sl_pct)
        return sl_long.astype(float), sl_short.astype(float)

    def _predict_sl_pct(self, X: pd.DataFrame, atr_pct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._predict_sl_with_models(X, atr_pct, self.sl_models)

    def _predict_factorized_raw(
        self, X: pd.DataFrame
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Predict Platt-calibrated q_TP(x), p_TO(x), and conditional mu_TO(x) for each (side, rr)."""
        n = len(X)
        n_rr = len(self.rr_ratios)
        q_out, to_out, mu_out = {}, {}, {}
        for side_name in ("long", "short"):
            q_mat = np.zeros((n, n_rr), dtype=float)
            to_mat = np.zeros((n, n_rr), dtype=float)
            mu_mat = np.zeros((n, n_rr), dtype=float)
            for j, rr in enumerate(self.rr_ratios):
                key = (side_name, rr)
                raw_q = self.dir_heads[key].predict_proba(X)[:, 1]
                cal_q = self.dir_cals[key].predict_proba(_logit(raw_q).reshape(-1, 1))[:, 1]
                if self.prob_shrinkage > 0.0:
                    cal_q = (1.0 - self.prob_shrinkage) * cal_q + self.prob_shrinkage * self.dir_priors[key]

                raw_to = self.to_heads[key].predict_proba(X)[:, 1]
                cal_to = self.to_cals[key].predict_proba(_logit(raw_to).reshape(-1, 1))[:, 1]
                if self.prob_shrinkage > 0.0:
                    cal_to = (1.0 - self.prob_shrinkage) * cal_to + self.prob_shrinkage * self.to_priors[key]

                reg_mu = self.to_payoff_regs[key]
                base_mu, _ = self.to_payoff_stats[key]
                if reg_mu is not None:
                    pred_mu = reg_mu.predict(X) + self.to_payoff_cal_shift[key]
                    # Shrink gently toward 0/prior and clip cleanly into [-0.95, 0.90*rr]
                    pred_mu = (1.0 - self.prob_shrinkage) * pred_mu + self.prob_shrinkage * base_mu
                    pred_mu = np.clip(pred_mu, -0.95, 0.90 * rr)
                else:
                    pred_mu = np.full(n, base_mu, dtype=float)

                q_mat[:, j] = cal_q
                to_mat[:, j] = cal_to
                mu_mat[:, j] = pred_mu

            q_out[side_name] = q_mat
            to_out[side_name] = to_mat
            mu_out[side_name] = mu_mat
        return q_out, to_out, mu_out

    def _assemble_probs_3class(
        self, q_mat: np.ndarray, to_mat: np.ndarray
    ) -> np.ndarray:
        """Assemble [p_SL, p_TP, p_TO] from (q_TP, p_TO) via exact chain rule + monotone projection."""
        p_to = np.clip(to_mat, 1e-5, 1.0 - 1e-5)
        q_tp = np.clip(q_mat, 1e-5, 1.0 - 1e-5)
        p_tp = (1.0 - p_to) * q_tp
        p_sl = (1.0 - p_to) * (1.0 - q_tp)
        tensor = np.stack([p_sl, p_tp, p_to], axis=-1)
        return _enforce_monotone_probs_3class(tensor)

    def _simulate_ev_backtest(
        self,
        ohlcv_df: pd.DataFrame,
        valid_mask: np.ndarray,
        sl_long_pct: np.ndarray,
        sl_short_pct: np.ndarray,
        probs_by_side: dict[str, np.ndarray],
        mu_to_by_side: dict[str, np.ndarray],
        var_to_by_side: dict[str, np.ndarray],
        min_ev_r: float | np.ndarray,
        entry_mask: np.ndarray | None = None,
        verbose: bool = False,
    ) -> dict:
        """Exact 3-outcome intraday backtest using conditional mu_to(x_t) and Per-Trade Sharpe."""
        opens = ohlcv_df["Open"].to_numpy(dtype=float)
        highs = ohlcv_df["High"].to_numpy(dtype=float)
        lows = ohlcv_df["Low"].to_numpy(dtype=float)
        closes = ohlcv_df["Close"].to_numpy(dtype=float)
        index = ohlcv_df.index
        n = len(ohlcv_df)

        bars_per_day = max(1, int(round(pd.Timedelta(days=1) / pd.Timedelta(self.timeframe))))
        ret_30d = pd.Series(closes).pct_change(30 * bars_per_day).to_numpy(dtype=float)

        if np.ndim(min_ev_r) == 0:
            min_ev_arr = np.full(n, float(min_ev_r), dtype=float)
            min_ev_label = "STAND_DOWN" if not np.isfinite(min_ev_r) else f"{float(min_ev_r):+.3f}R"
            min_ev_scalar = float(min_ev_r) if np.isfinite(min_ev_r) else None
        else:
            min_ev_arr = np.asarray(min_ev_r, dtype=float)
            finite_vals = min_ev_arr[np.isfinite(min_ev_arr)]
            if len(finite_vals) > 0:
                min_ev_scalar = float(np.mean(finite_vals))
                sd_tag = ",SD" if len(finite_vals) < len(min_ev_arr) else ""
                min_ev_label = f"wf[{np.min(finite_vals):+.2f}R..{np.max(finite_vals):+.2f}R{sd_tag}]"
            else:
                min_ev_scalar = None
                min_ev_label = "wf[STAND_DOWN]"

        balance = peak = float(self.initial_cash)
        max_dd = total_fees = 0.0
        position = None
        pending = None
        trades = []
        rejected_stand_down = 0
        rejected_session = 0
        rejected_comm = 0
        rejected_ev = 0
        ignored_while_open = 0

        p_long_t = probs_by_side["long"]
        p_short_t = probs_by_side["short"]
        mu_long = mu_to_by_side["long"]
        mu_short = mu_to_by_side["short"]
        var_long = var_to_by_side["long"]
        var_short = var_to_by_side["short"]
        rr_arr = np.asarray(self.rr_ratios, dtype=float)
        c = self.commission

        for i in range(n):
            o_i, h_i, l_i, c_i = opens[i], highs[i], lows[i], closes[i]

            if pending is not None:
                side, rr, sl_d, p_sl, p_tp, p_to, mu_to, var_to, sig_time, sig_regime = pending
                pending = None
                stop = o_i * (1.0 - side * sl_d)
                target = o_i * (1.0 + side * rr * sl_d)
                if stop > 0.0 and target > 0.0 and abs(o_i - stop) > 0.0:
                    if self.ev_mode == "conservative":
                        ev_r = compute_trade_ev(o_i, target, stop, p_tp, commission=c)
                        std_r = (rr + 1.0) * np.sqrt(max(p_tp * (1.0 - p_tp), 1e-6))
                    else:
                        ev_r, std_r = compute_trade_ev_3outcome(
                            o_i, target, stop, p_tp, p_sl, p_to, mu_to, var_to, commission=c
                        )
                    if ev_r > min_ev_arr[i]:
                        stop_dist = abs(o_i - stop)
                        risk_per_unit = stop_dist + c * (o_i + stop)
                        qty = (balance * self.risk_fraction) / risk_per_unit if risk_per_unit > 0.0 else 0.0
                        max_qty = (balance * self.max_leverage) / (o_i * (1.0 + c))
                        qty = min(qty, max_qty)
                        order_val = qty * o_i
                        entry_fee = order_val * c
                        initial_risk = qty * stop_dist
                        if qty > 0.0 and initial_risk > 0.0 and balance > entry_fee:
                            position = {
                                "side": side,
                                "rr": float(rr),
                                "entry": float(o_i),
                                "entry_index": i,
                                "entry_time": index[i],
                                "signal_time": sig_time,
                                "regime": sig_regime,
                                "stop": float(stop),
                                "target": float(target),
                                "sl_pct": float(sl_d),
                                "quantity": float(qty),
                                "initial_risk": float(initial_risk),
                                "entry_fee": float(entry_fee),
                                "p_sl": float(p_sl),
                                "p_tp": float(p_tp),
                                "p_to": float(p_to),
                                "pred_mu_to": float(mu_to),
                                "expected_ev_r": float(ev_r),
                                "expected_std_r": float(std_r),
                            }
                            balance -= entry_fee
                            total_fees += entry_fee
                    else:
                        rejected_ev += 1

            if position is not None:
                side = position["side"]
                stop = position["stop"]
                target = position["target"]
                exit_price = reason = None

                if (side == 1 and o_i <= stop) or (side == -1 and o_i >= stop):
                    exit_price, reason = o_i, "SL"
                elif (side == 1 and o_i >= target) or (side == -1 and o_i <= target):
                    exit_price, reason = o_i, "TP"
                else:
                    stop_hit = l_i <= stop if side == 1 else h_i >= stop
                    target_hit = h_i >= target if side == 1 else l_i <= target
                    if stop_hit:
                        exit_price, reason = stop, "SL"
                    elif target_hit:
                        exit_price, reason = target, "TP"
                    elif i - position["entry_index"] + 1 >= self.max_bars:
                        exit_price, reason = c_i, "TIMEOUT"

                if reason is not None:
                    gross = side * position["quantity"] * (exit_price - position["entry"])
                    exit_fee = position["quantity"] * exit_price * c
                    net = gross - position["entry_fee"] - exit_fee
                    balance += gross - exit_fee
                    total_fees += exit_fee
                    trades.append(
                        {
                            "signal_time": position["signal_time"],
                            "entry_time": position["entry_time"],
                            "exit_time": index[i],
                            "regime": position["regime"],
                            "side": "LONG" if side == 1 else "SHORT",
                            "rr": position["rr"],
                            "entry": position["entry"],
                            "exit": float(exit_price),
                            "stop": stop,
                            "target": target,
                            "sl_pct": position["sl_pct"],
                            "reason": reason,
                            "p_sl": position["p_sl"],
                            "p_tp": position["p_tp"],
                            "p_to": position["p_to"],
                            "pred_mu_to": position["pred_mu_to"],
                            "expected_ev_r": position["expected_ev_r"],
                            "gross_pnl": float(gross),
                            "commission": float(position["entry_fee"] + exit_fee),
                            "net_pnl": float(net),
                            "gross_r": float(gross / position["initial_risk"]),
                            "commission_r": float((position["entry_fee"] + exit_fee) / position["initial_risk"]),
                            "net_r": float(net / position["initial_risk"]),
                        }
                    )
                    position = None

            unrealized = (
                position["side"] * position["quantity"] * (c_i - position["entry"])
                if position is not None
                else 0.0
            )
            equity = balance + unrealized
            peak = max(peak, equity)
            if peak > 0.0:
                max_dd = max(max_dd, (peak - equity) / peak)

            if not valid_mask[i] or i + 1 >= n:
                continue

            if entry_mask is not None and not entry_mask[i]:
                rejected_session += 1
                continue

            thr_i = float(min_ev_arr[i])
            if not np.isfinite(thr_i):
                rejected_stand_down += 1
                continue

            sl_l = sl_long_pct[i]
            sl_s = sl_short_pct[i]
            comm_ok_l = sl_l > 0.0 and (2.0 * c / sl_l) <= self.max_comm_r
            comm_ok_s = sl_s > 0.0 and (2.0 * c / sl_s) <= self.max_comm_r
            if not comm_ok_l and not comm_ok_s:
                rejected_comm += 1
                continue

            best_score = -np.inf
            best_choice = None

            if self.ev_mode == "conservative":
                # Conservative lower-bound EV: treat every non-TP (SL or TIMEOUT) as -1R inside entry gate
                # so timeout predictability (AUC ~0.71) cannot hijack directional TP selection!
                if comm_ok_l:
                    stop_l = c_i * (1.0 - sl_l)
                    for j, rr in enumerate(rr_arr):
                        tp_l = c_i * (1.0 + rr * sl_l)
                        p_tp = float(p_long_t[i, j, 1])
                        ev_l = compute_trade_ev(c_i, tp_l, stop_l, p_tp, commission=c)
                        if ev_l > thr_i:
                            std_l = (rr + 1.0) * np.sqrt(max(p_tp * (1.0 - p_tp), 1e-6))
                            score_l = ev_l / std_l
                            if score_l > best_score:
                                best_score = score_l
                                best_choice = (1, float(rr), float(sl_l), 1.0 - p_tp, p_tp, 0.0, 0.0, 0.25)

                if comm_ok_s:
                    stop_s = c_i * (1.0 + sl_s)
                    for j, rr in enumerate(rr_arr):
                        tp_s = c_i * (1.0 - rr * sl_s)
                        p_tp = float(p_short_t[i, j, 1])
                        ev_s = compute_trade_ev(c_i, tp_s, stop_s, p_tp, commission=c)
                        if ev_s > thr_i:
                            std_s = (rr + 1.0) * np.sqrt(max(p_tp * (1.0 - p_tp), 1e-6))
                            score_s = ev_s / std_s
                            if score_s > best_score:
                                best_score = score_s
                                best_choice = (-1, float(rr), float(sl_s), 1.0 - p_tp, p_tp, 0.0, 0.0, 0.25)
            else:
                if comm_ok_l:
                    stop_l = c_i * (1.0 - sl_l)
                    for j, rr in enumerate(rr_arr):
                        tp_l = c_i * (1.0 + rr * sl_l)
                        p_sl, p_tp, p_to = float(p_long_t[i, j, 0]), float(p_long_t[i, j, 1]), float(p_long_t[i, j, 2])
                        mu_to, var_to = float(mu_long[i, j]), float(var_long[i, j])
                        ev_l, std_l = compute_trade_ev_3outcome(
                            c_i, tp_l, stop_l, p_tp, p_sl, p_to, mu_to, var_to, commission=c
                        )
                        if ev_l > thr_i:
                            score_l = ev_l / std_l
                            if score_l > best_score:
                                best_score = score_l
                                best_choice = (1, float(rr), float(sl_l), p_sl, p_tp, p_to, mu_to, var_to)

                if comm_ok_s:
                    stop_s = c_i * (1.0 + sl_s)
                    for j, rr in enumerate(rr_arr):
                        tp_s = c_i * (1.0 - rr * sl_s)
                        p_sl, p_tp, p_to = float(p_short_t[i, j, 0]), float(p_short_t[i, j, 1]), float(p_short_t[i, j, 2])
                        mu_to, var_to = float(mu_short[i, j]), float(var_short[i, j])
                        ev_s, std_s = compute_trade_ev_3outcome(
                            c_i, tp_s, stop_s, p_tp, p_sl, p_to, mu_to, var_to, commission=c
                        )
                        if ev_s > thr_i:
                            score_s = ev_s / std_s
                            if score_s > best_score:
                                best_score = score_s
                                best_choice = (-1, float(rr), float(sl_s), p_sl, p_tp, p_to, mu_to, var_to)

            if best_choice is not None:
                if position is not None:
                    ignored_while_open += 1
                else:
                    r30 = ret_30d[i]
                    regime = "BULL_30D" if (np.isfinite(r30) and r30 >= 0.0) else "BEAR_30D"
                    pending = (*best_choice, index[i], regime)
            else:
                rejected_ev += 1

        wins = sum(t["net_pnl"] > 0 for t in trades)
        losses = sum(t["net_pnl"] <= 0 for t in trades)
        counts = {r: sum(t["reason"] == r for t in trades) for r in ("TP", "SL", "TIMEOUT")}
        total_r = float(sum(t["net_r"] for t in trades))
        gross_gain_r = float(sum(max(t["gross_r"], 0.0) for t in trades))
        gross_loss_r = float(sum(min(t["gross_r"], 0.0) for t in trades))
        net_gain_r = float(sum(max(t["net_r"], 0.0) for t in trades))
        net_loss_r = float(sum(min(t["net_r"], 0.0) for t in trades))
        comm_r = float(sum(t["commission_r"] for t in trades))
        avg_exp_ev = float(np.mean([t["expected_ev_r"] for t in trades])) if trades else None
        avg_real_r = float(total_r / len(trades)) if trades else None
        profit_factor = (
            float(
                sum(t["net_pnl"] for t in trades if t["net_pnl"] > 0)
                / abs(sum(t["net_pnl"] for t in trades if t["net_pnl"] < 0))
            )
            if any(t["net_pnl"] < 0 for t in trades)
            else None
        )

        if trades:
            to_trades = [t for t in trades if t["reason"] == "TIMEOUT"]
            selected_calibration = {
                "pred_tp_rate_pct": float(100.0 * np.mean([t["p_tp"] for t in trades])),
                "real_tp_rate_pct": float(100.0 * counts["TP"] / len(trades)),
                "pred_sl_rate_pct": float(100.0 * np.mean([t["p_sl"] for t in trades])),
                "real_sl_rate_pct": float(100.0 * counts["SL"] / len(trades)),
                "pred_to_rate_pct": float(100.0 * np.mean([t["p_to"] for t in trades])),
                "real_to_rate_pct": float(100.0 * counts["TIMEOUT"] / len(trades)),
                "pred_timeout_mean_r": float(np.mean([t["pred_mu_to"] for t in to_trades])) if to_trades else None,
                "real_timeout_mean_gross_r": float(np.mean([t["gross_r"] for t in to_trades])) if to_trades else None,
                "avg_expected_ev_r": avg_exp_ev,
                "avg_realized_r": avg_real_r,
                "sl_pct_min": float(100.0 * min(t["sl_pct"] for t in trades)),
                "sl_pct_median": float(100.0 * np.median([t["sl_pct"] for t in trades])),
                "sl_pct_max": float(100.0 * max(t["sl_pct"] for t in trades)),
            }
        else:
            selected_calibration = {}

        by_rr = {}
        for rr in self.rr_ratios:
            sub = [t for t in trades if abs(t["rr"] - rr) < 1e-9]
            if sub:
                by_rr[f"{rr:g}:1"] = {
                    "trades": len(sub),
                    "longs": sum(t["side"] == "LONG" for t in sub),
                    "shorts": sum(t["side"] == "SHORT" for t in sub),
                    "tp_rate_pct": 100.0 * sum(t["reason"] == "TP" for t in sub) / len(sub),
                    "win_rate_pct": 100.0 * sum(t["net_pnl"] > 0 for t in sub) / len(sub),
                    "net_r": float(sum(t["net_r"] for t in sub)),
                    "avg_expected_ev_r": float(np.mean([t["expected_ev_r"] for t in sub])),
                    "avg_realized_r": float(np.mean([t["net_r"] for t in sub])),
                }

        by_side = {}
        for side_label in ("LONG", "SHORT"):
            sub = [t for t in trades if t["side"] == side_label]
            if sub:
                by_side[side_label] = {
                    "trades": len(sub),
                    "tp_rate_pct": 100.0 * sum(t["reason"] == "TP" for t in sub) / len(sub),
                    "win_rate_pct": 100.0 * sum(t["net_pnl"] > 0 for t in sub) / len(sub),
                    "net_r": float(sum(t["net_r"] for t in sub)),
                }

        by_regime = {}
        for reg_label in ("BULL_30D", "BEAR_30D"):
            sub = [t for t in trades if t["regime"] == reg_label]
            if sub:
                by_regime[reg_label] = {
                    "trades": len(sub),
                    "longs": sum(t["side"] == "LONG" for t in sub),
                    "shorts": sum(t["side"] == "SHORT" for t in sub),
                    "win_rate_pct": 100.0 * sum(t["net_pnl"] > 0 for t in sub) / len(sub),
                    "net_r": float(sum(t["net_r"] for t in sub)),
                }

        result = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start": index[0],
            "end": index[-1],
            "candles": int(n),
            "min_ev_r": min_ev_scalar,
            "commission": float(c),
            "closed_trades": len(trades),
            "wins": wins,
            "losses": losses,
            "win_rate_pct": (100.0 * wins / len(trades)) if trades else None,
            "exit_counts": counts,
            "rejected_stand_down": int(rejected_stand_down),
            "rejected_session": int(rejected_session),
            "rejected_comm": int(rejected_comm),
            "rejected_ev": int(rejected_ev),
            "ignored_while_open": int(ignored_while_open),
            "gross_gain_r": gross_gain_r,
            "gross_loss_r": gross_loss_r,
            "net_gain_r": net_gain_r,
            "net_loss_r": net_loss_r,
            "closed_commission_r": comm_r,
            "closed_net_r": total_r,
            "avg_expected_ev_r": avg_exp_ev,
            "avg_realized_r": avg_real_r,
            "profit_factor": profit_factor,
            "initial_cash": float(self.initial_cash),
            "final_equity": float(equity),
            "return_pct": float(100.0 * (equity / self.initial_cash - 1.0)),
            "max_close_drawdown_pct": float(100.0 * max_dd),
            "total_commission": float(total_fees),
            "selected_calibration": selected_calibration,
            "by_rr": by_rr,
            "by_side": by_side,
            "by_regime": by_regime,
            "open_position": position,
            "trades": trades,
        }

        if verbose:
            print(f"\n=== {self.symbol} {self.timeframe} Multi-R:R V2.1 Factorized 3-Outcome EV (OOS, sl_mode={self.sl_mode}) ===")
            print(f"Period: {result['start']} to {result['end']} | Candles: {n}")
            print(
                f"Settings: max_bars={self.max_bars} | session={self.entry_session} | c={c:.5f}/side | max_comm_r={self.max_comm_r:.2f}R | "
                f"min_ev_r={min_ev_label} | risk={self.risk_fraction*100:.1f}% | max_lev={self.max_leverage:.1f}x"
            )
            print(
                f"Rejections: session={rejected_session:,} | stand_down={rejected_stand_down:,} | comm_gate={rejected_comm:,} | "
                f"ev_gate={rejected_ev:,} | while_open={ignored_while_open:,}"
            )
            wr_s = f"{result['win_rate_pct']:.2f}%" if result["win_rate_pct"] is not None else "N/A"
            pf_s = f"{profit_factor:.2f}" if profit_factor is not None else "N/A"
            print(
                f"Closed trades: {len(trades)} (W={wins}, L={losses}) | WR={wr_s} | PF={pf_s} | "
                f"Exits: TP={counts['TP']}, SL={counts['SL']}, TO={counts['TIMEOUT']}"
            )
            if trades:
                sc = selected_calibration
                print(
                    f"Net Total: {total_r:+.2f}R (Comm={-comm_r:+.2f}R) | "
                    f"ExpEV={avg_exp_ev:+.3f}R vs RealR={avg_real_r:+.3f}R"
                )
                to_p_s = f"{sc['pred_timeout_mean_r']:+.2f}R" if sc["pred_timeout_mean_r"] is not None else "N/A"
                to_r_s = f"{sc['real_timeout_mean_gross_r']:+.2f}R" if sc["real_timeout_mean_gross_r"] is not None else "N/A"
                print(
                    f"Selected-Trade Calib: TP {sc['pred_tp_rate_pct']:.1f}%->{sc['real_tp_rate_pct']:.1f}% | "
                    f"SL {sc['pred_sl_rate_pct']:.1f}%->{sc['real_sl_rate_pct']:.1f}% | "
                    f"TO {sc['pred_to_rate_pct']:.1f}%->{sc['real_to_rate_pct']:.1f}% (E[R|TO] {to_p_s}->{to_r_s}) | "
                    f"SL%=[{sc['sl_pct_min']:.2f}%, med {sc['sl_pct_median']:.2f}%, {sc['sl_pct_max']:.2f}%]"
                )
            print(
                f"Initial: ${self.initial_cash:.2f} | Final: ${equity:.2f} | "
                f"Return: {result['return_pct']:+.2f}% | MaxDD: {result['max_close_drawdown_pct']:.2f}%"
            )
            if by_side:
                print("By Side:", {k: f"n={v['trades']} WR={v['win_rate_pct']:.1f}% Net={v['net_r']:+.2f}R" for k, v in by_side.items()})
            if by_regime:
                print("By Regime:", {k: f"n={v['trades']} (L={v['longs']},S={v['shorts']}) Net={v['net_r']:+.2f}R" for k, v in by_regime.items()})
            if by_rr:
                for k, v in by_rr.items():
                    print(
                        f"  {k:5s}: n={v['trades']:3d} (L={v['longs']:2d},S={v['shorts']:2d}) | "
                        f"TP={v['tp_rate_pct']:4.1f}% WR={v['win_rate_pct']:4.1f}% | "
                        f"Net={v['net_r']:+6.2f}R | Exp={v['avg_expected_ev_r']:+.3f}R Real={v['avg_realized_r']:+.3f}R"
                    )
        return result

    def _fit_on_history(
        self,
        df: pd.DataFrame,
        X_full: pd.DataFrame,
        atr_pct_full: np.ndarray,
        verbose: bool = True,
    ) -> float:
        t0 = time.perf_counter()
        n = len(df)
        gap = self.max_bars
        c = self.commission

        opens = df["Open"].to_numpy(dtype=float)
        highs = df["High"].to_numpy(dtype=float)
        lows = df["Low"].to_numpy(dtype=float)
        closes = df["Close"].to_numpy(dtype=float)
        atr_ok = np.isfinite(atr_pct_full) & (atr_pct_full > 0.0)
        feat_ok = X_full.notna().all(axis=1).to_numpy() & atr_ok

        s1_warmup = int(n * 0.22) if self.sl_mode == "learned" else 0
        s2_start = s1_warmup + gap if self.sl_mode == "learned" else 0
        s2_end = int(n * 0.72)
        cal_start = s2_end + gap
        cal_end = int(n * 0.86)
        val_start = cal_end + gap

        if (s2_end - s2_start) < 1000 or (cal_end - cal_start) < 500 or (n - val_start) < 500:
            raise ValueError("Not enough training candles for purged 2-stage Multi-R:R V2.1 training")

        if self.recency_half_life_days > 0.0:
            age_days = (df.index[-1] - df.index).total_seconds().to_numpy(dtype=float) / 86400.0
            recency_w_all = np.exp(-np.log(2.0) * np.maximum(age_days, 0.0) / self.recency_half_life_days)
        else:
            recency_w_all = np.ones(n, dtype=float)

        # --- Stage 1: Fresh Chronological Rolling-OOF Dynamic SL (or Fixed ATR Baseline) ---
        entry_mask_all = _compute_entry_mask(df.index, self.entry_session)
        sl_long_all = np.full(n, np.nan, dtype=float)
        sl_short_all = np.full(n, np.nan, dtype=float)
        if self.sl_mode == "fixed_atr":
            sl_l_f, sl_s_f = self._predict_sl_with_models(X_full, atr_pct_full, None)
            sl_long_all[:] = sl_l_f
            sl_short_all[:] = sl_s_f
            self._last_s1_window = ("fixed_atr", f"{self.fixed_sl_atr:.2f}_ATR")
        else:
            if verbose:
                print("\n--- Stage 1: Fresh Chronological Rolling-OOF ATR-MAE Quantile SL Models ---")
            mae_long, mae_short = _compute_mae_targets(opens, highs, lows, horizon_bars=self.sl_mae_bars)
            bars_per_year = max(1, int(round(pd.Timedelta(days=365) / pd.Timedelta(self.timeframe))))
            sl_win_bars = max(2000, int(round(self.sl_rolling_years * bars_per_year)))

            fold_edges = np.linspace(s2_start, n, self.sl_oof_folds + 1, dtype=int)
            for f_i in range(self.sl_oof_folds):
                f_start, f_end = int(fold_edges[f_i]), int(fold_edges[f_i + 1])
                if f_end <= f_start:
                    continue
                tr_end = f_start - self.sl_mae_bars
                tr_start = max(0, tr_end - sl_win_bars)
                if tr_end - tr_start < 500:
                    tr_start = 0
                oof_models = self._fit_stage1_pair(
                    X_full, atr_pct_full, feat_ok, mae_long, mae_short, df.index, tr_start, tr_end, entry_mask=entry_mask_all
                )
                sl_l_f, sl_s_f = self._predict_sl_with_models(
                    X_full.iloc[f_start:f_end], atr_pct_full[f_start:f_end], oof_models
                )
                sl_long_all[f_start:f_end] = sl_l_f
                sl_short_all[f_start:f_end] = sl_s_f

            final_end = n - self.sl_mae_bars
            final_start = max(0, final_end - sl_win_bars)
            self.sl_models = self._fit_stage1_pair(
                X_full, atr_pct_full, feat_ok, mae_long, mae_short, df.index, final_start, final_end, entry_mask=entry_mask_all
            )
            self._last_s1_window = (
                df.index[final_start].strftime("%Y-%m-%d"),
                df.index[final_end - 1].strftime("%Y-%m-%d"),
            )
            if verbose:
                med_l = float(np.nanmedian(sl_long_all[s2_start:] / np.maximum(atr_pct_full[s2_start:], 1e-10)))
                med_s = float(np.nanmedian(sl_short_all[s2_start:] / np.maximum(atr_pct_full[s2_start:], 1e-10)))
                print(
                    f"  Fresh Stage-1 test window: [{self._last_s1_window[0]} -> {self._last_s1_window[1]}] | "
                    f"OOS SL median: Long={np.nanmedian(sl_long_all[s2_start:])*100:.3f}% ({med_l:.2f} ATR), "
                    f"Short={np.nanmedian(sl_short_all[s2_start:])*100:.3f}% ({med_s:.2f} ATR)"
                )

        # --- Stage 2: Factorized Exact 3-Outcome Heads ---
        #   (a) Directional Head q_TP = P(TP | TP or SL, x) — early stops purely on directional logloss!
        #   (b) Timeout Head p_TO = P(TIMEOUT | x)
        #   (c) Conditional Timeout Payoff Regressor mu_TO(x) = E[R_TO | TIMEOUT, x]
        if verbose:
            print(f"\n--- Stage 2: Training {2 * len(self.rr_ratios)} Factorized 3-Outcome Heads (q_TP + p_TO + mu_TO(x)) ---")
        s2_slice = slice(s2_start, s2_end - gap)
        cal_slice = slice(cal_start, cal_end - gap)

        X_s2_base = X_full.iloc[s2_slice]
        X_cal_base = X_full.iloc[cal_slice]
        ok_s2 = feat_ok[s2_slice] & entry_mask_all[s2_slice]
        ok_cal = feat_ok[cal_slice] & entry_mask_all[cal_slice]
        y_all_by_head: dict[tuple[str, float], np.ndarray] = {}
        comm_ok_all = {
            "long": entry_mask_all & np.isfinite(sl_long_all) & (sl_long_all > 0.0) & ((2.0 * c / np.maximum(sl_long_all, 1e-10)) <= self.max_comm_r),
            "short": entry_mask_all & np.isfinite(sl_short_all) & (sl_short_all > 0.0) & ((2.0 * c / np.maximum(sl_short_all, 1e-10)) <= self.max_comm_r),
        }

        for side_name, side_int, sl_arr in (("long", 1, sl_long_all), ("short", -1, sl_short_all)):
            comm_cal = comm_ok_all[side_name][cal_slice]
            for rr in self.rr_ratios:
                key_t = (side_name, rr)
                y_all, to_r_all = _simulate_bracket_3class(
                    opens, highs, lows, closes, sl_arr, side=side_int, rr_ratio=rr, max_bars=self.max_bars
                )
                y_all_by_head[key_t] = y_all

                m_fit_all = ok_s2 & np.isfinite(y_all[s2_slice])
                m_cal_all = ok_cal & np.isfinite(y_all[cal_slice])
                # Calibrate strictly on tradable (comm_ok) bars if enough exist (avoids quiet-bar shift!)
                m_cal_eff = m_cal_all & comm_cal
                if int(m_cal_eff.sum()) < 400:
                    m_cal_eff = m_cal_all

                y_fit_3 = y_all[s2_slice][m_fit_all].astype(int)
                y_cal_3 = y_all[cal_slice][m_cal_eff].astype(int)
                X_fit_all = X_s2_base.loc[m_fit_all]
                w_fit_all = recency_w_all[s2_slice][m_fit_all]
                X_cal_eff = X_cal_base.loc[m_cal_eff]

                # (a) Directional Head q_TP = P(TP | barrier hit) trained ONLY on resolved TP/SL bars
                dir_fit_m = y_fit_3 != 2
                dir_cal_m = y_cal_3 != 2
                X_dir_fit = X_fit_all.loc[dir_fit_m]
                y_dir_fit = (y_fit_3[dir_fit_m] == 1).astype(int)
                w_dir_fit = w_fit_all[dir_fit_m]
                X_dir_cal = X_cal_eff.loc[dir_cal_m]
                y_dir_cal = (y_cal_3[dir_cal_m] == 1).astype(int)

                clf_dir = xgb.XGBClassifier(
                    n_estimators=300,
                    max_depth=3,
                    learning_rate=0.03,
                    min_child_weight=100,
                    subsample=0.75,
                    colsample_bytree=0.45,
                    reg_lambda=30.0,
                    reg_alpha=2.0,
                    eval_metric="logloss",
                    early_stopping_rounds=25,
                    tree_method="hist",
                    random_state=self.random_state,
                    n_jobs=4,
                )
                clf_dir.fit(X_dir_fit, y_dir_fit, sample_weight=w_dir_fit, eval_set=[(X_dir_cal, y_dir_cal)], verbose=False)
                raw_dir_cal = clf_dir.predict_proba(X_dir_cal)[:, 1]
                cal_dir = LogisticRegression(C=1.0, max_iter=500, random_state=self.random_state)
                cal_dir.fit(_logit(raw_dir_cal).reshape(-1, 1), y_dir_cal)
                q_cal = cal_dir.predict_proba(_logit(raw_dir_cal).reshape(-1, 1))[:, 1]
                dir_prior = float(np.clip(y_dir_cal.mean(), 0.01, 0.99))

                # (b) Timeout Head p_TO = P(TIMEOUT | x)
                y_to_fit = (y_fit_3 == 2).astype(int)
                y_to_cal = (y_cal_3 == 2).astype(int)
                clf_to = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.04,
                    min_child_weight=80,
                    subsample=0.75,
                    colsample_bytree=0.50,
                    reg_lambda=25.0,
                    reg_alpha=1.0,
                    eval_metric="logloss",
                    early_stopping_rounds=20,
                    tree_method="hist",
                    random_state=self.random_state,
                    n_jobs=4,
                )
                clf_to.fit(X_fit_all, y_to_fit, sample_weight=w_fit_all, eval_set=[(X_cal_eff, y_to_cal)], verbose=False)
                raw_to_cal = clf_to.predict_proba(X_cal_eff)[:, 1]
                cal_to = LogisticRegression(C=1.0, max_iter=500, random_state=self.random_state)
                cal_to.fit(_logit(raw_to_cal).reshape(-1, 1), y_to_cal)
                p_to_cal = cal_to.predict_proba(_logit(raw_to_cal).reshape(-1, 1))[:, 1]
                to_prior = float(np.clip(y_to_cal.mean(), 0.01, 0.99))

                # (c) Conditional Timeout Payoff Regressor mu_TO(x) = E[R_TO | TIMEOUT, x]
                to_r_fit_all = to_r_all[s2_slice][m_fit_all]
                to_r_cal_all = to_r_all[cal_slice][m_cal_eff]
                to_fit_m = (y_fit_3 == 2) & np.isfinite(to_r_fit_all)
                to_cal_m = (y_cal_3 == 2) & np.isfinite(to_r_cal_all)

                to_vals_cal = to_r_cal_all[to_cal_m]
                if len(to_vals_cal) >= 20:
                    shrink = len(to_vals_cal) / (len(to_vals_cal) + 50.0)
                    base_mu = float(shrink * np.mean(to_vals_cal))
                    var_to = float(np.clip(np.var(to_vals_cal), 0.04, (rr + 1.0) ** 2 / 4.0))
                else:
                    base_mu, var_to = 0.0, 0.25

                if int(to_fit_m.sum()) >= 200 and int(to_cal_m.sum()) >= 30:
                    reg_mu = xgb.XGBRegressor(
                        n_estimators=100,
                        max_depth=3,
                        learning_rate=0.04,
                        min_child_weight=120,
                        subsample=0.75,
                        colsample_bytree=0.45,
                        reg_lambda=50.0,
                        reg_alpha=2.0,
                        eval_metric="rmse",
                        early_stopping_rounds=15,
                        tree_method="hist",
                        random_state=self.random_state,
                        n_jobs=4,
                    )
                    reg_mu.fit(
                        X_fit_all.loc[to_fit_m],
                        to_r_fit_all[to_fit_m],
                        sample_weight=w_fit_all[to_fit_m],
                        eval_set=[(X_cal_eff.loc[to_cal_m], to_vals_cal)],
                        verbose=False,
                    )
                    pred_cal_mu = reg_mu.predict(X_cal_eff.loc[to_cal_m])
                    cal_shift = float(base_mu - np.mean(pred_cal_mu))
                    var_to = float(np.clip(np.var(to_vals_cal - (pred_cal_mu + cal_shift)), 0.04, (rr + 1.0) ** 2 / 4.0))
                else:
                    reg_mu = None
                    cal_shift = 0.0

                self.dir_heads[key_t] = clf_dir
                self.dir_cals[key_t] = cal_dir
                self.dir_priors[key_t] = dir_prior
                self.to_heads[key_t] = clf_to
                self.to_cals[key_t] = cal_to
                self.to_priors[key_t] = to_prior
                self.to_payoff_regs[key_t] = reg_mu
                self.to_payoff_cal_shift[key_t] = cal_shift
                self.to_payoff_stats[key_t] = (base_mu, var_to)

                # Diagnostics
                q_all_cal = cal_dir.predict_proba(_logit(clf_dir.predict_proba(X_cal_eff)[:, 1]).reshape(-1, 1))[:, 1]
                p_tp_cal = (1.0 - p_to_cal) * q_all_cal
                y_cal_tp = (y_cal_3 == 1).astype(int)
                auc_tp = float(roc_auc_score(y_cal_tp, p_tp_cal)) if len(np.unique(y_cal_tp)) > 1 else float("nan")
                auc_dir = float(roc_auc_score(y_dir_cal, q_cal)) if len(np.unique(y_dir_cal)) > 1 else float("nan")
                auc_to = float(roc_auc_score(y_to_cal, p_to_cal)) if len(np.unique(y_to_cal)) > 1 else float("nan")
                p_be = 1.0 / (rr + 1.0)
                q_fit_all = cal_dir.predict_proba(_logit(clf_dir.predict_proba(X_fit_all)[:, 1]).reshape(-1, 1))[:, 1]
                p_to_fit = cal_to.predict_proba(_logit(clf_to.predict_proba(X_fit_all)[:, 1]).reshape(-1, 1))[:, 1]
                p_tp_fit = (1.0 - p_to_fit) * q_fit_all
                train_acc = float(accuracy_score((y_fit_3 == 1).astype(int), (p_tp_fit >= p_be).astype(int)))
                dir_trees = int(getattr(clf_dir, "best_iteration", 0) or 0)
                to_trees = int(getattr(clf_to, "best_iteration", 0) or 0)

                key_s = f"{side_name}_{rr:g}"
                self.head_diagnostics[key_s] = {
                    "side": side_name,
                    "rr": rr,
                    "fit_samples": int(len(y_fit_3)),
                    "cal_samples": int(len(y_cal_3)),
                    "cal_dir_prior": dir_prior,
                    "cal_to_prior": to_prior,
                    "cal_timeout_mean_r": base_mu,
                    "cal_timeout_resid_std_r": float(var_to ** 0.5),
                    "train_accuracy": train_acc,
                    "cal_dir_auc": auc_dir,
                    "cal_to_auc": auc_to,
                    "cal_roc_auc": auc_tp,
                    "dir_best_iteration": dir_trees,
                    "to_best_iteration": to_trees,
                }
                if verbose:
                    print(
                        f"  Head {side_name:5s} {rr:3.1f}:1 -> dir_AUC={auc_dir:.4f} (trees={dir_trees:3d}) | "
                        f"to_AUC={auc_to:.4f} (trees={to_trees:3d}) | tp_AUC={auc_tp:.4f} | E[R|TO]={base_mu:+.3f}R"
                    )

        # --- Stage 3: Causal Rolling Re-Calibration (on tradable comm_ok bars) + EV Threshold ---
        if verbose:
            print(
                f"\n--- Stage 3: Validation EV Threshold Tuning (Factorized 3-Outcome EV, rolling_cal={self.rolling_cal_bars} bars) ---"
            )
        val_df = df.iloc[val_start:]
        X_val = X_full.iloc[val_start:]
        val_ok = feat_ok[val_start:]
        sl_l_val = sl_long_all[val_start:]
        sl_s_val = sl_short_all[val_start:]

        q_cal_raw, to_cal_raw, _ = self._predict_factorized_raw(X_full.iloc[cal_start:cal_end])
        q_val_raw, to_val_raw, mu_val = self._predict_factorized_raw(X_val)

        probs_val = {}
        var_val = {}
        min_p_roll = max(150, self.rolling_cal_bars // 12)
        for side_name in ("long", "short"):
            q_adj_cols, to_adj_cols, var_cols = [], [], []
            comm_cal_s = comm_ok_all[side_name][cal_start:cal_end]
            comm_val_s = comm_ok_all[side_name][val_start:]
            for j, rr in enumerate(self.rr_ratios):
                key_t = (side_name, rr)
                y_all = y_all_by_head[key_t]
                y_cal_s = y_all[cal_start:cal_end]
                y_val_s = y_all[val_start:]

                # Directional q_TP calibration strictly on resolved TP/SL & tradable bars
                y_cal_q = np.where(y_cal_s == 2, np.nan, (y_cal_s == 1).astype(float))
                y_val_q = np.where(y_val_s == 2, np.nan, (y_val_s == 1).astype(float))
                q_adj = _causal_rolling_prior_shift_binary(
                    q_val_raw[side_name][:, j],
                    y_val_q,
                    max_bars=self.max_bars,
                    window_bars=self.rolling_cal_bars,
                    min_periods=min_p_roll,
                    mask_active=comm_val_s,
                    seed_probs=q_cal_raw[side_name][:, j],
                    seed_y=y_cal_q,
                    seed_active=comm_cal_s,
                )

                # Timeout p_TO calibration strictly on tradable bars
                y_cal_to = np.where(np.isfinite(y_cal_s), (y_cal_s == 2).astype(float), np.nan)
                y_val_to = np.where(np.isfinite(y_val_s), (y_val_s == 2).astype(float), np.nan)
                to_adj = _causal_rolling_prior_shift_binary(
                    to_val_raw[side_name][:, j],
                    y_val_to,
                    max_bars=self.max_bars,
                    window_bars=self.rolling_cal_bars,
                    min_periods=min_p_roll,
                    mask_active=comm_val_s,
                    seed_probs=to_cal_raw[side_name][:, j],
                    seed_y=y_cal_to,
                    seed_active=comm_cal_s,
                )

                q_adj_cols.append(q_adj)
                to_adj_cols.append(to_adj)
                var_cols.append(np.full(len(val_df), self.to_payoff_stats[key_t][1], dtype=float))

                self._seed_q_p[key_t] = q_val_raw[side_name][:, j].copy()
                self._seed_q_y[key_t] = y_val_q.copy()
                self._seed_q_act[key_t] = comm_val_s.copy()
                self._seed_to_p[key_t] = to_val_raw[side_name][:, j].copy()
                self._seed_to_y[key_t] = y_val_to.copy()
                self._seed_to_act[key_t] = comm_val_s.copy()

            probs_val[side_name] = self._assemble_probs_3class(
                np.column_stack(q_adj_cols), np.column_stack(to_adj_cols)
            )
            var_val[side_name] = np.column_stack(var_cols)

        best_score = None
        best_ev_thr = max(self.min_ev_floor, 0.12)
        best_net_r = -np.inf
        val_grid = []
        ev_candidates = tuple(
            t for t in (0.10, 0.14, 0.18, 0.22, 0.26, 0.30)
            if t >= self.min_ev_floor - 1e-9
        ) or (self.min_ev_floor,)
        val_entry_mask = entry_mask_all[val_start:]
        for thr in ev_candidates:
            bt_v = self._simulate_ev_backtest(
                val_df, val_ok, sl_l_val, sl_s_val, probs_val, mu_val, var_val,
                min_ev_r=thr, entry_mask=val_entry_mask, verbose=False,
            )
            net_r = float(bt_v["closed_net_r"])
            max_dd = float(bt_v["max_close_drawdown_pct"])
            calmar_r = net_r / max(max_dd, 1.0)
            row = {
                "min_ev_r": thr,
                "trades": int(bt_v["closed_trades"]),
                "win_rate_pct": bt_v["win_rate_pct"],
                "net_r": net_r,
                "return_pct": float(bt_v["return_pct"]),
                "max_dd_pct": max_dd,
                "net_r_over_maxdd": float(calmar_r),
                "profit_factor": bt_v["profit_factor"],
                "avg_expected_ev_r": bt_v["avg_expected_ev_r"],
                "avg_realized_r": bt_v["avg_realized_r"],
            }
            val_grid.append(row)
            if verbose:
                wr_s = f"{bt_v['win_rate_pct']:5.1f}%" if bt_v["win_rate_pct"] is not None else "  N/A"
                print(
                    f"  min_ev={thr:+.2f}R | trades={bt_v['closed_trades']:4d} | WR={wr_s} | "
                    f"Net={net_r:+7.2f}R | MaxDD={max_dd:5.2f}% | NetR/DD={calmar_r:+5.2f} | Ret={bt_v['return_pct']:+6.2f}%"
                )
            eligible = bt_v["closed_trades"] >= 15
            score = (1 if eligible else 0, float(calmar_r), float(net_r))
            if best_score is None or score > best_score:
                best_score = score
                best_ev_thr = thr
                best_net_r = net_r

        if best_net_r <= 0.0:
            best_ev_thr = float("inf") if self.stand_down_on_neg_val else max(best_ev_thr, 0.12)

        self.val_ev_grid = val_grid
        self.selected_min_ev_r = float(best_ev_thr)
        elapsed = float(time.perf_counter() - t0)
        if verbose:
            sel_str = "STAND_DOWN (neg val)" if not np.isfinite(self.selected_min_ev_r) else f"{self.selected_min_ev_r:+.2f}R"
            print(f"Selected min_ev_r={sel_str} on validation (fit_time={elapsed:.1f}s)")
        return self.selected_min_ev_r

    def train(self):
        if self._train_df is None or self._X_train_full is None:
            raise RuntimeError("Run feature_engineering() before train()")
        t0 = time.perf_counter()
        res = self._fit_on_history(self._train_df, self._X_train_full, self._atr_pct_train, verbose=True)
        self.train_time = float(time.perf_counter() - t0)
        return res

    def test(self):
        if self._test_df is None or not self.dir_heads:
            raise RuntimeError("Run train() before test()")

        t0_wf = time.perf_counter()
        X_te_all = self._X_test_full
        te_ok = (
            X_te_all.notna().all(axis=1).to_numpy()
            & np.isfinite(self._atr_pct_test)
            & (self._atr_pct_test > 0.0)
        )
        n_te = len(self._test_df)
        n_rr = len(self.rr_ratios)
        c = self.commission

        sl_l_te = np.full(n_te, np.nan, dtype=float)
        sl_s_te = np.full(n_te, np.nan, dtype=float)
        probs_te = {
            "long": np.full((n_te, n_rr, 3), np.nan, dtype=float),
            "short": np.full((n_te, n_rr, 3), np.nan, dtype=float),
        }
        mu_te = {
            "long": np.zeros((n_te, n_rr), dtype=float),
            "short": np.zeros((n_te, n_rr), dtype=float),
        }
        var_te = {
            "long": np.full((n_te, n_rr), 0.25, dtype=float),
            "short": np.full((n_te, n_rr), 0.25, dtype=float),
        }
        min_ev_arr = np.full(n_te, self.selected_min_ev_r, dtype=float)

        if self.refit_days > 0:
            chunk_bars = max(1, int(round(pd.Timedelta(days=self.refit_days) / pd.Timedelta(self.timeframe))))
            chunks = [(s, min(n_te, s + chunk_bars)) for s in range(0, n_te, chunk_bars)]
        else:
            chunks = [(0, n_te)]

        min_p_roll = max(150, self.rolling_cal_bars // 12)
        self.refit_summary = []
        all_chunk_train_accs: dict[str, list[float]] = {
            f"{side}_{rr:g}": [] for side in ("long", "short") for rr in self.rr_ratios
        }

        if len(chunks) > 1:
            print(
                f"\n=== Walk-Forward Periodic Refitting ({len(chunks)} chunks x {self.refit_days}d across {n_te} OOS bars) ==="
            )

        for chunk_idx, (c_start, c_end) in enumerate(chunks):
            if chunk_idx > 0:
                hist_df = pd.concat([self._train_df, self._test_df.iloc[:c_start]])
                hist_X = pd.concat([self._X_train_full, X_te_all.iloc[:c_start]])
                hist_atr = np.concatenate([self._atr_pct_train, self._atr_pct_test[:c_start]])
                self._fit_on_history(hist_df, hist_X, hist_atr, verbose=False)

            for k in all_chunk_train_accs:
                all_chunk_train_accs[k].append(float(self.head_diagnostics[k]["train_accuracy"]))

            chunk_df = self._test_df.iloc[c_start:c_end]
            chunk_X = X_te_all.iloc[c_start:c_end]
            chunk_atr = self._atr_pct_test[c_start:c_end]
            sl_l_c, sl_s_c = self._predict_sl_pct(chunk_X, chunk_atr)
            q_c_raw, to_c_raw, mu_c = self._predict_factorized_raw(chunk_X)

            o_c = chunk_df["Open"].to_numpy(dtype=float)
            h_c = chunk_df["High"].to_numpy(dtype=float)
            l_c = chunk_df["Low"].to_numpy(dtype=float)
            c_c = chunk_df["Close"].to_numpy(dtype=float)

            chunk_entry_mask = _compute_entry_mask(chunk_df.index, self.entry_session)
            for side_name, side_int, sl_arr in (("long", 1, sl_l_c), ("short", -1, sl_s_c)):
                comm_c = chunk_entry_mask & np.isfinite(sl_arr) & (sl_arr > 0.0) & ((2.0 * c / np.maximum(sl_arr, 1e-10)) <= self.max_comm_r)
                q_adj_cols, to_adj_cols, var_cols = [], [], []
                for j, rr in enumerate(self.rr_ratios):
                    key_t = (side_name, rr)
                    y_c, _ = _simulate_bracket_3class(
                        o_c, h_c, l_c, c_c, sl_arr, side=side_int, rr_ratio=rr, max_bars=self.max_bars
                    )
                    y_c_q = np.where(y_c == 2, np.nan, (y_c == 1).astype(float))
                    y_c_to = np.where(np.isfinite(y_c), (y_c == 2).astype(float), np.nan)

                    q_adj = _causal_rolling_prior_shift_binary(
                        q_c_raw[side_name][:, j],
                        y_c_q,
                        max_bars=self.max_bars,
                        window_bars=self.rolling_cal_bars,
                        min_periods=min_p_roll,
                        mask_active=comm_c,
                        seed_probs=self._seed_q_p.get(key_t),
                        seed_y=self._seed_q_y.get(key_t),
                        seed_active=self._seed_q_act.get(key_t),
                    )
                    to_adj = _causal_rolling_prior_shift_binary(
                        to_c_raw[side_name][:, j],
                        y_c_to,
                        max_bars=self.max_bars,
                        window_bars=self.rolling_cal_bars,
                        min_periods=min_p_roll,
                        mask_active=comm_c,
                        seed_probs=self._seed_to_p.get(key_t),
                        seed_y=self._seed_to_y.get(key_t),
                        seed_active=self._seed_to_act.get(key_t),
                    )
                    q_adj_cols.append(q_adj)
                    to_adj_cols.append(to_adj)
                    var_cols.append(np.full(len(chunk_df), self.to_payoff_stats[key_t][1], dtype=float))

                probs_te[side_name][c_start:c_end] = self._assemble_probs_3class(
                    np.column_stack(q_adj_cols), np.column_stack(to_adj_cols)
                )
                mu_te[side_name][c_start:c_end] = mu_c[side_name]
                var_te[side_name][c_start:c_end] = np.column_stack(var_cols)

            sl_l_te[c_start:c_end] = sl_l_c
            sl_s_te[c_start:c_end] = sl_s_c
            min_ev_arr[c_start:c_end] = self.selected_min_ev_r

            best_val_row = (
                next((r for r in self.val_ev_grid if abs(r["min_ev_r"] - self.selected_min_ev_r) < 1e-9), {})
                if np.isfinite(self.selected_min_ev_r)
                else max(self.val_ev_grid, key=lambda r: r.get("net_r_over_maxdd", -1e9), default={})
            )
            chunk_info = {
                "chunk": chunk_idx + 1,
                "start": str(chunk_df.index[0]),
                "end": str(chunk_df.index[-1]),
                "bars": int(c_end - c_start),
                "train_bars": int(len(self._train_df) + c_start),
                "stage1_window": list(self._last_s1_window),
                "selected_min_ev_r": float(self.selected_min_ev_r) if np.isfinite(self.selected_min_ev_r) else None,
                "stand_down": not np.isfinite(self.selected_min_ev_r),
                "val_net_r": best_val_row.get("net_r"),
                "val_max_dd_pct": best_val_row.get("max_dd_pct"),
                "val_net_r_over_maxdd": best_val_row.get("net_r_over_maxdd"),
            }
            self.refit_summary.append(chunk_info)
            if len(chunks) > 1:
                ev_str = "STAND_DOWN" if not np.isfinite(self.selected_min_ev_r) else f"{self.selected_min_ev_r:+.2f}R"
                print(
                    f"  Chunk {chunk_idx+1}/{len(chunks)} [{chunk_df.index[0].strftime('%Y-%m-%d')} -> "
                    f"{chunk_df.index[-1].strftime('%Y-%m-%d')}] | S1={self._last_s1_window[0]}..{self._last_s1_window[1]} | "
                    f"min_ev={ev_str} (val Net={chunk_info['val_net_r']:+.1f}R, DD={chunk_info['val_max_dd_pct']:.1f}%)"
                )

        if len(chunks) > 1:
            self.train_time += float(time.perf_counter() - t0_wf)

        opens = self._test_df["Open"].to_numpy(dtype=float)
        highs = self._test_df["High"].to_numpy(dtype=float)
        lows = self._test_df["Low"].to_numpy(dtype=float)
        closes = self._test_df["Close"].to_numpy(dtype=float)

        te_entry_mask = _compute_entry_mask(self._test_df.index, self.entry_session)
        print("\n--- Test Head Factorized 3-Outcome Calibration, Balanced Acc & AUC (stitched OOS) ---")
        train_accs, test_accs, always0_accs, test_bal_accs, test_aucs = [], [], [], [], []
        for side_name, side_int, sl_arr in (("long", 1, sl_l_te), ("short", -1, sl_s_te)):
            for j, rr in enumerate(self.rr_ratios):
                y_te, to_r_te = _simulate_bracket_3class(
                    opens, highs, lows, closes, sl_arr, side=side_int, rr_ratio=rr, max_bars=self.max_bars
                )
                p3 = probs_te[side_name][:, j, :]
                m = te_ok & te_entry_mask & np.isfinite(y_te) & np.all(np.isfinite(p3), axis=1)
                yt = y_te[m].astype(int)
                pt = p3[m]
                yt_tp = (yt == 1).astype(int)
                p_tp = pt[:, 1]
                p_be = 1.0 / (rr + 1.0)
                y_pred_tp = (p_tp >= p_be).astype(int)

                te_acc = float(accuracy_score(yt_tp, y_pred_tp)) if len(yt) else float("nan")
                always0 = float(1.0 - yt_tp.mean()) if len(yt) else float("nan")
                te_bal = float(balanced_accuracy_score(yt_tp, y_pred_tp)) if len(np.unique(yt_tp)) > 1 else float("nan")
                auc = float(roc_auc_score(yt_tp, p_tp)) if len(np.unique(yt_tp)) > 1 else float("nan")
                to_vals_te = to_r_te[m][np.isfinite(to_r_te[m])]

                key = f"{side_name}_{rr:g}"
                tr_acc = float(np.mean(all_chunk_train_accs[key]))
                self.head_diagnostics[key].update(
                    {
                        "train_accuracy": tr_acc,
                        "test_accuracy": te_acc,
                        "test_always0_accuracy": always0,
                        "test_balanced_accuracy": te_bal,
                        "test_roc_auc": auc,
                        "test_base_sl_rate": float((yt == 0).mean()) if len(yt) else None,
                        "test_mean_pred_sl": float(pt[:, 0].mean()) if len(yt) else None,
                        "test_base_tp_rate": float((yt == 1).mean()) if len(yt) else None,
                        "test_mean_pred_tp": float(pt[:, 1].mean()) if len(yt) else None,
                        "test_base_to_rate": float((yt == 2).mean()) if len(yt) else None,
                        "test_mean_pred_to": float(pt[:, 2].mean()) if len(yt) else None,
                        "test_real_timeout_mean_r": float(np.mean(to_vals_te)) if len(to_vals_te) else None,
                    }
                )
                train_accs.append(tr_acc)
                test_accs.append(te_acc)
                always0_accs.append(always0)
                test_bal_accs.append(te_bal)
                test_aucs.append(auc)
                print(
                    f"  Test {side_name:5s} {rr:3.1f}:1 -> BalAcc={te_bal*100:5.2f}% (Acc={te_acc*100:5.2f}% vs 0-base={always0*100:5.2f}%) | "
                    f"TP={yt_tp.mean()*100:4.1f}/{p_tp.mean()*100:4.1f}% TO={(yt==2).mean()*100:4.1f}/{pt[:,2].mean()*100:4.1f}% | AUC={auc:.4f}"
                )

        self.avg_train_accuracy = float(np.nanmean(train_accs))
        self.avg_test_accuracy = float(np.nanmean(test_accs))
        self.avg_test_always0_accuracy = float(np.nanmean(always0_accs))
        self.avg_test_balanced_accuracy = float(np.nanmean(test_bal_accs))
        self.avg_test_roc_auc = float(np.nanmean(test_aucs))
        print(
            f"  OVERALL MEAN -> BalAcc={self.avg_test_balanced_accuracy*100:.2f}% | "
            f"Acc={self.avg_test_accuracy*100:.2f}% (always-0={self.avg_test_always0_accuracy*100:.2f}%) | "
            f"AUC={self.avg_test_roc_auc:.4f}"
        )

        self.test_backtest = self._simulate_ev_backtest(
            self._test_df,
            te_ok,
            sl_l_te,
            sl_s_te,
            probs_te,
            mu_te,
            var_te,
            min_ev_r=min_ev_arr if len(chunks) > 1 else self.selected_min_ev_r,
            entry_mask=te_entry_mask,
            verbose=True,
        )
        self.test_signals_df = self._test_df.copy()
        return self.test_backtest

    def evaluate(self):
        if self.test_backtest is None:
            raise RuntimeError("Run test() before evaluate()")

        from backtest import plot_backtest_positions

        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        tag = f"v21_{self.sl_mode}_{len(self.rr_ratios)}rr"
        chart_path = (
            self.results_path.parent
            / f"multi_rr_bracket_{self.symbol}_{self.timeframe}_{tag}_{stamp}_trades.png"
        )
        plot_backtest_positions(self.test_signals_df, self.test_backtest, chart_path, self.timeframe, recent_days=30)

        bt_clean = {
            k: v for k, v in self.test_backtest.items()
            if k not in ("trades", "open_position")
        }
        bt_clean["start"] = str(self.test_backtest["start"])
        bt_clean["end"] = str(self.test_backtest["end"])

        positions = [
            {
                "entry_time": str(t["entry_time"]),
                "exit_time": str(t["exit_time"]),
                "regime": t["regime"],
                "side": t["side"],
                "rr": t["rr"],
                "open_price": round(t["entry"], 4),
                "tp": round(t["target"], 4),
                "sl": round(t["stop"], 4),
                "close_price": round(t["exit"], 4),
                "sl_pct": round(t["sl_pct"], 5),
                "exit_reason": t["reason"],
                "p_sl": round(t["p_sl"], 4),
                "p_tp": round(t["p_tp"], 4),
                "p_to": round(t["p_to"], 4),
                "pred_mu_to": round(t["pred_mu_to"], 4),
                "expected_ev_r": round(t["expected_ev_r"], 4),
                "gross_r": round(t["gross_r"], 4),
                "net_r": round(t["net_r"], 4),
            }
            for t in self.test_backtest.get("trades", [])
        ]

        config = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "model_version": f"v2.2_{self.ev_mode}_fresh_s1",
            "ev_mode": self.ev_mode,
            "sl_mode": self.sl_mode,
            "fixed_sl_atr": self.fixed_sl_atr,
            "rr_ratios": list(self.rr_ratios),
            "max_bars": self.max_bars,
            "sl_mae_bars": self.sl_mae_bars,
            "sl_quantile": self.sl_quantile,
            "sl_rolling_years": self.sl_rolling_years,
            "sl_oof_folds": self.sl_oof_folds,
            "min_sl_atr": self.min_sl_atr,
            "max_sl_atr": self.max_sl_atr,
            "min_sl_pct": self.min_sl_pct,
            "max_sl_pct": self.max_sl_pct,
            "commission": self.commission,
            "risk_fraction": self.risk_fraction,
            "max_leverage": self.max_leverage,
            "rolling_cal_bars": self.rolling_cal_bars,
            "refit_days": self.refit_days,
            "recency_half_life_days": self.recency_half_life_days,
            "min_ev_floor": self.min_ev_floor,
            "max_comm_r": self.max_comm_r,
            "stand_down_on_neg_val": self.stand_down_on_neg_val,
            "prob_shrinkage": self.prob_shrinkage,
            "entry_session": self.entry_session,
            "random_state": self.random_state,
        }

        record = {
            "symbol": self.symbol,
            "time": datetime.now().isoformat(timespec="seconds"),
            "git_commit": _get_git_revision(),
            "model": "MultiRRBracketModel_V2.1",
            "config": config,
            "n_features": len(self.feature_names),
            "selected_min_ev_r": float(self.selected_min_ev_r) if np.isfinite(self.selected_min_ev_r) else None,
            "refit_summary": self.refit_summary,
            "train_time": self.train_time,
            "avg_train_accuracy": getattr(self, "avg_train_accuracy", None),
            "avg_test_accuracy": getattr(self, "avg_test_accuracy", None),
            "avg_test_always0_accuracy": getattr(self, "avg_test_always0_accuracy", None),
            "avg_test_balanced_accuracy": getattr(self, "avg_test_balanced_accuracy", None),
            "avg_test_roc_auc": getattr(self, "avg_test_roc_auc", None),
            "average_win_rate_pct": self.test_backtest.get("win_rate_pct"),
            "net_total_r": self.test_backtest.get("closed_net_r"),
            "head_diagnostics": self.head_diagnostics,
            "val_ev_grid": self.val_ev_grid,
            "test_backtest": bt_clean,
            "positions": positions,
            "chart_path": str(chart_path),
        }

        results = []
        if self.results_path.exists():
            with self.results_path.open() as f:
                results = json.load(f)
            if not isinstance(results, list):
                results = [results]
        results.append(record)
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with self.results_path.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {self.results_path} ({len(results)} runs total)")
        return record
