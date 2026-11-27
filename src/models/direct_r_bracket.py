from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

try:
    from ML_interface import Model
    from models.XGBoost import _normalize_ohlcv, compute_features
    from models.multi_rr_bracket import _compute_mae_targets, _get_git_revision
except ImportError:
    from src.ML_interface import Model
    from src.models.XGBoost import _normalize_ohlcv, compute_features
    from src.models.multi_rr_bracket import _compute_mae_targets, _get_git_revision


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RR_RATIOS = (2.0, 2.5, 3.0, 3.5, 4.0)


def _simulate_bracket_exact_net_r(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    sl_pct: np.ndarray,
    side: int,
    rr_ratio: float,
    max_bars: int,
    commission: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact vectorized realized net R-multiple matching the live backtest simulator 1-for-1.

    Includes:
      - Next-bar open entry e = opens[i + 1]
      - Conservative SL-first tie-breaking on both gaps and intrabar ranges
      - Exact timeout exit at closes[i + max_bars]
      - Exact round-trip commission c * (e + exit_price) / |e - SL|

    Returns (net_r, outcome_code) where outcome_code is 0=SL, 1=TP, 2=TIMEOUT, NaN=invalid.
    """
    n = len(opens)
    valid_n = max(0, n - max_bars - 1)
    net_r = np.full(n, np.nan, dtype=float)
    outcome = np.full(n, np.nan, dtype=float)
    if valid_n <= 0:
        return net_r, outcome

    entry = opens[1 : valid_n + 1]
    sl_d = sl_pct[:valid_n]
    ok = np.isfinite(entry) & (entry > 0.0) & np.isfinite(sl_d) & (sl_d > 0.0)

    stop = entry * (1.0 - side * sl_d)
    target = entry * (1.0 + side * rr_ratio * sl_d)
    ok &= (stop > 0.0) & (target > 0.0)

    # status: -2=invalid, -1=open, 0=SL, 1=TP
    status = np.full(valid_n, -1, dtype=np.int8)
    status[~ok] = -2
    exit_p = np.full(valid_n, np.nan, dtype=float)

    for k in range(1, max_bars + 1):
        still_open = status == -1
        if not np.any(still_open):
            break
        o_k = opens[k : valid_n + k]
        h_k = highs[k : valid_n + k]
        l_k = lows[k : valid_n + k]

        if side == 1:
            m = still_open & (o_k <= stop)
            status[m] = 0
            exit_p[m] = o_k[m]
            still_open = status == -1

            m = still_open & (o_k >= target)
            status[m] = 1
            exit_p[m] = o_k[m]
            still_open = status == -1

            m = still_open & (l_k <= stop)
            status[m] = 0
            exit_p[m] = stop[m]
            still_open = status == -1

            m = still_open & (h_k >= target)
            status[m] = 1
            exit_p[m] = target[m]
        else:
            m = still_open & (o_k >= stop)
            status[m] = 0
            exit_p[m] = o_k[m]
            still_open = status == -1

            m = still_open & (o_k <= target)
            status[m] = 1
            exit_p[m] = o_k[m]
            still_open = status == -1

            m = still_open & (h_k >= stop)
            status[m] = 0
            exit_p[m] = stop[m]
            still_open = status == -1

            m = still_open & (l_k <= target)
            status[m] = 1
            exit_p[m] = target[m]

    is_to = status == -1
    status[is_to] = 2
    exit_p[is_to] = closes[max_bars : valid_n + max_bars][is_to]

    risk = np.maximum(np.abs(entry - stop), 1e-10)
    gross = side * (exit_p - entry)
    fees = commission * (entry + exit_p)
    nr = (gross - fees) / risk
    # Clip extreme flash-crash gap outliers for training stability (backtest uses exact unclipped fills)
    nr = np.clip(nr, -2.0, rr_ratio + 0.5)

    valid_out = (status != -2) & np.isfinite(nr)
    net_r[:valid_n] = np.where(valid_out, nr, np.nan)
    outcome[:valid_n] = np.where(valid_out, status.astype(float), np.nan)
    return net_r, outcome


def _causal_rolling_mean_shift(
    pred_ev: np.ndarray,
    realized_r: np.ndarray,
    max_bars: int,
    window_bars: int = 5760,
    min_periods: int = 300,
    mask_active: np.ndarray | None = None,
    seed_pred: np.ndarray | None = None,
    seed_r: np.ndarray | None = None,
    seed_active: np.ndarray | None = None,
) -> np.ndarray:
    """Strictly causal additive residual-mean shift over resolved active past bars."""
    n = len(pred_ev)
    act = np.ones(n, dtype=bool) if mask_active is None else np.asarray(mask_active, dtype=bool)
    if seed_pred is not None and seed_r is not None and len(seed_pred) > 0:
        tail = min(len(seed_pred), window_bars + max_bars + 1)
        p_all = np.concatenate([seed_pred[-tail:], pred_ev])
        y_all = np.concatenate([seed_r[-tail:], realized_r])
        s_act = np.ones(tail, dtype=bool) if seed_active is None else np.asarray(seed_active[-tail:], dtype=bool)
        a_all = np.concatenate([s_act, act])
        offset = tail
    else:
        p_all = np.asarray(pred_ev, dtype=float)
        y_all = np.asarray(realized_r, dtype=float)
        a_all = act
        offset = 0

    lag = max_bars + 1
    p_s = pd.Series(p_all, dtype=float).shift(lag)
    y_s = pd.Series(y_all, dtype=float).shift(lag)
    a_s = pd.Series(a_all, dtype=bool).shift(lag, fill_value=False)

    valid = p_s.notna() & y_s.notna() & a_s
    resid_mean = (y_s - p_s).where(valid).rolling(window_bars, min_periods=min_periods).mean().to_numpy(dtype=float)
    delta = np.where(np.isfinite(resid_mean), np.clip(resid_mean, -0.50, 0.50), 0.0)
    adj_all = p_all + delta
    return adj_all[offset : offset + n]


class DirectNetRBracketModel(Model):
    """V3 Leakage-Free Direct Expected Net-R Multi-R:R Bracket Model.

    Directly learns E[R_net^(s,r) | x_t] against the EXACT simulator net R payoff
    (including TP, SL, exact market-close timeouts, and exact round-trip fees):
      - Zero timeout modeling mismatch (exact 1-to-1 match with simulator payoffs)
      - Zero survivorship-bias timeout hallucination on flat/low-vol bars
      - Purged 1D OLS affine calibration (a * pred + b, a in [0, 1]) on tradable bars
        automatically shrinks tail winner's curse AND silences uninformative heads (a <= 0.05 -> 0)
      - Fresh chronological 4-fold rolling-2y Stage-1 ATR-MAE quantile stops (zero 4-year lag)
      - Supports `sl_mode="fixed_atr"` for exact apples-to-apples baseline evaluation
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str = "15m",
        rr_ratios: tuple[float, ...] = DEFAULT_RR_RATIOS,
        max_bars: int = 32,
        sl_mode: str = "learned",
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
        min_ev_floor: float = 0.05,
        max_comm_r: float = 0.05,
        stand_down_on_neg_val: bool = True,
        results_path: str | Path | None = None,
        random_state: int = 42,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.rr_ratios = tuple(sorted(float(r) for r in rr_ratios))
        self.max_bars = int(max_bars)
        if sl_mode not in ("learned", "fixed_atr"):
            raise ValueError(f"Unsupported sl_mode={sl_mode!r}")
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
        self.random_state = int(random_state)
        self.refit_summary: list[dict] = []
        self.results_path = (
            Path(results_path)
            if results_path
            else PROJECT_DIR / "results" / f"direct_r_bracket_{symbol}_{timeframe}.json"
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
        self.heads: dict[tuple[str, float], xgb.XGBRegressor] = {}
        self.cal_affine: dict[tuple[str, float], tuple[float, float]] = {}  # (slope a, intercept b)
        self.head_std_r: dict[tuple[str, float], float] = {}
        self._seed_pred: dict[tuple[str, float], np.ndarray] = {}
        self._seed_r: dict[tuple[str, float], np.ndarray] = {}
        self._seed_act: dict[tuple[str, float], np.ndarray] = {}
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
            f"DirectNetR-V3 ({self.symbol} {self.timeframe}, sl_mode={self.sl_mode}, "
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
    ) -> dict[str, xgb.XGBRegressor]:
        sl_slice = slice(start_idx, end_idx)
        atr_s = atr_pct[sl_slice]
        age_days = (timestamps[end_idx - 1] - timestamps[sl_slice]).total_seconds().to_numpy(dtype=float) / 86400.0
        w_base = np.exp(-np.log(2.0) * np.maximum(age_days, 0.0) / 365.0)
        models = {}
        for side_name, mae_arr in (("long", mae_long), ("short", mae_short)):
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

    def _predict_calibrated_ev(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        n = len(X)
        n_rr = len(self.rr_ratios)
        out = {}
        for side_name in ("long", "short"):
            mat = np.zeros((n, n_rr), dtype=float)
            for j, rr in enumerate(self.rr_ratios):
                key = (side_name, rr)
                raw = self.heads[key].predict(X)
                a, b = self.cal_affine[key]
                mat[:, j] = a * raw + b
            out[side_name] = mat
        return out

    def _simulate_ev_backtest(
        self,
        ohlcv_df: pd.DataFrame,
        valid_mask: np.ndarray,
        sl_long_pct: np.ndarray,
        sl_short_pct: np.ndarray,
        ev_by_side: dict[str, np.ndarray],
        std_by_side: dict[str, np.ndarray],
        min_ev_r: float | np.ndarray,
        verbose: bool = False,
    ) -> dict:
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
        rejected_comm = 0
        rejected_ev = 0
        ignored_while_open = 0

        ev_long = ev_by_side["long"]
        ev_short = ev_by_side["short"]
        std_long = std_by_side["long"]
        std_short = std_by_side["short"]
        rr_arr = np.asarray(self.rr_ratios, dtype=float)
        c = self.commission

        for i in range(n):
            o_i, h_i, l_i, c_i = opens[i], highs[i], lows[i], closes[i]

            if pending is not None:
                side, rr, sl_d, exp_ev, sig_time, sig_regime = pending
                pending = None
                stop = o_i * (1.0 - side * sl_d)
                target = o_i * (1.0 + side * rr * sl_d)
                if stop > 0.0 and target > 0.0 and abs(o_i - stop) > 0.0 and exp_ev > min_ev_arr[i]:
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
                            "expected_ev_r": float(exp_ev),
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

            if comm_ok_l:
                for j, rr in enumerate(rr_arr):
                    ev_l = float(ev_long[i, j])
                    if ev_l > thr_i:
                        score_l = ev_l / max(float(std_long[i, j]), 0.25)
                        if score_l > best_score:
                            best_score = score_l
                            best_choice = (1, float(rr), float(sl_l), ev_l)

            if comm_ok_s:
                for j, rr in enumerate(rr_arr):
                    ev_s = float(ev_short[i, j])
                    if ev_s > thr_i:
                        score_s = ev_s / max(float(std_short[i, j]), 0.25)
                        if score_s > best_score:
                            best_score = score_s
                            best_choice = (-1, float(rr), float(sl_s), ev_s)

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
            "rejected_comm": int(rejected_comm),
            "rejected_ev": int(rejected_ev),
            "ignored_while_open": int(ignored_while_open),
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
            "sl_pct_summary": (
                {
                    "min": float(100.0 * min(t["sl_pct"] for t in trades)),
                    "median": float(100.0 * np.median([t["sl_pct"] for t in trades])),
                    "max": float(100.0 * max(t["sl_pct"] for t in trades)),
                }
                if trades
                else {}
            ),
            "by_rr": by_rr,
            "by_side": by_side,
            "by_regime": by_regime,
            "open_position": position,
            "trades": trades,
        }

        if verbose:
            print(f"\n=== {self.symbol} {self.timeframe} DirectNetR-V3 Exact-Payoff Backtest (OOS, sl_mode={self.sl_mode}) ===")
            print(f"Period: {result['start']} to {result['end']} | Candles: {n}")
            print(
                f"Settings: max_bars={self.max_bars} | c={c:.5f}/side | max_comm_r={self.max_comm_r:.2f}R | "
                f"min_ev_r={min_ev_label} | risk={self.risk_fraction*100:.1f}% | max_lev={self.max_leverage:.1f}x"
            )
            print(
                f"Rejections: stand_down={rejected_stand_down:,} | comm_gate={rejected_comm:,} | "
                f"ev_gate={rejected_ev:,} | while_open={ignored_while_open:,}"
            )
            wr_s = f"{result['win_rate_pct']:.2f}%" if result["win_rate_pct"] is not None else "N/A"
            pf_s = f"{profit_factor:.2f}" if profit_factor is not None else "N/A"
            print(
                f"Closed trades: {len(trades)} (W={wins}, L={losses}) | WR={wr_s} | PF={pf_s} | "
                f"Exits: TP={counts['TP']}, SL={counts['SL']}, TO={counts['TIMEOUT']}"
            )
            if trades:
                print(
                    f"Net Total: {total_r:+.2f}R (Comm={-comm_r:+.2f}R) | "
                    f"ExpEV={avg_exp_ev:+.3f}R vs RealR={avg_real_r:+.3f}R"
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
            raise ValueError("Not enough training candles for DirectNetR-V3 training")

        if self.recency_half_life_days > 0.0:
            age_days = (df.index[-1] - df.index).total_seconds().to_numpy(dtype=float) / 86400.0
            recency_w_all = np.exp(-np.log(2.0) * np.maximum(age_days, 0.0) / self.recency_half_life_days)
        else:
            recency_w_all = np.ones(n, dtype=float)

        # --- Stage 1 ---
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
                    X_full, atr_pct_full, feat_ok, mae_long, mae_short, df.index, tr_start, tr_end
                )
                sl_l_f, sl_s_f = self._predict_sl_with_models(
                    X_full.iloc[f_start:f_end], atr_pct_full[f_start:f_end], oof_models
                )
                sl_long_all[f_start:f_end] = sl_l_f
                sl_short_all[f_start:f_end] = sl_s_f

            final_end = n - self.sl_mae_bars
            final_start = max(0, final_end - sl_win_bars)
            self.sl_models = self._fit_stage1_pair(
                X_full, atr_pct_full, feat_ok, mae_long, mae_short, df.index, final_start, final_end
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

        # --- Stage 2: Direct Exact-Payoff Net-R Regressors + Purged OLS Affine Calibration ---
        if verbose:
            print(f"\n--- Stage 2: Training {2 * len(self.rr_ratios)} Direct Net-R Heads (Exact Simulator Payoff) ---")
        s2_slice = slice(s2_start, s2_end - gap)
        cal_slice = slice(cal_start, cal_end - gap)
        X_s2_base = X_full.iloc[s2_slice]
        X_cal_base = X_full.iloc[cal_slice]
        ok_s2 = feat_ok[s2_slice]
        ok_cal = feat_ok[cal_slice]

        comm_ok_all = {
            "long": np.isfinite(sl_long_all) & (sl_long_all > 0.0) & ((2.0 * c / np.maximum(sl_long_all, 1e-10)) <= self.max_comm_r),
            "short": np.isfinite(sl_short_all) & (sl_short_all > 0.0) & ((2.0 * c / np.maximum(sl_short_all, 1e-10)) <= self.max_comm_r),
        }
        r_all_by_head: dict[tuple[str, float], np.ndarray] = {}

        for side_name, side_int, sl_arr in (("long", 1, sl_long_all), ("short", -1, sl_short_all)):
            comm_s2 = comm_ok_all[side_name][s2_slice]
            comm_cal = comm_ok_all[side_name][cal_slice]
            for rr in self.rr_ratios:
                key_t = (side_name, rr)
                r_all, _ = _simulate_bracket_exact_net_r(
                    opens, highs, lows, closes, sl_arr, side=side_int, rr_ratio=rr, max_bars=self.max_bars, commission=c
                )
                r_all_by_head[key_t] = r_all

                m_fit = ok_s2 & np.isfinite(r_all[s2_slice]) & comm_s2
                if int(m_fit.sum()) < 1000:
                    m_fit = ok_s2 & np.isfinite(r_all[s2_slice])
                m_cal = ok_cal & np.isfinite(r_all[cal_slice]) & comm_cal
                if int(m_cal.sum()) < 400:
                    m_cal = ok_cal & np.isfinite(r_all[cal_slice])

                X_fit = X_s2_base.loc[m_fit]
                y_fit = r_all[s2_slice][m_fit]
                w_fit = recency_w_all[s2_slice][m_fit]
                X_cal = X_cal_base.loc[m_cal]
                y_cal = r_all[cal_slice][m_cal]

                reg = xgb.XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=300,
                    max_depth=3,
                    learning_rate=0.025,
                    min_child_weight=150,
                    subsample=0.70,
                    colsample_bytree=0.45,
                    reg_lambda=50.0,
                    reg_alpha=3.0,
                    eval_metric="rmse",
                    early_stopping_rounds=25,
                    tree_method="hist",
                    random_state=self.random_state,
                    n_jobs=4,
                )
                reg.fit(X_fit, y_fit, sample_weight=w_fit, eval_set=[(X_cal, y_cal)], verbose=False)

                raw_cal = reg.predict(X_cal)
                # 1D OLS affine calibration y_cal ~ a * raw_cal + b
                # Automatically shrinks overconfident tails (a < 1) AND silences dead heads (a <= 0.05 -> 0)!
                var_raw = float(np.var(raw_cal))
                cov_ry = float(np.cov(raw_cal, y_cal)[0, 1]) if len(y_cal) > 2 else 0.0
                if var_raw > 1e-10:
                    a_ols = cov_ry / var_raw
                else:
                    a_ols = 0.0
                a_cal = 0.0 if a_ols <= 0.05 else float(np.clip(a_ols, 0.0, 1.0))
                b_cal = float(np.mean(y_cal) - a_cal * np.mean(raw_cal))
                std_cal = float(max(np.std(y_cal), 0.25))
                cal_ic = float(np.corrcoef(raw_cal, y_cal)[0, 1]) if (var_raw > 1e-10 and np.std(y_cal) > 1e-10) else 0.0
                trees = int(getattr(reg, "best_iteration", 0) or 0)

                self.heads[key_t] = reg
                self.cal_affine[key_t] = (a_cal, b_cal)
                self.head_std_r[key_t] = std_cal

                key_s = f"{side_name}_{rr:g}"
                self.head_diagnostics[key_s] = {
                    "side": side_name,
                    "rr": rr,
                    "fit_samples": int(len(y_fit)),
                    "cal_samples": int(len(y_cal)),
                    "cal_mean_net_r": float(np.mean(y_cal)),
                    "cal_std_net_r": std_cal,
                    "cal_ols_slope_raw": float(a_ols),
                    "cal_affine_a": a_cal,
                    "cal_affine_b": b_cal,
                    "cal_ic": cal_ic,
                    "best_iteration": trees,
                }
                if verbose:
                    print(
                        f"  Head {side_name:5s} {rr:3.1f}:1 -> cal_mean={np.mean(y_cal):+.3f}R | "
                        f"cal_IC={cal_ic:+.4f} | affine_a={a_cal:.3f} (raw={a_ols:+.3f}) | std={std_cal:.2f}R | trees={trees:3d}"
                    )

        # --- Stage 3: Causal 60d Rolling Residual-Mean Calibration + Calmar_R Threshold ---
        if verbose:
            print(f"\n--- Stage 3: Validation EV Threshold Tuning (Direct Net-R, rolling_cal={self.rolling_cal_bars} bars) ---")
        val_df = df.iloc[val_start:]
        X_val = X_full.iloc[val_start:]
        val_ok = feat_ok[val_start:]
        sl_l_val = sl_long_all[val_start:]
        sl_s_val = sl_short_all[val_start:]

        ev_cal_raw = self._predict_calibrated_ev(X_full.iloc[cal_start:cal_end])
        ev_val_raw = self._predict_calibrated_ev(X_val)

        ev_val, std_val = {}, {}
        min_p_roll = max(150, self.rolling_cal_bars // 12)
        for side_name in ("long", "short"):
            adj_cols, std_cols = [], []
            comm_cal_s = comm_ok_all[side_name][cal_start:cal_end]
            comm_val_s = comm_ok_all[side_name][val_start:]
            for j, rr in enumerate(self.rr_ratios):
                key_t = (side_name, rr)
                r_all = r_all_by_head[key_t]
                r_cal_s = r_all[cal_start:cal_end]
                r_val_s = r_all[val_start:]
                adj = _causal_rolling_mean_shift(
                    ev_val_raw[side_name][:, j],
                    r_val_s,
                    max_bars=self.max_bars,
                    window_bars=self.rolling_cal_bars,
                    min_periods=min_p_roll,
                    mask_active=comm_val_s,
                    seed_pred=ev_cal_raw[side_name][:, j],
                    seed_r=r_cal_s,
                    seed_active=comm_cal_s,
                )
                adj_cols.append(adj)
                std_cols.append(np.full(len(val_df), self.head_std_r[key_t], dtype=float))
                self._seed_pred[key_t] = ev_val_raw[side_name][:, j].copy()
                self._seed_r[key_t] = r_val_s.copy()
                self._seed_act[key_t] = comm_val_s.copy()
            ev_val[side_name] = np.column_stack(adj_cols)
            std_val[side_name] = np.column_stack(std_cols)

        best_score = None
        best_ev_thr = max(self.min_ev_floor, 0.05)
        best_net_r = -np.inf
        val_grid = []
        ev_candidates = tuple(
            t for t in (0.04, 0.07, 0.10, 0.14, 0.18, 0.22)
            if t >= self.min_ev_floor - 1e-9
        ) or (self.min_ev_floor,)
        for thr in ev_candidates:
            bt_v = self._simulate_ev_backtest(
                val_df, val_ok, sl_l_val, sl_s_val, ev_val, std_val, min_ev_r=thr, verbose=False
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
            best_ev_thr = float("inf") if self.stand_down_on_neg_val else max(best_ev_thr, 0.08)

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
        if self._test_df is None or not self.heads:
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
        ev_te = {"long": np.full((n_te, n_rr), np.nan, dtype=float), "short": np.full((n_te, n_rr), np.nan, dtype=float)}
        std_te = {"long": np.full((n_te, n_rr), 1.0, dtype=float), "short": np.full((n_te, n_rr), 1.0, dtype=float)}
        min_ev_arr = np.full(n_te, self.selected_min_ev_r, dtype=float)

        if self.refit_days > 0:
            chunk_bars = max(1, int(round(pd.Timedelta(days=self.refit_days) / pd.Timedelta(self.timeframe))))
            chunks = [(s, min(n_te, s + chunk_bars)) for s in range(0, n_te, chunk_bars)]
        else:
            chunks = [(0, n_te)]

        min_p_roll = max(150, self.rolling_cal_bars // 12)
        self.refit_summary = []

        if len(chunks) > 1:
            print(f"\n=== Walk-Forward Periodic Refitting ({len(chunks)} chunks x {self.refit_days}d across {n_te} OOS bars) ===")

        for chunk_idx, (c_start, c_end) in enumerate(chunks):
            if chunk_idx > 0:
                hist_df = pd.concat([self._train_df, self._test_df.iloc[:c_start]])
                hist_X = pd.concat([self._X_train_full, X_te_all.iloc[:c_start]])
                hist_atr = np.concatenate([self._atr_pct_train, self._atr_pct_test[:c_start]])
                self._fit_on_history(hist_df, hist_X, hist_atr, verbose=False)

            chunk_df = self._test_df.iloc[c_start:c_end]
            chunk_X = X_te_all.iloc[c_start:c_end]
            chunk_atr = self._atr_pct_test[c_start:c_end]
            sl_l_c, sl_s_c = self._predict_sl_with_models(chunk_X, chunk_atr, self.sl_models)
            ev_c_raw = self._predict_calibrated_ev(chunk_X)

            o_c = chunk_df["Open"].to_numpy(dtype=float)
            h_c = chunk_df["High"].to_numpy(dtype=float)
            l_c = chunk_df["Low"].to_numpy(dtype=float)
            c_c = chunk_df["Close"].to_numpy(dtype=float)

            for side_name, side_int, sl_arr in (("long", 1, sl_l_c), ("short", -1, sl_s_c)):
                comm_c = np.isfinite(sl_arr) & (sl_arr > 0.0) & ((2.0 * c / np.maximum(sl_arr, 1e-10)) <= self.max_comm_r)
                adj_cols, std_cols = [], []
                for j, rr in enumerate(self.rr_ratios):
                    key_t = (side_name, rr)
                    r_c, _ = _simulate_bracket_exact_net_r(
                        o_c, h_c, l_c, c_c, sl_arr, side=side_int, rr_ratio=rr, max_bars=self.max_bars, commission=c
                    )
                    adj = _causal_rolling_mean_shift(
                        ev_c_raw[side_name][:, j],
                        r_c,
                        max_bars=self.max_bars,
                        window_bars=self.rolling_cal_bars,
                        min_periods=min_p_roll,
                        mask_active=comm_c,
                        seed_pred=self._seed_pred.get(key_t),
                        seed_r=self._seed_r.get(key_t),
                        seed_active=self._seed_act.get(key_t),
                    )
                    adj_cols.append(adj)
                    std_cols.append(np.full(len(chunk_df), self.head_std_r[key_t], dtype=float))
                ev_te[side_name][c_start:c_end] = np.column_stack(adj_cols)
                std_te[side_name][c_start:c_end] = np.column_stack(std_cols)

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

        # Stitched OOS IC & Calibration diagnostics on tradable bars
        opens = self._test_df["Open"].to_numpy(dtype=float)
        highs = self._test_df["High"].to_numpy(dtype=float)
        lows = self._test_df["Low"].to_numpy(dtype=float)
        closes = self._test_df["Close"].to_numpy(dtype=float)

        print("\n--- Test Head Direct Net-R Information Coefficient (IC) & Calibration (stitched OOS tradable bars) ---")
        ics = []
        for side_name, side_int, sl_arr in (("long", 1, sl_l_te), ("short", -1, sl_s_te)):
            comm_te = np.isfinite(sl_arr) & (sl_arr > 0.0) & ((2.0 * c / np.maximum(sl_arr, 1e-10)) <= self.max_comm_r)
            for j, rr in enumerate(self.rr_ratios):
                r_te, _ = _simulate_bracket_exact_net_r(
                    opens, highs, lows, closes, sl_arr, side=side_int, rr_ratio=rr, max_bars=self.max_bars, commission=c
                )
                pred = ev_te[side_name][:, j]
                m = te_ok & comm_te & np.isfinite(r_te) & np.isfinite(pred)
                pt, rt = pred[m], r_te[m]
                ic = float(np.corrcoef(pt, rt)[0, 1]) if (len(pt) > 20 and np.std(pt) > 1e-9 and np.std(rt) > 1e-9) else 0.0
                top10 = float(np.mean(rt[pt >= np.quantile(pt, 0.90)])) if len(pt) > 50 else float("nan")
                bot10 = float(np.mean(rt[pt <= np.quantile(pt, 0.10)])) if len(pt) > 50 else float("nan")
                key_s = f"{side_name}_{rr:g}"
                self.head_diagnostics[key_s].update(
                    {
                        "test_tradable_bars": int(len(pt)),
                        "test_ic": ic,
                        "test_mean_pred_r": float(np.mean(pt)) if len(pt) else None,
                        "test_mean_real_r": float(np.mean(rt)) if len(rt) else None,
                        "test_top10_real_r": top10,
                        "test_bot10_real_r": bot10,
                    }
                )
                ics.append(ic)
                print(
                    f"  Test {side_name:5s} {rr:3.1f}:1 -> IC={ic:+.4f} | mean pred={np.mean(pt):+.3f}R vs real={np.mean(rt):+.3f}R | "
                    f"top10%={top10:+.3f}R vs bot10%={bot10:+.3f}R"
                )
        self.avg_test_ic = float(np.nanmean(ics))
        print(f"  OVERALL MEAN TEST IC = {self.avg_test_ic:+.4f}")

        self.test_backtest = self._simulate_ev_backtest(
            self._test_df,
            te_ok,
            sl_l_te,
            sl_s_te,
            ev_te,
            std_te,
            min_ev_r=min_ev_arr if len(chunks) > 1 else self.selected_min_ev_r,
            verbose=True,
        )
        self.test_signals_df = self._test_df.copy()
        return self.test_backtest

    def evaluate(self):
        if self.test_backtest is None:
            raise RuntimeError("Run test() before evaluate()")

        from backtest import plot_backtest_positions

        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        tag = f"v3_{self.sl_mode}_{len(self.rr_ratios)}rr"
        chart_path = (
            self.results_path.parent
            / f"direct_r_bracket_{self.symbol}_{self.timeframe}_{tag}_{stamp}_trades.png"
        )
        plot_backtest_positions(self.test_signals_df, self.test_backtest, chart_path, self.timeframe, recent_days=30)

        bt_clean = {k: v for k, v in self.test_backtest.items() if k not in ("trades", "open_position")}
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
                "expected_ev_r": round(t["expected_ev_r"], 4),
                "gross_r": round(t["gross_r"], 4),
                "net_r": round(t["net_r"], 4),
            }
            for t in self.test_backtest.get("trades", [])
        ]

        config = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "model_version": "v3_direct_exact_net_r_fresh_s1",
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
            "commission": self.commission,
            "risk_fraction": self.risk_fraction,
            "max_leverage": self.max_leverage,
            "rolling_cal_bars": self.rolling_cal_bars,
            "refit_days": self.refit_days,
            "recency_half_life_days": self.recency_half_life_days,
            "min_ev_floor": self.min_ev_floor,
            "max_comm_r": self.max_comm_r,
            "stand_down_on_neg_val": self.stand_down_on_neg_val,
            "random_state": self.random_state,
        }

        record = {
            "symbol": self.symbol,
            "time": datetime.now().isoformat(timespec="seconds"),
            "git_commit": _get_git_revision(),
            "model": "DirectNetRBracketModel_V3",
            "config": config,
            "n_features": len(self.feature_names),
            "selected_min_ev_r": float(self.selected_min_ev_r) if np.isfinite(self.selected_min_ev_r) else None,
            "refit_summary": self.refit_summary,
            "train_time": self.train_time,
            "avg_test_ic": getattr(self, "avg_test_ic", None),
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
