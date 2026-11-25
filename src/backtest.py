import backtrader as bt
import numpy as np
import pandas as pd


def prepare_bracket_signals(ohlcv_df, backend):
    """Build causal classifier signals and signal-candle ATR from full history."""
    from models.XGBoost import _normalize_ohlcv, compute_features

    if backend.model is None or backend.X_train is None:
        raise RuntimeError("Train the XGBoost model before backtesting")
    data = _normalize_ohlcv(ohlcv_df)
    features = compute_features(data).loc[:, backend.X_train.columns]
    valid = features.notna().all(axis=1)
    probabilities = pd.Series(np.nan, index=data.index, dtype=float)
    if valid.any():
        raw = backend.model.predict_proba(features.loc[valid])[:, 1]
        probabilities.loc[valid] = backend.calibrator.predict_proba(raw)
    if ((probabilities.dropna() < 0) | (probabilities.dropna() > 1)).any():
        raise ValueError("Model returned probabilities outside [0, 1]")
    direction = pd.Series(0, index=data.index, dtype=int)
    bullish = probabilities >= backend.decision_threshold
    confidence = probabilities.where(bullish, 1.0 - probabilities)
    direction.loc[valid & (confidence >= backend.confidence_threshold) & bullish] = 1
    direction.loc[valid & (confidence >= backend.confidence_threshold) & ~bullish] = -1
    previous = data['Close'].shift(1)
    true_range = pd.concat([(data['High'] - data['Low']),
                            (data['High'] - previous).abs(),
                            (data['Low'] - previous).abs()], axis=1).max(axis=1)
    result = data.copy()
    result['probability'] = probabilities
    result['signal'] = direction
    result['atr'] = true_range.rolling(14).mean()
    return result


def run_bracket_backtest(symbol, timeframe, ohlcv_df, backend, start_time,
                         initial_cash=10000.0, position_value=100.0, commission=0.001):
    """Trade bracket probabilities at the next open using conservative OHLC fills.

    The final open position is marked at the last close; it is not forcibly sold.
    Drawdown is measured at candle closes, not from intrabar extremes.
    """
    if initial_cash <= 0 or position_value <= 0 or not 0 <= commission < 1:
        raise ValueError("Initial cash and position value must be positive; commission must be in [0, 1)")
    start = pd.to_datetime(start_time, utc=True, errors='raise')
    frame = prepare_bracket_signals(ohlcv_df, backend)
    frame = frame.loc[frame.index >= start].copy()
    if frame.empty:
        raise ValueError("No completed candles in the requested forward period")
    prices = frame[['Open', 'High', 'Low', 'Close']].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Forward OHLC prices must be finite and positive")
    if ((prices[:, 1] < np.maximum(prices[:, 0], prices[:, 3])) |
        (prices[:, 2] > np.minimum(prices[:, 0], prices[:, 3]))).any():
        raise ValueError("Forward candle high/low must contain its open and close")

    balance = peak = float(initial_cash)
    max_drawdown = total_fees = 0.0
    pending = position = None
    trades = []
    ignored_signals = rejected_entries = 0
    signals = frame['signal'].to_numpy(dtype=int)
    atr = frame['atr'].to_numpy(dtype=float)
    for i, (open_price, high, low, close) in enumerate(prices):
        if pending is not None:
            side, signal_atr, signal_time = pending
            pending = None
            stop = open_price - side * backend.sl_atr * signal_atr
            target = open_price + side * backend.tp_atr * signal_atr
            entry_fee = position_value * commission
            quantity = position_value / open_price
            initial_risk = quantity * abs(open_price - stop)
            if stop > 0 and target > 0 and initial_risk > 0 and balance >= position_value + entry_fee:
                position = {'side': side, 'entry': open_price, 'entry_index': i,
                            'entry_time': frame.index[i], 'signal_time': signal_time,
                            'quantity': quantity, 'initial_risk': initial_risk, 'stop': stop,
                            'target': target, 'entry_fee': entry_fee}
                balance -= entry_fee
                total_fees += entry_fee
            else:
                rejected_entries += 1

        if position is not None:
            side, stop, target = position['side'], position['stop'], position['target']
            exit_price = reason = None
            # A gap through a resting level fills at the next open.
            if (side == 1 and open_price <= stop) or (side == -1 and open_price >= stop):
                exit_price, reason = open_price, 'SL'
            elif (side == 1 and open_price >= target) or (side == -1 and open_price <= target):
                exit_price, reason = open_price, 'TP'
            else:
                stop_hit = low <= stop if side == 1 else high >= stop
                target_hit = high >= target if side == 1 else low <= target
                if stop_hit:
                    exit_price, reason = stop, 'SL'
                elif target_hit:
                    exit_price, reason = target, 'TP'
                elif i - position['entry_index'] + 1 >= backend.max_bars:
                    exit_price, reason = close, 'TIMEOUT'
            if reason is not None:
                gross = side * position['quantity'] * (exit_price - position['entry'])
                exit_fee = position['quantity'] * exit_price * commission
                net = gross - position['entry_fee'] - exit_fee
                balance += gross - exit_fee
                total_fees += exit_fee
                trades.append({'signal_time': position['signal_time'], 'entry_time': position['entry_time'],
                               'exit_time': frame.index[i], 'side': 'LONG' if side == 1 else 'SHORT',
                               'entry': float(position['entry']), 'exit': float(exit_price),
                               'stop': float(stop), 'target': float(target), 'reason': reason,
                               'initial_risk': float(position['initial_risk']),
                               'gross_pnl': float(gross), 'commission': float(position['entry_fee'] + exit_fee),
                               'net_pnl': float(net), 'gross_r': float(gross / position['initial_risk']),
                               'commission_r': float((position['entry_fee'] + exit_fee) / position['initial_risk']),
                               'net_r': float(net / position['initial_risk'])})
                position = None

        unrealized = (position['side'] * position['quantity'] * (close - position['entry'])
                      if position is not None else 0.0)
        equity = balance + unrealized
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, (peak - equity) / peak)
        if signals[i] and np.isfinite(atr[i]) and atr[i] > 0:
            if position is not None:
                ignored_signals += 1
            elif i + 1 < len(frame):
                pending = (signals[i], atr[i], frame.index[i])

    wins = sum(t['net_pnl'] > 0 for t in trades)
    losses = sum(t['net_pnl'] < 0 for t in trades)
    counts = {reason: sum(t['reason'] == reason for t in trades) for reason in ('TP', 'SL', 'TIMEOUT')}
    gain_r = sum(max(t['gross_r'], 0.0) for t in trades)
    loss_r = sum(min(t['gross_r'], 0.0) for t in trades)
    commission_r = sum(t['commission_r'] for t in trades)
    total_r = sum(t['net_r'] for t in trades)
    result = {
        'symbol': symbol, 'timeframe': timeframe, 'start': frame.index[0], 'end': frame.index[-1],
        'candles': len(frame), 'long_signals': int((signals == 1).sum()),
        'short_signals': int((signals == -1).sum()), 'ignored_signals': ignored_signals,
        'rejected_entries': rejected_entries, 'closed_trades': len(trades),
        'wins': wins, 'losses': losses, 'win_rate_pct': 100 * wins / len(trades) if trades else None,
        'exit_counts': counts, 'gross_closed_pnl': sum(t['gross_pnl'] for t in trades),
        'net_closed_pnl': sum(t['net_pnl'] for t in trades), 'total_commission': total_fees,
        'gross_gain_r': gain_r, 'gross_loss_r': loss_r,
        'closed_commission_r': commission_r, 'closed_net_r': total_r,
        'open_position': position, 'unrealized_gross_pnl': float(unrealized),
        'initial_cash': float(initial_cash), 'final_equity': float(equity),
        'return_pct': 100 * (equity / initial_cash - 1),
        'max_close_drawdown_pct': 100 * max_drawdown, 'trades': trades,
    }
    print(f"\n=== {symbol} {timeframe} forward bracket backtest ===")
    print(f"Period: {result['start']} to {result['end']} | Candles: {len(frame)}")
    print(f"Signals: {result['long_signals']} long, {result['short_signals']} short; ignored while open: {ignored_signals}")
    print(f"Closed trades: {len(trades)} | Wins: {wins} | Losses: {losses} | "
          f"Win rate: {result['win_rate_pct']:.2f}%" if trades else "Closed trades: 0 | Win rate: N/A")
    print(f"Exits: TP={counts['TP']}, SL={counts['SL']}, timeout={counts['TIMEOUT']}")
    print(f"Closed trades — Gain: {gain_r:+.2f}R | Loss: {loss_r:+.2f}R | "
          f"Commission: {-commission_r:+.2f}R | Total: {total_r:+.2f}R")
    print(f"Open position: {'yes' if position else 'no'} | Unrealized gross P&L: ${unrealized:.2f}")
    print(f"Initial: ${initial_cash:.2f} | Final equity: ${equity:.2f} | Return: {result['return_pct']:+.2f}% | Close-to-close max drawdown: {result['max_close_drawdown_pct']:.2f}%")
    return result


class NextCandleSignalData(bt.feeds.PandasData):
    """PandasData feed carrying next-candle displacement signals and TP/SL percentages."""

    lines = ("signal", "tp_pct", "sl_pct")
    params = (
        ("signal", "signal"),
        ("tp_pct", "tp_pct"),
        ("sl_pct", "sl_pct"),
    )


class NextCandleStrategy(bt.Strategy):
    """Enters Long (+1) / Short (-1) on signal and manages exits via TP%, SL%, or max_hold_bars."""

    params = (
        ("risk_per_trade", 25.0),
        ("max_stake", 2500.0),
        ("max_hold_bars", 12),
        ("commission", 0.001),
        ("min_tp_pct", 0.002),
        ("min_tp_fee_mult", 2.0),
    )

    def __init__(self):
        self.entry_order = None
        self.exit_order = None
        self._exit_reasons = {}
        self.last_closed_reason = None
        self.entry_bar = None
        self.entry_price = None
        self.entry_side = None
        self.entry_risk_usd = None
        self.entry_tp_pct = None
        self.entry_sl_pct = None
        self.tp_price = None
        self.sl_price = None
        self.exit_reason = None
        self.exit_fill_price = None
        self.total_commission = 0.0

    def _reset_active_trade(self):
        self.entry_bar = None
        self.entry_price = None
        self.entry_side = None
        self.entry_risk_usd = None
        self.entry_tp_pct = None
        self.entry_sl_pct = None
        self.tp_price = None
        self.sl_price = None
        self.exit_reason = None
        self.exit_fill_price = None

    def _submit_exit(self, reason: str, limit_price: float | None = None):
        if not self.position:
            return
        if self.exit_order is not None:
            if limit_price is not None and self.exit_reason == "TP":
                return
            self.cancel(self.exit_order)
            self.exit_order = None

        self.exit_reason = reason
        if limit_price is not None and np.isfinite(limit_price) and limit_price > 0.0:
            if self.position.size > 0:
                order = self.sell(
                    size=abs(self.position.size),
                    exectype=bt.Order.Limit,
                    price=limit_price,
                )
            elif self.position.size < 0:
                order = self.buy(
                    size=abs(self.position.size),
                    exectype=bt.Order.Limit,
                    price=limit_price,
                )
            else:
                order = None
        else:
            order = self.close()

        if order is not None:
            self._exit_reasons[order.ref] = reason
            self.exit_order = order

    def notify_order(self, order):
        if order.status == order.Completed:
            self.total_commission += float(order.executed.comm)

            if self.entry_order is not None and order.ref == self.entry_order.ref and self.position:
                fill_price = float(order.executed.price)
                size = abs(float(order.executed.size))
                tp_pct = self.entry_tp_pct
                sl_pct = self.entry_sl_pct

                if (
                    tp_pct is None
                    or sl_pct is None
                    or not (np.isfinite(tp_pct) and np.isfinite(sl_pct))
                    or tp_pct <= 0.0
                    or sl_pct <= 0.0
                    or fill_price <= 0.0
                    or size <= 0.0
                ):
                    self.entry_order = None
                    self._submit_exit("INVALID_BRACKET")
                    return

                self.entry_price = fill_price
                self.entry_bar = len(self)
                tp_dist = fill_price * tp_pct
                sl_dist = fill_price * sl_pct
                self.entry_risk_usd = size * sl_dist
                if self.position.size > 0:
                    self.entry_side = "LONG"
                    self.tp_price = fill_price + tp_dist
                    self.sl_price = fill_price - sl_dist
                elif self.position.size < 0:
                    self.entry_side = "SHORT"
                    self.tp_price = fill_price - tp_dist
                    self.sl_price = fill_price + sl_dist

            elif order.ref in self._exit_reasons:
                self.exit_fill_price = float(order.executed.price)
                self.last_closed_reason = self._exit_reasons[order.ref]

        if order.status in (
            order.Completed,
            order.Canceled,
            order.Margin,
            order.Rejected,
            order.Expired,
        ):
            if self.entry_order is not None and order.ref == self.entry_order.ref:
                self.entry_order = None
                if not self.position:
                    self._reset_active_trade()
            if self.exit_order is not None and order.ref == self.exit_order.ref:
                self.exit_order = None
            self._exit_reasons.pop(order.ref, None)

    def notify_trade(self, trade):
        if trade.isclosed:
            exit_px = (
                self.exit_fill_price
                if self.exit_fill_price is not None and np.isfinite(self.exit_fill_price)
                else float(self.data.close[0])
            )
            side = self.entry_side or ("LONG" if trade.size > 0 else "SHORT")
            opened = self.entry_price if self.entry_price is not None else exit_px
            tp_val = self.tp_price
            sl_val = self.sl_price

            reason = self.last_closed_reason or self.exit_reason
            if not reason:
                if side == "LONG":
                    reason = "TP" if exit_px >= opened else "SL"
                else:
                    reason = "TP" if exit_px <= opened else "SL"

            unit_r = float(self.p.risk_per_trade) if float(self.p.risk_per_trade) > 0.0 else 1.0
            r_gross = float(trade.pnl) / unit_r
            r_net = float(trade.pnlcomm) / unit_r
            tp_str = f"{tp_val:.2f}" if tp_val is not None else "N/A"
            sl_str = f"{sl_val:.2f}" if sl_val is not None else "N/A"

            capped_note = ""
            if (
                self.entry_risk_usd is not None
                and np.isfinite(self.entry_risk_usd)
                and abs(self.entry_risk_usd - unit_r) > 1e-6
            ):
                capped_note = f" [actual_risk=${self.entry_risk_usd:.2f}]"

            print(
                f"  [{side}] Open: {opened:.2f} | TP: {tp_str} | SL: {sl_str} | ExitPx: {exit_px:.2f} "
                f"| Exit: {reason} | Net: {r_net:+.2f}R (Gross: {r_gross:+.2f}R){capped_note}"
            )

            self.last_closed_reason = None
            if not self.position:
                self._reset_active_trade()

    def next(self):
        if self.entry_order is not None:
            return

        sig = int(round(float(self.data.signal[0]))) if np.isfinite(self.data.signal[0]) else 0
        price = float(self.data.close[0])
        if not np.isfinite(price) or price <= 0.0:
            return

        if self.position:
            bars_held = (len(self) - self.entry_bar) if self.entry_bar is not None else 0
            opposite_long = self.position.size > 0 and sig == -1
            opposite_short = self.position.size < 0 and sig == 1
            timed_out = self.p.max_hold_bars is not None and bars_held >= self.p.max_hold_bars

            if opposite_long or opposite_short:
                self._submit_exit("OPPOSITE_SIGNAL")
                return
            if timed_out:
                self._submit_exit("TIMEOUT")
                return

            if self.position.size > 0 and self.tp_price is not None and self.sl_price is not None:
                if float(self.data.low[0]) <= self.sl_price:
                    self._submit_exit("SL")
                    return
                if float(self.data.high[0]) >= self.tp_price:
                    self._submit_exit("TP", limit_price=self.tp_price)
                    return
            elif self.position.size < 0 and self.tp_price is not None and self.sl_price is not None:
                if float(self.data.high[0]) >= self.sl_price:
                    self._submit_exit("SL")
                    return
                if float(self.data.low[0]) <= self.tp_price:
                    self._submit_exit("TP", limit_price=self.tp_price)
                    return
            return

        if self.exit_order is not None or sig == 0:
            return

        tp_pct = float(self.data.tp_pct[0])
        sl_pct = float(self.data.sl_pct[0])
        round_trip_fee = 2.0 * float(self.p.commission)
        effective_min_tp = max(float(self.p.min_tp_pct), float(self.p.min_tp_fee_mult) * round_trip_fee)
        if (
            not (np.isfinite(tp_pct) and np.isfinite(sl_pct))
            or tp_pct <= 0.0
            or tp_pct < effective_min_tp
            or sl_pct <= 0.0
        ):
            return

        risk_per_unit = price * sl_pct
        if risk_per_unit <= 0.0:
            return

        size = min(self.p.risk_per_trade / risk_per_unit, self.p.max_stake / price)
        if size <= 0.0:
            return

        self.entry_tp_pct = tp_pct
        self.entry_sl_pct = sl_pct
        if sig == 1:
            self.entry_order = self.buy(size=size)
        elif sig == -1:
            self.entry_order = self.sell(size=size)


def _prepare_feed_frame(
    ohlcv_df: pd.DataFrame,
    pred_prices: pd.Series | pd.DataFrame,
    rr_ratio: float = 2.0,
    min_tp_pct: float = 0.002,
    max_tp_pct: float = 0.05,
    commission: float = 0.001,
    min_tp_fee_mult: float = 2.0,
    atr_pct_series: pd.Series | None = None,
    min_sl_atr_mult: float = 0.0,
) -> pd.DataFrame:
    """Build Backtrader feed from OHLCV and predicted next-candle closing prices."""
    base = ohlcv_df.copy()
    base.columns = [c.lower() for c in base.columns]
    close = base["close"].astype(float)

    if isinstance(pred_prices, pd.DataFrame):
        col = "predicted_price" if "predicted_price" in pred_prices.columns else pred_prices.columns[0]
        pred_series = pred_prices[col]
    else:
        pred_series = pred_prices

    pred_next = pred_series.reindex(base.index).astype(float).to_numpy()
    close_arr = close.to_numpy()

    round_trip_fee = 2.0 * float(commission)
    effective_min_tp = max(float(min_tp_pct), float(min_tp_fee_mult) * round_trip_fee)

    valid = np.isfinite(pred_next) & np.isfinite(close_arr) & (close_arr > 0.0)
    raw_move_pct = np.where(
        valid,
        np.abs(pred_next - close_arr) / np.maximum(close_arr, 1e-8),
        0.0,
    )
    if np.any(valid):
        q50, q75, q90, q99 = np.quantile(raw_move_pct[valid], [0.50, 0.75, 0.90, 0.99])
        print(
            f"Predicted move % quantiles -> p50={q50*100:.3f}%, p75={q75*100:.3f}%, "
            f"p90={q90*100:.3f}%, p99={q99*100:.3f}% (min_tp={effective_min_tp*100:.3f}%)"
        )

    tp_pct = np.clip(raw_move_pct, 0.0, max_tp_pct)
    sl_pct = np.where(tp_pct > 0.0, tp_pct / max(rr_ratio, 1e-6), 0.0)

    if atr_pct_series is not None and float(min_sl_atr_mult) > 0.0:
        atr_vals = atr_pct_series.reindex(base.index).astype(float).to_numpy()
        atr_floor_sl = np.where(
            np.isfinite(atr_vals) & (atr_vals > 0.0),
            float(min_sl_atr_mult) * atr_vals,
            0.0,
        )
        sl_pct = np.maximum(sl_pct, atr_floor_sl)

    signal = np.zeros(len(base), dtype=int)
    tradeable = (
        valid
        & (tp_pct >= effective_min_tp)
        & (tp_pct > 0.0)
        & (sl_pct > 0.0)
    )
    signal[tradeable & (pred_next > close_arr)] = 1
    signal[tradeable & (pred_next < close_arr)] = -1

    tp_pct = np.where(signal != 0, tp_pct, 0.0)
    sl_pct = np.where(signal != 0, sl_pct, 0.0)

    return pd.DataFrame(
        {
            "open": base["open"].astype(float),
            "high": base["high"].astype(float),
            "low": base["low"].astype(float),
            "close": close,
            "volume": base["volume"].astype(float),
            "signal": signal,
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
        },
        index=base.index,
    )


def run_displacement_backtest(
    symbol: str,
    timeframe: str,
    ohlcv_df: pd.DataFrame,
    pred_df: pd.Series | pd.DataFrame,
    initial_cash: float = 10000.0,
    risk_per_trade: float = 25.0,
    max_stake: float = 2500.0,
    max_hold_bars: int = 12,
    commission: float = 0.001,
    rr_ratio: float = 2.0,
    min_tp_pct: float = 0.002,
    max_tp_pct: float = 0.05,
    min_tp_fee_mult: float = 2.0,
    atr_pct_series: pd.Series | None = None,
    min_sl_atr_mult: float = 0.0,
) -> dict:
    print(f"\n=== Backtesting {symbol} {timeframe} Next-Candle XGBoost Model ===")
    feed_df = _prepare_feed_frame(
        ohlcv_df,
        pred_df,
        rr_ratio=rr_ratio,
        min_tp_pct=min_tp_pct,
        max_tp_pct=max_tp_pct,
        commission=commission,
        min_tp_fee_mult=min_tp_fee_mult,
        atr_pct_series=atr_pct_series,
        min_sl_atr_mult=min_sl_atr_mult,
    )
    sig_counts = feed_df["signal"].value_counts().to_dict()
    print(f"Signal distribution: {sig_counts} (bars={len(feed_df)})")

    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        NextCandleStrategy,
        risk_per_trade=risk_per_trade,
        max_stake=max_stake,
        max_hold_bars=max_hold_bars,
        commission=commission,
        min_tp_pct=min_tp_pct,
        min_tp_fee_mult=min_tp_fee_mult,
    )
    cerebro.adddata(NextCandleSignalData(dataname=feed_df, openinterest=None))

    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    results = cerebro.run()
    strat = results[0]
    final_value = cerebro.broker.getvalue()
    dd = strat.analyzers.drawdown.get_analysis()
    trades = strat.analyzers.trades.get_analysis()

    closed = trades.total.closed if "total" in trades and "closed" in trades.total else 0
    won = trades.won.total if "won" in trades and "total" in trades.won else 0
    lost = trades.lost.total if "lost" in trades and "total" in trades.lost else 0
    win_rate = (won / closed * 100.0) if closed > 0 else 0.0
    ret_pct = (final_value - initial_cash) / initial_cash * 100.0
    max_dd = dd.max.drawdown if hasattr(dd, "max") else 0.0

    summary = {
        "symbol": symbol,
        "timeframe": timeframe,
        "initial_cash": initial_cash,
        "final_value": final_value,
        "return_pct": ret_pct,
        "max_drawdown_pct": max_dd,
        "closed_trades": closed,
        "won_trades": won,
        "lost_trades": lost,
        "win_rate_pct": win_rate,
        "total_commission": strat.total_commission,
    }
    print(
        f"Closed Trades: {closed} (W={won}, L={lost}, WinRate={win_rate:.1f}%) | "
        f"Return: {ret_pct:+.2f}% | MaxDD: {max_dd:.2f}% | Comm: ${strat.total_commission:.2f}"
    )
    return summary
