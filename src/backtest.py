import backtrader as bt
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from strategy import compute_trade_ev
except ImportError:
    from src.strategy import compute_trade_ev


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


def prepare_multitimeframe_proxy_signals(higher_ohlcv, lower_ohlcv, backend,
                                         higher_timeframe, lower_timeframe):
    """Place each completed higher-timeframe signal on its final lower candle."""
    from models.XGBoost import _normalize_ohlcv

    high_interval = pd.Timedelta(higher_timeframe)
    low_interval = pd.Timedelta(lower_timeframe)
    if low_interval <= pd.Timedelta(0) or high_interval <= low_interval or high_interval % low_interval:
        raise ValueError('Higher timeframe must be an exact multiple of the lower timeframe')
    higher = _normalize_ohlcv(higher_ohlcv)
    lower = _normalize_ohlcv(lower_ohlcv)
    high_signals = prepare_price_signals(higher, backend)
    prediction = high_signals['predicted_price']
    displacement = prediction - higher['Close']
    signal_time = higher.index + high_interval
    signals = pd.DataFrame({'signal': high_signals['signal'].to_numpy(),
                            'proxy_displacement': displacement.to_numpy(),
                            'predicted_price': prediction.to_numpy(),
                            'prediction_origin_price': higher['Close'].to_numpy(),
                            'atr': high_signals['atr'].to_numpy()}, index=signal_time)
    result = lower.copy()
    events = signals.reindex(lower.index + low_interval)
    result['signal'] = events['signal'].fillna(0).to_numpy(dtype=int)
    result['proxy_displacement'] = events['proxy_displacement'].to_numpy(dtype=float)
    result['predicted_price'] = events['predicted_price'].to_numpy(dtype=float)
    result['prediction_origin_price'] = events['prediction_origin_price'].to_numpy(dtype=float)
    result['atr'] = events['atr'].to_numpy(dtype=float)
    result['horizon'] = backend.max_bars * int(high_interval / low_interval)
    return result


def plot_backtest_positions(frame, result, path, timeframe, recent_days=None, prediction_timeframe=None):
    """Plot actual closes, next-candle price forecasts and trade rectangles."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection
    from matplotlib.dates import date2num
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch, Rectangle

    interval = pd.Timedelta(timeframe)
    if frame.empty or interval <= pd.Timedelta(0):
        raise ValueError('A nonempty frame and positive timeframe are required for plotting')
    start = frame.index[0]
    end = frame.index[-1] + interval
    if recent_days is not None:
        start = max(start, end - pd.Timedelta(days=recent_days))
    fig = Figure(figsize=(16, 8), layout='constrained')
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    shown = frame.loc[frame.index + interval >= start]
    ax.plot(shown.index + interval, shown['Close'], color='#222222', linewidth=.8,
            label=f'Actual {timeframe} close', zorder=2)
    if 'predicted_price' in shown and 'prediction_origin_price' in shown:
        valid = (shown['predicted_price'].notna() & shown['prediction_origin_price'].notna())
        times = shown.index[valid] + interval
        forecast_times = times + pd.Timedelta(prediction_timeframe or timeframe)
        origin = shown.loc[valid, 'prediction_origin_price'].to_numpy(dtype=float)
        forecast = shown.loc[valid, 'predicted_price'].to_numpy(dtype=float)
        xs = date2num(times.to_pydatetime())
        future_xs = date2num(forecast_times.to_pydatetime())
        segments = np.stack((np.column_stack((xs, origin)), np.column_stack((future_xs, forecast))), axis=1)
        ax.add_collection(LineCollection(segments, colors='#356ac3', linewidths=.8, alpha=.7, zorder=3))
        ax.scatter(times, origin, s=12, color='#e18a11', alpha=.8,
                   label='Prediction time and higher-timeframe close', zorder=4)
        ax.scatter(forecast_times, forecast, s=12, color='#356ac3', alpha=.8,
                   label='Predicted next higher-timeframe close', zorder=4)

    def draw_rectangle(entry_time, exit_time, entry, exit_price, color, dashed=False):
        left = max(entry_time, start)
        right = min(exit_time, end)
        if right <= left:
            return
        bottom, top = sorted((entry, exit_price))
        rect = Rectangle((date2num(left.to_pydatetime()), bottom),
                         date2num(right.to_pydatetime()) - date2num(left.to_pydatetime()),
                         top - bottom, facecolor=color, edgecolor=color, alpha=.25,
                         linewidth=1, linestyle='--' if dashed else '-')
        ax.add_patch(rect)
        ax.scatter([entry_time, exit_time], [entry, exit_price], s=8, color=color, zorder=4)

    for trade in result['trades']:
        # OHLC data identifies the exit candle, not the intrabar exit time.
        exit_end = trade['exit_time'] + interval
        color = '#159166' if trade['net_pnl'] > 0 else '#d24a43'
        draw_rectangle(trade['entry_time'], exit_end, trade['entry'], trade['exit'], color)
    if result['open_position'] is not None:
        position = result['open_position']
        draw_rectangle(position['entry_time'], end, position['entry'], frame['Close'].iloc[-1], '#e3981a', True)
    ax.set_xlim(start, end)
    ax.set_title(f"{result['symbol']} backtest — actual price, next-close forecasts and positions\n"
                 'Position rectangle: entry to exit candle; height: entry to exit price')
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Price')
    ax.grid(alpha=.2)
    handles, labels = ax.get_legend_handles_labels()
    handles.extend([Patch(facecolor='#159166', alpha=.25, label='Profitable closed trade'),
                    Patch(facecolor='#d24a43', alpha=.25, label='Losing or flat closed trade'),
                    Patch(facecolor='#e3981a', alpha=.25, label='Open position (last close)')])
    ax.legend(handles=handles, loc='upper left')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    return path


def prepare_price_signals(ohlcv_df, backend):
    """Convert genuine next-close price forecasts into direction signals."""
    from models.XGBoost import _normalize_ohlcv

    frame = _normalize_ohlcv(ohlcv_df)
    prediction = backend.predict_next_candle(frame).reindex(frame.index)
    frame['predicted_price'] = prediction
    frame['signal'] = np.where(prediction.notna(), np.sign(prediction - frame['Close']), 0).astype(int)
    previous = frame['Close'].shift(1)
    true_range = pd.concat([(frame['High'] - frame['Low']),
                            (frame['High'] - previous).abs(),
                            (frame['Low'] - previous).abs()], axis=1).max(axis=1)
    frame['atr'] = true_range.rolling(14).mean()
    return frame


def prepare_direction_signals(ohlcv_df, backend):
    """Build signals from next-close direction probabilities."""
    from models.XGBoost import _normalize_ohlcv

    frame = _normalize_ohlcv(ohlcv_df)
    probability = backend.predict_direction_probability(frame).reindex(frame.index)
    frame['probability'] = probability
    frame['signal'] = np.where(probability.notna(),
                               np.where(probability >= backend.decision_threshold, 1, -1), 0)
    previous = frame['Close'].shift(1)
    true_range = pd.concat([(frame['High'] - frame['Low']),
                            (frame['High'] - previous).abs(),
                            (frame['Low'] - previous).abs()], axis=1).max(axis=1)
    frame['atr'] = true_range.rolling(14).mean()
    return frame


def prepare_adaptive_signals(ohlcv_df, backend):
    """Use each adaptive forecast's price, displacement and chosen horizon."""
    from models.XGBoost import _normalize_ohlcv

    frame = _normalize_ohlcv(ohlcv_df)
    forecasts = backend.predict_forecasts(frame).reindex(frame.index)
    for name in forecasts.columns:
        frame[name] = forecasts[name]
    frame['signal'] = np.where(frame['predicted_price'].notna(),
                               np.sign(frame['predicted_displacement']), 0).astype(int)
    previous = frame['Close'].shift(1)
    tr = pd.concat([(frame['High'] - frame['Low']), (frame['High'] - previous).abs(),
                    (frame['Low'] - previous).abs()], axis=1).max(axis=1)
    frame['atr'] = tr.rolling(14).mean()
    return frame


def run_bracket_backtest(symbol, timeframe, ohlcv_df, backend, start_time,
                         initial_cash=1000.0, position_value=1000.0, commission=0.0001,
                         prepared_signals=None, dynamic_forecast=False, reward_risk=2.0,
                         proxy_distance_mode=False, risk_fraction=None, stop_atr_multiplier=None,
                         min_ev_r=0.0, verbose=True, print_trades=True):
    """Trade bracket probabilities at the next open using conservative OHLC fills.

    The final open position is marked at the last close; it is not forcibly sold.
    Drawdown is measured at candle closes, not from intrabar extremes.
    """
    if initial_cash <= 0 or position_value <= 0 or not 0 <= commission < 1 or reward_risk <= 0:
        raise ValueError("Initial cash and position value must be positive; commission must be in [0, 1)")
    if (risk_fraction is None) != (stop_atr_multiplier is None):
        raise ValueError("risk_fraction and stop_atr_multiplier must be provided together")
    if risk_fraction is not None and (not 0 < risk_fraction < 1 or stop_atr_multiplier <= 0 or not proxy_distance_mode):
        raise ValueError("ATR risk sizing requires proxy mode, positive ATR multiplier and risk fraction between 0 and 1")
    start = pd.to_datetime(start_time, utc=True, errors='raise')
    frame = prepare_bracket_signals(ohlcv_df, backend) if prepared_signals is None else prepared_signals.copy()
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
    ignored_signals = rejected_entries = rejected_cost = rejected_unaffordable = rejected_ev = 0
    signals = frame['signal'].to_numpy(dtype=int)
    atr = frame['atr'].to_numpy(dtype=float)
    probabilities = (frame['probability'].to_numpy(dtype=float)
                     if 'probability' in frame.columns else np.full(len(frame), np.nan, dtype=float))
    if dynamic_forecast:
        forecast_prices = frame['predicted_price'].to_numpy(dtype=float)
        forecast_horizons = frame['horizon'].to_numpy(dtype=float)
    if proxy_distance_mode:
        proxy_displacements = frame['proxy_displacement'].to_numpy(dtype=float)
        forecast_horizons = frame['horizon'].to_numpy(dtype=float)
        low_interval = pd.Timedelta(timeframe)
    for i, (open_price, high, low, close) in enumerate(prices):
        if pending is not None:
            side, signal_atr, signal_time, forecast_price, forecast_horizon, signal_prob = pending
            pending = None
            if risk_fraction is not None:
                stop_distance = stop_atr_multiplier * signal_atr
                stop = open_price - side * stop_distance
                target = open_price + side * reward_risk * stop_distance
                distance = side * (target - open_price)
                max_bars = int(forecast_horizon)
            elif proxy_distance_mode:
                distance = side * forecast_price
                target = open_price + side * distance
                stop = open_price - side * distance / reward_risk
                max_bars = int(forecast_horizon)
            elif dynamic_forecast:
                target = forecast_price
                distance = side * (target - open_price)
                stop = open_price - side * distance / reward_risk
                max_bars = int(forecast_horizon)
            else:
                stop = open_price - side * backend.sl_atr * signal_atr
                target = open_price + side * backend.tp_atr * signal_atr
                distance = side * (target - open_price)
                max_bars = backend.max_bars
            if risk_fraction is not None:
                # Budget includes the entry fee and estimated exit fee at the stop.
                risk_per_unit = stop_distance + commission * (open_price + stop)
                quantity = balance * risk_fraction / risk_per_unit if risk_per_unit > 0 else 0.0
                order_value = quantity * open_price
            else:
                order_value = position_value
                quantity = order_value / open_price
            entry_fee = order_value * commission
            initial_risk = quantity * abs(open_price - stop)
            # Exact fee-adjusted EV gate in R-multiples when calibrated probability is available:
            # EV = ((|TP - e| / |SL - e|) + 1) * p - 1 - (p * c * (TP + e) + (1 - p) * c * (e + SL)) / |e - SL|
            expected_ev_r = None
            p_win = None
            ev_rejected = False
            if np.isfinite(signal_prob) and stop > 0 and target > 0 and abs(open_price - stop) > 0:
                p_win = float(signal_prob if side == 1 else (1.0 - signal_prob))
                p_win = float(np.clip(p_win, 0.0, 1.0))
                expected_ev_r = compute_trade_ev(open_price, target, stop, p_win, commission=commission)
                if min_ev_r is not None and expected_ev_r <= float(min_ev_r):
                    ev_rejected = True
                    rejected_ev += 1
            # A profitable target must at least cover the entry and exit fees.
            # This is a break-even condition, not a profitability guarantee.
            cost_exceeds_reward = ((dynamic_forecast or proxy_distance_mode) and distance > 0 and
                                   distance <= commission * (open_price + target))
            if cost_exceeds_reward:
                rejected_cost += 1
            has_immediate_entry = not proxy_distance_mode or frame.index[i] == signal_time
            if order_value + entry_fee > balance:
                rejected_unaffordable += 1
            if (has_immediate_entry and not cost_exceeds_reward and not ev_rejected and stop > 0 and target > 0
                    and distance > 0 and initial_risk > 0 and balance >= order_value + entry_fee):
                position = {'side': side, 'entry': open_price, 'entry_index': i,
                            'entry_time': frame.index[i], 'signal_time': signal_time,
                            'quantity': quantity, 'initial_risk': initial_risk, 'stop': stop,
                            'target': target, 'entry_fee': entry_fee, 'max_bars': max_bars,
                            'model_prob': p_win, 'expected_ev_r': expected_ev_r,
                            'planned_stop_loss': initial_risk + entry_fee + quantity * stop * commission}
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
                elif i - position['entry_index'] + 1 >= position['max_bars']:
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
                               'quantity': float(position['quantity']),
                               'planned_stop_loss': float(position['planned_stop_loss']),
                               'model_prob': position['model_prob'],
                               'expected_ev_r': position['expected_ev_r'],
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
                if proxy_distance_mode:
                    if ((risk_fraction is not None or np.isfinite(proxy_displacements[i])) and np.isfinite(forecast_horizons[i])
                            and forecast_horizons[i] >= 1 and forecast_horizons[i].is_integer()):
                        pending = (signals[i], atr[i], frame.index[i] + low_interval,
                                   proxy_displacements[i], forecast_horizons[i], probabilities[i])
                elif not dynamic_forecast:
                    pending = (signals[i], atr[i], frame.index[i], None, None, probabilities[i])
                elif (np.isfinite(forecast_prices[i]) and np.isfinite(forecast_horizons[i])
                      and forecast_horizons[i] >= 1 and forecast_horizons[i].is_integer()):
                    pending = (signals[i], atr[i], frame.index[i], forecast_prices[i], forecast_horizons[i], probabilities[i])

    wins = sum(t['net_pnl'] > 0 for t in trades)
    losses = sum(t['net_pnl'] < 0 for t in trades)
    counts = {reason: sum(t['reason'] == reason for t in trades) for reason in ('TP', 'SL', 'TIMEOUT')}
    gain_r = sum(max(t['gross_r'], 0.0) for t in trades)
    loss_r = sum(min(t['gross_r'], 0.0) for t in trades)
    commission_r = sum(t['commission_r'] for t in trades)
    total_r = sum(t['net_r'] for t in trades)
    ev_values = [t['expected_ev_r'] for t in trades if t['expected_ev_r'] is not None]
    avg_expected_ev_r = float(np.mean(ev_values)) if ev_values else None
    result = {
        'symbol': symbol, 'timeframe': timeframe, 'start': frame.index[0], 'end': frame.index[-1],
        'candles': len(frame), 'long_signals': int((signals == 1).sum()),
        'short_signals': int((signals == -1).sum()), 'ignored_signals': ignored_signals,
        'rejected_entries': rejected_entries, 'closed_trades': len(trades),
        'rejected_unaffordable': rejected_unaffordable,
        'rejected_cost': rejected_cost,
        'rejected_ev': rejected_ev,
        'avg_expected_ev_r': avg_expected_ev_r,
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
    if verbose:
        print(f"\n=== {symbol} {timeframe} forward bracket backtest ===")
        print(f"Period: {result['start']} to {result['end']} | Candles: {len(frame)}")
        if print_trades:
            print("Trade details (gain is net of commission):")
            for trade in trades:
                ev_note = f"   EV={trade['expected_ev_r']:+.3f}R" if trade['expected_ev_r'] is not None else ""
                print(f"[{trade['side'].title()}]: Open={trade['entry']:.8f}   "
                      f"TP={trade['target']:.8f}   SL={trade['stop']:.8f}   "
                      f"Close={trade['exit']:.8f}   Gain=${trade['net_pnl']:+.2f}{ev_note}")
            if position is not None:
                side = 'Long' if position['side'] == 1 else 'Short'
                open_net = unrealized - position['entry_fee']
                print(f"[{side}]: Open={position['entry']:.8f}   TP={position['target']:.8f}   "
                      f"SL={position['stop']:.8f}   Close=OPEN   "
                      f"Gain=${open_net:+.2f} (unrealized; entry commission included)")
        print(f"Signals: {result['long_signals']} long, {result['short_signals']} short; ignored while open: {ignored_signals}")
        if rejected_ev > 0:
            print(f"Rejected by EV gate (EV <= {min_ev_r:+.3f}R at c={commission:.5f}): {rejected_ev}")
        if dynamic_forecast or proxy_distance_mode:
            print(f"Rejected entries: {rejected_entries} (including {rejected_cost} whose target did not cover commission)")
            if risk_fraction is not None:
                print(f"Rejected for insufficient unleveraged cash: {rejected_unaffordable}")
        print(f"Closed trades: {len(trades)} | Wins: {wins} | Losses: {losses} | "
              f"Win rate: {result['win_rate_pct']:.2f}%" if trades else "Closed trades: 0 | Win rate: N/A")
        print(f"Exits: TP={counts['TP']}, SL={counts['SL']}, timeout={counts['TIMEOUT']}")
        ev_summary = f" | Avg expected EV: {avg_expected_ev_r:+.3f}R" if avg_expected_ev_r is not None else ""
        print(f"Closed trades — Gain: {gain_r:+.2f}R | Loss: {loss_r:+.2f}R | "
              f"Commission: {-commission_r:+.2f}R | Total: {total_r:+.2f}R{ev_summary}")
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
        ("commission", 0.0001),
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
    commission: float = 0.0001,
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
    commission: float = 0.0001,
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


def _compute_atr_series(ohlcv_df: pd.DataFrame, period: int = 14) -> pd.Series:
    previous = ohlcv_df["Close"].astype(float).shift(1)
    true_range = pd.concat(
        [
            (ohlcv_df["High"].astype(float) - ohlcv_df["Low"].astype(float)),
            (ohlcv_df["High"].astype(float) - previous).abs(),
            (ohlcv_df["Low"].astype(float) - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period, min_periods=period).mean()


def _simulate_bracket_hit_outcomes(
    ohlcv_df: pd.DataFrame,
    sides: np.ndarray,
    atr: np.ndarray,
    sl_atr: float,
    tp_atr: float,
    max_bars: int,
) -> np.ndarray:
    """Causal bracket hit labels matching run_bracket_backtest fills (next-open entry, SL-first)."""
    opens = ohlcv_df["Open"].to_numpy(dtype=float)
    highs = ohlcv_df["High"].to_numpy(dtype=float)
    lows = ohlcv_df["Low"].to_numpy(dtype=float)
    n = len(ohlcv_df)
    outcomes = np.full(n, np.nan, dtype=float)

    for i in range(n - 1):
        side = int(sides[i])
        if side == 0 or not np.isfinite(atr[i]) or atr[i] <= 0.0:
            continue
        entry = opens[i + 1]
        if not np.isfinite(entry) or entry <= 0.0:
            continue
        stop = entry - side * sl_atr * atr[i]
        target = entry + side * tp_atr * atr[i]
        if stop <= 0.0 or target <= 0.0:
            continue

        end_j = min(n, i + 1 + int(max_bars))
        hit = 0.0
        for j in range(i + 1, end_j):
            o_j, h_j, l_j = opens[j], highs[j], lows[j]
            if (side == 1 and o_j <= stop) or (side == -1 and o_j >= stop):
                hit = 0.0
                break
            if (side == 1 and o_j >= target) or (side == -1 and o_j <= target):
                hit = 1.0
                break
            stop_hit = l_j <= stop if side == 1 else h_j >= stop
            target_hit = h_j >= target if side == 1 else l_j <= target
            if stop_hit:
                hit = 0.0
                break
            if target_hit:
                hit = 1.0
                break
        outcomes[i] = hit
    return outcomes


def _build_bracket_calibrator_features(p_up: np.ndarray, sides: np.ndarray) -> np.ndarray:
    p_dir = np.where(sides == 1, p_up, 1.0 - p_up)
    p_clip = np.clip(p_dir, 1e-4, 1.0 - 1e-4)
    logit_dir = np.log(p_clip / (1.0 - p_clip))
    return np.column_stack([logit_dir, sides.astype(float)])


def run_consensus_ev_backtest(
    symbol: str,
    timeframe: str,
    train_ohlcv: pd.DataFrame,
    test_ohlcv: pd.DataFrame,
    val_preds_df: pd.DataFrame,
    test_preds_df: pd.DataFrame,
    commission: float = 0.0001,
    min_ev_r: float = 0.0,
    prob_col: str = "p_meta",
    initial_cash: float = 10000.0,
    position_value: float = 10000.0,
    candidate_brackets: tuple[tuple[float, float, int], ...] = (
        (1.0, 1.0, 24),
        (1.5, 1.5, 24),
        (2.0, 2.0, 24),
        (1.0, 2.0, 24),
        (1.5, 2.25, 24),
        (1.5, 3.0, 24),
        (2.0, 3.0, 48),
        (2.0, 4.0, 48),
    ),
    unanimous_options: tuple[bool, ...] = (False, True),
    min_val_trades: int = 15,
    print_trades: bool = False,
) -> dict:
    """Validation-calibrated EV-gated bracket backtest for the 3-model consensus.

    Step 1 (strictly on validation slice inside train_ohlcv):
      For each candidate (sl_atr, tp_atr, max_bars, require_unanimous), maps the
      consensus probability to calibrated P(hit TP before SL) using a purged
      first-half fit / second-half backtest evaluation under the exact fee-adjusted
      EV gate. Selects the highest net-R candidate without touching test_ohlcv.
    Step 2 (strictly out-of-sample on test_ohlcv):
      Refits the 1D bracket probability calibrator on all of validation, freezes
      the selected bracket + calibrator, and runs run_bracket_backtest() on test_ohlcv.
    """
    from types import SimpleNamespace
    from sklearn.linear_model import LogisticRegression
    from models.XGBoost import _normalize_ohlcv

    if val_preds_df is None or val_preds_df.empty:
        raise ValueError("val_preds_df is required to calibrate bracket hit probabilities causally")
    if test_preds_df is None or test_preds_df.empty:
        raise ValueError("test_preds_df is required for out-of-sample consensus backtesting")

    train_norm = _normalize_ohlcv(train_ohlcv)
    test_norm = _normalize_ohlcv(test_ohlcv)
    full_norm = pd.concat([train_norm, test_norm])
    full_norm = full_norm[~full_norm.index.duplicated(keep="last")].sort_index()
    full_atr = _compute_atr_series(full_norm, period=14)

    val_ohlcv = train_norm.loc[train_norm.index >= val_preds_df.index[0]].copy()
    val_ohlcv["atr"] = full_atr.reindex(val_ohlcv.index)
    val_aligned = val_preds_df.reindex(val_ohlcv.index)

    test_frame = test_norm.copy()
    test_frame["atr"] = full_atr.reindex(test_frame.index)
    test_aligned = test_preds_df.reindex(test_frame.index)

    def _prepare_signals(base_df, preds_df, require_unanimous, calibrator):
        frame = base_df.copy()
        p_up = preds_df[prob_col].to_numpy(dtype=float)
        unanimous = preds_df["unanimous"].fillna(False).to_numpy(dtype=bool)
        valid = np.isfinite(p_up) & np.isfinite(frame["atr"].to_numpy(dtype=float)) & (frame["atr"].to_numpy(dtype=float) > 0)
        sides = np.where(valid, np.where(p_up >= 0.5, 1, -1), 0).astype(int)
        if require_unanimous:
            sides = np.where(unanimous, sides, 0).astype(int)

        prob_for_bt = np.full(len(frame), np.nan, dtype=float)
        active = sides != 0
        if np.any(active):
            X_cal = _build_bracket_calibrator_features(p_up[active], sides[active])
            p_hit_tp = calibrator.predict_proba(X_cal)[:, 1]
            # run_bracket_backtest computes p_win = prob if side == 1 else (1 - prob)
            prob_for_bt[active] = np.where(sides[active] == 1, p_hit_tp, 1.0 - p_hit_tp)

        frame["signal"] = sides
        frame["probability"] = prob_for_bt
        return frame

    def _fit_calibrator(base_df, preds_df, sl_atr, tp_atr, max_bars, require_unanimous):
        p_up = preds_df[prob_col].to_numpy(dtype=float)
        unanimous = preds_df["unanimous"].fillna(False).to_numpy(dtype=bool)
        atr_vals = base_df["atr"].to_numpy(dtype=float)
        valid = np.isfinite(p_up) & np.isfinite(atr_vals) & (atr_vals > 0)
        sides = np.where(valid, np.where(p_up >= 0.5, 1, -1), 0).astype(int)
        if require_unanimous:
            sides = np.where(unanimous, sides, 0).astype(int)
        y_hit = _simulate_bracket_hit_outcomes(base_df, sides, atr_vals, sl_atr, tp_atr, max_bars)
        mask = (sides != 0) & np.isfinite(y_hit)
        if mask.sum() < 20 or len(np.unique(y_hit[mask])) < 2:
            return None
        X_cal = _build_bracket_calibrator_features(p_up[mask], sides[mask])
        cal = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        cal.fit(X_cal, y_hit[mask].astype(int))
        return cal

    n_val = len(val_ohlcv)
    val_split = int(n_val * 0.50)

    print(
        f"\n=== Consensus EV-Gated Bracket Selection on Validation "
        f"({symbol} {timeframe}, c={commission:.5f}/side, min_ev={min_ev_r:+.3f}R) ==="
    )
    val_summary = []
    best_cfg = None
    best_score = None

    for require_unanimous in unanimous_options:
        for sl_atr, tp_atr, max_bars in candidate_brackets:
            fit_end = max(50, val_split - int(max_bars))
            if fit_end >= val_split or val_split >= n_val - 20:
                continue
            cal_val = _fit_calibrator(
                val_ohlcv.iloc[:fit_end],
                val_aligned.iloc[:fit_end],
                sl_atr=sl_atr,
                tp_atr=tp_atr,
                max_bars=max_bars,
                require_unanimous=require_unanimous,
            )
            if cal_val is None:
                continue
            eval_df = val_ohlcv.iloc[val_split:]
            eval_preds = val_aligned.iloc[val_split:]
            sig_df = _prepare_signals(eval_df, eval_preds, require_unanimous, cal_val)
            cfg = SimpleNamespace(sl_atr=sl_atr, tp_atr=tp_atr, max_bars=int(max_bars))
            bt_val = run_bracket_backtest(
                symbol,
                timeframe,
                eval_df,
                cfg,
                eval_df.index[0],
                initial_cash=initial_cash,
                position_value=position_value,
                commission=commission,
                prepared_signals=sig_df,
                min_ev_r=min_ev_r,
                verbose=False,
            )
            entry = {
                "sl_atr": sl_atr,
                "tp_atr": tp_atr,
                "max_bars": int(max_bars),
                "require_unanimous": bool(require_unanimous),
                "val_closed_trades": int(bt_val["closed_trades"]),
                "val_rejected_ev": int(bt_val["rejected_ev"]),
                "val_win_rate_pct": bt_val["win_rate_pct"],
                "val_net_r": float(bt_val["closed_net_r"]),
                "val_return_pct": float(bt_val["return_pct"]),
                "val_max_dd_pct": float(bt_val["max_close_drawdown_pct"]),
            }
            val_summary.append(entry)
            wr_str = f"{bt_val['win_rate_pct']:.1f}%" if bt_val["win_rate_pct"] is not None else "N/A"
            print(
                f"  SL={sl_atr:.2f} TP={tp_atr:.2f} H={max_bars:2d} unan={str(require_unanimous):5s} | "
                f"trades={bt_val['closed_trades']:3d} (rej_ev={bt_val['rejected_ev']:4d}) | "
                f"WR={wr_str:>5s} | Net={bt_val['closed_net_r']:+6.2f}R | "
                f"Ret={bt_val['return_pct']:+6.2f}% | MaxDD={bt_val['max_close_drawdown_pct']:.2f}%"
            )

            eligible = bt_val["closed_trades"] >= min_val_trades
            score = (
                1 if eligible else 0,
                float(bt_val["closed_net_r"]),
                -float(bt_val["max_close_drawdown_pct"]),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_cfg = (sl_atr, tp_atr, max_bars, require_unanimous)

    if best_cfg is None:
        raise RuntimeError("No valid bracket configuration could be calibrated on the validation split")

    sel_sl, sel_tp, sel_bars, sel_unan = best_cfg
    print(
        f"\nSelected on validation -> SL={sel_sl:.2f} ATR, TP={sel_tp:.2f} ATR, "
        f"max_bars={sel_bars}, require_unanimous={sel_unan}"
    )

    # Refit calibrator on full validation slice (still strictly before test_ohlcv)
    final_cal = _fit_calibrator(
        val_ohlcv,
        val_aligned,
        sl_atr=sel_sl,
        tp_atr=sel_tp,
        max_bars=sel_bars,
        require_unanimous=sel_unan,
    )
    if final_cal is None:
        raise RuntimeError("Failed to refit final bracket calibrator on full validation slice")

    test_signals = _prepare_signals(test_frame, test_aligned, sel_unan, final_cal)
    final_cfg = SimpleNamespace(sl_atr=sel_sl, tp_atr=sel_tp, max_bars=int(sel_bars))
    test_result = run_bracket_backtest(
        symbol,
        timeframe,
        test_frame,
        final_cfg,
        test_frame.index[0],
        initial_cash=initial_cash,
        position_value=position_value,
        commission=commission,
        prepared_signals=test_signals,
        min_ev_r=min_ev_r,
        verbose=True,
        print_trades=print_trades,
    )
    test_result["selected_bracket"] = {
        "sl_atr": sel_sl,
        "tp_atr": sel_tp,
        "max_bars": int(sel_bars),
        "require_unanimous": bool(sel_unan),
        "prob_col": prob_col,
        "commission": float(commission),
        "min_ev_r": float(min_ev_r),
    }
    test_result["validation_grid"] = val_summary
    return test_result, test_signals
