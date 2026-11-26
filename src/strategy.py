"""
Developed by Benyamin Eslami
Strategy: trading rules.

Every trade has a stop-loss (SL) and a take-profit (TP):
    SL distance = sl_atr x ATR            (beyond normal candle noise)
    TP distance = alpha x SL distance     (alpha = reward : risk)

Results are measured in R, where 1R = |SL - e| (price distance to stop-loss).
With per-side broker commission rate c (e.g. 0.0001 = 0.01% per side) and model
win probability p = P(TP is hit before SL):

    EV = ((|TP - e| / |SL - e|) + 1) * p - 1
         - (p * c * (TP + e) + (1 - p) * c * (e + SL)) / |e - SL|

A trade is worth taking only when EV > 0 (or above a chosen min_ev_r margin).
"""


def compute_trade_ev(entry, tp, sl, p, commission=0.0001):
    """Exact fee-adjusted expected value in R-multiples for a long or short trade.

    Parameters
    ----------
    entry : float
        Entry price (e).
    tp : float
        Take-profit price (TP).
    sl : float
        Stop-loss price (SL).
    p : float
        Calibrated probability that TP is hit before SL, in [0, 1].
    commission : float, default 0.0001
        Broker commission rate per side (applied to entry notional and exit notional).
    """
    entry = float(entry)
    tp = float(tp)
    sl = float(sl)
    p = float(p)
    c = float(commission)

    risk = abs(entry - sl)
    if risk <= 0.0:
        raise ValueError("entry and stop loss cannot be equal")
    if not 0.0 <= p <= 1.0:
        raise ValueError("probability p must be in [0, 1]")
    if c < 0.0:
        raise ValueError("commission cannot be negative")

    reward_ratio = abs(tp - entry) / risk
    gross_ev = (reward_ratio + 1.0) * p - 1.0
    fee_ev = (p * c * (tp + entry) + (1.0 - p) * c * (entry + sl)) / risk
    return gross_ev - fee_ev


class Strategy:
    def __init__(self, alpha=2.0, sl_atr=1.0, cost_pct=None, commission=0.0001):
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if sl_atr <= 0:
            raise ValueError("sl_atr must be positive")
        if cost_pct is not None:
            if cost_pct < 0:
                raise ValueError("cost_pct cannot be negative")
            self.cost_pct = float(cost_pct)
            self.commission = self.cost_pct / 2.0
        else:
            if commission < 0:
                raise ValueError("commission cannot be negative")
            self.commission = float(commission)
            self.cost_pct = 2.0 * self.commission
        self.alpha = alpha          # reward : risk, TP distance = alpha x SL distance
        self.sl_atr = sl_atr        # SL distance in ATRs

    def stop_loss(self, entry, atr, direction):
        """SL price, sl_atr ATRs away from the entry on the losing side."""
        if atr <= 0:
            raise ValueError("atr must be positive")
        distance = self.sl_atr * atr
        if direction == "long":
            return entry - distance
        if direction == "short":
            return entry + distance
        raise ValueError("direction must be 'long' or 'short'")

    def take_profit(self, entry, stop_loss, direction):
        """TP price, alpha times the risk away from the entry on the winning side."""
        risk = abs(entry - stop_loss)
        if direction == "long":
            return entry + self.alpha * risk
        if direction == "short":
            return entry - self.alpha * risk
        raise ValueError("direction must be 'long' or 'short'")

    def risk_reward_ratio(self, entry, stop_loss, take_profit):
        """Reward / risk of a trade, e.g. 2.0 means the TP is twice as far as the SL."""
        risk = abs(entry - stop_loss)
        if risk == 0:
            raise ValueError("entry and stop loss cannot be equal")
        return abs(take_profit - entry) / risk

    def cost_in_r(self, entry, atr):
        """Approximate round-trip cost in R: (2 * c * e) / SL."""
        cost = self.cost_pct * entry
        sl_distance = self.sl_atr * atr
        return cost / sl_distance

    def trade_ev(self, entry, tp, sl, p):
        """Exact fee-adjusted EV in R-multiples using per-side commission."""
        return compute_trade_ev(entry, tp, sl, p, commission=self.commission)

    def break_even_win_rate(self, entry=None, tp=None, sl=None):
        """Minimum win probability p for EV = 0.

        When entry, tp, and sl are provided, solves the exact fee-adjusted EV = 0
        equation; otherwise returns the frictionless 1 / (1 + alpha).
        """
        if entry is None or tp is None or sl is None:
            return 1.0 / (1.0 + self.alpha)
        risk = abs(entry - sl)
        if risk <= 0.0:
            raise ValueError("entry and stop loss cannot be equal")
        c = self.commission
        numerator = risk + c * (entry + sl)
        denominator = abs(tp - entry) + risk - c * (tp - sl)
        if denominator <= 0.0:
            return 1.0
        return numerator / denominator

    def threshold(self, entry, atr, p0=None, direction="long"):
        """Smallest model probability q for which a trade has positive EV.

        When p0 is omitted, solves the exact fee-adjusted EV = 0 equation for the
        configured (sl_atr, alpha, commission). If a custom measured p0 is passed,
        falls back to the p0-shifted threshold.
        """
        if p0 is None:
            sl = self.stop_loss(entry, atr, direction)
            tp = self.take_profit(entry, sl, direction)
            return self.break_even_win_rate(entry=entry, tp=tp, sl=sl)
        return p0 + self.cost_in_r(entry, atr) / (1.0 + self.alpha)

    def signal(self, q, entry, atr, p0=None, direction="long", min_ev_r=0.0):
        """Return direction when trade EV clears min_ev_r, otherwise 'no trade'."""
        if p0 is None:
            sl = self.stop_loss(entry, atr, direction)
            tp = self.take_profit(entry, sl, direction)
            if self.trade_ev(entry, tp, sl, q) > min_ev_r:
                return direction
            return "no trade"
        if q > self.threshold(entry, atr, p0, direction=direction):
            return direction
        return "no trade"

    def expectancy(self, p, p0=None, cost_r=None, entry=None, tp=None, sl=None):
        """Average result per trade in R.

        Uses the exact fee-adjusted EV formula when (entry, tp, sl) are supplied;
        otherwise falls back to (1 + alpha) * (p - p0) - cost_r.
        """
        if entry is not None and tp is not None and sl is not None:
            return self.trade_ev(entry, tp, sl, p)
        if p0 is None:
            p0 = self.break_even_win_rate()
        if cost_r is None:
            cost_r = 0.0
        return (1.0 + self.alpha) * (p - p0) - cost_r
