"""
Developed by Benyamin Eslami
Strategy: trading rules.

Every trade has a stop-loss (SL) and a take-profit (TP):
    SL distance = sl_atr x ATR            (beyond normal candle noise)
    TP distance = alpha x SL distance     (alpha = reward : risk)

Results are measured in R, where 1R = the amount lost when the SL is hit.
A win earns alpha R, a loss costs 1 R, and costs take c/SL R from every trade:

    expectancy_net = (1 + alpha) * (p - p0) - c / SL

    p   : win rate of the trades the model takes (measured on test data)
    p0  : win rate with no model, = 1 / (1 + alpha) for a market with no pattern
    c   : round-trip cost of one trade (fees x 2 + spread + slippage), in price units

The ML model gives q = P(TP is hit before SL) for the current candle.
A trade is worth taking only when its own expectancy is positive:

    q > p0 + (c / SL) / (1 + alpha)
"""


class Strategy:
    def __init__(self, alpha=2.0, sl_atr=1.0, cost_pct=0.001):
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if sl_atr <= 0:
            raise ValueError("sl_atr must be positive")
        if cost_pct < 0:
            raise ValueError("cost_pct cannot be negative")
        self.alpha = alpha          # reward : risk, TP distance = alpha x SL distance
        self.sl_atr = sl_atr        # SL distance in ATRs
        self.cost_pct = cost_pct    # round-trip cost as a fraction of price (0.001 = 0.1%)

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
        """c / SL: the share of one trade's risk that is eaten by costs."""
        cost = self.cost_pct * entry            # c, in price units
        sl_distance = self.sl_atr * atr         # SL, in price units
        return cost / sl_distance

    def break_even_win_rate(self):
        """p0 = 1 / (1 + alpha): the win rate of a trade with no information."""
        return 1.0 / (1.0 + self.alpha)

    def threshold(self, entry, atr, p0=None):
        """Smallest model probability q for which a trade has positive expectancy.

        p0 defaults to the theoretical 1 / (1 + alpha); pass the measured
        no-model win rate instead when it is known.
        """
        if p0 is None:
            p0 = self.break_even_win_rate()
        return p0 + self.cost_in_r(entry, atr) / (1.0 + self.alpha)

    def signal(self, q, entry, atr, p0=None):
        """'long' when the model's probability q clears the threshold, otherwise 'no trade'."""
        if q > self.threshold(entry, atr, p0):
            return "long"
        return "no trade"

    def expectancy(self, p, p0, cost_r):
        """Average result per trade in R: (1 + alpha) * (p - p0) - c / SL."""
        return (1.0 + self.alpha) * (p - p0) - cost_r
