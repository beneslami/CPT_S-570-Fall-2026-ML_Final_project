"""
Developed by Benyamin Eslami
Strategy: trading rules.
"""

class Strategy:
    def __init__(self, reward_risk=2.0):
        self.reward_risk = 1 / reward_risk

    def signal(self, *args, **kwargs):
        raise NotImplementedError

    def stop_loss(self, entry, distance, direction):
        if distance <= 0:
            raise ValueError("distance must be positive")
        if direction == "long":
            return entry - distance
        if direction == "short":
            return entry + distance
        raise ValueError("direction must be 'long' or 'short'")

    def take_profit(self, entry, stop_loss, direction):
        risk = abs(entry - stop_loss)
        if direction == "long":
            return entry + self.reward_risk * risk
        if direction == "short":
            return entry - self.reward_risk * risk
        raise ValueError("direction must be 'long' or 'short'")

    def risk_reward_ratio(self, entry, stop_loss, take_profit):
        risk = abs(entry - stop_loss)
        if risk == 0:
            raise ValueError("entry and stop loss cannot be equal")
        return abs(take_profit - entry) / risk
