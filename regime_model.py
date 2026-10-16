"""
The contract between the candle stream and any regime model.

  - Candle       : what the stream hands to the model, one at a time.
  - RegimeModel  : the interface every model must implement (learn + predict).
  - NullModel    : a do-nothing model, used when no model is chosen.

To add a model, create a new file, subclass RegimeModel and implement both methods:

    # linreg_model.py
    from regime_model import RegimeModel, Candle

    class LinearRegressionModel(RegimeModel):
        def learn(self, candle: Candle) -> None:
            ...
        def predict(self, candle: Candle):
            return "bullish"

then run it with:

    python stream_sim.py datapoints/XAU/1h_tmp.csv --model linreg_model:LinearRegressionModel
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from types import MappingProxyType
from sklearn.linear_model import LinearRegression

import pandas as pd

REGIMES = ("bullish", "bearish", "consolidating")


@dataclass(frozen=True)
class Candle:
    """One closed candle. Immutable: a model can read it but never change it."""
    index: int                 # position in the stream (0, 1, 2, ...)
    time: pd.Timestamp         # candle open time, UTC
    open: float
    high: float
    low: float
    close: float
    volume: float
    data: MappingProxyType     # read-only: every column of this row, by column name

    def __getitem__(self, column):
        return self.data[column]

    def get(self, column, default=None):
        return self.data.get(column, default)


class RegimeModel(ABC):
    """
    Interface for every regime model (linear regression, neural network, ...).

    The stream calls, for every candle in time order:
        model.learn(candle)            # 1. the candle that just closed goes in
        regime = model.predict(candle) # 2. the model's output for that candle

    The model only ever receives one candle at a time. If it needs history
    (past closes, indicators, swings, ...) it must store that itself.
    A subclass that does not implement both methods cannot be created
    (Python raises TypeError), so a half-finished model fails immediately.
    """

    @abstractmethod
    def learn(self, candle: Candle) -> None:
        """Update the model with the candle that just closed."""

    @abstractmethod
    def predict(self, candle: Candle):
        """Return the regime for this candle: one of REGIMES, or None if not ready yet."""


class NullModel(RegimeModel):
    def __init__(self):
        pass

    def learn(self, candle: Candle) -> None:
        pass

    def predict(self, candle: Candle):
        return None
