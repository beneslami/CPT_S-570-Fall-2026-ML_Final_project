"""A7-4: 72-hour GRU for next hourly candle body category.

The two input DataFrames must be in chronological order and nonoverlapping; the first is
split internally into training and validation (8:1) and the second is test.
"""

import copy
import json
import pathlib
import random
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from ML_interface import Model


# PyTorch network architecture; separated from the higher-level Model wrapper.
# Input tensor: (batch_size, 72 time steps, 8 features).
class _GRUNetwork(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, dense_size=32):
        super().__init__()
        # GRU learns temporal dependencies across the 72 past hourly candles.
        # batch_first=True means input dimensions are (batch, time, features).
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        # Map final GRU state to four unnormalized class scores (logits).
        # CrossEntropyLoss applies the appropriate softmax internally.
        self.classifier = nn.Sequential(nn.Linear(hidden_size, dense_size), nn.ReLU(), nn.Linear(dense_size, 4))

    def forward(self, x):
        # h[-1] contains the final hidden state of the last GRU layer.
        _, h = self.gru(x)
        return self.classifier(h[-1])


class RNNGRU(Model):
    """Train once, test once, and append statistics to a JSON results file."""

    # A7-4 feature set: per-candle price shape, return, volatility, and volume.
    FEATURES = [
        "body_return", "upper_wick", "lower_wick", "close_return",
        "candle_range", "volatility_24", "avg_body_24", "relative_volume",
    ]
    # Class indexes are also used for direction and magnitude metrics below.
    CLASS_NAMES = ["Big bearish", "Small bearish", "Small bullish", "Big bullish"]

    def __init__(self, symbol: str, timeframe: str = "1h", random_state: int = 42,
                 sequence_length: int = 72, hidden_size: int = 64, dense_size: int = 32,
                 batch_size: int = 64, learning_rate: float = 0.001,
                 max_epochs: int = 50, patience: int = 7, results_path=None):
        # The sequence length and rolling periods are measured in hourly candles.
        if timeframe != "1h":
            raise ValueError("A7-4 uses 72 hourly candles; timeframe must be '1h'.")
        self.symbol, self.timeframe = symbol, timeframe
        self.random_state = random_state
        self.sequence_length = sequence_length
        self.hidden_size, self.dense_size = hidden_size, dense_size
        self.batch_size, self.learning_rate = batch_size, learning_rate
        self.max_epochs, self.patience = max_epochs, patience
        self.features = list(self.FEATURES)
        project_dir = pathlib.Path(__file__).resolve().parents[2]
        self.results_path = (pathlib.Path(results_path) if results_path else
                             project_dir / "results" / f"RNNGRU_{symbol}_{timeframe}.json")
        # Standardize feature scales using training data only (avoid leakage).
        self.scaler = StandardScaler()
        # Use an NVIDIA GPU when CUDA is available; otherwise train on CPU.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _GRUNetwork(len(self.features), hidden_size, dense_size).to(self.device)
        self.train_time = 0.0
        self.training_acc = 0.0
        self.test_acc = 0.0
        self.history = []
        self.best_epoch = None
        self.best_val_loss = None
        self.body_threshold = None
        self.splits = {}
        self.y_pred = None
        self.y_test = None
        self.n_train = self.n_val = self.n_test = 0

    @staticmethod
    def _enrich(frame):
        """Validate raw OHLCV candles and derive all eight causal A7-4 features.

        Each row uses information available at that candle close or earlier.
        Hourly timestamp gaps reset rolling calculations and later sequences.
        """
        frame = frame.copy().sort_values("time").reset_index(drop=True)
        required = ["time", "open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise ValueError(f"Missing OHLCV columns: {missing}")
        frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="raise")
        for col in required[1:]:
            frame[col] = pd.to_numeric(frame[col], errors="raise")
        if frame.time.duplicated().any() or not frame.time.is_monotonic_increasing:
            raise ValueError("Time must be sorted and timestamps unique")
        if not np.isfinite(frame[required[1:]].to_numpy(dtype=float)).all():
            raise ValueError("OHLCV contains missing/nonfinite values")
        if (frame[["open", "high", "low", "close"]] <= 0).any().any() or (frame.volume < 0).any():
            raise ValueError("Invalid prices or volume")
        if ((frame.high < frame[["open", "close", "low"]].max(axis=1)) |
                (frame.low > frame[["open", "close", "high"]].min(axis=1))).any():
            raise ValueError("Inconsistent OHLC rows")
        # A non-hourly jump indicates missing candle(s); never bridge a gap
        # when calculating rolling features or forming a 72-hour sequence.
        frame["segment_id"] = frame.time.diff().ne(pd.Timedelta(hours=1)).cumsum()
        # Signed candle body: positive = bullish, negative = bearish.
        frame["body_return"] = (frame.close - frame.open) / frame.open
        frame["upper_wick"] = (frame.high - frame[["open", "close"]].max(axis=1)) / frame.open
        frame["lower_wick"] = (frame[["open", "close"]].min(axis=1) - frame.low) / frame.open
        # Price change across consecutive closing prices, within one segment.
        frame["close_return"] = frame.groupby("segment_id").close.transform(
            lambda s: s.pct_change(fill_method=None))
        frame["candle_range"] = (frame.high - frame.low) / frame.open
        # Rolling standard deviation of the past 24 close-to-close returns.
        # First 24+ rows per segment may be NaN until enough history exists.
        frame["volatility_24"] = frame.groupby("segment_id").close_return.transform(
            lambda s: s.rolling(24, min_periods=24).std())
        frame["avg_body_24"] = frame.body_return.abs().groupby(frame.segment_id).transform(
            lambda s: s.rolling(24, min_periods=24).mean())
        # Relative volume compares this candle with its 24-hour mean.
        average_volume = frame.groupby("segment_id").volume.transform(
            lambda s: s.rolling(24, min_periods=24).mean())
        frame["relative_volume"] = frame.volume / average_volume.replace(0, np.nan)
        return frame

    def feature_engineering(self, train_data, test_data):
        """Prepare 80/10/10 chronological sequences, with training-only scaling.

        main.py passes the initial 90% as train_data and the last 10% as
        test_data. We reserve the last 1/9 of train_data for validation.
        """
        if len(train_data) < 100 or len(test_data) < 100:
            raise ValueError("Not enough data for GRU sequence construction")
        train_times = pd.to_datetime(train_data["time"], utc=True)
        test_times = pd.to_datetime(test_data["time"], utc=True)
        if train_times.max() >= test_times.min():
            raise ValueError("Training and test periods overlap. Split the CSV chronologically in main.py.")
        combined = pd.concat([train_data, test_data], ignore_index=True)
        frame = self._enrich(combined)
        train_end = int(len(train_data) * (8 / 9))
        val_end = len(train_data)
        if train_end <= self.sequence_length or val_end - train_end <= self.sequence_length:
            raise ValueError("Insufficient training/validation history")

        # The training median absolute body return defines small versus big.
        # Never use validation/test returns to choose this threshold.
        self.body_threshold = float(frame.body_return.iloc[:train_end].abs().median())
        r = frame.body_return.to_numpy()
        # Target is the NEXT candle category at index i, not the last
        # candle inside its historical input window.
        # 0 big bearish; 1 small bearish; 2 small bullish; 3 big bullish.
        labels = np.select([
            r < -self.body_threshold,
            (r < 0) & (r >= -self.body_threshold),
            (r >= 0) & (r <= self.body_threshold),
            r > self.body_threshold,
        ], [0, 1, 2, 3]).astype(np.int64)

        # Train-only normalization: validation/test are transformed with
        # exactly the scaler statistics learned from training candles.
        train_features = frame.loc[:train_end - 1, self.features]
        self.scaler.fit(train_features.replace([np.inf, -np.inf], np.nan).dropna())
        # Rolling warmup rows may be NaN; exclude windows containing them.
        valid = np.isfinite(frame[self.features].to_numpy(dtype=float)).all(axis=1)
        values = np.zeros((len(frame), len(self.features)), dtype=np.float32)
        values[valid] = self.scaler.transform(frame.loc[valid, self.features]).astype(np.float32)
        # Train / validation / test are chronological and nonoverlapping.
        split_regions = [("train", 0, train_end), ("val", train_end, val_end),
                         ("test", val_end, len(frame))]
        segments = frame.segment_id.to_numpy()
        self.splits = {}
        for name, start, end in split_regions:
            xs, ys = [], []
            # For target index i, input is [i-72, ..., i-1].
            # The current target candle is NEVER included in its inputs.
            for i in range(start + self.sequence_length, end):
                left = i - self.sequence_length
                # Skip sequences spanning missing timestamps or invalid features.
                # segment_id increases at every gap, so matching endpoints
                # guarantees the whole interval has the same segment ID.
                if segments[left] != segments[i] or not valid[left:i].all():
                    continue
                xs.append(values[left:i]); ys.append(labels[i])
            if not xs:
                raise ValueError(f"No valid {name} sequences. Check hourly gaps and split size.")
            self.splits[name] = (np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64))
        self.n_train = len(self.splits["train"][1])
        self.n_val = len(self.splits["val"][1])
        self.n_test = len(self.splits["test"][1])
        self.y_test = self.splits["test"][1]
        print(f"GRU sequences: train={self.n_train}, val={self.n_val}, test={self.n_test}")
        print(f"Training-only big-body threshold: {self.body_threshold*100:.4f}%")

    def _loader(self, name, shuffle=False):
        """Wrap one data split in mini-batches for efficient PyTorch training."""
        x, y = self.splits[name]
        # Shuffle only the training windows, not validation/test windows.
        # Randomized batch order does not shuffle candle order *inside* a window.
        generator = torch.Generator().manual_seed(self.random_state)
        return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                          batch_size=self.batch_size, shuffle=shuffle, generator=generator)

    @torch.no_grad()
    def _score(self, name):
        """Return mean cross-entropy, four-class accuracy, and predictions."""
        # Evaluation mode and no_grad disable training-only behavior and
        # avoid building gradient graphs during scoring.
        self.model.eval()
        predictions = []
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        for x, y in self._loader(name):
            logits = self.model(x.to(self.device))
            total_loss += criterion(logits, y.to(self.device)).item() * len(y)
            # Choose the class with the highest logit for each sequence.
            predictions.append(logits.argmax(dim=1).cpu().numpy())
        y_true = self.splits[name][1]
        y_pred = np.concatenate(predictions)
        return total_loss / len(y_true), accuracy_score(y_true, y_pred), y_pred

    def train(self):
        """Optimize GRU once; use validation loss for early stopping."""
        if not self.splits:
            raise RuntimeError("Call feature_engineering before train")
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        # Reinitialize AFTER setting random seeds for repeatability.
        self.model = _GRUNetwork(len(self.features), self.hidden_size, self.dense_size).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        best_loss, best_weights, wait = float("inf"), None, 0
        started = time.perf_counter()
        self.history = []
        # One epoch processes all training sequences (in mini-batches).
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            total_loss = 0.0
            for x, y in self._loader("train", shuffle=True):
                x, y = x.to(self.device), y.to(self.device)
                # Standard training step: reset gradients, forward pass,
                # loss, backpropagation, clipping, then optimizer update.
                optimizer.zero_grad()
                loss = criterion(self.model(x), y)
                loss.backward()
                # Prevent unusually large gradients from destabilizing the GRU.
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * len(y)
            # Validation data selects the checkpoint; test data is not
            # consulted during optimization or early stopping.
            val_loss, val_acc, _ = self._score("val")
            train_loss = total_loss / self.n_train
            self.history.append({"epoch": epoch, "train_loss": train_loss,
                                 "val_loss": val_loss, "val_accuracy": val_acc})
            print(f"Epoch {epoch:02d} | Train Loss {train_loss:.4f} | "
                  f"Val Loss {val_loss:.4f} | Val Acc {val_acc:.4f}")
            # Save a COPY of the best weights, not a reference that will
            # continue changing during subsequent optimizer steps.
            if val_loss < best_loss - 1e-5:
                best_loss, best_weights, self.best_epoch, wait = (
                    val_loss, copy.deepcopy(self.model.state_dict()), epoch, 0)
            else:
                wait += 1
            # Stop after seven consecutive epochs without meaningful gain.
            if wait >= self.patience:
                break
        self.train_time = time.perf_counter() - started
        self.best_val_loss = float(best_loss)
        # Restore best validation checkpoint before final test evaluation.
        self.model.load_state_dict(best_weights)
        _, self.training_acc, _ = self._score("train")

    def test(self):
        """Evaluate the held-out test set and derive direction/size metrics."""
        if self.best_epoch is None:
            raise RuntimeError("Call train before test")
        self.test_loss, self.test_acc, self.y_pred = self._score("test")
        # Classes 0/1 are bearish; 2/3 are bullish.
        direction_true, direction_pred = self.y_test >= 2, self.y_pred >= 2
        # Classes 0/3 are big; 1/2 are small.
        magnitude_true = np.isin(self.y_test, [0, 3])
        magnitude_pred = np.isin(self.y_pred, [0, 3])
        self.direction_acc = accuracy_score(direction_true, direction_pred)
        self.magnitude_acc = accuracy_score(magnitude_true, magnitude_pred)
        # Macro-F1 gives equal importance to each of the four classes.
        self.macro_f1 = f1_score(self.y_test, self.y_pred, labels=[0,1,2,3],
                                 average="macro", zero_division=0)
        # Of actual big candles, how often does the model predict big?
        self.big_recall = float(np.mean(magnitude_pred[magnitude_true])) if magnitude_true.any() else 0.0
        print(f"GRU test: accuracy={self.test_acc:.4f}, macro-F1={self.macro_f1:.4f}, "
              f"direction={self.direction_acc:.4f}, magnitude={self.magnitude_acc:.4f}")
        return self.test_acc

    def evaluate(self):
        """Package run settings and test metrics and append them to JSON."""
        if self.y_pred is None:
            raise RuntimeError("Call test before evaluate")
        result = {
            "symbol": self.symbol,
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": "RNNGRU", "timeframe": self.timeframe,
            "params": {"sequence_length": self.sequence_length, "hidden_size": self.hidden_size,
                       "dense_size": self.dense_size, "batch_size": self.batch_size,
                       "learning_rate": self.learning_rate, "max_epochs": self.max_epochs,
                       "patience": self.patience, "random_state": self.random_state},
            "features": self.features, "n_features": len(self.features),
            "n_train": self.n_train, "n_val": self.n_val, "n_test": self.n_test,
            "train_time": float(self.train_time), "training_acc": float(self.training_acc),
            "test_acc": float(self.test_acc), "test_loss": float(self.test_loss),
            "macro_f1": float(self.macro_f1), "direction_acc": float(self.direction_acc),
            "magnitude_acc": float(self.magnitude_acc), "big_recall": float(self.big_recall),
            "body_threshold": self.body_threshold, "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "labels": self.CLASS_NAMES,
            "confusion_matrix": confusion_matrix(self.y_test, self.y_pred, labels=[0,1,2,3]).tolist(),
            "classification_report": classification_report(
                self.y_test, self.y_pred, labels=[0,1,2,3],
                target_names=self.CLASS_NAMES, output_dict=True, zero_division=0),
        }
        # Keep earlier runs in the same JSON results file.
        results = []
        if self.results_path.exists():
            with self.results_path.open() as f:
                results = json.load(f)
        results.append(result)
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with self.results_path.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"Result appended to {self.results_path} ({len(results)} runs total)")
        return result
