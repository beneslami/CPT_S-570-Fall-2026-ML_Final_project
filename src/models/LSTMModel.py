"""LSTM classifier for the direction of the next completed candle."""

import json
import importlib.util
import os
import pathlib
import time
from datetime import datetime
from typing import Optional
import matplotlib

# This module only writes plots to disk; a non-interactive backend is safer for
# command-line training and must be selected before pyplot or Keras is loaded.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Keras 3 needs one backend selected before Keras is imported.
if "KERAS_BACKEND" not in os.environ:
    for backend_name in ("tensorflow", "jax", "torch"):
        if importlib.util.find_spec(backend_name) is not None:
            os.environ["KERAS_BACKEND"] = backend_name
            break

try:
    import keras
    from keras import Input, Sequential
    from keras.callbacks import EarlyStopping
    from keras.layers import Dense, Dropout, LSTM as KerasLSTM
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "Keras 3 needs a backend framework. Install TensorFlow, JAX, or PyTorch "
        "before importing models.LSTM."
    ) from error
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.preprocessing import StandardScaler

from ML_interface import Model


class LSTMModel(Model):
    def __init__(self, symbol: str, timeframe: str, results_path=None, lookback: int = 10, epochs: int = 50, batch_size: int = 64, confidence_threshold: float = 0.65, model_path: Optional[str] = None, random_seed: int = 42,):
        if lookback < 2:
            raise ValueError("lookback must be at least 2 candles")
        if epochs < 1 or batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if not 0.5 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.5 and 1.0")

        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.decision_threshold = 0.5
        self.model_path = pathlib.Path(model_path) if model_path else None
        self.random_seed = random_seed
        self.feature_names = ["body_pct", "range_pct", "upper_wick_pct", "lower_wick_pct", "return_1", "return_3", "return_6", "return_12", "atr_pct", "macd_hist_pct", "rsi_centered", "cci_scaled", "relative_volume_10", "volume_delta_ratio",]

        project_dir = pathlib.Path(__file__).resolve().parents[2]
        self.results_path = (pathlib.Path(results_path) if results_path else project_dir / "results" / f"lstm_{symbol}_{timeframe}.json")

        self.scaler = StandardScaler()
        self.model = None
        self.history = None
        self.split_method = ""
        self.train_times = self.validation_times = self.test_times = None
        self.X_train = self.X_validation = self.X_test = None
        self.y_train = self.y_validation = self.y_test = None
        self.y_probability = self.y_prediction = None
        self.train_time = 0.0
        self.training_accuracy = None
        self.validation_accuracy = None
        self.test_accuracy = None
        self.train_target_coverage = None
        self.validation_target_coverage = None
        self.test_target_coverage = None
        self.typical_abs_return = None
        self.test_current_close = None
        self.test_actual_next_close = None
        self.test_predicted_next_close = None
        np.random.seed(self.random_seed)

    def _validate_frame(self, data: pd.DataFrame, name: str):
        required = {"time", "open", "high", "low", "close", "volume", "volume_delta", "rsi_14", "cci", "atr_14", "macd_histogram"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
        if data.empty:
            raise ValueError(f"{name} is empty")
        times = pd.to_datetime(data["time"], utc=True, errors="raise")
        if not times.is_monotonic_increasing or times.duplicated().any():
            raise ValueError(f"{name} must have unique, increasing timestamps")

    def _numeric_features(self, data):
        open_price = pd.to_numeric(data["open"], errors="coerce")
        high = pd.to_numeric(data["high"], errors="coerce")
        low = pd.to_numeric(data["low"], errors="coerce")
        close = pd.to_numeric(data["close"], errors="coerce")
        volume = pd.to_numeric(data["volume"], errors="coerce")
        atr = pd.to_numeric(data["atr_14"], errors="coerce")

        features = pd.DataFrame(index=data.index)

        features["body_pct"] = (close - open_price) / open_price
        features["range_pct"] = (high - low) / open_price
        features["upper_wick_pct"] = (high - np.maximum(open_price, close)) / open_price
        features["lower_wick_pct"] = (np.minimum(open_price, close) - low) / open_price

        features["return_1"] = close.pct_change(1)
        features["return_3"] = close.pct_change(3)
        features["return_6"] = close.pct_change(6)
        features["return_12"] = close.pct_change(12)

        features["atr_pct"] = atr / close
        features["macd_hist_pct"] = (pd.to_numeric(data["macd_histogram"], errors="coerce") / close)
        features["rsi_centered"] = (pd.to_numeric(data["rsi_14"], errors="coerce") - 50) / 50
        features["cci_scaled"] = (pd.to_numeric(data["cci"], errors="coerce").clip(-300, 300) / 100)
        if "relative_volume_10" in data.columns:
            features["relative_volume_10"] = pd.to_numeric(data["relative_volume_10"], errors="coerce")
        else:
            volume_average_10 = volume.rolling(window=10, min_periods=10,).mean()
            features["relative_volume_10"] = (volume / volume_average_10.replace(0, np.nan))
        features["volume_delta_ratio"] = (pd.to_numeric(data["volume_delta"], errors="coerce") / volume.replace(0, np.nan))

        return features.replace([np.inf, -np.inf], np.nan)

    def _make_sequences(self, data: pd.DataFrame, fit_scaler: bool):
        """End each sequence at candle t and label it from candle t+1."""
        features = self._numeric_features(data)
        next_body = (pd.to_numeric(data["close"], errors="coerce").shift(-1) - pd.to_numeric(data["open"], errors="coerce").shift(-1))
        current_atr = pd.to_numeric(data["atr_14"], errors="coerce")
        minimum_move = 0.10 * current_atr
        labels = pd.Series(np.nan, index=data.index)
        labels.loc[next_body > minimum_move] = 1
        labels.loc[next_body < -minimum_move] = 0

        if fit_scaler:
            scale_rows = features.notna().all(axis=1)
            if scale_rows.sum() < self.lookback:
                raise ValueError("Not enough valid training rows to fit the scaler")
            self.scaler.fit(features.loc[scale_rows])
        scaled = pd.DataFrame(self.scaler.transform(features), columns=self.feature_names, index=data.index,)

        sequences = []
        targets = []
        sequence_times = []
        parsed_times = pd.to_datetime(data["time"], utc=True)
        for end in range(self.lookback - 1, len(data) - 1):
            start = end - self.lookback + 1
            window = scaled.iloc[start : end + 1]
            if window.isna().any().any() or pd.isna(labels.iloc[end]):
                continue
            sequences.append(window.to_numpy(dtype=np.float32))
            targets.append(int(labels.iloc[end]))
            sequence_times.append(parsed_times.iloc[end])

        if not sequences:
            raise ValueError("No complete LSTM sequences could be created")
        possible_sequences = max(1, len(data) - self.lookback)
        target_coverage = len(sequences) / possible_sequences
        return (np.asarray(sequences, dtype=np.float32), np.asarray(targets, dtype=np.float32), pd.DatetimeIndex(sequence_times, name="feature_end_time"), target_coverage,)

    @staticmethod
    def _aligned_closes(data: pd.DataFrame, sequence_times):
        prices = pd.DataFrame(
            {
                "time": pd.to_datetime(data["time"], utc=True),
                "close": pd.to_numeric(data["close"], errors="coerce"),
            }
        ).set_index("time")
        prices["next_close"] = prices["close"].shift(-1)
        aligned = prices.reindex(sequence_times)
        return (
            aligned["close"].to_numpy(dtype=float),
            aligned["next_close"].to_numpy(dtype=float),
        )

    def _build_model(self):
        keras.utils.set_random_seed(self.random_seed)
        model = Sequential([Input(shape=(self.lookback, len(self.feature_names))), KerasLSTM(50), Dense(10, activation="relu"), Dropout(0.20), Dense(1, activation="sigmoid"),])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"],)
        return model

    def feature_engineering(self, train_data, test_data):
        self._validate_frame(train_data, "train_data")
        self._validate_frame(test_data, "test_data")

        if train_data.equals(test_data):
            test_start = int(len(train_data) * 0.8)
            train_and_validation = train_data.iloc[:test_start].copy()
            test_frame = test_data.iloc[test_start:].copy()
            self.split_method = "same_csv_chronological_64_16_20"
        else:
            train_end = pd.to_datetime(train_data["time"].iloc[-1], utc=True)
            test_start_time = pd.to_datetime(test_data["time"].iloc[0], utc=True)
            if train_end >= test_start_time:
                raise ValueError("Training data must end before test data starts")
            train_and_validation = train_data.copy()
            test_frame = test_data.copy()
            self.split_method = "separate_chronological_files_80_20_train_validation"

        validation_start = int(len(train_and_validation) * 0.8)
        train_frame = train_and_validation.iloc[:validation_start].copy()
        validation_frame = train_and_validation.iloc[validation_start:].copy()

        (self.X_train, self.y_train, self.train_times, self.train_target_coverage,) = self._make_sequences(train_frame, fit_scaler=True)
        (self.X_validation, self.y_validation, self.validation_times, self.validation_target_coverage,) = self._make_sequences(validation_frame, fit_scaler=False)
        (self.X_test, self.y_test, self.test_times, self.test_target_coverage,) = self._make_sequences(test_frame, fit_scaler=False)

        train_current_close, train_actual_next_close = self._aligned_closes(
            train_frame,
            self.train_times,
        )
        train_abs_return = np.abs(
            train_actual_next_close / train_current_close - 1.0
        )
        self.typical_abs_return = float(np.nanmedian(train_abs_return))
        self.test_current_close, self.test_actual_next_close = (
            self._aligned_closes(test_frame, self.test_times)
        )

        if min(len(self.X_train), len(self.X_validation), len(self.X_test)) < 20:
            raise ValueError("Need at least 20 complete sequences in each split")
        if len(np.unique(self.y_train)) != 2 or len(np.unique(self.y_validation)) != 2:
            raise ValueError("Training and validation sequences must each contain bullish and bearish labels")
        self.model = self._build_model()

    def train(self):
        if self.model is None or self.X_train is None:
            raise RuntimeError("Run feature_engineering before train")

        early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True,)
        started = time.time()
        self.history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_validation, self.y_validation), epochs=self.epochs, batch_size=self.batch_size, shuffle=False, callbacks=[early_stopping], verbose=1,)
        self.train_time = time.time() - started
        train_probability = self.model.predict(self.X_train, batch_size=self.batch_size, verbose=0).reshape(-1)
        validation_probability = self.model.predict(self.X_validation, batch_size=self.batch_size, verbose=0).reshape(-1)

        thresholds = np.arange(0.40, 0.61, 0.01)
        threshold_scores = [balanced_accuracy_score(self.y_validation, validation_probability >= threshold,) for threshold in thresholds]
        self.decision_threshold = float(thresholds[np.argmax(threshold_scores)])

        self.training_accuracy = float(accuracy_score(self.y_train, train_probability >= self.decision_threshold,))
        self.validation_accuracy = float(accuracy_score(self.y_validation, validation_probability >= self.decision_threshold,))
        if self.model_path is not None:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.model_path)
        return self.training_accuracy

    def test(self):
        if self.model is None or self.history is None:
            raise RuntimeError("Run train before test")
        self.y_probability = self.model.predict(self.X_test, batch_size=self.batch_size, verbose=0).reshape(-1)
        self.y_prediction = (self.y_probability >= self.decision_threshold).astype(int)
        self.test_accuracy = float(accuracy_score(self.y_test, self.y_prediction))
        return self.test_accuracy

    def predict_next(self, recent_data: pd.DataFrame) -> dict:
        """Predict the candle immediately after the final completed input candle."""
        if self.model is None or self.history is None:
            raise RuntimeError("Train the model before predicting")
        self._validate_frame(recent_data, "recent_data")
        features = self._numeric_features(recent_data).tail(self.lookback)
        if len(features) != self.lookback or features.isna().any().any():
            raise ValueError(f"Need {self.lookback} complete recent candles for prediction")
        scaled = self.scaler.transform(features).astype(np.float32)
        probability = float(self.model.predict(scaled[None, :, :], verbose=0)[0, 0])
        direction = ("bullish" if probability >= self.decision_threshold else "bearish")
        confidence = (probability if direction == "bullish" else 1.0 - probability)
        return {
            "as_of_time": str(recent_data["time"].iloc[-1]),
            "prediction_for": "next_candle",
            "direction": direction,
            "bullish_probability": probability,
            "bearish_probability": 1.0 - probability,
            "confidence": confidence,
            "confident_signal": (direction if confidence >= self.confidence_threshold else "abstain"),
        }

    def plot_prediction_comparison(self, max_points: int = 500) -> pathlib.Path:
        """Save actual and direction-implied next-close prices for the test set.

        The LSTM is a direction classifier, not a price regression model.  Its
        implied price therefore uses the predicted direction together with the
        median absolute one-candle return measured on training data only.
        """
        if self.y_prediction is None:
            raise RuntimeError("Run test before plotting predictions")
        if max_points < 1:
            raise ValueError("max_points must be positive")
        if self.typical_abs_return is None or not np.isfinite(self.typical_abs_return):
            raise RuntimeError("Training price movement is unavailable")

        predicted_direction = np.where(self.y_prediction == 1, 1.0, -1.0)
        self.test_predicted_next_close = self.test_current_close * (
            1.0 + predicted_direction * self.typical_abs_return
        )

        plot_path = self.results_path.with_name(
            f"{self.results_path.stem}_prediction_comparison.png"
        )
        plot_path.parent.mkdir(parents=True, exist_ok=True)

        # A shorter recent segment remains readable when the test set contains
        # thousands of candles.  The complete test set is still used in every
        # reported evaluation metric.
        first = max(0, len(self.test_times) - max_points)
        plot_times = self.test_times[first:]
        actual_prices = self.test_actual_next_close[first:]
        predicted_prices = self.test_predicted_next_close[first:]

        figure, axis = plt.subplots(figsize=(14, 7))
        axis.plot(
            plot_times,
            actual_prices,
            label="Actual next-candle close",
            linewidth=1.8,
        )
        axis.plot(
            plot_times,
            predicted_prices,
            label="Direction-implied next-candle close",
            linewidth=1.3,
            alpha=0.85,
        )
        axis.set_title(
            f"{self.symbol} {self.timeframe}: actual vs predicted direction-implied price"
        )
        axis.set_xlabel("Feature window end time")
        axis.set_ylabel("Price")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.autofmt_xdate()
        figure.tight_layout()
        figure.savefig(plot_path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return plot_path

    def evaluate(self):
        if self.y_prediction is None:
            raise RuntimeError("Run test before evaluate")
        prediction_plot_path = self.plot_prediction_comparison()
        confidence = np.where(self.y_prediction == 1, self.y_probability, 1.0 - self.y_probability,)
        confident = confidence >= self.confidence_threshold
        result = {
            "symbol": self.symbol,
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": "LSTM",
            "timeframe": self.timeframe,
            "target": (
                "next candle body > 10% of current ATR: bullish; "
                "next candle body < -10% of current ATR: bearish; "
                "smaller moves omitted"
            ),
            "split_method": self.split_method,
            "lookback": self.lookback,
            "features": self.feature_names,
            "n_features": len(self.feature_names),
            "train_period": [str(self.train_times[0]), str(self.train_times[-1])],
            "validation_period": [str(self.validation_times[0]), str(self.validation_times[-1]),],
            "test_period": [str(self.test_times[0]), str(self.test_times[-1])],
            "n_train": len(self.X_train),
            "n_validation": len(self.X_validation),
            "n_test": len(self.X_test),
            "train_target_coverage": self.train_target_coverage,
            "validation_target_coverage": self.validation_target_coverage,
            "test_target_coverage": self.test_target_coverage,
            "epochs_requested": self.epochs,
            "epochs_trained": len(self.history.history["loss"]),
            "batch_size": self.batch_size,
            "train_time": self.train_time,
            "training_acc": self.training_accuracy,
            "validation_acc": self.validation_accuracy,
            "test_acc": self.test_accuracy,
            "majority_baseline_acc": float(max(np.mean(self.y_test == 0), np.mean(self.y_test == 1))),
            "test_balanced_acc": float(balanced_accuracy_score(self.y_test, self.y_prediction)),
            "test_macro_f1": float(f1_score(self.y_test, self.y_prediction, average="macro")),
            "test_log_loss": float(log_loss(self.y_test, self.y_probability, labels=[0, 1])),
            "prediction_plot": str(prediction_plot_path),
            "prediction_plot_points": min(500, len(self.test_times)),
            "predicted_price_method": (
                "current close adjusted by the predicted direction and the "
                "median absolute one-candle return learned from training data"
            ),
            "training_median_absolute_return": self.typical_abs_return,
            "labels": ["bearish", "bullish"],
            "confusion_matrix": confusion_matrix(self.y_test, self.y_prediction, labels=[0, 1]).tolist(),
            "classification_report": classification_report(self.y_test, self.y_prediction, labels=[0, 1], target_names=["bearish", "bullish"], output_dict=True, zero_division=0,),
            "decision_threshold": self.decision_threshold,
            "confidence_threshold": self.confidence_threshold,
            "n_confident": int(confident.sum()),
            "confident_coverage": float(confident.mean()),
            "confident_accuracy": (float(accuracy_score(self.y_test[confident], self.y_prediction[confident])) if confident.any() else None),
        }

        results = []
        if self.results_path.exists():
            with self.results_path.open() as file:
                results = json.load(file)
            if not isinstance(results, list):
                raise ValueError(f"Expected a list of runs in {self.results_path}")
        results.append(result)
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with self.results_path.open("w") as file:
            json.dump(results, file, indent=2)
        print(f"result appended to {self.results_path} ({len(results)} runs in total) for symbol {self.symbol}")
        return result
