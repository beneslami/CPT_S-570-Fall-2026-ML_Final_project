import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ML_interface import Model


PROJECT_DIR = Path(__file__).resolve().parents[2]
FEATURE_COLUMNS = [
    "Close", "High", "Low", "Open", "SMA_20", "SMA_50",
    "EMA_12", "EMA_26", "RSI", "MACD", "MACD_signal", "MACD_hist",
    "BB_upper", "BB_middle", "BB_lower", "FVG_Size", "FVG_Type",
    "OB_Type", "Close_lag1", "Close_lag2", "Close_lag3",
]


def load_data(path):
    """Load the raw OHLCV columns used by the source XGBoost implementation."""
    data = pd.read_csv(path, usecols=["time", "open", "high", "low", "close", "volume"])
    data["time"] = pd.to_datetime(data["time"], utc=True, errors="raise")
    if not data["time"].is_monotonic_increasing or data["time"].duplicated().any():
        raise ValueError(f"{path}: timestamps must be unique and chronological")
    data = data.set_index("time").rename(columns=str.capitalize)
    return data[["Close", "High", "Low", "Open", "Volume"]]


def prepare_features(data, timeframe):
    """Reproduce the original intraday project's features and five-bar target."""
    print(f"Processing {timeframe} data with {len(data)} rows...")
    data = data.copy()
    data["SMA_20"] = data["Close"].rolling(window=20).mean()
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["EMA_12"] = data["Close"].ewm(span=12).mean()
    data["EMA_26"] = data["Close"].ewm(span=26).mean()
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    data["RSI"] = 100 - (100 / (1 + gain / loss))
    data["MACD"] = data["EMA_12"] - data["EMA_26"]
    data["MACD_signal"] = data["MACD"].ewm(span=9).mean()
    data["MACD_hist"] = data["MACD"] - data["MACD_signal"]
    data["BB_middle"] = data["Close"].rolling(window=20).mean()
    bb_std = data["Close"].rolling(window=20).std()
    data["BB_upper"] = data["BB_middle"] + 2 * bb_std
    data["BB_lower"] = data["BB_middle"] - 2 * bb_std
    data["FVG_Size"] = 0.0
    data["FVG_Type"] = 0
    for i in range(1, len(data) - 1):
        low, previous_high, next_high = data["Low"].iloc[i], data["High"].iloc[i - 1], data["High"].iloc[i + 1]
        if low > previous_high and low > next_high:
            data.at[data.index[i], "FVG_Size"] = low - max(previous_high, next_high)
            data.at[data.index[i], "FVG_Type"] = 1
    data["OB_Type"] = 0
    for lag in (1, 2, 3):
        data[f"Close_lag{lag}"] = data["Close"].shift(lag)
    data = data.dropna().copy()
    # Match the source, including its final five unavailable labels as zeros.
    data["Target"] = (data["Close"].shift(-5) > data["Close"]).astype(int)
    data = data.dropna()
    X, y = data[FEATURE_COLUMNS], data["Target"]
    print(f"Prepared {len(X)} samples with {len(FEATURE_COLUMNS)} features")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    return X, y

class XGBoostModel(Model):
    def __init__(self, symbol, timeframe, results_path=None):
        self.symbol = symbol
        self.timeframe = timeframe
        self.results_path = Path(results_path) if results_path else PROJECT_DIR / "results" / f"xgboost_{symbol}_{timeframe}.json"
        self.model = None
        self.search = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = self.y_pred = None
        self.train_time = None
        self.test_acc = None

    def feature_engineering(self, train_data, test_data, **kwargs):
        data = pd.concat([train_data, test_data]).copy()
        # The shared frontend normally provides a time column and lowercase
        # names; accept that format as well as load_data's indexed OHLCV.
        if "time" in data.columns:
            data["time"] = pd.to_datetime(data["time"], utc=True, errors="raise")
            data = data.set_index("time")
        data = data.rename(columns=str.capitalize)
        data.index = pd.to_datetime(data.index, utc=True, errors="raise")
        if not data.index.is_monotonic_increasing or data.index.has_duplicates:
            raise ValueError("XGBoost requires unique, chronological input timestamps")
        data = data[["Close", "High", "Low", "Open", "Volume"]]
        print("WARNING: XGBoost reproduces the source random split and FVG lookahead; its test score is not a chronological holdout.")
        X, y = prepare_features(data, self.timeframe)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    def train(self):
        if self.X_train is None:
            raise RuntimeError("Run feature_engineering() before train()")
        scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()
        grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "scale_pos_weight": [scale_pos_weight],
        }
        self.search = GridSearchCV(XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=1), grid, cv=3, scoring="accuracy", n_jobs=1, verbose=1,)
        start = time.perf_counter()
        self.search.fit(self.X_train, self.y_train)
        self.train_time = time.perf_counter() - start
        self.model = self.search.best_estimator_
        print(f"Best parameters: {self.search.best_params_}")

    def test(self):
        if self.model is None:
            raise RuntimeError("Run train() before test()")
        self.y_pred = self.model.predict(self.X_test)
        self.test_acc = accuracy_score(self.y_test, self.y_pred)
        print(f"Random test accuracy: {self.test_acc:.4f}")
        print(classification_report(self.y_test, self.y_pred, zero_division=0))
        return self.test_acc

    def evaluate(self):
        if self.y_pred is None:
            raise RuntimeError("Run test() before evaluate()")
        result = {
            "symbol": self.symbol,
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": "XGBoost",
            "timeframe": self.timeframe,
            "method": "source-compatible random stratified 80/20 split; FVG uses the next candle",
            "target": "close five rows ahead exceeds current close; source final five labels are zero",
            "features": list(self.X_train.columns),
            "n_features": len(self.X_train.columns),
            "n_train": len(self.X_train),
            "n_test": len(self.X_test),
            "train_time": float(self.train_time),
            "params": self.search.best_params_,
            "test_acc": float(self.test_acc),
            "confusion_matrix": confusion_matrix(self.y_test, self.y_pred, labels=[0, 1]).tolist(),
            "classification_report": classification_report(self.y_test, self.y_pred, labels=[0, 1], output_dict=True, zero_division=0),
            "feature_importance": dict(zip(self.X_train.columns, map(float, self.model.feature_importances_))),
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
        print(f"Result appended to {self.results_path} ({len(results)} runs total)")
        return result
