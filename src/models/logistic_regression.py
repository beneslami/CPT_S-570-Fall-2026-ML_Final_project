"""
Developed by Benyamin Eslami.
"""

import json
import pathlib
from datetime import datetime
import time
from ML_interface import Model
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

class LogisticRegressionModel(Model):
    def __init__(self, symbol: str, timeframe: str, max_iter: int, random_state: int, results_path=None):
        self.symbol = symbol
        self.timeframe = timeframe
        self.model = LogisticRegression(max_iter=max_iter, random_state=random_state, solver='lbfgs')
        project_dir = pathlib.Path(__file__).resolve().parents[2]
        self.results_path = pathlib.Path(results_path) if results_path else project_dir / "results" / f"logistic_regression_{symbol}_{timeframe}.json"
        self.train_time = 0.0
        self.features = []
        self.n_train = 0
        self.n_test = 0
        self.scaler = StandardScaler()  # Standardize each feature. replaces each value with “how many standard deviations above average”
        self.training_acc = 0.0
        self.test_acc = 0.0
        self.X_train = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.X_test = pd.DataFrame()
        self.y_test = pd.DataFrame()
        self.y_pred = pd.DataFrame()

    def feature_engineering(self, train_data, test_data):
        self.features = ["time", "close", "cci"]

        def prepare(data):
            data = data.copy()
            data["signal"] = 0.0
            sell = (data["cci"] > 100)
            buy = (data["cci"] < -100)
            data.loc[sell, "signal"] = -1
            data.loc[buy, "signal"] = 1
            data["next_signal"] = data["signal"].shift(-1)
            data = data.dropna(subset=["next_signal"])
            return data[self.features], data["next_signal"]

        self.X_train, self.y_train = prepare(train_data)
        self.X_test, self.y_test = prepare(test_data)
        self.X_train = self.X_train.set_index("time")
        self.X_test = self.X_test.set_index("time")

    def train(self):
        self.features = list(self.X_train.columns)
        self.n_train = len(self.X_train)
        X_s = self.scaler.fit_transform(self.X_train)
        time_begin = time.time()
        self.model.fit(X_s, self.y_train)
        self.train_time = time.time() - time_begin
        self.training_acc = self.model.score(X_s, self.y_train)

    def test(self):
        X_s = self.scaler.transform(self.X_test)
        self.n_test = len(self.X_test)
        self.y_pred = self.model.predict(X_s)
        self.test_acc = accuracy_score(self.y_test, self.y_pred)

    def evaluate(self):
        labels = sorted(set(self.y_test))
        result = {
            "symbol": self.symbol,
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": "LogisticRegression",
            "timeframe": self.timeframe,
            "params": {"max_iter": self.model.max_iter, "random_state": self.model.random_state, "class_weight": self.model.class_weight},
            "features": self.features,
            "n_features": len(self.features),
            "n_train": self.n_train,
            "n_test": self.n_test,
            "train_time": float(self.train_time),
            "training_acc": float(self.training_acc),
            "test_acc": float(self.test_acc),
            "labels": [str(label) for label in labels],
            "confusion_matrix": confusion_matrix(self.y_test, self.y_pred, labels=labels).tolist(),
            "classification_report": classification_report(self.y_test, self.y_pred, output_dict=True),
        }

        # Append it to the list of earlier runs in the JSON file
        results = []
        if self.results_path.exists():
            with open(self.results_path) as f:
                results = json.load(f)
        results.append(result)
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"result appended to {self.results_path} ({len(results)} runs in total) for symbol {self.symbol}")
