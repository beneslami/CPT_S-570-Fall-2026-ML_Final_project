"""
Developed by Benyamin Eslami.

The idea of this learning model is to predict the color of the next candle based on historical features of the CURRENT candle.
It is suitable to train this model for 1-hour and 4-hour timeframes.

It takes current candle's features such as close, volume, volume_delta, SMA_10, SMA_20, etc., as input to predict the color of the next candle.
But the question is: Which set of features are most predictive of the next candle's color? 
"""

import json
import pathlib
from datetime import datetime
import time
from ML_interface import Model
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

class LogisticRegressionModel(Model):
    def __init__(self, symbol: str, max_iter: int, random_state: int, results_path=None):
        self.symbol = symbol
        self.model = LogisticRegression(max_iter=max_iter, random_state=random_state, class_weight='balanced')
        # default: <project>/results/logistic_regression.json, wherever the code is run from
        project_dir = pathlib.Path(__file__).resolve().parents[2]
        self.results_path = pathlib.Path(results_path) if results_path else project_dir / "results" / "logistic_regression.json"
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
        features = train_data.columns
        self.features = [c for c in features if c not in ("time", "candle_type", "open", "high", "low", "sma_60")]
        self.X_train = train_data[self.features].iloc[:-1]
        self.y_train = train_data["candle_type"].shift(-1).iloc[:-1]     # colour of the NEXT candle
        self.X_test = test_data[self.features].iloc[:-1]
        self.y_test = test_data["candle_type"].shift(-1).iloc[:-1]

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