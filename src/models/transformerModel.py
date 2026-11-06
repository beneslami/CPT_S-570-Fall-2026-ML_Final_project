from __future__ import annotations

import copy
import itertools
import json
import pathlib
from datetime import datetime

import numpy as np

# Import the existing model first: it chooses a Keras 3 backend and the safe
# non-interactive matplotlib backend before either library is imported here.
from models.LSTMModel import LSTMModel

import matplotlib.pyplot as plt
from keras import Input, Model
from keras.layers import (
    Add,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Layer,
    LayerNormalization,
    MultiHeadAttention,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)

class LearnedPositionEmbedding(Layer):
    def __init__(self, lookback, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.lookback = lookback
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        self.position_embeddings = self.add_weight(
            name="position_embeddings",
            shape=(1, self.lookback, self.embedding_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, inputs):
        return inputs + self.position_embeddings

    def get_config(self):
        return {
            **super().get_config(),
            "lookback": self.lookback,
            "embedding_dim": self.embedding_dim,
        }

class TransformerModel(LSTMModel):
    # The complete user-requested search space.  In progressive mode we sample
    # combinations from it deterministically, then retrain the best candidates.
    DEFAULT_LOOKBACK_CANDIDATES = (5, 10, 15, 20)
    DEFAULT_TARGET_THRESHOLD_CANDIDATES = (0.01, 0.02, 0.04, 0.08)
    DEFAULT_EMBEDDING_DIMENSIONS = (8, 12, 16, 20)
    DEFAULT_ATTENTION_HEADS = (2, 4, 6, 8)
    DEFAULT_ENCODER_BLOCKS = (1, 2, 4, 6, 8)
    DEFAULT_DROPOUTS = (0.10, 0.20, 0.30, 0.40)
    DEFAULT_FEATURE_NAMES = (
        "body_pct",
        "range_pct",
        "wick_imbalance",
        "atr_pct",
        "relative_volume",
        "volume_delta_ratio",
        "macd_hist_atr",
        "cci_scaled",
    )

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        results_path=None,
        lookback: int | str = "auto",
        target_threshold: float | str = "auto",
        transformer_config: dict | None = None,
        transformer_configs=None,
        epochs: int = 50,
        batch_size: int = 64,
        confidence_threshold: float | str = "auto",
        model_path: str | None = None,
        random_seed: int = 42,
        lookback_candidates=None,
        target_threshold_candidates=None,
        confidence_threshold_candidates=None,
        min_confident_coverage: float = 0.10,
        multi_timeframes=("15m", "4h"),
        search_strategy: str = "progressive",
        screening_trials: int = 32,
        finalist_trials: int = 6,
        screening_epochs: int = 8,
    ):
        # ``lstm_units='auto'`` prevents the inherited feature_engineering()
        # method from constructing an LSTM before a Transformer configuration is
        # selected in train().
        super().__init__(
            symbol=symbol,
            timeframe=timeframe,
            results_path=results_path,
            lookback=lookback,
            target_threshold=target_threshold,
            lstm_units="auto",
            epochs=epochs,
            batch_size=batch_size,
            confidence_threshold=confidence_threshold,
            model_path=model_path,
            random_seed=random_seed,
            lookback_candidates=lookback_candidates,
            target_threshold_candidates=target_threshold_candidates,
            lstm_unit_candidates=(1,),
            confidence_threshold_candidates=confidence_threshold_candidates,
            min_confident_coverage=min_confident_coverage,
            multi_timeframes=multi_timeframes,
        )
        # Keep the Transformer feature set explicit and independent from the
        # smaller LSTM baseline feature list.
        self.feature_names = list(self.DEFAULT_FEATURE_NAMES)
        self.primary_feature_names = list(self.DEFAULT_FEATURE_NAMES)

        if transformer_config is not None and transformer_configs is not None:
            raise ValueError("Pass transformer_config or transformer_configs, not both")
        if search_strategy not in {"progressive", "exhaustive"}:
            raise ValueError("search_strategy must be 'progressive' or 'exhaustive'")
        if screening_trials < 1 or finalist_trials < 1 or screening_epochs < 1:
            raise ValueError("screening_trials, finalist_trials, and screening_epochs must be positive")
        self.search_strategy = search_strategy
        self.screening_trials = screening_trials
        self.finalist_trials = finalist_trials
        self.screening_epochs = screening_epochs
        self.training_verbose = 0

        raw_configs = (
            (transformer_config,)
            if transformer_config is not None
            else (
                transformer_configs
                if transformer_configs is not None
                else self.default_transformer_configs()
            )
        )
        self.transformer_configs = tuple(
            self._validate_transformer_config(config) for config in raw_configs
        )
        self.transformer_config_mode = "fixed" if transformer_config is not None else "auto"
        self.transformer_config = (
            copy.deepcopy(self.transformer_configs[0])
            if transformer_config is not None
            else None
        )
        self.results_path = (
            pathlib.Path(results_path)
            if results_path
            else pathlib.Path(__file__).resolve().parents[2]
            / "results"
            / f"transformer_{symbol}_{timeframe}.json"
        )

    @classmethod
    def default_transformer_configs(cls):
        """Return only mathematically valid embedding/head combinations."""
        return tuple(
            {
                "embedding_dim": embedding_dim,
                "num_heads": num_heads,
                "num_encoder_blocks": blocks,
                "dropout": dropout,
            }
            for embedding_dim, num_heads, blocks, dropout in itertools.product(
                cls.DEFAULT_EMBEDDING_DIMENSIONS,
                cls.DEFAULT_ATTENTION_HEADS,
                cls.DEFAULT_ENCODER_BLOCKS,
                cls.DEFAULT_DROPOUTS,
            )
            if embedding_dim % num_heads == 0
        )

    @staticmethod
    def _validate_transformer_config(config) -> dict:
        required = {"embedding_dim", "num_heads", "num_encoder_blocks", "dropout"}
        if not isinstance(config, dict) or set(config) != required:
            raise ValueError(
                "Each transformer configuration must contain exactly "
                f"{sorted(required)}"
            )
        normalized = {
            "embedding_dim": int(config["embedding_dim"]),
            "num_heads": int(config["num_heads"]),
            "num_encoder_blocks": int(config["num_encoder_blocks"]),
            "dropout": float(config["dropout"]),
        }
        if normalized["embedding_dim"] < 1:
            raise ValueError("embedding_dim must be positive")
        if normalized["num_heads"] < 1:
            raise ValueError("num_heads must be positive")
        if normalized["embedding_dim"] % normalized["num_heads"] != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if normalized["num_encoder_blocks"] < 1:
            raise ValueError("num_encoder_blocks must be positive")
        if not 0 <= normalized["dropout"] < 1:
            raise ValueError("dropout must be in [0, 1)")
        return normalized

    def _transformer_encoder_block(self, inputs, config, block_number):
        embedding_dim = config["embedding_dim"]
        dropout = config["dropout"]

        normalized = LayerNormalization(epsilon=1e-6, name=f"encoder_{block_number}_attn_norm")(inputs)
        attention = MultiHeadAttention(
            num_heads=config["num_heads"],
            key_dim=embedding_dim // config["num_heads"],
            dropout=dropout,
            name=f"encoder_{block_number}_self_attention",
        )(normalized, normalized)
        attention = Dropout(dropout, name=f"encoder_{block_number}_attn_dropout")(attention)
        residual = Add(name=f"encoder_{block_number}_attn_residual")([inputs, attention])

        normalized = LayerNormalization(epsilon=1e-6, name=f"encoder_{block_number}_ffn_norm")(residual)
        feed_forward = Dense(2 * embedding_dim, activation="gelu", name=f"encoder_{block_number}_ffn_expand")(normalized)
        feed_forward = Dropout(dropout, name=f"encoder_{block_number}_ffn_dropout")(feed_forward)
        feed_forward = Dense(embedding_dim, name=f"encoder_{block_number}_ffn_project")(feed_forward)
        return Add(name=f"encoder_{block_number}_ffn_residual")([residual, feed_forward])

    def _build_model(self):
        if self.transformer_config is None:
            raise RuntimeError("Select a transformer configuration before building the model")

        config = self.transformer_config
        keras = __import__("keras")
        keras.utils.set_random_seed(self.random_seed)

        inputs = Input(shape=(self.lookback, len(self.feature_names)), name="candle_features")
        x = Dense(config["embedding_dim"], name="feature_projection")(inputs)

        # Self-attention has no inherent order.  Each completed candle therefore
        # receives a learned position embedding before entering the encoder.
        x = LearnedPositionEmbedding(
            self.lookback,
            config["embedding_dim"],
            name="position_embedding",
        )(x)

        for block_number in range(config["num_encoder_blocks"]):
            x = self._transformer_encoder_block(x, config, block_number)

        x = LayerNormalization(epsilon=1e-6, name="final_norm")(x)
        x = GlobalAveragePooling1D(name="sequence_pooling")(x)
        x = Dropout(config["dropout"], name="classifier_dropout")(x)
        outputs = Dense(1, activation="sigmoid", name="bullish_probability")(x)
        model = Model(inputs=inputs, outputs=outputs, name="candle_transformer")
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def _candidate_settings(self):
        lookbacks = (
            self.lookback_candidates if self.lookback_mode == "auto" else (self.lookback,)
        )
        thresholds = (
            self.target_threshold_candidates
            if self.target_threshold_mode == "auto"
            else (self.target_threshold,)
        )
        return [
            (float(threshold), lookback, copy.deepcopy(config))
            for threshold, lookback, config in itertools.product(
                thresholds, lookbacks, self.transformer_configs
            )
        ]

    def _progressive_screening_candidates(self, candidates):
        """Sample candidates reproducibly while covering every candidate value."""
        if len(candidates) <= self.screening_trials:
            return list(candidates)

        generator = np.random.default_rng(self.random_seed)
        order = generator.permutation(len(candidates)).tolist()
        selected_indices = []
        observed = {
            "threshold": set(),
            "lookback": set(),
            "embedding_dim": set(),
            "num_heads": set(),
            "num_encoder_blocks": set(),
            "dropout": set(),
        }
        for index in order:
            threshold, lookback, config = candidates[index]
            values = {
                "threshold": threshold,
                "lookback": lookback,
                "embedding_dim": config["embedding_dim"],
                "num_heads": config["num_heads"],
                "num_encoder_blocks": config["num_encoder_blocks"],
                "dropout": config["dropout"],
            }
            if any(values[name] not in observed[name] for name in observed):
                selected_indices.append(index)
                for name, value in values.items():
                    observed[name].add(value)
            if len(selected_indices) == self.screening_trials:
                break

        for index in order:
            if len(selected_indices) == self.screening_trials:
                break
            if index not in selected_indices:
                selected_indices.append(index)
        return [candidates[index] for index in selected_indices]

    def _run_candidate(self, setting, search_round, epoch_budget):
        target_threshold, lookback, config = setting
        self.target_threshold = target_threshold
        self.lookback = lookback
        self.transformer_config = copy.deepcopy(config)
        self._prepare_sequences()

        requested_epochs = self.epochs
        self.epochs = epoch_budget
        try:
            (
                validation_loss,
                validation_balanced_accuracy,
                validation_confident_coverage,
                validation_confident_accuracy,
            ) = self._train_current_lookback()
        finally:
            self.epochs = requested_epochs

        coverage_adjusted_balanced_accuracy = (
            validation_balanced_accuracy * self.validation_target_coverage
        )
        trial = {
            "search_round": search_round,
            "epoch_budget": epoch_budget,
            "target_threshold_atr_fraction": self.target_threshold,
            "lookback": self.lookback,
            "transformer_config": copy.deepcopy(self.transformer_config),
            "n_train": len(self.X_train),
            "n_validation": len(self.X_validation),
            "train_target_coverage": self.train_target_coverage,
            "validation_target_coverage": self.validation_target_coverage,
            "training_acc": self.training_accuracy,
            "training_error": 1.0 - self.training_accuracy,
            "validation_acc": self.validation_accuracy,
            "validation_error": 1.0 - self.validation_accuracy,
            "validation_balanced_acc": validation_balanced_accuracy,
            "coverage_adjusted_validation_balanced_acc": coverage_adjusted_balanced_accuracy,
            "validation_log_loss": validation_loss,
            "confidence_threshold": self.confidence_threshold,
            "validation_confident_coverage": validation_confident_coverage,
            "validation_confident_accuracy": validation_confident_accuracy,
            "epochs_trained": len(self.history.history["loss"]),
            "train_time": self.train_time,
        }
        return trial

    def _capture_state(self):
        return {
            "model": self.model,
            "scaler": self.scaler,
            "history": self.history,
            "lookback": self.lookback,
            "target_threshold": self.target_threshold,
            "transformer_config": copy.deepcopy(self.transformer_config),
            "decision_threshold": self.decision_threshold,
            "confidence_threshold": self.confidence_threshold,
            "training_accuracy": self.training_accuracy,
            "validation_accuracy": self.validation_accuracy,
            "train_time": self.train_time,
            "X_train": self.X_train,
            "y_train": self.y_train,
            "train_times": self.train_times,
            "train_target_coverage": self.train_target_coverage,
            "X_validation": self.X_validation,
            "y_validation": self.y_validation,
            "validation_times": self.validation_times,
            "validation_target_coverage": self.validation_target_coverage,
            "X_test": self.X_test,
            "y_test": self.y_test,
            "test_times": self.test_times,
            "test_target_coverage": self.test_target_coverage,
            "typical_abs_return": self.typical_abs_return,
            "test_current_close": self.test_current_close,
            "test_actual_next_close": self.test_actual_next_close,
        }

    @staticmethod
    def _trial_score(trial):
        return (
            -trial["coverage_adjusted_validation_balanced_acc"],
            trial["validation_log_loss"],
        )

    def train(self):
        if self._train_frame is None:
            raise RuntimeError("Run feature_engineering before train")
        self.lookback_trials = []
        self.target_threshold_trials = self.lookback_trials
        candidates = self._candidate_settings()
        if not candidates:
            raise ValueError("No Transformer hyperparameter candidates were supplied")

        if self.search_strategy == "exhaustive":
            finalists = candidates
            screening = []
        else:
            screening = self._progressive_screening_candidates(candidates)
            print(
                f"Progressive screen: {len(screening)} of {len(candidates)} "
                "candidate combinations"
            )
            for number, setting in enumerate(screening, start=1):
                print(f"  screening candidate {number}/{len(screening)}")
                self.lookback_trials.append(
                    self._run_candidate(setting, "screening", self.screening_epochs)
                )
            finalists = [
                (trial["target_threshold_atr_fraction"], trial["lookback"], trial["transformer_config"])
                for trial in sorted(self.lookback_trials, key=self._trial_score)[: self.finalist_trials]
            ]

        best_state = None
        best_score = None

        print(f"Finalist retraining: {len(finalists)} candidates")
        for number, setting in enumerate(finalists, start=1):
            print(f"  finalist {number}/{len(finalists)}")
            trial = self._run_candidate(setting, "finalist", self.epochs)
            self.lookback_trials.append(trial)
            # Hyperparameters are selected solely by chronological validation.
            score = self._trial_score(trial)
            if best_score is None or score < best_score:
                best_score = score
                best_state = self._capture_state()

        for name, value in best_state.items():
            setattr(self, name, value)
        if self.model_path is not None:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.model_path)
        return self.training_accuracy

    @staticmethod
    def _config_label(config) -> str:
        return (
            f"d{config['embedding_dim']}/h{config['num_heads']}/"
            f"b{config['num_encoder_blocks']}/p{config['dropout']:g}"
        )

    def plot_model_complexity_curve(self) -> pathlib.Path | None:
        matching_trials = [
            trial for trial in self.lookback_trials
            if (
                trial["target_threshold_atr_fraction"] == self.target_threshold
                and trial["lookback"] == self.lookback
            )
        ]
        if not matching_trials:
            return None

        labels = [self._config_label(trial["transformer_config"]) for trial in matching_trials]
        positions = np.arange(len(matching_trials))
        training_errors = [trial["training_error"] for trial in matching_trials]
        validation_errors = [trial["validation_error"] for trial in matching_trials]
        selected_label = self._config_label(self.transformer_config)
        selected_position = labels.index(selected_label)

        target_tag = f"{self.target_threshold * 100:g}".replace(".", "p")
        plot_path = self.results_path.with_name(
            f"{self.results_path.stem}_selected_lb{self.lookback}_"
            f"atr{target_tag}pct_hyperparameter_curve.png"
        )
        figure, axis = plt.subplots(figsize=(12, 6))
        axis.plot(positions, training_errors, marker="o", color="#2e7d32", label="Training error")
        axis.plot(positions, validation_errors, marker="o", color="#2468d4", label="Validation error")
        axis.axvline(selected_position, color="#d62728", linestyle="--", label=f"Selected: {selected_label}",)
        axis.set_title(f"{self.symbol} {self.timeframe}: Transformer architecture selection")
        axis.set_xlabel("Architecture: d=embedding, h=heads, b=blocks, p=dropout")
        axis.set_ylabel("Classification error")
        axis.set_xticks(positions, labels, rotation=25, ha="right")
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(plot_path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return plot_path

    def evaluate(self):
        if self.y_prediction is None:
            raise RuntimeError("Run test before evaluate")
        prediction_plot_path = self.plot_prediction_comparison()
        hyperparameter_plot_path = self.plot_model_complexity_curve()
        confidence = np.where(
            self.y_prediction == 1,
            self.y_probability,
            1.0 - self.y_probability,
        )
        confident = confidence >= self.confidence_threshold
        result = {
            "symbol": self.symbol,
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": "Transformer",
            "timeframe": self.timeframe,
            "target": (
                f"next candle body > {self.target_threshold:.0%} of current ATR: bullish; "
                f"next candle body < -{self.target_threshold:.0%} of current ATR: bearish; "
                "smaller moves omitted"
            ),
            "target_threshold_atr_fraction": self.target_threshold,
            "target_threshold_mode": self.target_threshold_mode,
            "target_threshold_candidates": list(self.target_threshold_candidates),
            "multi_timeframes": list(self.multi_timeframes),
            "split_method": self.split_method,
            "lookback": self.lookback,
            "lookback_mode": self.lookback_mode,
            "transformer_config": copy.deepcopy(self.transformer_config),
            "transformer_config_mode": self.transformer_config_mode,
            "transformer_config_candidates": [copy.deepcopy(config) for config in self.transformer_configs],
            "search_strategy": self.search_strategy,
            "full_candidate_count": len(self._candidate_settings()),
            "screening_trials_requested": self.screening_trials,
            "screening_epochs": self.screening_epochs,
            "finalist_trials_requested": self.finalist_trials,
            "hyperparameter_selection_metric": (
                "maximum coverage-adjusted validation balanced accuracy; "
                "validation log loss breaks ties"
            ),
            "lookback_candidates": list(self.lookback_candidates),
            "lookback_candidates_evaluated": sorted({trial["lookback"] for trial in self.lookback_trials}),
            "target_threshold_candidates_evaluated": sorted(
                {trial["target_threshold_atr_fraction"] for trial in self.lookback_trials}
            ),
            "hyperparameter_trials": self.lookback_trials,
            "features": self.feature_names,
            "n_features": len(self.feature_names),
            "train_period": [str(self.train_times[0]), str(self.train_times[-1])],
            "validation_period": [str(self.validation_times[0]), str(self.validation_times[-1])],
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
            "training_error": 1.0 - self.training_accuracy,
            "validation_acc": self.validation_accuracy,
            "validation_error": 1.0 - self.validation_accuracy,
            "test_acc": self.test_accuracy,
            "test_error": 1.0 - self.test_accuracy,
            "majority_baseline_acc": float(max(np.mean(self.y_test == 0), np.mean(self.y_test == 1))),
            "test_balanced_acc": float(balanced_accuracy_score(self.y_test, self.y_prediction)),
            "test_macro_f1": float(f1_score(self.y_test, self.y_prediction, average="macro")),
            "test_log_loss": float(log_loss(self.y_test, self.y_probability, labels=[0, 1])),
            "prediction_plot": str(prediction_plot_path),
            "prediction_plot_points": min(500, len(self.test_times)),
            "hyperparameter_curve": str(hyperparameter_plot_path) if hyperparameter_plot_path else None,
            "hyperparameter_curve_metric": "training and validation classification error; no test error",
            "predicted_price_method": (
                "current close adjusted by the predicted direction and the "
                "median absolute one-candle return learned from training data"
            ),
            "training_median_absolute_return": self.typical_abs_return,
            "labels": ["bearish", "bullish"],
            "confusion_matrix": confusion_matrix(self.y_test, self.y_prediction, labels=[0, 1]).tolist(),
            "classification_report": classification_report(
                self.y_test,
                self.y_prediction,
                labels=[0, 1],
                target_names=["bearish", "bullish"],
                output_dict=True,
                zero_division=0,
            ),
            "decision_threshold": self.decision_threshold,
            "confidence_threshold_mode": self.confidence_threshold_mode,
            "confidence_threshold": self.confidence_threshold,
            "confidence_threshold_candidates": list(self.confidence_threshold_candidates),
            "min_confident_coverage": self.min_confident_coverage,
            "n_confident": int(confident.sum()),
            "confident_coverage": float(confident.mean()),
            "confident_accuracy": (
                float(accuracy_score(self.y_test[confident], self.y_prediction[confident]))
                if confident.any()
                else None
            ),
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
        print(
            f"result appended to {self.results_path} "
            f"({len(results)} runs in total) for symbol {self.symbol}"
        )
        return result
