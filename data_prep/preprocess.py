from numbers import Integral
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


class preprocess:
    """Audit and transform candles, calculate indicators, and create walk-forward folds."""

    PRICE_COLUMNS = ["open", "high", "low", "close"]
    RAW_COLUMNS = ["time", *PRICE_COLUMNS, "volume"]
    SOURCE_COLUMNS = [*RAW_COLUMNS, "volume_delta"]
    BASELINE_COLUMNS = ["return_1", "body_pct"]
    INDICATOR_COLUMNS = [
        "sma_20", "ema_20", "dma_20_5", "bb_middle", "bb_upper", "bb_lower",
        "rsi_14", "momentum_10", "macd", "macd_signal", "macd_histogram",
        "volume_ma_20", "relative_volume_20", "volume_delta_estimate",
        "ichimoku_tenkan", "ichimoku_kijun", "ichimoku_senkou_a", "ichimoku_senkou_b",
        "true_range", "atr_sma_14", "cci_14", "cci_20",
    ]
    FEATURE_COLUMNS = [*BASELINE_COLUMNS, *INDICATOR_COLUMNS]
    MAX_WARMUP_ROWS = 77

    SYMBOLS = [
        "XAU", "USD", "QQQ.US", "MSFT.US", "HPE.US", "GOOGL.US", "GBP", "EUR",
        "ETH", "E_SANDP-500", "E_NQ-100", "E_D&J-IND", "BTC", "AMZN.US", "AMD.US", "AAPL.US",
    ]
    BAR_INTERVALS = {
        "1m": pd.Timedelta(minutes=1),
        "5m": pd.Timedelta(minutes=5),
        "15m": pd.Timedelta(minutes=15),
        "30m": pd.Timedelta(minutes=30),
        "1h": pd.Timedelta(hours=1),
        "4h": pd.Timedelta(hours=4),
    }
    SOURCE_TIMEFRAME_SUFFIXES = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1hour",
        "4h": "4hour",
    }
    PRICE_AUDIT_COLUMNS = [
        "invalid_price",
        "invalid_ohlc",
        "large_close_move",
        "large_candle_range",
        "isolated_spike_reversal",
    ]

    def __init__(
        self,
        data_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        write_outputs: bool = True,
        symbols: Sequence[str] | None = None,
        timeframes: Sequence[str] | None = None,
        bar_timestamp: str = "open",
        data_as_of_utc: pd.Timestamp | None = None,
        train_size: int = 12,
        val_size: int = 2,
        test_size: int = 2,
        step: int = 2,
        min_move: float = 0.15,
        robust_z: float = 12,
    ) -> None:
        sizes = (train_size, val_size, test_size, step)
        if any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or value < 1
            for value in sizes
        ):
            raise ValueError("train_size, val_size, test_size and step must be positive integers.")
        if (
            not np.isfinite(min_move)
            or not np.isfinite(robust_z)
            or min_move <= 0
            or robust_z <= 0
        ):
            raise ValueError("Price thresholds must be finite and positive.")

        project_dir = Path(__file__).resolve().parent.parent
        self.data_dir = (
            Path(data_dir).expanduser().resolve()
            if data_dir is not None
            else project_dir / "datapoints"
        )
        self.output_dir = (
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else self.data_dir.parent / "datapoints_preprocessed"
        )
        self.write_outputs = write_outputs
        self.symbols = list(symbols) if symbols is not None else self.SYMBOLS.copy()
        self.timeframes = list(timeframes) if timeframes is not None else ["15m"]
        self.bar_timestamp = bar_timestamp
        self.data_as_of_utc = (
            pd.Timestamp(data_as_of_utc)
            if data_as_of_utc is not None
            else pd.Timestamp.now(tz="UTC")
        )
        self.train_size, self.val_size, self.test_size, self.step = sizes
        self.min_move, self.robust_z = min_move, robust_z

        self.quality_summary = pd.DataFrame()
        self.gap_events = pd.DataFrame()
        self.price_events = pd.DataFrame()
        self.quality_reports: dict[
            tuple[str, str],
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        ] = {}
        self.segments: dict[tuple[str, str], list[dict]] = {}
        self.splits: dict[tuple[str, str], list[dict[str, tuple[int, int]]]] = {}
        self.processed_paths: dict[tuple[str, str], Path | None] = {}
        self.selected_folds: dict[
            tuple[str, str],
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        ] = {}
        self.train_data = pd.DataFrame()
        self.eval_data = pd.DataFrame()
        self.test_data = pd.DataFrame()
        self.column = self.FEATURE_COLUMNS.copy()

    def load_clean_bars(
        self,
        file_path: str | Path,
        interval: pd.Timedelta,
        raw_data: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        if raw_data is None:
            df = pd.read_csv(
                file_path,
                usecols=lambda column: column in self.SOURCE_COLUMNS,
            )
        else:
            columns = [
                column for column in raw_data.columns
                if column in self.SOURCE_COLUMNS
            ]
            df = raw_data.loc[:, columns].copy()
        missing_columns = sorted(set(self.RAW_COLUMNS) - set(df.columns))
        if missing_columns:
            raise ValueError(f"{file_path}: missing required columns {missing_columns}")

        df["source_file"] = str(Path(file_path).expanduser().resolve())
        df["source_line"] = np.arange(len(df)) + 2
        df["source_row"] = df["source_line"]
        df = df.rename(columns={"volume_delta": "source_volume_delta"})
        df["time"] = pd.to_datetime(
            df["time"], format="mixed", utc=True, errors="coerce"
        )
        invalid_time = df["time"].isna()
        audit = {
            "input_rows": len(df),
            "invalid_time_rows": int(invalid_time.sum()),
        }
        df = df.loc[~invalid_time].copy()

        for column in [*self.PRICE_COLUMNS, "volume"]:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype(float)
        if "source_volume_delta" in df:
            df["source_volume_delta"] = pd.to_numeric(
                df["source_volume_delta"], errors="coerce"
            )

        prices = df[self.PRICE_COLUMNS]
        valid_prices = np.isfinite(prices).all(axis=1) & prices.gt(0).all(axis=1)
        del prices
        valid_prices &= df["high"].ge(df[["open", "close", "low"]].max(axis=1))
        valid_prices &= df["low"].le(df[["open", "close"]].min(axis=1))
        audit["invalid_price_rows"] = int((~valid_prices).sum())
        df = df.loc[valid_prices].copy()
        df["volume"] = df["volume"].where(
            np.isfinite(df["volume"]) & df["volume"].ge(0)
        )

        df["available_at"] = df["time"]
        if self.bar_timestamp == "open":
            df["available_at"] = df["time"] + interval
        incomplete = df["available_at"].gt(self.data_as_of_utc)
        audit["incomplete_or_future_rows"] = int(incomplete.sum())
        df = df.loc[~incomplete].copy()

        rows_before = len(df)
        df = df.drop_duplicates(subset=self.RAW_COLUMNS)
        audit["identical_ohlcv_duplicates_removed"] = rows_before - len(df)
        duplicate_times = df.loc[df["time"].duplicated(), "time"]
        if not duplicate_times.empty:
            examples = duplicate_times.drop_duplicates().head(3).tolist()
            raise ValueError(f"{file_path}: conflicting OHLCV rows at {examples}")

        df = df.sort_values("time").reset_index(drop=True)
        time_gap = df["time"].diff()
        consecutive = time_gap.eq(interval)
        df["time_gap"] = time_gap
        df["gap_before"] = time_gap.notna() & ~consecutive
        observed_return = df["close"].pct_change(fill_method=None)
        df["return_1"] = observed_return.where(consecutive)
        df["price_jump"] = observed_return.abs().gt(0.10)
        audit.update(
            clean_rows=len(df),
            irregular_intervals=int(df["gap_before"].sum()),
            price_jumps_over_10pct=int(df["price_jump"].sum()),
            missing_or_invalid_volume_rows=int(df["volume"].isna().sum()),
            zero_volume_rows=int(df["volume"].eq(0).sum()),
        )
        return df, audit

    @staticmethod
    def add_sma(
        df: pd.DataFrame, period: int = 20, column: str = "close"
    ) -> pd.DataFrame:
        df[f"sma_{period}"] = df[column].rolling(
            window=period, min_periods=period
        ).mean()
        return df

    @staticmethod
    def add_ema(
        df: pd.DataFrame, period: int = 20, column: str = "close"
    ) -> pd.DataFrame:
        values = df[column].to_numpy(dtype=float)
        ema = np.full(len(values), np.nan)
        if len(values) >= period:
            k = 2 / (period + 1)
            ema[period - 1] = values[:period].mean()
            for i in range(period, len(values)):
                ema[i] = values[i] * k + ema[i - 1] * (1 - k)
        df[f"ema_{period}"] = ema
        return df

    @staticmethod
    def add_dma(
        df: pd.DataFrame,
        period: int = 20,
        displacement: int = 5,
        column: str = "close",
    ) -> pd.DataFrame:
        sma = df[column].rolling(window=period, min_periods=period).mean()
        df[f"dma_{period}_{displacement}"] = sma.shift(displacement)
        return df

    @staticmethod
    def add_bollinger_bands(
        df: pd.DataFrame, period: int = 20, num_std: float = 2, column: str = "close"
    ) -> pd.DataFrame:
        rolling = df[column].rolling(window=period, min_periods=period)
        middle = rolling.mean()
        std = rolling.std(ddof=0)
        df["bb_middle"] = middle
        df["bb_upper"] = middle + num_std * std
        df["bb_lower"] = middle - num_std * std
        return df

    @staticmethod
    def add_rsi(
        df: pd.DataFrame, period: int = 14, column: str = "close"
    ) -> pd.DataFrame:
        change = df[column].diff().to_numpy(dtype=float)
        gain = np.where(change > 0, change, 0.0)
        loss = np.where(change < 0, -change, 0.0)
        rsi = np.full(len(df), np.nan)
        if len(df) > period:
            avg_gain = gain[1 : period + 1].mean()
            avg_loss = loss[1 : period + 1].mean()
            for i in range(period, len(df)):
                if i > period:
                    avg_gain = (avg_gain * (period - 1) + gain[i]) / period
                    avg_loss = (avg_loss * (period - 1) + loss[i]) / period
                if avg_loss == 0:
                    rsi[i] = 100.0 if avg_gain > 0 else 50.0
                else:
                    rsi[i] = 100 - 100 / (1 + avg_gain / avg_loss)
        df[f"rsi_{period}"] = rsi
        return df

    @staticmethod
    def add_momentum(
        df: pd.DataFrame, period: int = 10, column: str = "close"
    ) -> pd.DataFrame:
        df[f"momentum_{period}"] = df[column] - df[column].shift(period)
        return df

    @staticmethod
    def add_macd(
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        column: str = "close",
    ) -> pd.DataFrame:
        fast_ema = df[column].ewm(
            span=fast, adjust=False, min_periods=fast
        ).mean()
        slow_ema = df[column].ewm(
            span=slow, adjust=False, min_periods=slow
        ).mean()
        macd = fast_ema - slow_ema
        macd_signal = macd.ewm(
            span=signal, adjust=False, min_periods=signal
        ).mean()
        df["macd"] = macd
        df["macd_signal"] = macd_signal
        df["macd_histogram"] = macd - macd_signal
        return df

    @staticmethod
    def add_volume(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        df[f"volume_ma_{period}"] = df["volume"].rolling(
            window=period, min_periods=period
        ).mean()
        volume_mean = df[f"volume_ma_{period}"]
        df[f"relative_volume_{period}"] = df["volume"] / volume_mean.where(
            volume_mean.gt(0)
        )
        return df

    @staticmethod
    def add_volume_delta_estimate(df: pd.DataFrame) -> pd.DataFrame:
        candle_range = (df["high"] - df["low"]).replace(0, np.nan)
        body_ratio = ((df["close"] - df["open"]) / candle_range).fillna(0.0)
        df["volume_delta_estimate"] = df["volume"] * body_ratio
        return df

    @staticmethod
    def add_ichimoku(
        df: pd.DataFrame,
        tenkan: int = 9,
        kijun: int = 26,
        senkou_b: int = 52,
        displacement: int = 26,
    ) -> pd.DataFrame:
        def midpoint(period: int) -> pd.Series:
            high = df["high"].rolling(window=period, min_periods=period).max()
            low = df["low"].rolling(window=period, min_periods=period).min()
            return (high + low) / 2

        tenkan_sen = midpoint(tenkan)
        kijun_sen = midpoint(kijun)
        df["ichimoku_tenkan"] = tenkan_sen
        df["ichimoku_kijun"] = kijun_sen
        df["ichimoku_senkou_a"] = ((tenkan_sen + kijun_sen) / 2).shift(
            displacement
        )
        df["ichimoku_senkou_b"] = midpoint(senkou_b).shift(displacement)
        return df

    @staticmethod
    def add_atr(
        df: pd.DataFrame, period: int = 14, method: str = "wilder"
    ) -> pd.DataFrame:
        prev_close = df["close"].shift(1)
        ranges = [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ]
        true_range = pd.concat(ranges, axis=1).max(axis=1).to_numpy(dtype=float)

        if method == "sma":
            atr = (
                pd.Series(true_range)
                .rolling(window=period, min_periods=period)
                .mean()
                .to_numpy()
            )
            name = f"atr_sma_{period}"
        elif method == "wilder":
            atr = np.full(len(true_range), np.nan)
            if len(true_range) >= period:
                atr[period - 1] = true_range[:period].mean()
                for i in range(period, len(true_range)):
                    atr[i] = (atr[i - 1] * (period - 1) + true_range[i]) / period
            name = f"atr_{period}"
        else:
            raise ValueError("method must be 'wilder' or 'sma'")

        df["true_range"] = true_range
        df[name] = atr
        return df

    @staticmethod
    def add_body_pct(df: pd.DataFrame) -> pd.DataFrame:
        open_price = df["open"].replace(0, np.nan)
        df["body_pct"] = (df["close"] - df["open"]) / open_price
        return df

    @staticmethod
    def cci(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        average = typical_price.rolling(window).mean()
        mean_deviation = pd.Series(np.nan, index=df.index)

        if len(df) >= window:
            windows = sliding_window_view(typical_price.to_numpy(), window)
            batch_size = 50_000
            for start in range(0, len(windows), batch_size):
                batch = windows[start : start + batch_size]
                center = batch.mean(axis=1, keepdims=True)
                first_row = start + window - 1
                mean_deviation.iloc[first_row : first_row + len(batch)] = (
                    np.abs(batch - center).mean(axis=1)
                )

        df[f"cci_{window}"] = (
            typical_price - average
        ) / (0.015 * mean_deviation.replace(0, np.nan))
        return df

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        self.add_body_pct(df)
        self.add_sma(df, period=20)
        self.add_ema(df, period=20)
        self.add_dma(df, period=20, displacement=5)
        self.add_bollinger_bands(df, period=20)
        self.add_rsi(df, period=14)
        self.add_momentum(df, period=10)
        self.add_macd(df)
        self.add_volume(df, period=20)
        self.add_volume_delta_estimate(df)
        self.add_ichimoku(df)
        self.add_atr(df, period=14, method="sma")
        self.cci(df, window=14)
        self.cci(df, window=20)
        df[self.FEATURE_COLUMNS] = df[self.FEATURE_COLUMNS].replace(
            [np.inf, -np.inf], np.nan
        )
        return df

    @staticmethod
    def add_next_bar_targets(
        df: pd.DataFrame, interval: pd.Timedelta
    ) -> pd.DataFrame:
        next_bar_time = df["available_at"].shift(-1)
        has_next_bar = (next_bar_time - df["available_at"]).eq(interval)
        next_close = df["close"].shift(-1).where(has_next_bar)

        df["target_end"] = next_bar_time.where(has_next_bar)
        df["target_log_return"] = np.log(next_close) - np.log(df["close"])
        df["target_up"] = next_close.gt(df["close"]).where(has_next_bar).astype("Int8")
        return df

    def _inspect_frame(
        self,
        raw: pd.DataFrame,
        market: str,
        timeframe: str,
    ) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
        df = raw.copy()
        df["source_row"] = np.arange(len(df)) + 2
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
        invalid_times = int(df["time"].isna().sum())
        df = (
            df.dropna(subset=["time"])
            .sort_values("time", kind="stable")
            .reset_index(drop=True)
        )
        for column in self.PRICE_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        interval = self.BAR_INTERVALS[timeframe]
        previous = df["time"].shift()
        delta = df["time"] - previous
        day_jump = (df["time"].dt.normalize() - previous.dt.normalize()).dt.days
        crypto = market in {"BTC", "ETH"}
        weekend = (
            (previous.dt.dayofweek == 4)
            & (df["time"].dt.dayofweek.isin([6, 0]))
            & day_jump.isin([2, 3])
        ) if not crypto else pd.Series(False, index=df.index)
        overnight = (
            (day_jump == 1)
            & (previous.dt.dayofweek < 4)
            & (delta > interval)
        )
        baseline = delta[overnight].median() if not crypto else pd.NaT
        adjusted = delta - pd.to_timedelta(weekend.astype(int) * 2, unit="D")
        expected_session = pd.Series(False, index=df.index)
        if pd.notna(baseline):
            tolerance = max(interval, pd.Timedelta(hours=1))
            expected_session = (
                (overnight | weekend)
                & ((adjusted - baseline).abs() <= tolerance)
            )
        normal_weekend = weekend & (
            (adjusted <= interval) | expected_session
        )
        has_gap = delta > interval
        classification = pd.Series("unexplained_gap", index=df.index)
        classification.loc[expected_session & ~weekend] = "typical_session_closure"
        classification.loc[normal_weekend] = "weekend_closure"
        classification.loc[weekend & ~normal_weekend] = "weekend_plus_extra_gap"
        gaps = pd.DataFrame({
            "source_row": df["source_row"],
            "previous_time": previous,
            "time": df["time"],
            "gap": delta,
            "weekend_adjusted_gap": adjusted,
            "classification": classification,
        })[has_gap].copy()
        gaps["needs_review"] = ~gaps["classification"].isin(
            ["weekend_closure", "typical_session_closure"]
        )

        prices = df[self.PRICE_COLUMNS]
        invalid = (
            prices.isna().any(axis=1)
            | prices.le(0).any(axis=1)
            | (~np.isfinite(prices)).any(axis=1)
        )
        inconsistent = (
            (df["high"] < prices[["open", "close", "low"]].max(axis=1))
            | (df["low"] > prices[["open", "close", "high"]].min(axis=1))
        )
        close = df["close"].where(df["close"] > 0)
        log_return = np.log(close / close.shift())
        magnitude = log_return.abs()
        trailing = magnitude.shift().rolling(200, min_periods=30)
        median = trailing.median()
        mad = (
            (magnitude - magnitude.rolling(200, min_periods=30).median())
            .abs()
            .shift()
            .rolling(200, min_periods=30)
            .median()
        )
        threshold = np.maximum(
            np.log1p(self.min_move),
            median + self.robust_z * 1.4826 * mad,
        ).fillna(np.log1p(self.min_move))
        large_move = magnitude > threshold
        wick = np.log(
            df["high"].where(df["high"] > 0) / df["low"].where(df["low"] > 0)
        )
        large_range = wick > threshold * 2
        next_return = log_return.shift(-1)
        reversal = (
            large_move
            & (log_return * next_return < 0)
            & (next_return.abs() >= magnitude * 0.7)
            & ((log_return + next_return).abs() <= magnitude * 0.3)
        )

        events = df.loc[
            invalid | inconsistent | large_move | large_range,
            ["source_row", "time", *self.PRICE_COLUMNS],
        ].copy()
        events["previous_close"] = close.shift().reindex(events.index)
        events["next_close"] = close.shift(-1).reindex(events.index)
        events["return_pct"] = (
            close.pct_change(fill_method=None) * 100
        ).reindex(events.index)
        events["threshold_pct"] = (np.expm1(threshold) * 100).reindex(events.index)
        events["gap_before"] = has_gap.reindex(events.index)
        events["invalid_price"] = invalid.reindex(events.index)
        events["invalid_ohlc"] = inconsistent.reindex(events.index)
        events["large_close_move"] = large_move.reindex(events.index)
        events["large_candle_range"] = large_range.reindex(events.index)
        events["isolated_spike_reversal"] = reversal.reindex(events.index)
        events["assessment"] = "review_move_or_wick"
        events.loc[
            events["isolated_spike_reversal"], "assessment"
        ] = "possible_bad_tick_reversal"
        events.loc[
            events["invalid_price"] | events["invalid_ohlc"], "assessment"
        ] = "invalid_price_or_ohlc"

        for table in (gaps, events):
            table.insert(0, "timeframe", timeframe)
            table.insert(0, "market", market)

        summary = {
            "market": market,
            "timeframe": timeframe,
            "rows": len(raw),
            "start": df["time"].min(),
            "end": df["time"].max(),
            "invalid_times": invalid_times,
            "duplicate_timestamps": int(df["time"].duplicated(keep=False).sum()),
            "missing_ohlc": int(prices.isna().any(axis=1).sum()),
            "typical_overnight_gap": baseline,
            "total_gaps": len(gaps),
            "gaps_to_review": int(gaps["needs_review"].sum()),
            "price_events": len(events),
            "invalid_price_or_ohlc": int((invalid | inconsistent).sum()),
            "spike_reversals": int(reversal.sum()),
        }
        return summary, gaps, events

    def _quality_flags(
        self,
        raw: pd.DataFrame,
        gaps: pd.DataFrame,
        events: pd.DataFrame,
    ) -> pd.DataFrame:
        flags = pd.DataFrame({"source_line": np.arange(len(raw)) + 2})
        times = pd.to_datetime(raw["time"], errors="coerce", utc=True)
        flags["duplicate_timestamp"] = times.notna() & times.duplicated(keep=False)

        gap_lookup = gaps.set_index("source_row")
        flags["gap_classification"] = flags["source_line"].map(
            gap_lookup["classification"]
        )
        flags["gap_needs_review"] = flags["source_line"].map(
            gap_lookup["needs_review"]
        ).eq(True)

        price_lookup = events.set_index("source_row")
        for reason in self.PRICE_AUDIT_COLUMNS:
            flags[reason] = flags["source_line"].map(price_lookup[reason]).eq(True)
        flags["suspicious_price"] = flags[self.PRICE_AUDIT_COLUMNS].any(axis=1)
        return flags

    def _process_source(
        self,
        source_path: Path,
        symbol: str,
        timeframe: str,
    ) -> tuple[pd.DataFrame, dict, dict, pd.DataFrame, pd.DataFrame]:
        raw = pd.read_csv(source_path)
        required = {"time", *self.PRICE_COLUMNS, "volume"}
        missing = required.difference(raw.columns)
        if missing:
            raise ValueError(f"{source_path}: missing required columns {sorted(missing)}")

        quality_summary, gaps, events = self._inspect_frame(
            raw, symbol, timeframe
        )
        interval = self.BAR_INTERVALS[timeframe]
        df, audit = self.load_clean_bars(source_path, interval, raw_data=raw)
        flags = self._quality_flags(raw, gaps, events)
        df = df.merge(flags, on="source_line", how="left", validate="one_to_one")
        df = self.add_indicators(df)
        df = self.add_next_bar_targets(df, interval)
        df.insert(0, "symbol", symbol)
        df.insert(1, "timeframe", timeframe)
        return df, audit, quality_summary, gaps, events

    @staticmethod
    def _segment_data(df: pd.DataFrame, freq: str = "7D") -> list[dict]:
        segments = []
        for start, group in df.groupby(pd.Grouper(key="time", freq=freq)):
            if not group.empty:
                segments.append({
                    "start": start,
                    "end": group["time"].max(),
                    "rows": len(group),
                })
        return segments

    def _walk_forward_split(
        self,
        segments: list[dict],
    ) -> list[dict[str, tuple[int, int]]]:
        splits = []
        train_end = self.train_size
        while train_end + self.val_size + self.test_size <= len(segments):
            val_end = train_end + self.val_size
            splits.append({
                "train": (0, train_end),
                "val": (train_end, val_end),
                "test": (val_end, val_end + self.test_size),
            })
            train_end += self.step
        return splits

    @staticmethod
    def _combine_segments(
        df: pd.DataFrame,
        segments: list[dict],
    ) -> pd.DataFrame:
        if not segments:
            return pd.DataFrame(columns=df.columns)
        start, end = segments[0]["start"], segments[-1]["end"]
        partition = df.loc[df["time"].between(start, end)].copy()
        if "target_end" in partition and "available_at" in partition:
            cutoff = partition["available_at"].max()
            crosses_boundary = partition["target_end"].gt(cutoff)
            partition = partition.loc[~crosses_boundary].copy()
        return partition.reset_index(drop=True)

    def _resolve_key(
        self,
        symbol: str | None,
        timeframe: str | None,
        available: dict,
    ) -> tuple[str, str]:
        if symbol is not None and timeframe is not None:
            key = (symbol, timeframe)
            if key not in available:
                raise KeyError(f"No processed data for {symbol}/{timeframe}.")
            return key
        if symbol is None and timeframe is None and len(available) == 1:
            return next(iter(available))
        raise ValueError("Specify both symbol and timeframe when multiple datasets are configured.")

    def select_fold(
        self,
        fold_index: int = 0,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        key = self._resolve_key(symbol, timeframe, self.splits)
        if isinstance(fold_index, bool) or not isinstance(fold_index, Integral):
            raise ValueError("fold_index must be an integer")
        folds = self.splits[key]
        if not 0 <= fold_index < len(folds):
            segment_count = len(self.segments[key])
            required = self.train_size + self.val_size + self.test_size
            if not folds:
                raise ValueError(
                    f"Need at least {required} observed weeks; found {segment_count}."
                )
            raise IndexError(f"Choose a fold from 0 to {len(folds) - 1}.")

        path = self.processed_paths[key]
        if path is not None:
            df = pd.read_csv(path)
            for column in ("time", "available_at", "target_end"):
                if column in df:
                    df[column] = pd.to_datetime(df[column], utc=True, errors="coerce")
        else:
            symbol, timeframe = key
            source_path = (
                self.data_dir
                / symbol
                / f"{symbol}_{self.SOURCE_TIMEFRAME_SUFFIXES[timeframe]}.csv"
            )
            df, _, _, _, _ = self._process_source(source_path, symbol, timeframe)

        fold = folds[fold_index]
        partitions = {}
        for name, (start, end) in fold.items():
            partitions[name] = self._combine_segments(
                df, self.segments[key][start:end]
            )
        train, evaluation, test = (
            partitions["train"], partitions["val"], partitions["test"]
        )
        if not (
            train["time"].max() < evaluation["time"].min()
            and evaluation["time"].max() < test["time"].min()
        ):
            raise ValueError("Fold partitions must be strictly chronological.")

        self.selected_folds[key] = (train, evaluation, test)
        self.train_data, self.eval_data, self.test_data = train, evaluation, test
        self.fold_index = fold_index
        self.selected_key = key
        return train, evaluation, test

    def get_train_data(
        self, symbol: str | None = None, timeframe: str | None = None
    ) -> pd.DataFrame:
        return self._get_selected_partition("train", symbol, timeframe)

    def get_eval_data(
        self, symbol: str | None = None, timeframe: str | None = None
    ) -> pd.DataFrame:
        return self._get_selected_partition("eval", symbol, timeframe)

    def get_test_data(
        self, symbol: str | None = None, timeframe: str | None = None
    ) -> pd.DataFrame:
        return self._get_selected_partition("test", symbol, timeframe)

    def _get_selected_partition(
        self,
        name: str,
        symbol: str | None,
        timeframe: str | None,
    ) -> pd.DataFrame:
        key = self._resolve_key(symbol, timeframe, self.selected_folds)
        index = {"train": 0, "eval": 1, "test": 2}[name]
        return self.selected_folds[key][index]

    def get_features(self) -> list[str]:
        return self.FEATURE_COLUMNS.copy()

    def get_quality_reports(
        self,
        symbol: str | None = None,
        timeframe: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if symbol is None and timeframe is None:
            return self.quality_summary, self.gap_events, self.price_events
        key = self._resolve_key(symbol, timeframe, self.quality_reports)
        return self.quality_reports[key]

    def get_fold_summary(
        self,
        symbol: str | None = None,
        timeframe: str | None = None,
    ) -> pd.DataFrame:
        key = self._resolve_key(symbol, timeframe, self.splits)
        records = []
        for index, fold in enumerate(self.splits[key]):
            record = {"fold": index}
            for name, (start, end) in fold.items():
                segments = self.segments[key][start:end]
                record.update({
                    f"{name}_start": segments[0]["start"],
                    f"{name}_end": segments[-1]["end"],
                    f"{name}_weeks": len(segments),
                    f"{name}_rows": sum(segment["rows"] for segment in segments),
                })
            records.append(record)
        return pd.DataFrame(records)

    def run_preprocessing(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
        if self.bar_timestamp not in {"open", "close"}:
            raise ValueError("bar_timestamp must be 'open' or 'close'.")
        unsupported = sorted(set(self.timeframes) - set(self.BAR_INTERVALS))
        if unsupported:
            raise ValueError(f"Unsupported timeframes: {unsupported}")
        if self.data_as_of_utc.tzinfo is None:
            raise ValueError("data_as_of_utc must include a timezone.")
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"Set data_dir to the candle CSV directory. Checked {self.data_dir}"
            )
        if self.write_outputs and self.output_dir == self.data_dir:
            raise ValueError("Use a separate output directory to protect source CSVs.")

        summaries = []
        missingness = []
        quality_summaries = []
        gap_events = []
        price_events = []
        self.quality_reports.clear()
        self.segments.clear()
        self.splits.clear()
        self.processed_paths.clear()
        self.selected_folds.clear()
        self.train_data = pd.DataFrame()
        self.eval_data = pd.DataFrame()
        self.test_data = pd.DataFrame()
        preview = None

        for symbol in self.symbols:
            for timeframe in self.timeframes:
                identity = {"symbol": symbol, "timeframe": timeframe}
                key = (symbol, timeframe)
                suffix = self.SOURCE_TIMEFRAME_SUFFIXES[timeframe]
                source_path = self.data_dir / symbol / f"{symbol}_{suffix}.csv"
                if not source_path.is_file():
                    summaries.append({**identity, "status": "file missing"})
                    continue

                print(f"Processing {symbol} {timeframe}", flush=True)
                (
                    df,
                    audit,
                    quality_summary,
                    gaps,
                    events,
                ) = self._process_source(source_path, symbol, timeframe)
                quality_summaries.append(quality_summary)
                gap_events.append(gaps)
                price_events.append(events)
                self.quality_reports[key] = (
                    pd.DataFrame([quality_summary]),
                    gaps,
                    events,
                )
                self.segments[key] = self._segment_data(df)
                self.splits[key] = self._walk_forward_split(self.segments[key])

                features = df[self.FEATURE_COLUMNS]
                labeled = df["target_end"].notna()
                summaries.append({
                    **identity,
                    "status": "processed",
                    **audit,
                    "labeled_rows": int(labeled.sum()),
                    "flat_target_rows": int(df["target_log_return"].eq(0).sum()),
                    "rows_with_all_features_and_label": int(
                        (labeled & features.notna().all(axis=1)).sum()
                    ),
                })
                after_warmup = features.iloc[self.MAX_WARMUP_ROWS :]
                missing_total = features.isna().sum()
                missing_after_warmup = after_warmup.isna().sum()
                for feature in self.FEATURE_COLUMNS:
                    missingness.append({
                        **identity,
                        "feature": feature,
                        "missing_total": int(missing_total[feature]),
                        "rows_after_warmup": len(after_warmup),
                        "missing_after_warmup": int(
                            missing_after_warmup[feature]
                        ),
                    })

                del features, after_warmup

                if preview is None and not df.empty:
                    preview = df[[
                        "symbol", "timeframe", "available_at", "close",
                        "relative_volume_20", "atr_sma_14", "cci_14", "cci_20",
                        "target_end", "target_up",
                    ]].tail(8).copy()

                if self.write_outputs:
                    destination = (
                        self.output_dir
                        / symbol
                        / f"{symbol}_{timeframe}_features.csv"
                    )
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with NamedTemporaryFile(
                        dir=destination.parent, suffix=".tmp", delete=False
                    ) as handle:
                        temporary_path = Path(handle.name)
                    try:
                        df.to_csv(temporary_path, index=False)
                        temporary_path.replace(destination)
                    finally:
                        temporary_path.unlink(missing_ok=True)
                    self.processed_paths[key] = destination
                else:
                    self.processed_paths[key] = None

                del df

        self.quality_summary = pd.DataFrame(quality_summaries)
        self.gap_events = (
            pd.concat(gap_events, ignore_index=True)
            if gap_events
            else pd.DataFrame()
        )
        self.price_events = (
            pd.concat(price_events, ignore_index=True)
            if price_events
            else pd.DataFrame()
        )
        if not missingness:
            print("No matching CSVs were processed. Check data_dir, symbols and timeframes.")
        return pd.DataFrame(summaries), pd.DataFrame(missingness), preview


if __name__ == "__main__":
    processor = preprocess()
    quality_summary, feature_audit, preview = processor.run_preprocessing()
    print(quality_summary.to_string(index=False))
    if not feature_audit.empty:
        missing_features = feature_audit["missing_after_warmup"] > 0
        print(feature_audit.loc[missing_features].head(30).to_string(index=False))
    if preview is not None:
        print(preview.to_string(index=False))