"""
Developed by Benaymin Eslami

In-memory quality review and weekly walk-forward data preparation.
Quality checks run before segmentation; flags never remove or correct candles.
"""
from pathlib import Path
from numbers import Integral
import numpy as np
import pandas as pd

INTERVALS = {"1m": "1min", "5m": "5min", "15m": "15min",
             "30m": "30min", "1h": "1h", "4h": "4h"}


class Data:
    def __init__(self, symbol, csv_path: str,
                 timeframe: str, *, train_size=12, val_size=2, test_size=2,
                 step=2, min_move=0.15, robust_z=12):
        """Load source candles, audit them, and select the first weekly fold.

        Paths identify source files, not fixed partitions: distinct files are pooled, while a
        repeated path is read once. Sizes and step count observed weekly bins.
        Training always expands; validation and test sizes remain fixed.
        """
        if timeframe not in INTERVALS:
            raise ValueError(f"Unsupported timeframe: {timeframe!r}")
        sizes = (train_size, val_size, test_size, step)
        if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 1 for v in sizes):
            raise ValueError("All sizes and step must be positive integers.")
        if not np.isfinite(min_move) or not np.isfinite(robust_z) or min_move <= 0 or robust_z <= 0:
            raise ValueError("Price thresholds must be finite and positive.")
        self.symbol = symbol
        self.timeframe = timeframe
        self.source_path = csv_path
        self.train_size, self.val_size, self.test_size, self.step = sizes
        self.min_move, self.robust_z = min_move, robust_z
        self.df = pd.DataFrame()
        self.train_data = pd.DataFrame()
        self.eval_data = pd.DataFrame()
        self.test_data = pd.DataFrame()
        self.column = []
        self.__load()

    ## Private methods
    def __load(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Read each unique file once, flag full-history issues, then split."""
        paths = dict.fromkeys(Path(p).expanduser().resolve()
                              for p in (self.source_path,))
        frames = []
        for path in paths:
            frame = pd.read_csv(path)
            # Retain physical CSV provenance even after chronological sorting.
            frame["source_file"] = str(path)
            frame["source_line"] = np.arange(len(frame)) + 2
            frames.append(frame)
        raw = pd.concat(frames, ignore_index=True)
        required = {"time", "open", "high", "low", "close"}
        missing = required.difference(raw.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        # Audit columns are excluded from the original feature-column accessor.
        self.column = [c for c in frames[0].columns if c not in {"source_file", "source_line"}]
        self.df, self.quality_summary, self.gap_events, self.price_events = self.__prepare_dataset(raw)
        self.segments = self.__segment_data(self.df)
        self.splits = self.__walk_forward_split(self.segments)
        if not self.splits:
            raise ValueError(f"Need at least {self.train_size + self.val_size + self.test_size} "
                             f"observed weeks; found {len(self.segments)}.")
        self.select_fold(0)
        return self.train_data, self.eval_data, self.test_data

    def __clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reject empty inputs and drop rows with missing timestamps or OHLC values."""
        assert not df.empty, "DataFrame is empty"
        df = df.copy().dropna().reset_index(drop=True)
        return df

    def __prepare_dataset(self, raw):
        """Attach price and gap flags before any weekly boundaries are made.

        source_row identifies a row in the concatenated input (plus header).
        source_file/source_line locate the original CSV record. Invalid times
        raise an error; missing or suspicious prices remain for manual review.
        """
        raw = self.__clean(raw)
        summary, gaps, prices = self.__inspect_frame(
            raw, self.symbol, self.timeframe, self.min_move, self.robust_z
        )
        frame = raw.copy()
        frame["source_row"] = range(2, len(frame) + 2)
        frame["time"] = pd.to_datetime(frame["time"], errors="coerce", utc=True)
        if frame["time"].isna().any():
            raise ValueError(f"{self.symbol}/{self.timeframe}: invalid timestamps; review source rows "
                             f"{frame.loc[frame.time.isna(), 'source_row'].tolist()[:20]}")
        frame = frame.sort_values("time", kind="stable").reset_index(drop=True)
        frame["duplicate_timestamp"] = frame.time.duplicated(keep=False)
        # Convert OHLC for downstream numerical work; invalid prices retain audit flags.
        for col in ("open", "high", "low", "close"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        gap_lookup = gaps.set_index("source_row")
        frame["time_gap"] = frame.source_row.map(gap_lookup["gap"])
        frame["gap_classification"] = frame.source_row.map(gap_lookup["classification"])
        frame["gap_needs_review"] = frame.source_row.map(gap_lookup["needs_review"]).eq(True)
        price_lookup = prices.set_index("source_row")
        reasons = ["invalid_price", "invalid_ohlc", "large_close_move",
                   "large_candle_range", "isolated_spike_reversal"]
        for reason in reasons:
            frame[reason] = frame.source_row.map(price_lookup[reason]).eq(True)
        frame["suspicious_price"] = frame[reasons].any(axis=1)
        return frame, pd.DataFrame([summary]), gaps, prices

    def __inspect_frame(self, raw, market, timeframe, min_move=0.15, robust_z=12):
        """Return summary, gap events, and price events for one file.

        Weekend allowance applies only to non-crypto Friday-to-Sunday/Monday gaps.
        Session closures are estimated from typical weekday overnight gaps, not
        an exchange calendar. Holidays and unusual sessions remain review items.
        Price thresholds use a trailing median/MAD of absolute log returns.
        """
        # Work on a copy: quality flags must never alter the caller's raw prices.
        # Keep CSV line numbers so a flagged event can be traced to its original row.
        df = raw.copy()
        df["source_row"] = np.arange(len(df)) + 2  # CSV line, including header
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
        invalid_times = int(df.time.isna().sum())
        df = df.dropna(subset=["time"]).sort_values("time", kind="stable").reset_index(drop=True)
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        step = pd.Timedelta(INTERVALS[timeframe])
        # Compare adjacent candles after sorting; crypto has no weekend allowance.
        previous = df.time.shift()
        delta = df.time - previous
        day_jump = (df.time.dt.normalize() - previous.dt.normalize()).dt.days
        crypto = market in {"BTC", "ETH"}
        weekend = ((previous.dt.dayofweek == 4) & (df.time.dt.dayofweek.isin([6, 0]))
                   & (day_jump.isin([2, 3]))) if not crypto else pd.Series(False, index=df.index)
        # Daily session closures are distinct from missing intraday candles.
        overnight = (day_jump == 1) & (previous.dt.dayofweek < 4) & (delta > step)
        baseline = delta[overnight].median() if not crypto else pd.NaT
        # Subtract only the two-day closure, leaving extra missing hours visible.
        adjusted = delta - pd.to_timedelta(weekend.astype(int) * 2, unit="D")
        expected_session = pd.Series(False, index=df.index)
        if pd.notna(baseline):
            # DST transitions may shift UTC session boundaries by one hour.
            tolerance = max(step, pd.Timedelta(hours=1))
            expected_session = ((overnight | weekend) & ((adjusted - baseline).abs() <= tolerance))
        normal_weekend = weekend & ((adjusted <= step) | expected_session)
        has_gap = delta > step
        kind = pd.Series("unexplained_gap", index=df.index)
        kind.loc[expected_session & ~weekend] = "typical_session_closure"
        kind.loc[normal_weekend] = "weekend_closure"
        kind.loc[weekend & ~normal_weekend] = "weekend_plus_extra_gap"
        # Export all gaps, including explained closures, so exclusions stay auditable.
        gaps = pd.DataFrame({"source_row": df.source_row, "previous_time": previous,
                             "time": df.time, "gap": delta, "weekend_adjusted_gap": adjusted,
                             "classification": kind})[has_gap].copy()
        gaps["needs_review"] = ~gaps.classification.isin(["weekend_closure", "typical_session_closure"])

        # Check structural price errors separately from statistically unusual moves.
        prices = df[["open", "high", "low", "close"]]
        invalid = prices.isna().any(axis=1) | prices.le(0).any(axis=1) | (~np.isfinite(prices)).any(axis=1)
        inconsistent = ((df.high < prices[["open", "close", "low"]].max(axis=1))
                        | (df.low > prices[["open", "close", "high"]].min(axis=1)))
        close = df.close.where(df.close > 0)
        # Log returns put upward/downward moves on a symmetric scale. Shifted
        # rolling statistics ensure thresholds use only earlier recorded candles.
        log_return = np.log(close / close.shift())
        magnitude = log_return.abs()
        trailing = magnitude.shift().rolling(200, min_periods=30)
        median = trailing.median()
        # Rolling median absolute deviations from local medians; prior candles only.
        mad = (magnitude - magnitude.rolling(200, min_periods=30).median()).abs().shift().rolling(200, min_periods=30).median()
        threshold = np.maximum(np.log1p(min_move), median + robust_z * 1.4826 * mad)
        threshold = threshold.fillna(np.log1p(min_move))
        big_move = magnitude > threshold
        wick = np.log(df.high.where(df.high > 0) / df.low.where(df.low > 0))
        big_range = wick > threshold * 2
        # A quick reversal is evidence for review, not proof of an erroneous tick.
        # This future-looking check is for data auditing, never a model input.
        next_return = log_return.shift(-1)
        reversal = (big_move & (log_return * next_return < 0)
                    & (next_return.abs() >= magnitude * 0.7)
                    & ((log_return + next_return).abs() <= magnitude * 0.3))
        # Include nearby closes and individual reason flags for manual investigation.
        events = df.loc[invalid | inconsistent | big_move | big_range,
                        ["source_row", "time", "open", "high", "low", "close"]].copy()
        events["previous_close"] = close.shift().reindex(events.index)
        events["next_close"] = close.shift(-1).reindex(events.index)
        events["return_pct"] = (close.pct_change(fill_method=None) * 100).reindex(events.index)
        events["threshold_pct"] = (np.expm1(threshold) * 100).reindex(events.index)
        events["gap_before"] = has_gap.reindex(events.index)
        events["invalid_price"] = invalid.reindex(events.index)
        events["invalid_ohlc"] = inconsistent.reindex(events.index)
        events["large_close_move"] = big_move.reindex(events.index)
        events["large_candle_range"] = big_range.reindex(events.index)
        events["isolated_spike_reversal"] = reversal.reindex(events.index)
        events["assessment"] = "review_move_or_wick"
        events.loc[events.isolated_spike_reversal, "assessment"] = "possible_bad_tick_reversal"
        events.loc[events.invalid_price | events.invalid_ohlc, "assessment"] = "invalid_price_or_ohlc"
        for table in (gaps, events):
            table.insert(0, "timeframe", timeframe)
            table.insert(0, "market", market)
        # Summaries provide one compact quality record per instrument/timeframe.
        summary = dict(market=market, timeframe=timeframe, rows=len(raw),
                       start=df.time.min(), end=df.time.max(), invalid_times=invalid_times,
                       duplicate_timestamps=int(df.time.duplicated(keep=False).sum()),
                       missing_ohlc=int(prices.isna().any(axis=1).sum()),
                       typical_overnight_gap=baseline, total_gaps=len(gaps),
                       gaps_to_review=int(gaps.needs_review.sum()), price_events=len(events),
                       invalid_price_or_ohlc=int((invalid | inconsistent).sum()),
                       spike_reversals=int(reversal.sum()))
        return summary, gaps, events


    def __segment_data(self, df, freq="7D"):
        """Group candles into half-open seven-day UTC bins, skipping empty bins.

        Bins start at midnight on the first observed date. Edge bins may be
        partial; sizes count observed weeks when entire calendar weeks are absent.
        """
        segments = []
        for start, group in df.groupby(pd.Grouper(key="time", freq=freq)):
            if not group.empty:
                segments.append({"start": start, "end": group.time.max(),
                                 "data": group.copy()})
        return segments

    def __walk_forward_split(self, segments):
        """Keep segment references for expanding train and subsequent val/test."""
        splits = []
        train_end = self.train_size
        while train_end + self.val_size + self.test_size <= len(segments):
            val_end = train_end + self.val_size
            splits.append({"train": segments[:train_end],
                           "val": segments[train_end:val_end],
                           "test": segments[val_end:val_end + self.test_size]})
            train_end += self.step
        return splits

    @staticmethod
    def __combine_segments(segments):
        """Materialize only a requested partition instead of copying every fold."""
        if not segments:
            return pd.DataFrame()
        return pd.concat([segment["data"] for segment in segments], ignore_index=True)

    ## Public methods
    def select_fold(self, fold_index=0):
        """Select a zero-based fold and update the existing train/eval/test getters.

        Later folds reuse earlier evaluation periods as training history. Flags
        involving future reversals or estimated sessions are audit-only, never
        model features. Fit preprocessing on this fold's training data only.
        """
        if isinstance(fold_index, bool) or not isinstance(fold_index, Integral):
            raise ValueError("fold_index must be an integer")
        if not 0 <= fold_index < len(self.splits):
            raise IndexError(f"Choose a fold from 0 to {len(self.splits) - 1}.")
        fold = self.splits[fold_index]
        train = self.__combine_segments(fold["train"])
        evaluation = self.__combine_segments(fold["val"])
        test = self.__combine_segments(fold["test"])
        if not (train.time.max() < evaluation.time.min() and evaluation.time.max() < test.time.min()):
            raise ValueError("Fold partitions must be strictly chronological.")
        self.fold_index = fold_index
        self.train_data, self.eval_data, self.test_data = train, evaluation, test
        return train, evaluation, test

    def get_train_data(self) -> pd.DataFrame:
        """Return training candles and audit flags for the selected fold."""
        return self.train_data

    def get_eval_data(self) -> pd.DataFrame:
        """Return validation candles and audit flags for the selected fold."""
        return self.eval_data

    def get_test_data(self) -> pd.DataFrame:
        """Return test candles and audit flags for the selected fold."""
        return self.test_data

    def get_features(self) -> list[str]:
        """Return original CSV column names, excluding added audit columns."""
        return self.column

    def get_quality_reports(self):
        """Return the summary, all gap events, and suspicious-price events."""
        return self.quality_summary, self.gap_events, self.price_events

    def get_fold_summary(self) -> pd.DataFrame:
        """Describe dates, week counts and row counts without copying candles."""
        records = []
        for index, fold in enumerate(self.splits):
            record = {"fold": index}
            for name, segments in fold.items():
                record.update({f"{name}_start": segments[0]["data"].time.min(),
                               f"{name}_end": segments[-1]["data"].time.max(),
                               f"{name}_weeks": len(segments),
                               f"{name}_rows": sum(len(s["data"]) for s in segments)})
            records.append(record)
        return pd.DataFrame(records)
