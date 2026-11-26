"""Download and build strictly causal Binance Futures + Spot derivatives datasets.

Saves all outputs under `datapoints_new/<SYMBOL>/` without touching `datapoints/`.

Sources (Binance Public Data Archive — free, no API key required):
  1. USD-M Futures 1h klines: OHLCV, trade count, exact taker buy/sell volume
  2. Spot 1h klines: spot close, volume, exact spot taker buy/sell volume (for perp-spot basis)
  3. USD-M Futures 1h premium index klines: hourly perpetual-vs-index premium OHLC
  4. USD-M Futures 8h funding rate history: causally aligned via merge_asof on close_time
  5. (Optional, --include-metrics) Daily 5m Open Interest & Long/Short ratios aggregated to 1h

Usage:
    python3 new_data.py --symbols BTC --start 2020-01
    python3 new_data.py --symbols BTC ETH --start 2020-01 --include-metrics
"""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import pathlib
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd

BASE_URL = "https://data.binance.vision"
PROJECT_DIR = pathlib.Path(__file__).resolve().parent
OUT_DIR = PROJECT_DIR / "datapoints_new"

KLINE_COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]


def _month_range(start_ym: str, end_ym: str) -> list[str]:
    start = pd.Timestamp(f"{start_ym}-01", tz="UTC")
    end = pd.Timestamp(f"{end_ym}-01", tz="UTC")
    months = pd.date_range(start=start, end=end, freq="MS", tz="UTC")
    return [m.strftime("%Y-%m") for m in months]


def _day_range(start_ymd: str, end_ymd: str) -> list[str]:
    days = pd.date_range(start=start_ymd, end=end_ymd, freq="D", tz="UTC")
    return [d.strftime("%Y-%m-%d") for d in days]


def _parse_binance_ts(series: pd.Series) -> pd.DatetimeIndex:
    """Convert Binance epoch timestamps (handles both ms and post-2025 us timestamps)."""
    vals = pd.to_numeric(series, errors="coerce")
    # Post-Jan-2025 Binance spot archives use microseconds (> 1e14); earlier use milliseconds (~1.7e12).
    is_us = vals > 1e14
    ms_vals = np.where(is_us, vals // 1000, vals)
    return pd.to_datetime(ms_vals, unit="ms", utc=True)


def _fetch_zip_csv(url: str, expected_cols: list[str] | None = None, timeout: int = 20) -> pd.DataFrame | None:
    """Download a single Binance Vision ZIP archive and return its CSV as a DataFrame (or None if 404)."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code in (404, 403):
            return None
        raise
    except Exception:
        return None

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            return None
        with zf.open(csv_names[0]) as f:
            head_line = f.readline().decode("utf-8", errors="ignore").strip()
            f.seek(0)
            first_token = head_line.split(",")[0].strip().strip('"')
            has_header = not first_token.isdigit()
            if has_header:
                df = pd.read_csv(f)
            elif expected_cols is not None:
                df = pd.read_csv(f, header=None, names=expected_cols)
            else:
                df = pd.read_csv(f, header=None)
    return df


def _fetch_many(urls: list[str], expected_cols: list[str] | None = None, max_workers: int = 12) -> pd.DataFrame:
    frames = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_fetch_zip_csv, u, expected_cols) for u in urls]
        for fut in concurrent.futures.as_completed(futures):
            df = fut.result()
            if df is not None and not df.empty:
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def download_klines(market: str, pair: str, timeframe: str, months: list[str]) -> pd.DataFrame:
    """Download monthly klines for market in {'futures/um', 'spot'}."""
    urls = [
        f"{BASE_URL}/data/{market}/monthly/klines/{pair}/{timeframe}/{pair}-{timeframe}-{ym}.zip"
        for ym in months
    ]
    df = _fetch_many(urls, expected_cols=KLINE_COLS)
    if df.empty:
        return df

    # Normalize column names across header/headerless files
    if len(df.columns) >= 11:
        df = df.iloc[:, :11].copy()
        df.columns = KLINE_COLS[:11]
    df["time"] = _parse_binance_ts(df["open_time"])
    num_cols = ["open", "high", "low", "close", "volume", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["time", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    return df[["time"] + num_cols]


def download_premium_index(pair: str, timeframe: str, months: list[str]) -> pd.DataFrame:
    """Download monthly USD-M Futures premium index klines."""
    urls = [
        f"{BASE_URL}/data/futures/um/monthly/premiumIndexKlines/{pair}/{timeframe}/{pair}-{timeframe}-{ym}.zip"
        for ym in months
    ]
    df = _fetch_many(urls, expected_cols=KLINE_COLS)
    if df.empty:
        return df
    df = df.iloc[:, :5].copy()
    df.columns = ["open_time", "premium_open", "premium_high", "premium_low", "premium_close"]
    df["time"] = _parse_binance_ts(df["open_time"])
    for col in ("premium_open", "premium_high", "premium_low", "premium_close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["time", "premium_close"]).drop_duplicates(subset="time").sort_values("time")
    return df[["time", "premium_open", "premium_high", "premium_low", "premium_close"]].reset_index(drop=True)


def download_funding_rates(pair: str, months: list[str]) -> pd.DataFrame:
    """Download monthly USD-M Futures 8h funding rate history."""
    urls = [
        f"{BASE_URL}/data/futures/um/monthly/fundingRate/{pair}/{pair}-fundingRate-{ym}.zip"
        for ym in months
    ]
    df = _fetch_many(urls, expected_cols=["calc_time", "funding_interval_hours", "last_funding_rate"])
    if df.empty:
        return df
    # Normalize possible column names
    time_col = "calc_time" if "calc_time" in df.columns else df.columns[0]
    rate_col = "last_funding_rate" if "last_funding_rate" in df.columns else df.columns[-1]
    out = pd.DataFrame(
        {
            "funding_time": _parse_binance_ts(df[time_col]),
            "funding_rate": pd.to_numeric(df[rate_col], errors="coerce"),
        }
    )
    return out.dropna().drop_duplicates(subset="funding_time").sort_values("funding_time").reset_index(drop=True)


def download_futures_metrics(pair: str, start_ymd: str, end_ymd: str) -> pd.DataFrame:
    """Download daily 5m Open Interest & Long/Short metrics (available from 2021-12-01) and aggregate causally to 1h."""
    eff_start = max(pd.Timestamp(start_ymd, tz="UTC"), pd.Timestamp("2021-12-01", tz="UTC")).strftime("%Y-%m-%d")
    days = _day_range(eff_start, end_ymd)
    urls = [
        f"{BASE_URL}/data/futures/um/daily/metrics/{pair}/{pair}-metrics-{d}.zip"
        for d in days
    ]
    print(f"  [{pair}] Fetching {len(urls)} daily OI/Long-Short metric archives ({eff_start} -> {end_ymd})...")
    df = _fetch_many(urls, max_workers=20)
    if df.empty:
        return df
    df["create_time"] = pd.to_datetime(df["create_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["create_time"]).sort_values("create_time")
    # Causal hourly snapshot: take the last 5m metric snapshot inside [t, t+1h)
    df["time"] = df["create_time"].dt.floor("1h")
    keep_cols = [
        "sum_open_interest",
        "sum_open_interest_value",
        "count_toptrader_long_short_ratio",
        "sum_toptrader_long_short_ratio",
        "count_long_short_ratio",
        "sum_taker_long_short_vol_ratio",
    ]
    present = [c for c in keep_cols if c in df.columns]
    for col in present:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    hourly = df.groupby("time", sort=True)[present].last().reset_index()
    return hourly


def build_symbol_dataset(
    symbol: str,
    start_ym: str,
    end_ym: str,
    include_metrics: bool = False,
) -> pathlib.Path:
    pair = f"{symbol.upper()}USDT"
    sym_dir = OUT_DIR / symbol.upper()
    sym_dir.mkdir(parents=True, exist_ok=True)
    months = _month_range(start_ym, end_ym)

    print(f"\n=== Downloading {pair} ({start_ym} -> {end_ym}, {len(months)} months) ===")
    fut_1h = download_klines("futures/um", pair, "1h", months)
    if fut_1h.empty:
        raise RuntimeError(f"No futures 1h klines downloaded for {pair}. Check network access or symbol.")
    print(f"  Futures 1h klines : {len(fut_1h):,} rows ({fut_1h['time'].iloc[0]} -> {fut_1h['time'].iloc[-1]})")

    spot_1h = download_klines("spot", pair, "1h", months)
    print(f"  Spot 1h klines    : {len(spot_1h):,} rows")

    prem_1h = download_premium_index(pair, "1h", months)
    print(f"  Premium 1h klines : {len(prem_1h):,} rows")

    funding = download_funding_rates(pair, months)
    print(f"  Funding 8h rates  : {len(funding):,} rows")

    # Also save 4h and 15m futures klines so multi-timeframe models work seamlessly on datapoints_new/
    fut_4h = download_klines("futures/um", pair, "4h", months)
    print(f"  Futures 4h klines : {len(fut_4h):,} rows")

    # Build enriched strictly-causal 1h table
    df = fut_1h.copy()
    eps = 1e-10
    df["taker_sell_volume"] = (df["volume"] - df["taker_buy_volume"]).clip(lower=0.0)
    df["volume_delta"] = df["taker_buy_volume"] - df["taker_sell_volume"]
    df["taker_buy_ratio"] = (df["taker_buy_volume"] / (df["volume"] + eps)).clip(0.0, 1.0)
    df["avg_trade_size"] = df["quote_volume"] / (df["count"] + eps)

    if not spot_1h.empty:
        spot_sub = spot_1h[["time", "close", "volume", "count", "taker_buy_volume"]].rename(
            columns={
                "close": "spot_close",
                "volume": "spot_volume",
                "count": "spot_count",
                "taker_buy_volume": "spot_taker_buy_volume",
            }
        )
        df = df.merge(spot_sub, on="time", how="left")
        df["spot_taker_buy_ratio"] = (df["spot_taker_buy_volume"] / (df["spot_volume"] + eps)).clip(0.0, 1.0)
        df["perp_spot_basis"] = (df["close"] - df["spot_close"]) / (df["spot_close"] + eps)
        df["perp_to_spot_vol_ratio"] = df["volume"] / (df["spot_volume"] + eps)

    if not prem_1h.empty:
        df = df.merge(prem_1h, on="time", how="left")

    if not funding.empty:
        df["candle_close_time"] = df["time"] + pd.Timedelta("1h")
        df = pd.merge_asof(
            df.sort_values("candle_close_time"),
            funding.sort_values("funding_time"),
            left_on="candle_close_time",
            right_on="funding_time",
            direction="backward",
            allow_exact_matches=True,
        )
        df["hours_since_funding"] = (
            (df["candle_close_time"] - df["funding_time"]).dt.total_seconds() / 3600.0
        )
        df = df.drop(columns=["candle_close_time", "funding_time"])

    if include_metrics:
        end_ymd = (pd.Timestamp(f"{end_ym}-01", tz="UTC") + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
        metrics_1h = download_futures_metrics(pair, f"{start_ym}-01", end_ymd)
        print(f"  Metrics 1h (OI/LS): {len(metrics_1h):,} rows")
        if not metrics_1h.empty:
            df = df.merge(metrics_1h, on="time", how="left")

    df = df.sort_values("time").drop_duplicates(subset="time").reset_index(drop=True)

    # Save enriched 1h dataset + clean 4h dataset
    out_1h = sym_dir / f"{symbol.upper()}_1h.csv"
    df.to_csv(out_1h, index=False)
    print(f"Saved enriched 1h dataset -> {out_1h} ({len(df):,} rows, {len(df.columns)} columns)")

    if not fut_4h.empty:
        fut_4h["taker_sell_volume"] = (fut_4h["volume"] - fut_4h["taker_buy_volume"]).clip(lower=0.0)
        fut_4h["volume_delta"] = fut_4h["taker_buy_volume"] - fut_4h["taker_sell_volume"]
        fut_4h["taker_buy_ratio"] = (fut_4h["taker_buy_volume"] / (fut_4h["volume"] + eps)).clip(0.0, 1.0)
        out_4h = sym_dir / f"{symbol.upper()}_4h.csv"
        fut_4h.to_csv(out_4h, index=False)
        print(f"Saved futures 4h dataset  -> {out_4h} ({len(fut_4h):,} rows, {len(fut_4h.columns)} columns)")

    return out_1h


def main():
    now_ym = datetime.now(timezone.utc).strftime("%Y-%m")
    parser = argparse.ArgumentParser(
        description="Download Binance USD-M Futures + Spot derivatives data into datapoints_new/"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC"],
        help="Crypto base symbols to download (default: BTC; e.g., BTC ETH)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2020-01",
        help="Start month YYYY-MM (default: 2020-01)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=now_ym,
        help=f"End month YYYY-MM (default: {now_ym})",
    )
    parser.add_argument(
        "--include-metrics",
        action="store_true",
        help="Also download daily 5m Open Interest & Long/Short ratio archives (slower, ~1,500 files)",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sym in args.symbols:
        build_symbol_dataset(sym, args.start, args.end, include_metrics=args.include_metrics)


if __name__ == "__main__":
    main()
