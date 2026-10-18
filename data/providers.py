"""Real historical market-data providers for NSE (and the shared cleaning/caching layer).

Providers
---------
* ``YahooProvider`` - free, no account. Daily history for NSE stocks/indices
  (``RELIANCE`` -> ``RELIANCE.NS``, ``NIFTY50`` -> ``^NSEI``). Hourly bars are
  limited by Yahoo to roughly the last 730 days. OHLC are adjusted for
  splits *and* dividends using Yahoo's adjusted close (factor = adjclose/close).
* ``KiteProvider`` - Zerodha Kite Connect historical API (needs your own
  Kite Connect app with historical-data access; credentials from the
  ``KITE_API_KEY`` and ``KITE_ACCESS_TOKEN`` environment variables). Requests
  are chunked (2000 days for daily, 400 days for 60-minute) and paced at
  ~3 requests/second. Prices are used as returned by Kite (not dividend-adjusted).
* ``CsvProvider`` - any CSV with ``date, open, high, low, close, volume``
  (e.g. a broker export or NSE bhavcopy you assembled yourself).

Only OHLCV is supplied by these sources, so they are labelled OBSERVED and
every order-flow / depth / sentiment / funding / on-chain column is labelled
UNAVAILABLE. Agents that need those inputs abstain instead of seeing fake
values.

Timestamps use the exchange timezone (Asia/Kolkata). Daily bars are labelled
at the session open (09:15 IST) of their trading date; the strategy still
decides only after the bar closes and fills at the next bar's open.
Downloads are cached as CSV under the cache directory so repeated runs are
offline and reproducible; ``refresh=True`` forces a new download.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from core.config import FrameworkConfig
from core.models import FeatureStatus
from data.feed import CONTEXT, MICROSTRUCTURE, OHLCV, ONCHAIN, OPTIONS, SENTIMENT, DataValidationError, MarketData, validate_bars

logger = logging.getLogger(__name__)

Fetcher = Callable[[str, dict[str, str]], bytes]
USER_AGENT = "Mozilla/5.0 (multi-agent-research-framework)"
NSE_INDEX_ALIASES: dict[str, tuple[str, str]] = {
    # user symbol -> (yahoo ticker, kite tradingsymbol)
    "NIFTY50": ("^NSEI", "NIFTY 50"),
    "NIFTY": ("^NSEI", "NIFTY 50"),
    "BANKNIFTY": ("^NSEBANK", "NIFTY BANK"),
    "INDIAVIX": ("^INDIAVIX", "INDIA VIX"),
}
CONTEXT_SERIES: tuple[tuple[str, str], ...] = (("INDIAVIX", "india_vix"), ("NIFTY50", "index_close"))


class DataSourceError(RuntimeError):
    """Raised when a real data source cannot be reached or returns unusable data."""


def http_get(url: str, headers: dict[str, str], attempts: int = 4, backoff: float = 3.0) -> bytes:
    """GET ``url`` with ``headers`` using the standard library (30 s timeout).

    Rate limits (HTTP 429), server errors (5xx) and network glitches are retried with
    a growing pause (3 s, 6 s, 12 s); other HTTP errors fail at once.

    Raises:
        DataSourceError: With the HTTP status / network reason.
    """
    import time

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **headers})
    host = urllib.parse.urlsplit(url).netloc
    for attempt in range(attempts):
        last = attempt == attempts - 1
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - fixed https hosts
                return resp.read()
        except urllib.error.HTTPError as exc:
            if (exc.code == 429 or exc.code >= 500) and not last:
                time.sleep(backoff * 2 ** attempt)
                continue
            body = exc.read()[:300].decode("utf-8", "replace")
            raise DataSourceError(f"HTTP {exc.code} from {host}: {body}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if not last:
                time.sleep(backoff * 2 ** attempt)
                continue
            reason = getattr(exc, "reason", exc)
            raise DataSourceError(
                f"cannot reach {host}: {reason}. "
                "Check your internet connection or use --source csv with a downloaded file."
            ) from exc
    raise DataSourceError(f"cannot reach {host}")  # pragma: no cover


def clean_ohlcv(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop unusable rows and repair minor OHLC inconsistencies.

    * rows with missing/non-positive open/high/low/close are dropped;
    * duplicate timestamps keep the last row;
    * high/low are widened to contain open and close (vendor rounding glitches);
    * missing volume becomes 0.

    Returns:
        (clean frame sorted by time, counts of dropped/repaired rows).
    """
    out = df.copy()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    price = out[["open", "high", "low", "close"]]
    bad = price.isna().any(axis=1) | (price <= 0).any(axis=1)
    dropped = int(bad.sum())
    out = out[~bad].copy()
    hi = out[["open", "high", "close"]].max(axis=1)
    lo = out[["open", "low", "close"]].min(axis=1)
    repaired = int(((hi > out["high"]) | (lo < out["low"])).sum())
    out["high"], out["low"] = hi, lo
    out["volume"] = out["volume"].fillna(0.0).clip(lower=0.0)
    return out, {"dropped": dropped, "repaired": repaired}


def _session_open_index(ts: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    """Map daily timestamps to 09:15 exchange time on their trading date."""
    local = ts.tz_convert(tz)
    return local.normalize() + pd.Timedelta(hours=9, minutes=15)


def parse_yahoo_chart(payload: dict, interval: str, tz: str = "Asia/Kolkata") -> pd.DataFrame:
    """Convert a Yahoo v8 chart JSON payload into an adjusted OHLCV frame.

    Raises:
        DataSourceError: If Yahoo reports an error or returns no candles.
    """
    chart = payload.get("chart") or {}
    if chart.get("error"):
        raise DataSourceError(f"Yahoo error: {chart['error']}")
    results = chart.get("result") or []
    if not results or not results[0].get("timestamp"):
        raise DataSourceError("Yahoo returned no candles (check the symbol and date range)")
    r = results[0]
    q = r["indicators"]["quote"][0]
    idx = pd.to_datetime(pd.Series(r["timestamp"], dtype="int64"), unit="s", utc=True)
    df = pd.DataFrame(
        {k: pd.to_numeric(pd.Series(q.get(k)), errors="coerce") for k in ("open", "high", "low", "close", "volume")}
    )
    df.index = pd.DatetimeIndex(idx)
    adj = (r["indicators"].get("adjclose") or [{}])[0].get("adjclose")
    if adj is not None:
        a = pd.to_numeric(pd.Series(adj), errors="coerce").to_numpy()
        factor = np.where(np.isfinite(a) & (df["close"].to_numpy() > 0), a / df["close"].to_numpy(), 1.0)
        for col in ("open", "high", "low", "close"):
            df[col] = df[col].to_numpy() * factor
    if interval == "1d":
        df.index = _session_open_index(df.index, tz)
    else:
        df.index = df.index.tz_convert(tz)
    return df


def parse_kite_candles(payload: dict, interval: str, tz: str = "Asia/Kolkata") -> pd.DataFrame:
    """Convert a Kite Connect historical-candles JSON payload into an OHLCV frame."""
    if payload.get("status") != "success":
        raise DataSourceError(f"Kite error: {payload.get('message') or payload}")
    rows = payload.get("data", {}).get("candles") or []
    if not rows:
        return pd.DataFrame(columns=list(OHLCV), index=pd.DatetimeIndex([], tz=tz))
    df = pd.DataFrame([r[:6] for r in rows], columns=["ts", "open", "high", "low", "close", "volume"])
    idx = pd.DatetimeIndex(pd.to_datetime(df.pop("ts"), utc=True)).tz_convert(tz)
    df.index = _session_open_index(idx, tz) if interval == "1d" else idx
    return df.astype(float)


def kite_chunks(start: datetime, end: datetime, interval: str) -> list[tuple[datetime, datetime]]:
    """Split [start, end) into request windows within Kite's per-request day limits."""
    max_days = {"1d": 2000, "1h": 400}[interval]
    out = []
    cur = start
    while cur < end:
        nxt = min(end, cur + timedelta(days=max_days))
        out.append((cur, nxt))
        cur = nxt
    return out


def warmup_calendar_days(config: FrameworkConfig) -> int:
    """Calendar days to fetch before ``start`` so ``warmup_bars`` sessions are available."""
    sessions = config.data.warmup_bars / (7.0 if config.data.timeframe == "1h" else 1.0)
    return int(math.ceil(sessions * 365.0 / 250.0 * 1.1)) + 7


class _BaseCacheMarker:
    """Cache files already (re)downloaded in this process, so ``--refresh-data`` fetches the shared
    Nifty / India VIX series once per run instead of once per stock."""

    enabled: bool = False  # switched on by the daily-signal run
    done: set[str] = set()


class _BaseRealProvider:
    """Shared caching, slicing, labelling and validation for real sources."""

    source = "real"

    def __init__(self, config: FrameworkConfig, cache_dir: str | Path | None = None, refresh: bool = False) -> None:
        """Create the provider; ``cache_dir=None`` disables caching."""
        if config.data.market != "nse":
            raise DataSourceError("real-data providers currently support data.market='nse' only")
        if config.data.timeframe not in ("1d", "1h"):
            raise DataSourceError("NSE data supports timeframe '1d' or '1h'")
        self.config = config
        self.tz = config.app.timezone
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.refresh = refresh

    def _download(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        raise NotImplementedError

    def _cache_path(self, symbol: str, start: datetime, end: datetime) -> Path | None:
        if self.cache_dir is None:
            return None
        safe = "".join(ch if ch.isalnum() else "_" for ch in symbol.upper())
        return self.cache_dir / f"{self.source}_{safe}_{self.config.data.timeframe}_{start:%Y%m%d}_{end:%Y%m%d}.csv"

    def get_bars(self, symbol: str, start: datetime, end: datetime) -> MarketData:
        """Return validated OHLCV for ``[start, end)`` plus warm-up history (cached)."""
        if end <= start:
            raise DataValidationError(f"end ({end}) must be after start ({start})")
        fetch_start = start - timedelta(days=warmup_calendar_days(self.config))
        path = self._cache_path(symbol, fetch_start, end)
        if path is not None and path.exists() and (not self.refresh or (_BaseCacheMarker.enabled and str(path) in _BaseCacheMarker.done)):
            raw = pd.read_csv(path, index_col=0)
            raw.index = pd.DatetimeIndex(pd.to_datetime(raw.index, utc=True)).tz_convert(self.tz)
            logger.info("loaded %d cached bars from %s", len(raw), path)
            origin = f"{self.source} (cache {path.name})"
        else:
            raw = self._download(symbol, fetch_start, end)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                raw.to_csv(path)
                _BaseCacheMarker.done.add(str(path))
                logger.info("cached %d bars to %s", len(raw), path)
            origin = self.source
        return self._finalize(symbol, raw, fetch_start, start, end, origin)

    def _finalize(self, symbol: str, raw: pd.DataFrame, fetch_start: datetime, start: datetime, end: datetime, origin: str) -> MarketData:
        """Clean, slice, label provenance, validate."""
        df, counts = clean_ohlcv(raw[list(OHLCV)])
        df = df[(df.index >= pd.Timestamp(fetch_start)) & (df.index < pd.Timestamp(end))]
        first = int((df.index < pd.Timestamp(start)).sum())
        if first == len(df):
            raise DataValidationError(f"no bars for {symbol} inside the requested range {start:%Y-%m-%d}..{end:%Y-%m-%d}")
        if first < self.config.data.warmup_bars:
            logger.warning(
                "%s: only %d warm-up bars before start (configured %d); early indicators may be unavailable",
                symbol, first, self.config.data.warmup_bars,
            )
        if counts["dropped"] or counts["repaired"]:
            logger.warning("%s: dropped %d invalid rows, repaired %d high/low values", symbol, counts["dropped"], counts["repaired"])
        validate_bars(df, min_bars=self.config.data.min_history_bars)
        availability = {c: FeatureStatus.OBSERVED for c in OHLCV}
        for c in MICROSTRUCTURE + SENTIMENT + ONCHAIN + OPTIONS + CONTEXT:
            availability[c] = FeatureStatus.UNAVAILABLE
        if float(df["volume"].sum()) == 0.0:
            logger.warning("%s has no volume data (typical for indices); volume-based agents will abstain", symbol)
        return MarketData(
            symbol=symbol.upper(), bars=df, availability=availability, source=origin, first_trade_index=first,
            meta={"timeframe": self.config.data.timeframe, "dropped": str(counts["dropped"]), "repaired": str(counts["repaired"])},
        )


class YahooProvider(_BaseRealProvider):
    """Free Yahoo Finance chart API (no account). See module docstring for limits."""

    source = "yahoo"

    def __init__(self, config: FrameworkConfig, cache_dir: str | Path | None = None, refresh: bool = False,
                 fetcher: Fetcher = http_get) -> None:
        """Create the provider; ``fetcher`` is injectable for offline tests."""
        super().__init__(config, cache_dir, refresh)
        self.fetcher = fetcher

    @staticmethod
    def ticker(symbol: str) -> str:
        """Map an NSE symbol to a Yahoo ticker (``RELIANCE`` -> ``RELIANCE.NS``)."""
        s = symbol.strip().upper()
        if s in NSE_INDEX_ALIASES:
            return NSE_INDEX_ALIASES[s][0]
        if s.startswith("^") or s.endswith(".NS") or s.endswith(".BO"):
            return s
        return f"{s}.NS"

    def _download(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        interval = self.config.data.timeframe
        if interval == "1h" and (datetime.now(start.tzinfo) - start).days > 725:
            raise DataSourceError(
                "Yahoo only serves ~730 days of hourly data; use timeframe 1d, a later --start, or --source kite"
            )
        q = urllib.parse.urlencode({
            "period1": int(pd.Timestamp(start).timestamp()), "period2": int(pd.Timestamp(end).timestamp()),
            "interval": {"1d": "1d", "1h": "60m"}[interval], "events": "div,splits", "includeAdjustedClose": "true",
        })
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(self.ticker(symbol))}?{q}"
        logger.info("downloading %s %s from Yahoo", self.ticker(symbol), interval)
        try:
            payload = json.loads(self.fetcher(url, {"Accept": "application/json"}))
        except json.JSONDecodeError as exc:
            raise DataSourceError("Yahoo returned a non-JSON response") from exc
        return parse_yahoo_chart(payload, interval, self.tz)


class KiteProvider(_BaseRealProvider):
    """Zerodha Kite Connect historical candles (requires your Kite Connect credentials)."""

    source = "kite"
    BASE = "https://api.kite.trade"

    def __init__(self, config: FrameworkConfig, cache_dir: str | Path | None = None, refresh: bool = False,
                 api_key: str | None = None, access_token: str | None = None, fetcher: Fetcher = http_get,
                 pause_seconds: float = 0.35) -> None:
        """Create the provider; credentials default to KITE_API_KEY / KITE_ACCESS_TOKEN."""
        super().__init__(config, cache_dir, refresh)
        self.api_key = api_key or os.environ.get("KITE_API_KEY", "")
        self.access_token = access_token or os.environ.get("KITE_ACCESS_TOKEN", "")
        self.fetcher = fetcher
        self.pause_seconds = pause_seconds

    def _headers(self) -> dict[str, str]:
        if not self.api_key or not self.access_token:
            raise DataSourceError(
                "Kite credentials missing: set KITE_API_KEY and KITE_ACCESS_TOKEN (today's access token) "
                "as environment variables, or use --source yahoo"
            )
        return {"X-Kite-Version": "3", "Authorization": f"token {self.api_key}:{self.access_token}"}

    def instrument_token(self, symbol: str) -> int:
        """Look up the instrument token for an NSE equity or index tradingsymbol."""
        s = symbol.strip().upper()
        target = NSE_INDEX_ALIASES[s][1] if s in NSE_INDEX_ALIASES else s
        raw = self.fetcher(f"{self.BASE}/instruments/NSE", self._headers()).decode("utf-8")
        for row in csv.DictReader(io.StringIO(raw)):
            if row.get("tradingsymbol", "").upper() == target and row.get("exchange") == "NSE":
                return int(row["instrument_token"])
        raise DataSourceError(f"symbol '{symbol}' not found in Kite's NSE instrument list")

    def _download(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        interval = self.config.data.timeframe
        token = self.instrument_token(symbol)
        kite_interval = {"1d": "day", "1h": "60minute"}[interval]
        frames = []
        for i, (a, b) in enumerate(kite_chunks(start, end, interval)):
            if i:
                time.sleep(self.pause_seconds)  # respect Kite's ~3 requests/second historical limit
            q = urllib.parse.urlencode({"from": f"{a:%Y-%m-%d %H:%M:%S}", "to": f"{b:%Y-%m-%d %H:%M:%S}"})
            url = f"{self.BASE}/instruments/historical/{token}/{kite_interval}?{q}"
            frames.append(parse_kite_candles(json.loads(self.fetcher(url, self._headers())), interval, self.tz))
        if not frames or all(f.empty for f in frames):
            raise DataSourceError(f"Kite returned no candles for {symbol}")
        return pd.concat([f for f in frames if not f.empty])


class CsvProvider(_BaseRealProvider):
    """Load OHLCV from a CSV file with columns date, open, high, low, close, volume."""

    source = "csv"

    def __init__(self, config: FrameworkConfig, path: str | Path) -> None:
        """Create the provider for ``path`` (no caching needed)."""
        super().__init__(config, None, False)
        self.path = Path(path)

    def _download(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        if not self.path.exists():
            raise DataSourceError(f"CSV file not found: {self.path}")
        df = pd.read_csv(self.path)
        df.columns = [c.strip().lower() for c in df.columns]
        date_col = next((c for c in ("date", "datetime", "timestamp", "time") if c in df.columns), None)
        missing = [c for c in OHLCV if c not in df.columns]
        if date_col is None or missing:
            raise DataSourceError(f"CSV needs a date column and {list(OHLCV)}; missing {missing or ['date']}")
        idx = pd.to_datetime(df.pop(date_col))
        idx = pd.DatetimeIndex(idx)
        idx = idx.tz_localize(self.tz) if idx.tz is None else idx.tz_convert(self.tz)
        if self.config.data.timeframe == "1d":
            idx = idx.normalize() + pd.Timedelta(hours=9, minutes=15)
        df.index = idx
        return df[list(OHLCV)].apply(pd.to_numeric, errors="coerce")


def attach_market_context(md: MarketData, provider: _BaseRealProvider, start: datetime, end: datetime) -> MarketData:
    """Join India VIX and Nifty 50 closes onto a stock's bars (for the MarketMoodAgent).

    Each context series is aligned to the stock's bar timestamps with a
    backward as-of match (the latest context value at or before the bar), so
    no future value can leak in. A series that cannot be fetched is labelled
    UNAVAILABLE with a warning; the strategy still runs and the agent abstains.
    """
    bars = md.bars.copy()
    for ctx_symbol, column in CONTEXT_SERIES:
        try:
            ctx = provider.get_bars(ctx_symbol, start, end).bars["close"]
        except (DataSourceError, DataValidationError) as exc:
            logger.warning("market context %s unavailable (%s); market_mood will run without it", ctx_symbol, exc)
            md.availability[column] = FeatureStatus.UNAVAILABLE
            continue
        aligned = ctx.reindex(ctx.index.union(bars.index)).sort_index().ffill().reindex(bars.index)
        bars[column] = aligned.to_numpy()
        md.availability[column] = FeatureStatus.OBSERVED
        logger.info("joined %s (%d of %d bars covered)", ctx_symbol, int(aligned.notna().sum()), len(bars))
    md.bars = bars
    return md
