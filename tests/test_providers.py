"""Offline tests for real-data providers (HTTP is mocked; no network needed)."""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta

from helpers import ROOT, make_state

import numpy as np
import pandas as pd

from agents.market_intel import MarketIntelAgent
from core.config import load_config
from core.feature_bus import compute_features
from core.models import FeatureStatus, SignalType
from data.providers import (
    CsvProvider, DataSourceError, KiteProvider, YahooProvider, clean_ohlcv, kite_chunks, parse_kite_candles,
    parse_yahoo_chart,
)
from execution.backtester import Backtester, bars_per_day

NSE_CFG = os.path.join(ROOT, "config_nse.yaml")
IST = "Asia/Kolkata"


def nse_cfg(**o):
    return load_config(NSE_CFG, o or None)


def ist(y, m, d):
    return pd.Timestamp(datetime(y, m, d)).tz_localize(IST).to_pydatetime()


def fake_daily(n=700, start="2021-01-01"):
    """Business-day random walk OHLCV."""
    rng = np.random.default_rng(3)
    days = pd.bdate_range(start, periods=n)
    c = 1000 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n)))
    o = np.concatenate([[c[0]], c[:-1]]) * np.exp(rng.normal(0, 0.003, n))
    h = np.maximum(o, c) * 1.01
    l = np.minimum(o, c) * 0.99
    v = rng.uniform(1e6, 3e6, n)
    return days, o, h, l, c, v


def yahoo_payload(days, o, h, l, c, v, adj_factor=1.0):
    ts = [int(pd.Timestamp(d).tz_localize(IST).replace(hour=9, minute=15).timestamp()) for d in days]
    return {"chart": {"result": [{
        "meta": {"symbol": "TEST.NS"},
        "timestamp": ts,
        "indicators": {"quote": [{"open": list(o), "high": list(h), "low": list(l), "close": list(c), "volume": list(v)}],
                       "adjclose": [{"adjclose": list(np.asarray(c) * adj_factor)}]},
    }], "error": None}}


class TestParsing(unittest.TestCase):
    def test_yahoo_parse_adjusts_and_labels_ist(self):
        days, o, h, l, c, v = fake_daily(5)
        df = parse_yahoo_chart(yahoo_payload(days, o, h, l, c, v, adj_factor=0.5), "1d")
        self.assertEqual(str(df.index.tz), IST)
        self.assertEqual((df.index[0].hour, df.index[0].minute), (9, 15))
        self.assertAlmostEqual(df["close"].iloc[0], c[0] * 0.5)
        self.assertAlmostEqual(df["high"].iloc[0], h[0] * 0.5)
        self.assertAlmostEqual(df["volume"].iloc[0], v[0])

    def test_yahoo_error_and_empty(self):
        with self.assertRaises(DataSourceError):
            parse_yahoo_chart({"chart": {"result": None, "error": {"code": "Not Found"}}}, "1d")
        with self.assertRaises(DataSourceError):
            parse_yahoo_chart({"chart": {"result": [{}], "error": None}}, "1d")

    def test_yahoo_ticker_mapping(self):
        self.assertEqual(YahooProvider.ticker("reliance"), "RELIANCE.NS")
        self.assertEqual(YahooProvider.ticker("NIFTY50"), "^NSEI")
        self.assertEqual(YahooProvider.ticker("TCS.NS"), "TCS.NS")

    def test_kite_parse(self):
        payload = {"status": "success", "data": {"candles": [
            ["2024-01-02T00:00:00+0530", 100, 105, 99, 104, 1000],
            ["2024-01-03T00:00:00+0530", 104, 106, 101, 102, 1500]]}}
        df = parse_kite_candles(payload, "1d")
        self.assertEqual(len(df), 2)
        self.assertEqual(df.index[0], pd.Timestamp("2024-01-02 09:15", tz=IST))
        with self.assertRaises(DataSourceError):
            parse_kite_candles({"status": "error", "message": "TokenException"}, "1d")

    def test_kite_chunks_cover_range_without_overlap(self):
        a, b = ist(2010, 1, 1), ist(2026, 1, 1)
        ch = kite_chunks(a, b, "1d")
        self.assertEqual(ch[0][0], a)
        self.assertEqual(ch[-1][1], b)
        for (x0, x1), (y0, _) in zip(ch, ch[1:]):
            self.assertEqual(x1, y0)
            self.assertLessEqual(x1 - x0, timedelta(days=2000))
        self.assertTrue(all(x1 - x0 <= timedelta(days=400) for x0, x1 in kite_chunks(a, b, "1h")))

    def test_clean_repairs_and_drops(self):
        idx = pd.date_range("2024-01-01", periods=4, freq="D", tz=IST)
        df = pd.DataFrame({"open": [10, 10, np.nan, 10], "high": [11, 9.5, 11, 11], "low": [9, 9, 9, 10.5],
                           "close": [10.5, 10, 10, 10.2], "volume": [1, np.nan, 1, 1]}, index=idx, dtype=float)
        out, counts = clean_ohlcv(df)
        self.assertEqual(counts["dropped"], 1)
        self.assertEqual(counts["repaired"], 2)
        self.assertEqual(out["high"].iloc[1], 10.0)
        self.assertEqual(out["low"].iloc[2], 10.0)
        self.assertEqual(out["volume"].iloc[1], 0.0)


class TestProviders(unittest.TestCase):
    def setUp(self):
        self.cfg = nse_cfg()
        self.days, o, h, l, c, v = fake_daily(900, "2020-01-01")
        self.payload = json.dumps(yahoo_payload(self.days, o, h, l, c, v)).encode()
        self.calls = []

    def fetch(self, url, headers):
        self.calls.append(url)
        return self.payload

    def test_yahoo_provider_labels_observed_and_warmup(self):
        with tempfile.TemporaryDirectory() as tmp:
            prov = YahooProvider(self.cfg, tmp, fetcher=self.fetch)
            md = prov.get_bars("TEST", ist(2021, 6, 1), ist(2023, 6, 1))
            self.assertEqual(md.availability["close"], FeatureStatus.OBSERVED)
            self.assertEqual(md.availability["bid_volume"], FeatureStatus.UNAVAILABLE)
            self.assertEqual(md.availability["social_sentiment"], FeatureStatus.UNAVAILABLE)
            self.assertGreaterEqual(md.first_trade_index, 250)
            self.assertGreaterEqual(md.bars.index[md.first_trade_index], pd.Timestamp(ist(2021, 6, 1)))
            self.assertLess(md.bars.index[-1], pd.Timestamp(ist(2023, 6, 1)))
            self.assertIn("RELIANCE.NS".replace("RELIANCE", "TEST"), self.calls[0])
            # second call is served from the cache (no new HTTP request)
            md2 = YahooProvider(self.cfg, tmp, fetcher=self.fetch).get_bars("TEST", ist(2021, 6, 1), ist(2023, 6, 1))
            self.assertEqual(len(self.calls), 1)
            self.assertIn("cache", md2.source)
            pd.testing.assert_frame_equal(md.bars, md2.bars, check_freq=False)
            YahooProvider(self.cfg, tmp, refresh=True, fetcher=self.fetch).get_bars("TEST", ist(2021, 6, 1), ist(2023, 6, 1))
            self.assertEqual(len(self.calls), 2)

    def test_end_to_end_backtest_on_observed_ohlcv(self):
        md = YahooProvider(self.cfg, None, fetcher=self.fetch).get_bars("TEST", ist(2021, 6, 1), ist(2023, 6, 1))
        ff = compute_features(md, self.cfg)
        self.assertEqual(ff.availability["ofi"], FeatureStatus.UNAVAILABLE)
        self.assertEqual(ff.state(ff.first_trade_index).value("cvd_is_proxy"), 1.0)
        res = Backtester(self.cfg, ff).run(ff.first_trade_index, len(ff))
        sigs = {s.agent_name: s for s in res.signals[-1]}
        self.assertFalse(sigs["market_mood"].available)               # no VIX/Nifty context joined -> abstains
        self.assertTrue(sigs["market_intel"].available)               # proxy mode
        self.assertTrue(sigs["market_intel"].features["proxy_mode"])
        self.assertIn("NOT real aggressor data", sigs["market_intel"].reason)
        self.assertEqual(res.metrics.n_bars, len(ff) - ff.first_trade_index)
        self.assertEqual(bars_per_day(self.cfg), 1.0)
        self.assertEqual(self.cfg.bars_per_year, 250)

    def test_kite_provider_needs_credentials(self):
        prov = KiteProvider(self.cfg, None, api_key="", access_token="", fetcher=self.fetch)
        prov.api_key = prov.access_token = ""
        with self.assertRaises(DataSourceError):
            prov.get_bars("TEST", ist(2021, 6, 1), ist(2023, 6, 1))

    def test_kite_provider_chunked_download(self):
        instruments = "instrument_token,exchange_token,tradingsymbol,name,exchange\n738561,2885,RELIANCE,RELIANCE IND,NSE\n"
        candles = [[f"{d:%Y-%m-%d}T00:00:00+0530", 100.0, 101.0, 99.0, 100.5, 1000] for d in self.days]
        seen = []

        def fetch(url, headers):
            seen.append((url, headers))
            if url.endswith("/instruments/NSE"):
                return instruments.encode()
            q = dict(p.split("=", 1) for p in url.split("?", 1)[1].split("&"))
            a = pd.Timestamp(q["from"].replace("+", " ").replace("%3A", ":")).date()
            b = pd.Timestamp(q["to"].replace("+", " ").replace("%3A", ":")).date()
            rows = [r for r in candles if a <= pd.Timestamp(r[0][:10]).date() < b]
            return json.dumps({"status": "success", "data": {"candles": rows}}).encode()

        prov = KiteProvider(self.cfg, None, api_key="k", access_token="t", fetcher=fetch, pause_seconds=0)
        md = prov.get_bars("RELIANCE", ist(2021, 6, 1), ist(2023, 6, 1))
        self.assertTrue(all(h["Authorization"] == "token k:t" for _, h in seen))
        self.assertTrue(any("/historical/738561/day" in u for u, _ in seen))
        self.assertFalse(md.bars.index.has_duplicates)
        with self.assertRaises(DataSourceError):
            prov.instrument_token("NOSUCH")

    def test_csv_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.csv")
            days, o, h, l, c, v = fake_daily(600, "2020-01-01")
            pd.DataFrame({"Date": days.strftime("%Y-%m-%d"), "Open": o, "High": h, "Low": l, "Close": c, "Volume": v}).to_csv(path, index=False)
            md = CsvProvider(self.cfg, path).get_bars("X", ist(2021, 3, 1), ist(2022, 3, 1))
            self.assertEqual(md.bars.index[0].hour, 9)
            self.assertGreater(len(md.bars), 300)
            with self.assertRaises(DataSourceError):
                CsvProvider(self.cfg, os.path.join(tmp, "missing.csv")).get_bars("X", ist(2021, 3, 1), ist(2022, 3, 1))

    def test_crypto_config_rejected_for_real_data(self):
        with self.assertRaises(DataSourceError):
            YahooProvider(load_config(os.path.join(ROOT, "config.yaml")), None, fetcher=self.fetch)


class TestNseConfigAndProxy(unittest.TestCase):
    def test_nse_config_rules(self):
        cfg = nse_cfg()
        self.assertTrue(cfg.strategy.long_only)
        with self.assertRaises(Exception):
            nse_cfg(data={"timeframe": "4h"})
        with self.assertRaises(Exception):
            nse_cfg(app={"timezone": "UTC"})
        self.assertEqual(nse_cfg(data={"timeframe": "1h"}).bars_per_year, 1750)

    def test_market_intel_proxy_mode_and_default_off(self):
        cfg = nse_cfg()
        st = make_state({"cvd_norm": 0.6, "inst_momentum": 0.5}, unavailable=("ofi",))
        s = MarketIntelAgent(1.0, cfg.agents.market_intel).analyze(st)
        p = cfg.agents.market_intel
        score = (p.w_cvd * 0.6 + p.w_momentum * 0.5) / (p.w_cvd + p.w_momentum)
        self.assertEqual(s.signal, SignalType.BUY)
        self.assertAlmostEqual(s.confidence, score * p.proxy_confidence_scale)
        crypto = load_config(os.path.join(ROOT, "config.yaml"))
        s2 = MarketIntelAgent(1.0, crypto.agents.market_intel).analyze(st)
        self.assertFalse(s2.available)


if __name__ == "__main__":
    unittest.main()
