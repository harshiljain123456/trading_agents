"""Universe runner: per-symbol isolation, pooled stats and agent verdict rules (offline)."""

import os
import tempfile
import unittest
from pathlib import Path

from helpers import ROOT

from core.config import load_config
from data.feed import SyntheticFeed
from execution.universe import (
    SymbolResult, _verdicts, load_universe, run_universe, write_universe_csv,
)
from main import parse_market_date

NSE = os.path.join(ROOT, "config_nse.yaml")


class TestUniverseFile(unittest.TestCase):
    def test_nifty50_has_50_unique_symbols(self):
        syms = load_universe("nifty50")
        self.assertEqual(len(syms), 50)
        self.assertEqual(len(set(syms)), 50)
        self.assertIn("M&M", syms)
        self.assertNotIn("WIPRO", syms)

    def test_former_and_ever_lists(self):
        cur, former, ever = load_universe("nifty50"), load_universe("nifty50_former"), load_universe("nifty50_ever")
        self.assertFalse(set(cur) & set(former))          # no stock is both current and former
        self.assertEqual(set(ever), set(cur) | set(former))
        self.assertIn("YESBANK", former)
        self.assertIn("WIPRO", former)

    def test_custom_file_and_comments(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "u.txt"
            p.write_text("# header\nreliance\nTCS  # it\n\nreliance\n")
            self.assertEqual(load_universe(str(p)), ["RELIANCE", "TCS"])
        with self.assertRaises(ValueError):
            load_universe("no_such_universe")


class TestVerdicts(unittest.TestCase):
    def results(self, helped, hurt, neutral=0):
        out = []
        for i in range(helped):
            out.append(SymbolResult(f"H{i}", ablation={"a": (0.05, 0.1)}, hits={"a": (6, 10)}))
        for i in range(hurt):
            out.append(SymbolResult(f"X{i}", ablation={"a": (-0.05, -0.1)}, hits={"a": (4, 10)}))
        for i in range(neutral):
            out.append(SymbolResult(f"N{i}", ablation={"a": (0.0, 0.0)}))
        return out

    def test_helps_requires_significance(self):
        v = _verdicts(self.results(15, 3))[0]
        self.assertEqual(v.verdict, "HELPS")
        self.assertLess(v.sign_p, 0.05)
        self.assertAlmostEqual(v.hit_rate, (15 * 6 + 3 * 4) / 180)

    def test_small_sample_unclear(self):
        self.assertEqual(_verdicts(self.results(4, 1))[0].verdict, "UNCLEAR")

    def test_hurts(self):
        self.assertEqual(_verdicts(self.results(2, 14))[0].verdict, "HURTS")

    def test_neutral_not_counted(self):
        v = _verdicts(self.results(5, 5, neutral=10))[0]
        self.assertEqual((v.helped, v.hurt, v.n_symbols), (5, 5, 20))
        self.assertEqual(v.verdict, "UNCLEAR")


class TestRunUniverse(unittest.TestCase):
    def test_runs_isolates_errors_and_writes_csv(self):
        cfg = load_config(NSE)
        start = parse_market_date("2018-01-01", cfg.app.timezone)
        end = parse_market_date("2020-01-01", cfg.app.timezone)
        seeds = {"AAA": 1, "BBB": 2}

        def loader(sym):
            if sym == "BAD":
                raise RuntimeError("no data")
            md = SyntheticFeed(cfg, seed=seeds[sym]).get_bars(sym, start, end)
            return md

        rep = run_universe(cfg, ["AAA", "BAD", "BBB"], loader, ablation=True)
        self.assertEqual([r.symbol for r in rep.results], ["AAA", "BAD", "BBB"])
        self.assertIsNotNone(rep.results[1].error)
        self.assertEqual(len(rep.results[0].ablation), 5)
        self.assertEqual({v.agent for v in rep.verdicts}, set(rep.results[0].ablation))
        self.assertEqual(rep.pooled_trades, sum(len(r.trade_pnls) for r in rep.results))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "u.csv"
            write_universe_csv(rep, p)
            self.assertTrue(p.exists() and (Path(tmp) / "u_agents.csv").exists())


if __name__ == "__main__":
    unittest.main()


class TestParallel(unittest.TestCase):
    def test_parallel_matches_sequential(self):
        cfg = load_config(NSE)
        start = parse_market_date("2018-01-01", cfg.app.timezone)
        end = parse_market_date("2019-06-01", cfg.app.timezone)
        loader = lambda sym: SyntheticFeed(cfg, seed=len(sym)).get_bars(sym, start, end)  # noqa: E731
        a = run_universe(cfg, ["AB", "ABC"], loader, ablation=False, jobs=1)
        b = run_universe(cfg, ["AB", "ABC"], loader, ablation=False, jobs=2)
        self.assertEqual([r.ret for r in a.results], [r.ret for r in b.results])
