"""Command-line interface for the multi-agent trading framework.

Examples::

    python main.py --mode backtest --symbol BTCUSDT --start 2023-01-01 --end 2024-01-01 --synthetic
    python main.py --mode backtest --symbol BTCUSDT --start 2023-01-01 --end 2024-01-01 --synthetic --walk-forward
    python main.py --mode paper --symbol BTCUSDT --synthetic
    python main.py --market nse --real --symbol RELIANCE --start 2015-01-01 --end 2026-01-01
    python main.py --market nse --real --symbols RELIANCE,TCS,HDFCBANK --start 2015-01-01 --end 2026-01-01

Dates are midnight in the market timezone (UTC for crypto, Asia/Kolkata for NSE).
The range is [start, end): end is exclusive.
"""

from __future__ import annotations

import argparse
import os
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from core.config import ConfigError, FrameworkConfig, load_config
from core.feature_bus import compute_features
from data.feed import DataValidationError, LiveFeedStub, LiveFeedUnavailableError, MarketData, SyntheticFeed
from data.providers import CsvProvider, DataSourceError, KiteProvider, YahooProvider, attach_market_context
from core.models import AgentScore
from execution.backtester import BacktestResult, Backtester, bars_per_day, run_ablation, run_benchmarks, run_sensitivity
from execution.evaluator import Evaluator
from execution.designs import compare_designs, write_designs_csv
from execution.portfolio_test import run_portfolio_test
from execution.robustness import run_robustness, write_robustness_csv
from execution.daily_signal import run_daily_signal
from execution.dashboard import build_dashboard
from execution.universe import load_universe, run_universe, write_universe_csv
from execution.walk_forward import run_walk_forward
import reporting

logger = logging.getLogger("main")


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser (all defaults documented in --help)."""
    p = argparse.ArgumentParser(
        description="Multi-agent trading strategy framework (research/simulation only; never sends real orders).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["backtest", "paper"], default="backtest", help="backtest or paper (simulated) execution")
    p.add_argument("--market", choices=["crypto", "nse"], default="crypto",
                   help="market; selects the default config (config.yaml or config_nse.yaml)")
    p.add_argument("--symbol", default=None,
                   help="instrument (default BTCUSDT for crypto; required for NSE, e.g. RELIANCE, NIFTY50). "
                        "Synthetic mode: label only")
    p.add_argument("--symbols", default=None,
                   help="comma-separated list (real data, backtest mode): run each symbol, print tables and agent verdicts")
    p.add_argument("--jobs", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)),
                   help="parallel worker processes for --symbols/--universe runs")
    p.add_argument("--compare-designs", action="store_true",
                   help="Step 3: compare ensemble vs trend-only vs stay-invested designs (needs --symbols or --universe)")
    p.add_argument("--daily-signal", action="store_true",
                   help="Step 7: today's paper-trading signal for the Nifty crash filter basket (run after the close)")
    p.add_argument("--site-dir", default=None, help="folder for the web dashboard (default: site/)")
    p.add_argument("--paper-dir", default=None, help="folder for the paper book state, journal and orders (default: paper/)")
    p.add_argument("--robustness", action="store_true",
                   help="Step 5: robustness test of the Nifty crash filter (option C) vs holding the basket")
    p.add_argument("--portfolio-test", action="store_true",
                   help="Step 3c: equal-weight basket: hold vs per-stock trend vs Nifty crash filter (needs --universe/--symbols)")
    p.add_argument("--universe", default=None,
                   help="universe name in data/universes (e.g. nifty50) or a path to a symbol-list file")
    p.add_argument("--start", default="2023-01-01", help="start date, inclusive (YYYY-MM-DD, market timezone)")
    p.add_argument("--end", default="2024-01-01", help="end date, exclusive (YYYY-MM-DD, market timezone)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", action="store_true", help="use the deterministic synthetic feed (all data labelled SYNTHETIC)")
    src.add_argument("--real", action="store_true", help="use real historical data (NSE; see --source)")
    src.add_argument("--live-feed", action="store_true", help="use the live-feed STUB (raises: no provider connected)")
    p.add_argument("--source", choices=["yahoo", "kite", "csv"], default="yahoo",
                   help="real-data source: yahoo (free), kite (needs KITE_API_KEY/KITE_ACCESS_TOKEN), csv (--csv)")
    p.add_argument("--csv", default=None, help="CSV path for --source csv (columns: date,open,high,low,close,volume)")
    p.add_argument("--cache-dir", default=str(Path(__file__).with_name("data_cache")),
                   help="where downloaded real data is cached")
    p.add_argument("--refresh-data", action="store_true", help="ignore the cache and download again")
    p.add_argument("--walk-forward", action="store_true", help="run chronological walk-forward validation (backtest mode)")
    p.add_argument("--seed", type=int, default=None, help="synthetic seed (default: app.seed from config)")
    p.add_argument("--config", default=None, help="YAML config path (default: config.yaml, or config_nse.yaml with --market nse)")
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default=None, help="override app.log_level")
    p.add_argument("--no-ablation", action="store_true", help="skip execution-level agent ablation (5 extra backtests)")
    p.add_argument("--sensitivity", action="store_true", help="run threshold/regime-weight sensitivity grid (18 backtests)")
    p.add_argument("--show-decision", type=int, default=None,
                   help="bar offset (from --start) to print in full; default: first directional decision")
    p.add_argument("--paper-bars", type=int, default=300, help="paper mode: number of bars to replay from --start")
    p.add_argument("--output-dir", default=None, help="write equity/trades/decisions CSV and leaderboard JSON here")
    return p


def parse_market_date(text: str, tz: str) -> "pd.Timestamp":
    """Parse YYYY-MM-DD as midnight in the market timezone (aware inputs are converted)."""
    ts = pd.Timestamp(text)
    ts = ts.tz_localize(tz) if ts.tzinfo is None else ts.tz_convert(tz)
    return ts.to_pydatetime()


def _load_data(args: argparse.Namespace, cfg: FrameworkConfig, symbol: str) -> MarketData:
    """Fetch data from the selected source (never silently falls back)."""
    start, end = parse_market_date(args.start, cfg.app.timezone), parse_market_date(args.end, cfg.app.timezone)
    if end <= start:
        raise DataValidationError(f"--end ({args.end}) must be after --start ({args.start}); end is exclusive")
    if args.live_feed:
        return LiveFeedStub().get_bars(symbol, start, end)
    if args.real:
        if cfg.data.market != "nse":
            raise DataSourceError("real data is currently implemented for NSE only: add --market nse")
        if args.source == "csv":
            if not args.csv:
                raise DataSourceError("--source csv requires --csv PATH")
            return CsvProvider(cfg, args.csv).get_bars(symbol, start, end)
        prov = (KiteProvider(cfg, args.cache_dir, args.refresh_data) if args.source == "kite"
                else YahooProvider(cfg, args.cache_dir, args.refresh_data))
        md = prov.get_bars(symbol, start, end)
        if cfg.features.sentiment_slot == "market_mood":
            md = attach_market_context(md, prov, start, end)
        return md
    return SyntheticFeed(cfg, seed=args.seed).get_bars(symbol, start, end)


def _data_note(md: MarketData) -> str:
    """Honest caveat line depending on the data source."""
    if md.source.startswith("synthetic"):
        return "NOTE: synthetic data - results say nothing about real-market profitability."
    unavailable = sorted(k for k, v in md.availability.items() if v.value == "unavailable")
    return (
        "NOTE: real historical OHLCV (OBSERVED). Unavailable feeds (agents needing them abstain): "
        + ", ".join(unavailable)
        + ". Past performance on one symbol is not evidence of a durable edge; costs approximate, taxes not modelled."
    )


def _run_backtest(args: argparse.Namespace, cfg: FrameworkConfig, md: MarketData) -> list[str]:
    """Run a single backtest with benchmarks, leaderboard, ablation and adversarial report."""
    ff = compute_features(md, cfg)
    start, end = ff.first_trade_index, len(ff)
    bt = Backtester(cfg, ff)
    res = bt.run(start, end)
    bpd = bars_per_day(cfg)
    out = [f"Data source: {md.source} | symbol {md.symbol} | market {cfg.data.market} | timeframe {cfg.data.timeframe} | "
           f"bars traded {end - start} ({ff.index[start]} .. {ff.index[end - 1]}, {cfg.app.timezone}) | warm-up bars {start}",
           _data_note(md)]
    pick = args.show_decision
    if pick is None:
        pick = next((k for k, d in enumerate(res.decisions) if d.action.value != "HOLD"), 0)
    if not 0 <= pick < len(res.decisions):
        raise ValueError(f"--show-decision must be in [0, {len(res.decisions) - 1}]")
    out.append(reporting.format_decision(res.decisions[pick], res.signals[pick]))
    closed = [t for t in res.portfolio.trades if t.is_closed]
    out.append(reporting.format_trade(closed[0]) if closed else "== Sample trade ==\n  no closed trades")
    ev = Evaluator(cfg, bt.boss.base_weights)
    scores = ev.score_agents(res.signal_records, ff.column("close"), as_of_index=end - 1, current_weights=res.final_weights, min_index=start)
    out.append(reporting.format_leaderboard(scores, ev.suggest_weights(scores)))
    if not args.no_ablation:
        out.append(reporting.format_ablation(run_ablation(cfg, ff, start, end, res)))
    if args.sensitivity:
        out.append(reporting.format_sensitivity(run_sensitivity(cfg, ff, start, end)))
    if res.adversarial:
        out.append(reporting.format_adversarial(res.adversarial))
    rc = ", ".join(f"{k}={v}" for k, v in sorted(res.rule_counts.items()))
    out.append(f"== Forced-HOLD rule counts (a bar can trigger several) ==\n  {rc or 'none'}")
    ks = "; ".join(f"{e.kind}@{e.timestamp:%Y-%m-%d %H:%M} ({e.detail})" for e in res.kill_switch_events)
    out.append(f"== Kill-switch events ==\n  {ks or 'none'}")
    out.append(reporting.format_benchmarks(res.metrics, run_benchmarks(cfg, ff, start, end)))
    out.append(reporting.format_metrics("Strategy performance (net of fees and slippage)", res.metrics, bpd))
    if res.unfilled_orders:
        out.append(f"  unfilled orders at window end: {len(res.unfilled_orders)} (no next bar; no fill invented)")
    if args.output_dir:
        _write_outputs(Path(args.output_dir), res, scores)
        out.append(f"Outputs written to {args.output_dir}")
    return out


def _write_outputs(path: Path, res: BacktestResult, scores: dict[str, AgentScore]) -> None:
    """Persist equity curve, fills, trades, decisions and leaderboard."""
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"equity": res.equity, "exposure": res.exposures}).to_csv(path / "equity.csv")
    pd.DataFrame([asdict(f) for f in res.portfolio.fills]).to_csv(path / "fills.csv", index=False)
    pd.DataFrame([{**asdict(t), "net_pnl": t.net_pnl} for t in res.portfolio.trades]).to_csv(path / "trades.csv", index=False)
    pd.DataFrame([{
        "timestamp": d.timestamp, "action": d.action.value, "raw_action": d.raw_action.value, "strength": d.confidence,
        "directional_score": d.directional_score, "agreement": d.agreement, "coverage": d.coverage, "regime": d.regime.value,
        "risk_score": d.risk_score, "risk_veto": d.risk_veto, "kill_switch": d.kill_switch_active,
        "forced_hold_rule": d.forced_hold_rule, "reason": d.reason} for d in res.decisions]).to_csv(path / "decisions.csv", index=False)
    (path / "leaderboard.json").write_text(json.dumps({k: asdict(v) for k, v in scores.items()}, indent=2, default=str))


def _run_paper(args: argparse.Namespace, cfg: FrameworkConfig, md: MarketData) -> list[str]:
    """Replay bars one at a time through the PaperExecutor (simulated fills only)."""
    if args.paper_bars < 2:
        raise ValueError("--paper-bars must be >= 2")
    ff = compute_features(md, cfg)
    start = ff.first_trade_index
    end = min(len(ff), start + args.paper_bars)
    bt = Backtester(cfg, ff, paper=True)
    res = bt.run(start, end)
    out = [
        f"PAPER MODE (simulated fills, NO broker connection, no real orders) | source {md.source} | "
        f"symbol {md.symbol} | bars {end - start}",
        _data_note(md),
        f"  simulated fills: {len(res.portfolio.fills)}; closed trades: {res.metrics.n_trades}; "
        f"unfilled at end: {len(res.unfilled_orders)}",
    ]
    for f in res.portfolio.fills[:10]:
        out.append(f"  SIM FILL {f.timestamp} {f.side.value} {f.quantity:.4f} @ {f.exec_price:,.2f} fee {f.fee:,.2f}")
    if len(res.portfolio.fills) > 10:
        out.append(f"  ... {len(res.portfolio.fills) - 10} more simulated fills (see log)")
    out.append(reporting.format_metrics("Paper session performance", res.metrics, bars_per_day(cfg)))
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    try:
        overrides: dict = {}
        if args.seed is not None:
            overrides.setdefault("app", {})["seed"] = args.seed
        if args.walk_forward:
            if args.mode != "backtest":
                raise ConfigError("--walk-forward is only valid with --mode backtest")
            overrides.setdefault("walk_forward", {})["enabled"] = True
        if args.log_level:
            overrides.setdefault("app", {})["log_level"] = args.log_level
        default_cfg = "config_nse.yaml" if args.market == "nse" else "config.yaml"
        cfg = load_config(args.config or str(Path(__file__).with_name(default_cfg)), overrides)
        if cfg.data.market != args.market:
            raise ConfigError(f"--market {args.market} but config has data.market={cfg.data.market}")
        logging.basicConfig(level=cfg.app.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
        symbol = args.symbol or ("BTCUSDT" if args.market == "crypto" else None)
        if (args.compare_designs or args.portfolio_test or args.robustness or args.daily_signal) and not (args.symbols or args.universe):
            raise ConfigError("--compare-designs / --portfolio-test / --robustness / --daily-signal need --symbols or --universe")
        if args.daily_signal:
            # always fresh data from a fixed history start through today; dates on the command line are ignored
            now = pd.Timestamp.now(tz=cfg.app.timezone)
            args.start, args.end = "2014-01-01", (now + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            args.refresh_data = True
            from data.providers import _BaseCacheMarker
            _BaseCacheMarker.enabled = True  # download the shared Nifty / VIX series once per run
        if args.symbols or args.universe:
            if args.symbols and args.universe:
                raise ConfigError("use either --symbols or --universe, not both")
            if not args.real or args.mode != "backtest" or args.walk_forward:
                raise ConfigError("--symbols/--universe are supported with --real in plain backtest mode")
            syms = (load_universe(args.universe) if args.universe
                    else [x.strip().upper() for x in args.symbols.split(",") if x.strip()])
            label = args.universe or "custom list"

            def progress(i: int, n: int, sym: str) -> None:
                print(f"[{i}/{n}] {sym} ...", file=sys.stderr, flush=True)

            out_dir = Path(args.output_dir) if args.output_dir else Path(__file__).with_name("reports")
            stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
            safe = label.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "")
            if args.daily_signal:
                paper_dir = Path(args.paper_dir) if args.paper_dir else Path(__file__).with_name("paper")
                srep = run_daily_signal(cfg, syms, lambda sym: _load_data(args, cfg, sym), paper_dir, label,
                                        progress=progress)
                print(reporting.format_signal(srep, label))
                site = build_dashboard(srep, paper_dir, Path(args.site_dir) if args.site_dir else Path(__file__).with_name("site"), label)
                print(f"\nJournal: {paper_dir / 'journal.csv'}")
                print(f"Dashboard: {site}")
                if srep.orders:
                    print(f"Orders:  {paper_dir / ('orders_for_' + srep.next_session.strftime('%Y%m%d') + '.csv')}")
                return 0
            if args.robustness:
                rrep = run_robustness(cfg, syms, lambda sym: _load_data(args, cfg, sym),
                                      pd.Timestamp(parse_market_date(args.start, cfg.app.timezone)),
                                      pd.Timestamp(parse_market_date(args.end, cfg.app.timezone)) if args.end else None,
                                      progress=progress)
                csv_path = out_dir / f"robustness_{safe}_{stamp}.csv"
                out_dir.mkdir(parents=True, exist_ok=True)
                write_robustness_csv(rrep, csv_path)
                print(reporting.format_robustness(rrep, f"{label} ({args.start} to {args.end})"))
                print(f"\nSaved: {csv_path}")
                return 0
            if args.portfolio_test:
                prep = run_portfolio_test(cfg, syms, lambda sym: _load_data(args, cfg, sym),
                                          pd.Timestamp(parse_market_date(args.start, cfg.app.timezone)),
                                          progress=progress, jobs=args.jobs)
                csv_path = out_dir / f"portfolio_{safe}_{stamp}.csv"
                out_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({r.name: r.equity for r in prep.options}).to_csv(csv_path)
                print(reporting.format_portfolio(prep, f"{label} ({args.start} to {args.end})"))
                print(f"\nSaved daily portfolio values: {csv_path}")
                return 0
            if args.compare_designs:
                drep = compare_designs(cfg, syms, lambda sym: _load_data(args, cfg, sym), progress=progress, jobs=args.jobs)
                csv_path = out_dir / f"designs_{safe}_{stamp}.csv"
                write_designs_csv(drep, csv_path)
                print(reporting.format_designs(drep, f"{label} ({args.start} to {args.end})"))
                print(f"\nSaved: {csv_path}")
                return 0
            rep = run_universe(cfg, syms, lambda sym: _load_data(args, cfg, sym), ablation=not args.no_ablation,
                               progress=progress, jobs=args.jobs)
            csv_path = out_dir / f"universe_{safe}_{stamp}.csv"
            write_universe_csv(rep, csv_path)
            print(reporting.format_universe(rep, f"{label} ({args.start} to {args.end})"))
            print(f"\nSaved: {csv_path} and {csv_path.stem}_agents.csv")
            return 0
        if symbol is None:
            raise ConfigError("--symbol is required for --market nse (e.g. --symbol RELIANCE or NIFTY50)")
        md = _load_data(args, cfg, symbol)
        if args.mode == "paper":
            lines = _run_paper(args, cfg, md)
        elif cfg.walk_forward.enabled:
            ff = compute_features(md, cfg)
            rep = run_walk_forward(cfg, ff)
            lines = [f"Data source: {md.source} | symbol {md.symbol} | {len(ff) - ff.first_trade_index} tradable bars "
                     f"({cfg.app.timezone}, end exclusive)",
                     _data_note(md),
                     reporting.format_walk_forward(rep, bars_per_day(cfg))]
        else:
            lines = _run_backtest(args, cfg, md)
    except LiveFeedUnavailableError as exc:
        print(f"LIVE FEED UNAVAILABLE: {exc}", file=sys.stderr)
        return 2
    except DataSourceError as exc:
        print(f"DATA SOURCE ERROR: {exc}", file=sys.stderr)
        return 2
    except (ConfigError, DataValidationError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print("\n\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
