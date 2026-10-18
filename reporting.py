"""Plain-text report formatting for the CLI (no printing here; main.py prints)."""

from __future__ import annotations

from typing import Optional

from agents.adversarial_agent import AdversarialReport
from core.config import FrameworkConfig
import numpy as np
from core.models import AgentScore, AgentSignal, BossDecision, PerformanceMetrics, Trade
from execution.backtester import AblationRow, SensitivityRow
from execution.designs import DesignReport
from execution.portfolio_test import PortfolioReport
from execution.universe import UniverseReport
from execution.walk_forward import WalkForwardReport


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:+.2%}"


def _num(x: Optional[float], fmt: str = "{:.2f}") -> str:
    return "n/a" if x is None else fmt.format(x)


def format_metrics(title: str, m: PerformanceMetrics, bars_per_day: float) -> str:
    """Render performance metrics with explicit 'n/a' for unavailable values."""
    hold = "n/a" if m.avg_holding_bars is None else f"{m.avg_holding_bars:.1f} bars ({m.avg_holding_bars / bars_per_day:.1f} days)"
    lines = [
        f"== {title} ==",
        f"  initial equity      {m.initial_equity:,.2f}",
        f"  final equity        {m.final_equity:,.2f}",
        f"  total net return    {_pct(m.total_return)}",
        f"  CAGR                {_pct(m.cagr)}",
        f"  Sharpe (annualized) {_num(m.sharpe)}",
        f"  max drawdown        {m.max_drawdown:.2%}",
        f"  closed trades       {m.n_trades}",
        f"  win rate            {_num(m.win_rate, '{:.1%}')}",
        f"  profit factor       {_num(m.profit_factor)}",
        f"  turnover            {m.turnover:.2f}x mean equity",
        f"  total fees          {m.total_fees:,.2f}",
        f"  est. slippage       {m.total_slippage:,.2f}",
        f"  avg holding period  {hold}",
        f"  avg |exposure|      {m.avg_exposure:.2%}  (long {m.avg_long_exposure:.2%}, short {m.avg_short_exposure:.2%})",
        f"  bars                {m.n_bars}",
    ]
    for k, v in m.notes.items():
        lines.append(f"  note[{k}]: {v}")
    return "\n".join(lines)


def format_benchmarks(strategy: PerformanceMetrics, bench: dict[str, PerformanceMetrics]) -> str:
    """Side-by-side comparison on the identical window and cost model."""
    rows = [("strategy", strategy)] + list(bench.items())
    out = ["== Benchmark comparison (same window, same fees/slippage, mark-to-market) ==",
           f"  {'name':<14}{'return':>10}{'sharpe':>9}{'maxDD':>9}{'trades':>8}{'fees':>11}"]
    for name, m in rows:
        out.append(f"  {name:<14}{_pct(m.total_return):>10}{_num(m.sharpe):>9}{m.max_drawdown:>9.2%}{m.n_trades:>8}{m.total_fees:>11,.0f}")
    return "\n".join(out)


def format_leaderboard(scores: dict[str, AgentScore], suggested: dict[str, float]) -> str:
    """Agent leaderboard from matured outcomes."""
    out = ["== Agent leaderboard (matured outcomes only; hit = direction of 20-bar forward return) ==",
           f"  {'agent':<18}{'preds':>6}{'dir':>6}{'mat':>6}{'hit':>7}{'cw_ret':>10}{'standalone':>11}{'incr(bp)':>10}{'t':>7}{'w_now':>7}{'w_sugg':>8}"]
    ordered = sorted(scores.values(), key=lambda s: (s.incremental_contribution is None, -(s.incremental_contribution or 0)))
    for s in ordered:
        out.append(
            f"  {s.agent_name:<18}{s.prediction_count:>6}{s.directional_count:>6}{s.matured_count:>6}"
            f"{_num(s.hit_rate, '{:.1%}'):>7}{_num(s.confidence_weighted_return, '{:+.4f}'):>10}"
            f"{_num(s.standalone_pnl, '{:+.3f}'):>11}{_num(s.incremental_contribution, '{:+.2f}'):>10}"
            f"{_num(s.incremental_tstat, '{:+.2f}'):>7}{s.current_weight:>7.2f}{suggested.get(s.agent_name, s.current_weight):>8.2f}"
        )
        if s.note:
            out.append(f"      note: {s.note}")
    out.append("  (standalone = non-compounded unit-position return sum after costs; incr = signal-level ablation)")
    return "\n".join(out)


def format_ablation(rows: list[AblationRow]) -> str:
    """Execution-level ablation table."""
    out = ["== Execution-level ablation (full minus without-agent; in-sample, descriptive) ==",
           f"  {'removed agent':<18}{'return w/o':>12}{'sharpe w/o':>12}{'d_return':>11}{'d_sharpe':>10}"]
    for r in rows:
        out.append(f"  {r.agent_name:<18}{_pct(r.total_return_without):>12}{_num(r.sharpe_without):>12}"
                   f"{_pct(r.delta_return):>11}{_num(r.delta_sharpe, '{:+.2f}'):>10}")
    return "\n".join(out)


def format_sensitivity(rows: list[SensitivityRow]) -> str:
    """Threshold/regime-weight sensitivity grid."""
    out = ["== Sensitivity (in-sample; do not pick parameters from this table) ==",
           f"  {'min_conf':>9}{'floor':>7}{'regimeW':>9}{'return':>10}{'sharpe':>8}{'maxDD':>8}{'trades':>7}"]
    for r in rows:
        out.append(f"  {r.min_confidence:>9.2f}{r.score_floor:>7.2f}{str(r.regime_weights):>9}{_pct(r.total_return):>10}"
                   f"{_num(r.sharpe):>8}{r.max_drawdown:>8.2%}{r.n_trades:>7}")
    return "\n".join(out)


def format_decision(decision: BossDecision, signals: list[AgentSignal]) -> str:
    """Bar-level view: every agent's signal plus the boss aggregation."""
    out = [f"== Bar-level decision @ {decision.timestamp.isoformat()} =="]
    for s in signals:
        feats = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in s.features.items())
        out.append(f"  [{s.agent_name}] {s.signal.value} conf={s.confidence:.3f} available={s.available}")
        out.append(f"      reason: {s.reason}")
        out.append(f"      features: {feats}")
    out.append("  weights (base x regime x family -> learned/effective):")
    for c in decision.breakdown:
        out.append(f"      {c.agent_name:<18} base {c.base_weight:.2f} learned {c.learned_weight:.2f} regime x{c.regime_factor:.2f} "
                   f"family x{c.family_factor:.2f} -> eff {c.effective_weight:.2f}; input {c.signed_input:+.3f} contrib {c.contribution:+.3f}")
    out += [
        f"  regime={decision.regime.value} directional_score={decision.directional_score:+.4f} agreement={decision.agreement:.3f} "
        f"coverage={decision.coverage:.3f} data_coverage={decision.data_coverage:.3f}",
        f"  ensemble strength={decision.confidence:.3f} ({decision.confidence_label})",
        f"  risk_score={_num(decision.risk_score)} risk_veto={decision.risk_veto} kill_switch={decision.kill_switch_active}",
        f"  raw_action={decision.raw_action.value} action={decision.action.value} forced_hold_rule={decision.forced_hold_rule}",
        f"  analyst note: {decision.reason}",
    ]
    return "\n".join(out)


def format_trade(t: Trade) -> str:
    """One round-trip trade."""
    side = "LONG" if t.direction > 0 else "SHORT"
    return (
        f"== Sample trade ==\n  {side} entry {t.entry_time} @ {t.entry_price:,.2f} (qty peak {t.max_quantity:.4f})\n"
        f"  exit {t.exit_time} @ {_num(t.exit_price, '{:,.2f}')} after {t.bars_held} bars\n"
        f"  gross PnL {t.gross_pnl:,.2f} (slippage inside fill prices: {t.slippage:,.2f}) fees {t.fees:,.2f} net PnL {t.net_pnl:,.2f}\n"
        f"  entry reason: {t.entry_reason[:300]}\n  exit reason: {t.exit_reason[:300]}"
    )


def format_adversarial(r: AdversarialReport) -> str:
    """Shadow-agent summary."""
    verdict = ("counter-consensus OUTPERFORMED consensus in this sample" if r.counter_outperformed
               else "consensus outperformed counter-consensus in this sample")
    return (
        "== Adversarial shadow agent (never votes; same timing/costs; ignores risk controls) ==\n"
        f"  disagreement observations {r.observations}, matured outcomes {r.matured_outcomes}, "
        f"counter hit rate {_num(r.counter_hit_rate, '{:.1%}')}\n"
        f"  consensus book: final {r.consensus_final_equity:,.0f} maxDD {r.consensus_max_drawdown:.1%} | "
        f"counter book: final {r.counter_final_equity:,.0f} maxDD {r.counter_max_drawdown:.1%}\n  {verdict}"
    )


def format_walk_forward(rep: WalkForwardReport, bars_per_day: float) -> str:
    """Per-fold table plus aggregated out-of-sample metrics."""
    out = ["== Walk-forward folds (IS = train with feedback, VAL = threshold selection, OOS = frozen test) ==",
           f"  {'fold':>4}  {'train':>11} {'val':>11} {'test':>11}  {'IS ret':>8} {'VAL ret':>8} {'OOS ret':>8} {'B&H OOS':>8} {'min_conf':>8} {'w_upd':>5} {'Brier cal/base':>15}"]
    for f in rep.folds:
        w = f.windows
        cal = "n/a" if f.calibration_brier is None else f"{f.calibration_brier:.3f}/{f.baseline_brier:.3f}"
        out.append(
            f"  {f.fold:>4}  {w.train[0]:>5}-{w.train[1]:<5} {w.validation[0]:>5}-{w.validation[1]:<5} {w.test[0]:>5}-{w.test[1]:<5}  "
            f"{_pct(f.train_metrics.total_return):>8} {_pct(f.validation_metrics.total_return):>8} {_pct(f.test_metrics.total_return):>8} "
            f"{_pct(f.test_buy_and_hold.total_return):>8} {f.chosen_min_confidence:>8.2f} {f.weight_updates:>5} {cal:>15}"
        )
    out.append(f"  mean in-sample (train) return {rep.in_sample_mean_return:+.2%}; mean validation return {rep.validation_mean_return:+.2%}")
    last = rep.folds[-1].learned_weights
    out.append("  learned weights after last training window: " + ", ".join(f"{k}={v:.2f}" for k, v in last.items()))
    out.append(format_metrics("Aggregated OUT-OF-SAMPLE (chained test windows)", rep.oos_metrics, bars_per_day))
    b = rep.oos_buy_and_hold
    out.append(f"  buy-and-hold over the same chained test windows: return {_pct(b.total_return)}, Sharpe {_num(b.sharpe)}, maxDD {b.max_drawdown:.2%}")
    return "\n".join(out)


def format_universe(rep: UniverseReport, title: str) -> str:
    """Per-symbol table, pooled trade stats and the cross-sectional agent verdicts."""
    ok = [r for r in rep.results if r.error is None]
    out = [f"== {title}: per-symbol results (same config, window, costs) ==",
           f"  {'symbol':<12}{'bars':>6}{'return':>10}{'sharpe':>8}{'maxDD':>8}{'trades':>7}{'win':>6}{'expo':>6}"
           f"{'B&H ret':>10}{'B&H shp':>8}{'B&H DD':>8}"]
    for r in sorted(rep.results, key=lambda r: (r.error is not None, -(r.sharpe or -9))):
        if r.error:
            out.append(f"  {r.symbol:<12}SKIPPED: {r.error[:100]}")
            continue
        out.append(
            f"  {r.symbol:<12}{r.bars:>6}{_pct(r.ret):>10}{_num(r.sharpe):>8}{r.mdd:>8.1%}{r.trades:>7}"
            f"{_num(r.win, '{:.0%}'):>6}{r.expo:>6.0%}{_pct(r.bh_ret):>10}{_num(r.bh_sharpe):>8}{r.bh_mdd:>8.1%}"
        )
    if ok:
        n = len(ok)
        beat_ret = sum(r.ret > r.bh_ret for r in ok)
        beat_sh = sum((r.sharpe or -9) > (r.bh_sharpe or -9) for r in ok)
        pos = sum(r.ret > 0 for r in ok)
        med = float(np.median([r.ret for r in ok]))
        med_bh = float(np.median([r.bh_ret for r in ok]))
        out += [
            "",
            "== Universe summary ==",
            f"  symbols run {n}/{len(rep.results)}; strategy profitable on {pos}/{n}; median return {med:+.1%} vs buy-and-hold {med_bh:+.1%}",
            f"  beat buy-and-hold on return {beat_ret}/{n}, on Sharpe {beat_sh}/{n}",
            f"  pooled closed trades {rep.pooled_trades}; win rate {_num(rep.pooled_win_rate, '{:.1%}')}; "
            f"profit factor {_num(rep.pooled_profit_factor)}",
            f"  median time invested {float(np.median([r.expo for r in ok])):.0%}",
        ]
    if rep.verdicts:
        out += [
            "",
            "== Agent verdicts (remove-one-agent test on every symbol; + means the agent added value) ==",
            f"  {'agent':<18}{'n':>4}{'mean dRet':>11}{'med dRet':>10}{'mean dShp':>11}{'helped':>8}{'hurt':>6}{'sign p':>8}"
            f"{'hit@20':>8}{'verdict':>10}",
        ]
        for v in sorted(rep.verdicts, key=lambda v: -(v.mean_d_sharpe or -9)):
            out.append(
                f"  {v.agent:<18}{v.n_symbols:>4}{_pct(v.mean_d_ret):>11}{_pct(v.median_d_ret):>10}"
                f"{_num(v.mean_d_sharpe, '{:+.3f}'):>11}{v.helped:>8}{v.hurt:>6}{_num(v.sign_p, '{:.3f}'):>8}"
                f"{_num(v.hit_rate, '{:.1%}'):>8}{v.verdict:>10}"
            )
        out += [
            "  verdict rule (fixed in advance): HELPS/HURTS needs >=60% of symbols one way AND sign-test p < 0.05.",
            "  hit@20 = share of the agent's BUY/SELL calls whose 20-day forward return had the right sign (50% = coin flip).",
        ]
    out.append("  NOTE: in-sample, today's index members (survivorship bias). Use it to decide what to test next, not to trade.")
    return "\n".join(out)


DESIGN_LABELS = {
    "ensemble": "Ensemble (5-agent vote)",
    "trend_only": "Trend only (+risk sizing)",
    "defensive": "Stay invested, step aside",
    "buy_and_hold": "Buy and hold",
}


def format_designs(rep: DesignReport, title: str) -> str:
    """Design comparison: summary table, crash windows, per-symbol Sharpe and the pre-registered winner."""
    ok = [s for s in rep.symbols if s.error is None]
    out = [f"== Step 3 design comparison: {title} ({len(ok)}/{len(rep.symbols)} symbols) =="]
    out.append(f"  {'design':<28}{'beat B&H Sharpe':>16}{'beat B&H ret':>13}{'med return':>12}{'med Sharpe':>11}"
               f"{'med maxDD':>10}{'invested':>9}{'trades':>7}{'vs ensemble':>13}")
    for d in rep.summaries:
        vs = "" if d.vs_ensemble_wins is None else f"{d.vs_ensemble_wins}/{d.n} p={d.vs_ensemble_p:.3f}" if d.vs_ensemble_p is not None else f"{d.vs_ensemble_wins}/{d.n}"
        beat_sh = "-" if d.design == "buy_and_hold" else f"{d.beat_bh_sharpe}/{d.n}"
        beat_r = "-" if d.design == "buy_and_hold" else f"{d.beat_bh_return}/{d.n}"
        out.append(
            f"  {DESIGN_LABELS.get(d.design, d.design):<28}{beat_sh:>16}{beat_r:>13}{_pct(d.median_return):>12}"
            f"{_num(d.median_sharpe):>11}{d.median_mdd:>10.1%}{d.median_expo:>9.0%}{d.median_trades:>7.0f}{vs:>13}"
        )
    crashes = list(rep.summaries[0].crash) if rep.summaries else []
    if crashes:
        out += ["", "== Median return through market crashes (stocks with data in the window) =="]
        out.append(f"  {'design':<28}" + "".join(f"{c:>16}" for c in crashes))
        for d in rep.summaries:
            out.append(f"  {DESIGN_LABELS.get(d.design, d.design):<28}" + "".join(f"{_pct(d.crash.get(c)):>16}" for c in crashes))
    losers = [x for x in ok if x.rows["buy_and_hold"].ret < 0]
    out += ["", f"== Stocks where buy-and-hold LOST money: {len(losers)}/{len(ok)} (the exit rule's real job) =="]
    if losers:
        out.append(f"  {'design':<28}{'median return':>15}{'lost less than B&H':>20}{'still lost money':>18}")
        for k in ("ensemble", "trend_only", "defensive", "buy_and_hold"):
            rets = [x.rows[k].ret for x in losers]
            better = sum(x.rows[k].ret > x.rows["buy_and_hold"].ret for x in losers) if k != "buy_and_hold" else None
            neg = sum(r < 0 for r in rets)
            out.append(f"  {DESIGN_LABELS[k]:<28}{_pct(float(np.median(rets))):>15}"
                       f"{('-' if better is None else f'{better}/{len(losers)}'):>20}{f'{neg}/{len(losers)}':>18}")
    worst = sorted(ok, key=lambda x: x.rows["buy_and_hold"].ret)[:10]
    out += ["", "== 10 worst stocks for buy-and-hold: total return by design =="]
    out.append(f"  {'symbol':<12}{'Ensemble':>11}{'Trend':>11}{'Stay inv.':>11}{'Buy&Hold':>11}{'B&H maxDD':>11}")
    for x in worst:
        r = x.rows
        out.append(f"  {x.symbol:<12}{_pct(r['ensemble'].ret):>11}{_pct(r['trend_only'].ret):>11}{_pct(r['defensive'].ret):>11}"
                   f"{_pct(r['buy_and_hold'].ret):>11}{r['buy_and_hold'].mdd:>11.0%}")
    out += ["", "== Per-symbol Sharpe (E = ensemble, T = trend only, D = defensive, B = buy and hold) =="]
    out.append(f"  {'symbol':<12}{'bars':>6}{'E':>8}{'T':>8}{'D':>8}{'B':>8}   best")
    for s in sorted(rep.symbols, key=lambda s: s.symbol):
        if s.error:
            out.append(f"  {s.symbol:<12}SKIPPED: {s.error[:90]}")
            continue
        sh = {k: s.rows[k].sharpe for k in ("ensemble", "trend_only", "defensive", "buy_and_hold")}
        best = max(sh, key=lambda k: sh[k] if sh[k] is not None else -9)
        out.append(f"  {s.symbol:<12}{s.bars:>6}" + "".join(f"{_num(sh[k]):>8}" for k in sh) + f"   {DESIGN_LABELS[best].split(' (')[0]}")
    out += [
        "",
        f"WINNER (rule fixed in advance: most stocks beating buy-and-hold on Sharpe, tie -> median Sharpe): "
        f"{DESIGN_LABELS.get(rep.winner, rep.winner) if rep.winner else 'n/a'}",
        "  'vs ensemble' = stocks where the design's Sharpe beat the ensemble's, with a sign-test p-value.",
        "  In-sample on today's Nifty 50 members (survivorship bias). The winner still has to pass walk-forward (Step 5).",
    ]
    return "\n".join(out)


PORTFOLIO_LABELS = {
    "A_hold": "A. Hold the basket",
    "B_trend": "B. Basket + per-stock trend",
    "C_market": "C. Basket + Nifty crash filter",
}


def format_portfolio(rep: PortfolioReport, title: str) -> str:
    """Step 3c: portfolio-level comparison of the three options."""
    out = [f"== Step 3c portfolio test: {title} ==",
           f"  stocks used {len(rep.symbols_used)}; skipped {len(rep.symbols_skipped)}"
           + (f" ({', '.join(rep.symbols_skipped)})" if rep.symbols_skipped else ""),
           "  equal-weight slots, monthly rebalance, decisions at close -> trades at next open, costs on every trade",
           "",
           f"  {'option':<32}{'final value':>14}{'total':>11}{'per year':>10}{'Sharpe':>8}{'worst fall':>12}{'invested':>10}{'costs':>11}"]
    for r in rep.options:
        m = r.metrics
        out.append(
            f"  {PORTFOLIO_LABELS[r.name]:<32}{m.final_equity:>14,.0f}{_pct(m.total_return):>11}{_pct(m.cagr):>10}"
            f"{_num(m.sharpe):>8}{m.max_drawdown:>12.1%}{float(r.invested.mean()):>10.0%}{r.costs:>11,.0f}"
        )
    crashes = list(rep.options[0].crash)
    out += ["", "== Through the crashes ==", f"  {'option':<32}" + "".join(f"{c:>14}" for c in crashes)]
    for r in rep.options:
        out.append(f"  {PORTFOLIO_LABELS[r.name]:<32}" + "".join(f"{_pct(r.crash.get(c)):>14}" for c in crashes))
    years = sorted(rep.options[0].yearly)
    out += ["", "== Calendar-year returns ==", f"  {'year':<6}" + "".join(f"{PORTFOLIO_LABELS[r.name][:2]:>9}" for r in rep.options)]
    for y in years:
        out.append(f"  {y:<6}" + "".join(f"{_pct(r.yearly.get(y)):>9}" for r in rep.options))
    beats = ", ".join(PORTFOLIO_LABELS[b] for b in rep.beats_hold) or "none"
    out += [
        "",
        f"WINNER (rule fixed in advance: highest Sharpe, tie -> smaller worst fall): {PORTFOLIO_LABELS[rep.winner]}",
        f"Options that beat plain holding on BOTH Sharpe and worst fall: {beats}",
        "  In-sample; basket = the chosen universe file (survivorship caveats apply). Next: walk-forward (Step 5).",
    ]
    return "\n".join(out)


def format_robustness(rep, title: str) -> str:
    """Step 5: robustness of option C (Nifty crash filter) vs A (hold the basket)."""

    def line(label: str, r, width: int = 30) -> list[str]:
        out = []
        for name, o in (("A hold", r.a), ("C filter", r.c)):
            m = o.metrics
            exit_v = "n/a" if o.after_tax_exit_value is None else f"{o.after_tax_exit_value:,.0f}"
            out.append(f"  {label if name == 'A hold' else '':<{width}}{name:<10}{_pct(m.cagr):>9}{_num(m.sharpe):>8}"
                       f"{m.max_drawdown:>11.1%}{o.taxes_paid:>13,.0f}{exit_v:>16}")
        out.append(f"  {'':<{width}}{'C beats A on both Sharpe and worst fall:':<42}{'YES' if r.c_wins else 'no':>5}"
                   f"   (C invested {r.c_invested:.0%} of days)")
        return out

    head = f"  {'case':<30}{'option':<10}{'per year':>9}{'Sharpe':>8}{'worst fall':>11}{'tax paid':>13}{'after-tax exit':>16}"
    out = [f"== Step 5 robustness test: {title} ==",
           f"  stocks used {len(rep.symbols_used)}; skipped {len(rep.symbols_skipped)}"
           + (f" ({', '.join(rep.symbols_skipped)})" if rep.symbols_skipped else ""),
           f"  starting capital {rep.capital:,.0f} rupees; 'after-tax exit' = value if everything were sold on the last day",
           "", "== 3. Real money (base setting, full period) ==", head]
    for r in rep.realism:
        out += line(r.label, r)
    out += ["", "== 1. Nearby filter settings (cash interest + tax on) ==",
            f"  {'setting':<24}{'C per year':>11}{'C Sharpe':>10}{'C worst fall':>14}{'C beats A?':>12}"]
    a = rep.settings[0].a.metrics if rep.settings else None
    if a is not None:
        out.append(f"  {'(A hold, for reference)':<24}{_pct(a.cagr):>11}{_num(a.sharpe):>10}{a.max_drawdown:>14.1%}{'':>12}")
    for r in rep.settings:
        m = r.c.metrics
        out.append(f"  {r.label:<24}{_pct(m.cagr):>11}{_num(m.sharpe):>10}{m.max_drawdown:>14.1%}{'YES' if r.c_wins else 'no':>12}")
    need = rep.pass_share * len(rep.settings)
    out.append(f"  C wins in {rep.settings_wins} of {len(rep.settings)} settings (needs >= {need:g}) -> "
               f"{'PASS' if rep.settings_pass else 'FAIL'}")
    out += ["", "== 2. Split periods (fresh capital each half, cash interest + tax on) ==", head]
    for r in rep.periods:
        out += line(r.label, r)
    if not rep.periods:
        out.append("  (window does not span the split date; period test cannot pass)")
    out.append(f"  C beats A in every half -> {'PASS' if rep.periods_pass else 'FAIL'}")
    out += ["",
            f"OVERALL (rules fixed in advance: settings PASS and periods PASS): {'PASS' if rep.passed else 'FAIL'}",
            "  In-sample, survivorship caveats apply. PASS means 'worth paper trading', not 'profitable'."]
    return "\n".join(out)


def format_signal(rep, title: str, max_orders: int = 60) -> str:
    """Step 7: daily paper-trading signal for the Nifty crash filter basket (option C)."""
    on = rep.filter_now > 0
    pnl = rep.paper_value / rep.capital - 1.0
    out = [f"== Step 7 daily paper signal: {title} ==",
           f"  as of close: {rep.as_of:%a %d-%b-%Y}   (stocks used {len(rep.symbols_used)}; skipped {len(rep.symbols_skipped)}"
           + (f": {', '.join(rep.symbols_skipped)})" if rep.symbols_skipped else ")"),
           "",
           f"  NIFTY FILTER: {'INVESTED' if on else 'IN CASH'}"
           + ("" if rep.filter_now == rep.filter_prev else "   <-- CHANGED TODAY"),
           f"  Nifty {rep.nifty_close:,.2f} | EMA fast {rep.ema_fast:,.2f} | EMA slow {rep.ema_slow:,.2f} | ADX {rep.adx:.1f}",
           f"  {rep.regime_summary}",
           "",
           f"  ACTION FOR NEXT SESSION (~{rep.next_session:%a %d-%b}, at the open): {rep.action}"]
    if rep.orders:
        out += [f"  {'stock':<14}{'side':<6}{'amount (Rs)':>14}{'~shares':>9}{'at close':>12}"]
        for o in rep.orders[:max_orders]:
            out.append(f"  {o.symbol:<14}{o.side:<6}{o.rupees:>14,.0f}{o.approx_shares:>9}{o.ref_price:>12,.2f}")
        if len(rep.orders) > max_orders:
            out.append(f"  ... {len(rep.orders) - max_orders} more (see the orders CSV)")
    out += ["",
            f"  PAPER BOOK: started {rep.paper_start:%d-%b-%Y} with Rs {rep.capital:,.0f}; sessions traded {rep.sessions_traded}",
            f"  value Rs {rep.paper_value:,.0f} ({pnl:+.2%}); invested {rep.paper_invested:.0%}; cash Rs {rep.paper_cash:,.0f};"
            f" tax provisioned Rs {rep.taxes_paid:,.0f}"]
    for w in rep.warnings:
        out.append(f"  WARNING: {w}")
    if getattr(rep, "panel", None):
        from execution.agent_panel import format_panel
        out += ["", format_panel(rep.panel, rep.scorecard)]
    out += ["",
            "  Paper only: no order is sent anywhere. Fills assumed at the next open with fees, slippage, cash interest and tax.",
            "  Run once a day after 4 pm IST. Judge it after 8-12 weeks against the rules, not day by day."]
    return "\n".join(out)
