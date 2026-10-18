# What to build next (ranked by expected practical impact)

This ranking is a research hypothesis, not a guarantee. The framework's biggest weakness is that every
signal is evaluated on synthetic data, so the first priorities are about getting real evidence, not adding
features.

1. **Real trade-level and order-book data (aggressor-flagged prints, L2 depth, funding history).** OFI, CVD,
   sweeps and absorption are only meaningful on genuine aggressor and depth data. Today they are synthetic.
2. **Statistical validation on real data.** This means many walk-forward folds across symbols, block-bootstrap
   confidence intervals for Sharpe, and multiple-testing control (deflated Sharpe) for the threshold grid.
   Synthetic out-of-sample results are not evidence of alpha.
3. **Transaction-cost and market-impact model.** This would replace fixed bps with spread-, depth- and
   participation-dependent costs. Turnover-heavy agents may only look good under the flat-cost assumption.
4. **Agent-level calibration and regime-conditional evaluation.** This would score each agent per regime on
   matured outcomes and learn regime weights from data instead of the illustrative table.
5. **Portfolio-level sizing and multi-asset risk.** This means volatility targeting across instruments,
   correlation-aware exposure caps, and stop/exit rules decoupled from the entry vote, which is currently the
   main driver of the long holding periods.
