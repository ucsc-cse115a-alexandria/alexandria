# P0–P6 Compression-Strength Sweep: Curve Summary

**Benchmark:** BABILong 8k, n=50, seed 42
**Date:** 2026-07-20
**Related:** #79 (sweep matrix), #80 (execution), #81 (decision), #82 (this report)

## Overview

This sweep tested whether loosening or tightening the `cos_sim_diff_budget` safety rail around
Alexandria's default compression settings could save more tokens without breaking the
accuracy-retention release gate (95% CI lower bound ≥90%). Seven points were measured: the
uncompressed original (P0), the current default (P1), and five budget variants (P2–P6, budgets
0.0025 to 0.02).

## Results

| Point | Setting | Accuracy | Retention (95% CI) | Token reduction | Release gate |
|---|---|---|---|---|---|
| P0 | Original (uncompressed) | 72.0% (36/50) | — (baseline) | 0.0% | — |
| P1 | Default (`Params()`) | 70.0% vs 74.0% | ~95% (CI low 81.8%) | 0.8% | FAIL |
| P2 | Budget 0.0025 | 66.7% vs 72.0% | ~91.4% (79.4%–103.1%) | 0.7% | FAIL |
| P3 | Budget 0.005 | 64.6% vs 72.0% | ~91.2% (78.4%–103.2%) | 0.7% | FAIL |
| P4 | Budget 0.01 | 72.9% vs 72.0% | ~103% (CI low 86.5%) | 0.7% | FAIL |
| P5 | Budget 0.015 | 70.8% vs 72.0% | ~100% (82.9%–120.6%) | 0.7% | FAIL |
| P6 | Budget 0.02 | 64.6% vs 72.0% | ~91.2% (75.0%–108.6%) | 0.7% | FAIL |

*Note: P1's original-prompt baseline (74.0%) differs slightly from P0 and P2–P6 (72.0%) due to
separate benchmark runs — normal run-to-run variance, not a data error.*

## Time and cost

Wall-clock time and estimated API cost per point (`2026-07-20-sweep-index-v1.json`):

| Point | Wall clock | Est. cost |
|---|---:|---:|
| P1 | 1,218s (~20.3 min) | $0.00 |
| P2 | 1,105s (~18.4 min) | $0.58 |
| P3 | 1,033s (~17.2 min) | $0.59 |
| P4 | 1,019s (~17.0 min) | $0.58 |
| P5 | 1,052s (~17.5 min) | $0.58 |
| P6 | 1,054s (~17.6 min) | $0.58 |

P1's $0.00 reflects the inline library-default runner, which did not emit `api_events.jsonl`.
P2–P6 each cost roughly $0.58 and took 17–18 minutes — flat across the budget range, with no
cheaper operating point emerging as budgets loosened.

## Curve interpretation

Token reduction is flat across the entire sweep, roughly 0.7–0.8% regardless of how the
`cos_sim_diff_budget` safety rail is loosened or tightened. Tightening the budget (P2, P3) does
not meaningfully reduce cuts made, and loosening it (P5, P6) does not meaningfully increase them —
the algorithm is bottlenecked by how few near-duplicate sentence pairs BABILong 8k prompts
actually contain, not by the budget itself.

No point — including the shipped default — clears the accuracy-retention release gate with 95%
confidence at n=50. Point estimates for P4 and P5 look strong (~100–103% retention), but their
confidence intervals are wide enough at this sample size that the lower bound still falls below
the 90% threshold.

## Decision

**Selected release default: unchanged.** `Params()` (P1) remains the shipped default. No point
in the sweep beats it while clearing the release gate, so per #81's test plan, no default or
parameter change was made.

This does not mean compression doesn't work — separate `--keep`-mode sweeps show 13–29%+ token
reduction is achievable at more aggressive settings, with a larger accuracy trade-off. That is a
different compression mode, out of scope for this sweep, which was restricted to the conservative
`cos_sim_diff_budget` mechanism only.

## Evidence

- Combined index: `benchmarks/babilong_8k/results/2026-07-20-sweep-index-v1.json`
- Per-point reports: `benchmarks/babilong_8k/results/2026-07-20-p{0..6}-*/report.md`
- Decision record: #81 (closed)