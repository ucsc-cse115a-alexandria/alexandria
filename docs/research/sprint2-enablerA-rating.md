# Enabler A: Candidate Rating and Shortlist

> **Decision update (2026-07-18):** This document records the Sprint 2 shortlist before BABILong was evaluated.
> The release benchmark is now [BABILong 8k](../../benchmarks/babilong_8k/README.md): its roughly 8k-token prompts
> provide meaningful compression headroom, and its `qa1`-`qa5` labels have a deterministic verifier. IFEval remains
> useful for instruction-format compliance, but its short prompts are not the canonical compression benchmark.

## Overview
Rates the four surveyed candidates, IFEval, RULER, NIAH, and LongBench, against Enabler A's acceptance criteria and shortlists the strongest for our project's benchmark selection.

## Rating

| Candidate | Established | Length-sensitive | Fast/cheap subset (≤10 min, ≤$1) | Fit to our failure mode |
|---|---|---|---|---|
| IFEval | Yes, 484 citations | No, prompts aren't length-varied | Unknown, likely cheap given 541 prompts and no LLM judge | Strong, only candidate testing per-instruction compliance in code |
| RULER | Yes, 1.6k GitHub stars | Yes, length-sensitivity is the core method | Unknown, likely cheap given synthetic scalable generation | Weak, tests retrieval and tracing in distractor text |
| NIAH | Yes, 2.3k stars, no paper | Designed around length but no published accuracy-vs-length numbers in the tool itself | Unknown, likely cheap given synthetic scalable generation | Weak, narrower match than RULER since it has no established baseline or threshold |
| LongBench | Yes, 862 citations | Yes, LongBench-E publishes bucketed drop numbers | Unknown, likely moderate given 21 datasets | Weak on instruction-following, but PassageCount is structurally adjacent to duplicate detection |

The fast/cheap criterion can't be confirmed from any of the four research notes. None of them report actual run time or LLM cost, that only comes from running the benchmark, which is what task 3 is for. Marking it unknown here rather than guessing a pass or fail.

## Shortlist: IFEval

IFEval is the strongest pick. It's the only one that tests instruction-following rather than long-context retrieval, and that's what our project's failure mode is, dropping a near-duplicate instruction and losing a unique directive underneath it.

RULER and LongBench clear the length-sensitivity criterion more cleanly, and NIAH is designed around the same idea, but length-sensitivity in a retrieval task doesn't tell us whether compression broke instruction compliance. IFEval doesn't vary prompt length at all, which is a gap against the must-have bar, but it points at a benchmark that already verifies instruction compliance directly, rather than one built around retrieval that would need compliance checking retrofitted onto it.

The three retrieval-style candidates share the same gap: none of them test whether a model still follows an instruction after a near-duplicate restatement is present or removed. That gap is what our project's compression is supposed to prove doesn't break things, so a benchmark that can't observe it at all isn't a fit no matter how well it scores on established or length-sensitive.

## Next step
Task 3 runs a small trial on IFEval to confirm the time and cost budget and that scores hold or drop appropriately, then locks in the base benchmark with a written rationale.
