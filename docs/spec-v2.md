# Alexandria v2 — Behavioral Fidelity

A delta on [spec.md](spec.md). v1 keeps a prompt edit only if it barely moves the **sentence
embedding**. v2 adds a sharper, model-grounded fidelity signal: keep an edit only if it barely
moves the **decoder's next-token distribution** — what the model is actually about to do. The three
phases, the IR, the plugin/registry model, and every invariant in spec.md are unchanged; v2 is two
new strategy files and one new impure boundary.

## Why embedding similarity is not enough

A sentence embedding is a mean-pooled summary of a span's *topic*. It is **near-necessary but not
sufficient** for fidelity: the tokens that flip a downstream answer (a negation, a digit, a name,
`v1`→`v2`) barely move the semantic centroid, so cosine stays high while the behavior breaks.

- Minimal negation/antonym pairs routinely score cosine **> 0.95** on bi-encoders; the cosine reward
  tracks topic overlap, not truth value. (Hsieh et al., HEROS, [2306.05083]; "Daunting Dilemma with
  Sentence Encoders" [2309.03747]; negation-aware embeddings [2504.00584].)
- Perplexity, the other cheap scalar, also fails to track correctness ([2405.06105]; EvoLM
  [2506.16029]) — so simply switching from cosine to perplexity does not fix it.
- The decisive signal lives in the **output distribution, at a few tokens**: one analysis found ~82%
  of total KL between two prompts' outputs concentrated in 5 of 128 positions. An aggregate metric
  (mean cosine, mean KL, perplexity) averages the decisive token away; a **per-token max** over the
  output distribution recovers it. (Knowledge distillation [1503.02531]; "Prompts have evil twins"
  [2311.07064]; KV-Distill [2503.10337].)

**Prior art and the gap.** Training-free compressors (Selective Context [2304.12102], LLMLingua
[2310.05736], LongLLMLingua [2310.06839]) score tokens by a **single-step** self-information /
perplexity proxy — never a before/after comparison of the full output distribution. "Evil twins"
[2311.07064] defines prompt equivalence as **KL between the prompt-induced output distributions** but
optimizes adversarial strings, not compression. PCRL [2308.08758] frames the objective as output
divergence yet substitutes ROUGE. **A label-free compressor whose accept/reject gate is the
before/after decoder-distribution divergence is unclaimed** — that is v2's core.

## The idea: a behavioral signature replaces the embedding as the fidelity proxy

For a prompt `p`, run a deployment-class causal LM, teacher-force a fixed **probe continuation** `c`,
and record the per-position next-token distributions. That array **is** the behavioral signature —
the decoder's analogue of v1's embedding. Two prompts are behaviorally equivalent to the extent
their signatures coincide.

```
signature(p, c) = [ softmax(logits(p · c_<t))  for t in positions of c ]   # (|c|, vocab)
drift(p, p') = max over positions t of  JSD( signature(p,c)[t] ‖ signature(p',c)[t] )
```

- **JSD, not KL.** Jensen–Shannon is symmetric and bounded in `[0, log 2]`, so it thresholds cleanly
  and is stable when supports differ; KL is unbounded and tail-fragile under top-k truncation.
  (Generalized JSD distillation, GKD [2306.13649].)
- **Max per-token, not mean** — the decisive token, not the average, decides fidelity (see above).
- **Truncated logits fail safe.** Local backends expose top-k logprobs; assign the residual mass
  `1 − Σ top-k` uniformly over the tail before computing JSD — a conservative **upper bound** on
  drift, so truncation can only over-reject, never silently accept.

**The probe `c` is the reasoning prefix.** Default `c` = the model's own greedy decode of its
reasoning/answer prefix under the *original* prompt — literally "what the model was about to do."
Decoded **once per Document** and reused for every trial, so the cost of a trial is one *teacher-
forced* forward pass (no autoregression). The reasoning *content* is never trusted (CoT is an
unfaithful self-report — Turpin et al. [2305.04388]; Anthropic [2505.05410]); only its *distribution*
is compared, which sidesteps faithfulness entirely. The answer is decodable within ~4 reasoning
tokens ([2511.14773]), so a short prefix suffices.

## Two-tier pipeline (this is the "layer your metrics" point, made concrete)

| Tier | Signal | Role | Cost |
|---|---|---|---|
| 1 — screen | embedding `redundancy` (v1, unchanged) | **propose** candidate drops | cheap |
| 2 — gate | behavioral `drift` (v2, new) | **accept/reject** each drop | one teacher-forced pass / trial |

Embedding redundancy keeps proposing candidates; its negation-blindness is acceptable here because
the Tier-2 gate catches the false positives it lets through. Fidelity is *guaranteed* only at the
gate — which is exactly where v1 used embedding cosine and v2 uses decoder JSD.

## What changes in the current implementation

Five touch points. Everything else (IR, `apply`, `redundancy`, `greedy_pairwise`, all invariants,
Parquet) is untouched. New strategies are **new files**, per spec.md's "one strategy, one file" rule.

**1. `ir/contracts.py` — a second impure boundary, beside `Embedder`.**

```python
type Signature = NDArray[np.float32]          # (probe_len, top_k+1): per-position top-k logprobs + tail

class Decoder(Protocol):
    @property
    def model_id(self) -> str: ...
    def signature(self, prompt: str, continuation: str) -> Signature: ...   # teacher-forced logprobs
    def decode(self, prompt: str, max_tokens: int) -> str: ...              # greedy probe generation
```

`Optimizer.__call__` gains a `decoder: Decoder` parameter (mirroring the always-passed `embedder`);
optimizers that don't need it ignore it, as `greedy_pairwise` already ignores scorers it didn't
request. Registration declares the need so the (expensive) decoder is **built lazily** — only when a
chosen optimizer asks for it:

```python
@register_optimizer("behavioral_pairwise", requires=("redundancy",), decoder=True)
```

`OptimizerParams` gains `max_behavioral_drift: Drift` (the JSD gate) and `probe_tokens: int`
(prefix length). **The default `max_behavioral_drift` has no literature constant** — it must be
calibrated (see Evaluation); ship it flagged as provisional, not authoritative.

**2. `ir/divergence.py` — new, pure.** `token_jsd(a, b)` and
`max_token_jsd(sig_a, sig_b, *, tail="uniform")` over arrays. No model in the loop, so it is tested
on hand-written distributions exactly like `similarity.py`.

**3. `utils/decoding.py` — new, the only place a generative model is built.** `build_decoder(model)`
returning a `Decoder`, plus a deterministic `HashDecoder` fake so the whole pipeline stays a pure
function under test (the v1 `HashEmbedder` pattern). The `import-linter` forbidden contract extends:
`ir`, `ops`, and `cli` may not import a generative backend — only this shell may.

**4. `ops/features/optimize/behavioral_pairwise.py` — new, one file.** Same candidate ranking as
`greedy_pairwise` (redundant pairs, sorted by similarity), but the drop-safety guard swaps cosine
drift for behavioral drift:

```
c = decoder.decode(document.text, params.probe_tokens)     # probe: the reasoning prefix, once
base = decoder.signature(document.text, c)
for (a, b) in redundant pairs, sorted desc:                # Tier-1 screen (embedding redundancy)
    for cand in (a, b) still present:
        trial = decoder.signature(text_without(cand), c)
        d = max_token_jsd(trial, base)                     # Tier-2 gate (decoder distribution)
    drop = argmin over the pair of d
    if min d <= params.max_behavioral_drift:               # was: cosine_distance <= max_drift
        propose Delete(drop)
```

This is v1's `_drift` replaced by `_behavioral_drift`; the structure, the unique-sentence guard, and
the `Plan`/`apply` contract are identical.

**5. CLI + pipeline.** `reduce`/`review` accept `--decoder MODEL` (mirrors `--model` for the
embedder). The pipeline builds the decoder iff a chosen optimizer declares `decoder=True`. Signatures
are content-addressed by `(prompt_hash, probe_hash)` exactly as `encode` caches embeddings, so
identical trial texts are scored once.

## Cost and efficiency

The probe is decoded **once** under the original prompt; every trial reuses it. How much of the
original forward pass a trial can reuse is the whole efficiency story — in three layers, from certain
to aspirational:

- **Layer 1 — reuse the probe (certain, biggest win).** Decode the reasoning prefix `c` once, then
  *score* it under each edited prompt by teacher forcing — a single parallel forward pass, no
  autoregressive generation. This is what makes a trial cheap relative to "re-run the model."
- **Layer 2 — reuse the KV cache (real, but with a floor).** Causal attention bounds reuse: `c` sits
  at the end and attends to the whole prompt, so **`c`'s positions always recompute** after any edit;
  prefix caching salvages only the head *before* the cut (RadixAttention/automatic prefix caching,
  SGLang [2312.07104]). Reusing the KV of *unchanged middle chunks* needs CacheBlend-style selective
  recompute of the ~10–15% of post-cut tokens whose cross-attention mattered ([2405.16444]) —
  PromptCache [2311.04934] is cheaper but distorts cross-chunk attention, so it corrupts the very
  distribution being measured. Net: a trial costs ≈ `|c|` + the changed tail, not zero.
- **Layer 3 — predict the edit instead of running it (the true "light computation").** Skip a forward
  pass per edit entirely: spend a **fixed ~32 ablation passes** to fit a Lasso surrogate from
  "sentences kept → output probability," then score any candidate edit as a dot product, independent
  of prompt length (ContextCite [2409.00729]). AttriBoT approximates exact leave-one-out with
  prefix-cache + a small proxy model + coarse-to-fine for >300× ([2411.15102]). The classical ideal —
  predicting leave-one-out without re-running, via influence functions ([1703.04730]) — is known to
  approximate true LOO poorly beyond a few steps, which is precisely why these methods use empirical
  ablation rather than a first-order estimate.

## Stretch: a behavioral redundancy scorer

`ops/features/score/behavioral_redundancy.py` (one file) makes **Tier 1 itself** behavioral: a sentence is
redundant iff the model still predicts the probe ≈ as well without it — pointwise mutual information
between a sentence and the rest, computed from the LM (Padmakumar & He, PMI summarization
[2102.06272]). Or estimate it in constant passes with ContextCite. A genuine unclaimed variant:
**fill-in-the-middle surprisal** — "removable iff reconstructable from prefix + suffix" — which no
prior compressor uses. Kept as a stretch; the embedding screen is the cheap default.

## Honest limits

- **The decoder-beats-embedding-on-negation claim is a hypothesis, not a result.** Bi-encoder
  negation-blindness is well documented; that a decoder's *next-token distribution* fixes it is
  indirectly supported but unproven. The benchmark below is what validates it.
- **Proxy ≠ deployment model.** A small probe model is a biased estimate of the target's behavior.
  Equivalence judged by output-distribution matching does transfer across models (evil twins
  [2311.07064]), but LLMLingua needed an alignment step to close the proxy gap — prefer a probe model
  from the deployment family.
- **No canonical safe threshold exists** — `max_behavioral_drift` is empirical; the divergence→safety
  link is correlational, not a proven bound. Calibrate, don't import a constant.
- **The probe biases the metric** — a generic suffix may be insensitive to the deleted content; the
  reasoning-prefix default is on-distribution but recomputed per Document.

# Evaluation

v2 makes a falsifiable claim: the decoder-distribution gate catches meaning-changing edits that
embedding cosine misses. An evaluation that cannot **leak** is what turns that claim from plausible
into demonstrated.

## What we measure — two layers

| Layer | Question | Ground truth | Leak risk |
|---|---|---|---|
| **B — Metric validation** (the core) | Does `drift` separate meaning-changing from meaning-preserving edits, and beat cosine/perplexity? | **Label by construction** (a program decides) | **None** if labels are programmatic — this is why Layer B leads |
| **A — System evaluation** | Does the full pipeline cut tokens while preserving downstream behavior? | Downstream task accuracy + multi-model behavior consensus | Contamination + circularity — controlled below |

Layer B validates the *gate*; Layer A validates the *system*. Layer B is primary because it is the
novel claim and the only layer that can be made leak-proof by construction.

## The leakage threat model (read first)

Three distinct leaks, each with a specific neutralizer. The constraint — *do not leak* — is met only
if all three are closed.

1. **Pretraining contamination** — a benchmark prompt is in the proxy/deployment model's training
   data, so "fidelity" is memorization. *Neutralize:* freshly authored or **template-generated**
   items (answer follows from the construction rule, not recall — Functional Benchmarks [2402.19450];
   LiveBench [2406.19314], LiveCodeBench [2403.07974]); a **canary GUID** in any released artifact;
   screen every candidate model with **Min-K% Prob** [2310.16789] / paraphrase decontamination
   [2311.04850] before trusting its score; record each model's training cutoff vs. item date.
2. **Evaluation-set leakage** — tuning the gate threshold on the test set. *Neutralize:* strict
   **train / dev / test** split; `max_behavioral_drift` and every hyperparameter tuned on **dev
   only**; test touched once (reusable-holdout discipline, Dwork et al. [1506.02629]). A freshly
   re-collected mirror split as a sanity check, interpreting drops as shift vs. overfit (ImageNetV2
   [1902.10811]).
3. **Circularity** — the proxy that defines the gate also generates items, judges fidelity, or is the
   model under test. Reference-free metrics provably drift toward their own model [2210.12563] and
   reward low-perplexity/familiar text regardless of fidelity [2410.21819]; judges favor their own
   generations [2404.13076]. *Neutralize:* the **role-separation matrix** below, and — wherever
   possible — **eliminate the judge entirely** by using programmatic labels (Layer B).

**Role-separation matrix.** Any model that *produces* text being scored must differ in family from any
model that *scores* it.

| Role | Must differ from | Why |
|---|---|---|
| **Proxy** (gate model) | Deployment, Judge, Item-generator | Else the gate is graded by its own signal (self-recognition [2404.13076]) |
| **Deployment** (under test) | Judge | Self-enhancement inflates own outputs [2306.05685] |
| **Item-generator** | Judge | Generator-as-judge favors its own phrasing (verbosity/perplexity bias [2407.01085]) |
| **Item-generator** | — (prefer rule-based) | A program generating items removes the generator from the trust path entirely |

Target: proxy, deployment, judge, generator are four distinct families. Best case: items are
rule-generated and labels are programmatic, so *no judge exists* to be biased.

## Layer B — metric validation (leak-proof by construction)

**Prompt minimal pairs with a programmatic verifier.** Build on IFEval-style verifiable instructions
[2311.07911]: each prompt `P` pairs with a deterministic checker `V` (regex / token-count / keyword)
that decides pass/fail of an output. An edit's label is then **certain, because `V` is a program**:

- **Meaning-changing (MC)** — the edit changes the set of outputs `V` accepts. *Catalog (label by
  construction):* delete a verifiable constraint (`"respond in JSON"` → removed); invert it (`"do NOT
  include X"` → `"include X"`); move a numeric bound (`"≤50 words"` → `"≤500"`); swap a versioned
  value (`"use v1"` → `"v2"`); counterfactual rule (RuleBench [2407.08440]). The gate **must reject**.
- **Meaning-preserving (MP)** — the edit leaves `V`'s accepted set identical: paraphrase a
  non-verified sentence, reorder independent instructions, **drop a redundant restatement of a
  constraint that still appears elsewhere** (exactly Alexandria's job), drop a redundant few-shot
  example, compress whitespace. The gate **must accept**.

Edit operators come from CheckList INV/DIR [2005.04118], Contrast Sets [2004.02709], CondaQA's
paraphrase/scope/reverse edits [2211.00295], and NL-Augmenter's deterministic transforms [2112.02721].
**Hard MP negatives** matter: FormatSpread [2310.11324] and the Butterfly Effect [2401.03729] show
cosmetic edits can still move a model's output — so MP is defined by `V` (the gold contract), not by
"looks cosmetic," and the gate is graded against `V`, not against one model's wobble. `CompressionAttack`
[2510.22963] is the live threat this layer measures: edits that survive compression yet flip the
output — precisely the MC-disguised-as-MP cases the gate exists to catch.

## Layer A — system evaluation

**Corpus.** Instruction-heavy, genuinely redundant prompts: Super-NaturalInstructions [2204.07705]
(definitions restated in examples), Unnatural Instructions' paraphrase expansions [2212.09689], and
real system/agent prompts (leaked-system-prompt collections — research-only, **check licensing per
source, do not redistribute derived text**). Few-shot blocks harvested from the above.

**Downstream tasks with labels.** IFEval [2311.07911] (programmatic), LongBench-v2 [2412.15204]
(multiple-choice → clean accuracy, no free-form-metric noise), GSM8K, BBH, MeetingBank [2305.17529].

**Behavioral ground truth without labels (the differential oracle).** Where no programmatic verifier
exists, "behavior changed" = **consensus shift across K proxy-independent deployment models**
(differential testing, DeepXplore pattern [1705.06640]). Aggregate by **minority-veto / diversity-
weighted**, not plurality — LLM juries over-confirm "no change" (TNR collapse [2510.11822]). The
proxy family is excluded from the jury. A **~100-item, ≥3-annotator human anchor** (Cohen/Fleiss κ
or Krippendorff α + CIs) audits the oracle; it **validates, never tunes**.

## How to compute scores

**Compression.** Ratio in the **deployment model's tokenizer** (tiktoken), over the **compressible
span only** (exclude the protected instruction/question) — state the denominator; report as both
`1 − Lc/Lo` and the `Lo/Lc` multiple to avoid the two-convention confusion.

**System fidelity.** Downstream accuracy retention `acc(compressed)/acc(original)` per task, plus
**answer-flip rate** on the differential jury. Operating-point curves: accuracy vs. compression ratio.

**Gate quality (Layer B — the headline).** Treat `drift` as a soft classifier over MC/MP:
- **AUROC** + **AUPRC** (AUPRC is the headline under MC-rarity) + **MCC** [Chicco & Jurman 2020].
- **Recall-of-MC @ fixed FPR** (e.g. FPR=1%) — a safety gate's real operating summary; a false-accept
  of an MC edit is the dangerous error.
- **Asymmetric-cost threshold:** fix cost(false-accept) ≫ cost(false-reject); pick the operating
  point where the ROC tangent slope = cost·prevalence ratio (iso-cost, [1107.5930]), **on dev**, then
  report confusion metrics at it on test.

**The comparative claim (decoder-JSD beats the alternatives).** On the *same* pairs, compute `drift`,
`1 − cosine(emb(P), emb(P'))`, and `Δperplexity`. Compare AUROCs with **DeLong's paired test**
[DeLong 1988]. JSD wins iff its AUROC/AUPRC is significantly higher.

**Non-spuriousness (the anti-circularity test).** A reference-free distributional metric can secretly
be a perplexity/length/overlap proxy [2204.09890; 2410.21819]. Require `drift` to retain predictive
power for the MC/MP label in **partial correlation controlling for Δperplexity, Δlength, and token
overlap**. If it collapses, the gate is spurious — report it.

**Correlation with the oracle.** Spearman/Kendall between `drift` and the jury's answer-flip rate
(meta-evaluation recipe of MT metrics — BERTScore [1904.09675], BLEURT [2004.04696]; prefer rank /
pairwise-accuracy framing [2305.14324], [2409.09598]).

**Cross-model transfer (the load-bearing unverified assumption).** No prior work shows a compression-
fidelity judgment transfers across models; evil-twins transfer [2311.07064] is only adjacent. So
**test it:** run the same gate-accepted compressions on ≥2–3 held-out deployment models; report
**per-model accuracy retention + variance/CI**, the transfer gap, and **rank stability** (Spearman of
the gate's prompt ordering across models). Treat the proxy as one more held-out model.

**Statistical rigor.** Paired bootstrap [Koehn 2004] for continuous metrics; **McNemar** [Dietterich
1998] for paired pass/fail between gates; **DeLong** CIs for AUROC; Benjamini–Hochberg FDR across the
model × operating-point sweep; an a-priori **power analysis** for minimal-pair set size [2010.06595];
report CIs and effect sizes, not bare p-values.

## Acceptance criteria (what "it works" means)

1. **Gate beats baselines:** decoder-JSD AUROC/AUPRC on MC/MP significantly above cosine and
   perplexity (DeLong p<0.05, FDR-corrected).
2. **Non-spurious:** survives partial-correlation control for perplexity/length/overlap.
3. **Safe operating point:** recall-of-MC ≥ target at FPR ≤ 1% on the held-out test split.
4. **System value:** higher accuracy retention than v1's embedding gate at matched compression ratio.
5. **Transfers:** positive accuracy retention with bounded variance across ≥2 held-out deployment
   models; stable gate ranking.

Falsification is explicit: failing (1) or (2) means the decoder signal adds nothing over a cheaper
scalar, and v2's premise is wrong — report it plainly.

## Evaluation implementation hooks

The `benchmark` verb (spec.md) already runs the pipeline over a corpus. v2 adds, all as pure
functions over plain data so they test without a model:

- `eval/minimal_pairs.py` — deterministic edit operators + IFEval-style verifiers → labeled `(P, P',
  label)` triples. Pure; the label is the construction.
- `eval/gate_metrics.py` — AUROC/AUPRC/MCC/recall@FPR/DeLong/partial-correlation over score arrays.
- `eval/transfer.py` — run accepted compressions across a model panel; report retention + rank
  stability. Panel models are injected (the `Decoder`/deployment boundary), so the analysis core
  stays pure and the only impurity is the model calls at the edge.

Splits (`train`/`dev`/`test`), the canary GUID, and per-item provenance (source, creation date) are
recorded with the corpus, not in code.

## References

Embedding blindness: [2306.05083], [2309.03747], [2504.00584], [2405.06105], [2506.16029].
Output-distribution equivalence: [1503.02531], [2306.13649], [2311.07064], [2503.10337], [2305.14739].
Prompt compression: [2304.12102], [2310.05736], [2310.06839], [2403.12968], [2308.08758], [2102.06272], [2503.19114].
Reasoning as signal / CoT faithfulness: [2201.11903], [2501.12948], [2511.14773], [2305.04388], [2505.05410].
Efficiency / KV reuse / attribution: [2409.00729], [2411.15102], [2405.16444], [2311.04934], [2312.07104], [1703.04730].
Prompt sensitivity / compression attack: [2401.03729], [2310.11324], [2306.04528], [2510.22963].
Verifiable instructions / constraints: [2311.07911], [2401.03601], [2310.20410], [2407.08440].
Redundant-instruction corpora: [2204.07705], [2212.10560], [2212.09689].
Minimal-pair / counterfactual methodology: [2005.04118], [2004.02709], [1909.12434], [1912.00582], [2211.00295], [2112.02721].
Contamination & leak-resistant construction: [2406.04244], [2310.16789], [2404.02936], [2308.08493], [2311.04850], [2311.09783], [2404.00699], [2403.07974], [2406.19314], [2402.19450], [1506.02629], [1902.10811].
Judge bias / role separation / juries: [2306.05685], [2305.17926], [2404.13076], [2410.21819], [2410.02736], [2407.01085], [2404.18796].
Oracle / reference-free / metamorphic: [2511.02108], [1705.06640], [2210.12563], [2204.09890], [2510.11822].
Metric meta-evaluation: [1904.09675], [2004.04696], [2305.14324], [2409.09598].
Benchmarks: [2412.15204], [2305.14196], [2305.17529], [2308.14508].
Cross-model transfer / statistical power: [2010.06595].
Non-arXiv: Koehn 2004 (paired bootstrap); Dietterich 1998 (McNemar); DeLong 1988 (AUROC); Chicco & Jurman 2020 (MCC); Chen et al. 2018 CSUR (metamorphic testing); van der Lee et al. 2021 (human eval).

(arXiv ids; `https://arxiv.org/abs/<id>`.)
