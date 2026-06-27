# Long-Context and Long-Prompt Degradation

## Overview
This note surveys empirical evidence that LLM quality drops as prompts and
context grow longer, even when the model stays well inside its advertised
context window. It answers a question central to Alexandria: is there real,
measured harm in feeding a model long, instruction-heavy prompts, or is
"more context is free" a safe assumption? The literature is consistent and
blunt: longer inputs degrade accuracy, more simultaneous instructions degrade
adherence, and the harm appears at token counts far below the stated limit.
Two effects matter for us. First, raw input *length* hurts performance
independent of content quality. Second, *instruction density* — the number of
distinct directives packed into one prompt — degrades how reliably each
instruction is followed. Both are the empirical case for compressing prompts
and pruning redundant instructions.

## Methods
The papers isolate degradation by holding the task fixed and varying only one
dimension at a time.

- **Positional isolation ("Lost in the Middle", Liu et al. 2023/2024).** Fix
  the question and the gold evidence, then slide the evidence to different
  positions inside a long context. Any change in accuracy is attributable to
  position, not difficulty. The result is a U-shaped curve: accuracy is highest
  when evidence sits at the start or end and sags in the middle.
- **Length isolation via padding ("Same Task, More Tokens", Levy et al. 2024).**
  The FLenQA dataset wraps the *same* reasoning sample in padding of varying
  length, type, and location. Holding the reasoning constant and only inflating
  surrounding tokens isolates the effect of length itself.
- **Length isolation despite perfect retrieval ("Context Length Alone Hurts",
  Du et al. 2025).** The sharpest control yet. They first guarantee the model
  can retrieve the needed evidence, then shrink distraction to nothing by
  replacing irrelevant tokens with whitespace, and finally *mask* irrelevant
  tokens so attention is forced onto the relevant ones. Degradation survives all
  three, proving length harms performance even with no retrieval failure and no
  distractor content.
- **Instruction-density isolation ("How Many Instructions Can LLMs Follow at
  Once?", Jaroslawicz et al. 2025).** The IFScale benchmark stacks up to 500
  verifiable keyword-inclusion instructions onto one business-report task and
  ramps the count from few to many. Because each instruction is independently
  checkable, they get a clean adherence-vs-density curve.

## How it is validated
- **Lost in the Middle** (arXiv:2307.03172; TACL 2024): multi-document QA and
  key-value retrieval; metric is answer accuracy as a function of gold-evidence
  position; evaluated across open and closed long-context models.
- **Same Task, More Tokens** (arXiv:2402.14848; ACL 2024): FLenQA QA-reasoning
  framework; accuracy versus input length, with the same sample padded to
  different lengths/types/locations; includes GPT-4. Degradation is measurable
  by ~3,000 tokens — far below model limits — and is worst when evidence must be
  integrated from two non-adjacent spans.
- **Context Length Alone Hurts LLM Performance Despite Perfect Retrieval**
  (arXiv:2510.05381; Findings of EMNLP 2025): five open and closed models across
  mathematics, QA, and coding; whitespace-replacement and attention-masking
  ablations; RULER for the mitigation. Reported degradation of **13.9%–85%**
  despite perfect retrieval. Their fix — prompting the model to recite the
  retrieved evidence before solving, turning a long-context task into a short
  one — recovers up to 4% for GPT-4o on RULER.
- **How Many Instructions Can LLMs Follow at Once?** (arXiv:2507.11538): IFScale,
  20 models across seven providers; metric is fraction of instructions satisfied
  as density rises to 500. Even the best frontier models reach only **68%
  accuracy at 500 instructions**, with three degradation shapes (threshold,
  linear, exponential decay) and a measured bias toward following *earlier*
  instructions over later ones.

## Relevance to our project
This body of work is the empirical justification for Alexandria's core bet:
shrinking a prompt should help, not just save tokens.

- **Length itself is a cost.** "Context Length Alone Hurts" shows accuracy falls
  by double digits purely from input length, even with perfect retrieval and
  zero distractors. So every redundant sentence we drop from a `Document` (our
  `Section -> Sentence` IR in `core/ir.py`) is not neutral padding — it is
  reclaimed accuracy. This validates compression as a quality lever, not only a
  cost lever.
- **Redundant instructions are distinct directives.** IFScale shows adherence
  decays as instruction count grows, with a bias toward earlier instructions.
  Two sentences that say the same thing in different words are, to the model,
  two instructions competing for the same budget. Our `redundancy` scorer
  (`score/redundancy.py`) — which scores each sentence by its maximum cosine
  similarity to any peer — is exactly the signal needed to find and prune those
  duplicates. Removing a near-duplicate sentence lowers effective instruction
  density without losing intent, which the literature predicts should improve
  following.
- **Order and position matter.** Lost-in-the-Middle and IFScale's earlier-
  instruction bias mean our `greedy_pairwise` selection should care not just
  *which* sentences survive but *where* they land. A shorter, denser prompt also
  shrinks the "middle" region where evidence gets lost.
- **Accuracy validation must be label-free but real.** Because the harm is
  measurable (13.9%–85%), our validation should confirm that a compressed prompt
  preserves task accuracy. The recite-the-evidence mitigation from Du et al.
  is essentially manual compression by the model at inference time; Alexandria
  does the same thing ahead of time and statically. That parallel is a useful
  sanity check: if reciting only the relevant evidence helps the model, then
  pre-pruning to only the relevant instructions should too.

## Related papers
- [Liu et al. — Lost in the Middle: How Language Models Use Long Contexts (2023/2024)] — Accuracy follows a U-shaped curve over evidence position; models lose information placed in the middle of long contexts. arXiv:2307.03172; TACL 2024. https://arxiv.org/abs/2307.03172
- [Levy, Jacoby, Goldberg — Same Task, More Tokens: the Impact of Input Length on the Reasoning Performance of Large Language Models (2024)] — Padding the same reasoning sample to greater length degrades accuracy by ~3,000 tokens, worst when integrating non-adjacent evidence. arXiv:2402.14848; ACL 2024. https://arxiv.org/abs/2402.14848
- [Du et al. — Context Length Alone Hurts LLM Performance Despite Perfect Retrieval (2025)] — Input length alone degrades performance 13.9%–85% even with perfect retrieval and distractors masked out; reciting evidence first recovers up to 4% on RULER. arXiv:2510.05381; Findings of EMNLP 2025. https://arxiv.org/abs/2510.05381
- [Jaroslawicz et al. — How Many Instructions Can LLMs Follow at Once? (2025)] — IFScale benchmark of up to 500 instructions; best frontier models hit only 68% adherence at max density, with bias toward earlier instructions. arXiv:2507.11538. https://arxiv.org/abs/2507.11538
