# Prompt Compression and Optimization

## Overview
This note surveys recent prompt compression work to answer one question: how do
state-of-the-art methods shorten long instruction-heavy prompts without losing
task accuracy, and what does that imply for Alexandria? We cover three concrete
works spanning the two dominant compression paradigms — token-level pruning
(LLMLingua-2) and sentence-level relevance filtering (CPC) — plus a 2025
evaluation framework that measures what compression actually destroys
(information preservation). Together they frame the trade-off space Alexandria
operates in: most published methods are query-dependent and need labeled or
distilled supervision, whereas Alexandria targets label-free, instruction-level
redundancy removal.

## Methods

### LLMLingua-2 — token classification via data distillation (ACL 2024 Findings, arXiv 2403.12968)
LLMLingua-2 reframes prompt compression as a binary token-classification problem:
for each token, keep or drop. It first builds an extractive compression dataset by
prompting GPT-4 to losslessly compress texts (MeetingBank), then distills that
signal into a small bidirectional Transformer encoder (XLM-RoBERTa-large / mBERT)
that labels each token keep/drop. This is a deliberate move away from the original
LLMLingua line, which scored tokens by the *perplexity* assigned by a small causal
LM (GPT-2 / LLaMA-7B) and dropped low-perplexity tokens under a budget controller.
The classification approach is task-agnostic and faithful (it can only delete
original tokens, never hallucinate), runs 3x–6x faster than the perplexity-based
methods, and delivers 1.6x–2.9x end-to-end latency reduction at 2x–5x compression
ratios. Key idea: a cheap, bidirectional, distilled classifier replaces an
expensive unidirectional perplexity estimate.

### CPC — context-aware sentence encoding (AAAI-25, arXiv 2409.01227)
CPC (Context-aware Prompt Compression) moves the unit of compression from tokens to
whole sentences. A custom sentence encoder, trained contrastively on a
Context-aware Question-Relevance (CQR) dataset of (question, positive sentence,
negative sentence) triples, produces a relevance score as the cosine similarity
between the question embedding and each context-sentence embedding. Compression is
then simply: rank sentences by relevance and drop the least-relevant ones until the
token budget is met. Because it deletes whole sentences, the compressed prompt stays
human-readable — unlike token-level LLMLingua output, which is often fragmented.
CPC reports up to 10.93x faster inference than the best token-level baseline and
beats LongLLMLingua on long-context benchmarks. Key idea: question-conditioned
sentence relevance, learned via contrastive embeddings, gives readable
sentence-granular compression. A 2025 follow-up, TPC (arXiv 2502.13374), removes the
need for an explicit question by generating a task description with an
RL-trained task descriptor, making the same sentence-relevance machinery
task-agnostic.

### Understanding and Improving Information Preservation (EMNLP 2025 Findings, arXiv 2503.19114)
Rather than proposing a faster compressor, this work builds a holistic evaluation
framework that scores compression methods on three axes: downstream task
performance, grounding in the input context, and fine-grained information
preservation (how many entities, numbers, and proper nouns survive). The central
finding is that several state-of-the-art soft and hard methods silently drop key
details, which caps their performance on complex tasks. By controlling
*compression granularity* in a soft-prompting method, the authors recover up to
+23% downstream performance, +8 BERTScore points in grounding, and 2.7x more
entities preserved. Their conclusion: the best effectiveness/compression trade-off
comes from soft prompting combined with sequence-level compression. Key idea:
compression quality is not just an accuracy number — it is whether the specific
facts a task depends on survive the cut.

## How it is validated
- **Benchmarks.** LongBench and ZeroSCROLLS are the shared long-context evaluation
  suites across CPC and LLMLingua-2; LLMLingua-2 adds MeetingBank (in-domain),
  GSM8K, and BBH for out-of-domain generalization.
- **Metrics.** Downstream task scores (QA accuracy, summarization quality),
  compression ratio (tokens kept vs. original), and end-to-end latency / inference
  speedup. LLMLingua-2 reports 1.6x–2.9x latency reduction at 2x–5x compression;
  CPC reports up to 10.93x faster inference than the best token-level method.
- **Baselines.** Methods are compared against each other in a chain: Selective
  Context and LLMLingua/LongLLMLingua (perplexity / self-information) are the
  baselines for LLMLingua-2; LongLLMLingua is the baseline CPC and TPC beat on
  LongBench/ZeroSCROLLS.
- **Information-level evaluation.** The EMNLP 2025 framework goes beyond aggregate
  accuracy, adding grounding (BERTScore against source) and entity/number
  retention counts — measuring *what* is lost, not just *how much* accuracy drops.

## Relevance to our project
Alexandria sits in a different corner of this design space, and the contrast is the
point.

- **Unit of compression.** LLMLingua-2 deletes *tokens*; CPC deletes *sentences*
  relative to a question. Alexandria deletes/merges *instructions* — the natural
  unit of an instruction-heavy prompt. Our `InstructionSet` (split → encode → score
  → greedily merge/drop) is closest in spirit to CPC's sentence-level, readable
  compression, but our merge step can collapse two redundant instructions into one
  rather than only dropping.
- **What we score.** LLMLingua-2 scores *importance* via a GPT-4-distilled
  classifier; CPC scores *query relevance* via a contrastive encoder. Both require
  supervision (distilled labels or constructed positive/negative pairs). Alexandria
  scores *redundancy* — pairwise embedding similarity between instructions — which
  is **label-free and query-free**. We drop or merge an instruction because another
  instruction already covers it, not because a model judged it unimportant or
  irrelevant to a question.
- **Why label-free fits us.** CPC's relevance is conditioned on a question and TPC
  needs an RL-trained task descriptor; LLMLingua-2 needs a GPT-4 distillation
  pipeline. Alexandria deliberately avoids all of that: redundancy among
  instructions is computable directly from sentence embeddings (the
  `greedy_pairwise` scorer already in the repo), with no labels, no question, and no
  teacher LLM. This is the cheapest possible supervision signal and the right one
  for self-redundant instruction sets.
- **Accuracy validation.** The EMNLP 2025 information-preservation work is the most
  directly actionable for us: it warns that aggregate accuracy can stay flat while
  critical entities/numbers silently vanish. Our validation phase should therefore
  not only confirm task accuracy is preserved, but check that compression did not
  drop instructions carrying load-bearing specifics (formats, constraints, named
  values) — an instruction-level analogue of their entity-retention metric.
- **Borrowable evaluation harness.** LongBench / ZeroSCROLLS and the
  compression-ratio-vs-accuracy curve are the field-standard way to report results;
  we can adopt the same axes (token-reduction ratio on the x-axis, accuracy
  retention on the y-axis) to make Alexandria comparable to these baselines, while
  noting our label-free setting.

## Related papers
- [Pan, Wu, Jiang et al. — LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression (2024)] — token-classification compression distilled from GPT-4; 1.6x–2.9x latency cut at 2x–5x ratios. ACL 2024 Findings, arXiv:2403.12968 — https://arxiv.org/abs/2403.12968
- [Liskavets, Ushakov, Roy et al. — Prompt Compression with Context-Aware Sentence Encoding for Fast and Improved LLM Inference (2024)] — sentence-level, query-relevance compression via a contrastive encoder; readable output, up to 10.93x faster inference. AAAI-25, arXiv:2409.01227 — https://arxiv.org/abs/2409.01227
- [Łajewska, Hardalov, Aina et al. — Understanding and Improving Information Preservation in Prompt Compression for LLMs (2025)] — evaluation framework measuring downstream accuracy, grounding, and entity retention; +23% downstream, 2.7x more entities preserved via granularity control. EMNLP 2025 Findings, arXiv:2503.19114 — https://arxiv.org/abs/2503.19114
- [Liskavets, Roy, Ushakov et al. — Task-agnostic Prompt Compression with Context-aware Sentence Embedding and Reward-guided Task Descriptor (2025)] — removes the explicit-question requirement of CPC via an RL-trained task descriptor. arXiv:2502.13374 — https://arxiv.org/abs/2502.13374
- [Jiang, Wu, Lin et al. — LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression (2023)] — question-aware coarse-to-fine perplexity compression with document reordering; +17.1% over original prompt at 4x fewer tokens. ACL 2024, arXiv:2310.06839 — https://arxiv.org/abs/2310.06839
- [Li, Liu, Su, Collier — Prompt Compression for Large Language Models: A Survey (2024)] — taxonomy of hard- vs. soft-prompt compression and the key representative methods. NAACL 2025 (Oral), arXiv:2410.12388 — https://arxiv.org/abs/2410.12388
