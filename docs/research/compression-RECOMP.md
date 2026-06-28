# [RECOMP](https://arxiv.org/abs/2310.04408)

## Overview
This paper proposes RECOMP (Retrieve, Compress, Prepend), a framework that compresses retrieved documents into concise textual summaries before prepending them to a language model's input. Instead of feeding raw documents directly into the model, RECOMP inserts a compression step that reduces token count while preserving the information needed for the downstream task. It presents two compressors: an extractive one that selects relevant sentences, and an abstractive one that generates summaries by synthesizing information across multiple documents. If retrieved documents are irrelevant, the compressor returns an empty string, implementing selective augmentation.

## Methods
- **Extractive compressor:** trains a dual-encoder model using contrastive learning to rank sentences from retrieved documents by how much they improve end task performance when prepended to the input.
- **Abstractive compressor:** distills summarization ability from GPT-3.5 by generating training data with it, filtering examples where the summary hurts performance, and fine-tuning a smaller T5-large (775M) encoder-decoder model on the filtered data.
- **Selective augmentation:** if retrieved documents are irrelevant or unhelpful, the compressor returns an empty string rather than prepending noise.
- **End-task-driven training:** both compressors are trained to optimize downstream task performance (perplexity for language modeling, exact match for QA) rather than intrinsic summarization quality.

## How it is validated
**Benchmarks:**

- WikiText-103 (language modeling)
- Natural Questions (open-domain QA)
- TriviaQA (open-domain QA)
- HotpotQA (multi-hop QA)

**Baselines compared against:**

- Heuristic: Bag of Words, Named Entity extraction
- Retrieval: BM25, Contriever, DPR
- Summarization: off-the-shelf T5, GPT-3.5

**Key results:**

- 6% compression rate with minimal performance loss on language modeling
- On NQ, abstractive compressor achieves 5% token compression with less than 2 EM point drop vs full documents
- Compressors trained on one LM transfer well to others

## Relevance to our project
- RECOMP's abstractive compressor is the closest existing analogue to our merge optimizer collapsing redundant sentences into a single representative one via a Replace edit
- Their end-task-driven training contrasts with our label-free approach. We use embedding similarity to detect redundancy without any target output to optimize against
- Their finding that selective augmentation (returning empty when retrieval is unhelpful) supports our premise that removing content can help rather than hurt
- RECOMP operates on retrieved documents while we operate on instruction prompts, a different domain but the same core problem of identifying and removing redundant or irrelevant content

## Related papers
- [Chevalier et al. - AutoCompressor (2023)](https://arxiv.org/abs/2305.14788) - compresses long contexts into soft prompt summary vectors, a soft compression alternative to RECOMP's text-based approach

- [Mu et al. - GIST tokens (2024)](https://arxiv.org/abs/2304.08467) - trains LMs to compress prompts into special "gist" tokens, achieving up to 26x compression with minimal quality loss