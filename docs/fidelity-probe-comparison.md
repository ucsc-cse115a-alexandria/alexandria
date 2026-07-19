# Fidelity signal comparison — sentence-embedding cosine vs spec-v2 `behavioral_jsd`

Both signals scored on the **same** `literature` minimal-pair catalog (`scripts/fidelity_pairs.json`). 
This report is generated from a live run; regenerate it by rerunning the two probes.

## Setup

- **Catalog:** `literature` — 23 pairs (16 meaning-changing, 6 meaning-preserving, 1 unrelated control)
- **Embedding (v1):** bi-encoder `sentence-transformers/all-MiniLM-L6-v2`, cosine similarity (high = keep)
- **Behavioral (spec-v2):** `mlx-community/Qwen2.5-0.5B-Instruct-bf16` via MLX, probe=8 tokens, max-position JSD in bits (low = keep)

## Headline — does the signal separate meaning-changing (MC) from meaning-preserving (MP)?

| metric | cosine (v1) | `behavioral_jsd` (spec-v2) |
|---|---|---|
| **AUROC** (MC vs MP, higher = better separation) | 0.260 | 0.552 |
| MC flips leaking under the keep-all-paraphrases cutoff | 14/16 | 12/16 |
| MC flips indistinguishable from the best paraphrase (BLIND) | 10/16 | 5/16 |
| MC range | [0.023, 0.998] | [0.004, 0.998] |
| MP range | [0.444, 0.982] | [0.046, 0.632] |

AUROC 0.5 = no separation, 1.0 = perfect. Cosine **0.260** vs behavioral **0.552**.

## Per-pair — both signals side by side

`cos` = cosine (↑ keep). `behavioral_jsd` = JSD bits (↓ keep). Rows are sorted by behavioral_jsd ascending (behaviorally closest first), so an ideal behavioral signal would put every MP row at the top and every MC row at the bottom.

| label | kind | cos | behavioral_jsd | original → edited |
|---|---|---:|---:|---|
| MC | version-swap | 0.945 | 0.004 | `Use the v1 API endpoint.` → `Use the v2 API endpoint.` |
| MC | unit-swap | 0.986 | 0.023 | `Set the request timeout to 30 seconds.` → `Set the request timeout to 30 minutes.` |
| MC | attribute-binding | 0.990 | 0.044 | `Make the title bold and the body italic.` → `Make the title italic and the body bold.` |
| MC | numeric-bound | 0.869 | 0.045 | `Summarize in 3 bullet points.` → `Summarize in 30 bullet points.` |
| MP | reorder | 0.982 | 0.046 | `Respond in JSON. Keep it short.` → `Keep it short. Respond in JSON.` |
| MC | role-swap | 0.994 | 0.048 | `The reviewer must approve the author's changes.` → `The author must approve the reviewer's changes.` |
| MP | synonym | 0.875 | 0.063 | `Respond in a concise manner.` → `Respond in a brief manner.` |
| MP | redundant-drop | 0.842 | 0.093 | `Respond in JSON. Your final answer must be valid JSON.` → `Respond in JSON.` |
| MC | role-swap | 0.998 | 0.156 | `Forward the summary from the agent to the user.` → `Forward the summary from the user to the agent.` |
| MC | numeric-bound | 0.826 | 0.169 | `Use at most 50 words.` → `Use at most 500 words.` |
| MP | paraphrase | 0.731 | 0.384 | `Respond in JSON.` → `Format your answer as JSON.` |
| MC | negation-verbal | 0.890 | 0.406 | `Include code examples in the answer.` → `Do not include code examples in the answer.` |
| MC | negation-lexical | 0.875 | 0.418 | `Keep the original formatting.` → `Strip the original formatting.` |
| MC | antonym | 0.919 | 0.533 | `Keep the response short.` → `Keep the response long.` |
| MP | paraphrase | 0.530 | 0.546 | `Summarize the text in three bullet points.` → `Condense the passage into 3 bullets.` |
| MC | boilerplate-inflation | 0.990 | 0.584 | `Please make sure that you carefully follow every formatting rule below and respond in valid JSON.` → `Please make sure that you carefully follow every formatting rule below and do not respond in valid JSON.` |
| MC | comparator-flip | 0.993 | 0.618 | `Flag any response above 0.8 confidence.` → `Flag any response below 0.8 confidence.` |
| MP | paraphrase | 0.444 | 0.632 | `Keep the response short.` → `Be concise in your reply.` |
| MC | quantifier-scope | 0.928 | 0.651 | `Translate all comments into French.` → `Translate some comments into French.` |
| unrelated | control | -0.009 | 0.977 | `Respond in JSON.` → `The weather in Santa Cruz is sunny today.` |
| MC | negation-verbal | 0.845 | 0.981 | `Respond in JSON.` → `Do not respond in JSON.` |
| MC | format-swap | 0.440 | 0.997 | `Respond in JSON.` → `Respond in YAML.` |
| MC | topic-change | 0.023 | 0.998 | `Respond in valid JSON.` → `Write a haiku about autumn leaves.` |

## Where they disagree most — the diagnostic cases

MC flips that **cosine misses but behavioral_jsd catches** (cosine high, behavioral_jsd high):

- **role-swap** — cos 0.998 (looks safe) / behavioral_jsd 0.156 — `Forward the summary from the agent to the user.` → `Forward the summary from the user to the agent.`
- **comparator-flip** — cos 0.993 (looks safe) / behavioral_jsd 0.618 — `Flag any response above 0.8 confidence.` → `Flag any response below 0.8 confidence.`
- **boilerplate-inflation** — cos 0.990 (looks safe) / behavioral_jsd 0.584 — `Please make sure that you carefully follow every formatting rule below and respond in valid JSON.` → `Please make sure that you carefully follow every formatting rule below and do not respond in valid JSON.`
- **quantifier-scope** — cos 0.928 (looks safe) / behavioral_jsd 0.651 — `Translate all comments into French.` → `Translate some comments into French.`
- **antonym** — cos 0.919 (looks safe) / behavioral_jsd 0.533 — `Keep the response short.` → `Keep the response long.`
- **negation-verbal** — cos 0.890 (looks safe) / behavioral_jsd 0.406 — `Include code examples in the answer.` → `Do not include code examples in the answer.`

MC flips that **both still miss** (cosine high, behavioral_jsd low — the hard residual):

- **version-swap** — cos 0.945 / behavioral_jsd 0.004 — `Use the v1 API endpoint.` → `Use the v2 API endpoint.`
- **unit-swap** — cos 0.986 / behavioral_jsd 0.023 — `Set the request timeout to 30 seconds.` → `Set the request timeout to 30 minutes.`
- **attribute-binding** — cos 0.990 / behavioral_jsd 0.044 — `Make the title bold and the body italic.` → `Make the title italic and the body bold.`
- **role-swap** — cos 0.994 / behavioral_jsd 0.048 — `The reviewer must approve the author's changes.` → `The author must approve the reviewer's changes.`

## Context dilution — one negation, growing shared context

Same meaning flip (`respond` → `do not respond`) buried under more shared context. A good fidelity signal should stay decisive as the prompt grows.

| shared tokens | cosine (→1 = hides flip) | behavioral_jsd (stays high = keeps flip visible) |
|---:|---:|---:|
| 7 | 0.891 | 0.875 |
| 10 | 0.896 | 0.842 |
| 12 | 0.937 | 0.964 |
| 19 | 0.952 | 0.824 |

## Speed — warm cache, Apple Silicon arm64

- Embedding probe: 23 pairs in 0.53s (23 ms/pair)
- Behavioral probe (MLX): 23 pairs in 2.56s (111 ms/pair)

The embedding signal is two forward passes of a 22M bi-encoder; the behavioral signal is a greedy probe decode plus two teacher-forced passes of a 0.5B decoder — far heavier per pair, which is why spec-v2 keeps it as the Tier-2 gate behind the cheap embedding screen.
