"""Probe spec-v2's *behavioral* fidelity signal on the same minimal pairs as the embedding probe.

Where `sentence_embedding_fidelity_probe.py` scores each pair by sentence-embedding cosine (a topic
summary), this script scores it by spec-v2's decoder-distribution gate: run a causal LM, teacher-force
a fixed probe continuation `c`, and compare the per-position next-token distributions the two prompts
induce.

    c            = greedy decode of the model under the *original* prompt   (decoded once per pair)
    signature(p) = [ softmax(logits(p . c_<t)) for t in positions of c ]    (teacher-forced, one pass)
    behavioral_jsd(p, p') = max over positions t of  JSD( signature(p)[t] || signature(p')[t] )

JSD is in bits, so behavioral_jsd is bounded in [0, 1]: 0 means the two prompts leave the model about
to do the same thing, 1 means maximally different. A meaning-changing (MC) edit should have high
behavioral_jsd; a meaning-preserving (MP) edit should have low behavioral_jsd -- the separation
cosine fails to make on flips.

Two backends produce the distributions behind a common interface: `mlx` (Apple-silicon native,
`mlx-lm`) and `torch` (`transformers`). The JSD math is pure numpy, so the signal is identical across
backends and only the speed differs. Running a local model exposes the *full* logits, so JSD is
computed exactly over the whole vocabulary; the top-k truncated-tail handling spec-v2 describes is only
needed for logprob-capped API backends.
"""

from __future__ import annotations

import argparse
import time
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
from fidelity_pairs import DEFAULT_PAIRS_PATH, Label, Pair, load_catalogs
from pydantic import BaseModel

if TYPE_CHECKING:
    from numpy.typing import NDArray

SCORING_CATALOGS = ("literature", "agent")
DILUTION_CATALOG = "dilution"
_EPS = 1e-12

_DEFAULT_MODELS = {
    "mlx": "mlx-community/Qwen2.5-0.5B-Instruct-bf16",
    "torch": "Qwen/Qwen2.5-0.5B-Instruct",
}


class Backend(StrEnum):
    """Which framework produces the next-token distributions."""

    MLX = "mlx"
    TORCH = "torch"


class PairResult(BaseModel):
    """Probe and behavioral_jsd (max-position JSD, in bits) for a scored pair."""

    label: Label
    kind: str
    source: str
    original: str
    edited: str
    probe: str
    probe_tokens: int
    behavioral_jsd: float


class Decoder(Protocol):
    """The three model operations the probe needs, behind one backend-agnostic surface."""

    model_id: str

    def encode_prompt(self, text: str) -> list[int]: ...
    def greedy_probe(self, prompt_ids: list[int], probe_tokens: int) -> list[int]: ...
    def signature(self, prompt_ids: list[int], probe_ids: list[int]) -> NDArray[np.float32]: ...
    def detokenize(self, ids: list[int]) -> str: ...


class MlxDecoder:
    """mlx-lm backend: Apple-silicon native, full-precision logits."""

    def __init__(self, model_id: str) -> None:
        from mlx_lm import load

        self.model_id = model_id
        self._model, self._tokenizer = load(model_id)

    def encode_prompt(self, text: str) -> list[int]:
        return _encode_prompt(self._tokenizer, text)

    def greedy_probe(self, prompt_ids: list[int], probe_tokens: int) -> list[int]:
        import mlx.core as mx

        ids = list(prompt_ids)
        probe: list[int] = []
        for _ in range(probe_tokens):
            next_logits = self._model(mx.array([ids]))[0, -1]
            token = int(mx.argmax(next_logits).item())
            probe.append(token)
            ids.append(token)
        return probe

    def signature(self, prompt_ids: list[int], probe_ids: list[int]) -> NDArray[np.float32]:
        import mlx.core as mx

        logits = self._model(mx.array([prompt_ids + probe_ids]))[0]
        start = len(prompt_ids) - 1
        window = logits[start : start + len(probe_ids)].astype(mx.float32)
        return np.asarray(mx.softmax(window, axis=-1), dtype=np.float32)

    def detokenize(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids)


class TorchDecoder:
    """transformers backend on the best available torch device (cuda / mps / cpu)."""

    def __init__(self, model_id: str, device: str | None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self._torch = torch
        self._device = device or _pick_torch_device(torch)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(model_id).to(self._device).eval()

    def encode_prompt(self, text: str) -> list[int]:
        return _encode_prompt(self._tokenizer, text)

    def greedy_probe(self, prompt_ids: list[int], probe_tokens: int) -> list[int]:
        with self._torch.no_grad():
            inputs = self._torch.tensor([prompt_ids], device=self._device)
            generated = self._model.generate(
                inputs, do_sample=False, min_new_tokens=probe_tokens, max_new_tokens=probe_tokens
            )
        return generated[0, len(prompt_ids) :].tolist()

    def signature(self, prompt_ids: list[int], probe_ids: list[int]) -> NDArray[np.float32]:
        with self._torch.no_grad():
            full = self._torch.tensor([prompt_ids + probe_ids], device=self._device)
            logits = self._model(full).logits[0]
            start = len(prompt_ids) - 1
            window = logits[start : start + len(probe_ids)].float()
            probs = self._torch.softmax(window, dim=-1)
        return probs.cpu().numpy().astype(np.float32)

    def detokenize(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids)


def _encode_prompt(tokenizer: object, text: str) -> list[int]:
    """Token ids for a prompt: through the chat template when the model has one, else raw encode."""
    template = getattr(tokenizer, "chat_template", None)
    if template:
        ids = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True, return_dict=False
        )
        return list(ids)
    return list(tokenizer.encode(text))  # type: ignore[attr-defined]


def _pick_torch_device(torch: object) -> str:
    if torch.cuda.is_available():  # type: ignore[attr-defined]
        return "cuda"
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps"
    return "cpu"


def build_decoder(backend: Backend, model_id: str, device: str | None) -> Decoder:
    """Construct the requested backend's decoder."""
    match backend:
        case Backend.MLX:
            return MlxDecoder(model_id)
        case Backend.TORCH:
            return TorchDecoder(model_id, device)


def max_token_jsd(p: NDArray[np.float32], q: NDArray[np.float32]) -> float:
    """Max over positions of the Jensen-Shannon divergence (bits) between two distribution stacks."""
    m = 0.5 * (p + q)
    kl_pm = (p * (np.log2(p + _EPS) - np.log2(m + _EPS))).sum(axis=-1)
    kl_qm = (q * (np.log2(q + _EPS) - np.log2(m + _EPS))).sum(axis=-1)
    jsd = 0.5 * kl_pm + 0.5 * kl_qm
    return float(np.clip(jsd.max(), 0.0, 1.0))


def score_pair(pair: Pair, decoder: Decoder, probe_tokens: int) -> PairResult:
    """Decode under the original prompt, then compare the edited signature with behavioral_jsd."""
    original_ids = decoder.encode_prompt(pair.original)
    edited_ids = decoder.encode_prompt(pair.edited)
    probe_ids = decoder.greedy_probe(original_ids, probe_tokens)
    base = decoder.signature(original_ids, probe_ids)
    trial = decoder.signature(edited_ids, probe_ids)
    return PairResult(
        label=pair.label,
        kind=pair.kind,
        source=pair.source,
        original=pair.original,
        edited=pair.edited,
        probe=decoder.detokenize(probe_ids),
        probe_tokens=len(probe_ids),
        behavioral_jsd=max_token_jsd(base, trial),
    )


def print_dilution(results: list[PairResult]) -> None:
    """Show behavioral_jsd staying high as shared context buries a flip and cosine climbs."""
    print("\nContext dilution -- one negation, growing shared context:")
    print(f"  {'kind':<10}  {'behavioral_jsd':>6}  edit")
    for r in sorted(results, key=lambda r: r.kind):
        print(f"  {r.kind:<10}  {r.behavioral_jsd:>6.3f}  {r.edited!r}")
    print(
        "  => Same meaning flip; behavioral_jsd should stay high regardless of context length -- "
        "the flip stays visible."
    )


def print_table(results: list[PairResult]) -> None:
    """Print one row per pair, sorted by behavioral_jsd ascending (behaviorally closest first)."""
    header = f"{'label':<9} {'kind':<22} {'src':<11} {'behavioral_jsd':>6}  original -> edited"
    print(header)
    print("-" * len(header))
    for result in sorted(results, key=lambda r: r.behavioral_jsd):
        print(
            f"{result.label.value:<9} {result.kind:<22} {result.source:<11} {result.behavioral_jsd:>6.3f}  "
            f"{result.original!r} -> {result.edited!r}"
        )


def print_threshold_squeeze(mc: list[PairResult], mp: list[PairResult]) -> None:
    """A gate keeps an edit when behavioral_jsd <= cutoff. Report both ends: the cutoff that keeps every
    paraphrase (and how many flips leak under it) and the cutoff that blocks all-but-one flip (and how
    many paraphrases it costs). If JSD separates MC from MP, both ends can be good at once.
    """
    keep_all_cutoff = max(r.behavioral_jsd for r in mp)
    leaked = sum(r.behavioral_jsd <= keep_all_cutoff for r in mc)
    print(
        f"\nKeep all {len(mp)} paraphrases (cutoff {keep_all_cutoff:.3f}): "
        f"{leaked}/{len(mc)} meaning flips leak through with them."
    )

    safety_cutoff = sorted(r.behavioral_jsd for r in mc)[1] - 1e-6
    survivors = sum(r.behavioral_jsd <= safety_cutoff for r in mp)
    print(
        f"Allow at most one leaked flip (cutoff {safety_cutoff:.3f}): "
        f"{survivors}/{len(mp)} paraphrases survive the gate."
    )


def print_summary(results: list[PairResult]) -> None:
    """Show whether behavioral_jsd separates MC from MP -- and call out the dangerous low-behavioral_jsd flips."""
    mp = [r for r in results if r.label is Label.MEANING_PRESERVING]
    mc = [r for r in results if r.label is Label.MEANING_CHANGING]
    if not (mp and mc):
        return

    print("\nMean behavioral_jsd by kind:")
    for kind in dict.fromkeys(r.kind for r in results):
        behavioral_jsd_values = [r.behavioral_jsd for r in results if r.kind == kind]
        label = next(r.label.value for r in results if r.kind == kind)
        mean_behavioral_jsd = sum(behavioral_jsd_values) / len(behavioral_jsd_values)
        print(f"  {label:<9} {kind:<22} n={len(behavioral_jsd_values)}  mean={mean_behavioral_jsd:.3f}")

    print(
        f"\nDistributions: MC behavioral_jsd in "
        f"[{min(r.behavioral_jsd for r in mc):.3f}, {max(r.behavioral_jsd for r in mc):.3f}], "
        f"MP behavioral_jsd in [{min(r.behavioral_jsd for r in mp):.3f}, {max(r.behavioral_jsd for r in mp):.3f}]."
    )

    print_threshold_squeeze(mc, mp)

    paraphrase_floor = min(r.behavioral_jsd for r in mp if r.kind in {"paraphrase", "synonym"})
    blind = sorted((r for r in mc if r.behavioral_jsd < paraphrase_floor), key=lambda r: r.behavioral_jsd)
    print(f"\nBLIND -- meaning flips below the closest paraphrase ({paraphrase_floor:.3f}): {len(blind)}/{len(mc)}")
    for r in blind:
        print(f"  {r.behavioral_jsd:.3f}  [{r.kind}]  {r.original!r} -> {r.edited!r}")

    overeager = sorted(
        (r for r in mp if r.behavioral_jsd > min(r.behavioral_jsd for r in mc)),
        key=lambda r: r.behavioral_jsd,
        reverse=True,
    )
    rejected = [r for r in overeager if r.kind in {"paraphrase", "synonym"}]
    print(f"\nOVER-EAGER -- valid paraphrases above some meaning flip: {len(rejected)}/{len(mp)}")
    for r in rejected:
        print(f"  {r.behavioral_jsd:.3f}  [{r.kind}]  {r.original!r} -> {r.edited!r}")

    print(
        "\n=> A behavioral gate works iff MC behavioral_jsd sits above MP behavioral_jsd with a usable "
        "margin -- catching "
        "the truth-flipping operators cosine is blind to while keeping honest paraphrases."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", type=Backend, choices=list(Backend), default=Backend.MLX, help="decoder backend")
    parser.add_argument("--model", default=None, help="causal LM id (default: a small Qwen for the chosen backend)")
    parser.add_argument("--probe-tokens", type=int, default=8, help="length of the teacher-forced probe continuation")
    parser.add_argument(
        "--catalog",
        choices=SCORING_CATALOGS,
        default="literature",
        help="which minimal-pair catalog to score: paper-grounded or agent-prompt-grounded",
    )
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS_PATH, help="path to the minimal-pair catalog JSON")
    parser.add_argument("--device", default=None, help="torch device override (torch backend only)")
    args = parser.parse_args()

    model_id = args.model or _DEFAULT_MODELS[args.backend.value]
    catalogs = load_catalogs(args.pairs)

    load_start = time.perf_counter()
    decoder = build_decoder(args.backend, model_id, args.device)
    load_seconds = time.perf_counter() - load_start

    def score(pairs: list[Pair]) -> tuple[list[PairResult], float]:
        start = time.perf_counter()
        scored = [score_pair(pair, decoder, args.probe_tokens) for pair in pairs]
        return scored, time.perf_counter() - start

    results, score_seconds = score(catalogs[args.catalog])

    print(f"backend={args.backend.value}  model={model_id}  probe_tokens={args.probe_tokens}")
    print(
        f"load {load_seconds:.2f}s  |  scored {len(results)} pairs in {score_seconds:.2f}s "
        f"({score_seconds / len(results) * 1000:.0f} ms/pair, {len(results) / score_seconds:.1f} pairs/s)\n"
    )
    print_table(results)
    print_summary(results)
    if args.catalog == "literature":
        dilution, _ = score(catalogs[DILUTION_CATALOG])
        print_dilution(dilution)


if __name__ == "__main__":
    main()
