import tiktoken

from alexandria.ir.similarity import compute_cos_sim_diff
from alexandria.utils.embedders import default_embedder

COS_SIM_DIFF_BUDGET = 0.5
SIMILARITY_GATE = 1.0 - COS_SIM_DIFF_BUDGET


def check_compression(original: str, compressed: str) -> dict:
    embedder = default_embedder()
    enc = tiktoken.get_encoding("cl100k_base")

    orig_vec, comp_vec = embedder.embed([original, compressed])

    cos_sim_diff = compute_cos_sim_diff(orig_vec, comp_vec)
    cosine = 1.0 - cos_sim_diff

    source_tokens = len(enc.encode(original))
    reduced_tokens = len(enc.encode(compressed))

    token_reduction = 1.0 - reduced_tokens / source_tokens if source_tokens else 0.0

    return {
        "source_tokens": source_tokens,
        "reduced_tokens": reduced_tokens,
        "token_reduction": token_reduction,
        "cosine_similarity": cosine,
        "cos_sim_diff": cos_sim_diff,
        "passed": cos_sim_diff <= COS_SIM_DIFF_BUDGET,
    }
