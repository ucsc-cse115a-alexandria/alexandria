import tiktoken

from alexandria.ir.similarity import cosine_distance
from alexandria.utils.embedders import default_embedder

DRIFT_BUDGET = 0.5
SIMILARITY_GATE = 1.0 - DRIFT_BUDGET


def check_compression(original: str, compressed: str) -> dict:
    embedder = default_embedder()
    enc = tiktoken.get_encoding("cl100k_base")

    orig_vec, comp_vec = embedder.embed([original, compressed])

    drift = cosine_distance(orig_vec, comp_vec)
    cosine = 1.0 - drift

    source_tokens = len(enc.encode(original))
    reduced_tokens = len(enc.encode(compressed))

    token_reduction = 1.0 - reduced_tokens / source_tokens if source_tokens else 0.0

    return {
        "source_tokens": source_tokens,
        "reduced_tokens": reduced_tokens,
        "token_reduction": token_reduction,
        "cosine_similarity": cosine,
        "cosine_drift": drift,
        "passed": drift <= DRIFT_BUDGET,
    }
