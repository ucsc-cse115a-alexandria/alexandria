# Vendored IFEval checkers

Source: https://github.com/google-research/google-research/tree/master/instruction_following_eval
Commit: 5b09c22d73a9d35eb6c5d2a99b95677a45053466
License: Apache-2.0 (see LICENSE)

Files: instructions.py, instructions_registry.py, instructions_util.py, evaluation_lib.py.
Only modification: `from instruction_following_eval import X` rewritten to
`from benchmarks.ifeval.vendor import X`. evaluation_lib.pyi is ours — a minimal stub so
pyright-strict callers type-check against the untyped vendored module.
Dataset is not vendored; fetch it with `uv run python -m scripts.download_ifeval_data`.
