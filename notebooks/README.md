# Alexandria notebooks

Notebook-only packages are kept in the root `pyproject.toml` development dependency group. `uv sync --dev`
creates the root `.venv` automatically and installs Alexandria in editable mode, so notebook cells import the
working tree rather than a separately published package.

## Setup

From the repository root:

```bash
uv sync --dev
uv run python -m scripts.download_babilong_8k_data
uv run jupyter lab notebooks/section_embedding_clusters.ipynb
```

Set `OPENAI_API_KEY` before running the notebook. The notebook uses Alexandria's default
`text-embedding-3-small` embedder through `represent()`.

The skill input defaults to the sibling checkout at:

```text
../skill-corpus/corpus/obra__superpowers/skills/subagent-driven-development/SKILL.md
```

Set `ALEXANDRIA_SKILL_PATH` to an absolute path if the corpus lives elsewhere. In an editor, select
`.venv/bin/python` as the notebook kernel.

To reproduce the committed outputs non-interactively:

```bash
uv run jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=600 \
  notebooks/section_embedding_clusters.ipynb
```

The independently fitted plots are also written to:

- `notebooks/outputs/skill_section_embedding_clusters.png`
- `notebooks/outputs/babilong_qa1_42_section_embedding_clusters.png`
- `notebooks/outputs/section_cohesion_metrics.csv`

## Skill corpus section cohesion

`scripts/skill_corpus_section_cohesion.py` measures deepest-section cohesion, aggregates section medians through
skills to repositories, caches completed skills under `.cache/`, and writes section, skill, repo, statistical-test,
and graph outputs. The committed exploratory run uses 10 repositories with complete cached coverage:

```bash
uv run python scripts/skill_corpus_section_cohesion.py \
  --repo ClawBio__ClawBio \
  --repo CloudAI-X__claude-workflow-v2 \
  --repo Eronred__aso-skills \
  --repo K-Dense-AI__scientific-agent-skills \
  --repo NVIDIA__SkillSpector \
  --repo NVIDIA__skills \
  --repo Owl-Listener__designer-skills \
  --repo Prat011__awesome-llm-skills \
  --repo aipoch__medical-research-skills \
  --repo alirezarezvani__claude-skills
uv run jupyter lab notebooks/skill_corpus_section_cohesion.ipynb
```

The notebook reads and plots the generated files under
`notebooks/outputs/skill_corpus_section_cohesion/`. Delete the corresponding `.cache/` directory only when a
fresh embedding run is intentionally required.
