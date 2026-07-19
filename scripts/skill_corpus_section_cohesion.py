#!/usr/bin/env python3
"""Measure within-section semantic cohesion for every repository in a skill corpus.

The primary unit is a skill, not a sentence: section scores are summarized within each
SKILL.md, then skill medians are summarized within each repository. This prevents large
skills and repositories from dominating the comparison merely because they contain more text.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import numpy as np
from scipy.stats import kruskal, mannwhitneyu

from alexandria.ops.features.represent import RawSection, RawSentence, split
from alexandria.utils.embedders import default_embedder

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import TextIO

    from numpy.typing import NDArray

    from alexandria.ir.contracts import Embedder

METRIC_VERSION = 1
DEFAULT_CORPUS_ROOT = Path("../skill-corpus/corpus")
DEFAULT_OUTPUT_DIR = Path("notebooks/outputs/skill_corpus_section_cohesion")
DEFAULT_CACHE_DIR = Path(".cache/skill_corpus_section_cohesion")
SECTION_FIELDS = (
    "repo",
    "skill_path",
    "content_sha256",
    "section",
    "section_kind",
    "section_path",
    "sentence_count",
    "leave_one_out_centroid_cosine",
    "mean_pairwise_cosine",
)


@dataclass(frozen=True)
class ParsedSection:
    label: str
    kind: str
    path: str
    texts: tuple[str, ...]


@dataclass(frozen=True)
class ParsedSkill:
    repo: str
    skill_path: str
    content_sha256: str
    sections: tuple[ParsedSection, ...]

    @property
    def sentence_count(self) -> int:
        return sum(len(section.texts) for section in self.sections)


def discover_skill_paths(
    corpus_root: Path,
    *,
    repos: set[str] | None = None,
    max_skills_per_repo: int | None = None,
    seed: int = 42,
) -> list[Path]:
    """Return a deterministic corpus selection, optionally sampled within each repository."""
    paths = sorted(path for path in corpus_root.rglob("SKILL.md") if path.is_file())
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        repo = path.relative_to(corpus_root).parts[0]
        if repos is None or repo in repos:
            grouped[repo].append(path)

    selected: list[Path] = []
    for repo, repo_paths in sorted(grouped.items()):
        if max_skills_per_repo is not None and len(repo_paths) > max_skills_per_repo:
            repo_seed = int.from_bytes(hashlib.sha256(f"{seed}:{repo}".encode()).digest()[:8], "big")
            indexes = np.random.default_rng(repo_seed).choice(len(repo_paths), size=max_skills_per_repo, replace=False)
            repo_paths = [repo_paths[index] for index in sorted(indexes)]
        selected.extend(repo_paths)
    return selected


def parse_skill(path: Path, corpus_root: Path) -> ParsedSkill:
    """Parse one SKILL.md and attach optimizable sentences to their deepest section."""
    text = path.read_text(encoding="utf-8", errors="replace")
    relative = path.relative_to(corpus_root)
    parsed_sections: list[ParsedSection] = []

    def visit(section: RawSection, index_path: tuple[int, ...]) -> None:
        path_label = ".".join(str(part) for part in index_path)
        name = section.header.strip() or section.kind.value
        texts = tuple(
            child.text.strip()
            for child in section.children
            if isinstance(child, RawSentence) and child.optimizable and child.text.strip()
        )
        if texts:
            parsed_sections.append(
                ParsedSection(label=f"{name} [{path_label}]", kind=section.kind.value, path=path_label, texts=texts)
            )
        nested_index = 0
        for child in section.children:
            if isinstance(child, RawSection):
                nested_index += 1
                visit(child, (*index_path, nested_index))

    for root_index, section in enumerate(split(text), start=1):
        visit(section, (root_index,))
    return ParsedSkill(
        repo=relative.parts[0],
        skill_path=relative.as_posix(),
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
        sections=tuple(parsed_sections),
    )


def _cohesion(vectors: NDArray[np.float32]) -> tuple[float | None, float | None]:
    """Return leave-one-out centroid and pairwise cosine without forming an n-by-n matrix."""
    sentence_count = len(vectors)
    if sentence_count < 2:
        return None, None
    normalized = vectors.astype(float)
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    normalized = normalized / np.maximum(norms, np.finfo(float).eps)
    vector_sum = normalized.sum(axis=0)
    other_centroids = vector_sum - normalized
    other_norms = np.linalg.norm(other_centroids, axis=1, keepdims=True)
    other_centroids = other_centroids / np.maximum(other_norms, np.finfo(float).eps)
    leave_one_out = float(np.mean(np.sum(normalized * other_centroids, axis=1)))
    pairwise = float((np.dot(vector_sum, vector_sum) - sentence_count) / (sentence_count * (sentence_count - 1)))
    return leave_one_out, pairwise


def analyze_group(
    skills: Sequence[ParsedSkill], embedder: Embedder, *, api_batch_size: int
) -> dict[str, list[dict[str, object]]]:
    """Embed a bounded group of skills, deduplicating identical sentence text within the group."""
    unique_texts: list[str] = []
    text_indexes: dict[str, int] = {}
    section_indexes: dict[tuple[str, str], list[int]] = {}
    for skill in skills:
        for section in skill.sections:
            indexes: list[int] = []
            for text in section.texts:
                if text not in text_indexes:
                    text_indexes[text] = len(unique_texts)
                    unique_texts.append(text)
                indexes.append(text_indexes[text])
            section_indexes[(skill.skill_path, section.path)] = indexes

    embedded: list[NDArray[np.float32]] = []
    for start in range(0, len(unique_texts), api_batch_size):
        embedded.extend(embedder.embed(unique_texts[start : start + api_batch_size]))

    results: dict[str, list[dict[str, object]]] = {}
    for skill in skills:
        rows: list[dict[str, object]] = []
        for section in skill.sections:
            indexes = section_indexes[(skill.skill_path, section.path)]
            vectors = np.stack([embedded[index] for index in indexes])
            leave_one_out, pairwise = _cohesion(vectors)
            rows.append(
                {
                    "repo": skill.repo,
                    "skill_path": skill.skill_path,
                    "content_sha256": skill.content_sha256,
                    "section": section.label,
                    "section_kind": section.kind,
                    "section_path": section.path,
                    "sentence_count": len(indexes),
                    "leave_one_out_centroid_cosine": leave_one_out,
                    "mean_pairwise_cosine": pairwise,
                }
            )
        results[skill.skill_path] = rows
    return results


def _cache_path(cache_dir: Path, skill_path: str) -> Path:
    key = hashlib.sha256(skill_path.encode()).hexdigest()[:24]
    return cache_dir / f"{key}.json"


def load_cached_rows(skill: ParsedSkill, cache_dir: Path, model_id: str) -> list[dict[str, object]] | None:
    path = _cache_path(cache_dir, skill.skill_path)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return None
    if (
        payload.get("metric_version") != METRIC_VERSION
        or payload.get("embedding_model") != model_id
        or payload.get("skill_path") != skill.skill_path
        or payload.get("content_sha256") != skill.content_sha256
    ):
        return None
    rows = payload.get("sections")
    return rows if isinstance(rows, list) else None


def save_cached_rows(skill: ParsedSkill, rows: list[dict[str, object]], cache_dir: Path, model_id: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "metric_version": METRIC_VERSION,
        "embedding_model": model_id,
        "skill_path": skill.skill_path,
        "content_sha256": skill.content_sha256,
        "sections": rows,
    }
    _cache_path(cache_dir, skill.skill_path).write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8"
    )


def collect_section_rows(
    paths: Sequence[Path],
    corpus_root: Path,
    embedder: Embedder,
    *,
    cache_dir: Path,
    api_batch_size: int,
    group_sentence_limit: int,
) -> list[dict[str, object]]:
    """Load cached results and analyze uncached skills in bounded, resumable groups."""
    all_rows: list[dict[str, object]] = []
    pending: list[ParsedSkill] = []
    pending_sentences = 0
    completed = 0

    def flush() -> None:
        nonlocal pending, pending_sentences, completed
        if not pending:
            return
        analyzed = analyze_group(pending, embedder, api_batch_size=api_batch_size)
        for skill in pending:
            rows = analyzed[skill.skill_path]
            save_cached_rows(skill, rows, cache_dir, embedder.model_id)
            all_rows.extend(rows)
        completed += len(pending)
        print(f"analyzed {completed}/{len(paths)} skills ({len(all_rows)} section rows)", flush=True)
        pending = []
        pending_sentences = 0

    for path in paths:
        skill = parse_skill(path, corpus_root)
        cached = load_cached_rows(skill, cache_dir, embedder.model_id)
        if cached is not None:
            all_rows.extend(cached)
            completed += 1
            continue
        if pending and pending_sentences + skill.sentence_count > group_sentence_limit:
            flush()
        pending.append(skill)
        pending_sentences += skill.sentence_count
    flush()
    print(f"collected {len(all_rows)} sections from {len(paths)} skills", flush=True)
    return all_rows


def summarize_skills(section_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in section_rows:
        grouped[(str(row["repo"]), str(row["skill_path"]))].append(row)

    summaries: list[dict[str, object]] = []
    for (repo, skill_path), rows in sorted(grouped.items()):
        valid = [row for row in rows if row["mean_pairwise_cosine"] is not None]
        if not valid:
            continue
        summaries.append(
            {
                "repo": repo,
                "skill_path": skill_path,
                "section_count": len(rows),
                "analyzed_section_count": len(valid),
                "sentence_count": sum(int(row["sentence_count"]) for row in rows),
                "median_section_pairwise_cosine": float(
                    np.median([float(row["mean_pairwise_cosine"]) for row in valid])
                ),
                "median_section_leave_one_out_cosine": float(
                    np.median([float(row["leave_one_out_centroid_cosine"]) for row in valid])
                ),
            }
        )
    return summaries


def _bootstrap_median(values: NDArray[np.float64], samples: int, seed: int) -> tuple[float, float]:
    if len(values) == 1 or samples == 0:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    medians = np.median(values[rng.integers(0, len(values), size=(samples, len(values)))], axis=1)
    low, high = np.quantile(medians, (0.025, 0.975))
    return float(low), float(high)


def _benjamini_hochberg(p_values: dict[str, float]) -> dict[str, float]:
    if not p_values:
        return {}
    ordered = sorted(p_values, key=p_values.get)
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 1.0
    for rank_from_end, repo in enumerate(reversed(ordered), start=1):
        rank = count - rank_from_end + 1
        running = min(running, p_values[repo] * count / rank)
        adjusted[repo] = min(1.0, running)
    return adjusted


def summarize_repos(
    skill_rows: Sequence[dict[str, object]],
    selected_paths: Sequence[Path],
    corpus_root: Path,
    *,
    bootstrap_samples: int,
    min_skills_for_test: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    values_by_repo: dict[str, list[float]] = defaultdict(list)
    skill_rows_by_repo: dict[str, list[dict[str, object]]] = defaultdict(list)
    selected_skill_counts = Counter(path.relative_to(corpus_root).parts[0] for path in selected_paths)
    for row in skill_rows:
        repo = str(row["repo"])
        values_by_repo[repo].append(float(row["median_section_pairwise_cosine"]))
        skill_rows_by_repo[repo].append(row)

    eligible = {repo: values for repo, values in values_by_repo.items() if len(values) >= min_skills_for_test}
    if len(eligible) >= 2:
        observation_count = sum(len(values) for values in eligible.values())
        group_count = len(eligible)
        eligible_values = [value for values in eligible.values() for value in values]
        if np.ptp(eligible_values) == 0:
            statistic, p_value, epsilon_squared = 0.0, 1.0, 0.0
        else:
            statistic, p_value = kruskal(*(eligible[repo] for repo in sorted(eligible)))
            epsilon_squared = max(0.0, float((statistic - group_count + 1) / (observation_count - group_count)))
    else:
        statistic = p_value = epsilon_squared = math.nan

    all_values = [value for values in values_by_repo.values() for value in values]
    repo_p_values: dict[str, float] = {}
    repo_effects: dict[str, float] = {}
    for repo, values in eligible.items():
        rest = [value for other_repo, other in values_by_repo.items() if other_repo != repo for value in other]
        if len(rest) < min_skills_for_test:
            continue
        test = mannwhitneyu(values, rest, alternative="two-sided")
        repo_p_values[repo] = float(test.pvalue)
        repo_effects[repo] = float(2 * test.statistic / (len(values) * len(rest)) - 1)
    adjusted = _benjamini_hochberg(repo_p_values)

    summaries: list[dict[str, object]] = []
    for repo in sorted(selected_skill_counts):
        values = np.asarray(values_by_repo.get(repo, []), dtype=float)
        repo_seed = int.from_bytes(hashlib.sha256(f"{seed}:{repo}".encode()).digest()[:8], "big")
        if len(values):
            median = float(np.median(values))
            ci_low, ci_high = _bootstrap_median(values, bootstrap_samples, repo_seed)
        else:
            median = ci_low = ci_high = math.nan
        source_rows = skill_rows_by_repo.get(repo, [])
        summaries.append(
            {
                "repo": repo,
                "selected_skill_count": selected_skill_counts[repo],
                "analyzed_skill_count": len(values),
                "analyzed_section_count": sum(int(row["analyzed_section_count"]) for row in source_rows),
                "median_skill_section_pairwise_cosine": median,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "median_skill_section_leave_one_out_cosine": (
                    float(np.median([float(row["median_section_leave_one_out_cosine"]) for row in source_rows]))
                    if source_rows
                    else math.nan
                ),
                "vs_rest_p_value": repo_p_values.get(repo, math.nan),
                "vs_rest_bh_q_value": adjusted.get(repo, math.nan),
                "vs_rest_cliffs_delta": repo_effects.get(repo, math.nan),
            }
        )

    global_test = {
        "test": "Kruskal-Wallis on per-skill median section pairwise cosine",
        "eligible_repo_count": len(eligible),
        "skill_observation_count": sum(len(values) for values in eligible.values()),
        "min_skills_per_repo": min_skills_for_test,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "epsilon_squared": float(epsilon_squared),
        "corpus_skill_median": float(np.median(all_values)) if all_values else math.nan,
        "repo_vs_rest_test": "two-sided Mann-Whitney U with Benjamini-Hochberg correction",
    }
    return summaries, global_test


def _write_csv(
    path: Path, rows: Sequence[dict[str, object]], fields: Sequence[str], *, compressed: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def write(output: TextIO) -> None:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    if compressed:
        with (
            path.open("wb") as raw_output,
            gzip.GzipFile(fileobj=raw_output, mode="wb", mtime=0) as compressed_output,
            io.TextIOWrapper(compressed_output, encoding="utf-8", newline="") as output,
        ):
            write(output)
    else:
        with path.open(mode="w", encoding="utf-8", newline="") as output:
            write(output)


def plot_repo_summary(
    skill_rows: Sequence[dict[str, object]],
    repo_rows: Sequence[dict[str, object]],
    global_test: dict[str, object],
    output_path: Path,
) -> None:
    values_by_repo: dict[str, list[float]] = defaultdict(list)
    for row in skill_rows:
        values_by_repo[str(row["repo"])].append(float(row["median_section_pairwise_cosine"]))
    summary_by_repo = {str(row["repo"]): row for row in repo_rows}
    repos = sorted(
        values_by_repo,
        key=lambda repo: float(summary_by_repo[repo]["median_skill_section_pairwise_cosine"]),
    )
    distributions = [values_by_repo[repo] for repo in repos]
    figure_height = max(8.0, 0.65 * len(repos) + 2.5)
    fig, axis = plt.subplots(figsize=(15, figure_height), constrained_layout=True)
    y_positions = np.arange(1, len(repos) + 1)
    boxplot = axis.boxplot(
        distributions,
        positions=y_positions,
        orientation="horizontal",
        widths=0.55,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"color": "0.45"},
        capprops={"color": "0.45"},
    )

    for index, (repo, values) in enumerate(zip(repos, distributions, strict=True), start=1):
        row = summary_by_repo[repo]
        q_value = float(row["vs_rest_bh_q_value"])
        effect = float(row["vs_rest_cliffs_delta"])
        color = ("#D55E00" if effect > 0 else "#0072B2") if math.isfinite(q_value) and q_value < 0.05 else "#777777"
        boxplot["boxes"][index - 1].set_facecolor(color)
        boxplot["boxes"][index - 1].set_alpha(0.25)
        jitter_seed = int.from_bytes(hashlib.sha256(repo.encode()).digest()[:8], "big")
        jitter = np.random.default_rng(jitter_seed).uniform(-0.18, 0.18, size=len(values))
        axis.scatter(values, index + jitter, color=color, s=14, alpha=0.42, linewidths=0, zorder=2)

    labels = [f"{repo}  (n={len(values_by_repo[repo])})" for repo in repos]
    axis.set_yticks(y_positions, labels, fontsize=8)
    axis.axvline(float(global_test["corpus_skill_median"]), color="black", linestyle="--", linewidth=1)
    all_values = np.asarray([value for distribution in distributions for value in distribution])
    lower, upper = np.quantile(all_values, (0.005, 0.995))
    padding = max(0.01, 0.08 * (upper - lower))
    axis.set_xlim(max(-1.0, lower - padding), min(1.0, upper + padding))
    clipped_count = int(np.sum((all_values < lower - padding) | (all_values > upper + padding)))
    axis.set_xlabel("Per-skill median within-section pairwise cosine")
    axis.set_ylabel("Repository (n = analyzed skills)")
    p_value = float(global_test["p_value"])
    p_text = f"{p_value:.2e}" if math.isfinite(p_value) else "n/a"
    axis.set_title(
        "Distribution of section cohesion across skill repositories\n"
        f"Each point = one skill; box = interquartile range; Kruskal–Wallis p={p_text}, "
        f"epsilon²={float(global_test['epsilon_squared']):.3f}"
    )
    axis.grid(axis="x", alpha=0.2)
    axis.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#D55E00", label="Higher than rest, q < .05"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#0072B2", label="Lower than rest, q < .05"),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#777777",
                label="No adjusted evidence / n < minimum",
            ),
            Line2D([0], [0], color="black", linestyle="--", label="Corpus skill median"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=8,
    )
    fig.text(
        0.01,
        0.002,
        f"Repo-vs-rest colors use Mann–Whitney U with BH correction; {clipped_count} extreme point(s) outside "
        "the robust x-axis range are clipped; "
        "language, domain, duplication, and repository size remain potential confounders.",
        fontsize=8,
        color="0.35",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_analysis(
    corpus_root: Path,
    output_dir: Path,
    cache_dir: Path,
    embedder: Embedder,
    *,
    repos: set[str] | None = None,
    max_skills_per_repo: int | None = None,
    api_batch_size: int = 512,
    group_sentence_limit: int = 4096,
    bootstrap_samples: int = 2000,
    min_skills_for_test: int = 3,
    seed: int = 42,
) -> dict[str, object]:
    corpus_root = corpus_root.resolve()
    paths = discover_skill_paths(corpus_root, repos=repos, max_skills_per_repo=max_skills_per_repo, seed=seed)
    if not paths:
        raise ValueError(f"no SKILL.md files found under {corpus_root}")
    repo_count = len({path.relative_to(corpus_root).parts[0] for path in paths})
    print(f"selected {len(paths)} skills across {repo_count} repos")
    section_rows = collect_section_rows(
        paths,
        corpus_root,
        embedder,
        cache_dir=cache_dir,
        api_batch_size=api_batch_size,
        group_sentence_limit=group_sentence_limit,
    )
    skill_rows = summarize_skills(section_rows)
    repo_rows, global_test = summarize_repos(
        skill_rows,
        paths,
        corpus_root,
        bootstrap_samples=bootstrap_samples,
        min_skills_for_test=min_skills_for_test,
        seed=seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "sections.csv.gz", section_rows, SECTION_FIELDS, compressed=True)
    _write_csv(output_dir / "skills.csv", skill_rows, tuple(skill_rows[0]) if skill_rows else ())
    _write_csv(output_dir / "repos.csv", repo_rows, tuple(repo_rows[0]) if repo_rows else ())
    (output_dir / "global_test.json").write_text(
        json.dumps(global_test, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8"
    )
    plot_repo_summary(skill_rows, repo_rows, global_test, output_dir / "repos.png")
    print(f"wrote analysis to {output_dir}")
    return global_test


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--repo", action="append", dest="repos", help="analyze only this repo directory; repeatable")
    parser.add_argument("--max-skills-per-repo", type=int, help="deterministic sample size for exploratory runs")
    parser.add_argument("--api-batch-size", type=int, default=512)
    parser.add_argument("--group-sentence-limit", type=int, default=4096)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--min-skills-for-test", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    for name in ("api_batch_size", "group_sentence_limit", "min_skills_for_test"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples must be non-negative")
    if args.max_skills_per_repo is not None and args.max_skills_per_repo < 1:
        raise ValueError("--max-skills-per-repo must be positive")
    run_analysis(
        args.corpus_root,
        args.output_dir,
        args.cache_dir,
        default_embedder(),
        repos=set(args.repos) if args.repos else None,
        max_skills_per_repo=args.max_skills_per_repo,
        api_batch_size=args.api_batch_size,
        group_sentence_limit=args.group_sentence_limit,
        bootstrap_samples=args.bootstrap_samples,
        min_skills_for_test=args.min_skills_for_test,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
