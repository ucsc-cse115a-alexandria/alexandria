#!/usr/bin/env python3
"""Render the per-repo SKILL.md token statistics as a self-contained HTML chart.

Reads the JSON produced by count_skill_tokens.py and embeds the data so the page
opens directly from the filesystem (no server required).
"""

import argparse
import json
import statistics
from pathlib import Path

from pydantic import BaseModel

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SKILL.md Token Distribution</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 48px 24px;
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    background: radial-gradient(circle at 20% 0%, #1b2540 0%, #0b0f1a 60%);
    color: #e7ecf5; min-height: 100vh;
  }
  .wrap { max-width: 980px; margin: 0 auto; }
  h1 { font-size: 1.6rem; font-weight: 650; margin: 0 0 4px; letter-spacing: -0.02em; }
  .sub { color: #8b96ad; margin: 0 0 32px; font-size: 0.95rem; }
  .stats {
    display: grid; gap: 14px; margin-bottom: 32px;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  }
  .stat {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; padding: 18px 20px;
  }
  .stat .label { color: #8b96ad; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; }
  .stat .value { font-size: 1.55rem; font-weight: 650; margin-top: 6px; font-variant-numeric: tabular-nums; }
  .card {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px; padding: 24px; margin-bottom: 28px;
  }
  .card h2 { font-size: 1.05rem; font-weight: 600; margin: 0 0 18px; }
  canvas { width: 100% !important; }
</style>
</head>
<body>
<div class="wrap">
  <h1>SKILL.md Token Distribution</h1>
  <p class="sub">Average tiktoken count per repository &middot; <span id="enc"></span></p>
  <div class="stats" id="stats"></div>
  <div class="card"><h2>Distribution of per-repo average tokens</h2><canvas id="hist"></canvas></div>
  <div class="card"><h2>Distribution within the 95th percentile</h2><canvas id="hist95"></canvas></div>
  <div class="card"><h2>Average tokens by repository</h2><canvas id="byRepo"></canvas></div>
</div>
<script>
  const DATA = __DATA__;
  const ENCODING = "__ENCODING__";

  const fmt = (n) => Math.round(n).toLocaleString();
  const grid = "rgba(255,255,255,0.07)";
  const tick = "#8b96ad";
  Chart.defaults.color = tick;
  Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;

  document.getElementById("enc").textContent = ENCODING;

  const stats = DATA.stats;
  const statRows = [
    ["Repositories", fmt(stats.count)],
    ["Mean", fmt(stats.mean)],
    ["Median", fmt(stats.median)],
    ["Min", fmt(stats.min)],
    ["95th pct", fmt(stats.p95)],
    ["Max", fmt(stats.max)],
  ];
  document.getElementById("stats").innerHTML = statRows
    .map(([label, value]) =>
      `<div class="stat"><div class="label">${label}</div><div class="value">${value}</div></div>`)
    .join("");

  const makeHistogram = (canvasId, hist, color) => new Chart(document.getElementById(canvasId), {
    type: "bar",
    data: {
      labels: hist.labels,
      datasets: [{ label: "Repositories", data: hist.counts, backgroundColor: color, borderRadius: 6 }],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: "avg tokens" }, grid: { color: grid } },
        y: { beginAtZero: true, ticks: { precision: 0 }, grid: { color: grid } },
      },
    },
  });

  makeHistogram("hist", DATA.histogram, "#6ea8ff");
  makeHistogram("hist95", DATA.histogram_p95, "#b69dff");

  const repos = [...DATA.repos].sort((a, b) => b.avg_tokens - a.avg_tokens);
  new Chart(document.getElementById("byRepo"), {
    type: "bar",
    data: {
      labels: repos.map((r) => r.full_name),
      datasets: [{
        label: "avg tokens", data: repos.map((r) => r.avg_tokens),
        backgroundColor: "#7ee0c0", borderRadius: 4,
      }],
    },
    options: {
      indexAxis: "y",
      plugins: { legend: { display: false } },
      scales: {
        x: { beginAtZero: true, grid: { color: grid } },
        y: { ticks: { autoSkip: false, font: { size: 10 } }, grid: { display: false } },
      },
    },
  });
</script>
</body>
</html>
"""


class RepoTokens(BaseModel):
    """Per-repo SKILL.md token statistics produced by count_skill_tokens.py."""

    full_name: str
    skill_md_count: int
    total_tokens: int
    avg_tokens: float


class Histogram(BaseModel):
    """Equal-width histogram bins: bucket start labels and per-bucket counts."""

    labels: list[str]
    counts: list[int]


class Stats(BaseModel):
    """Summary statistics of per-repo average token counts."""

    count: int
    mean: float
    median: float
    min: float
    max: float
    p95: float


class ChartPayload(BaseModel):
    """Everything the HTML page needs, embedded as a single JSON blob."""

    repos: list[RepoTokens]
    stats: Stats
    histogram: Histogram
    histogram_p95: Histogram


def percentile(values: list[float], pct: float) -> float:
    """Return the `pct` percentile of `values` using linear interpolation."""
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def histogram(values: list[float], bins: int) -> Histogram:
    """Bin `values` into `bins` equal-width buckets, returning labels and counts."""
    low, high = min(values), max(values)
    width = (high - low) / bins or 1.0
    counts = [0] * bins
    for value in values:
        index = min(int((value - low) / width), bins - 1)
        counts[index] += 1
    labels = [f"{round(low + index * width)}" for index in range(bins)]
    return Histogram(labels=labels, counts=counts)


def build_payload(repos: list[RepoTokens], bins: int) -> ChartPayload:
    """Assemble the chart payload: raw repos, summary stats, and full + p95 histograms."""
    values = [repo.avg_tokens for repo in repos]
    p95 = percentile(values, 95)
    stats = Stats(
        count=len(values),
        mean=statistics.mean(values),
        median=statistics.median(values),
        min=min(values),
        max=max(values),
        p95=p95,
    )
    return ChartPayload(
        repos=repos,
        stats=stats,
        histogram=histogram(values, bins),
        histogram_p95=histogram([value for value in values if value <= p95], bins),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/skill_token_counts.json"), help="token counts JSON")
    parser.add_argument("--output", type=Path, default=Path("data/skill_token_chart.html"), help="output HTML file")
    parser.add_argument("--bins", type=int, default=20, help="histogram bin count")
    parser.add_argument("--encoding", default="cl100k_base", help="tiktoken encoding label shown in the page")
    args = parser.parse_args()

    repos = [RepoTokens.model_validate(item) for item in json.loads(args.input.read_text(encoding="utf-8"))]
    payload = build_payload(repos, args.bins)

    html = TEMPLATE.replace("__DATA__", payload.model_dump_json()).replace("__ENCODING__", args.encoding)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Rendered chart for {len(repos)} repos -> {args.output}")


if __name__ == "__main__":
    main()
