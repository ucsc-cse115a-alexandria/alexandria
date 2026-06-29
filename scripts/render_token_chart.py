#!/usr/bin/env python3
"""Render the per-repo SKILL.md token statistics as a self-contained HTML chart.

Joins the token counts from count_skill_tokens.py with the repository metadata
from search_skill_repos.py (for star counts) and embeds the merged data so the
page opens directly from the filesystem (no server required).
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
  :root {
    color-scheme: light;
    --bg: #f7f8fa;
    --card: #ffffff;
    --border: #e7e9ee;
    --ink: #1b2430;
    --muted: #6b7585;
    --grid: rgba(27, 36, 48, 0.07);
    --blue: #3b82f6;
    --violet: #8b5cf6;
    --teal: #14b8a6;
    --amber: #f59e0b;
    --shadow: 0 1px 2px rgba(16, 24, 40, 0.04), 0 8px 24px rgba(16, 24, 40, 0.06);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 56px 24px;
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    background: var(--bg); color: var(--ink); min-height: 100vh;
    -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: 1040px; margin: 0 auto; }
  h1 { font-size: 1.7rem; font-weight: 680; margin: 0 0 6px; letter-spacing: -0.02em; }
  .sub { color: var(--muted); margin: 0 0 36px; font-size: 0.95rem; }
  .stats {
    display: grid; gap: 14px; margin-bottom: 36px;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  }
  .stat {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 14px; padding: 18px 20px; box-shadow: var(--shadow);
  }
  .stat .label {
    color: var(--muted); font-size: 0.72rem; text-transform: uppercase;
    letter-spacing: 0.07em; font-weight: 600;
  }
  .stat .value {
    font-size: 1.55rem; font-weight: 680; margin-top: 8px;
    font-variant-numeric: tabular-nums; letter-spacing: -0.01em;
  }
  .grid2 {
    display: grid; gap: 24px; margin-bottom: 24px;
    grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
  }
  .card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 18px; padding: 24px 24px 20px; margin-bottom: 24px; box-shadow: var(--shadow);
  }
  .card h2 { font-size: 1.02rem; font-weight: 640; margin: 0 0 4px; letter-spacing: -0.01em; }
  .card .note { color: var(--muted); font-size: 0.84rem; margin: 0 0 16px; }
  canvas { width: 100% !important; }
</style>
</head>
<body>
<div class="wrap">
  <h1>SKILL.md Token Distribution</h1>
  <p class="sub">Per-repository SKILL.md token statistics &middot; tiktoken <span id="enc"></span></p>
  <div class="stats" id="stats"></div>

  <div class="grid2">
    <div class="card">
      <h2>Stars vs. average tokens</h2>
      <p class="note">
        All <span id="nAll"></span> repos &middot; log&ndash;log &middot; Pearson r = <span id="rAll"></span>
      </p>
      <canvas id="scatterAll"></canvas>
    </div>
    <div class="card">
      <h2>Stars vs. average tokens (within 95th percentile)</h2>
      <p class="note"><span id="nP95"></span> repos &middot; linear &middot; Pearson r = <span id="rP95"></span></p>
      <canvas id="scatterP95"></canvas>
    </div>
  </div>

  <div class="grid2">
    <div class="card">
      <h2>Distribution of average tokens</h2>
      <p class="note">All repositories</p>
      <canvas id="hist"></canvas>
    </div>
    <div class="card">
      <h2>Distribution within the 95th percentile</h2>
      <p class="note">Outliers above the 95th percentile removed</p>
      <canvas id="hist95"></canvas>
    </div>
  </div>

  <div class="card">
    <h2>Average tokens by repository</h2>
    <p class="note">Sorted high to low</p>
    <canvas id="byRepo"></canvas>
  </div>
</div>
<script>
  const DATA = __DATA__;
  const ENCODING = "__ENCODING__";

  const fmt = (n) => Math.round(n).toLocaleString();
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const grid = css("--grid");
  Chart.defaults.color = css("--muted");
  Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
  Chart.defaults.font.size = 12;

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
      datasets: [{
        label: "Repositories", data: hist.counts,
        backgroundColor: color, borderRadius: 6, maxBarThickness: 48,
      }],
    },
    options: {
      animation: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: "avg tokens" }, grid: { display: false } },
        y: { beginAtZero: true, ticks: { precision: 0 }, grid: { color: grid }, border: { display: false } },
      },
    },
  });

  makeHistogram("hist", DATA.histogram, css("--blue"));
  makeHistogram("hist95", DATA.histogram_p95, css("--violet"));

  const point = (r) => ({ x: r.stars, y: r.avg_tokens, name: r.full_name });
  const scatterTooltip = {
    callbacks: {
      label: (ctx) => `${ctx.raw.name}: ${fmt(ctx.raw.x)}★, ${fmt(ctx.raw.y)} tok`,
    },
  };

  const makeScatter = (canvasId, points, color, axisType) => new Chart(document.getElementById(canvasId), {
    type: "scatter",
    data: {
      datasets: [{
        data: points,
        backgroundColor: color + "cc",
        borderColor: color,
        borderWidth: 1,
        radius: 4,
        hoverRadius: 6,
      }],
    },
    options: {
      animation: false,
      plugins: { legend: { display: false }, tooltip: scatterTooltip },
      scales: {
        x: {
          type: axisType, title: { display: true, text: "stars" },
          grid: { color: grid }, border: { display: false },
        },
        y: {
          type: axisType, title: { display: true, text: "avg tokens" },
          grid: { color: grid }, border: { display: false },
        },
      },
    },
  });

  const allPoints = DATA.repos.map(point);
  const p95Points = DATA.repos
    .filter((r) => r.stars <= DATA.stars_p95 && r.avg_tokens <= DATA.tokens_p95)
    .map(point);

  document.getElementById("nAll").textContent = fmt(allPoints.length);
  document.getElementById("nP95").textContent = fmt(p95Points.length);
  document.getElementById("rAll").textContent = DATA.corr_all.toFixed(2);
  document.getElementById("rP95").textContent = DATA.corr_p95.toFixed(2);

  makeScatter("scatterAll", allPoints, css("--teal"), "logarithmic");
  makeScatter("scatterP95", p95Points, css("--amber"), "linear");

  const repos = [...DATA.repos].sort((a, b) => b.avg_tokens - a.avg_tokens);
  new Chart(document.getElementById("byRepo"), {
    type: "bar",
    data: {
      labels: repos.map((r) => r.full_name),
      datasets: [{
        label: "avg tokens", data: repos.map((r) => r.avg_tokens),
        backgroundColor: css("--blue"), borderRadius: 4,
      }],
    },
    options: {
      animation: false,
      indexAxis: "y",
      plugins: { legend: { display: false } },
      scales: {
        x: { beginAtZero: true, grid: { color: grid }, border: { display: false } },
        y: { ticks: { autoSkip: false, font: { size: 10 } }, grid: { display: false }, border: { display: false } },
      },
    },
  });
</script>
</body>
</html>
"""


class RepoPoint(BaseModel):
    """Per-repo SKILL.md token statistics joined with its GitHub star count."""

    full_name: str
    skill_md_count: int
    total_tokens: int
    avg_tokens: float
    stars: int


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

    repos: list[RepoPoint]
    stats: Stats
    histogram: Histogram
    histogram_p95: Histogram
    stars_p95: float
    tokens_p95: float
    corr_all: float
    corr_p95: float


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


def load_repos(tokens_path: Path, repos_path: Path) -> list[RepoPoint]:
    """Join token counts with repository star counts by `full_name`."""
    stars_by_name = {item["full_name"]: item["stars"] for item in json.loads(repos_path.read_text(encoding="utf-8"))}
    return [
        RepoPoint.model_validate({**item, "stars": stars_by_name[item["full_name"]]})
        for item in json.loads(tokens_path.read_text(encoding="utf-8"))
    ]


def correlation(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation, returning 0.0 when fewer than two points are given."""
    if len(xs) < 2:
        return 0.0
    return statistics.correlation(xs, ys)


def build_payload(repos: list[RepoPoint], bins: int) -> ChartPayload:
    """Assemble the chart payload: repos, summary stats, histograms, and correlations."""
    tokens = [repo.avg_tokens for repo in repos]
    stars = [float(repo.stars) for repo in repos]
    tokens_p95 = percentile(tokens, 95)
    stars_p95 = percentile(stars, 95)
    stats = Stats(
        count=len(tokens),
        mean=statistics.mean(tokens),
        median=statistics.median(tokens),
        min=min(tokens),
        max=max(tokens),
        p95=tokens_p95,
    )
    within_p95 = [repo for repo in repos if repo.stars <= stars_p95 and repo.avg_tokens <= tokens_p95]
    return ChartPayload(
        repos=repos,
        stats=stats,
        histogram=histogram(tokens, bins),
        histogram_p95=histogram([value for value in tokens if value <= tokens_p95], bins),
        stars_p95=stars_p95,
        tokens_p95=tokens_p95,
        corr_all=correlation(stars, tokens),
        corr_p95=correlation(
            [float(repo.stars) for repo in within_p95],
            [repo.avg_tokens for repo in within_p95],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/skill_token_counts.json"), help="token counts JSON")
    parser.add_argument("--repos", type=Path, default=Path("data/skill_repos.json"), help="repository metadata JSON")
    parser.add_argument("--output", type=Path, default=Path("data/skill_token_chart.html"), help="output HTML file")
    parser.add_argument("--bins", type=int, default=20, help="histogram bin count")
    parser.add_argument("--encoding", default="cl100k_base", help="tiktoken encoding label shown in the page")
    args = parser.parse_args()

    repos = load_repos(args.input, args.repos)
    payload = build_payload(repos, args.bins)

    html = TEMPLATE.replace("__DATA__", payload.model_dump_json()).replace("__ENCODING__", args.encoding)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Rendered chart for {len(repos)} repos -> {args.output}")


if __name__ == "__main__":
    main()
