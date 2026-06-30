#!/usr/bin/env bash
# Render an HTML file to a PNG via headless Chrome.
# Usage: scripts/render_html_to_png.sh <input.html> [output.png] [width]
set -euo pipefail

input="${1:?usage: render_html_to_png.sh <input.html> [output.png] [width]}"
output="${2:-${input%.html}.png}"
width="${3:-1400}"
height="${RENDER_HEIGHT:-2000}"

if [[ ! -f "$input" ]]; then
  echo "error: input file not found: $input" >&2
  exit 1
fi

find_chrome() {
  local candidates=(
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    "/Applications/Chromium.app/Contents/MacOS/Chromium"
  )
  for c in "${candidates[@]}"; do
    [[ -x "$c" ]] && { echo "$c"; return 0; }
  done
  for name in google-chrome chromium chromium-browser; do
    command -v "$name" >/dev/null 2>&1 && { command -v "$name"; return 0; }
  done
  return 1
}

chrome="$(find_chrome)" || { echo "error: Chrome/Chromium not found" >&2; exit 1; }

# Chrome needs absolute paths for file:// URLs and --screenshot.
abs() { [[ "$1" = /* ]] && echo "$1" || echo "$PWD/$1"; }

"$chrome" \
  --headless=new \
  --disable-gpu \
  --hide-scrollbars \
  --force-device-scale-factor=2 \
  --window-size="${width},${height}" \
  --screenshot="$(abs "$output")" \
  "file://$(abs "$input")"

echo "wrote $output"
