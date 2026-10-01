#!/usr/bin/env bash
# The whole video: narration -> timeline -> frames (3 workers) -> MP4 with audio and chapters.
#   ./make.sh            full build into build/jev-harness-walkthrough.mp4 (+ a smaller -small.mp4 under 30 MB)
#   ./make.sh stills     one still per beat in build/stills/, to check layouts before the long render
# Needs: node, python3 (venv with requirements.txt), ffmpeg with libx264, Chromium for Playwright
# (npx playwright install chromium, or CHROMIUM_PATH=/path/to/chrome), and ./fetch-assets.sh run once.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
mkdir -p build/parts
node export-script.js
$PY tts.py
node timeline.js
if [ "${1:-}" = "stills" ]; then node stills.js; exit 0; fi
$PY audio.py
FRAMES=$(node -e 'console.log(require("./build/timeline.json").frames)')
A=$((FRAMES / 3)); B=$((2 * FRAMES / 3))
node render.js 0 $A build/parts/p0.mp4 & node render.js $A $B build/parts/p1.mp4 & node render.js $B $FRAMES build/parts/p2.mp4 & wait
printf "file 'parts/p0.mp4'\nfile 'parts/p1.mp4'\nfile 'parts/p2.mp4'\n" > build/concat.txt
ffmpeg -loglevel error -y -f concat -safe 0 -i build/concat.txt -i build/narration.wav -i build/chapters.txt \
  -map 0:v -map 1:a -map_metadata 2 -map_chapters 2 -c:v copy -c:a aac -b:a 160k -shortest -movflags +faststart \
  build/jev-harness-walkthrough.mp4
ffmpeg -loglevel error -y -i build/jev-harness-walkthrough.mp4 -map 0 -map_chapters 0 -c:v libx264 -preset slow -crf 28 \
  -tune animation -pix_fmt yuv420p -c:a aac -b:a 96k -movflags +faststart build/jev-harness-walkthrough-small.mp4
ls -la build/*.mp4
