#!/usr/bin/env bash
# Downloads what the video needs but git doesn't keep: IBM Plex fonts (OFL) and the Kokoro voice model (Apache-2.0).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p assets/fonts build
tmp=$(mktemp -d)
get_plex() {  # $1 = release tag, $2 = zip name, then the font files to keep
  local tag=$1 zip=$2; shift 2
  curl -sSfL -o "$tmp/$zip" "https://github.com/IBM/plex/releases/download/$tag/$zip"
  unzip -q -o "$tmp/$zip" -d "$tmp/x"
  for f in "$@"; do cp "$(find "$tmp/x" -path "*complete/woff2/$f" | head -1)" assets/fonts/; done
}
get_plex "%40ibm%2Fplex-sans%401.1.0" ibm-plex-sans.zip IBMPlexSans-{Regular,Medium,SemiBold,Bold}.woff2
get_plex "%40ibm%2Fplex-mono%401.1.0" ibm-plex-mono.zip IBMPlexMono-{Regular,Medium}.woff2
get_plex "%40ibm%2Fplex-sans-condensed%401.1.0" ibm-plex-sans-condensed.zip IBMPlexSansCondensed-{SemiBold,Bold}.woff2
for f in kokoro-v1.0.onnx voices-v1.0.bin; do
  [ -s "assets/$f" ] || curl -sSfL -o "assets/$f" "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/$f"
done
rm -rf "$tmp"
echo "assets ready: $(ls assets/fonts | wc -l) fonts, voice model $(du -h assets/kokoro-v1.0.onnx | cut -f1)"
