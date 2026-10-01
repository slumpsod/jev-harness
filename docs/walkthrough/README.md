# Walkthrough video

A narrated, captioned walkthrough of how the harness works (about 16 minutes, 1080p, 19 chapters), made
entirely from files in this folder: the script and animations in `scenes.js`, rendered frame by frame
in Chromium, voiced offline with [Kokoro](https://github.com/thewh1teagle/kokoro-onnx).

```bash
cd docs/walkthrough
./fetch-assets.sh                       # once: IBM Plex fonts + the Kokoro voice model (~340 MB, kept out of git)
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
npm install                             # Chromium: npx playwright install chromium, or set CHROMIUM_PATH
PYTHON=.venv/bin/python ./make.sh stills   # one still per beat in build/stills/ - check layouts first
PYTHON=.venv/bin/python ./make.sh          # build/jev-harness-walkthrough.mp4 (+ -small.mp4, under 30 MB)
```

A full build takes about 10 minutes on 4 cores, plus about 5 minutes the first time for the voice.

## Changing it

`scenes.js` holds every chapter. A scene has `nodes` (boxes at fixed positions on a 1920×1080 canvas,
coloured by `kind`: `jev`, `model`, `repl`, `disk`, `fail`, `ui`, `h`) and `beats`. A beat is one
narrated line:

```js
{ say: "The caption, as shown.", voice: "Optional: how the narrator should say it (e.g. 'repple').",
  acts: [{ show: ["id"] }, { flow: ["from", "to", "packet label"], at: 0.4, c: "jev" }, { link: ["a", "b", "label"] }] }
```

`acts` run at `at` (0-1, a share of the line's length): `show`/`hide`, `hl`/`unhl` (highlight), `dim`/`undim`,
`link` (an arrow that draws itself), `flow` (a labelled packet travelling between two boxes). Timing
comes from the narration, so after editing a line delete its `build/audio/<beat id>.wav` to re-voice it.

| File | Does |
|---|---|
| `stage.html` | the renderer: lays out the timeline from the clip lengths and draws any moment with `renderAt(t)` |
| `export-script.js` → `tts.py` | narration text → one WAV per beat, and their lengths |
| `timeline.js` | each beat's start, the frame count, and the MP4 chapter markers |
| `audio.py` | the clips placed on the timeline as one narration track |
| `render.js` | frames `[a, b)` → an MP4 part (an unchanged frame is reused, not re-captured) |
| `stills.js` | one still at the end of every beat |
| `make.sh` | all of the above, three render workers in parallel, then ffmpeg |
