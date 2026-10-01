"""build/audio/*.wav + build/timeline.json -> build/narration.wav: each beat's clip at the moment its beat starts."""
import json, numpy as np, soundfile as sf
B = "build/"
tl = json.load(open(B + "timeline.json"))
SR = 24000
out = np.zeros(int(np.ceil(tl["total"] * SR)) + SR, dtype=np.float32)
for b in tl["beats"]:
    s, sr = sf.read(f"{B}audio/{b['id']}.wav", dtype="float32")
    assert sr == SR, sr
    i = int(round(b["start"] * SR)); out[i:i + len(s)] += s
out *= 0.89 / np.abs(out).max()
sf.write(B + "narration.wav", out[:int(tl["total"] * SR)], SR)
print(f"narration.wav: {tl['total']:.1f} s")
