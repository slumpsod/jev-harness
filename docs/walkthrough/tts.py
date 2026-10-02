"""build/script.json -> build/audio/<beat>.wav (Kokoro, voice af_heart) and build/durations.json.
Clips that already exist are kept: delete one to re-voice that beat after editing its text."""
import json, os, soundfile as sf
from kokoro_onnx import Kokoro
B = "build/"
k = Kokoro("assets/kokoro-v1.0.onnx", "assets/voices-v1.0.bin")
os.makedirs(B + "audio", exist_ok=True)
dur = {}
for b in json.load(open(B + "script.json")):
    p = f"{B}audio/{b['id']}.wav"
    if not os.path.exists(p):
        s, sr = k.create(b["text"], voice=os.environ.get("VOICE", "af_heart"), speed=1.0, lang="en-us")
        sf.write(p, s, sr)
    info = sf.info(p)
    dur[b["id"]] = round(info.frames / info.samplerate, 3)
json.dump(dur, open(B + "durations.json", "w"), indent=1)
print(len(dur), "clips,", round(sum(dur.values()) / 60, 2), "min of speech")
