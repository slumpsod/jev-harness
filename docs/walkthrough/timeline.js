// durations.json -> timeline.json (each beat's start, total length, frame count) and chapters.txt (MP4 chapters).
// The spacing constants must match stage.html (GAP, INTRO, OUTRO), which lays out the same timeline for the frames.
const fs = require("fs");
const scenes = require("./scenes.js");
const dur = JSON.parse(fs.readFileSync(__dirname + "/build/durations.json"));
const GAP = 0.5, INTRO = 0.9, OUTRO = 0.8, FPS = 30;
let t = 0; const beats = [], chapters = [];
for (const sc of scenes) {
  const start = t; t += INTRO;
  for (const b of sc.beats) { beats.push({ id: b.id, start: t }); t += dur[b.id] + GAP; }
  t += OUTRO;
  chapters.push([start, t, sc.title ? `${sc.num} ${sc.title}` : sc.key === "intro" ? "Intro" : "Outro"]);
}
const frames = Math.ceil(t * FPS);
fs.writeFileSync(__dirname + "/build/timeline.json", JSON.stringify({ total: t, fps: FPS, frames, beats }, null, 1));
let meta = ";FFMETADATA1\ntitle=Jev Harness: how it works\n";
for (const [a, b, name] of chapters) meta += `[CHAPTER]\nTIMEBASE=1/1000\nSTART=${Math.round(a * 1000)}\nEND=${Math.round(b * 1000)}\ntitle=${name}\n`;
fs.writeFileSync(__dirname + "/build/chapters.txt", meta);
for (const [a, , name] of chapters) console.log(`${Math.floor(a / 60)}:${String(Math.floor(a % 60)).padStart(2, "0")}  ${name}`);
console.log(`total ${t.toFixed(1)} s, ${frames} frames`);
