// scenes.js -> script.json: what the narrator says for each beat (voice text where it differs from the caption).
const scenes = require("./scenes.js");
const out = [];
scenes.forEach(sc => sc.beats.forEach(b => out.push({ id: b.id, text: b.voice || b.say })));
require("fs").writeFileSync(__dirname + "/build/script.json", JSON.stringify(out, null, 1));
console.log(out.length, "beats,", out.reduce((n, b) => n + b.text.split(" ").length, 0), "words");
