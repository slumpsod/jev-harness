// node render.js <firstFrame> <endFrame> <out.mp4>   (frames [first, end) at FPS)
const { chromium } = require('playwright-core');
const { spawn } = require('child_process');
const fs = require('fs');
const FPS = 30;
(async () => {
  const [a, b, out] = [+process.argv[2], +process.argv[3], process.argv[4]];
  const DUR = JSON.parse(fs.readFileSync(__dirname + '/build/durations.json'));
  const br = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH || undefined, args: ['--allow-file-access-from-files'] });
  const p = await br.newPage({ viewport: { width: 1920, height: 1080 } });
  await p.addInitScript(d => { window.DUR = d; }, DUR);
  await p.goto('file://' + __dirname + '/stage.html');
  await p.evaluate(async () => { await Promise.all(['400 25px Plex','600 30px Plex','500 34px Plex','400 21px PlexMono','700 50px PlexCond','700 104px PlexCond','600 22px Plex'].map(f => document.fonts.load(f))); });
  const ff = spawn('ffmpeg', ['-loglevel', 'error', '-y', '-f', 'image2pipe', '-c:v', 'mjpeg', '-framerate', String(FPS), '-i', '-',
    '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', '-r', String(FPS), out], { stdio: ['pipe', 'inherit', 'inherit'] });
  let last = null, buf = null, shots = 0;
  const t0 = Date.now();
  for (let f = a; f < b; f++) {
    const key = await p.evaluate(t => window.renderAt(t), f / FPS);
    if (key !== last || !buf) { buf = await p.screenshot({ type: 'jpeg', quality: 92 }); last = key; shots++; }
    if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
    if ((f - a) % 900 === 0) console.log(out, f - a, '/', b - a, shots, 'shots', ((Date.now() - t0) / 1000).toFixed(0) + 's');
  }
  ff.stdin.end();
  await new Promise(r => ff.on('close', r));
  console.log(out, 'done', b - a, 'frames,', shots, 'screenshots,', ((Date.now() - t0) / 1000).toFixed(0) + 's');
  await br.close();
})();
