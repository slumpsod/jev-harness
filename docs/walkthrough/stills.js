const { chromium } = require('playwright-core');
const fs = require('fs');
(async () => {
  const DUR = JSON.parse(fs.readFileSync(__dirname + '/build/durations.json'));
  const b = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH || undefined, args: ['--allow-file-access-from-files'] });
  const p = await b.newPage({ viewport: { width: 1920, height: 1080 } });
  const errs = []; p.on('pageerror', e => errs.push(e.message)); p.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
  await p.addInitScript(d => { window.DUR = d; }, DUR);
  await p.goto('file://' + __dirname + '/stage.html');
  await p.evaluate(async () => { await Promise.all(['400 25px Plex','600 30px Plex','500 34px Plex','400 21px PlexMono','700 50px PlexCond'].map(f => document.fonts.load(f))); });
  const beats = await p.evaluate(() => window.BEATS);
  fs.mkdirSync(__dirname + '/build/stills', { recursive: true });
  const only = process.argv[2];
  for (const bt of beats) {
    if (only && !bt.id.startsWith(only)) continue;
    await p.evaluate(t => window.renderAt(t), bt.start + bt.dur - 0.05);
    await p.screenshot({ path: `${__dirname}/build/stills/${bt.id}.jpg`, type: 'jpeg', quality: 70 });
  }
  console.log('total', (await p.evaluate(() => window.TOTAL)).toFixed(1), 's', errs.slice(0, 5));
  await b.close();
})();
