const { chromium } = require('playwright-core');
const fs = require('fs'), path = require('path');
const dir = process.argv[2];
(async () => {
  const blocks = [];
  for (const f of fs.readdirSync(dir).filter(f => f.endsWith('.md')).sort()) {
    const t = fs.readFileSync(path.join(dir, f), 'utf8');
    const re = /```mermaid\n([\s\S]*?)```/g; let m, i = 0;
    while ((m = re.exec(t))) blocks.push({ id: `${f}#${++i}`, code: m[1] });
  }
  const b = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH || undefined });
  const p = await b.newPage();
  await p.setContent('<html><body></body></html>');
  await p.addScriptTag({ path: path.join(__dirname, 'node_modules/mermaid/dist/mermaid.min.js') });
  const res = await p.evaluate(async (blocks) => {
    mermaid.initialize({ startOnLoad: false });
    const out = [];
    let n = 0;
    for (const bl of blocks) {
      try { await mermaid.parse(bl.code); const r = await mermaid.render('m' + (n++), bl.code); out.push([bl.id, 'ok', r.svg.length]); }
      catch (e) { out.push([bl.id, 'ERR', String(e.message || e).slice(0, 300)]); }
    }
    return out;
  }, blocks);
  for (const r of res) console.log(r.join('  '));
  await b.close();
})();
