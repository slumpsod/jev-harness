"""Build evals/bench/serve_compare_chart.html (one self-contained page) from the rows serve_compare.py saved.

    python evals/bench/serve_compare_chart.py          # reads runs/serve_compare/rows.jsonl, writes the html beside this file

Offline: no server, no model. Re-run it after more rows land (the control section appears once control_nospec rows exist).
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROWS = HERE / "runs" / "serve_compare" / "rows.jsonl"
OUT = HERE / "serve_compare_chart.html"

TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen3.8 27B serving test</title>
<style>
:root{--surface:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#7a7873;--grid:#e6e5e1;--axis:#c9c8c3;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--bad:#d03b3b;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--surface:#1a1a19;--card:#232321;--ink:#fff;--ink2:#c3c2b7;--muted:#8d8c84;--grid:#33332f;--axis:#4a4a45;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--bad:#ec6a6a;color-scheme:dark}}
:root[data-theme="dark"]{--surface:#1a1a19;--card:#232321;--ink:#fff;--ink2:#c3c2b7;--muted:#8d8c84;--grid:#33332f;--axis:#4a4a45;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--bad:#ec6a6a;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1040px;margin:0 auto;padding:28px 16px 56px}
h1{font-size:24px;margin:0 0 4px;letter-spacing:-.01em}
.sub{color:var(--ink2);margin:0 0 22px;max-width:70ch}
h2{font-size:16px;margin:0 0 2px}
.note{color:var(--ink2);font-size:13px;margin:0 0 12px;max-width:75ch}
.card{background:var(--card);border:1px solid var(--grid);border-radius:12px;padding:18px 18px 14px;margin:0 0 18px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:0 0 18px}
.tile{background:var(--card);border:1px solid var(--grid);border-radius:12px;padding:14px 16px;border-top:4px solid var(--c)}
.tile .nm{font-weight:650;display:flex;gap:8px;align-items:center}
.tile .lb{color:var(--ink2);font-size:13px;margin-bottom:8px}
.tile .big{font-size:30px;font-weight:700;letter-spacing:-.02em;line-height:1.1}
.tile .big small{font-size:14px;font-weight:500;color:var(--ink2)}
.tile dl{margin:10px 0 0;display:grid;grid-template-columns:1fr auto;gap:2px 10px;font-size:13px;color:var(--ink2)}
.tile dd{margin:0;color:var(--ink);font-variant-numeric:tabular-nums;text-align:right}
.sw{width:11px;height:11px;border-radius:3px;background:var(--c);display:inline-block}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:13px;color:var(--ink2);margin:0 0 10px}
.legend span{display:inline-flex;gap:6px;align-items:center}
svg{display:block;width:100%;height:auto;overflow:visible}
svg text{fill:var(--ink2);font:12px system-ui,sans-serif}
svg .val{fill:var(--ink);font-weight:600;font-variant-numeric:tabular-nums}
svg .tick{fill:var(--muted)}
svg .bad{fill:var(--bad);font-weight:700}
.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px}
#tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--surface);padding:6px 9px;border-radius:7px;font-size:12px;line-height:1.35;opacity:0;transition:opacity .08s;z-index:9;max-width:260px}
button{font:inherit;font-size:13px;background:var(--card);color:var(--ink);border:1px solid var(--axis);border-radius:8px;padding:5px 11px;cursor:pointer}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums;margin-top:10px}
th,td{padding:5px 8px;border-bottom:1px solid var(--grid);text-align:right}th:first-child,td:first-child{text-align:left}
th{color:var(--ink2);font-weight:600}
.hidden{display:none}
.foot{color:var(--muted);font-size:12px;margin-top:6px}
@media (max-width:560px){main{padding-top:18px}.tile .big{font-size:26px}}
</style></head>
<body><main>
<h1>Qwen3.8 27B on one RTX 3090: which setup gets the job done?</h1>
<p class="sub" id="sub"></p>
<div class="tiles" id="tiles"></div>

<section class="card"><h2>Time per task</h2>
<p class="note">Minutes of wall-clock time for a full harness turn, thinking, tool calls and checks included. Bars are the mean of each task's repeats (coding tasks ran 3 times, ARC puzzles once); dots are the individual runs. A red ✕ is a wrong answer.</p>
<div class="legend" id="leg1"></div><div id="c1"></div></section>

<section class="card"><h2>Why a faster decoder is not always faster</h2>
<p class="note">Same runs, three views. The fast server reads tokens out fastest, but it writes more of them per task than the recommended one, so the two end up level on time. The precision server's average is pulled up by one ARC puzzle that ran into the 20-minute limit.</p>
<div class="panels"><div id="p1"></div><div id="p2"></div><div id="p3"></div></div></section>

<section class="card hidden" id="ctl"><h2>Control: does speculation change the answers?</h2>
<p class="note" id="ctlnote"></p><div id="c4"></div></section>

<section class="card"><div style="display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap">
<h2 style="margin:0">Numbers behind the charts</h2><button id="tb">Show table</button></div>
<div id="tbl" class="hidden"></div>
<p class="foot">Every server ran with 80,000 tokens of context, two slots, the same harness code, Jev, personas and sampling; only the GPU was the 3090. Each run's hidden tests (coding) or exact-match grid (ARC) decide right or wrong.</p></section>
</main><div id="tip"></div>
<script>
const ROWS = __ROWS__;
const CTL = __CTL__;
const META = __META__;
const SERVERS = META.servers;   // [{id,label,short,color}]
const col = id => `var(--${SERVERS.find(s=>s.id===id).slot})`;
const $ = id => document.getElementById(id);
const fmt = (x,d=1)=>Number(x).toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d});
const mean = a => a.reduce((x,y)=>x+y,0)/(a.length||1);
const tip = $('tip');
function showTip(e,html){tip.innerHTML=html;tip.style.opacity=1;const w=tip.offsetWidth;tip.style.left=Math.min(innerWidth-w-8,e.clientX+12)+'px';tip.style.top=(e.clientY+14)+'px'}
function hideTip(){tip.style.opacity=0}
function el(tag,attrs,parent,text){const n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const k in attrs)n.setAttribute(k,attrs[k]);if(text!==undefined)n.textContent=text;parent&&parent.appendChild(n);return n}
function hover(n,html){n.addEventListener('mousemove',e=>showTip(e,html));n.addEventListener('mouseleave',hideTip);n.addEventListener('focus',e=>{const r=n.getBoundingClientRect();showTip({clientX:r.left,clientY:r.top},html)});n.addEventListener('blur',hideTip);n.setAttribute('tabindex','0')}

const main = SERVERS.filter(s=>!s.control);
const byCfg = id => ROWS.filter(r=>r.config===id);
const stats = id => { const rs=byCfg(id), code=rs.filter(r=>r.kind==='code'), arc=rs.filter(r=>r.kind==='arc');
  const dec=rs.filter(r=>r.server&&r.server.decode_tps); const dt=dec.reduce((s,r)=>s+r.server.decoded/r.server.decode_tps,0);
  return {n:rs.length, solved:rs.filter(r=>r.correct).length, code:code.filter(r=>r.correct).length, nc:code.length, arc:arc.filter(r=>r.correct).length, na:arc.length,
   min:mean(rs.map(r=>r.seconds))/60, out:mean(rs.map(r=>r.output)), dec: dt? dec.reduce((s,r)=>s+r.server.decoded,0)/dt : 0, to:rs.filter(r=>r.timed_out).length}; };

$('sub').textContent = `${META.nruns} full harness runs: ${META.ntasks} real tasks (6 coding tasks with hidden tests, repeated 3 times, and 6 ARC-AGI puzzles) on each of three ways to serve the model. Fastest, recommended for the harness, and best precision.`;

// tiles
$('tiles').innerHTML = main.map(s=>{const t=stats(s.id);return `<div class="tile" style="--c:var(--${s.slot})"><div class="nm"><span class="sw"></span>${s.name}</div><div class="lb">${s.label}</div>
<div class="big">${t.solved}<small> / ${t.n} solved</small></div>
<dl><dt>Coding</dt><dd>${t.code} / ${t.nc}</dd><dt>ARC puzzles</dt><dd>${t.arc} / ${t.na}</dd><dt>Time per task</dt><dd>${fmt(t.min)} min</dd>
<dt>Output per task</dt><dd>${fmt(t.out,0)} tokens</dd><dt>Server decode</dt><dd>${fmt(t.dec,0)} tok/s</dd><dt>Timeouts</dt><dd>${t.to}</dd></dl></div>`}).join('');
$('leg1').innerHTML = main.map(s=>`<span><i class="sw" style="--c:var(--${s.slot})"></i>${s.name}: ${s.label}</span>`).join('');

// chart 1: grouped horizontal bars, one group per task
(function(){
  const tasks=[]; const seen=new Set();
  ROWS.forEach(r=>{if(r.config==='control_nospec')return;if(!seen.has(r.task)){seen.add(r.task);tasks.push({id:r.task,kind:r.kind})}});
  const order=['ledger','ttlcache','mdtable','freeslot','bugfix','todocli'];
  tasks.sort((a,b)=>(a.kind===b.kind?(a.kind==='code'?order.indexOf(a.id)-order.indexOf(b.id):a.id.localeCompare(b.id)):(a.kind==='code'?-1:1)));
  const W=980, left=118, right=70, top=8, bw=11, gap=2, grp=bw*main.length+gap*(main.length-1), pad=16;
  const rowH=grp+pad; let y=top; const H=top+tasks.length*rowH+34; const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':'Minutes per task for each server'});
  const max=Math.ceil(Math.max(...ROWS.filter(r=>r.config!=='control_nospec').map(r=>r.seconds/60))/5)*5||5; const x=v=>left+(W-left-right)*v/max;
  for(let v=0;v<=max;v+=5){el('line',{x1:x(v),x2:x(v),y1:top-2,y2:H-26,stroke:'var(--grid)','stroke-width':1},svg);el('text',{x:x(v),y:H-8,'text-anchor':'middle',class:'tick'},svg,v+(v===max?' min':''))}
  let lastKind=null;
  tasks.forEach((t,i)=>{
    const gy=top+i*rowH;
    if(t.kind!==lastKind&&lastKind!==null) el('line',{x1:8,x2:W-right,y1:gy-pad/2,y2:gy-pad/2,stroke:'var(--axis)','stroke-dasharray':'3 4'},svg);
    lastKind=t.kind;
    el('text',{x:left-10,y:gy+grp/2+4,'text-anchor':'end'},svg,(t.kind==='arc'?'ARC ':'')+t.id);
    main.forEach((s,k)=>{
      const rs=ROWS.filter(r=>r.task===t.id&&r.config===s.id); if(!rs.length)return;
      const by=gy+k*(bw+gap), m=mean(rs.map(r=>r.seconds/60));
      const bar=el('rect',{x:left,y:by,width:Math.max(2,x(m)-left),height:bw,rx:3,fill:col(s.id)},svg);
      const wrong=rs.filter(r=>!r.correct).length;
      hover(bar,`<b>${s.name}</b> · ${t.kind==='arc'?'ARC ':''}${t.id}<br>${fmt(m)} min on average (${rs.length} run${rs.length>1?'s':''})<br>${rs.map(r=>fmt(r.seconds/60)+' min'+(r.correct?'':' ✕')+(r.timed_out?' (timed out)':'')+', '+fmt(r.output,0)+' tokens').join('<br>')}`);
      rs.forEach(r=>{if(rs.length>1)el('circle',{cx:x(r.seconds/60),cy:by+bw/2,r:2.6,fill:'var(--card)',stroke:col(s.id),'stroke-width':1.5},svg)});
      el('text',{x:x(Math.max(m,...rs.map(r=>r.seconds/60)))+8,y:by+bw-1.5,class:'val'},svg,fmt(m));
      if(wrong) el('text',{x:x(Math.max(m,...rs.map(r=>r.seconds/60)))+44,y:by+bw-1,class:'bad'},svg,'✕'+(rs.length>1?' '+wrong+'/'+rs.length:'')+(rs.some(r=>r.timed_out)?' timed out':''));
    });
  });
  $('c1').appendChild(svg);
})();

// panels: three single-metric bar charts
function panel(host,title,unit,get,d){
  const W=300,H=40+main.length*34, left=0, svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':title});
  el('text',{x:0,y:12,style:'fill:var(--ink);font-weight:600'},svg,title);
  el('text',{x:0,y:28,class:'tick'},svg,unit);
  const vals=main.map(s=>get(s.id)); const max=Math.max(...vals)*1.18;
  main.forEach((s,i)=>{const y=40+i*34,w=(W-70)*vals[i]/max;
    const b=el('rect',{x:0,y,width:w,height:20,rx:4,fill:col(s.id)},svg);
    hover(b,`<b>${s.name}</b><br>${title}: ${fmt(vals[i],d)}`);
    el('text',{x:w+6,y:y+15,class:'val'},svg,fmt(vals[i],d));
    el('text',{x:6,y:y+15,style:'fill:#fff;font-weight:600;paint-order:stroke;stroke:rgba(0,0,0,.35);stroke-width:2.5px'},svg,s.name.split(' ')[0]);});
  host.appendChild(svg);
}
panel($('p1'),'Server decode speed','tokens per second, higher is faster',id=>stats(id).dec,0);
panel($('p2'),'Output per task','tokens written, lower is leaner',id=>stats(id).out,0);
panel($('p3'),'Time per task','minutes, lower is faster',id=>stats(id).min,1);

// control
(function(){
  const c=CTL; const ids=SERVERS.filter(s=>c.some(r=>r.config===s.id)); const hasCtl=c.some(r=>r.config==='control_nospec');
  if(!hasCtl)return; $('ctl').classList.remove('hidden');
  $('ctlnote').textContent='The bugfix task (seven hidden tests, one of them tricky) repeated on every server, plus Q5_K_M with speculation switched off. If speculation changed answers, its pass rate would differ from the plain model\'s. Small samples: a one-run gap is noise.';
  const W=600,H=24+ids.length*38,left=170,svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':'bugfix pass rate'});
  ids.forEach((s,i)=>{const rs=c.filter(r=>r.config===s.id),ok=rs.filter(r=>r.correct).length,y=8+i*38,w=(W-left-90)*ok/rs.length,full=W-left-90;
    el('text',{x:left-10,y:y+15,'text-anchor':'end'},svg,s.name);
    el('rect',{x:left,y,width:full,height:20,rx:4,fill:'var(--grid)'},svg);
    const b=el('rect',{x:left,y,width:Math.max(w,0),height:20,rx:4,fill:col(s.id)},svg);
    hover(b,`<b>${s.name}</b><br>${ok} of ${rs.length} runs passed all 7 hidden tests`);
    el('text',{x:left+full+8,y:y+15,class:'val'},svg,`${ok} / ${rs.length}`)});
  $('c4').appendChild(svg);
})();

// table
(function(){
  const rows=SERVERS.filter(s=>byCfg(s.id).length).map(s=>{const t=stats(s.id);return `<tr><td>${s.name} · ${s.label}</td><td>${t.n}</td><td>${t.solved}</td><td>${t.code}/${t.nc}</td><td>${t.arc}/${t.na}</td><td>${fmt(t.min)}</td><td>${fmt(t.out,0)}</td><td>${fmt(t.dec,0)}</td><td>${t.to}</td></tr>`}).join('');
  $('tbl').innerHTML=`<table><thead><tr><th>Server</th><th>Runs</th><th>Solved</th><th>Coding</th><th>ARC</th><th>Min/task</th><th>Tokens/task</th><th>Decode tok/s</th><th>Timeouts</th></tr></thead><tbody>${rows}</tbody></table>`;
  $('tb').onclick=()=>{const h=$('tbl').classList.toggle('hidden');$('tb').textContent=h?'Show table':'Hide table'};
})();
</script></body></html>
"""


def main() -> None:
    rows = [json.loads(line) for line in ROWS.read_text(encoding="utf-8").splitlines()]
    keys = ("task", "kind", "config", "rep", "seconds", "output", "thinking", "passed", "total", "correct", "timed_out",
            "steps", "server")
    slim = lambda rs: [{k: r.get(k) for k in keys} for r in rs]   # noqa: E731
    # the main comparison is the first 3 repeats of every coding task (the bugfix control adds more bugfix repeats and
    # a no-speculation server; those only feed the control chart)
    main_rows = [r for r in rows if r["config"] != "control_nospec" and r["rep"] <= 3]
    ctl_rows = [r for r in rows if r["task"] == "bugfix" and r["kind"] == "code"]
    servers = [
        {"id": "fast", "name": "Fastest", "label": "Q4_K_XL + DFlash2 draft, q8 KV", "slot": "s1"},
        {"id": "recommended", "name": "Recommended", "label": "Q5_K_M + MTP, q8 KV", "slot": "s2"},
        {"id": "precision", "name": "Precision", "label": "Q6_K + MTP, q4 KV", "slot": "s3"},
        {"id": "control_nospec", "name": "Q5_K_M plain", "label": "Q5_K_M, no speculation", "slot": "s4", "control": True},
    ]
    meta = {"servers": servers, "nruns": len(main_rows), "ntasks": len({r["task"] for r in main_rows})}
    html = (TEMPLATE.replace("__ROWS__", json.dumps(slim(main_rows))).replace("__CTL__", json.dumps(slim(ctl_rows)))
            .replace("__META__", json.dumps(meta)))
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
