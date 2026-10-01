// Builds the Jev Harness Atlas page from docs/architecture/*.md + components.js.
// usage: node build-atlas.js <docs dir> <out.html> [--local-mermaid]
// Code links point at the commit checked out now (git rev-parse --short HEAD).
const fs = require("fs"), path = require("path");
const { Marked } = require("marked");
const { groups, nodes, edges } = require("./components.js");

const [docs, out] = process.argv.slice(2);
const localMermaid = process.argv.includes("--local-mermaid");
const COMMIT = require("child_process").execSync("git rev-parse --short HEAD", { cwd: __dirname }).toString().trim();
const GH = `https://github.com/slumpsod/jev-harness/blob/${COMMIT}/jevh/`;

const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const slug = (s) => s.toLowerCase().replace(/<[^>]+>/g, "").replace(/[^\p{L}\p{N} _-]/gu, "").replace(/ /g, "-");
const pageId = (file) => "p-" + (file === "README.md" ? "overview" : file.replace(/\.md$/, ""));

const files = fs.readdirSync(docs).filter((f) => f.endsWith(".md")).sort((a, b) =>
  a === "README.md" ? -1 : b === "README.md" ? 1 : a.localeCompare(b));

function refLink(text) {
  const m = /^([\w/]+\.py):(\d+)(?:-(\d+))?$/.exec(text);
  if (!m) return null;
  const anchor = m[3] ? `#L${m[2]}-L${m[3]}` : `#L${m[2]}`;
  return `<a class="ref" href="${GH}${m[1]}${anchor}" target="_blank" rel="noopener">${esc(text)}</a>`;
}

const sections = [];
for (const file of files) {
  const id = pageId(file);
  const md = fs.readFileSync(path.join(docs, file), "utf8");
  const marked = new Marked();
  let title = "";
  marked.use({
    renderer: {
      code({ text, lang }) {
        if (lang === "mermaid") return `<div class="diagram"><pre class="mermaid">${esc(text)}</pre></div>\n`;
        return `<pre class="code"><code>${esc(text)}</code></pre>\n`;
      },
      heading({ tokens, depth, text }) {
        const inner = this.parser.parseInline(tokens);
        if (depth === 1) { title = inner.replace(/<[^>]+>/g, ""); return ""; }
        return `<h${depth + 1} id="${id}--${slug(text.replace(/`/g, ""))}">${inner}</h${depth + 1}>\n`;
      },
      link({ href, tokens }) {
        const inner = this.parser.parseInline(tokens);
        let m;
        if ((m = /^([\w-]+)\.md(?:#(.+))?$/.exec(href))) {
          const target = pageId(m[1] + ".md");
          return `<a href="#${m[2] ? target + "--" + m[2] : target}">${inner}</a>`;
        }
        if (href.startsWith("#")) return `<a href="#${id}--${href.slice(1)}">${inner}</a>`;
        return `<a href="${esc(href)}" target="_blank" rel="noopener">${inner}</a>`;
      },
      codespan({ text }) {
        // marked passes codespan text already HTML-escaped
        const raw = text.replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&#39;/g, "'");
        return refLink(raw) || `<code>${text}</code>`;
      },
    },
  });
  let body = marked.parse(md);
  body = body.replace(/<table>/g, '<div class="table-wrap"><table>').replace(/<\/table>/g, "</table></div>");
  const short = file === "README.md" ? "System map" : title.replace(/^\d+\s*·\s*/, "");
  const num = file === "README.md" ? "00" : file.slice(0, 2);
  sections.push({ id, title: file === "README.md" ? "System map and where state lives" : short, short, num, body, file });
}

const nav = sections.map((s) =>
  `<a href="#${s.id}" data-sec="${s.id}"><span class="n">${s.num}</span><span>${esc(s.short)}</span></a>`).join("\n");
const content = sections.map((s) => `
<section class="doc" id="${s.id}">
  <header class="doc-head"><span class="n">${s.num}</span><h2>${esc(s.title)}</h2>
  <a class="src" href="https://github.com/slumpsod/jev-harness/blob/main/docs/architecture/${s.file}" target="_blank" rel="noopener">docs/architecture/${s.file}</a></header>
  ${s.body}
</section>`).join("\n");

const data = JSON.stringify({ groups, nodes, edges, gh: GH, sections: sections.map((s) => ({ id: s.id, short: s.short, num: s.num })) });
const tpl = fs.readFileSync(path.join(__dirname, "atlas-template.html"), "utf8");
let html = tpl.replace("%%NAV%%", nav).replace("%%CONTENT%%", content).replace("%%DATA%%", data.replace(/</g, "\\u003c"))
  .replace(/%%COMMIT%%/g, COMMIT).replace("%%COUNTS%%", `${nodes.length} components · ${edges.length} connections · ${(content.match(/class="mermaid"/g) || []).length} diagrams · ${sections.length} pages`);
if (localMermaid) {
  html += `<script src="file://${path.join(__dirname, "node_modules/mermaid/dist/mermaid.min.js")}"></script>
<script>mermaid.initialize({startOnLoad:false, theme: matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'default'}); mermaid.run();</script>`;
}
fs.writeFileSync(out, html);
console.log("wrote", out, (html.length / 1024).toFixed(0) + " KB");
