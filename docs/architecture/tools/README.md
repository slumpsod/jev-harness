# Architecture tools

Keep the diagrams in `docs/architecture/` true to the code, and build the Atlas page from them.

```bash
cd docs/architecture/tools && npm install
# Chromium for Playwright: `npx playwright install chromium`, or point CHROMIUM_PATH at a Chrome/Chromium binary.
```

| Task | Command (from the repo root) |
|---|---|
| Check that every Mermaid diagram parses and renders | `node docs/architecture/tools/check-mermaid.js docs/architecture` |
| Re-anchor every `file:line` after the code moved (matches each old line by content) | `python docs/architecture/tools/remap.py <old-commit> docs/architecture/*.md docs/architecture/tools/components.js` |
| Build the Atlas page (code links point at the commit checked out) | `node docs/architecture/tools/build-atlas.js docs/architecture atlas.html` |
| …with Mermaid bundled, to preview it locally in a browser | add `--local-mermaid` |

`remap.py` prints the refs it couldn't place (the line was changed or deleted): fix those by hand.
After a refactor, also search the docs for the functions you renamed.

## Files

- `build-atlas.js` renders each `docs/architecture/*.md` page (Mermaid blocks become `<pre class="mermaid">`,
  `file:line` refs become GitHub links) into `atlas-template.html`.
- `components.js` is the component map at the top of the page: groups, components (with a `file:line`,
  a blurb, where to watch it, and the doc pages it appears in), and the edges between them with what
  each sends. Add a component here when the harness grows a new part.
- `check-mermaid.js` parses and renders every diagram in Chromium with Mermaid 11.
