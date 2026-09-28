
<!-- 2026-09-28: could you search the web and find out how to make a change of address, i need it to get rid of my old gym membership. -->
- Skip `web.ask()` — it timed out after 120s twice and wiped all variables when the REPL restarted; use `web.search(n=5)` instead, which returned structured results reliably.
- Prefer `web.read(url)` to pull full article text from a specific source (e.g., the cancelmygym.app guide) instead of relying on search snippets that get truncated.
- Don't fire multiple `web.ask()` calls in one turn — they compound timeouts and lose state on REPL restart; run one query per turn or use `web.search()` for batched structured results.
