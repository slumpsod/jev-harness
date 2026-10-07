---
name: nango-setup
description: Walk the user through connecting their own accounts and apps (Gmail, Google Calendar, Drive, GitHub, Slack, Notion and ~1000 more) to the assistant with Nango - start Nango, register an app at the provider, enter its ID and secret, sign in, test it. Use when the user wants to connect, add, set up or reconnect an account or app, or when the nango tool says it isn't set up or an account isn't connected.
---
# Connecting an account (Nango), step by step with the user

How to actually connect, add or reconnect one of the user's own accounts or apps (Gmail, Google Calendar, Drive,
GitHub, Slack, Notion, Todoist and more) so the assistant can use it: start Nango, register an app at the provider,
give Nango its ID and secret, sign in, test. Not for explaining how OAuth or APIs work in general.

You are the guide. The user may never have done this: plain words, one step at a time, say what and why in one
sentence. Code does everything else. Never ask for a password or secret in chat, never type a password, never
print a secret. Do it all yourself: no request() to another assistant.

**Waiting for the user.** After each thing THEY must do: `ask_user(question, ["Done", "I'm stuck"])` (keeps your
place), or end your turn with ONE short message: what to do now and "reply Done". One step or question per
message. If they answer in a new message, FIRST run `print(nango.status())` and `g = nango.guide(app)`, say only
what that output shows (never "connected" or "set up" from memory), then do the next step; don't redo finished
ones. Never end a turn with "I'll now do X": do X in code in the same turn, then report it.

## 1. Where things stand
```python
s = nango.status(); print(s)        # set_up, docker, nango_running, connected, next
```
Do what `next` says. If it's all set up, skip to step 3.

## 2. Start Nango on this machine (first time, or after a restart)
Tell them: "I'll start Nango, the service that keeps your logins, using Docker. First time it downloads about
400 MB and takes a few minutes." Ask OK. Ask once: `ask_user("Nango has a dashboard on this PC. Keep it open to
this PC only (simplest), or ask for a login?", ["Open to this PC", "Ask for a login"])`. Then:
```python
print(nango.start(dashboard_login=False))   # True for "Ask for a login"; it can take minutes
```
- "Docker isn't installed": give them https://www.docker.com/products/docker-desktop/ , wait, retry.
- Docker's own leftover-files problem (Windows): start() asks permission and fixes it.
- Another failure: show the message, don't guess. (Nango Cloud: sign up at app.nango.dev, set NANGO_SECRET_KEY.)

## 3. Which app, and how much access
```python
print(nango.providers("calendar"))      # name, display_name, auth_mode; Gmail is "google-mail"
```
Confirm the app. Ask on its own: read only, or read and make changes? (`access="read"` / `"write"`; either way every
change asks them first.) If it is already connected (`status()["connected"]`), say so and go to step 7.

## 4. What the user does at the app
```python
g = nango.guide("google-mail"); print(g["auth_mode"], g["callback_url"]); print(g["steps"])
```
- `API_KEY` / `BASIC`: nothing to register. Skip to step 6 (the sign-in page asks for the key).
- `OAUTH2`: go through `g["steps"]` ONE at a time. Say its title and `do`; open its page for them with
  `nango.open_page(step["url"])` (their own browser, already signed in). The callback URL must be typed EXACTLY
  as `g["callback_url"]`. Wait for "Done" before the next. "Already done": skip it.
- No written steps: `web.read(g["setup_guide_url"], "the steps to register an OAuth app")`, then do the same.
Before the first step, offer once, on its own: "Want me to click through it myself? You sign in once in a window I
open; I never see your password." If yes, follow "Driving the website yourself" below. If no (or it fails), keep
guiding by hand.

## 5. Give Nango the app's ID and secret
`setup()` asks the user for both, itself: the ID, then the secret typed hidden. YOU never ask for them and never
say "paste them here": a secret in chat is kept in the transcript and memory. Say "I'll ask for the ID, then the
secret (hidden)", then call it right away (by hand: no browser):
```python
print(nango.setup("google-mail", access="read"))   # the questions come from this call
```
Pass `scopes="a,b"` for an app with no known scopes (read its docs, ask what they want to do). A "look swapped"
or "doesn't look like" error means a wrong paste: say which, run it again. It returns no secret; never print one.
A second Google app (Calendar after Gmail) reuses the project and its ID/secret: only "turn on the API" is new.

## 6. Sign in
```python
c = nango.connect_and_wait("google-mail"); print(c)   # opens the page in their browser, waits up to 5 min
```
Tell them first: "A page opens: sign in and allow access. Google may say the app is unverified. That's your own
app: Advanced, then Continue." `connected` True: step 7. False: ask what they saw (see Problems) and call it again.

## 7. Check it and finish
```python
print(nango.test("google-mail"))        # {ok, detail}, e.g. the email address it is signed in as
```
Tell them plainly what works now ("I can read your Gmail"), what they can ask next, and that anything that
changes something shows them a question first. Offer to connect another app.

## Driving the website yourself (only if the user said yes)
```python
print(nango.browser_mode())       # BEFORE the first browser.goto(): a visible window, stays signed in
browser.goto(step["url"])
```
- Needs sign-in? `ask_user("Please sign in in the browser window I opened, then reply Done", ["Done"])`. Never type
  credentials, 2-step codes or card details.
- Google says "This browser or app may not be secure" or blocks the sign-in: STOP. Don't retry or work around
  it. Say so, and carry on by hand with `nango.open_page(url)` in their own browser.
- One outcome per `browser.do("...", values={"App name": "Jev harness"})`, then check with `browser.read()`.
  Redirect URI: exactly `g["callback_url"]`. A billing, card or paid-plan page: stop and tell the user (none of
  this costs money).
- The ID and secret show once the client is created: `print(nango.setup("google-mail", access="read",
  page=browser.read()))` finds them itself (Google, GitHub), never prints them, and asks the user for any it
  can't see.

## Problems
| What the user sees or you get | Cause and fix |
|---|---|
| Google: `redirect_uri_mismatch` | the redirect URI in the Google client differs from `g["callback_url"]` by a character (http vs https, trailing slash, wrong port). Fix it there, wait a minute, connect again. |
| Google: "Access blocked: app has not completed verification" | the app is in Testing: add their email under Audience > Test users, or Publish app. |
| Google: connected, then 401 a week later | the app is still in Testing (7-day tokens): Publish app, then connect again. |
| `invalid_client` / `unauthorized_client` at sign-in | the Client ID or secret is wrong or from another app: run `nango.setup()` again. |
| API call: 403 "API has not been used in project" | step "turn on the API" was skipped for that app. Open its page, Enable, retry. |
| API call: 401 or 403, `ok` False in `status()` | the login expired or was revoked: `nango.connect_and_wait(app)` again. |
| `connect_and_wait` times out | the user didn't finish, or closed the page: ask, run it again (a new link each time). |
| A scope changed with `setup()` | existing accounts keep old scopes: `connect_and_wait` again. |
| `can't reach Nango` | Docker or Nango stopped: `nango.start()`. |
| `Docker Desktop didn't come up` | the user opens Docker Desktop, reads its error, tells you. |

## Lessons from past runs

<!-- 2026-10-02: Could you help me connect my Facebook account? -->
- Use `nango.guide()["steps"]` directly (it already lists titles, actions, and URLs) instead of relying on `web.read`, which returned truncated content in this run and left you without the full registration details.
- Before the first step, offer once to "click through it myself" using `nango.open_page(step["url"])` in the user's own already-signed-in browser, as the guide's "Driving the website yourself" section describes—this wasn't offered.
- Remember that for OAUTH2 you must walk through `g["steps"]` ONE AT A TIME: state the title and `do`, open the page, type the callback URL EXACTLY as `g["callback_url"]`, and wait for "Done" before moving on; don't dump all steps at once.
- The Facebook provider list includes both `facebook` (OAUTH2) and `facebook-system-user` (API_KEY)—confirm which the user actually needs before proceeding, since the setup flow differs.

<!-- 2026-10-02: I can use your browser to click let’s just try it -->
- Check `browser_mode()` and whether the user can actually click/type in the visible window before asking them to sign in; if they’re on mobile or blocked, stop and switch to a secure alternative instead of waiting for “Done”.
- Check whether the user has already pasted credentials into chat before continuing; if yes, stop, tell them to rotate that password, and do not type, store, or reuse those credentials.
- Check whether the user can create a Google Cloud project from the current device before opening `console.cloud.google.com/projectcreate`; if not, do not start that OAuth setup step.
- Check that provider setup pages are opened with `nango.open_page(step["url"])` and that the user has completed the step before moving to `nango.setup()`; if not, stay on that step.

<!-- 2026-10-05: could you help me connect my gmail. -->
- Before driving the Google Auth Platform form, check `nango.guide("google-mail")` for the exact `callback_url` and verify the signed-in Google account/project first.
- If `browser.do(...)` returns `needs_value` or `max_actions`, stop using high-level actions and call `browser.elements()`, then use direct commands like `browser.fill(i, text, enter=False)` and `browser.click(i)`.
- In the Google consent-screen branding form, the “User support email” field is a custom dropdown: `browser.select(i, "<their email>")` opens it and picks the entry (or `browser.do(..., values={"User support email": "<their email>"})`). (Updated 2026-10-07: `select` handles custom dropdowns now; `click` on a dropdown is refused.)
- If you must pause mid-Google-Cloud setup, save an artifact with the project, current URL, filled fields, and remaining steps; on restart, re-run `nango.status()` and `nango.guide("google-mail")` before continuing.

<!-- 2026-10-06: could you help me connect my gmail. -->
- Check `browser.elements("User support email")` before entering it; if it reports a dropdown, use `browser.select(index, exact_email)` instead of `browser.do("enter ...")`.
- If `browser.html` does not expose the dropdown options, ask the user for the signed-in email once and pass that exact string to `browser.select`; do not try clicking the avatar to infer it.
- After every Google Cloud “Next” click, verify the new step with `browser.read(...)` or `r["page"]` before issuing the next action, because the page can remain on the same screen if a required field is missing.
- If a final “finish”/“create” click returns `stuck`, inspect the visible control with `browser.read` or `browser.look` and click the specific button/checkbox (for example “Agree and continue”) rather than retrying a generic “finish”.
