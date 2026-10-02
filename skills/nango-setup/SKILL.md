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
