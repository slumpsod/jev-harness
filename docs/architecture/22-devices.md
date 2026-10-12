# 22 · Devices and sign-in

Other computers the worker can use (files, a shell, the screen, the browser and desktop tools), connected to the
dashboard, and the accounts that sign in to it from anywhere on the tailnet. User guide: [../devices.md](../devices.md).

Files: `jevh/accounts.py` (accounts, sign-in sessions, devices, pairing codes; `jevh users`), `jevh/devices/hub.py` (the
connections, a worker's call), `jevh/devices/agent.py` (the device's side; `jevh device`), `jevh/devices/ops.py` (what a
device does), `jevh/devices/wire.py` (the messages, files inside them), `jevh/tools/devices.py` (the worker's tool),
`jevh/app.py` (`jevh app`: window, tray, agent), `jevh/web/server.py` (the guard, `/login`, `/auth/*`, `/api/devices*`,
the WebSocket), `jevh/web/static/login.html`, the Devices panel in `index.html`. Tests: `tests/test_devices.py`. Live:
`evals/devices_cases.py` (Jev's tool pick), `evals/bench/devices.py` (full turns).

## A call, end to end

```mermaid
sequenceDiagram
  participant W as worker REPL<br/>tools/devices.py
  participant H as Harness<br/>orchestrator._device_call
  participant D as DeviceHub<br/>(dashboard event loop)
  participant A as Agent<br/>(the device)
  participant O as Ops<br/>(the device)
  participant J as Jev adapter<br/>(the hub)
  W->>H: _host("device", "laptop", "browser.do", args, kwargs)
  H->>D: call(ref, op, args, kwargs, timeout, cancel=turn Stop, on_host)
  D->>A: {t: call, id, op, args, kwargs}
  A->>O: run(op) on a thread (browser/desktop/sh: one at a time)
  O->>A: the tool asks Jev (JEVH_JEV_URL = the agent's 127.0.0.1 proxy)
  A->>D: {t: jev, rid, body}
  D->>J: POST /v1/systemone
  J-->>D: answers
  D-->>A: {t: jev_reply}
  O->>A: a password? ask_user (desktop tool, via __main__._call)
  A->>D: {t: host, id, name: ask_user}
  D->>H: on_host("ask_user", "[on laptop] ...") -> the dashboard asks you
  H-->>D: "Allow"
  D-->>A: {t: host_reply}
  O-->>A: result (a screenshot path becomes the file itself)
  A-->>D: {t: result, id, ok, value}
  D-->>H: value
  H-->>W: value, files written to from-devices/laptop/
```

## What decides what

| Decision | Who | Where |
|---|---|---|
| Does this task's prompt get the devices guide | Jev (`pick_tools`, the `tool` threshold), told which devices are connected | `Harness._pick_tools` |
| Does the worker have the tool at all | code: only while a device is online (else no import, no guide, no pointer line) | `Harness.run_task` |
| Which device, which operation | the worker (it names the device) | `tools/devices.py` |
| Each click and field on the device | Jev, as on the hub (the browser and desktop tools run there unchanged) | `ops.Ops.tool` |
| A shell command needs a yes | code: `ops.RISKY` (delete, kill, format, shutdown, force-push...) | `ops.Ops.sh` |
| Delete / overwrite | code: always undoable (trash/, backups/ on the device) | `ops.Ops._keep` |
| What a device may ask the hub | code: `look` and `ask_user` only (`hub.HOST_NAMES`); Jev on `/v1/systemone` and `/health` only | `hub.DeviceHub._answer_host/_answer_jev` |
| Who gets in from another device | code: a signed-in session (accounts exist) or the launch key (none yet) | `server.guard` |

## Design facts

- **The device connects out.** Nothing listens on a device but a proxy on 127.0.0.1, so it needs no firewall rule and no
  address. The hub is already behind `tailscale serve` for the phone, and the WebSocket goes through the same front
  (`wss://<hub>:10000/api/devices/ws`).
- **The key is the WebSocket's sign-in.** It is a bearer header, which no web page can set on a WebSocket, so a page
  elsewhere cannot open a device connection with someone's cookie. A wrong or revoked key is accepted and then closed
  with code 4401. The agent stops trying on 4401, and retries any other close with backoff (2 s doubling to 60 s).
- **Jev stays on the hub.** The browser and desktop tools build their Jev client from `JEVH_JEV_URL`. On a device that
  is the agent's proxy, which forwards over the connection. So a device needs no Jev and no GPU, and its decisions use
  the same backend as the hub's.
- **The device's own tools, unchanged.** `ops.Ops.tool` imports `jevh.tools.browser` / `desktop` on the device and calls
  the function asked for (public functions only). A result's path to a file the tool made (a screenshot) is sent as
  the file, and the harness writes it into the task's folder.
- **Stop reaches the device.** `DeviceHub.call` waits with the turn's cancel event. On Stop it sends `cancel`, and a shell
  command there is killed. A browser or desktop step in progress finishes its current action. The call returns
  `Cancelled` at once.
- **One app per device process.** The agent sets `JEVH_JEV_URL` for its whole process, so it must not run inside a hub
  process: its workers would inherit it. The bench runs its device as its own `jevh device run` process for this
  reason.
- **The device folder is not `~/.jevh`.** Once `~/.jevh` exists the harness reads personas and skills from it and writes
  learned skills there. A device creating it would change how a hub on the same machine behaves. Device state is in
  `~/.jevh-device` (`JEVH_DEVICE_HOME`).
- **Sign-in replaces the key link.** With no accounts, nothing changes: the per-launch `?key=` link and its cookie. Once
  an account exists, a remote host needs a `jevh_session` cookie (HttpOnly, Secure, SameSite=Lax, 30 days). Lax rather
  than Strict, so that opening the hub from a link in another app keeps you signed in. The API token and the Origin check
  still guard every POST.
- **The phone tab bar scrolls.** Eleven tabs need about 400 px. Shrunk to fit 360 px, the last one slid under the
  Appearance button (caught by `tests/test_dashboard_ui.py`). The tabs now keep 34 px and scroll sideways on their own.
