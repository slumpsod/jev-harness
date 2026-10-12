# Devices, sign-in and the desktop app

Your assistant runs on one computer, **the hub**: the one with Jev and the models. Your other computers sign in to it and
become **devices**. The assistant can then use them as it uses the hub: their files, a shell, their browser and their
apps. You chat from any of them, or from your phone.

```
  phone ──https──┐                                ┌── laptop (device agent)
                 ├── tailscale serve ── hub ──────┼── studio-pc (device agent)
  laptop app ────┘   (your tailnet)   (dashboard, └── ...
                                       Jev, models)
```

Status: **partly verified**. The offline tests cover sign-in, pairing, the connection, files, the shell, asking back,
pausing, cancelling and revoking. What was run live on the hub machine itself, and what was not, is in
[What has been checked](#what-has-been-checked).

## Set up the hub

1. Run the dashboard behind `tailscale serve`, as for the phone (README, "From your phone"):

   ```bash
   tailscale serve --bg --https=10000 http://127.0.0.1:8740
   jevh dashboard --no-browser --allow-host <this-pc>.<tailnet>.ts.net:10000
   ```

   Or use the desktop app: `jevh app`, choose **This computer is the hub**, and tick the Tailscale box (it runs the
   `tailscale serve` line for you). `jevh app install` adds a Start menu entry.

2. Make your account. In the dashboard on the hub computer, open **Devices**, then **Make an account**. Or run
   `jevh users add jed` in a terminal. From then on other devices **sign in** (the account lasts across restarts), and
   the per-launch `?key=` link stops working. The hub computer itself (127.0.0.1) never asks.

## Add a computer

On the other computer (Windows), install the harness with the device extras:

```bash
git clone <this repo> jev-harness && cd jev-harness
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[app,browser,desktop]"
python -m playwright install chromium     # for the browser on that computer
```

Then pair it in either of two ways:

- **The app.** Run `jevh app`, choose **Connect to my hub**, and give the hub's address
  (`https://<hub>.<tailnet>.ts.net:10000`). Sign in. In **Devices**, choose **Let the assistant use this computer**.
- **A terminal.** On the hub, **Devices** then **Add a device** shows a one-time code (valid for 10 minutes) and the
  command. On the computer, run:

  ```bash
  jevh device pair https://<hub>.<tailnet>.ts.net:10000 ABCD-EFGH --name laptop
  jevh device run
  ```

The device makes the connection itself, out to the hub. Nothing on it listens to the network. Its key is in
`~/.jevh-device/device.json`, readable by you only, and the hub keeps only a hash of it.

## What the assistant can do there

Ask normally: *"what's in the Downloads folder on my laptop?"*, *"copy report.pdf from bilbo to this PC"*,
*"on studio-pc, open Chrome and check my site loads"*. The worker's `devices` tool offers:

| | |
|---|---|
| files | list, find (by name words or text inside), read, write, move, copy, make folders, delete, open in its app; bring a file here (`fetch`), send one there; copy between two devices. Up to 48 MB a file |
| shell | a PowerShell (or cmd) command, with a time limit; Stop kills it |
| screen | a screenshot of the whole screen |
| browser | the browser tool, running on that computer (you see its window there) |
| apps | the desktop tool, running on that computer |

The worker only gets the tool while a device is online. Otherwise its prompt doesn't mention it.

## What stays guarded

You chose "full access, keep the confirmations":

- **Undoable by default.** Deleting moves the file to `~/.jevh-device/trash/` on that computer. Writing over a file keeps
  the old one in `~/.jevh-device/backups/`.
- **Destructive commands ask first.** A shell command that deletes, kills processes, force-pushes, formats or shuts down
  returns `needs_confirmation`. The worker has to ask you, then repeat it with your yes.
- **The browser and desktop tools keep their own rules** on every device: orders, payments, sends and deletes ask you
  first. A password is typed only after you allow it, and the question says which computer is asking. Card numbers are
  never typed.
- **Pausing a device.** The tray icon, or `jevh device pause`, refuses every call and stops a running command. The tray
  icon shows a notice when the assistant starts using that computer, and **Activity** lists everything it did there,
  also kept in `~/.jevh-device/activity.jsonl`.
- **Removing a device.** **Remove** in the Devices panel ends its key at once. `jevh device forget` deletes the key on
  the device.
- **What a device may ask the hub.** Only to show you a question, or to let the worker's model read a screenshot. A
  device cannot hand out tasks, read your memory or use your accounts.

These are guards against mistakes, not a sandbox: a shell can do anything, as on the hub. Treat a paired computer as
something that runs what your assistant decides.

## Sign-in details

- Accounts live in `workspace/accounts.db` on the hub: scrypt password hashes, and the hashes of sign-in and device
  tokens. Every account is an owner of the hub: there is one assistant, one memory and one set of devices.
- A sign-in lasts 30 days on that browser. Changing a password signs that account out everywhere. Repeated wrong
  passwords or pairing codes from one address are refused for a few minutes.
- `jevh users list | add NAME | password NAME | remove NAME` manages accounts from the hub's terminal.
- The hub is reachable only on your tailnet (`tailscale serve`, never `funnel`). Sign-in is the second lock.

## The desktop app

`jevh app` opens the dashboard in its own window, with a tray icon. Closing the window keeps the app in the tray. On a
device, the app also runs the device agent while it is open. The tray menu has **Open**, **Pause the assistant here**,
**Activity**, **Start with Windows** and **Quit**. `jevh app --hidden` starts in the tray (what Start with Windows
runs). `jevh app reset` asks the hub-or-device question again.

It is a Python app (pywebview with WebView2, and pystray), started from your checkout. There is no installer `.exe` yet.

## What has been checked

See the README's eval list and `evals/devices_cases_summary.txt` / `evals/bench/devices_last_summary.txt` for the numbers
of the last run. Not checked live yet:

- a real second computer over Tailscale (the live runs used a device on the hub machine itself, connected the same way);
- the desktop tool on a device (the bench avoids typing into real windows);
- the app's window and tray on a computer other than the hub.

## Limits

- Devices are Windows computers. Files and the shell would work elsewhere, but the desktop tool and the app's Start
  menu entry are Windows-only. The iPhone is a chat client only.
- The hub runs one turn at a time, so two devices can't each run a separate task at once. A task can use several devices
  in turn.
- Files move inside the connection: 48 MB per file, per call.
- A device's browser and desktop tools ask the hub's Jev (through the hub), so they are slower than on the hub by the
  network's round trip.
