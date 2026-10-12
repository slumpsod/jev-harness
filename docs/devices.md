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

Status: **verified across two computers** (the hub and a second Windows 11 PC over Tailscale, 2026-10-12: 13/13 tasks).
The offline tests cover sign-in, pairing, the connection, files, the shell, asking back, pausing, cancelling and revoking.
The desktop tool on a device too (11/11 on the same PC). Not checked yet: the app's window on a computer other than the
hub. See
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

## Which computer is "this computer"

The one you are writing on. The hub sees where each message comes from: `tailscale serve` passes the tailnet address of
the computer the request comes from, and each device's connection comes from its own address. So a message typed in the
app on bilbo, or in any browser there, is from bilbo. One typed on the hub itself is from the hub.

- **From a device.** "This computer", "here", "my Downloads", "my home folder" and an app you mention all mean that
  device, unless you name another computer. The worker always gets the devices guide then, and its Environment section
  says that its own Python, and the project, desktop and browser tools, run on the hub, not where you are.
- **From the hub.** "This computer" is the hub; another computer only when you name it.
- **From anywhere else** (your phone, a browser on a computer that isn't paired): nothing is assumed, and the hub is the
  default as before.
- **Naming a computer always wins**: "on studio-pc, ..." from bilbo goes to studio-pc.

The chat shows "from bilbo" under each message while other computers are connected. A request queued while a task runs
keeps the computer of the turn it follows.

Measured (`evals/bench/devices.py --origin`, 2026-10-12, 3 repeats, a device on the hub machine connected like a remote
one): requests about "this computer" or "my home folder" written on the device, 0/9 without telling the harness where and
**9/9** told; held out 0/6 against **6/6**. Requests that must not reach the device ("this computer" written on the hub,
"on the hub, ..." written on the device, a sum) stayed 9/9 and 3/3 held out. Cost: a question from a device that needs no
computer at all carries the devices guide (median 4.8 s became 8.4 s on "what's 17% of 2,340?"). Then tried by hand on
bilbo itself (2026-10-12), in `jevh app` there: the messages showed "from Bilbo", "what's in my Downloads folder?" and
"what's this computer's name?" were answered from bilbo, and "on the hub, ..." from the hub.

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

Live, 2026-10-11, Qwen3.8-27B workers with the semif Jev, on the hub machine. The device was a separate
`jevh device run` process, paired and connected through the hub like any other.

| | without devices | with devices |
|---|---|---|
| device tasks: list, read, find, fetch and sum a CSV, write, a shell command, delete, the browser, a device that is not there (9 tasks x 3) | 1/27 | **27/27** |
| held-out device tasks (2 x 3) | 0/6 | **6/6** |
| "delete this folder" with the user answering No (x3) | 1/3 (twice it deleted the folder on the hub without asking) | 2/3 (the miss: the model server stalled mid-turn; nothing was deleted) |
| lookalikes that must not touch a device (4 x 3, one held out) | 12/12 | **12/12**, no device touched |
| lookalike cost (median) | 9.0 s, 1,897 input tokens | 8.4 s, 1,977 input tokens |

Median times with devices: 10-33 s a task (the shell task: 9.5 s, against the 8-minute limit 3/3 times without).
Results: `evals/bench/devices_last_summary.txt`, `devices_raw.json`, `devices_holdout_raw.json`.

Jev's choice to give the worker the tool (`evals/devices_cases.py`, 3 repeats): recall 45/45 on the cases its
description was written against, **9/18 on held-out cases**; no lookalike on the cases, 3/15 held out and 3/225 on
`tool_cases.py` got it needlessly. When Jev misses, the worker still has the tool, and its prompt names the connected
computers. The end-to-end tasks passed either way.

**Across two computers** (2026-10-12): the hub and bilbo, a second Windows 11 PC on the tailnet running `jevh device run`,
with devices, once each: **13/13**. That covers the nine device tasks, the gate (it asked, the user said No, the folder
stayed), and the three lookalikes with no device touched. Median 12 s a task; Chrome opened on bilbo for the browser task;
its Jev questions went through the hub. Raw: `evals/bench/devices_bilbo_raw.json`. The run without devices was stopped
after 2 tasks: on a real other computer it searched the hub until the 8-minute limit each time.

**The desktop tool on that computer** (2026-10-12, `--desktop`): JevLab, the desktop bench's test app, was built on bilbo
from source the hub sent, and each check read the app's own state file on bilbo. **11/11**: fill and submit a form, remove
one list item, send `/rc`, a modal settings dialog, a counter, the right one of two windows, count a list without changing
it, delete a list with the user's yes, sign in with the password refused (the question reached the hub as "[on bilbo] May
I type a password ...", No was kept, nothing typed) and allowed (typed, not repeated in the answer), and Windows
Calculator. Median 35 s a task; the delete-all took 269 s and 28 steps, and on the way the tool typed "Confirm delete"
into the app's Notes field, which the worker noticed and cleared (a desktop-tool quirk, not one of devices). Raw:
`evals/bench/devices_bilbo_desktop_raw.json`.

Not checked live yet:

- the app's window and tray on a computer other than the hub.

## Limits

- Devices are Windows computers. Files and the shell would work elsewhere, but the desktop tool and the app's Start
  menu entry are Windows-only. The iPhone is a chat client only.
- The hub runs one turn at a time, so two devices can't each run a separate task at once. A task can use several devices
  in turn.
- Files move inside the connection: 48 MB per file, per call.
- A device's browser and desktop tools ask the hub's Jev (through the hub), so they are slower than on the hub by the
  network's round trip.
