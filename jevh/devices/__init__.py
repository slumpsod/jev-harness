"""Devices: other computers the assistant may use (files, shell, browser, desktop), signed in to one hub.

- `hub.py`    in the dashboard: the devices' connections, and the calls a worker makes on one of them
- `agent.py`  on each device: one connection out to the hub; runs what it is asked (`ops.py`) and answers
- `ops.py`    what a device can be asked to do; no networking, so it is tested on its own
- `wire.py`   the messages both ends send, and files carried inside them

The worker uses them through `jevh/tools/devices.py`. See docs/architecture/22-devices.md.
"""
