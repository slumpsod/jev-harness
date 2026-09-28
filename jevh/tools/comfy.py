"""ComfyUI client, pre-imported as `comfy` in the comfyui_manager's REPL.

    wf = comfy.load("Qwen Image 2.1 workflows/<file>.json")   # API-format workflow
    comfy.find(wf, "CLIPTextEncode")                           # -> {node_id: node}
    comfy.set(wf, {"6.text": "a red fox", "3.seed": comfy.seed()})
    paths = comfy.run(wf)                                      # queue, wait, save outputs as artifacts

Server URL from JEVH_COMFY_URL (default http://127.0.0.1:8188); workflow folder from
JEVH_COMFY_WORKFLOWS (default C:/Comfyui workflows).
"""
from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

URL = os.environ.get("JEVH_COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
WORKFLOWS = Path(os.environ.get("JEVH_COMFY_WORKFLOWS", "C:/Comfyui workflows"))
ARTIFACTS = Path(os.environ.get("JEVH_ARTIFACTS", "artifacts"))


class ComfyDown(RuntimeError):
    pass


def _down(e: Exception) -> ComfyDown:
    return ComfyDown(f"ComfyUI is not reachable at {URL} ({getattr(e, 'reason', e)}). It is not "
                     f"running - retrying will not help. Call final() now and report that images "
                     f"are unavailable until ComfyUI is started.")


def _get(path: str, raw: bool = False, timeout: float = 30):
    try:
        with urllib.request.urlopen(f"{URL}{path}", timeout=timeout) as r:
            data = r.read()
    except urllib.error.HTTPError:
        raise
    except (urllib.error.URLError, ConnectionError) as e:
        raise _down(e) from None
    return data if raw else json.loads(data.decode("utf-8"))


def _post(path: str, body: dict, timeout: float = 60) -> dict:
    req = urllib.request.Request(f"{URL}{path}", data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"ComfyUI {e.code}: {e.read().decode('utf-8', 'replace')[:1500]}") from e
    except (urllib.error.URLError, ConnectionError) as e:
        raise _down(e) from None


def status() -> dict:
    """Server up? GPUs + free VRAM (check before big jobs - local LLMs share the GPUs)."""
    stats = _get("/system_stats")
    return {"up": True, "devices": [
        {"name": d.get("name"), "vram_free_gb": round(d.get("vram_free", 0) / 2**30, 1),
         "vram_total_gb": round(d.get("vram_total", 0) / 2**30, 1)} for d in stats.get("devices", [])]}


def workflows(pattern: str = "**/*.json") -> list[str]:
    """Workflow files available (paths relative to the workflows folder)."""
    return sorted(str(p.relative_to(WORKFLOWS)) for p in WORKFLOWS.glob(pattern))


def load(path: str) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = WORKFLOWS / p
    wf = json.loads(p.read_text(encoding="utf-8"))
    if "nodes" in wf and "links" in wf:
        raise ValueError(f"{p.name} is a UI-format workflow; export it with 'Save (API format)'")
    return wf


def find(wf: dict, class_type: str | None = None, title: str | None = None) -> dict:
    """Nodes by class_type and/or _meta.title substring."""
    out = {}
    for nid, node in wf.items():
        if class_type and node.get("class_type") != class_type:
            continue
        if title and title.lower() not in str(node.get("_meta", {}).get("title", "")).lower():
            continue
        out[nid] = node
    return out


def set(wf: dict, values: dict) -> dict:  # noqa: A001 - short name inside the REPL
    """values: {"<node_id>.<input>": value}. Mutates and returns wf."""
    for key, value in values.items():
        nid, _, inp = key.partition(".")
        if nid not in wf:
            raise KeyError(f"node {nid} not in workflow")
        wf[nid]["inputs"][inp] = value
    return wf


def seed() -> int:
    return random.randint(0, 2**48)


def upload(path: str) -> str:
    """Upload an image to ComfyUI's input dir; returns the name for LoadImage.image."""
    p = Path(path)
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{p.name}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n").encode() + p.read_bytes() + \
           f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"{URL}/upload/image", data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))["name"]


def run(wf: dict, timeout: float = 1800, poll: float = 2.0, prefix: str = "comfy") -> list[str]:
    """Queue, wait, download every output file into the artifact store. Returns paths."""
    pid = _post("/prompt", {"prompt": wf, "client_id": uuid.uuid4().hex})["prompt_id"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        hist = _get(f"/history/{pid}")
        if pid in hist:
            entry = hist[pid]
            st = entry.get("status", {})
            if st.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI job failed: {json.dumps(st.get('messages', []))[-1500:]}")
            return _save_outputs(entry.get("outputs", {}), prefix)
        time.sleep(poll)
    raise TimeoutError(f"ComfyUI job {pid} not done after {timeout}s")


def _save_outputs(outputs: dict, prefix: str) -> list[str]:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    paths = []
    for node_out in outputs.values():
        for kind in ("images", "gifs", "videos", "audio", "files"):
            for f in node_out.get(kind, []) or []:
                if not isinstance(f, dict) or "filename" not in f:
                    continue
                q = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f.get("subfolder", ""),
                                            "type": f.get("type", "output")})
                dest = ARTIFACTS / f"{prefix}_{f['filename']}"
                dest.write_bytes(_get(f"/view?{q}", raw=True, timeout=300))
                paths.append(str(dest))
    return paths
