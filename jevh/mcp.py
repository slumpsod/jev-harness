"""MCP (Model Context Protocol) servers over stdio, shared by every worker.

config.yaml:
    mcp_servers:
      blender: {command: uvx, args: [blender-mcp], env: {...}, timeout: 60}

Servers start on first use. Their tools form one catalog ("server.tool"); per task, Jev picks
the few whose descriptions fit (same Choice + fits design as skills), and only those reach the
worker's prompt. Workers call them from the REPL with mcp("server.tool", arg=value).
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

PROTOCOL = "2025-06-18"


@dataclass
class McpTool:
    server: str
    name: str
    description: str
    schema: dict

    @property
    def key(self) -> str:
        return f"{self.server}.{self.name}"

    def signature(self) -> str:
        props = (self.schema or {}).get("properties", {})
        req = set((self.schema or {}).get("required", []))
        args = ", ".join(f"{k}{'' if k in req else '?'}: {v.get('type', 'any')}" for k, v in props.items())
        return f'mcp("{self.key}", {args})'


class McpError(RuntimeError):
    pass


class McpServer:
    def __init__(self, name: str, command: str, args: list[str] | None = None, env: dict | None = None,
                 timeout: float = 60, cwd: str | None = None):
        self.name, self.command, self.args = name, command, list(args or [])
        self.env, self.timeout, self.cwd = env or {}, float(timeout), cwd
        self.proc: subprocess.Popen | None = None
        self._ids = itertools.count(1)
        self._pending: dict[int, Future] = {}
        self._lock = threading.Lock()
        self.tools: list[McpTool] = []

    def start(self) -> None:
        env = {**os.environ, **{k: str(v) for k, v in self.env.items()}}
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen([self.command, *self.args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, env=env, cwd=self.cwd, creationflags=flags)
        threading.Thread(target=self._read, daemon=True, name=f"mcp-{self.name}").start()
        self.request("initialize", {"protocolVersion": PROTOCOL, "capabilities": {},
                                    "clientInfo": {"name": "jev-harness", "version": "0.1"}})
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        cursor, tools = None, []
        while True:
            res = self.request("tools/list", {"cursor": cursor} if cursor else {})
            tools += res.get("tools", [])
            cursor = res.get("nextCursor")
            if not cursor:
                break
        self.tools = [McpTool(self.name, t["name"], (t.get("description") or "").strip(), t.get("inputSchema") or {})
                      for t in tools]

    def _send(self, msg: dict) -> None:
        with self._lock:
            self.proc.stdin.write((json.dumps(msg) + "\n").encode("utf-8"))
            self.proc.stdin.flush()

    def _read(self) -> None:
        for line in self.proc.stdout:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue  # servers sometimes log to stdout
            if "id" in msg and ("result" in msg or "error" in msg):
                fut = self._pending.pop(msg["id"], None)
                if fut:
                    fut.set_result(msg)
            elif "id" in msg and "method" in msg:  # server -> client request: answer pings, refuse the rest
                reply = {"jsonrpc": "2.0", "id": msg["id"]}
                reply.update({"result": {}} if msg["method"] == "ping" else
                             {"error": {"code": -32601, "message": "not supported by this client"}})
                self._send(reply)
        for fut in list(self._pending.values()):
            fut.set_result({"error": {"message": f"MCP server {self.name} exited"}})

    def request(self, method: str, params: dict) -> dict:
        if not self.proc or self.proc.poll() is not None:
            raise McpError(f"MCP server {self.name} is not running")
        rid = next(self._ids)
        fut: Future = Future()
        self._pending[rid] = fut
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        msg = fut.result(self.timeout)
        if "error" in msg:
            raise McpError(f"{self.name} {method}: {msg['error'].get('message', msg['error'])}")
        return msg.get("result") or {}

    def call(self, tool: str, arguments: dict) -> str:
        res = self.request("tools/call", {"name": tool, "arguments": arguments})
        parts = []
        for c in res.get("content", []):
            if c.get("type") == "text":
                parts.append(c["text"])
            elif c.get("type") == "resource":
                parts.append(json.dumps(c.get("resource"))[:4000])
            else:
                parts.append(f"[{c.get('type')} content]")
        if res.get("structuredContent") and not parts:
            parts.append(json.dumps(res["structuredContent"]))
        text = "\n".join(parts)
        if res.get("isError"):
            raise McpError(f"{self.name}.{tool} failed: {text[:2000]}")
        return text

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.kill()


class McpHub:
    def __init__(self, servers: dict[str, dict] | None):
        self.config = dict(servers or {})
        self.servers: dict[str, McpServer] = {}
        self.errors: dict[str, str] = {}
        self._lock = threading.Lock()

    def __bool__(self) -> bool:
        return bool(self.config)

    def _server(self, name: str) -> McpServer | None:
        with self._lock:
            if name not in self.servers and name not in self.errors:
                spec = dict(self.config[name])
                srv = McpServer(name, spec.pop("command"), spec.pop("args", None), spec.pop("env", None),
                                spec.pop("timeout", 60), spec.pop("cwd", None))
                try:
                    srv.start()
                    self.servers[name] = srv
                except Exception as e:  # noqa: BLE001 - a broken server must not take the turn down
                    srv.close()
                    self.errors[name] = f"{type(e).__name__}: {e}"
            return self.servers.get(name)

    def catalog(self) -> dict[str, McpTool]:
        out = {}
        for name in self.config:
            srv = self._server(name)
            for t in srv.tools if srv else []:
                out[t.key] = t
        return out

    def call(self, key: str, arguments: dict) -> str:
        server, _, tool = key.partition(".")
        if server not in self.config:
            raise McpError(f"unknown MCP server '{server}'")
        srv = self._server(server)
        if srv is None:
            raise McpError(f"MCP server {server} failed to start: {self.errors.get(server)}")
        return srv.call(tool, arguments)

    def close(self) -> None:
        for s in self.servers.values():
            s.close()
