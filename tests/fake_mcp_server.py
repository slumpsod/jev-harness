"""A tiny stdio MCP server for tests: tools `echo` and `add`, plus a failing `boom` and a `hang` that never answers.
FAKE_MCP_NOISE=1: before each answer it also logs a bare JSON number and bytes that are not UTF-8 to stdout."""
import json
import os
import sys

NOISE = os.environ.get("FAKE_MCP_NOISE") == "1"

TOOLS = [
    {"name": "echo", "description": "Echo text back.", "inputSchema": {"type": "object",
     "properties": {"text": {"type": "string"}}, "required": ["text"]}},
    {"name": "add", "description": "Add two numbers.", "inputSchema": {"type": "object",
     "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]}},
    {"name": "boom", "description": "Always fails.", "inputSchema": {"type": "object", "properties": {}}},
]

print("fake server starting (log noise on stdout)", flush=True)
for line in sys.stdin:
    msg = json.loads(line)
    if "id" not in msg:
        continue
    method, params = msg["method"], msg.get("params") or {}
    if method == "initialize":
        result = {"protocolVersion": params.get("protocolVersion"), "capabilities": {"tools": {}},
                  "serverInfo": {"name": "fake", "version": "0"}}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        name, args = params["name"], params.get("arguments") or {}
        if name == "hang":
            continue
        if NOISE:
            print("42", flush=True)
            sys.stdout.buffer.write(b"\xff\xfe progress \x81\n")
            sys.stdout.buffer.flush()
        if name == "echo":
            result = {"content": [{"type": "text", "text": args["text"]}]}
        elif name == "add":
            result = {"content": [{"type": "text", "text": str(args["a"] + args["b"])}]}
        else:
            result = {"content": [{"type": "text", "text": "it broke"}], "isError": True}
    else:
        print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": "nope"}}), flush=True)
        continue
    print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}), flush=True)
