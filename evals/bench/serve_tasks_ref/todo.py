import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="todo.json")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("text", nargs="+")
    ls = sub.add_parser("list")
    ls.add_argument("--all", action="store_true")
    for c in ("done", "rm"):
        sub.add_parser(c).add_argument("n", type=int)
    args = ap.parse_args()
    data = {"next": 1, "items": []}
    if os.path.exists(args.file):
        data = json.load(open(args.file, encoding="utf-8"))
    items = data["items"]
    if args.cmd == "add":
        items.append({"id": data["next"], "text": " ".join(args.text), "done": False})
        print(f"added {data['next']}")
        data["next"] += 1
    elif args.cmd == "list":
        for it in items:
            if args.all or not it["done"]:
                print(f"{it['id']} [{'x' if it['done'] else ' '}] {it['text']}")
    else:
        it = next((i for i in items if i["id"] == args.n), None)
        if it is None:
            print(f"no such item: {args.n}", file=sys.stderr)
            sys.exit(1)
        if args.cmd == "done":
            it["done"] = True
            print(f"done {args.n}")
        else:
            items.remove(it)
            print(f"removed {args.n}")
    json.dump(data, open(args.file, "w", encoding="utf-8"))


main()
