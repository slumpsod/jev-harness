"""Make an ONNX export of a Laya checkpoint safe to run batched (laya_bridge.py, LAYA_ONNX).

    pip install onnx
    python laya_onnx.py model.onnx                 # rewrites it in place
    python laya_onnx.py model.onnx fixed.onnx

Why: ModernBERT's local-attention layers see 64 tokens either side. Laya's ONNXAgent pads every question of a request to
the longest one, and where a question is shorter by more than that, a padding position sees only padding: its attention
row is all -inf, the exported softmax returns NaN, and the NaN spreads into the real tokens (measured on Laya v2's export:
135 NaN answers in 300 held-out harness requests). PyTorch's attention kernel does not do this.

The fix: each attention layer masks with Where(mask, 0, -inf); the -inf becomes -1e30. A real token's attention is
unchanged (exp(-1e30) is 0 as exp(-inf) is), a padding-only row gets a finite softmax instead of NaN, and the graph runs
no extra work. The answers then match PyTorch's (within 1e-4) in one batched run per request.

The fixed file carries `laya_nan_fix` in its metadata; laya_bridge.py runs a file without it question group by question
group instead (correct, about twice as slow).
"""
from __future__ import annotations

import sys

MARK = "laya_nan_fix"
FLOOR = -1e30


def fix(src: str, dst: str | None = None) -> int:
    """Rewrite the graph at `src` into `dst` (default: in place). Returns the number of attention masks fixed."""
    import numpy as np
    import onnx
    from onnx import numpy_helper

    m = onnx.load(src)
    if any(p.key == MARK for p in m.metadata_props):
        print(f"{src}: already fixed")
        return 0
    g = m.graph
    prod = {o: n for n in g.node for o in n.output}
    inits = {i.name: i for i in g.initializer}
    n = 0
    for sm in (x for x in g.node if x.op_type == "Softmax" and "/attn/" in x.name):
        add = prod.get(sm.input[0])
        if add is None or add.op_type != "Add":
            continue
        for w in (prod.get(i) for i in add.input):
            if w is None or w.op_type != "Where":
                continue
            fill = w.input[2]   # Where(mask, 0, -inf)
            if fill in inits:
                t = inits[fill]
            elif fill in prod and prod[fill].op_type == "Constant":
                t = prod[fill].attribute[0].t
            else:
                continue
            a = numpy_helper.to_array(t)
            if a.dtype.kind == "f" and a.size and np.isneginf(a).all():
                t.CopyFrom(numpy_helper.from_array(np.full_like(a, FLOOR), t.name))
                n += 1
    if not n:
        raise SystemExit(f"{src}: no -inf attention mask found (is it a Laya export?)")
    m.metadata_props.append(onnx.StringStringEntryProto(key=MARK, value=str(n)))
    onnx.save(m, dst or src)
    print(f"{dst or src}: {n} attention masks fixed")
    return n


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit(__doc__)
    fix(*sys.argv[1:])
