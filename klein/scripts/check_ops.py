"""GPU-compat op enumerator using ai_edge_litert (no tensorflow dependency)."""
import collections
import sys

from ai_edge_litert.interpreter import Interpreter

BANNED = {
    "GATHER_ND", "GATHER", "SELECT", "SELECT_V2",
    "NOT_EQUAL", "EQUAL", "GREATER", "LESS",
    "TOPK_V2", "CAST", "PACK", "SPLIT",
    # ML Drift rejects BROADCAST_TO outright ("Operation is not supported"), which
    # is what `Tensor.expand` lowers to. Rewrite the expand as a concat instead.
    "BROADCAST_TO",
}


def check_ops(path):
    interp = Interpreter(model_path=path)
    interp.allocate_tensors()
    details = interp._get_ops_details()
    counts = collections.Counter(d.get("op_name", "?") for d in details)
    banned = {k: v for k, v in counts.items() if k in BANNED}
    flex = {k: v for k, v in counts.items() if "Flex" in k}

    over4d = []
    for d in interp.get_tensor_details():
        shp = d.get("shape", [])
        if len(shp) > 4:
            over4d.append((d["name"], list(shp)))

    print(f"total ops: {len(details)}")
    print(f"distinct op types: {len(counts)}")
    print("op distribution:")
    for op, c in counts.most_common():
        print(f"  {op}: {c}")
    print(f"\nBANNED ops present: {banned or 'NONE'}")
    print(f"FLEX ops present:   {flex or 'NONE'}")
    print(f">4D tensors:        {len(over4d)}")
    for name, shp in over4d[:8]:
        print(f"    {shp}  {name}")
    ok = not banned and not flex
    print(f"\nOP-COMPAT: {'PASS (GPU-clean op set)' if ok else 'FAIL'}"
          f"{'  (note: >4D tensors present, warning-only)' if over4d and ok else ''}")
    return {"ok": ok, "banned": banned, "flex": flex, "over4d": over4d, "counts": counts}


if __name__ == "__main__":
    check_ops(sys.argv[1])
