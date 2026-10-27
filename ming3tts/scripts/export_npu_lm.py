"""The language model of the all-NPU placement: lm2 with one change in SafeRMSNorm, one file per signature.

  python scripts/export_npu_lm.py     -> $MING3TTS_OUT/npu/lm2_epsdiv_prefill_wfp16.tflite, lm2_epsdiv_step_wfp16.tflite

1. SafeRMSNorm adds eps / (m * m), m = max(1, max|x|). The Qualcomm HTP computes in fp16, where a divide by a
   finite value above 2^15 returns 0 or NaN, and m * m reaches that range for some tokens. These files compute the
   same term as (eps / m) / m (exact in real arithmetic; m itself stays below 2^15). Nothing else changes: the
   weights, modules and export path of `export_graphs.py --graph lm2 --fp16` (prefill + step, float16 weights).
2. An NPU compile on the phone keeps one copy of the weights per signature of a file, which a Galaxy S26 did not
   have the memory for, so each signature becomes its own file: the kept subgraph, its tensors and ops are
   byte-for-byte the two-signature file's; the other subgraph and the buffers only it uses are dropped.
The float32 export is a byproduct and is deleted. Run it in its own process, like the other exports.
"""
import os
import sys

import flatbuffers
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import export_graphs as E  # noqa: E402
import litert_helpers as LH  # noqa: E402
import ming_modules as M  # noqa: E402

NPU = C.OUT / "npu"


def forward_epsdiv(self, x):
    m = torch.clamp(torch.amax(torch.abs(x), dim=-1, keepdim=True), min=1.0)
    xs = x / m
    var = torch.mean(xs * xs, dim=-1, keepdim=True)
    return xs * torch.rsqrt(var + (self.eps / m) / m) * self.weight


def split(src, sig_key, dst):
    """One signature of a multi-signature .tflite as its own file."""
    from ai_edge_litert import schema_py_generated as schema
    with open(src, "rb") as f:
        buf = f.read()
    model = schema.ModelT.InitFromPackedBuf(buf, 0)
    sig = next(s for s in model.signatureDefs if s.signatureKey.decode() == sig_key)
    g = model.subgraphs[sig.subgraphIndex]
    keep = {0}                                       # buffer 0 = the empty sentinel
    keep |= {t.buffer for t in g.tensors}
    keep |= {m.buffer for m in (model.metadata or [])}
    order = sorted(keep)
    remap = {old: new for new, old in enumerate(order)}
    for t in g.tensors:
        t.buffer = remap[t.buffer]
    for m in model.metadata or []:
        m.buffer = remap[m.buffer]
    model.buffers = [model.buffers[i] for i in order]
    model.subgraphs = [g]
    sig.subgraphIndex = 0
    model.signatureDefs = [sig]
    # The exporter keeps large buffers outside the flatbuffer (offset/size from the file start, "only valid
    # if > 1"). Re-append those blobs after the new flatbuffer, 64-byte aligned. Pass 1 packs with a nonzero
    # placeholder offset (scalars are fixed width, so pass 2 has the same length) to learn where data starts.
    ext = [b for b in model.buffers if (b.offset or 0) > 1]
    blobs = [buf[b.offset:b.offset + b.size] for b in ext]

    def pack():
        fb = flatbuffers.Builder(1 << 24)
        fb.Finish(model.Pack(fb), file_identifier=b"TFL3")
        return bytes(fb.Output())

    align = lambda n: (n + 63) // 64 * 64  # noqa: E731
    for b in ext:
        b.offset = 1 << 40
    head = pack()
    cur = align(len(head))
    for b, blob in zip(ext, blobs):
        b.offset, b.size = cur, len(blob)
        cur = align(cur + len(blob))
    head2 = pack()
    assert len(head2) == len(head), (len(head2), len(head))
    with open(dst, "wb") as f:
        f.write(head2)
        for b, blob in zip(ext, blobs):
            f.write(b"\0" * (b.offset - f.tell()))
            f.write(blob)


def same_outputs(src, dst, sig_key):
    """The split file against its signature in the source, on LiteRT CPU, on zero inputs of the right shapes."""
    a = LH.MultiSigRunner(src)
    args = [np.zeros(shape, np.float32) for _, shape, _ in a.specs[sig_key]["inputs"]]
    ra = a.call(sig_key, *args)
    a.close()
    b = LH.MultiSigRunner(dst)
    assert list(b.specs) == [sig_key], list(b.specs)
    rb = b.call(sig_key, *args)
    b.close()
    return all(bool(np.array_equal(x, y)) for x, y in zip(ra, rb))


def main():
    NPU.mkdir(parents=True, exist_ok=True)
    fp32, f16 = NPU / "lm2_epsdiv_fp32.tflite", NPU / "lm2_epsdiv_wfp16.tflite"
    orig = M.SafeRMSNorm.forward
    M.SafeRMSNorm.forward = forward_epsdiv
    try:
        sigs = E.build("lm2", 8, 10)
        import litert_torch
        conv = None
        for key, mod, sample in sigs:
            conv = litert_torch.signature(key, mod, sample) if conv is None else conv.signature(key, mod, sample)
        conv.convert().export(str(fp32))
    finally:
        M.SafeRMSNorm.forward = orig
    LH.to_fp16(str(fp32), str(f16))
    fp32.unlink()
    for sig in ("prefill", "step"):
        dst = NPU / f"lm2_epsdiv_{sig}_wfp16.tflite"
        split(f16, sig, dst)
        print("NPU_LM", dst.name, os.path.getsize(dst), "bytes sha256", C.sha256(dst),
              "same outputs as the two-signature file:", same_outputs(f16, dst, sig), flush=True)
    f16.unlink()   # the two single-signature files are what the app loads


if __name__ == "__main__":
    main()
