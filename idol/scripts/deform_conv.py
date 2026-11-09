"""torchvision.ops.deform_conv2d (modulated, stride 1, dilation 1, one offset group) in plain tensor ops that the LiteRT
converter lowers (GATHER on the token axis, elementwise bilinear weights, one matmul), for a CPU graph of BiRefNet's
decoder: the GPU has no gather, so the dense deformable convolution is a GPU wall (memory project_birefnet_park).

Same arithmetic as torchvision: for output pixel (h, w) and kernel tap k = (i, j), sample the input at
(h - pad + i + dy_k, w - pad + j + dx_k) bilinearly, corners outside the image count as 0, multiply by mask_k, then
contract over (tap, channel) with the weight."""
import torch
import torch.nn.functional as F


def deform_conv2d_gather(x, offset, weight, bias, padding, mask, max_bytes=None, lean=False, row0=0):
    """max_bytes: run the output rows in tiles so that no gathered [taps, rows, width, channels] float32 tensor passes
    it (the CPU's peak memory; each output element keeps the same arithmetic). None: one tile.
    lean: the corner-validity mask in min / max arithmetic (exact: the corner coordinates are integers) and the row
    lookup as an embedding lookup (EMBEDDING_LOOKUP, no bounds-check ops: the indices are clamped), so the CPU runs
    two ops per corner outside XNNPACK (the cast to int and the lookup) instead of about ten.
    row0: offset and mask hold the output rows [row0, row0 + their height) only (a row tile of the whole output; the
    input x is the whole map)."""
    b, c, h, w = x.shape
    o, _, kh, kw = weight.shape
    k = kh * kw
    ph, pw = (padding, padding) if isinstance(padding, int) else padding
    ho, wo = offset.shape[2], w + 2 * pw - kw + 1
    dev = x.device
    # base sampling grid per tap: [k, ho, wo]
    gy = torch.arange(ho, device=dev, dtype=x.dtype).view(1, ho, 1) + (row0 - ph)
    gx = torch.arange(wo, device=dev, dtype=x.dtype).view(1, 1, wo) - pw
    ki = torch.arange(kh, device=dev, dtype=x.dtype).repeat_interleave(kw).view(k, 1, 1)
    kj = torch.arange(kw, device=dev, dtype=x.dtype).repeat(kh).view(k, 1, 1)
    off = offset.view(b, k, 2, ho, wo)
    msk = mask.view(b, k, ho, wo)
    tokens = x.permute(0, 2, 3, 1).reshape(b * h * w, c)   # token-major input: row = (batch, y, x)
    base = (torch.arange(b, device=dev) * (h * w)).view(b, 1, 1, 1)
    if lean:
        base = base.to(x.dtype)
    wmat = weight.permute(2, 3, 1, 0).reshape(k * c, o)               # weight[o, c, i, j] -> [(i, j, c), o]
    rows = ho if max_bytes is None else max(1, min(ho, int(max_bytes) // (b * k * wo * c * 4)))
    outs = []
    for r0 in range(0, ho, rows):
        r1 = min(ho, r0 + rows)
        n = r1 - r0
        py = gy[:, r0:r1] + ki + off[:, :, 0, r0:r1]                 # [b, k, n, wo]
        px = gx + kj + off[:, :, 1, r0:r1]
        y0 = torch.floor(py)
        x0 = torch.floor(px)
        ly, lx = py - y0, px - x0
        acc = 0
        for dy, dx, wgt in ((0, 0, (1 - ly) * (1 - lx)), (0, 1, (1 - ly) * lx), (1, 0, ly * (1 - lx)), (1, 1, ly * lx)):
            yy, xx = y0 + dy, x0 + dx
            if lean:
                # 1 inside, 0 outside; relu(t) - relu(t - 1) = clamp(t, 0, 1) without RELU_0_TO_1 (no XNNPACK kernel)
                u = lambda t: F.relu(t) - F.relu(t - 1)  # noqa: E731
                valid = u(yy + 1) * u(h - yy) * u(xx + 1) * u(w - xx)
                idx = (yy.clamp(0, h - 1) * w + xx.clamp(0, w - 1) + base).to(torch.int64)   # [b, k, n, wo]
                v = F.embedding(idx.reshape(-1), tokens).view(b, k, n, wo, c)
            else:
                valid = ((yy >= 0) & (yy <= h - 1) & (xx >= 0) & (xx <= w - 1)).to(x.dtype)
                idx = (yy.clamp(0, h - 1) * w + xx.clamp(0, w - 1)).to(torch.int64) + base   # [b, k, n, wo]
                v = torch.index_select(tokens, 0, idx.reshape(-1)).view(b, k, n, wo, c)
            acc = acc + v * (wgt * valid * msk[:, :, r0:r1]).unsqueeze(-1)
        cols = acc.permute(0, 2, 3, 1, 4).reshape(b * n * wo, k * c)    # (tap, channel) per output pixel
        outs.append((cols @ wmat).view(b, n, wo, o))
    out = (outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)).permute(0, 3, 1, 2)
    if bias is not None:
        out = out + bias.view(1, o, 1, 1)
    return out


if __name__ == '__main__':
    from torchvision.ops import deform_conv2d
    torch.manual_seed(0)
    for ks, pad in ((1, 0), (3, 1), (7, 3)):
        x = torch.randn(1, 64, 33, 40)
        off = torch.randn(1, 2 * ks * ks, 33, 40) * 4
        m = torch.rand(1, ks * ks, 33, 40) * 2
        wt = torch.randn(256, 64, ks, ks) * 0.05
        ref = deform_conv2d(x, off, wt, None, padding=pad, mask=m)
        got = deform_conv2d_gather(x, off, wt, None, pad, m)
        tiled = deform_conv2d_gather(x, off, wt, None, pad, m, max_bytes=ks * ks * 40 * 64 * 4 * 5, lean=True)
        print(f'k={ks}: max |diff| {(ref - got).abs().max():.3e}  tiled lean vs one tile {(tiled - got).abs().max():.3e}  '
              f'max |ref| {ref.abs().max():.3e}')
