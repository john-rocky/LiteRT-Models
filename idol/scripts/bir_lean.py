"""BiRefNet (idol.models.birefnet, the author's code) with the same arithmetic in a form whose peak memory fits a
phone CPU: the shipped bir.tflite is one CPU graph (its deformable convolutions are gathers, which the LiteRT GPU
does not run), and its first form peaked at 9.3 GB in one call on the Mac CPU and was killed on the Galaxy S26.

Three rewrites, each exact in real arithmetic (only the order of float additions changes):
  1. the deformable convolutions run their output rows in tiles (_deform): no gathered [taps, rows, width,
     channels] tensor passes GATHER_MAX_BYTES, and a data dependency makes the tiles run one after another;
  2. _ASPPDeformable: conv1 (1x1) over the concatenation of its five branches = the sum of conv1's weight slices
     applied to each branch as it is made (one 256-channel branch alive at a time, no 1280-channel concatenation);
     the pooled branch is constant over the map (a 1x1 map resized), so its slice is a per-channel bias;
  3. the decoder's last stage: conv_out1 (1x1, 240 -> 1) over [resize(p1) (192 channels), ipt_blk1(image) (48)]
     = resize(conv_out1[:192](p1)) + (conv_out1[192:] folded into ipt_blk1's last 3x3 conv, which has no
     activation after it), so neither the 1024 x 1024 x 192 resize nor the 240-channel concatenation exists.

  import bir_lean; bir_lean.apply(BR)   # BR = idol.models.birefnet, before the forward runs
"""
import torch
import torch.nn.functional as F

from deform_conv import deform_conv2d_gather

GATHER_MAX_BYTES = 64 << 20


_chain = [None]   # the last deformable-convolution tile output of the running forward (see _deform)


def _deform(input, offset, weight, bias=None, padding=0, mask=None, stride=(1, 1)):
    """deform_conv2d in output-row tiles, each tile's offsets plus (r - r) = 0 with r taken from the previous tile's
    output. The LiteRT CPU runs the lookups outside XNNPACK, and its partitioner otherwise runs every lookup that is
    ready before the XNNPACK partition that consumes them, so all tiles' gathered tensors were alive at once (the
    tiling saved nothing: 6.7 GB peak on the Mac CPU). The dependency makes the tiles of all 20 convolutions run one
    after another; adding an exact zero changes no value."""
    assert tuple(stride) == (1, 1)
    b, c, _, _ = input.shape
    k = weight.shape[2] * weight.shape[3]
    ho, wo = offset.shape[2], offset.shape[3]
    rows = max(1, min(ho, GATHER_MAX_BYTES // (b * k * wo * c * 4)))
    outs = []
    for r0 in range(0, ho, rows):
        r1 = min(ho, r0 + rows)
        off = offset[:, :, r0:r1]
        if _chain[0] is not None:
            r = _chain[0].reshape(-1)[:1]
            off = off + (r - r)
        y = deform_conv2d_gather(input, off, weight, None, padding, None if mask is None else mask[:, :, r0:r1],
                                 lean=True, row0=r0)
        _chain[0] = y
        outs.append(y)
    y = torch.cat(outs, dim=2) if len(outs) > 1 else outs[0]
    return y if bias is None else y + bias.view(1, -1, 1, 1)


def _aspp_forward(self, x):
    w = self.conv1.weight                                     # [out, 256 * 5, 1, 1], no bias
    inter = w.shape[1] // (2 + len(self.aspp_deforms))
    branches = [self.aspp1] + list(self.aspp_deforms)
    acc = None
    for i, m in enumerate(branches):
        y = F.conv2d(m(x), w[:, i * inter:(i + 1) * inter])
        acc = y if acc is None else acc + y
    x5 = self.global_avg_pool(x)                              # [1, 256, 1, 1]; resized to the map = constant
    acc = acc + F.conv2d(x5, w[:, len(branches) * inter:])
    return self.relu(self.bn1(acc))


def _decoder_forward(self, x, x1, x2, x3, x4):
    from idol.models.birefnet import _image2patches
    x4 = torch.cat((x4, self.ipt_blk5(_image2patches(x, x4))), 1)
    p4 = self.decoder_block4(x4)
    p4 = p4 * self.gdt_convs_attn_4(self.gdt_convs_4(p4)).sigmoid()
    _p4 = F.interpolate(p4, size=x3.shape[2:], mode='bilinear', align_corners=True)
    _p3 = _p4 + self.lateral_block4(x3)

    _p3 = torch.cat((_p3, self.ipt_blk4(_image2patches(x, _p3))), 1)
    p3 = self.decoder_block3(_p3)
    p3 = p3 * self.gdt_convs_attn_3(self.gdt_convs_3(p3)).sigmoid()
    _p3 = F.interpolate(p3, size=x2.shape[2:], mode='bilinear', align_corners=True)
    _p2 = _p3 + self.lateral_block3(x2)

    _p2 = torch.cat((_p2, self.ipt_blk3(_image2patches(x, _p2))), 1)
    p2 = self.decoder_block2(_p2)
    p2 = p2 * self.gdt_convs_attn_2(self.gdt_convs_2(p2)).sigmoid()
    _p2 = F.interpolate(p2, size=x1.shape[2:], mode='bilinear', align_corners=True)
    _p1 = _p2 + self.lateral_block2(x1)

    _p1 = torch.cat((_p1, self.ipt_blk2(_image2patches(x, _p1))), 1)
    _p1 = self.decoder_block1(_p1)                             # [1, 192, H / 4, W / 4]
    head = self.conv_out1[0]                                   # 1x1, 240 -> 1, bias
    n = _p1.shape[1]
    a = F.conv2d(_p1, head.weight[:, :n])                       # the 1x1 before the resize (both linear)
    a = F.interpolate(a, size=x.shape[2:], mode='bilinear', align_corners=True)
    blk = self.ipt_blk1                                         # conv1 (3 -> 64), conv_out (64 -> 48), no activation
    wb = head.weight[:, n:, 0, 0]                               # [1, 48]
    w_fold = torch.einsum('oc,cikl->oikl', wb, blk.conv_out.weight)        # [1, 64, 3, 3]
    b_fold = wb @ blk.conv_out.bias + head.bias                 # [1]
    b = F.conv2d(blk.conv1(_image2patches(x, x)), w_fold, b_fold, padding=1)
    return a + b


def apply(BR):
    """Patch the module in place; returns a function that restores the author's forward."""
    saved = (BR.deform_conv2d, BR._ASPPDeformable.forward, BR._BiRefNetDecoder.forward, BR.BiRefNet.forward)
    author_forward = BR.BiRefNet.forward

    def forward(self, x):
        _chain[0] = None   # each forward (and each export trace) starts its own chain
        try:
            return author_forward(self, x)
        finally:
            _chain[0] = None

    BR.deform_conv2d = _deform
    BR._ASPPDeformable.forward = _aspp_forward
    BR._BiRefNetDecoder.forward = _decoder_forward
    BR.BiRefNet.forward = forward

    def restore():
        BR.deform_conv2d, BR._ASPPDeformable.forward, BR._BiRefNetDecoder.forward, BR.BiRefNet.forward = saved
    return restore
