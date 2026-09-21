"""Exact round-one torch graph boundaries; spectral work remains on the host."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from sopro.vocoder import _conv_causal


class GroupNorm4D(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.groups, self.eps = source.num_groups, source.eps
        self.weight, self.bias = source.weight, source.bias

    def forward(self, x):
        b, c, t = x.shape
        z = x.reshape(b, self.groups, c // self.groups, t)
        mean = z.mean(dim=(2, 3), keepdim=True)
        d = z - mean
        y = d * torch.rsqrt((d * d).mean(dim=(2, 3), keepdim=True) + self.eps)
        y = y.reshape(b, c, t)
        return y * self.weight[None, :, None] + self.bias[None, :, None]


def replace_group_norm(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.GroupNorm):
            setattr(module, name, GroupNorm4D(child))
        else:
            replace_group_norm(child)


class SpeakerGraph(nn.Module):
    output_names = ('id_emb', 'style_emb', 'style_ctrl', 'cond_vec')

    def __init__(self, tts, rewrite=True):
        super().__init__()
        self.speaker = copy.deepcopy(tts.speaker_encoder)
        self.speaker.frontend = nn.Identity()
        self.cond_proj = copy.deepcopy(tts.model.cond_proj)
        if rewrite:
            replace_group_norm(self.speaker)

    def forward(self, speaker_mel):
        s = self.speaker
        x, feats = s.stem(speaker_mel), []
        for transition, stage in zip(s.transitions, s.stages):
            x = stage(transition(x))
            feats.append(x)
        x = s.fuse(torch.cat(feats, dim=1))
        ident = F.normalize(s.id_head(s.id_pool(x)), p=2, dim=-1)
        pooled = s.style_pool(x)
        style, control = s.style_head(pooled), s.style_ctrl_head(pooled)
        cond = self.cond_proj(torch.cat((ident, style, control), dim=-1))
        return ident, style, control, cond


class VocoderGraph(nn.Module):
    def __init__(self, vocoder):
        super().__init__()
        self.backbone = copy.deepcopy(vocoder.backbone)
        self.out = copy.deepcopy(vocoder.head.out)

    def forward(self, mel, frame_mask):
        b = self.backbone
        x = _conv_causal(mel * frame_mask, b.embed, b.lookahead_frames)
        # The residual stream starts AFTER the affine embed LayerNorm: masking
        # before it alone would inject its learned bias into all padded frames.
        x = b.norm(x.transpose(1, 2)).transpose(1, 2) * frame_mask
        for block in b.convnext:
            x = block(x) * frame_mask
        return self.out(b.final_layer_norm(x.transpose(1, 2)))
