"""Static 64-frame native Sopro streaming backbone + fp32 spectral head.

The source's seven-tap depthwise MUL/ADD order and erf GELU are preserved.
Spectral operations remain on the host. First call emits 64 - 27 = 37 frames.
"""
import copy
import torch
from torch import nn


def pack(state):
    return (state['embed'], torch.stack([s['conv'] for s in state['blocks']]),
            torch.stack([s['pending'] for s in state['blocks']]))


def unpack(embed, conv, pending):
    return {'embed': embed, 'blocks': [{'conv': conv[i], 'pending': pending[i]} for i in range(8)]}


class VocoderStreamGraph(nn.Module):
    def __init__(self, vocoder, mode):
        super().__init__()
        assert mode in ('start', 'step', 'flush')
        self.mode = mode
        self.backbone = copy.deepcopy(vocoder.backbone)
        self.out = copy.deepcopy(vocoder.head.out)

    def forward(self, *args):
        if self.mode == 'start':
            hidden, state = self.backbone.forward_stream(args[0], None, False)
        elif self.mode == 'step':
            mel, embed, conv, pending = args
            hidden, state = self.backbone.forward_stream(mel, unpack(embed, conv, pending), False)
        else:
            embed, conv, pending = args
            mel = embed.new_zeros((1, 100, 0))
            hidden, state = self.backbone.forward_stream(mel, unpack(embed, conv, pending), True)
        features = self.out(hidden)
        return features if self.mode == 'flush' else (features, *pack(state))


def source_examples(vocoder, mel):
    with torch.inference_mode():
        _, state = vocoder.backbone.forward_stream(mel[:, :, :64], None, False)
        return {'start': (mel[:, :, :64].contiguous(),),
                'step': (mel[:, :, 64:128].contiguous(), *pack(state)),
                'flush': pack(state)}
