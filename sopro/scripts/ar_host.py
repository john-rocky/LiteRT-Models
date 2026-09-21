"""NumPy AR prefix/tables/RoPE/KV host contract and reusable CPU runner."""
import numpy as np
from common import ROOT
from litert_utils import CM

P_MAX, CAP = 256, 1024


def rotary_cos_sin(positions, dim=64, base=10000.0):
    positions = np.asarray(positions, dtype=np.float32).reshape(-1)
    exponent = np.arange(0, dim, 2, dtype=np.float32) / np.float32(dim)
    inv = np.float32(1.0) / np.power(np.float32(base), exponent)
    freq = positions[:, None] * inv[None]
    emb = np.concatenate((freq, freq), axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def prefill_bias(valid_length, bucket=P_MAX, dtype=np.float32):
    pos = np.arange(bucket)
    valid = (pos[None, :] <= pos[:, None]) & (pos[None, :] < valid_length)
    return np.where(valid, 0.0, -10000.0).astype(dtype)[None, None]


def step_bias(position, capacity=CAP):
    assert 0 <= position < capacity
    return np.where(np.arange(capacity) <= position, 0.0, -10000.0).astype(np.float32)[None, None, None]


def call_graph(cm, *arrays):
    # Explicit names prevent ambiguity for same-shape cos/sin and k/v.
    result = cm.call_named({f'args_{i}': a for i, a in enumerate(arrays)})
    return [result[f'output_{i}'] for i in range(len(result))]


class ARRunner:
    """Each replay actually calls graphs 3/4/5; no cached graph output reuse."""
    def __init__(self, threads=4, environment=None):
        with np.load(ROOT/'exports/sopro_ar_tables_fp32.npz') as d:
            self.text = d['text_tok_emb'].copy()
            self.semantic = d['sem_emb512'].copy()
            self.bos_id = int(d['bos_id'])
            self.max_text_len = int(d['max_text_len'])
        self.style = CM(ROOT/'exports/sopro_style_prefix_fp32.tflite', threads, environment=environment)
        self.prefill = CM(ROOT/'exports/sopro_ar_prefill_fp32.tflite', threads, environment=environment)
        self.step = CM(ROOT/'exports/sopro_ar_step_fp32.tflite', threads, environment=environment)
        self.cos, self.sin = rotary_cos_sin(np.arange(CAP))

    def build_prefix(self, ref_tokens, text_ids):
        ref_tokens = np.asarray(ref_tokens, dtype=np.int64).reshape(1, -1)
        text_ids = np.asarray(text_ids, dtype=np.int64).reshape(1, -1)[:, :self.max_text_len]
        style = call_graph(self.style, self.semantic[ref_tokens[:, :160]])[0]
        prefix = np.concatenate((style, self.text[text_ids], self.semantic[ref_tokens[:, :120]],
                                 self.semantic[np.array([[self.bos_id]])]), axis=1)
        assert prefix.shape[1] <= P_MAX, prefix.shape
        return prefix, style

    def start(self, prefix):
        length = prefix.shape[1]
        padded = np.pad(prefix, ((0, 0), (0, P_MAX-length), (0, 0)))
        logits, k, v = call_graph(self.prefill, padded, prefill_bias(length), np.array([length-1], np.int32))
        self.pk = np.zeros((1, 96, CAP, 64), np.float32)
        self.pv = np.zeros_like(self.pk)
        self.pk[:, :, :length], self.pv[:, :, :length] = k[:, :, :length], v[:, :, :length]
        self.position = length
        return logits, k[:, :, :length], v[:, :, :length]

    def advance(self, token):
        p = self.position
        assert p < CAP
        args = (self.semantic[np.array([[int(token)]])], self.cos[p:p+1][None, None],
                self.sin[p:p+1][None, None], step_bias(p), self.pk, self.pv)
        logits, k, v = call_graph(self.step, *args)
        self.pk[:, :, p:p+1], self.pv[:, :, p:p+1] = k, v
        self.position += 1
        return logits, k, v

    def replay(self, ref_tokens, text_ids, sampled_tokens):
        prefix, style = self.build_prefix(ref_tokens, text_ids)
        logits, _, _ = self.start(prefix)
        all_logits, greedy = [logits.copy()], [int(logits.argmax())]
        for token in np.asarray(sampled_tokens).ravel():
            logits, _, _ = self.advance(token)
            all_logits.append(logits.copy())
            greedy.append(int(logits.argmax()))
        return {'prefix_length': prefix.shape[1], 'style_prefix': style,
                'logits': np.stack(all_logits), 'greedy': np.array(greedy, np.int32),
                'graph_calls': {'style_prefix': 1, 'ar_prefill': 1, 'ar_step': len(greedy)-1},
                'cache_length': self.position}

    def close(self):
        for graph in (self.style, self.prefill, self.step):
            graph.close()
