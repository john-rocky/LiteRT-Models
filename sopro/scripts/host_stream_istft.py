"""NumPy mirror of sopro.vocoder.ISTFT.forward_stream, including flush crop."""
from dataclasses import dataclass
import numpy as np
from host_dsp import HostDSP


@dataclass
class ISTFTState:
    processed_frames: int = 0
    emitted_samples: int = 0
    tail_start: int = 0
    ola: object = None
    env: object = None


class StreamingISTFT:
    def __init__(self, window, n_fft=1024, hop_length=256):
        self.window = np.asarray(window, np.float32)
        self.n_fft, self.hop_length = n_fft, hop_length
        self.pad = n_fft // 2

    def overlap_add(self, spec):
        batch, _, frames = spec.shape
        if not frames:
            return np.zeros((batch, 0), np.float32), np.zeros(0, np.float32)
        ifft = np.fft.irfft(spec, n=self.n_fft, axis=1).astype(np.float32) * self.window[None, :, None]
        size = (frames-1)*self.hop_length + self.n_fft
        y, env = np.zeros((batch, size), np.float32), np.zeros(size, np.float32)
        wsq = self.window*self.window
        for i in range(frames):
            start = i*self.hop_length
            y[:, start:start+self.n_fft] += ifft[:, :, i]
            env[start:start+self.n_fft] += wsq
        return y, env

    def __call__(self, features, state=None, flush=False):
        return self.from_spectrum(HostDSP.spectrum(features), state, flush)

    def from_spectrum(self, spec, state=None, flush=False):
        st = state or ISTFTState()
        y_chunk, env_chunk = self.overlap_add(spec)
        offset = st.processed_frames*self.hop_length-st.tail_start
        required = offset+y_chunk.shape[-1]
        cur = 0 if st.ola is None else st.ola.shape[-1]
        length = max(cur, required)
        ola, env = np.zeros((spec.shape[0], length), np.float32), np.zeros(length, np.float32)
        if cur:
            ola[:, :cur], env[:cur] = st.ola, st.env
        if y_chunk.shape[-1]:
            ola[:, offset:required] += y_chunk
            env[offset:required] += env_chunk
        st.processed_frames += spec.shape[-1]
        target = max(0, st.processed_frames*self.hop_length-self.pad)
        if flush and st.processed_frames:
            target = max(target, (st.processed_frames-1)*self.hop_length+self.n_fft-2*self.pad)
        emit_count = max(0, target-st.emitted_samples)
        rel_start = st.emitted_samples+self.pad-st.tail_start
        rel_end = rel_start+emit_count
        out = ola[:, rel_start:rel_end] / np.maximum(env[None, rel_start:rel_end], np.float32(1e-8))
        st.emitted_samples = target
        trim = st.emitted_samples+self.pad-st.tail_start
        st.ola, st.env = ola[:, trim:].copy(), env[trim:].copy()
        st.tail_start += trim
        return out, ISTFTState() if flush else st


def static_stream_features(mel, invoke):
    """Yield exact retained features from fixed64 start/step/flush calls.

    For a partial final chunk, replay the final128 REAL frames in a separate
    state and retain only the last remainder frames of its step plus flush27.
    The earliest retained frame has >=38 left frames, exceeding context27.
    This explicitly supports total lengths >=128, matching all24 fixtures.
    ``invoke(mode, arrays)`` must return a list of feature/state arrays.
    """
    mel = np.ascontiguousarray(mel, np.float32)
    length = mel.shape[-1]
    if length < 128:
        raise ValueError('Static stream host contract requires at least128 real mel frames')
    full, remainder = divmod(length, 64)
    outputs = invoke('start', (mel[:, :, :64],))
    yield outputs[0], False
    state = outputs[1:]
    for i in range(1, full):
        outputs = invoke('step', (mel[:, :, i*64:(i+1)*64], *state))
        yield outputs[0], False
        state = outputs[1:]
    if remainder:
        tail = mel[:, :, -128:]
        restart = invoke('start', (tail[:, :, :64],))
        tail_step = invoke('step', (tail[:, :, 64:], *restart[1:]))
        yield tail_step[0][:, -remainder:], False
        state = tail_step[1:]
    yield invoke('flush', tuple(state))[0], True


def decode_static(mel, invoke, istft):
    pieces, features, st = [], [], None
    for f, flush in static_stream_features(mel, invoke):
        features.append(f)
        wav, st = istft(f, st, flush)
        pieces.append(wav)
    return np.concatenate(pieces, axis=-1), np.concatenate(features, axis=1)
