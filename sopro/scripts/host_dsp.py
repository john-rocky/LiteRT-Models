"""NumPy-only host DSP for round one, using frozen exact frontend coefficients.

The coefficients are torchaudio buffers, frozen by oracle.py and hashed. This
module has no torch dependency and performs FFT, filterbank, resample and iSTFT
on the host. No learned operation or spectral approximation is introduced.
"""
import numpy as np


class HostDSP:
    def __init__(self, constants):
        with np.load(constants) as data:
            self.c = {k: data[k] for k in data.files}

    def resample_24_16(self, wav):
        x = np.asarray(wav, dtype=np.float32)
        width = int(self.c['resample_24_16_width'])
        kernel = self.c['resample_24_16_kernel'][:, 0]
        orig, new = 3, 2
        padded = np.pad(x, ((0, 0), (width, width + orig)))
        windows = np.lib.stride_tricks.sliding_window_view(padded, kernel.shape[-1], axis=-1)[:, ::orig, :]
        y = np.einsum('btk,pk->btp', windows, kernel, optimize=True).reshape(x.shape[0], -1)
        return np.ascontiguousarray(y[:, :(new*x.shape[-1]+orig-1)//orig], dtype=np.float32)

    def mel(self, wav, kind):
        if kind == 'speaker':
            nfft, hop, power = 1024, 160, 2
        elif kind == 'semantic':
            nfft, hop, power = 400, 160, 2
        elif kind == 'acoustic':
            nfft, hop, power = 1024, 256, 1
        else:
            raise ValueError(kind)
        x = np.asarray(wav, dtype=np.float32)
        frames = (x.shape[-1] + hop - 1) // hop
        if kind == 'semantic':
            x = np.pad(x, ((0, 0), (0, nfft)))
        x = np.pad(x, ((0, 0), (nfft//2, nfft//2)), mode='reflect')
        blocks = np.lib.stride_tricks.sliding_window_view(x, nfft, axis=-1)[:, ::hop]
        w = self.c[kind+'_window']
        if w.size < nfft:
            left = (nfft-w.size)//2
            w = np.pad(w, (left, nfft-w.size-left))
        if kind == 'acoustic':
            # NumPy 2.5's default fct=1 is a Python integer and resolves the
            # pocketfft loop to double precision even with complex64 output.
            # Forward normalization uses the float32 loop, matching torch.
            # 1/1024 and 1024 are exact binary scales; undoing this scale
            # introduces no approximation (see acoustic_frontend_diagnosis).
            spectrum = np.fft.rfft(blocks * w, axis=-1, norm='forward') * np.float32(nfft)
        else:
            spectrum = np.fft.rfft(blocks * w, axis=-1)
        magnitude = np.abs(spectrum).astype(np.float32)
        if power == 2:
            magnitude = magnitude * magnitude
        mel = np.matmul(magnitude, self.c[kind+'_melbank']).transpose(0, 2, 1)
        if kind == 'speaker':
            z = np.log(np.maximum(mel, np.float32(1e-5)))
            mean = z.mean(axis=1, keepdims=True)
            d = z - mean
            return np.ascontiguousarray(d / np.sqrt((d*d).mean(axis=1, keepdims=True) + np.float32(1e-5)))
        if kind == 'semantic':
            z = np.log10(np.maximum(mel[:, :, :frames+2], np.float32(1e-10)))
            return (np.maximum(z, z.max(axis=(1, 2), keepdims=True)-np.float32(8)) + np.float32(4)) / np.float32(4)
        return np.log(np.maximum(mel, np.float32(1e-7)))

    @staticmethod
    def spectrum(features):
        f = np.asarray(features, dtype=np.float32).transpose(0, 2, 1)
        logmag, phase = np.split(f, 2, axis=1)
        mag = np.exp(np.minimum(logmag, np.float32(np.log(100.0))))
        return mag * (np.cos(phase) + np.complex64(1j) * np.sin(phase))

    def istft(self, spec):
        spec = np.asarray(spec)
        b, bins, frames = spec.shape
        nfft, hop = (bins-1)*2, 256
        assert nfft == 1024
        w = self.c['istft_window']
        ifft = np.fft.irfft(spec, n=nfft, axis=1).astype(np.float32) * w[None, :, None]
        size = (frames-1)*hop+nfft
        out = np.zeros((b, size), np.float32)
        env = np.zeros(size, np.float32)
        for t in range(frames):
            start = t*hop
            out[:, start:start+nfft] += ifft[:, :, t]
            env[start:start+nfft] += w*w
        pad = nfft//2
        return out[:, pad:-pad] / env[None, pad:-pad]

    def decode_features(self, features):
        return self.istft(self.spectrum(features))
