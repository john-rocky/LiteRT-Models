"""NumPy acoustic host contract and CompiledModel runner for graphs six/seven."""
import numpy as np
from common import ROOT
from litert_utils import CM

N_MAX, T_MAX = 512, 2048


def build_time_grid(steps=2, sway=-1.0):
    times = np.linspace(0, 1, steps + 1, dtype=np.float32)
    return times + np.float32(sway) * (np.cos(np.float32(0.5 * np.pi) * times) - np.float32(1) + times)


def prepare_inputs(ref_tokens, generated_tokens, x0, cond_vec, ref_mel):
    tokens = np.concatenate((ref_tokens, generated_tokens), axis=1).astype(np.int32)
    n, valid = tokens.shape[-1], x0.shape[-1]
    assert n <= N_MAX and valid <= T_MAX, (n, valid)
    assert valid == ref_mel.shape[-1] + 4 * generated_tokens.shape[-1]
    token_array = np.zeros((1, N_MAX), np.int32)
    token_array[:, :n] = tokens
    token_mask = np.zeros((1, 1, N_MAX), np.float32)
    token_mask[:, :, :n] = 1
    # Use ACTUAL source/target lengths. Invalid gathers repeat the last token;
    # their mu outputs are explicitly zeroed by the host before velocity.
    frame_map = np.minimum(np.arange(T_MAX, dtype=np.int64) * n // valid, n-1).astype(np.int32)
    x = np.zeros((1, 100, T_MAX), np.float32)
    x[:, :, :valid] = x0
    cond_mel = np.zeros_like(x)
    prompt = ref_mel.shape[-1]
    cond_mel[:, :, :prompt] = ref_mel
    cond_mask = np.zeros((1, 1, T_MAX), np.float32)
    cond_mask[:, :, :prompt] = 1
    key_bias = np.full((1, 1, 1, T_MAX), -1e4, np.float32)
    key_bias[:, :, :, :valid] = 0
    return {'semantic_tokens': token_array, 'token_mask': token_mask, 'frame_to_token': frame_map,
            'x': x, 'cond_vec': np.asarray(cond_vec, dtype=np.float32), 'cond_mel': cond_mel,
            'cond_mask': cond_mask, 'key_bias': key_bias, 'valid_frames': valid, 'valid_tokens': n}


def euler_update(x, velocity, x0, cond_mel, cond_mask, t0, t1, sigma_min=1e-6):
    x = x + np.float32(t1-t0) * velocity
    x_prompt = (np.float32(1) - np.float32(1-sigma_min) * t1) * x0 + t1 * cond_mel
    return cond_mask * x_prompt + (np.float32(1) - cond_mask) * x


class AcousticRunner:
    def __init__(self, threads=4, environment=None):
        self.condition = CM(ROOT / 'exports/sopro_acoustic_condition_fp32.tflite', threads=threads, environment=environment)
        self.velocity = CM(ROOT / 'exports/sopro_acoustic_velocity_fp32.tflite', threads=threads, environment=environment)

    def mu(self, inputs):
        mu = self.condition.call_ordered(*(inputs[k] for k in ('semantic_tokens', 'token_mask', 'frame_to_token')))[0]
        mu[:, :, inputs['valid_frames']:] = 0
        return mu

    def velocity_at(self, inputs, x, t, mu):
        kwargs = {k: inputs[k] for k in ('cond_vec', 'cond_mel', 'cond_mask', 'key_bias')}
        kwargs.update(x=x, t=np.asarray([t], np.float32), mu=mu)
        return self.velocity.call_ordered(*(kwargs[k] for k in ('x', 't', 'mu', 'cond_vec', 'cond_mel', 'cond_mask', 'key_bias')))[0]

    def solve(self, ref_tokens, generated_tokens, x0, cond_vec, ref_mel):
        inputs = prepare_inputs(ref_tokens, generated_tokens, x0, cond_vec, ref_mel)
        valid = inputs['valid_frames']
        mu = self.mu(inputs)
        x0_pad = inputs['x'].copy()
        x = x0_pad.copy()
        grid = build_time_grid()
        velocities, states = [], [x[:, :, :valid].copy()]
        for t0, t1 in zip(grid[:-1], grid[1:]):
            v = self.velocity_at(inputs, x, t0, mu)
            velocities.append(v[:, :, :valid].copy())
            x = euler_update(x, v, x0_pad, inputs['cond_mel'], inputs['cond_mask'], t0, t1)
            x[:, :, valid:] = 0
            states.append(x[:, :, :valid].copy())
        solved = inputs['cond_mask'] * inputs['cond_mel'] + (np.float32(1)-inputs['cond_mask']) * x
        return {'solved_mel_normalized': solved[:, :, :valid], 'mu': mu[:, :, :valid],
                'velocities': velocities, 'states': states, 'valid_frames': valid,
                'valid_tokens': inputs['valid_tokens'], 'time_grid': grid}

    def close(self):
        self.condition.close()
        self.velocity.close()
