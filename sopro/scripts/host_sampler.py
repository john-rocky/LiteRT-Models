"""NumPy mirror of sopro.sampling, with one uniform inverse-CDF draw.

The distribution matches the source. NumPy's generator and inverse-CDF draw
deliberately do not claim the bitstream/algorithm of native torch.multinomial.
"""
import numpy as np

BOS_ID, EOS_ID = 4375, 4376
MAX_STEPS, MIN_STEPS = 704, 10


def probabilities(logits, temperature=.8, top_p=.9, top_k=25,
                  bos_id=BOS_ID, eos_id=EOS_ID, allow_eos=False):
    x = np.array(logits, dtype=np.float32, copy=True)
    assert x.ndim == 2 and x.shape[0] == 1
    x[:, bos_id] = np.float32(-1e9)
    if not allow_eos:
        x[:, eos_id] = np.float32(-1e9)
    if temperature <= 0:
        out = np.zeros_like(x)
        out[0, int(x.argmax())] = 1
        return out
    x /= np.float32(max(1e-5, float(temperature)))
    p = np.exp(x-x.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    if 0 < int(top_k) < p.shape[-1]:
        kth = np.partition(p, p.shape[-1]-int(top_k), axis=-1)[:, -int(top_k): -int(top_k)+1 or None]
        p = np.where(p < kth, np.float32(0), p)
        p /= np.maximum(p.sum(axis=-1, keepdims=True), np.float32(1e-8))
    if float(top_p) < 1:
        threshold = np.float32(max(0, min(1, float(top_p))))
        order = np.argsort(-p, axis=-1, stable=True)
        sorted_p = np.take_along_axis(p, order, axis=-1)
        remove = np.cumsum(sorted_p, axis=-1, dtype=np.float32) > threshold
        remove[:, 1:] = remove[:, :-1].copy()
        remove[:, :1] = False
        sorted_p = np.where(remove, np.float32(0), sorted_p)
        nucleus = np.zeros_like(p)
        np.put_along_axis(nucleus, order, sorted_p, axis=-1)
        p = nucleus / np.maximum(nucleus.sum(axis=-1, keepdims=True), np.float32(1e-8))
    return p


def inverse_cdf(probs, uniform, return_trace=False):
    p = np.asarray(probs, np.float32).reshape(-1)
    assert np.isfinite(p).all() and np.all(p >= 0)
    cdf = np.cumsum(p, dtype=np.float64)
    total = float(cdf[-1])
    assert total > 0 and 0 <= uniform < 1
    # Weighted sampling normalizes the accumulated fp32 probabilities once in
    # fp64, avoiding an out-of-range draw if their sum is 1-epsilon.
    cdf /= total
    index = int(np.searchsorted(cdf, float(uniform), side='right'))
    index = min(index, p.size-1)
    if not return_trace:
        return index
    lower = float(cdf[index-1]) if index else 0.0
    upper = float(cdf[index])
    trace = {'uniform':float(uniform), 'picked_token':index,
             'probability':float(p[index]), 'probability_sum':total,
             'cdf_lower':lower, 'cdf_upper':upper,
             'cdf_boundary_distance':min(float(uniform)-lower, upper-float(uniform))}
    return index, trace


def sample_next_token(logits, rng, temperature=.8, top_p=.9, top_k=25,
                      bos_id=BOS_ID, eos_id=EOS_ID, allow_eos=False,
                      return_trace=False):
    probs = probabilities(logits, temperature, top_p, top_k, bos_id, eos_id, allow_eos)
    if float(temperature) <= 0:
        token = int(probs.argmax())
        return (token, {'uniform':None, 'picked_token':token, 'greedy':True}) if return_trace else token
    uniform = float(rng.random())
    return inverse_cdf(probs, uniform, return_trace)


def generate_tokens(initial_logits, advance, rng, max_steps=MAX_STEPS, min_steps=MIN_STEPS,
                    bos_id=BOS_ID, eos_id=EOS_ID):
    """Source loop: no step on EOS and none after the max_steps-th token.

    A final speech token is advanced when another prediction is needed to
    discover EOS. Every positive-temperature prediction consumes one draw.
    """
    logits = initial_logits
    tokens, traces, saved_logits = [], [], []
    step_calls, stop_reason = 0, 'max_steps'
    for step in range(int(max_steps)):
        allow_eos = step+1 >= max(1, int(min_steps))
        token, trace = sample_next_token(logits, rng, bos_id=bos_id, eos_id=eos_id,
                                         allow_eos=allow_eos, return_trace=True)
        traces.append({'step':step, 'allow_eos':allow_eos, **trace})
        saved_logits.append(np.asarray(logits).copy())
        if allow_eos and token == eos_id:
            stop_reason = 'eos'
            break
        token = min(4374, max(0, token))
        tokens.append(token)
        if step+1 < int(max_steps):
            output = advance(token)
            logits = output[0] if isinstance(output, tuple) else output
            step_calls += 1
    return {'tokens':np.asarray(tokens, np.int32)[None], 'traces':traces,
            'logits':np.stack(saved_logits), 'ar_step_calls':step_calls,
            'draw_count':len(traces), 'stop_reason':stop_reason}
