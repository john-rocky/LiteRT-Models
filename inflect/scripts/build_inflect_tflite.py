"""Build Inflect-Nano-v2 LiteRT graphs (dynamic length) from extracted weights.

Runs in the TF venv. Reads out/inflect_weights.npz + out/inflect_golden.npz
(produced by extract_weights.py in the torch venv), re-authors the VITS
inference graph in TF (channels-last, weights baked as constants), verifies
each stage against the torch goldens, then converts:

    out/inflect_text_encoder.tflite : tokens[1,N] int32 -> m_p[1,N,128], logs_p[1,N,128], logw[1,N,1]
    out/inflect_decoder.tflite      : z_p[1,T,128]      -> wav[1, 256*T]

Both sequence axes are dynamic (-1 in the TFLite shape signature).
"""
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"

SD = dict(np.load(OUT / "inflect_weights.npz"))
GOLD = dict(np.load(OUT / "inflect_golden.npz"))

HIDDEN = 72
HEADS = 2
DK = 36
WINDOW = 4
INTER = 128
LN_EPS = 1e-5
LRELU_SLOPE = 0.1


def cw(name):
    """torch Conv1d weight [out,in,k] -> TF conv1d kernel [k,in,out]."""
    return tf.constant(SD[name].transpose(2, 1, 0))


def cb(name):
    return tf.constant(SD[name])


def conv1d(x, prefix, dilation=1):
    y = tf.nn.conv1d(x, cw(prefix + ".weight"), stride=1, padding="SAME",
                     dilations=dilation)
    return y + cb(prefix + ".bias")


def layer_norm(x, prefix):
    mean, var = tf.nn.moments(x, axes=[-1], keepdims=True)
    x = (x - mean) * tf.math.rsqrt(var + LN_EPS)
    return x * tf.constant(SD[prefix + ".gamma"]) + tf.constant(SD[prefix + ".beta"])


def rel_attention(x, prefix):
    """VITS window-4 relative self-attention, band-mask formulation."""
    q = conv1d(x, prefix + ".conv_q")
    k = conv1d(x, prefix + ".conv_k")
    v = conv1d(x, prefix + ".conv_v")
    n = tf.shape(x)[1]

    def heads(t):
        return tf.transpose(tf.reshape(t, [1, -1, HEADS, DK]), [0, 2, 1, 3])

    q, k, v = heads(q) / DK ** 0.5, heads(k), heads(v)
    scores = tf.matmul(q, k, transpose_b=True)  # [1,H,N,N]

    emb_k = tf.constant(SD[prefix + ".emb_rel_k"][0])  # [9, DK]
    emb_v = tf.constant(SD[prefix + ".emb_rel_v"][0])
    rel_logits = tf.matmul(q, tf.transpose(emb_k))  # [1,H,N,9]
    pos = tf.range(n)
    offset = pos[None, :] - pos[:, None]  # [N,N] = j - i
    local = tf.zeros_like(scores)
    for r in range(2 * WINDOW + 1):
        band = tf.cast(tf.equal(offset, r - WINDOW), tf.float32)[None, None]
        local += rel_logits[..., r:r + 1] * band
    p = tf.nn.softmax(scores + local, axis=-1)

    out = tf.matmul(p, v)  # [1,H,N,DK]
    rel_w = tf.stack(
        [tf.reduce_sum(p * tf.cast(tf.equal(offset, r - WINDOW), tf.float32)[None, None], axis=-1)
         for r in range(2 * WINDOW + 1)], axis=-1)  # [1,H,N,9]
    out += tf.matmul(rel_w, emb_v)
    out = tf.reshape(tf.transpose(out, [0, 2, 1, 3]), [1, -1, HEADS * DK])
    return conv1d(out, prefix + ".conv_o")


def encoder(x):
    for i in range(3):
        y = rel_attention(x, f"enc_p.encoder.attn_layers.{i}")
        x = layer_norm(x + y, f"enc_p.encoder.norm_layers_1.{i}")
        h = tf.nn.relu(conv1d(x, f"enc_p.encoder.ffn_layers.{i}.conv_1"))
        y = conv1d(h, f"enc_p.encoder.ffn_layers.{i}.conv_2")
        x = layer_norm(x + y, f"enc_p.encoder.norm_layers_2.{i}")
    return x


def duration_predictor(x):
    x = tf.nn.relu(conv1d(x, "dp.conv_1"))
    x = layer_norm(x, "dp.norm_1")
    x = tf.nn.relu(conv1d(x, "dp.conv_2"))
    x = layer_norm(x, "dp.norm_2")
    return conv1d(x, "dp.proj")


def text_encoder_fn(tokens):
    emb = tf.constant(SD["enc_p.emb.weight"])
    x = tf.gather(emb, tokens) * HIDDEN ** 0.5  # [1,N,72]
    x = encoder(x)
    stats = conv1d(x, "enc_p.proj")
    m_p, logs_p = stats[..., :INTER], stats[..., INTER:]
    logw = duration_predictor(x)
    return m_p, logs_p, logw


def wn(x, prefix):
    """WaveNet block of the coupling layers: k=5, dilation 1, 4 layers, g=None."""
    output = tf.zeros_like(x)
    for i in range(4):
        x_in = conv1d(x, f"{prefix}.in_layers.{i}")
        acts = tf.tanh(x_in[..., :HIDDEN]) * tf.sigmoid(x_in[..., HIDDEN:])
        rs = conv1d(acts, f"{prefix}.res_skip_layers.{i}")
        if i < 3:
            x = x + rs[..., :HIDDEN]
            output += rs[..., HIDDEN:]
        else:
            output += rs
    return output


def flow_reverse(x):
    for i in (6, 4, 2, 0):
        x = tf.reverse(x, axis=[-1])  # Flip
        x0, x1 = x[..., :INTER // 2], x[..., INTER // 2:]
        h = conv1d(x0, f"flow.flows.{i}.pre")
        h = wn(h, f"flow.flows.{i}.enc")
        m = conv1d(h, f"flow.flows.{i}.post")
        x = tf.concat([x0, x1 - m], axis=-1)
    return x


def resblock1(x, idx, kernel, dilations=(1, 3, 5)):
    for j, d in enumerate(dilations):
        xt = tf.nn.leaky_relu(x, LRELU_SLOPE)
        xt = conv1d(xt, f"dec.resblocks.{idx}.convs1.{j}", dilation=d)
        xt = tf.nn.leaky_relu(xt, LRELU_SLOPE)
        xt = conv1d(xt, f"dec.resblocks.{idx}.convs2.{j}")
        x = x + xt
    return x


def generator(x):
    x = conv1d(x, "dec.conv_pre")  # [1,T,192]
    kernels = [3, 7, 11]
    ups = [(16, 8), (16, 8), (4, 2), (4, 2)]
    for i, (k, s) in enumerate(ups):
        x = tf.nn.leaky_relu(x, LRELU_SLOPE)
        w = tf.constant(SD[f"dec.ups.{i}.weight"].transpose(2, 1, 0))  # [k,out,in]
        ch = w.shape[1]
        t = tf.shape(x)[1]
        x = tf.nn.conv1d_transpose(
            x, w, output_shape=[1, t * s, ch], strides=s, padding="SAME")
        x = x + cb(f"dec.ups.{i}.bias")
        xs = resblock1(x, 3 * i, kernels[0])
        for j in (1, 2):
            xs += resblock1(x, 3 * i + j, kernels[j])
        x = xs / 3.0
    # torch F.leaky_relu default slope is 0.01; TF's default is 0.2 — be explicit
    x = tf.nn.leaky_relu(x, alpha=0.01)
    w_post = tf.constant(SD["dec.conv_post.weight"].transpose(2, 1, 0))
    x = tf.nn.conv1d(x, w_post, stride=1, padding="SAME")  # no bias
    # reshape, not [..., 0]: shrink-axis STRIDED_SLICE is rejected by the GPU
    # delegate and CompiledModel GPU requires every op to be accepted
    return tf.reshape(tf.tanh(x), [1, -1])  # [1, 256*T]


def decoder_fn(z_p):
    z = flow_reverse(z_p)
    return generator(z)


def to_cl(a):
    """golden [1,C,N] -> channels-last [1,N,C]."""
    return np.transpose(a, (0, 2, 1))


def verify_eager():
    tokens = tf.constant(GOLD["tokens"].astype(np.int32))
    m_p, logs_p, logw = text_encoder_fn(tokens[0][None] if tokens.ndim == 3 else tokens)
    checks = [
        ("m_p", m_p.numpy(), to_cl(GOLD["m_p"])),
        ("logs_p", logs_p.numpy(), to_cl(GOLD["logs_p"])),
        ("logw", logw.numpy(), to_cl(GOLD["logw"])),
    ]
    z_p = tf.constant(to_cl(GOLD["z_p"]))
    wav = decoder_fn(z_p)
    checks.append(("wav", wav.numpy()[0], GOLD["wav"][0, 0]))
    for name, got, ref in checks:
        err = np.abs(got - ref).max()
        denom = max(np.abs(ref).max(), 1e-9)
        print(f"  eager {name}: maxerr={err:.3e} (rel {err/denom:.2e})")


def convert(fn, spec, path):
    cf = tf.function(fn, input_signature=[spec]).get_concrete_function()
    conv = tf.lite.TFLiteConverter.from_concrete_functions([cf])
    flat = conv.convert()
    Path(path).write_bytes(flat)
    print(f"[convert] {Path(path).name}  {len(flat)/1e6:.1f} MB")


def main():
    print("== eager verification vs torch goldens ==")
    verify_eager()
    print("== convert ==")
    convert(text_encoder_fn, tf.TensorSpec([1, None], tf.int32),
            OUT / "inflect_text_encoder.tflite")
    convert(decoder_fn, tf.TensorSpec([1, None, INTER], tf.float32),
            OUT / "inflect_decoder.tflite")


if __name__ == "__main__":
    main()
