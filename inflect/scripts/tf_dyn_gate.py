"""Gate test: tf_keras (Keras 2) -> TFLite with a DYNAMIC time axis.

Covers the op classes needed by Inflect (Conv1D, ConvTranspose1D, attention
matmuls) and KittenTTS (bidirectional LSTM). Runs each converted model at two
lengths, with and without XNNPACK, against Keras eager.
"""
import numpy as np
import tensorflow as tf
import tf_keras as keras


def convert_dynamic(model):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    return conv.convert()


def run_tflite(flat, x, use_xnnpack=True):
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
    kwargs = {}
    if not use_xnnpack:
        kwargs["experimental_op_resolver_type"] = (
            OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it = Interpreter(model_content=flat, **kwargs)
    ind = it.get_input_details()[0]
    it.resize_tensor_input(ind["index"], list(x.shape))
    it.allocate_tensors()
    it.set_tensor(ind["index"], x)
    it.invoke()
    return it.get_tensor(it.get_output_details()[0]["index"])


def check(name, model, make_input):
    try:
        flat = convert_dynamic(model)
        print(f"{name}: converted {len(flat)/1e3:.0f} KB")
        for L in (24, 96):
            x = make_input(L)
            ref = model(x).numpy()
            for xnn in (True, False):
                out = run_tflite(flat, x, use_xnnpack=xnn)
                err = np.abs(out - ref).max()
                tag = "xnnpack" if xnn else "builtin"
                print(f"  {name} L={L} {tag}: maxerr={err:.2e}")
    except Exception as e:
        print(f"{name}: FAIL", " ".join(str(e).split())[:220])


def conv_stack():
    x = keras.Input(shape=(None, 8), batch_size=1)
    h = keras.layers.Conv1D(16, 5, padding="same", activation="relu")(x)
    h = keras.layers.Conv1D(8, 5, padding="same", dilation_rate=3,
                            activation="relu")(h)
    y = keras.layers.Conv1DTranspose(8, 16, strides=8, padding="same")(h)
    return keras.Model(x, y)


def attn():
    x = keras.Input(shape=(None, 72), batch_size=1)
    q = keras.layers.Dense(72)(x)
    k = keras.layers.Dense(72)(x)
    v = keras.layers.Dense(72)(x)

    def sdpa(t):
        q, k, v = t
        q = tf.transpose(tf.reshape(q, [1, -1, 2, 36]), [0, 2, 1, 3])
        k = tf.transpose(tf.reshape(k, [1, -1, 2, 36]), [0, 2, 1, 3])
        v = tf.transpose(tf.reshape(v, [1, -1, 2, 36]), [0, 2, 1, 3])
        p = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) / 6.0, axis=-1)
        o = tf.transpose(tf.matmul(p, v), [0, 2, 1, 3])
        return tf.reshape(o, [1, -1, 72])

    y = keras.layers.Lambda(sdpa)([q, k, v])
    return keras.Model(x, y)


def bilstm():
    x = keras.Input(shape=(None, 24), batch_size=1)
    y = keras.layers.Bidirectional(
        keras.layers.LSTM(32, return_sequences=True))(x)
    return keras.Model(x, y)


if __name__ == "__main__":
    np.random.seed(0)
    check("conv+convT", conv_stack(),
          lambda L: np.random.randn(1, L, 8).astype(np.float32))
    check("attention", attn(),
          lambda L: np.random.randn(1, L, 72).astype(np.float32))
    check("bilstm", bilstm(),
          lambda L: np.random.randn(1, L, 24).astype(np.float32))
