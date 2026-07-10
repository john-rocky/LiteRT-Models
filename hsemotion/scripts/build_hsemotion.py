"""HSEmotion (EfficientNet-B0, AffectNet-8) -> LiteRT CompiledModel GPU.

Facial emotion recognition: a cropped face -> 8 emotion logits. The HSEmotion
weights are an old-timm pickle whose forward is broken under new timm; we lift
its state_dict into a fresh timm tf_efficientnet_b0 (num_classes=8, classifier
key remapped) which has a working forward, then convert with litert-torch.

Emotions (index order): Anger, Contempt, Disgust, Fear, Happiness, Neutral,
Sadness, Surprise.

Run:  python build_hsemotion.py [parity|convert|fp16|device|all]
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "enet_b0_8_best_afew.pt")
FP32 = os.path.join(HERE, "hsemotion_b0.tflite")
FP16 = os.path.join(HERE, "hsemotion_b0_fp16.tflite")
SIZE = 224
EMOTIONS = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral",
            "Sadness", "Surprise"]
IN_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IN_STD = np.array([0.229, 0.224, 0.225], np.float32)


def safe_global_mean(x):
    """Exact global spatial mean via small avg_pool stages (fp16-safe).

    The delegate computes a single mean((2,3)) over a 112x112 map as one fp16
    reduction whose partial sum overflows 65504 -> NaN. Hierarchical averaging
    over equal-size tiling windows (each <= 49 elements) is mathematically
    identical but keeps every accumulation small.
    """
    while x.shape[2] > 1 or x.shape[3] > 1:
        kh = 2 if x.shape[2] > 1 and x.shape[2] % 2 == 0 else x.shape[2]
        kw = 2 if x.shape[3] > 1 and x.shape[3] % 2 == 0 else x.shape[3]
        x = F.avg_pool2d(x, kernel_size=(kh, kw))
    return x


def _safe_se_forward(self, x):
    """SqueezeExcite forward with an fp16-safe global mean."""
    x_se = safe_global_mean(x)
    x_se = self.conv_reduce(x_se)
    x_se = self.act1(x_se)
    x_se = self.conv_expand(x_se)
    return x * self.gate(x_se)


def build_model():
    """Rebuilds HSEmotion EfficientNet-B0 in current timm from its weights."""
    import timm
    from timm.models._efficientnet_blocks import SqueezeExcite
    SqueezeExcite.forward = _safe_se_forward
    pickle = torch.load(CKPT, map_location="cpu", weights_only=False)
    src = pickle.state_dict()
    model = timm.create_model("tf_efficientnet_b0", pretrained=False,
                              num_classes=8)
    dst = model.state_dict()
    remap = {}
    for k, v in src.items():
        key = k.replace("classifier.0.", "classifier.")
        if key in dst and dst[key].shape == v.shape:
            remap[key] = v
    missing = [k for k in dst if k not in remap]
    assert not missing, "unmapped keys: %s" % missing[:4]
    model.load_state_dict(remap)
    return model.eval()


def preprocess(path):
    """Loads a face image -> ImageNet-normalized NCHW tensor [1,3,224,224]."""
    im = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    arr = (np.asarray(im, np.float32) / 255.0 - IN_MEAN) / IN_STD
    return torch.from_numpy(arr.transpose(2, 0, 1)[None])


def stage_parity(model):
    """Predicts emotions on the test face; happy face -> Happiness."""
    x = preprocess(os.path.join(HERE, "happy.jpg"))
    with torch.no_grad():
        probs = torch.softmax(model(x)[0], -1)
    order = torch.argsort(probs, descending=True)
    print("happy.jpg emotions:")
    for i in order[:4]:
        print("   %5.1f%%  %s" % (probs[i] * 100, EMOTIONS[i]))


def stage_convert(model):
    """litert-torch conversion of the classifier (fp32 flatbuffer)."""
    ex = (torch.zeros(1, 3, SIZE, SIZE),)
    import litert_torch
    litert_torch.convert(model, ex).export(FP32)
    print("convert: %.1f MB -> %s" % (os.path.getsize(FP32) / 1e6, FP32))


def stage_fp16():
    """fp32 -> fp16 flatbuffer via ai_edge_quantizer FLOAT_CASTING."""
    from ai_edge_quantizer import quantizer, recipe_manager
    from ai_edge_quantizer.recipe import AlgorithmName, qtyping
    rm = recipe_manager.RecipeManager()
    rm.add_quantization_config(
        regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=qtyping.TensorQuantizationConfig(
                num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
            compute_precision=qtyping.ComputePrecision.FLOAT),
        algorithm_key=AlgorithmName.FLOAT_CASTING)
    if os.path.exists(FP16):
        os.remove(FP16)
    qt = quantizer.Quantizer(float_model=FP32)
    qt.load_quantization_recipe(rm.get_quantization_recipe())
    qt.quantize().export_model(FP16)
    print("fp16: %.1f MB -> %s" % (os.path.getsize(FP16) / 1e6, FP16))


def stage_device(model):
    """Runs the fp16 tflite on the Pixel 8a GPU and checks top-1 vs fp32."""
    import subprocess
    dev = "/data/local/tmp"
    x = preprocess(os.path.join(HERE, "happy.jpg"))
    with torch.no_grad():
        ref = model(x)[0].numpy()
    x.numpy().astype(np.float32).tofile(os.path.join(HERE, "hin.bin"))
    for f in ("hin.bin", "hsemotion_b0_fp16.tflite"):
        subprocess.run(["adb", "push", os.path.join(HERE, f), "%s/%s" % (dev, f)],
                       capture_output=True)
    r = subprocess.run(
        ["adb", "shell", "cd %s && LD_LIBRARY_PATH=. ./gpu_test_bin "
         "hsemotion_b0_fp16.tflite 5 hin.bin hout.bin" % dev],
        capture_output=True, text=True)
    print([l for l in r.stderr.splitlines() if "RUN OK" in l or "Replacing" in l])
    subprocess.run(["adb", "pull", "%s/hout.bin.0" % dev,
                    os.path.join(HERE, "hout.bin.0")], capture_output=True)
    dev_logits = np.fromfile(os.path.join(HERE, "hout.bin.0"), np.float32)
    corr = np.corrcoef(dev_logits, ref)[0, 1]
    print("device fp16 top-1: %s   fp32 top-1: %s   logits corr %.5f" %
          (EMOTIONS[int(dev_logits.argmax())], EMOTIONS[int(ref.argmax())], corr))
    print("top-1 match:", "YES" if dev_logits.argmax() == ref.argmax() else "NO")


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    model = build_model()
    if stage in ("parity", "all"):
        stage_parity(model)
    if stage in ("convert", "all"):
        stage_convert(model)
    if stage in ("fp16", "all"):
        stage_fp16()
    if stage in ("device", "all"):
        stage_device(model)


if __name__ == "__main__":
    main()
