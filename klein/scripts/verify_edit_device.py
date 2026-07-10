"""Run one image-editing DiT step on the device, graph by graph, and score it.

Drives the eight `kce_*` chunks over adb in the order the app will, doing the one
host-side step the graphs do not cover (the `cat([encoder, hidden])` that joins
the text and image streams before the single-stream blocks). The result is
compared against the fp32 torch DiT on identical inputs.

This is the editing analogue of `gen_verify_klein.py`: if the device output
matches here, the remaining work is host wiring, not numerics.

Usage:
    python verify_edit_device.py            # editing graphs (kce_*), S=1024
    python verify_edit_device.py --t2i      # the shipped graphs (kc_*), S=768
    python verify_edit_device.py --no-push  # graphs already on the device
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from build_klein_dit import CTX_DIM, IN_CH, ExportKleinDiT
from build_klein_real import N_TXT, REPO, TOKEN_GRID, edit_ids, latent_ids
from chunked_export_klein import DIM, DOUBLE_SPLIT, SINGLE_CHUNK

DEVICE_DIR = "/data/local/tmp/klein_edit"
RUNNER = os.path.expanduser(
    "~/Downloads/litert-upstream/midas-gpu-test/build/midas_gpu_test")
RUNNER_LIBS = os.path.expanduser(
    "~/Downloads/litert-upstream/midas-gpu-test/libs")
SEED = 0
TOKENS_PER_IMAGE = TOKEN_GRID * TOKEN_GRID


def adb(command):
    return subprocess.run(["adb", "shell", command], check=True,
                          capture_output=True, text=True).stdout


def push(local, remote):
    subprocess.run(["adb", "push", local, remote], check=True,
                   capture_output=True)


def pull(remote, local):
    subprocess.run(["adb", "pull", remote, local], check=True,
                   capture_output=True)


def run_graph(name, inputs, num_outputs):
    """Runs one chunk on the device GPU and returns its outputs as numpy arrays.

    Args:
        name: Graph basename, e.g. "kce_prep".
        inputs: Tensors in the module's forward order.
        num_outputs: How many output buffers the graph has.

    Returns:
        A list of flat float32 arrays, one per output.
    """
    adb(f"rm -f {DEVICE_DIR}/in.* {DEVICE_DIR}/out.bin.*")
    for index, tensor in enumerate(inputs):
        path = os.path.join(_SCRIPT_DIR, f"_in{index}.bin")
        tensor.detach().numpy().astype("<f4").tofile(path)
        push(path, f"{DEVICE_DIR}/in.{index}")
        os.remove(path)

    log = adb(f"cd {DEVICE_DIR} && FP32=1 LD_LIBRARY_PATH=. "
              f"./runner {name}.tflite 1 in out.bin 2>&1")
    delegated = [line for line in log.splitlines() if "Replacing" in line]
    print(f"  [{name}] {delegated[0].strip() if delegated else 'NO DELEGATE LINE'}")

    outputs = []
    for index in range(num_outputs):
        local = os.path.join(_SCRIPT_DIR, f"_out{index}.bin")
        pull(f"{DEVICE_DIR}/out.bin.{index}", local)
        outputs.append(np.fromfile(local, dtype=np.float32))
        os.remove(local)
    return outputs


def as_tensor(flat, shape):
    return torch.from_numpy(flat.reshape(shape).copy())


def score(name, reference, actual):
    reference = reference.detach().numpy().ravel()
    actual = actual.ravel()
    correlation = np.corrcoef(reference, actual)[0, 1]
    print(f"  {name:28s} corr {correlation:.6f}  "
          f"max|diff| {np.abs(reference - actual).max():.4f}  "
          f"NaN={bool(np.isnan(actual).any())}")
    return correlation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--t2i", action="store_true",
                        help="score the shipped text-to-image graphs instead")
    args = parser.parse_args()

    from diffusers import Flux2Transformer2DModel

    # Editing doubles the image tokens; text-to-image is the `image=None` case.
    prefix = "kc" if args.t2i else "kce"
    s_img = TOKENS_PER_IMAGE if args.t2i else 2 * TOKENS_PER_IMAGE
    seq = N_TXT + s_img
    image_ids = (latent_ids if args.t2i else edit_ids)(TOKEN_GRID)
    print(f"[mode] {'text-to-image' if args.t2i else 'image editing'}: "
          f"s_img={s_img}, joint sequence={seq}, graphs {prefix}_*")

    torch.manual_seed(SEED)
    print("[load] real transformer (fp32) ...")
    model = Flux2Transformer2DModel.from_pretrained(
        REPO, subfolder="transformer", torch_dtype=torch.float32).eval()
    dit = ExportKleinDiT(model, N_TXT).eval()

    img_tokens = torch.randn(1, s_img, IN_CH) * 0.5
    enc_hidden = torch.randn(1, N_TXT, CTX_DIM) * 0.5
    timestep = torch.tensor([0.5])
    with torch.no_grad():
        temb = model.time_guidance_embed(timestep * 1000, None)
        img_cos, img_sin = model.pos_embed(image_ids)
        txt_cos, txt_sin = model.pos_embed(torch.zeros(N_TXT, 4))
        cos = torch.cat([txt_cos, img_cos], dim=0)[:, 0::2][None, :, None, :]
        sin = torch.cat([txt_sin, img_sin], dim=0)[:, 0::2][None, :, None, :]
        expected = dit(img_tokens, enc_hidden, temb, cos, sin)
    del model, dit
    print(f"[host] fp32 reference {tuple(expected.shape)}")

    graphs = ([f"{prefix}_double{i}" for i in range(len(DOUBLE_SPLIT))]
              + [f"{prefix}_single{i}" for i in range(20 // SINGLE_CHUNK)])
    adb(f"mkdir -p {DEVICE_DIR}")
    if not args.no_push:
        push(RUNNER, f"{DEVICE_DIR}/runner")
        for lib in os.listdir(RUNNER_LIBS):
            if lib.endswith(".so"):
                push(os.path.join(RUNNER_LIBS, lib), f"{DEVICE_DIR}/")
        adb(f"chmod +x {DEVICE_DIR}/runner")
        for name in [f"{prefix}_prep", f"{prefix}_final"] + graphs:
            print(f"[push] {name}.tflite")
            push(os.path.join(_SCRIPT_DIR, f"{name}.tflite"), f"{DEVICE_DIR}/")

    print("\n[device] sequential residency, FP32, one graph at a time")
    prep = run_graph(f"{prefix}_prep", (img_tokens, enc_hidden, temb), 5)
    hidden = as_tensor(prep[0], (1, s_img, DIM))
    encoder = as_tensor(prep[1], (1, N_TXT, DIM))
    mod_i = as_tensor(prep[2], (1, 1, 6 * DIM))
    mod_t = as_tensor(prep[3], (1, 1, 6 * DIM))
    mod_s = as_tensor(prep[4], (1, 1, 3 * DIM))

    for index in range(len(DOUBLE_SPLIT)):
        out = run_graph(f"{prefix}_double{index}",
                        (hidden, encoder, cos, sin, mod_i, mod_t), 2)
        hidden = as_tensor(out[0], (1, s_img, DIM))
        encoder = as_tensor(out[1], (1, N_TXT, DIM))

    joint = torch.cat([encoder, hidden], dim=1)  # host-side, as the app does
    for index in range(20 // SINGLE_CHUNK):
        out = run_graph(f"{prefix}_single{index}", (joint, cos, sin, mod_s), 1)
        joint = as_tensor(out[0], (1, seq, DIM))

    patches = as_tensor(run_graph(f"{prefix}_final", (joint, temb), 1)[0],
                        (1, s_img, IN_CH))

    print("\n[score] device int8 GPU vs host fp32 torch")
    score("full DiT output", expected, patches.numpy())
    # The pipeline keeps only the noise tokens: noise_pred[:, :latents.size(1)]
    score("noise tokens (what is used)", expected[:, :TOKENS_PER_IMAGE],
          patches[:, :TOKENS_PER_IMAGE].numpy())


if __name__ == "__main__":
    main()
