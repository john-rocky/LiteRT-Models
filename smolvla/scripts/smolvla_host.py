"""Host-side (numpy) pieces of the SmolVLA LiteRT pipeline.

Everything here mirrors lerobot 0.6.1 (`modeling_smolvla.py`, `vla_utils.py`,
`flow_matching.py`, the tokenizer/normalizer processor steps) so that a Kotlin
port can follow it line by line:

  image  -> letterbox_512 (bilinear, align_corners=False, pad left/top with 0) -> x*2-1
  task   -> task + "\\n" -> SmolVLM2 tokenizer (max_length 48, right padding, truncation)
         -> embedding rows from embed_tokens_f16.bin (fp16 table, looked up here)
  state  -> (x - mean) / (std + eps) if the checkpoint has state stats -> zero-pad to 32
  masks  -> additive biases (0 visible / -30000 masked) from lerobot's make_att_2d_masks
  RoPE   -> cos/sin tables of positions (base 10 000, split-half)
  time   -> create_sinusoidal_pos_embedding(t, 720, 0.004, 4.0) in float64 -> float32
  loop   -> 10 Euler steps, dt = -0.1, t = 1.0 + step*dt
  output -> x[..., :action_dim] -> x * std + mean if the checkpoint has action stats
"""

import json
import math

import numpy as np

HIDDEN = 960
EXPERT_HIDDEN = 720
HEAD_DIM = 64
CHUNK = 50
MAX_DIM = 32
LANG_LEN = 48
IMG_TOKENS = 64
IMG_SIZE = 512
MASK_NEG = np.float32(-30000.0)
ROPE_BASE = 10_000
MIN_PERIOD = 0.004
MAX_PERIOD = 4.0
NUM_STEPS = 10
TOKENIZER_REPO = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"


# ------------------------------------------------------------- graph files
GRAPH_BASENAMES = {"vision": "smolvla_vision", "prefix": "smolvla_prefix",
                   "expert": "smolvla_expert_step"}
FP16_SAFE_STAGES = ("vision", "prefix")   # the graphs whose norms see |x| > 255.9
DEFAULT_NORMS = "fp16safe"


def graph_file(stage, num_cameras=1, fp16_weights=False, norms=DEFAULT_NORMS):
    """File name of a graph variant.

    norms="fp16safe" (the default) names the vision/prefix graphs whose RMSNorm/LayerNorm
    use the exact down-scaled forms. norms="exact" names the opt-in reference graphs with
    the unmodified norm formulas (suffix _exactnorm); they overflow at fp16 GPU precision.
    The expert graph has a single variant.
    """
    base = GRAPH_BASENAMES[stage]
    if stage != "vision" and num_cameras != 1:
        base += f"_cam{num_cameras}"
    if norms == "exact" and stage in FP16_SAFE_STAGES:
        base += "_exactnorm"
    return base + ("_f16" if fp16_weights else "") + ".tflite"


# -------------------------------------------------------------------- image
def load_image_rgb01(path):
    """RGB image as float32 [3, H, W] in [0, 1] (uint8 / 255)."""
    from PIL import Image
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / np.float32(255.0)
    return np.ascontiguousarray(img.transpose(2, 0, 1))


def _linear_indices(in_size, out_size):
    """PyTorch upsample_bilinear2d (align_corners=False) source indices/weights."""
    scale = np.float32(in_size) / np.float32(out_size)
    dst = np.arange(out_size, dtype=np.float32)
    src = scale * (dst + np.float32(0.5)) - np.float32(0.5)
    src = np.maximum(src, np.float32(0.0))
    i0 = np.minimum(np.floor(src).astype(np.int64), in_size - 1)
    lam1 = np.clip(src - i0.astype(np.float32), np.float32(0.0), np.float32(1.0)).astype(np.float32)
    lam0 = (np.float32(1.0) - lam1).astype(np.float32)
    i1 = np.where(i0 < in_size - 1, i0 + 1, i0)
    return i0, i1, lam0, lam1


def resize_bilinear(img, out_h, out_w):
    """F.interpolate(mode='bilinear', align_corners=False, antialias=False), CHW float32."""
    _, h, w = img.shape
    h0, h1, lh0, lh1 = _linear_indices(h, out_h)
    w0, w1, lw0, lw1 = _linear_indices(w, out_w)
    rows0 = img[:, h0, :]
    rows1 = img[:, h1, :]
    top = rows0[:, :, w0] * lw0 + rows0[:, :, w1] * lw1
    bot = rows1[:, :, w0] * lw0 + rows1[:, :, w1] * lw1
    return (top * lh0[:, None] + bot * lh1[:, None]).astype(np.float32)


def letterbox_512(img01, size=IMG_SIZE):
    """lerobot resize_with_pad(img, 512, 512, pad_value=0) then img*2-1 -> [1,3,512,512]."""
    _, h, w = img01.shape
    if (h, w) != (size, size):
        ratio = max(w / size, h / size)
        rh, rw = int(h / ratio), int(w / ratio)
        resized = resize_bilinear(img01, rh, rw)
        out = np.zeros((3, size, size), np.float32)
        out[:, size - rh:, size - rw:] = resized        # pad LEFT and TOP
    else:
        out = img01.astype(np.float32)
    return (out * np.float32(2.0) - np.float32(1.0))[None]


# ------------------------------------------------------------------ language
def load_tokenizer(repo=TOKENIZER_REPO):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(repo)


def tokenize(tokenizer, task, max_length=LANG_LEN):
    """lerobot NewLineTaskProcessorStep + TokenizerProcessorStep -> ids[48], mask[48]."""
    if not task.endswith("\n"):
        task = task + "\n"
    enc = tokenizer(task, max_length=max_length, truncation=True, padding="max_length",
                    padding_side="right", return_tensors="np")
    return enc["input_ids"][0].astype(np.int64), enc["attention_mask"][0].astype(bool)


def load_embed_table(path):
    return np.fromfile(path, dtype="<f2").reshape(-1, HIDDEN)


def embed_lookup(table_f16, ids):
    """Raw (unscaled) embedding rows [1, 48, 960] float32."""
    return table_f16[ids].astype(np.float32)[None]


# --------------------------------------------------------- masks / positions
def make_att_2d_masks(pad, att):
    """lerobot make_att_2d_masks for one sequence (bool [N], int [N]) -> bool [N, N]."""
    cum = np.cumsum(att.astype(np.int64))
    att_2d = cum[None, :] <= cum[:, None]
    pad_2d = pad[None, :] & pad[:, None]
    return att_2d & pad_2d


def to_bias(mask):
    return np.where(mask, np.float32(0.0), MASK_NEG).astype(np.float32)


class PrefixLayout:
    """Pad/att masks, bias and positions for [images | language | state]."""

    def __init__(self, lang_mask, num_cameras=1):
        n_img = IMG_TOKENS * num_cameras
        self.pad = np.concatenate([np.ones(n_img, bool), lang_mask.astype(bool), np.ones(1, bool)])
        self.att = np.concatenate([np.zeros(n_img + LANG_LEN, np.int64), np.ones(1, np.int64)])
        self.length = self.pad.size
        self.mask_2d = make_att_2d_masks(self.pad, self.att)
        self.positions = np.cumsum(self.pad.astype(np.int64)) - 1
        self.offset = int(self.pad.sum())        # prefix_offsets in denoise_step

    def attn_bias(self):
        return to_bias(self.mask_2d)[None, None]                      # [1,1,S,S]

    def expert_biases(self):
        prefix_cols = np.broadcast_to(self.pad[None, :], (CHUNK, self.length))
        suffix = make_att_2d_masks(np.ones(CHUNK, bool), np.ones(CHUNK, np.int64))  # causal
        full = np.concatenate([prefix_cols, suffix], axis=1)
        bias_self = to_bias(full)[None, None]                          # [1,1,50,S+50]
        bias_cross = to_bias(prefix_cols)[None, None]                  # [1,1,50,S]
        return bias_self, bias_cross

    def valid_index(self):
        return np.nonzero(self.pad)[0]


def rope_tables(positions):
    """lerobot apply_rope tables: radians = pos / 10000**(2i/64) in float32 -> [1,1,N,32]."""
    freq_exp = np.float32(2.0 / HEAD_DIM) * np.arange(HEAD_DIM // 2, dtype=np.float32)
    timescale = np.power(np.float64(ROPE_BASE), freq_exp.astype(np.float64)).astype(np.float32)
    radians = np.asarray(positions, np.float32)[:, None] / timescale[None, :]
    rad64 = radians.astype(np.float64)
    cos = np.cos(rad64).astype(np.float32)
    sin = np.sin(rad64).astype(np.float32)
    return cos[None, None], sin[None, None]


# ---------------------------------------------------------------- time/loop
def time_embedding(t, dim=EXPERT_HIDDEN, min_period=MIN_PERIOD, max_period=MAX_PERIOD):
    """create_sinusoidal_pos_embedding for a float32 timestep t -> [1,1,dim] float32."""
    fraction = np.linspace(0.0, 1.0, dim // 2, dtype=np.float64)
    period = min_period * (max_period / min_period) ** fraction
    scaling = 1.0 / period * 2 * math.pi
    sin_input = scaling * np.float64(np.float32(t))
    return np.concatenate([np.sin(sin_input), np.cos(sin_input)]).astype(np.float32)[None, None]


def euler_times(num_steps=NUM_STEPS):
    dt = -1.0 / num_steps
    return [np.float32(1.0 + step * dt) for step in range(num_steps)], np.float32(dt)


def euler_integrate(velocity_fn, noise, num_steps=NUM_STEPS):
    """lerobot euler_integrate without RTC: x <- x + dt * v(x, t), t = 1.0 + step*dt."""
    times, dt = euler_times(num_steps)
    x = noise.astype(np.float32)
    trace = []
    for step, t in enumerate(times):
        v = velocity_fn(x, time_embedding(t), step)
        x = (x + dt * v).astype(np.float32)
        trace.append((x.copy(), v.copy()))
    return x, trace


# ----------------------------------------------------------- normalization
class NormStats:
    """MEAN_STD (un)normalization exactly as lerobot's (Un)NormalizerProcessorStep.

    A missing entry means identity, which is what lerobot does when the processor
    state file has no matching key (the case for lerobot/smolvla_base).
    """

    def __init__(self, state=None, action=None, eps=1e-8):
        self.state = state      # dict(mean=[...], std=[...]) or None
        self.action = action
        self.eps = eps

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            d = json.load(f)
        modes = d.get("norm_map") or {}
        for key, feature in (("state", "STATE"), ("action", "ACTION")):
            if d.get(key) is not None and modes.get(feature, "MEAN_STD") != "MEAN_STD":
                raise NotImplementedError(
                    f"{feature} uses {modes[feature]}; only MEAN_STD is implemented here")
        return cls(d.get("state"), d.get("action"), d.get("eps") or 1e-8)

    def normalize_state(self, state):
        s = np.asarray(state, np.float32)
        if self.state is None:
            return s
        mean = np.asarray(self.state["mean"], np.float32)
        std = np.asarray(self.state["std"], np.float32)
        return ((s - mean) / (std + np.float32(self.eps))).astype(np.float32)

    def unnormalize_action(self, action):
        a = np.asarray(action, np.float32)
        if self.action is None:
            return a
        mean = np.asarray(self.action["mean"], np.float32)
        std = np.asarray(self.action["std"], np.float32)
        return (a * std + mean).astype(np.float32)


def pad_state(state, dim=MAX_DIM):
    s = np.zeros((1, dim), np.float32)
    s[0, :len(state)] = state
    return s


# --------------------------------------------------------------- pipeline
class SmolVLAHost:
    """Drives the three graphs. vision/prefix/expert are callables on numpy arrays:

    vision(image, pos_embed) -> img_emb [1,64,960]
    prefix(img_emb, lang_emb, state, attn_bias, rope_cos, rope_sin) -> (k_all, v_all)
    expert(x_t, time_emb, k_all, v_all, bias_self, bias_cross,
           rope_cos_s, rope_sin_s, rope_cos_c, rope_sin_c) -> v_t
    """

    def __init__(self, vision, prefix, expert, embed_table, pos_embed, tokenizer,
                 stats=None, num_cameras=1, action_dim=6):
        self.vision = vision
        self.prefix = prefix
        self.expert = expert
        self.embed_table = embed_table
        self.pos_embed = pos_embed
        self.tokenizer = tokenizer
        self.stats = stats or NormStats()
        self.num_cameras = num_cameras
        self.action_dim = action_dim

    def prepare(self, images01, task, state):
        assert len(images01) == self.num_cameras
        ids, lang_mask = tokenize(self.tokenizer, task)
        layout = PrefixLayout(lang_mask, self.num_cameras)
        cos, sin = rope_tables(layout.positions)
        cos_s, sin_s = rope_tables(layout.offset + np.arange(CHUNK))
        cos_c, sin_c = rope_tables(np.arange(CHUNK))
        bias_self, bias_cross = layout.expert_biases()
        return {
            "images": [letterbox_512(im) for im in images01],
            "ids": ids, "lang_mask": lang_mask, "layout": layout,
            "lang_emb": embed_lookup(self.embed_table, ids),
            "state": pad_state(self.stats.normalize_state(state)),
            "attn_bias": layout.attn_bias(), "rope_cos": cos, "rope_sin": sin,
            "bias_self": bias_self, "bias_cross": bias_cross,
            "rope_cos_s": cos_s, "rope_sin_s": sin_s, "rope_cos_c": cos_c, "rope_sin_c": sin_c,
        }

    def run(self, images01, task, state, noise):
        p = self.prepare(images01, task, state)
        img_emb = np.concatenate([self.vision(im, self.pos_embed) for im in p["images"]], axis=1)
        k_all, v_all = self.prefix(img_emb, p["lang_emb"], p["state"], p["attn_bias"],
                                   p["rope_cos"], p["rope_sin"])

        def velocity(x, temb, step):
            return self.expert(x, temb, k_all, v_all, p["bias_self"], p["bias_cross"],
                               p["rope_cos_s"], p["rope_sin_s"], p["rope_cos_c"], p["rope_sin_c"])

        x, trace = euler_integrate(velocity, noise)
        actions = self.stats.unnormalize_action(x[..., :self.action_dim])
        return {"prepared": p, "img_emb": img_emb, "k_all": k_all, "v_all": v_all,
                "x": x, "trace": trace, "actions": actions}
