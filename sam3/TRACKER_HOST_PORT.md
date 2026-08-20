# SAM 3.1 Object-Multiplex tracker — host state machine port spec (2026-08-20)

`scripts/tracker_host_loop.py` is the **executable spec** for the Kotlin/Swift port: the
complete tracker orchestration re-implemented in numpy + seven LiteRT CompiledModel
graphs, with **zero dependency on torch or the official sam3 code at runtime**. This
document is the map of that file: every rule, constant, and graph contract, plus the
verified agreement numbers. Host constants are dumped by
`scripts/dump_tracker_host_assets.py` into `models/tracker_host/{consts.npz,flags.json}`.

## Verified agreement (vs the all-PyTorch reference `tracker_reference_cpu.py`)

| clip | mode | ids | min mask IoU | max \|Δprob\| | wall (M4 Max) |
|---|---|---|---|---|---|
| clip8 (8f, 4 obj) | graphs f32 (`enforce_f32`) | identical, all frames | 0.9924 | 0.0000 | 28.4 s |
| clip24 (24f, recondition @0/@16) | graphs f32 | identical, all frames | 0.9925 | 0.0000 | 71.8 s |
| clip8 | graphs fp16 (Mali reality) | identical | 0.9976 | 0.0018 | 21.7 s |
| clip24 | graphs fp16 | identical | 0.9936 | 0.0018 | 49.7 s |

(The residual IoU gap vs 1.0 is the fp16 *weights* of the .tflite files vs the fp32 torch
reference — identical to the gap the official-loop-with-graph-swaps milestone showed.)

Per-frame graph calls (B = bucket count, 1 for ≤16 objects): vision_tri ×1, head ×1,
memattn ×B, maskdec ×B, memenc ×B (+B on recondition frames), initdec ×(new objects).
Mac f32 per-frame: vision 800 ms, memattn 1.31 s, head 175 ms, memenc 16 ms, maskdec 6 ms;
fp16: vision 600 ms, memattn 640 ms, head 100 ms.

## Graphs and exact I/O packing (flat float32, single input / single output)

| graph | input layout | output layout |
|---|---|---|
| `sam3_vision_tri` | image (1,3,1008,1008), (x/255−0.5)/0.5, **computed in fp16 then upcast** | concat of 9 maps: sam3_fpn288 (256·288²), sam3_fpn144 (256·144²), sam3_fpn72 (256·72²), inter_h0 (32·288²), inter_h1 (64·144²), inter_f2 (256·72²), prop_h0 (32·288²), prop_h1 (64·144²), prop_f2 (256·72²) |
| `sam3_text` | token embeddings (1,32,1024) — host BPE (ctx 32, BOS 49406, EOT 49407, 0-pad) + fp16 table lookup | text_mem (32·256) |
| `sam3_head` | [sam3_fpn288 \| fpn144 \| fpn72 \| text_mem (32·256) \| pad (32, 1.0 where id==0)] | [logits (200) \| boxes cxcywh (200·4) \| presence (1) \| mask logits (200·288²)] — presence unused by the tracker |
| `trk_memattn_n7` | [pix (5184·256, prop_f2 flattened HW×C) \| mem_img (7·5184·256) \| mem_img_pos \| maskmem (7·5184·256) \| ptr (256·256) \| ptr_pos (256·256) \| keep (7·5184+256)] — zero-pad unused slots, keep=1 on valid keys | conditioned pix (5184·256) |
| `trk_maskdec` | [pix_with_mem (256·72²) \| prop_h0 \| prop_h1 \| extra_per_object_emb (16·256)] | [masks (16·3·288²) \| ious (16·3) \| obj_score_logits (16) \| tokens (16·3·256)] |
| `trk_memenc` | [prop_f2 (256·72²) \| 32ch mask stack @1008² (16 mux'd mask_for_mem + 16 mux'd condition channels)] | memory features (256·72²) |
| `trk_initdec` | [inter_f2 + interactivity_no_mem_embed (256·72²) \| inter_h0 \| inter_h1 \| sparse_const (2·256) \| dense (256·72²)] | [mask (288²) \| iou (1) \| token (256) \| obj_score_logit (1)] |

One call per bucket for memattn/maskdec/memenc; one call per object for initdec.

## Host-side constants (`consts.npz`)

`pos_72` (sine pos-enc of the 72×72 grid, used as image pos in memattn),
`memenc_pos_72` (memory pos-enc), `maskmem_tpos_enc` (7,256), `interactivity_no_mem_embed`
(256), `no_obj_embed_spatial` (16,256), `output_valid_embed`/`output_invalid_embed`
(16,256), `obj_ptr_tpos_proj` (Linear 256→256), `obj_ptr_proj` and
`interactive_obj_ptr_proj` (3-layer ReLU MLPs 256→256), `no_obj_ptr_linear` (Linear),
`interactive_mask_downsample` (Conv 1→1 k4 s4), `mask_downscaling` (Conv 1→4 k2s2, LN,
GELU, Conv 4→16 k2s2, LN, GELU, Conv 16→256 k1 — all stride==kernel ⇒ block reshapes),
`sparse_const` (2,256 — the prompt-encoder sparse embeddings are CONSTANT for the
mask-as-output flow: one (0,0) point with label −1 plus the pad point).

## Frame pipeline (per frame t)

1. **Load** frame: PIL bilinear resize to 1008², /255, **store fp16**, normalize
   (x−0.5)/0.5 in fp16, upcast to f32 for the graph (fp16 storage replicates the
   reference's image buffer).
2. **Detection**: head graph → 200 logits/boxes/masks.
   - probs = sigmoid(logits); `is_valid = probs > 0.4`.
   - **NMS** (quirk-exact): pairwise "IoM" where the denominator is the ROW mask's own
     area + 1e-8 (upstream perflib broadcast quirk — NOT true min-area IoM); vectorized
     keep loop over rows sorted by prob descending: a row above the 0.1 threshold
     suppresses every lower-scored overlapping row **even if that row was itself already
     suppressed**. Suppressed rows: logits −= 1e4.
   - re-threshold probs > 0.4; boundary suppression: drop dets whose box center is
     within 0.025 of any edge.
   - **Sort**: kept rows first. Order within the kept group = torch's unstable bool
     argsort in the reference; the port uses the deterministic rule
     "[lowest kept row, then remaining kept rows in descending row order]" — verified to
     match torch on every frame of both clips. (Only affects the ID assignment order of
     ≥2 new objects appearing in the same frame; a labeling artifact.)
3. **Tracker propagation** (per multiplex state, objects sorted by obj id):
   - Frames present in consolidated cond/non-cond storage are FETCHED, not re-inferred
     (frame 0 is re-processed by the propagation loop and fetches its consolidated entry).
   - **Memory bank**: ≤4 temporally closest cond frames (insertion order; only frame 0 is
     ever cond in this flow) with t_pos = t−t_cond, then non-cond frames t−6…t−1 with
     t_pos 1…6. Slot temporal embedding (tpos_v2): `maskmem_tpos_enc[6]` when t_pos≤0 or
     ≥7, else `maskmem_tpos_enc[7−t_pos−1]`. Memory features are stored
     **bfloat16-rounded** (round-to-nearest-even) — replicate or the numerics drift.
     mem_img = stored per-frame image features; mem_img_pos = pos_72 + tpos;
     maskmem_pos = memenc_pos_72 + tpos.
   - **Object pointers**: same cond frames first, then t_diff 1…min(num_frames,16)−1
     (skip missing frames). tpos = Linear(sine_pe(dist / (min(num_frames,16)−1))),
     repeated ×16 per frame slot. Note the normalizer depends on the CLIP LENGTH.
   - If NO spatial memory exists, skip the graph and use raw pix features (fallback).
   - **Mask decode**: maskdec per bucket; extra_per_object_emb = valid_mask·valid_embed +
     (1−valid)·invalid_embed. Demux by slot map. Where obj_score_logit ≤ 0 the 3 masks
     become −1024. Best mask = argmax of the 3 ious. obj_ptr = MLP(token of best mask);
     then ptr = λ·ptr + (1−λ)·no_obj_ptr_linear(ptr) with λ = (osl > 0).
4. **Association** (dets vs propagated 288² masks; det rows zeroed where not kept):
   TRUE min-area IoM matrix (this one IS min-area, from train/masks_ops).
   - track matched if any det IoM ≥ 0.5 (trk_assoc); unmatched = nonempty & !matched.
   - new det: score ≥ 0.65, kept, and matches NO track at IoM ≥ 0.1 (assoc).
   - ambiguity-zeroing for recondition: zero columns (tracks) matching >1 det at ≥0.5 and
     rows (dets) matching >1 track at ≥0.5; `im_mask` (det↔trk matches for hotstart
     bookkeeping) = zeroed metric ≥ 0.1; high-conf recondition pair: det score ≥ 0.8,
     kept, not-new, zeroed-metric max ≥ 0.5 → (argmax track → det).
5. **Hotstart** (vectorized replica): keep_alive += 1 if matched else −1, clamped
   [−4, 8]; unmatch_cnt += 1 when unmatched (never resets); pairwise overlap counter +=
   1 for track pairs matched to the same det. Remove a track if first_frame >
   t−15 AND (unmatch_cnt ≥ 8 OR overlap-with-earlier-track ≥ 8). The keep-alive
   "suppression" mask is computed but **never consumed** upstream (vestigial) — the
   suppressed set is always empty in this flow.
6. **Recondition** (t % 16 == 0, incl. the frame-0 re-process, when high-conf pairs
   exist and the track's sigmoid(osl) > 0.8): replace track-mask pixels where binary
   disagreement with the det mask (agreement keeps old logits); re-run the mask-as-output
   init (initdec) with det mask @1152 for the affected objects, merging into the stored
   frame entry (rows of pred_masks/osl/obj_ptr; conditioning set); then consolidate +
   re-encode memory (preflight). At cond frames (frame 0) the consolidated entry lands in
   non-cond storage and is then dropped by the cond/non-cond dedup — i.e. only the
   in-place row merge + the trk-mask pixel update survive (replicate exactly).
7. **Occlusion suppression**: pairwise IoU (matmul, union clamp ≥1) of the propagated
   binary masks ≥ 0.7 (upper triangle): the more-recently-occluded member of the pair is
   zeroed (−10), only if the other has been occluded before (last_occl > −1); removed
   tracks count as occluded at +∞. Track last_occluded ← t for empty or suppressed masks.
8. **Memory encoding** (every frame with tracks): masks → 1152² bilinear; pixel-argmax
   non-overlap + area-shrinkage suppression (mask whose area shrinks below 0.3× is fully
   zeroed to −10); osl = ±10 by area; mask_for_mem = sigmoid(logits)·2 − 1;
   mux into 16 channels + 16 condition channels (1.0 for objects in the frame's
   conditioning set, else 0); **resize the 32-channel stack 1152→1008 bilinear** (the
   graph takes 1008); memenc graph; add Σ_slots (1−is_obj)·no_obj_embed_spatial;
   store bf16-rounded into the frame's entry, with the frame's image features.
9. **New objects**: obj ids = max_obj_id+1…; if count would exceed 16, keep top scores.
   Det masks → 1152 bilinear > 0 → mask-as-output init: video-res mask (aa-resize >0.5)
   for output; low-res = aa(±10@1152 → 288); obj_ptr via initdec (dense embedding =
   mask_downscaling(interactive_mask_downsample(mask))); double no-obj-ptr blend (graph
   osl then mask-nonemptiness). Then preflight: consolidate per-object video-res masks
   (±1024, cross-suppressed between simultaneous objects) → 288 aa → 1008 → pixel-argmax
   non-overlap → memenc (conditioning = the new objects). Bucket assignment: fill
   existing buckets' padding slots, else new bucket of 16.
10. **Confirmation**: consecutive matched-detection counter per object (reset to 0 on a
    frame with no match); status CONFIRMED at ≥ 3.
11. **Outputs**: per-object video-res binary masks (bilinear >0) from the
    (suppression/recondition-modified) track masks + the new dets' masks; probs = the
    object's FIRST detection score (never updated); per-frame sam2 prob = sigmoid(osl).
12. **Hotstart yield + postprocess**: outputs are buffered 15 frames; when a frame is
    yielded, hide objects that are (a) removed (cumulative set snapshot at yield time),
    (b) unconfirmed at frame min(t+2, last) — then drop empty masks and apply the
    object-wise non-overlap constraint at video res using the per-frame sam2 probs
    (pixel goes to the highest-scoring claimant; losers lose those pixels).

## Traps found while porting (beyond the memory's list)

- **perflib is enabled by default on CPU** (`USE_PERFLIB` unset) — the reference's NMS
  used the row-area-IoM quirk and the suppressed-suppresses NMS loop. Port those, not
  textbook NMS.
- torch cat of bf16 memory + f32 pointers type-promotes: memories act bf16-rounded.
- The obj-ptr tpos normalizer `min(num_frames, 16) − 1` depends on clip length.
- The frame-0 output is computed TWICE (add_prompt, then propagate re-processes frame 0
  with the tracker state live — counters tick twice, recondition fires at frame 0).
- Hotstart's keep-alive suppression path is dead code upstream; suppressed set is empty.
- `_run_single_frame_inference` stores only a compact entry (no high-res masks) — the
  recondition merge on `pred_masks_high_res` is a silent no-op upstream.
- Frame images are stored fp16 and normalized in fp16 before the f32 upcast.

## Not exercised / simplified

- Object REMOVAL execution (`remove_objects`) is ported in simplified form (slot
  remapping preserves bucket positions) — no removal occurs in either verification clip.
- Multi-bucket (>16 objects), reverse tracking, point/box prompts, image-only input, and
  memory-selection are not ported (not reachable in this flow).
- `models/tracker_host/trace{8,24}/` + `scripts/_debug_trace_official.py` are disposable
  debug artifacts (stage-by-stage dumps of the official loop with graph swaps).

## Kotlin port (2026-08-20, device autotest — awaiting Pixel run)

Ported 1:1 from `tracker_host_loop.py` into the sam3 Android app (compile-checked,
`:app:compileDebugKotlin` green; NOT yet run on device — the Pixel was contended):

| file | role |
|---|---|
| `app/src/main/java/com/sam3/TrackerMath.kt` | numeric primitives: fp16 (RNE) / bf16 round, `interp_bilinear` (+AA triangle filter, weight cache), erf/GELU (NR erfc, \|err\|<1.2e-7), stride==kernel convs, LayerNorm2d, linears, sine PE, 64-bit packed mask bitsets (LSB-first, popcount IoU/IoM) |
| `app/src/main/java/com/sam3/TrackerConsts.kt` | loads `tracker/consts/*.bin` per manifest + `flags.json`; mlp3 / no-obj-ptr blend |
| `app/src/main/java/com/sam3/Sam3Tracker.kt` | the state machine: detection decode + quirk NMS, association, hotstart, occl suppression, recondition, memory bank + tpos, obj-ptr bank, mask-as-output init (host conv chain), multiplex state, per-frame outputs + hotstart-delayed postprocess |
| `app/src/main/java/com/sam3/TrackerAutotest.kt` | fixture comparison (ids / \|Δprob\| / video-res mask IoU via packed popcount), logcat tag `SAM3TRK`, writes `files/tracker_result.txt` |
| `scripts/dump_tracker_device_assets.py` | consts → raw f32 bins + manifest; clip copy; expected outputs via `tracker_host_loop.py --fp16 --dump-device` |
| `scripts/install_tracker_to_device.sh` | adb staging (bash -n checked, NOT executed) |

MainActivity boots into tracker mode when `files/tracker/expected/manifest.json`
exists (image mode untouched otherwise; `run-as com.sam3 rm -r files/tracker` to go
back). Graphs: vision_tri/head/memattn_n7/maskdec/memenc/initdec on
`Accelerator.GPU`, text on CPU — same split the Mac fp16 fixture run used.

Fixture format (`models/device_tracker/expected/`): per frame `f<i>_ids.bin` (<i4),
`f<i>_probs.bin` (<f4), `f<i>_masks.bin` (per object, row-major H×W, 1 bit/px,
LSB-first = `np.packbits(bitorder="little")`); `manifest.json` has frames/H/W/prompt.
Fixtures were regenerated from the fp16 (Mali-reality) host loop run: clip8 ids
identical to the reference, IoU ≥0.9976. After adding `--dump-device` the f32
verification was re-run: clip8 min IoU 0.9924, clip24 0.9925, ids identical — unchanged.

Device run (for later, from `sam3/`):
```
scripts/install_tracker_to_device.sh
adb logcat -c && adb shell am force-stop com.sam3 && adb shell am start -n com.sam3/.MainActivity
adb logcat -s SAM3TRK          # per-frame ids/IoU + per-graph ms; final verdict line
adb shell run-as com.sam3 cat files/tracker_result.txt
```

Deliberate deviations from the Python loop (all justified, none expected to move the
ids/IoU gate):
- **JPEG decode + frame resize**: Android `BitmapFactory` + a float triangle filter
  rounded to uint8 stands in for PIL's libjpeg + fixed-point BILINEAR (±1 LSB class
  differences; downstream of this the pipeline is arithmetic-exact ports).
- **erf** via the Numerical Recipes rational approximation (float-level error) instead
  of libm erf; AA-resize accumulates in plain f32 order instead of numpy einsum's
  pairwise order (~1e-7 relative).
- **preflight base resize skipped**: the aa-downsampled `pred_masks_video_res` base is
  provably dead (every row is overwritten by either the temp entry or the stored 288²
  row); the Kotlin builds `cons` from those directly.
- **stale bank entries pruned**: non-cond entries older than t−6 drop their
  `maskmem`/`image_features` (the bank cannot reach them; obj_ptr is kept for the
  16-frame pointer window). Pure memory relief, provably value-identical.
- memattn is pinned to the N=7 graph (zero-padded bank); the N-variant seam is the
  single call site in `memoryConditionedFeatures()`.

## Swift port (sam3/ios, 2026-08-20)

`ios/Sources/TrackerMath.swift` (330 LoC), `TrackerConsts.swift` (72),
`Sam3Tracker.swift` (~1180), `TrackerAutotest.swift` (135); wired into
`Sam3App.swift` (tracker autotest mode when `Documents/tracker/expected/
manifest.json` exists; `.gpu` graphs on iPhone). No UIKit in the tracker files —
the SAME sources compile into a macOS CLI harness (scratchpad `swifttrk/`,
all graphs `.cpu`; the mac CLiteRTLM dylib's GPU accelerator is WebGPU and
mis-dispatches these graphs).

Mac-CPU verification, clip8 "person" (f32 fixtures via
`tracker_host_loop.py --dump-device`):
- vs the PIL-decoded fixtures: ids identical 8/8 frames, min mask IoU **0.9789**,
  max |Δprob| 0.0070. The gap is the ImageIO-vs-PIL JPEG decoder (mean |diff|
  0.27, max 23 on frame 0); the python loop run on CG-decoded frames shows the
  same drop vs the torch reference (min IoU 0.9776).
- vs decode-matched fixtures (CG-decoded frames re-encoded as lossless PNG,
  fixtures regenerated by the python loop on them): ids identical, min mask IoU
  **0.9980**, max |Δprob| 0.0066 (residual = f32 accumulation-order class:
  numpy pairwise summation vs sequential loops in resize/matvec, surfacing
  through uint8 rounding ties).

Swift-specific notes:
- fp16 via native `Float16` (arm64 RNE ≡ numpy); bf16 round via bit math;
  erf via libm (closer to numpy than the Kotlin NR approximation).
- Kotlin's stable `sortedBy` became explicit (value, index) comparators —
  Swift's `sorted` is not stability-guaranteed.
- `trk_maskdec.tflite`'s declared input is 61,440 floats LARGER than the real
  packing (oversized export dummy: 16·16·256 instead of 16·256 for
  extra_per_object_embeddings; the tail is sliced off inside the graph). The
  python/Kotlin wrappers get away with short writes into zero-init buffers;
  the Swift wrapper zero-pads explicitly. ⚠ The Kotlin `writeFloat` short-write
  relies on the same undefined-but-zero tail behavior.
- The text encoder is loaded → run once → released BEFORE the GPU graphs
  compile (same lmkd lesson as Android; mac-CPU peak ~3.3 GB).
