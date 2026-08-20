#!/usr/bin/env python3
"""One-time dumper: every host-side constant the LiteRT-only tracker host loop
(tracker_host_loop.py) needs, extracted from the OFFICIAL sam3.1 multiplex model into
sam3/models/tracker_host/ as .npy/.json. This script may import torch + vendor_sam3;
the host loop may not.

Usage: dump_tracker_host_assets.py [--ckpt models/sam3.1_multiplex.pt]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import precheck_sam3 as P  # noqa: E402,F401  (triton stub etc.)
from tracker_reference_cpu import cpu_stubs  # noqa: E402

OUT = os.path.join(ROOT, "models", "tracker_host")


def np32(t):
    return t.detach().cpu().float().numpy().astype(np.float32)


def dump_linear(d, name, lin):
    d[f"{name}.w"] = np32(lin.weight)
    d[f"{name}.b"] = np32(lin.bias)


def dump_mlp(d, name, mlp):
    for i, lin in enumerate(mlp.layers):
        dump_linear(d, f"{name}.{i}", lin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "models", "sam3.1_multiplex.pt"))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cpu_stubs()
    from sam3 import model_builder as mb
    pred = mb.build_sam3_multiplex_video_predictor(checkpoint_path=a.ckpt, use_fa3=False,
                                                   use_rope_real=True, compile=False,
                                                   async_loading_frames=False)
    model = pred.model            # Sam3MultiplexTrackingWithInteractivity
    model.to("cpu")
    trk = model.tracker.model     # Sam3VideoTrackingMultiplexDemo

    d = {}
    # --- position encodings (constant across frames) ---
    with torch.no_grad():
        pe = model.detector.backbone.vision_backbone.position_encoding
        d["pos_72"] = np32(pe(torch.zeros(1, 256, 72, 72)))            # (1,256,72,72)
        d["memenc_pos_72"] = np32(trk.maskmem_backbone.position_encoding(
            torch.zeros(1, 256, 72, 72)))                              # (1,256,72,72)
        # propagation dense PE is baked into trk_maskdec; interactive dense PE baked into
        # trk_initdec -- not dumped.

    # --- temporal / object-pointer constants ---
    d["maskmem_tpos_enc"] = np32(trk.maskmem_tpos_enc)                 # (7,1,1,256)
    d["interactivity_no_mem_embed"] = np32(trk.interactivity_no_mem_embed)  # (1,1,256)
    d["no_obj_embed_spatial"] = np32(trk.no_obj_embed_spatial)         # (16,256)
    d["output_valid_embed"] = np32(trk.output_valid_embed)             # (16,256)
    d["output_invalid_embed"] = np32(trk.output_invalid_embed)         # (16,256)
    dump_linear(d, "obj_ptr_tpos_proj", trk.obj_ptr_tpos_proj)
    dump_mlp(d, "obj_ptr_proj", trk.obj_ptr_proj)
    dump_mlp(d, "interactive_obj_ptr_proj", trk.interactive_obj_ptr_proj)
    dump_linear(d, "no_obj_ptr_linear", trk.no_obj_ptr_linear)

    # --- interactive prompt path (mask-as-output init) ---
    dump_linear(d, "interactive_mask_downsample", trk.interactive_mask_downsample)  # conv k4 s4
    pe_enc = trk.interactive_sam_prompt_encoder
    md = pe_enc.mask_downscaling  # Conv(1,4,k2,s2), LN, GELU, Conv(4,16,k2,s2), LN, GELU, Conv(16,256,k1)
    for i, m in enumerate(md):
        if hasattr(m, "weight") and m.weight is not None:
            d[f"mask_downscaling.{i}.w"] = np32(m.weight)
            d[f"mask_downscaling.{i}.b"] = np32(m.bias)
    # constant sparse embeddings for the mask-as-output call: one (0,0) point with label -1,
    # padded with another (0,0)/-1 point (boxes=None). Constant for every object.
    with torch.no_grad():
        sparse, dense_nomask = pe_enc(
            points=(torch.zeros(1, 1, 2), -torch.ones(1, 1, dtype=torch.int32)),
            boxes=None, masks=None)
    d["sparse_const"] = np32(sparse[0])                                # (2,256)
    d["no_mask_embed"] = np32(pe_enc.no_mask_embed.weight)             # (1,256) unused but cheap

    # --- flags / scalars the host loop must honor ---
    flags = dict(
        num_maskmem=trk.num_maskmem,
        max_obj_ptrs_in_encoder=trk.max_obj_ptrs_in_encoder,
        use_signed_tpos_enc_to_obj_ptrs=trk.use_signed_tpos_enc_to_obj_ptrs,
        only_obj_ptrs_in_the_past_for_eval=trk.only_obj_ptrs_in_the_past_for_eval,
        proj_tpos_enc_in_obj_ptrs=trk.proj_tpos_enc_in_obj_ptrs,
        use_maskmem_tpos_v2=trk.use_maskmem_tpos_v2,
        max_cond_frames_in_attn=trk.max_cond_frames_in_attn,
        keep_first_cond_frame=trk.keep_first_cond_frame,
        memory_temporal_stride_for_eval=trk.memory_temporal_stride_for_eval,
        sigmoid_scale_for_mem_enc=trk.sigmoid_scale_for_mem_enc,
        sigmoid_bias_for_mem_enc=trk.sigmoid_bias_for_mem_enc,
        binarize_mask_from_pts_for_mem_enc=trk.binarize_mask_from_pts_for_mem_enc,
        object_score_logit_threshold=trk.object_score_logit_threshold,
        low_res_mask_size=trk.low_res_mask_size,
        input_mask_size=trk.input_mask_size,
        image_size=trk.image_size,
        interpol_size=list(trk.maskmem_backbone.mask_downsampler.interpol_size),
        multiplex_count=trk.multiplex_count,
        condition_as_mask_input=trk.condition_as_mask_input,
        condition_fg=trk.condition_as_mask_input_fg,
        condition_bg=trk.condition_as_mask_input_bg,
        add_output_suppression_embeddings=trk.add_output_suppression_embeddings,
        add_object_conditional_embeddings=trk.add_object_conditional_embeddings,
        decode_mask_with_shared_tokens=trk.decode_mask_with_shared_tokens,
        stability_score_attentuation=trk.stability_score_attentuation,
        multimask_output_in_sam=trk.multimask_output_in_sam,
        multimask_output_for_tracking=trk.multimask_output_for_tracking,
        multimask_min_pt_num=trk.multimask_min_pt_num,
        multimask_max_pt_num=trk.multimask_max_pt_num,
        num_multimask_outputs=trk.num_multimask_outputs,
        use_memory_selection=trk.use_memory_selection,
        non_overlap_masks_for_output=trk.non_overlap_masks_for_output,
        fill_hole_area_demo=trk.fill_hole_area,
        # outer (Sam3MultiplexTrackingWithInteractivity) heuristics
        score_threshold_detection=model.score_threshold_detection,
        det_nms_thresh=model.det_nms_thresh,
        det_nms_use_iom=model.det_nms_use_iom,
        assoc_iou_thresh=model.assoc_iou_thresh,
        trk_assoc_iou_thresh=model.trk_assoc_iou_thresh,
        new_det_thresh=model.new_det_thresh,
        hotstart_delay=model.hotstart_delay,
        hotstart_unmatch_thresh=model.hotstart_unmatch_thresh,
        hotstart_dup_thresh=model.hotstart_dup_thresh,
        suppress_unmatched_only_within_hotstart=model.suppress_unmatched_only_within_hotstart,
        init_trk_keep_alive=model.init_trk_keep_alive,
        max_trk_keep_alive=model.max_trk_keep_alive,
        min_trk_keep_alive=model.min_trk_keep_alive,
        suppress_overlap_recent_occl_thresh=model.suppress_overlapping_based_on_recent_occlusion_threshold,
        allow_unoccluded_to_suppress=model.allow_unoccluded_to_suppress,
        decrease_trk_keep_alive_for_empty_masklets=model.decrease_trk_keep_alive_for_empty_masklets,
        suppress_det_close_to_boundary=model.suppress_det_close_to_boundary,
        fill_hole_area=model.fill_hole_area,
        sprinkle_removal_area=model.sprinkle_removal_area,
        max_num_objects=model.max_num_objects,
        recondition_every_nth_frame=model.recondition_every_nth_frame,
        use_iom_recondition=model.use_iom_recondition,
        iom_thresh_recondition=model.iom_thresh_recondition,
        iou_thresh_recondition=model.iou_thresh_recondition,
        masklet_confirmation_enable=model.masklet_confirmation_enable,
        masklet_confirmation_consecutive_det_thresh=model.masklet_confirmation_consecutive_det_thresh,
        reconstruction_bbox_iou_thresh=model.reconstruction_bbox_iou_thresh,
        reconstruction_bbox_det_score=model.reconstruction_bbox_det_score,
        reapply_no_object_pointer=model.reapply_no_object_pointer,
        postprocess_batch_size=model.postprocess_batch_size,
        clear_non_cond_mem_around_input=trk.clear_non_cond_mem_around_input,
        add_all_frames_to_correct_as_cond=trk.add_all_frames_to_correct_as_cond,
        always_start_from_first_ann_frame=trk.always_start_from_first_ann_frame,
    )
    with open(os.path.join(OUT, "flags.json"), "w") as f:
        json.dump(flags, f, indent=2, default=str)
    np.savez(os.path.join(OUT, "consts.npz"), **d)
    print("[dump] flags:", json.dumps(flags, indent=2, default=str))
    print("[dump] arrays:", {k: v.shape for k, v in d.items()})
    print("[dump] wrote", os.path.join(OUT, "consts.npz"))


if __name__ == "__main__":
    main()
