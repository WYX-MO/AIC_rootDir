#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SAM (class-agnostic segmentation) as an alternative framing-centre estimator.

Why this probe exists
---------------------
The shipped box arm estimates each video's framing centre from YOLO *person box*
centres (FLAT: conf*area weighted, pooled to one per-video constant). On val it
goes 0.3688 (centred) -> 0.4214 (ungated) -> 0.4305 (gated). The oracle constant
box placed at the true GT centre is 0.5315, and the entire remaining gap is
**residual bias** -- `recal_probe.py` showed a leave-one-out regression over YOLO
summary statistics can only remove 5% of it (0.0667 -> 0.0632, F1 0.4214 ->
0.4087). The bias is a latent variable: you have to read pixels.

YOLO cannot read them here. COCO has no class for the subjects in these clips
(rockets, fireworks, race cars) and 36% of the test set has <0.5 person
detections per sampled frame -- precisely the videos the current gate retreats
to a centred box on. A class-agnostic segmenter is the obvious next thing to try,
and the official task brief points at SAM for the same reason.

The falsifiable question
------------------------
Does a mask centroid estimate the framing centre with *less bias* than a person
box centroid? Note the memory's warning: GT is a *framing*, not a subject --
qvh_000214's GT box spans an entire group of dancers while YOLO locks onto one
of them, scoring F1 0.341 -> 0.001. A mask-based "extent of the busy region" may
match that convention better than "the most salient person" does.

Nothing here trains anything and nothing is tuned on the test set. It is one
GPU pass plus an offline sweep of selection rules.

Phases
------
    python3 sam_probe.py gpu   --stride 40          # -> sam_val_masks.json
    python3 sam_probe.py rules --rule area_w        # -> sam_val_cache.json
    python3 sam_probe.py bias  --rule area_w        # bias diagnostics vs GT

The cache written by `rules` is byte-compatible with yolo_val_cache.json
(W/H/n_frames/frames/dets, cls="person", cx/cy normalised), so `emit_gated.py`
consumes it unchanged and the comparison against 0.4214 / 0.4305 is apples to
apples on the same frame set.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from stage_data import crop_size, free_axis                       # noqa: E402

SAM_MODEL = "facebook/sam-vit-base"
SAM_CACHE = os.path.expanduser("~/.cache/hf_sam")
MASKS_JSON = os.path.join(HERE, "sam_val_masks.json")


# --------------------------------------------------------------------------
# phase 1: GPU pass
# --------------------------------------------------------------------------
def phase_gpu(args):
    import cv2
    import torch
    from PIL import Image
    from transformers import pipeline

    index = json.load(open(args.index, "r", encoding="utf-8"))
    meta = json.load(open(args.metadata, "r", encoding="utf-8"))

    done = {}
    if os.path.exists(MASKS_JSON) and args.resume:
        done = json.load(open(MASKS_JSON, "r", encoding="utf-8"))
        print("resume: %d/%d videos already done" % (len(done), len(index)))

    gen = pipeline("mask-generation", model=SAM_MODEL, cache_dir=SAM_CACHE,
                   device=0, points_per_batch=64)

    for k, item in enumerate(index):
        vid = str(item["video_id"])
        if vid in done:
            continue
        m = meta.get(vid)
        if not m:
            print("  [skip] no metadata for %s" % vid)
            continue
        path = os.path.join(args.video_dir, "%s.mp4" % vid)
        cap = cv2.VideoCapture(path)
        frames, masks = [], []
        n = int(m["n_frames"])
        for fi in range(0, n, args.stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, bgr = cap.read()
            if not ok:
                continue
            img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            W, H = img.size
            out = gen(img, points_per_side=args.pps,
                      pred_iou_thresh=args.iou_thresh,
                      stability_score_thresh=args.stability)
            recs = []
            for mask, score in zip(out["masks"], out["scores"]):
                a = mask.numpy() if hasattr(mask, "numpy") else np.asarray(mask)
                area = float(a.sum())
                if area <= 0:
                    continue
                frac = area / (W * H)
                if frac < args.min_area:
                    continue
                ys, xs = np.nonzero(a)
                recs.append({"cx": float(xs.mean()) / W,
                             "cy": float(ys.mean()) / H,
                             "area": frac,
                             "iou": float(score),
                             "x0": float(xs.min()) / W, "x1": float(xs.max()) / W,
                             "y0": float(ys.min()) / H, "y1": float(ys.max()) / H})
            frames.append(fi)
            masks.append(recs)
        cap.release()
        done[vid] = {"W": int(m["W"]), "H": int(m["H"]), "n_frames": n,
                     "frames": frames, "masks": masks}
        nk = sum(len(x) for x in masks)
        print("  [%2d/%d] %-22s %2d frames %4d masks  (%.1fs cum)"
              % (k + 1, len(index), vid, len(frames), nk,
                 __import__("time").time() - T0))
        json.dump(done, open(MASKS_JSON, "w", encoding="utf-8"))
    print("wrote %s (%d videos)" % (MASKS_JSON, len(done)))
    return 0


# --------------------------------------------------------------------------
# selection rules: per frame -> one centre
# --------------------------------------------------------------------------
def frame_centre(recs, rule):
    if not recs:
        return None
    if rule == "largest":
        r = max(recs, key=lambda d: d["area"])
        return r["cx"], r["cy"]
    if rule == "extent":
        # centre of the union bbox of everything above min-area
        return (min(r["x0"] for r in recs) + max(r["x1"] for r in recs)) / 2.0, \
               (min(r["y0"] for r in recs) + max(r["y1"] for r in recs)) / 2.0
    if rule == "area_w":
        w = np.array([r["area"] for r in recs])
    elif rule == "sqrt_area_w":
        w = np.sqrt([r["area"] for r in recs])
    elif rule == "iou_area_w":
        w = np.array([r["area"] * r["iou"] for r in recs])
    elif rule == "iou_w":
        w = np.array([r["iou"] for r in recs])
    else:
        raise ValueError(rule)
    w = w / w.sum()
    return float(np.dot(w, [r["cx"] for r in recs])), \
           float(np.dot(w, [r["cy"] for r in recs]))


def build_cache(masks, rule, top_k=None):
    """yolo_val_cache.json-compatible: one synthetic det per sampled frame."""
    cache = {}
    for vid, d in masks.items():
        dets = []
        for recs in d["masks"]:
            if top_k:
                recs = sorted(recs, key=lambda r: -r["area"])[:top_k]
            c = frame_centre(recs, rule)
            if c is None:
                dets.append([])
                continue
            # one det carries the whole frame's answer: person_series then pools
            # conf*area weighted, so conf=area=1 makes the pooled centre == c.
            dets.append([{"cls": "person", "conf": 1.0, "cx": c[0], "cy": c[1],
                          "area": 1.0}])
        cache[vid] = {"W": d["W"], "H": d["H"], "n_frames": d["n_frames"],
                      "frames": d["frames"], "dets": dets}
    return cache


# --------------------------------------------------------------------------
# bias diagnostics vs GT
# --------------------------------------------------------------------------
def gt_centres():
    """{vid: (free_axis, [constant GT centre per video...])}"""
    gt = [json.loads(l) for l in open(os.path.join(HERE, "val_gt.jsonl"))
          if l.strip()]
    meta = json.load(open(os.path.join(HERE, "val_metadata.json")))
    out = {}
    for rec in gt:
        vid = str(rec["video_id"])
        m = meta[vid]
        W, H = int(m["W"]), int(m["H"])
        tw, th = rec["targetRatioWH"]
        ax = free_axis(W, H, tw, th)
        cs = []
        for p in rec["predictions"]:
            x, y, w = p["bboxes"]
            if ax == "x":
                cs.append((x + w / 2.0) / W)
            else:
                h = w * float(th) / float(tw)
                cs.append((y + h / 2.0) / H)
        out[vid] = (ax, cs)
    return out


def rule_constant(vid, d, rule):
    """Per-video constant centre (median over frames), same as emit_gated."""
    key = 0
    vals = []
    for recs in d["masks"]:
        c = frame_centre(recs, rule)
        if c is not None:
            vals.append(c[key])
    return float(np.median(vals)) if vals else None


def main():
    global T0
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["gpu", "rules", "bias", "sweep"])
    ap.add_argument("--index", default=os.path.join(HERE, "val_index.json"))
    ap.add_argument("--metadata", default=os.path.join(HERE, "val_metadata.json"))
    ap.add_argument("--video-dir", default=os.path.join(HERE, "val_video"))
    ap.add_argument("--stride", type=int, default=40)
    ap.add_argument("--pps", type=int, default=24)
    ap.add_argument("--iou-thresh", type=float, default=0.88)
    ap.add_argument("--stability", type=float, default=0.92)
    ap.add_argument("--min-area", type=float, default=0.002)
    ap.add_argument("--rule", default="area_w")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "sam_val_cache.json"))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    import time
    T0 = time.time()

    if args.phase == "gpu":
        return phase_gpu(args)

    masks = json.load(open(MASKS_JSON, "r", encoding="utf-8"))
    if args.phase == "rules":
        cache = build_cache(masks, args.rule, args.top_k)
        json.dump(cache, open(args.out, "w", encoding="utf-8"))
        print("wrote %s  rule=%s top_k=%s  (%d videos)"
              % (args.out, args.rule, args.top_k, len(cache)))
        return 0

    gt = gt_centres()
    if args.phase == "bias":
        errs, signed = [], []
        for vid, d in masks.items():
            if vid not in gt:
                continue
            ax, gcs = gt[vid]
            if not gcs or ax is None:
                continue
            gt_c = float(np.median(gcs))
            c = rule_constant(vid, d, args.rule)
            if c is None:
                continue
            errs.append(abs(c - gt_c))
            signed.append(c - gt_c)
        print("rule=%s  n=%d" % (args.rule, len(errs)))
        print("  mean|c-GT| = %.4f   mean(c-GT) = %+.4f   sd = %.4f"
              % (np.mean(errs), np.mean(signed), np.std(signed)))
        return 0

    # sweep: bias for every rule, cheap (no GPU)
    for rule in ["largest", "extent", "area_w", "sqrt_area_w", "iou_area_w", "iou_w"]:
        errs = []
        for vid, d in masks.items():
            if vid not in gt:
                continue
            ax, gcs = gt[vid]
            if not gcs or ax is None:
                continue
            c = rule_constant(vid, d, rule)
            if c is not None:
                errs.append(abs(c - float(np.median(gcs))))
        print("  %-14s mean|c-GT| = %.4f  (n=%d)" % (rule, np.mean(errs), len(errs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
