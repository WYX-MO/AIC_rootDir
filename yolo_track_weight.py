#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Same idea as yolo_main_object.py, but used as a WEIGHT instead of a SELECT.

yolo_main_object.py hard-*selects* one track per video and throws the others
away; all five selection rules measured <= +0.0013 (held-out negative). This
script asks the fair follow-up: keep the exact same per-track criterion
("how long is this object on screen" x "how big is it"), but use it to WEIGHT
a per-frame pooling of every track's centre instead of to pick a winner.

Rationale for expecting the weight version to behave differently: the YOLO
centre tracks GT at r = +0.489 for the *per-frame pooled* centre (yolo_experiment
--diagnose) but only +0.10..+0.35 for a selected main-object median. Selection
discards the frames where the main object is invisible (the crop then has to
come from somewhere else anyway), so all it can do is lower the effective sample
size of the same statistic.

Arms
  select  : argmax_t w_t  ->  median centre of that track, then shrink beta
  weight  : per frame, c_f = sum_t w_t * c_t(f) / sum_t w_t  (tracks present at
            f only), then per-video median, then shrink beta
with w_t in {presence*meanarea, presence, meanarea} and an optional person-only
filter on the detections being pooled.

Usage
    python yolo_track_weight.py
    python yolo_track_weight.py --person-only
"""

import os
import sys
import json
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from stage_data import crop_size, free_axis                        # noqa: E402
from eval_local import iou_xyxy, box_from_triplet                  # noqa: E402
import yolo_experiment as YE                                       # noqa: E402
from yolo_main_object import track_stats                           # noqa: E402

RULES = ("pres*area", "presence", "meanarea")


def track_weight(entries, n_sampled, rule):
    pres, marea, _, _ = track_stats(entries, n_sampled)
    if rule == "pres*area":
        return pres * marea
    if rule == "presence":
        return pres
    if rule == "meanarea":
        return marea
    raise ValueError(rule)


def use_track(entries, person_only):
    """A track is one ByteTrack id; its class is whatever it was first seen as."""
    return (not person_only) or entries[0][5] == "person"


def flat_series(tracks, ax, person_only, power):
    """yolo_experiment §4.7's champion, in this harness: ignore track identity,
    pool every DETECTION in a frame with weight conf * area**power.

    Included so the track-based arms above are compared against the best known
    per-frame aggregation on identical footing (same cache, same scorer).
    """
    num, den = {}, {}
    for entries in tracks.values():
        for e in entries:
            if person_only and e[5] != "person":
                continue
            f, c = int(e[0]), (e[1] if ax == "x" else e[2])
            w = e[4] * (max(e[3], 0.0) ** power)
            if w <= 0:
                continue
            num[f] = num.get(f, 0.0) + w * c
            den[f] = den.get(f, 0.0) + w
    frames = sorted(num)
    centres = [min(1.0, max(0.0, num[f] / den[f])) for f in frames]
    return np.array(frames, dtype=np.float64), np.array(centres, dtype=np.float64)


def fused_series(tracks, n_sampled, ax, rule, person_only):
    """Per-sampled-frame fused centre from all tracks, weighted by the rule.

    Returns (frames[], centre[]) -- only frames where at least one usable track
    is present. A frame with no detection is dropped here rather than pinned to
    0.5, exactly as _ymed does in yolo_experiment, so the two are comparable.
    """
    num, den = {}, {}
    for tid, entries in tracks.items():
        if not use_track(entries, person_only):
            continue
        w = track_weight(entries, n_sampled, rule)
        if w <= 0:
            continue
        for e in entries:
            f, c = int(e[0]), (e[1] if ax == "x" else e[2])
            num[f] = num.get(f, 0.0) + w * c
            den[f] = den.get(f, 0.0) + w
    frames = sorted(num)
    centres = [min(1.0, max(0.0, num[f] / den[f])) for f in frames]
    return np.array(frames, dtype=np.float64), np.array(centres, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(HERE, "yolo_track_cache.json"))
    ap.add_argument("--gt", default=os.path.join(HERE, "val_gt.jsonl"))
    ap.add_argument("--metadata", default=os.path.join(HERE, "val_metadata.json"))
    ap.add_argument("--drop-tail", type=float, default=0.08)
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--person-only", action="store_true")
    args = ap.parse_args()

    cache = json.load(open(args.cache, "r", encoding="utf-8"))
    meta = json.load(open(args.metadata, "r", encoding="utf-8"))
    gt = {r["video_id"]: r for r in
          (json.loads(l) for l in open(args.gt, "r", encoding="utf-8") if l.strip())}

    D = {}
    for vid, g in gt.items():
        if vid not in cache:
            continue
        m = meta[vid]
        W, H, n = m["W"], m["H"], int(m["n_frames"])
        tw, th = g["targetRatioWH"]
        ax = free_axis(W, H, tw, th)
        cs = []
        for e in g["predictions"]:
            x, y, w = e["bboxes"]
            cs.append(((x + w / 2.0) / W) if ax == "x"
                      else ((y + (w * float(th) / float(tw)) / 2.0) / H))
        if not cs:
            continue
        D[vid] = {"W": W, "H": H, "n": n, "ax": ax, "tw": tw, "th": th,
                  "mu_gt": float(np.median(cs)), "rec": cache[vid]}
    vids = sorted(D)

    def f1_video(vid, centre_of_video):
        d = D[vid]
        gm = {}
        for e in gt[vid]["predictions"]:
            fi = int(e["frame"])
            if fi not in gm:
                gm[fi] = box_from_triplet(*e["bboxes"], d["tw"], d["th"])
        cw, ch = crop_size(d["W"], d["H"], d["tw"], d["th"])
        f0, f1 = YE.window(d["n"], args.drop_tail)
        p = YE.boxes_for({"n_frames": d["n"], "_drop_tail": args.drop_tail},
                         d["ax"], cw, ch, d["W"], d["H"],
                         np.full(f1 - f0, centre_of_video))
        S = 0.0
        for e in p:
            fi = int(e["frame"])
            if fi in gm:
                S += iou_xyxy(box_from_triplet(*e["bboxes"], d["tw"], d["th"]), gm[fi])
        den = len(p) + len(gm)
        return 2 * S / den if den else 0.0

    ctrl = {v: f1_video(v, 0.5) for v in vids}
    ctrl_mean = float(np.mean(list(ctrl.values())))
    print("control (centred box, frames 0..n-round(%.2f n)-1): %.4f  (n=%d)"
          % (args.drop_tail, ctrl_mean, len(vids)))

    def vid_centre(vid, rule, mode, person_only):
        """The single per-video centre an arm proposes, before shrinking."""
        rec = D[vid]["rec"]
        ns = len(rec["sampled"])
        ax = D[vid]["ax"]
        if mode == "select":
            best, best_w = None, -1.0
            for tid, entries in rec["tracks"].items():
                if not use_track(entries, person_only):
                    continue
                w = track_weight(entries, ns, rule)
                if w > best_w:
                    best, best_w = entries, w
            if best is None:
                return None
            return float(np.median([e[1] if ax == "x" else e[2] for e in best]))
        fi, ce = fused_series(rec["tracks"], ns, ax, rule, person_only)
        if len(fi) == 0:
            return None
        return float(np.median(ce))

    def flat_centre(vid, person_only, power):
        rec = D[vid]["rec"]
        fi, ce = flat_series(rec["tracks"], D[vid]["ax"], person_only, power)
        if len(fi) == 0:
            return None
        return float(np.median(ce))

    rng = np.random.default_rng(0)

    def report(label, cf):
        usable = [v for v in vids if cf[v] is not None]
        if len(usable) < 10:
            print("%-34s  <too few usable videos>" % label)
            return
        gg = np.array([D[v]["mu_gt"] for v in usable])
        yy = np.array([cf[v] for v in usable])
        corr = (float(np.corrcoef(gg, yy)[0, 1])
                if yy.std() > 1e-9 else float("nan"))
        # unshrunk and half-shrunk F1 (videos with no usable track keep control)
        def f1_with(beta):
            return float(np.mean([
                f1_video(v, 0.5 + beta * (cf[v] - 0.5)) if cf[v] is not None
                else ctrl[v] for v in vids]))
        f1_b1, f1_b05 = f1_with(1.0), f1_with(0.5)
        # 200 random half/half splits: fit beta on train, score on test
        ds = []
        for _ in range(args.splits):
            idx = rng.permutation(len(usable))
            te = [usable[i] for i in idx[:len(usable) // 2]]
            tr = [usable[i] for i in idx[len(usable) // 2:]]
            if len(tr) < 5 or not te:
                continue
            gy = np.array([D[u]["mu_gt"] for u in tr])
            yy_ = np.array([cf[u] for u in tr])
            if yy_.std() < 1e-9:
                continue
            cc = np.cov(gy, yy_, ddof=1)
            b = float(cc[0, 1] / cc[1, 1])
            ds.append(np.mean([f1_video(v, 0.5 + b * (cf[v] - 0.5)) - ctrl[v]
                               for v in te]))
        ds = np.array(ds) if ds else np.array([0.0])
        print("%-34s %+6.3f %8.4f %8.4f %+10.4f %7.1f%% %6d"
              % (label, corr, f1_b1, f1_b05,
                 ds.mean(), 100 * (ds > 0).mean(), len(usable)))

    print("\n%-34s %6s %8s %8s %10s %8s %6s"
          % ("arm", "corr", "F1 b=1", "F1 b=0.5", "held-out d", "win%", "n"))
    for person_only in ((True, False) if args.person_only else (False,)):
        p = "P" if person_only else "all"
        for rule in RULES:
            for mode in ("select", "weight"):
                report("%s %s %s" % (mode, rule, p),
                       {v: vid_centre(v, rule, mode, person_only) for v in vids})
    # the §4.7 champion, for an apples-to-apples reference
    for person_only in ((True, False) if args.person_only else (False,)):
        p = "P" if person_only else "all"
        for power in (0.5, 1.0):
            report("FLAT conf*a^%.1f %s (== 4.7)" % (power, p),
                   {v: flat_centre(v, person_only, power) for v in vids})
    return 0


if __name__ == "__main__":
    sys.exit(main())
