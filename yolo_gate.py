#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO 检出的物体太多时，退回居中框。

The red arm (yolo_track_weight.py FLAT) wins on average but loses badly on
cluttered videos: when several people are on screen the weighted mean has no
single subject to lock onto, and the per-video constant it produces is noise.
The proposed fix is the obvious one -- if the detector sees too many objects,
believe nothing and keep the centred box.

Two ways to spend the same evidence:

  HARD   per video, drop the top q fraction of videos by track count
         (rank within the submission, so no absolute threshold to calibrate)
  SOFT   per frame, shrink the fused centre toward 0.5 by how many detections
         are in that frame:  c' = 0.5 + (c - 0.5) * exp(-n_f / K)

SOFT is the more honest of the two -- it has no per-video judgement, one
parameter, and the parameter turns out to have a broad plateau -- so it should
transfer better. Both are measured here under the same protocol: the parameter
is chosen on one half of the videos and scored on the other, 500 splits.

Usage
    python yolo_gate.py
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

K_GRID = (2, 3, 4, 5, 6, 8, 10, 12, 16, 20)
Q_GRID = (0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)


def frame_counts(tracks, ax):
    """(frames, fused person-weighted centre, n person detections per frame)."""
    num, den, cnt = {}, {}, {}
    for entries in tracks.values():
        for e in entries:
            if e[5] != "person":
                continue
            f = int(e[0])
            w = e[4] * max(e[3], 1e-9)
            if w <= 0:
                continue
            c = e[1] if ax == "x" else e[2]
            num[f] = num.get(f, 0.0) + w * c
            den[f] = den.get(f, 0.0) + w
            cnt[f] = cnt.get(f, 0) + 1
    frames = sorted(num)
    centres = np.clip(np.array([num[f] / den[f] for f in frames]), 0.0, 1.0)
    return np.array(frames, dtype=np.float64), centres, \
        np.array([cnt[f] for f in frames], dtype=np.float64)


# --------------------------------------------------------------------------
# the gate statistic
# --------------------------------------------------------------------------
# Three candidates were compared on val. All three ask "how many people are
# around", and all three are usable as a gate, but they are not equally safe
# once the length distribution changes (val 7.2x spread in frame counts, test
# 87.5x: 98 .. 8574):
#
#   stat        definition                             corr(len)   held-out gate
#   ntr         unique track ids                        +0.346      +0.0577
#   meanperson  mean person detections per sampled frame +0.116     +0.0555
#   dens        ntr / n_sampled                          +0.061     +0.0491
#
# ntr wins on val, but at corr(len) +0.346 it is largely a length proxy, so on
# the test set its top-quantile would mostly be "the longest videos". That is a
# gamble against a distribution we cannot see, for 0.002 of val F1 -- not worth
# it. `dens` is the most length-neutral but costs 0.006. `meanperson` sits in the
# middle and is the default: it needs no tracker at all (plain per-frame person
# detections), it is the cheapest to produce for the hidden test videos, and its
# corr(len) is a third of ntr's. Nothing here beats nothing by a wide margin
# except ntr, and ntr's margin is exactly the part we do not trust.
def gate_stats(R):
    """The three candidate statistics, per video, as arrays."""
    meanperson, dens, ntr = [], [], []
    for r in R:
        n_person = sum(len(es) for es in r["tracks"].values() if es[0][5] == "person")
        ns = max(1, r["ns"])
        meanperson.append(len(r["cn"]) and float(np.mean(r["cn"])) or 0.0)
        dens.append(n_person / float(ns))
        ntr.append(len(r["tracks"]))
    return (np.array(meanperson, dtype=np.float64),
            np.array(dens, dtype=np.float64),
            np.array(ntr, dtype=np.float64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=os.path.join(HERE, "val_gt.jsonl"))
    ap.add_argument("--metadata", default=os.path.join(HERE, "val_metadata.json"))
    ap.add_argument("--cache", default=os.path.join(HERE, "yolo_track_cache.json"))
    ap.add_argument("--drop-tail", type=float, default=0.08)
    ap.add_argument("--splits", type=int, default=500)
    ap.add_argument("--gate-stat", default="meanperson",
                    choices=["meanperson", "dens", "ntr"],
                    help="which clutter statistic the hard gate ranks on; "
                         "meanperson needs no tracker and is the default "
                         "(see the comment block above gate_stats)")
    args = ap.parse_args()

    meta = json.load(open(args.metadata, "r", encoding="utf-8"))
    cache = json.load(open(args.cache, "r", encoding="utf-8"))
    gt = {r["video_id"]: r for r in
          (json.loads(l) for l in open(args.gt, "r", encoding="utf-8") if l.strip())}

    def score_triplet(rec, ax, cw, ch, W, H, centres, gm):
        p = YE.boxes_for(rec, ax, cw, ch, W, H, np.asarray(centres))
        S = sum(iou_xyxy(box_from_triplet(*e["bboxes"], tw, th), gm[int(e["frame"])])
                for e in p if int(e["frame"]) in gm)
        den = len(p) + len(gm)
        return 2 * S / den if den else 0.0

    R = []
    for vid, g in gt.items():
        m = meta[vid]
        W, H, n = m["W"], m["H"], int(m["n_frames"])
        tw, th = g["targetRatioWH"]
        ax = free_axis(W, H, tw, th)
        cw, ch = crop_size(W, H, tw, th)
        f0, f1 = YE.window(n, args.drop_tail)
        gm = {}
        for e in g["predictions"]:
            fi = int(e["frame"])
            if fi not in gm:
                gm[fi] = box_from_triplet(*e["bboxes"], tw, th)
        rec = {"n_frames": n, "_drop_tail": args.drop_tail}
        ctrl = score_triplet(rec, ax, cw, ch, W, H, np.full(f1 - f0, 0.5), gm)
        fi, ce, cn = frame_counts(cache[vid]["tracks"], ax)
        R.append({"vid": vid, "n": f1 - f0, "ctl": ctrl, "ce": ce, "cn": cn,
                  "rec": rec, "ax": ax, "cw": cw, "ch": ch, "W": W, "H": H,
                  "gm": gm, "tw": tw, "th": th,
                  "ntr": len(cache[vid]["tracks"]),
                  "tracks": cache[vid]["tracks"],
                  "ns": len(cache[vid]["sampled"])})
    N = len(R)
    STATS = dict(zip(("meanperson", "dens", "ntr"), gate_stats(R)))
    ntr = STATS[args.gate_stat]

    def red_delta(r):
        """The ungated red arm's per-video F1 minus control (0 if no person)."""
        if len(r["ce"]) == 0:
            return 0.0
        c = np.full(r["n"], float(np.median(r["ce"])))
        return score_triplet(r["rec"], r["ax"], r["cw"], r["ch"], r["W"], r["H"],
                             c, r["gm"]) - r["ctl"]

    def soft_delta(r, K):
        if len(r["ce"]) == 0:
            return 0.0
        c = 0.5 + (r["ce"] - 0.5) * np.exp(-r["cn"] / K)
        return score_triplet(r["rec"], r["ax"], r["cw"], r["ch"], r["W"], r["H"],
                             np.full(r["n"], float(np.median(c))), r["gm"]) - r["ctl"]

    RED = np.array([red_delta(r) for r in R])
    SOFT = {K: np.array([soft_delta(r, K) for r in R]) for K in K_GRID}

    print("n=%d videos" % N)
    ln = np.array([r["n"] for r in R], dtype=np.float64)
    goff = np.array([abs(np.median(r["ce"]) - 0.5) if len(r["ce"]) else 0.0
                     for r in R], dtype=np.float64)
    print("\n--- gate statistic (length corr must stay small: the test set's "
          "frame counts span 87.5x vs val's 7.2x) ---")
    for name in ("meanperson", "dens", "ntr"):
        s = STATS[name]
        print("  %-11s corr(len)=%+0.3f  corr(|GT-0.5|)=%+0.3f"
              % (name, np.corrcoef(ln, s)[0, 1], np.corrcoef(goff, s)[0, 1]))
    print("  stat used by the hard gate: %s" % args.gate_stat)

    print("\n--- in-sample (parameter-picking view, NOT a result) ---")
    print("  %-34s %+.4f" % ("red, no gate", RED.mean()))
    kb = max(K_GRID, key=lambda k: SOFT[k].mean())
    print("  %-34s %+.4f  (over the K grid: %s)"
          % ("soft shrink, best K", SOFT[kb].mean(),
             " ".join("%.0f:%+.3f" % (k, SOFT[k].mean()) for k in K_GRID)))

    def hard_vec(delta, q, ref):
        """Keep delta where the track count is not in the top-q of `ref`."""
        thr = np.quantile(ntr[ref], 1 - q) if q > 0 else np.inf
        keep = ntr <= thr
        out = delta.copy()
        out[~keep] = 0.0
        return out

    def hard_holdout(delta):
        ds = []
        for _ in range(args.splits):
            idx = rng.permutation(N)
            tr, te = idx[:N // 2], idx[N // 2:]
            q = max(Q_GRID, key=lambda q: hard_vec(delta, q, tr)[tr].mean())
            ds.append(hard_vec(delta, q, te)[te].mean())
        return np.array(ds)

    def soft_holdout(delta_by_k):
        ds = []
        for _ in range(args.splits):
            idx = rng.permutation(N)
            tr, te = idx[:N // 2], idx[N // 2:]
            k = max(K_GRID, key=lambda k: delta_by_k[k][tr].mean())
            ds.append(delta_by_k[k][te].mean())
        return np.array(ds)

    rng = np.random.default_rng(0)
    print("\n--- held-out: parameter chosen on half, scored on the other "
          "(%d splits) ---" % args.splits)
    print("  %-34s %10s %10s" % ("arm", "held-out d", "se"))
    ds = []
    for _ in range(args.splits):
        idx = rng.permutation(N)
        ds.append(RED[idx[N // 2:]].mean())
    ds = np.array(ds)
    print("  %-34s %+10.4f %10.4f" % ("red, no gate", ds.mean(),
                                      ds.std() / np.sqrt(len(ds))))
    for name, arr in (("soft shrink (K fitted)", soft_holdout(SOFT)),
                      ("hard gate (q fitted)", hard_holdout(RED))):
        print("  %-34s %+10.4f %10.4f"
              % (name, arr.mean(), arr.std() / np.sqrt(len(arr))))

    # the soft shrink applied to the hard-gate survivors
    both = {}
    for k in K_GRID:
        both[k] = SOFT[k]
    ds = []
    for _ in range(args.splits):
        idx = rng.permutation(N)
        tr, te = idx[:N // 2], idx[N // 2:]
        k = max(K_GRID, key=lambda k: both[k][tr].mean())
        q = max(Q_GRID, key=lambda q: hard_vec(both[k], q, tr)[tr].mean())
        ds.append(hard_vec(both[k], q, te)[te].mean())
    ds = np.array(ds)
    print("  %-34s %+10.4f %10.4f"
          % ("soft + hard", ds.mean(), ds.std() / np.sqrt(len(ds))))

    print("\n--- mechanism: does the statistic predict WHERE red loses? ---")
    for name in ("meanperson", "dens", "ntr"):
        print("  %-11s corr(stat, red delta) = %+0.3f"
              % (name, np.corrcoef(STATS[name], RED)[0, 1]))
    srt = np.argsort(STATS[args.gate_stat])
    for q in (0.10, 0.20, 0.30, 0.40):
        m = int(round(N * q))
        drop, keep = srt[-m:], srt[:N - m]
        print("  q=%.2f dropped %2d videos (their red delta %+.4f) | kept %2d (%+.4f)"
              % (q, len(drop), RED[drop].mean() if len(drop) else 0.0,
                 len(keep), RED[keep].mean() if len(keep) else 0.0))

    print("\n--- q used as a CONSTANT (no fitting on the test side) ---")
    print("  %-6s %10s %10s %10s" % ("q", "in-sample", "held-out", "se"))
    for q in Q_GRID:
        ds = []
        for _ in range(args.splits):
            idx = rng.permutation(N)
            ds.append(hard_vec(RED, q, idx[N // 2:])[idx[N // 2:]].mean())
        ds = np.array(ds)
        print("  %-6.2f %+10.4f %+10.4f %10.4f"
              % (q, hard_vec(RED, q, np.arange(N)).mean(), ds.mean(),
                 ds.std() / np.sqrt(len(ds))))

    print("\n--- K plateau (in-sample mean delta per K) ---")
    for k in K_GRID:
        print("  K=%-3d  %+.4f   (red = %+.4f)" % (k, SOFT[k].mean(), RED.mean()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
