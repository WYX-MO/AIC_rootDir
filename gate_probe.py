#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe: can the red-box gate be improved by also looking at HOW FAR the
red centre sits from 0.5, and by SHRINKING the offset instead of gating?

Motivation
----------
The shipped gate (emit_gated.py) refuses the red box when a video looks
cluttered -- mean person detections per sampled frame > threshold. It ignores a
second, equally simple fact: the arm's whole value is moving the box off centre,
so a video whose pooled centre already lands at ~0.5 has nothing to gain and
only variance to lose. val says corr(|GT-0.5|, red delta) = +0.657, which is the
single largest unexploited effect we have measured; |pred-0.5| is the GT-free
proxy for it.

Three policies are compared on the same 57 val videos, all scored with the exact
official metric (eval_local.iou_xyxy / box_from_triplet, yolo_experiment.window):

  * shipped   -- gate if clutter > --thr, else the per-video constant red box
  * +neargate -- as shipped, but ALSO gate when |c-0.5| < --dmin
  * shrink    -- never gate on distance; emit 0.5 + lam*(c-0.5) instead, so a
                 doubtful video is nudged rather than snapped to the centre.
                 lam is swept globally, then combined with the clutter gate.

Also printed: the per-video ORACLE (choose red/centred per video with hindsight),
which is the hard ceiling any per-video binary gate can reach.

Usage
    python gate_probe.py                       # shipped vs +neargate vs shrink
    python gate_probe.py --dmin 0.02 0.04 0.06
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
from emit_gated import person_series                               # noqa: E402


def load(args):
    gt = {r["video_id"]: r for r in
          (json.loads(l) for l in open(args.gt, "r", encoding="utf-8") if l.strip())}
    meta = json.load(open(args.metadata, "r", encoding="utf-8"))
    cache = json.load(open(args.cache, "r", encoding="utf-8"))
    rows = []
    for vid, g in sorted(gt.items()):
        if vid not in meta or vid not in cache:
            continue
        W, H, n = meta[vid]["W"], meta[vid]["H"], int(meta[vid]["n_frames"])
        tw, th = g["targetRatioWH"]
        ax = free_axis(W, H, tw, th)
        gcs = {}
        for e in g["predictions"]:
            f = int(e["frame"])
            if f not in gcs:
                gcs[f] = box_from_triplet(e["bboxes"][0], e["bboxes"][1],
                                          e["bboxes"][2], tw, th)
        if not gcs:
            continue
        centres, counts = person_series(cache[vid], ax)
        cs = np.array(sorted(centres.values()), dtype=np.float64)
        c = float(np.median(cs)) if len(cs) else None
        stat = (float(np.mean([counts[f] for f in sorted(counts)]))
                if counts else 0.0)
        cw, ch = crop_size(W, H, tw, th)
        # how one-sided are the per-frame pooled centres? The signed centre is
        # the informative part (corr(c-0.5, GT-0.5) = +0.62); a video whose
        # frames disagree about the direction has a meaningless median and is a
        # candidate for the same fallback the clutter gate already performs.
        cf = np.array(sorted(centres.values()), dtype=np.float64)
        cons = (float(max((cf > 0.5).mean(), (cf < 0.5).mean()))
                if len(cf) else 0.0)
        rows.append({"vid": vid, "n": n, "W": W, "H": H, "tw": tw, "th": th,
                     "ax": ax, "cw": cw, "ch": ch, "gcs": gcs, "n_gt": len(gcs),
                     "c": c, "stat": stat, "n_det": len(cs), "cons": cons,
                     "side": float(np.median(cf) - 0.5) if len(cf) else 0.0})
    return rows


def f1_of(d, centre, drop_tail):
    """Official per-video F1 for a constant free-axis centre."""
    f0, f1 = YE.window(d["n"], drop_tail)
    centres = np.full(f1 - f0, centre, dtype=np.float64)
    p = YE.boxes_for({"n_frames": d["n"], "_drop_tail": drop_tail},
                     d["ax"], d["cw"], d["ch"], d["W"], d["H"], centres)
    S = 0.0
    for e in p:
        f = int(e["frame"])
        if f in d["gcs"]:
            S += iou_xyxy(box_from_triplet(*e["bboxes"], d["tw"], d["th"]),
                          d["gcs"][f])
    den = len(p) + d["n_gt"]
    return 2 * S / den if den else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=os.path.join(HERE, "val_gt.jsonl"))
    ap.add_argument("--metadata", default=os.path.join(HERE, "val_metadata.json"))
    ap.add_argument("--cache", default=os.path.join(HERE, "yolo_val_cache.json"))
    ap.add_argument("--drop-tail", type=float, default=0.08)
    ap.add_argument("--thr", type=float, default=2.0,
                    help="shipped clutter cut (mean person dets / sampled frame)")
    ap.add_argument("--dmin", type=float, nargs="+", default=[0.01, 0.02, 0.03, 0.05],
                    help="candidate extra gate: also gate when |c-0.5| < dmin")
    ap.add_argument("--lam", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--table", action="store_true", help="print the per-video table")
    args = ap.parse_args()

    rows = load(args)
    for d in rows:
        d["f1_ctrl"] = f1_of(d, 0.5, args.drop_tail)
        # the red arm with no gate at all; a video with no person keeps centred
        d["f1_red"] = (f1_of(d, d["c"], args.drop_tail)
                       if d["c"] is not None else d["f1_ctrl"])
        d["delta"] = d["f1_red"] - d["f1_ctrl"]
        d["off"] = abs(d["c"] - 0.5) if d["c"] is not None else 0.0
        d["cluttered"] = (d["c"] is None) or (d["stat"] > args.thr)

    def mean(xs):
        return float(np.mean(xs)) if len(xs) else 0.0

    ctrl = mean([d["f1_ctrl"] for d in rows])
    red = mean([d["f1_red"] for d in rows])
    ship = mean([d["f1_ctrl"] if d["cluttered"] else d["f1_red"] for d in rows])
    orac = mean([max(d["f1_red"], d["f1_ctrl"]) for d in rows])

    print("=" * 78)
    print("val videos = %d   drop_tail = %.2f   clutter cut = %.1f"
          % (len(rows), args.drop_tail, args.thr))
    print("  centred (champion)        %.4f" % ctrl)
    print("  red, no gate              %.4f   (%+.4f)" % (red, red - ctrl))
    print("  shipped gate              %.4f   (%+.4f)" % (ship, ship - ctrl))
    print("  ORACLE per-video pick     %.4f   (%+.4f)  <- ceiling for any"
          " per-video binary gate" % (orac, orac - ctrl))
    print("  ... of which the shipped gate already captures %.0f%%"
          % (100.0 * (ship - ctrl) / (orac - ctrl) if orac > ctrl else 0.0))
    print()

    offs = [d["off"] for d in rows]
    dels = [d["delta"] for d in rows]
    sts = [d["stat"] for d in rows]
    print("corr(|c-0.5|, delta) = %+.3f    corr(clutter, delta) = %+.3f"
          % (np.corrcoef(offs, dels)[0, 1], np.corrcoef(sts, dels)[0, 1]))
    print()

    print("--- A. shipped gate + an extra |c-0.5| < dmin cut --------------")
    print("%8s %10s %10s %10s" % ("dmin", "mean F1", "vs ship", "gated"))
    for dm in args.dmin:
        vals, ng = [], 0
        for d in rows:
            g = d["cluttered"] or (d["off"] < dm)
            ng += g
            vals.append(d["f1_ctrl"] if g else d["f1_red"])
        print("%8.3f %10.4f %+10.4f %10d" % (dm, mean(vals), mean(vals) - ship, ng))
    # the pure distance gate, ignoring clutter, for reference
    for dm in args.dmin:
        vals = [d["f1_ctrl"] if d["off"] < dm else d["f1_red"] for d in rows]
        print("  [distance only, dmin=%.3f -> %.4f]" % (dm, mean(vals)))
    print()

    print("--- B. shrink the offset:  centre = 0.5 + lam*(c-0.5) ----------")
    print("%8s %10s %10s %10s" % ("lam", "no gate", "with clutter gate",
                                  "gate+lam<"))
    for lam in args.lam:
        vals, vals_g = [], []
        for d in rows:
            cm = 0.5 if d["c"] is None else 0.5 + lam * (d["c"] - 0.5)
            vals.append(f1_of(d, cm, args.drop_tail))
            vals_g.append(d["f1_ctrl"] if d["cluttered"]
                          else f1_of(d, cm, args.drop_tail))
        print("%8.2f %10.4f %10.4f" % (lam, mean(vals), mean(vals_g)))
    print("  (lam=1.0 is the red arm, lam=0.0 is centred)")
    print()

    # where does the extra gain come from? bin by |c-0.5|
    print("--- C. delta by |c-0.5| bin (does the near-centre tail really hurt) --")
    edges = [0.0, 0.01, 0.02, 0.04, 0.08, 0.15, 1.0]
    for lo, hi in zip(edges, edges[1:]):
        b = [d for d in rows if lo <= d["off"] < hi]
        if not b:
            continue
        print("  |c-0.5| %.3f-%.3f  n=%-3d  mean delta %+.4f  "
              "mean clutter %.2f" % (lo, hi, len(b), mean([d["delta"] for d in b]),
                                     mean([d["stat"] for d in b])))
    print()

    print("--- D. why: is |c-0.5| even a proxy for |GT-0.5|? -------------")
    # GT centre, same normalisation the judge uses on the free axis
    for d in rows:
        mus = []
        for f, gb in d["gcs"].items():
            # gb is the xyxy box, so the centre is the mean of the two edges --
            # NOT x1 + x2/2, which is what a missing bracket silently computes
            if d["ax"] == "x":
                mus.append(((gb[0] + gb[2]) / 2.0) / d["W"])
            else:
                mus.append(((gb[1] + gb[3]) / 2.0) / d["H"])
        d["mu_gt"] = float(np.median(mus)) if mus else 0.5
        if not 0.0 <= d["mu_gt"] <= 1.0:
            raise SystemExit("%s: GT centre %.4f outside [0,1] -- the box is not "
                             "in normalised free-axis coordinates"
                             % (d["vid"], d["mu_gt"]))
        d["off_gt"] = abs(d["mu_gt"] - 0.5)
    print("  corr(|c-0.5|, |GT-0.5|)   = %+.3f   <- how well the predicted"
          " offset tracks the true one"
          % np.corrcoef([d["off"] for d in rows],
                        [d["off_gt"] for d in rows])[0, 1])
    print("  corr(|GT-0.5|, delta)     = %+.3f   <- the effect we actually"
          " want, but it is NOT observable at test time"
          % np.corrcoef([d["off_gt"] for d in rows],
                        [d["delta"] for d in rows])[0, 1])
    # the cheat gate: gate on the TRUE framing offset. This is what a perfect
    # subject-judger would be worth, so it bounds the whole VLM direction.
    best = (-9, None)
    for dm in np.arange(0.0, 0.21, 0.01):
        v = mean([d["f1_ctrl"] if d["off_gt"] < dm else d["f1_red"] for d in rows])
        if v > best[0]:
            best = (v, dm)
    print("  CHEAT gate on true |GT-0.5| < %.2f -> %.4f  (%+.4f vs shipped)"
          % (best[1], best[0], best[0] - ship))
    print("  ... vs the per-video ORACLE (full hindsight)     -> %.4f  (%+.4f)"
          % (orac, orac - ship))
    print()

    print("--- G. ceiling of each box FAMILY (what is actually left) ------")
    # (1) a per-video constant placed exactly at the GT median centre -- the
    #     best a per-video-constant box can ever do, with full hindsight.
    # (2) the GT's own per-frame centre -- the best any trajectory can do.
    o_const = mean([f1_of(d, d["mu_gt"], args.drop_tail) for d in rows])
    o_traj = []
    for d in rows:
        f0, f1 = YE.window(d["n"], args.drop_tail)
        gcs = d["gcs"]
        fs = sorted(gcs)
        ctr = []
        for f in range(f0, f1):
            # nearest GT frame's centre (GT is a slow-moving framing)
            near = min(fs, key=lambda x: abs(x - f))
            gb = gcs[near]
            ctr.append(((gb[0] + gb[2]) / 2.0) / d["W"] if d["ax"] == "x"
                       else ((gb[1] + gb[3]) / 2.0) / d["H"])
        p = YE.boxes_for({"n_frames": d["n"], "_drop_tail": args.drop_tail},
                         d["ax"], d["cw"], d["ch"], d["W"], d["H"],
                         np.array(ctr))
        S = sum(iou_xyxy(box_from_triplet(*e["bboxes"], d["tw"], d["th"]),
                         gcs[int(e["frame"])])
                for e in p if int(e["frame"]) in gcs)
        den = len(p) + d["n_gt"]
        o_traj.append(2 * S / den if den else 0.0)
    print("  shipped (per-video const, gated)        %.4f" % ship)
    print("  oracle per-video CONSTANT at GT centre  %.4f  (%+.4f vs shipped)"
          % (o_const, o_const - ship))
    print("  oracle per-frame TRAJECTORY on GT centre %.4f  (%+.4f vs shipped)"
          % (mean(o_traj), mean(o_traj) - ship))
    print("  => a perfect subject-judger that still emits ONE box per video can"
          " reach at most %.4f val" % o_const)
    print("     (= %+.4f online at the measured 1/3.4 val->online discount)"
          % ((o_const - ship) / 3.4))
    print()

    print("--- F. sign-consistency gate (the +0.62 signal, not |.|) --------")
    # cons = max(frac frames left of 0.5, frac right of 0.5). 1.0 = the whole
    # video agrees on a direction; 0.5 = the median is meaningless.
    devg = np.array([d["off_gt"] for d in rows])
    cons = np.array([d["cons"] for d in rows])
    sign_ok = np.array([np.sign(d["side"]) == np.sign(d["mu_gt"] - 0.5)
                        for d in rows])
    am = np.abs(devg) > 0.02
    print("  corr(cons, delta) = %+.3f" % np.corrcoef(cons, dels)[0, 1])
    print("  sign correctness, all %d videos          : %.0f%%"
          % (len(rows), 100 * sign_ok.mean()))
    print("  sign correctness, |GT-0.5|>0.02 (n=%d)  : %.0f%%"
          % (am.sum(), 100 * sign_ok[am].mean()))
    print("  mean delta when the lean is RIGHT: %+.4f (n=%d) | WRONG: %+.4f (n=%d)"
          % (mean([d["delta"] for d, s in zip(rows, sign_ok) if s]), sign_ok.sum(),
             mean([d["delta"] for d, s in zip(rows, sign_ok) if not s]),
             (~sign_ok).sum()))
    print()
    print("  %8s %10s %10s %8s   (gate if clutter>s OR cons<cut)"
          % ("cut", "mean F1", "vs ship", "gated"))
    fbest = (-9, None, None)
    for s in [1.5, 2.0, 3.0, 1e9]:
        for cut in [0.5, 0.6, 0.7, 0.8, 0.9, 1.01]:
            v = mean([d["f1_ctrl"] if (d["c"] is None or d["stat"] > s
                                       or d["cons"] < cut) else d["f1_red"]
                      for d in rows])
            ng = sum(1 for d in rows if (d["c"] is None or d["stat"] > s
                                         or d["cons"] < cut))
            if v > fbest[0]:
                fbest = (v, s, cut, ng)
            print("  s=%-5s %8.2f %10.4f %+10.4f %8d"
                  % ("inf" if s > 1e8 else "%.1f" % s, cut, v, v - ship, ng))
    print("  best: clutter>%s OR cons<%.2f -> %.4f (%+.4f)  gated=%d"
          % ("inf" if fbest[1] > 1e8 else fbest[1], fbest[2], fbest[0],
             fbest[0] - ship, fbest[3]))
    print()

    print("--- E. 2D grid: gate if clutter > s OR |c-0.5| < dm -------------")
    grid_s = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 1e9]
    grid_d = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03]
    print("  %6s" % "s\\dm" + "".join("%9.3f" % dm for dm in grid_d))
    gbest = (-9, None, None)
    for s in grid_s:
        cells = []
        for dm in grid_d:
            v = mean([d["f1_ctrl"] if (d["c"] is None or d["stat"] > s
                                       or d["off"] < dm) else d["f1_red"]
                      for d in rows])
            cells.append(v)
            if v > gbest[0]:
                gbest = (v, s, dm)
        print("  %6s" % ("inf" if s > 1e8 else "%.1f" % s)
              + "".join("%9.4f" % v for v in cells))
    print("  best cell: clutter>%s OR |c-0.5|<%.3f -> %.4f (%+.4f vs shipped)"
          % ("inf" if gbest[1] > 1e8 else "%.1f" % gbest[1], gbest[2],
             gbest[0], gbest[0] - ship))
    print()

    if args.table:
        print("%-22s %5s %7s %7s %8s %8s %8s %6s"
              % ("video", "n", "clutt", "|c-.5|", "f1_red", "f1_ctrl",
                 "delta", "gated"))
        for d in sorted(rows, key=lambda d: -d["delta"]):
            print("%-22s %5d %7.2f %7.3f %8.4f %8.4f %+8.4f %6s"
                  % (d["vid"], d["n_det"], d["stat"], d["off"], d["f1_red"],
                     d["f1_ctrl"], d["delta"], "Y" if d["cluttered"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
