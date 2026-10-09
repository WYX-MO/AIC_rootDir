#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Before spending GPU hours on a VLM, ask the cheap question:

    does the YOLO person candidate list even CONTAIN a better choice than the
    conf*area weighted pool the shipped arm already uses?

Every selector below runs on the cx/cy/area/conf already in yolo_val_cache.json,
so this costs nothing -- no GPU, no re-detection. The decisive row is
`near_gt`: it picks, per frame, the person whose centre is closest to where the
GT framing actually is. That is the PERFECT subject-judger restricted to
choosing among YOLO's own candidates. If near_gt does not beat `pool`, then no
VLM, however good, can help by selecting a person -- the information is not in
the candidate set, and the whole direction is dead.

Two aggregations, because they are different products:
  const -- pool the per-frame pick to ONE box for the video (the shipped design)
  traj  -- keep the per-frame pick (a trajectory; measured worse in practice,
           but it is the higher ceiling, so print it for reference)

Usage
    python select_probe.py                 # all selectors, both aggregations
    python select_probe.py --table
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

MODES = ["pool", "pool_med", "largest", "conf", "central", "near_gt"]
# near_gt peeks at the ground truth; it is the ceiling, not a proposal.
CHEAT = {"near_gt"}


def gt_centres(rec, ax):
    """frame -> GT framing centre on the free axis (from the loaded gt row)."""
    return rec


def pick(dets, ax, mode, gt_c):
    """One frame's person candidates -> a single free-axis centre, or None."""
    if not dets:
        return None
    c = np.array([d["cx"] if ax == "x" else d["cy"] for d in dets], dtype=np.float64)
    a = np.array([d["area"] for d in dets], dtype=np.float64)
    f = np.array([d["conf"] for d in dets], dtype=np.float64)
    if mode == "pool":
        w = f * np.maximum(a, 1e-9)
        return float((c * w).sum() / w.sum())
    if mode == "pool_med":
        return float(np.median(c))
    if mode == "largest":
        return float(c[np.argmax(a)])
    if mode == "conf":
        return float(c[np.argmax(f)])
    if mode == "central":
        return float(c[np.argmin(np.abs(c - 0.5))])
    if mode == "near_gt":
        if gt_c is None:
            return float(np.median(c))
        return float(c[np.argmin(np.abs(c - gt_c))])
    raise SystemExit("unknown mode " + mode)


def load(args):
    gt = {r["video_id"]: r for r in
          (json.loads(l) for l in open(args.gt, "r", encoding="utf-8") if l.strip())}
    meta = json.load(open(args.metadata, "r", encoding="utf-8"))
    cache = json.load(open(args.cache, "r", encoding="utf-8"))
    rows = []
    for vid, g in sorted(gt.items()):
        if vid not in meta or vid not in cache:
            continue
        m, rec = meta[vid], cache[vid]
        W, H, n = m["W"], m["H"], int(m["n_frames"])
        tw, th = g["targetRatioWH"]
        ax = free_axis(W, H, tw, th)
        gcs = {}
        for e in g["predictions"]:
            fb = box_from_triplet(e["bboxes"][0], e["bboxes"][1], e["bboxes"][2],
                                  tw, th)
            f = int(e["frame"])
            if f not in gcs:
                gcs[f] = fb
        if not gcs:
            continue
        gctr = {}
        for f, gb in gcs.items():
            gctr[f] = (((gb[0] + gb[2]) / 2.0) / W if ax == "x"
                       else ((gb[1] + gb[3]) / 2.0) / H)
        cw, ch = crop_size(W, H, tw, th)
        rows.append({"vid": vid, "n": n, "W": W, "H": H, "tw": tw, "th": th,
                     "ax": ax, "cw": cw, "ch": ch, "gcs": gcs, "n_gt": len(gcs),
                     "gctr": gctr, "frames": rec["frames"], "dets": rec["dets"],
                     "clutt": float(np.mean([sum(1 for d in dd
                                                if d["cls"] == "person")
                                            for dd in rec["dets"]]))})
    return rows


def f1(d, centres, drop_tail):
    """Official per-video F1 for a per-frame centre array (densified)."""
    f0, f1n = YE.window(d["n"], drop_tail)
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


def simulate(args, rows):
    """How accurate must a pointing model be to be worth downloading?

    A pointer is not restricted to YOLO candidates, so it escapes the negative
    result above -- but it must beat it. This replaces the true framing centre
    with a noisy version of itself and sweeps the noise. The sigma at which the
    curve crosses the shipped score is the accuracy bar, in normalised
    free-axis units; multiply by the source width for a pixel budget.
    """
    sigmas = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.0675, 0.08]
    print("--- H. accuracy bar for any NEW centre estimator (pointer / VLM) ----")
    print("  replacing the true framing centre with  c_gt + N(0, sigma);")
    print("  one point per sampled frame, then same box machinery as the arm")
    print()
    print("  %8s %10s %10s %10s %10s" % ("sigma", "px@534", "const", "traj",
                                        "const+gate"))
    print("  %8s %10s %10s %10s %10s" % ("pool", "%.0f" % (0.0675 * 534),
                                        "0.4214", "0.4018", "(shipped 0.4302)"))
    for sig in sigmas:
        cs, ts, gs = [], [], []
        for d in rows:
            a, b = YE.window(d["n"], args.drop_tail)
            gfs = sorted(d["gctr"])
            est, eid = [], []
            for f in range(a, b, 5):
                gc = d["gctr"][min(gfs, key=lambda x: abs(x - f))]
                est.append(float(np.clip(gc + np.random.normal(0.0, sig), 0.0, 1.0)))
                eid.append(f)
            est = np.array(est)
            eid = np.array(eid, dtype=np.float64)
            cm = float(np.median(est))
            # a gated arm would keep the clutter fallback; approximate its
            # effect by refusing the estimate where clutter exceeds the cut
            g = d["clutt"] > 2.0
            cs.append(f1(d, np.full(b - a, cm), args.drop_tail))
            ts.append(f1(d, YE.densify(eid, est, b - a, b - a), args.drop_tail))
            gs.append(f1(d, np.full(b - a, 0.5 if g else cm), args.drop_tail))
        print("  %8.4f %10.0f %10.4f %10.4f %10.4f"
              % (sig, sig * 534, float(np.mean(cs)), float(np.mean(ts)),
                 float(np.mean(gs))))
    print()
    print("  read off the sigma where 'const' falls to 0.4214 (the free pool) and")
    print("  to 0.4302 (the shipped gated arm) -- those are the bars.")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=os.path.join(HERE, "val_gt.jsonl"))
    ap.add_argument("--metadata", default=os.path.join(HERE, "val_metadata.json"))
    ap.add_argument("--cache", default=os.path.join(HERE, "yolo_val_cache.json"))
    ap.add_argument("--drop-tail", type=float, default=0.08)
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--simulate", action="store_true",
                    help="sweep pointing noise to find the accuracy bar")
    args = ap.parse_args()

    rows = load(args)
    if args.simulate:
        np.random.seed(0)
        simulate(args, rows)
        return 0

    # nearest GT frame centre, for the cheat selector and nothing else
    for d in rows:
        gfs = sorted(d["gctr"])
        d["gfc"] = {f: d["gctr"][min(gfs, key=lambda x: abs(x - f))] for f in gfs}

    f0 = lambda d: YE.window(d["n"], args.drop_tail)               # noqa: E731

    print("=" * 76)
    print("val videos = %d   selectors run on the cx/cy/area/conf already in the"
          " cache" % len(rows))
    print()
    print("%-10s %10s %10s %10s %10s   %s" % ("selector", "const", "traj",
                                              "d const", "d traj", "note"))
    base_c = base_t = None
    per = {}
    for mode in MODES:
        cs, ts = [], []
        for d in rows:
            a, b = f0(d)
            ser, gts = [], []
            for fi, dd in zip(d["frames"], d["dets"]):
                ps = [x for x in dd if x["cls"] == "person"]
                gc = d["gfc"].get(int(fi))
                v = pick(ps, d["ax"], mode, gc)
                if v is not None:
                    ser.append((int(fi), v))
                gts.append(gc)
            if not ser:
                cs.append(f1(d, np.full(b - a, 0.5), args.drop_tail))
                ts.append(cs[-1])
                continue
            sid = np.array([s[0] for s in ser], dtype=np.float64)
            sv = np.array([s[1] for s in ser], dtype=np.float64)
            cs.append(f1(d, np.full(b - a, float(np.median(sv))), args.drop_tail))
            ts.append(f1(d, YE.densify(sid, sv, b - a, b - a), args.drop_tail))
        mc, mt = float(np.mean(cs)), float(np.mean(ts))
        per[mode] = (mc, mt)
        if mode == "pool":
            base_c, base_t = mc, mt
        print("%-10s %10.4f %10.4f %+10.4f %+10.4f   %s"
              % (mode, mc, mt, mc - base_c, mt - base_t,
                 "CHEAT (sees GT)" if mode in CHEAT else ""))
    print()
    print("  pool is the shipped arm (conf*area weighted); its const number must"
          " reproduce 0.4214 ungated / 0.4302 gated")
    print("  largest / conf / central are free, GT-blind selectors")
    print()
    # how often do the free selectors even DISAGREE with the pool?
    print("--- do the free selectors pick a different person than pool? --------")
    for mode in ["largest", "conf", "central"]:
        dis = tot = 0
        for d in rows:
            for fi, dd in zip(d["frames"], d["dets"]):
                ps = [x for x in dd if x["cls"] == "person"]
                if len(ps) < 2:
                    continue
                tot += 1
                gc = d["gfc"].get(int(fi))
                if abs(pick(ps, d["ax"], mode, gc) - pick(ps, d["ax"], "pool", gc)) > 1e-9:
                    dis += 1
        print("  %-8s differs from pool on %5d / %5d frames (%.1f%%)"
              % (mode, dis, tot, 100.0 * dis / tot if tot else 0.0))
    print()
    free = [m for m in MODES if m not in CHEAT and m != "pool"]
    best_free = max(per[m][0] for m in free)
    print("best GT-blind selector (const): %.4f  vs pool %.4f  (%+.4f)"
          % (best_free, base_c, best_free - base_c))
    print("CHEAT selector  near_gt (const): %.4f  vs pool %.4f  (%+.4f)"
          % (per["near_gt"][0], base_c, per["near_gt"][0] - base_c))
    print("CHEAT selector  near_gt (traj) : %.4f  vs pool %.4f  (%+.4f)"
          % (per["near_gt"][1], base_c, per["near_gt"][1] - base_c))
    print()
    print("  oracle over selectors, const = %.4f   traj = %.4f"
          % (max(per[m][0] for m in MODES), max(per[m][1] for m in MODES)))

    if args.table:
        print()
        print("%-22s %5s %6s %6s %9s %9s" % ("video", "ndet", "clutt",
                                             "|c-.5|", "f1 pool", "f1 ng"))
        for d in rows:
            a, b = f0(d)
            out = {}
            for mode in ["pool", "near_gt"]:
                ser = []
                for fi, dd in zip(d["frames"], d["dets"]):
                    ps = [x for x in dd if x["cls"] == "person"]
                    v = pick(ps, d["ax"], mode, d["gfc"].get(int(fi)))
                    if v is not None:
                        ser.append(v)
                out[mode] = (f1(d, np.full(b - a, float(np.median(ser))),
                                args.drop_tail) if ser else 0.0)
            print("%-22s %5d %6.2f %6.3f %9.4f %9.4f"
                  % (d["vid"], sum(len(x) for x in d["dets"]), d["clutt"],
                     abs(np.median([pick([y for y in dd if y["cls"] == "person"],
                                         d["ax"], "pool", None)
                                    for dd in d["dets"] if any(
                                        y["cls"] == "person" for y in dd)] or [0.5]) - 0.5),
                     out["pool"], out["near_gt"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
