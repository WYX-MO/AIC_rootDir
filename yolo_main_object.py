#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pick ONE main object per video (by how long it is on screen x how big it is)
and use it for the crop -- the "main object track" strategy.

Why this is a different idea from everything in yolo_experiment.py
-----------------------------------------------------------------
yolo_experiment.py treats every frame independently: argmax (what
baseline_cv.py does) or a weighted mean over that frame's boxes. Both re-decide
*which* object matters on every single frame, and the frames disagree -- which
is exactly where the 0.117 frame-to-frame jitter comes from (near-tied
top1/top2 scores on 22% of frames cause 2.7x larger jumps).

This script instead makes the decision ONCE per video, on evidence accumulated
over the whole video:
    the object that is present for the largest fraction of the video AND
    occupies the largest area
is declared the main object, and its boxes (and only its boxes) define the crop.

Identity comes from ByteTrack (`model.track(..., persist=True)`), so "present
for X% of the video" means the *same instance* re-identified across frames, not
merely "a box of the same class".

The factual backdrop that shapes what to expect
-----------------------------------------------
The GT crop barely moves inside a video: within-video centre sd 0.025 vs
between-video sd 0.108 (68% of videos below 0.03, only 4% above 0.10). So the
GT is essentially ONE static framing per video, and YOLO's 0.117 jitter is 4.7x
the GT's own motion. Two consequences:

  * a per-frame track can at best imitate a near-constant;
  * what a useful signal must supply is a robust PER-VIDEO offset estimate.

Hence the two arms below are reported separately: `const` (collapse the main
object's trajectory to one per-video offset, then shrink) and `frame` (keep the
per-frame trajectory, then shrink). If the main-object idea helps, it should
help mostly through `const`.

Usage
-----
    python yolo_main_object.py --build-cache     # one tracked pass, cached
    python yolo_main_object.py --eval
    python yolo_main_object.py --eval --dump-best
"""

import os
import sys
import json
import math
import time
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from stage_data import crop_size, free_axis, probe_one          # noqa: E402
from eval_local import iou_xyxy, box_from_triplet               # noqa: E402
import yolo_experiment as YE                                    # noqa: E402

DEFAULT_WEIGHTS = "/mnt/data/ML/模型实践与复现/YOLO/yolo11n.pt"
DEFAULT_TRACKER = "bytetrack.yaml"


# --------------------------------------------------------------------------
# pass 1 -- tracked inference
# --------------------------------------------------------------------------
def reset_tracker(model):
    """ByteTrack keeps its state across calls when persist=True. Clear it so
    track ids do not leak from one video into the next (which would let two
    different videos' objects be treated as the same instance)."""
    try:
        for t in model.predictor.trackers:
            t.reset()
    except Exception:
        pass


def build_cache(args):
    import cv2
    from ultralytics import YOLO

    index = json.load(open(args.index, "r", encoding="utf-8"))
    meta = json.load(open(args.metadata, "r", encoding="utf-8"))
    model = YOLO(args.weights)
    cache = {}
    t0 = time.time()
    for k, rec in enumerate(index):
        vid = rec["video_id"]
        m = meta.get(vid) or probe_one(os.path.join(args.video_dir, vid + ".mp4"))
        if not m or not m.get("n_frames"):
            continue
        W, H, n = m["W"], m["H"], int(m["n_frames"])
        cap = cv2.VideoCapture(os.path.join(args.video_dir, vid + ".mp4"))
        frames, keep = [], []
        for i in range(0, n, args.stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, bgr = cap.read()
            if not ok or bgr is None:
                continue
            frames.append(bgr)
            keep.append(i)
        cap.release()
        if not frames:
            continue

        reset_tracker(model)
        # one call over the whole list => the tracker sees a continuous stream
        res = model.track(frames, persist=True, verbose=False, imgsz=args.imgsz,
                          device=args.device, tracker=args.tracker)
        tracks = {}
        for f, r in zip(keep, res):
            b = getattr(r, "boxes", None)
            if b is None or len(b) == 0 or b.id is None:
                continue
            xywh = b.xywh.cpu().numpy()
            cf = b.conf.cpu().numpy()
            ids = b.id.cpu().numpy().astype(int)
            cls = b.cls.cpu().numpy().astype(int)
            for (bx, by, bw, bh), c, tid, cl in zip(xywh, cf, ids, cls):
                tracks.setdefault(str(int(tid)), []).append(
                    [int(f), round(float(bx) / W, 5), round(float(by) / H, 5),
                     round(float(bw) * float(bh) / float(W * H), 6),
                     round(float(c), 4),
                     model.names.get(int(cl), str(int(cl)))])
        cache[vid] = {"W": W, "H": H, "n_frames": n,
                      "sampled": keep, "tracks": tracks}
        lens = sorted((len(v) for v in tracks.values()), reverse=True)[:3]
        print("  %s  %d sampled, %d tracks, top lengths %s"
              % (vid, len(keep), len(tracks), lens))
        if (k + 1) % 10 == 0 or k + 1 == len(index):
            print("  ... %d/%d (%.0fs)" % (k + 1, len(index), time.time() - t0))

    with open(args.cache, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    print("\nwrote %s (%d videos, %.1f MB)"
          % (args.cache, len(cache), os.path.getsize(args.cache) / 1e6))
    return 0


# --------------------------------------------------------------------------
# scoring a track
# --------------------------------------------------------------------------
def track_stats(entries, n_sampled):
    """(presence fraction, mean area fraction, integrated area, mean conf)."""
    present = len(entries)
    areas = [e[3] for e in entries]
    confs = [e[4] for e in entries]
    return (present / max(1, n_sampled),
            float(np.mean(areas)), float(np.sum(areas)),
            float(np.mean(confs)))


def pick_main(tracks, n_sampled, rule):
    """rank = declared main-object criterion. 'which object owns the video'."""
    if not tracks:
        return None
    best, best_s = None, -1.0
    for tid, entries in tracks.items():
        pres, marea, integ, mconf = track_stats(entries, n_sampled)
        if rule == "pres*area":      s = pres * marea
        elif rule == "pres*maxarea": s = pres * max(e[3] for e in entries)
        elif rule == "integrated":   s = integ            # sum of area over time
        elif rule == "presence":     s = pres
        elif rule == "meanarea":     s = marea
        elif rule == "pres*area*conf": s = pres * marea * mconf
        else:
            raise ValueError(rule)
        if s > best_s:
            best, best_s = tid, s
    return best


def track_centres(entries, ax):
    """(frame, centre along the free axis) for one track."""
    fi = np.array([e[0] for e in entries], dtype=np.float64)
    ce = np.array([e[1] if ax == "x" else e[2] for e in entries], dtype=np.float64)
    order = np.argsort(fi)
    return fi[order], ce[order]


# --------------------------------------------------------------------------
# pass 2 -- evaluation
# --------------------------------------------------------------------------
def evaluate(args):
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
                  "mu_gt": float(np.median(cs)),
                  "gt_sd": float(np.std(cs)) if len(cs) > 1 else 0.0,
                  "rec": cache[vid]}
    vids = sorted(D)

    def f1_video(vid, centre_map, per_frame=None):
        d = D[vid]
        gm = {}
        for e in gt[vid]["predictions"]:
            fi = int(e["frame"])
            if fi not in gm:
                gm[fi] = box_from_triplet(*e["bboxes"], d["tw"], d["th"])
        cw, ch = crop_size(d["W"], d["H"], d["tw"], d["th"])
        f0, f1 = YE.window(d["n"], args.drop_tail)
        rec = {"n_frames": d["n"], "_drop_tail": args.drop_tail}
        if per_frame is None:
            centres = np.full(f1 - f0, centre_map[vid])
        else:
            centres = per_frame[vid]
        p = YE.boxes_for(rec, d["ax"], cw, ch, d["W"], d["H"], centres)
        S = 0.0
        for e in p:
            fi = int(e["frame"])
            if fi in gm:
                S += iou_xyxy(box_from_triplet(*e["bboxes"], d["tw"], d["th"]), gm[fi])
        den = len(p) + len(gm)
        return 2 * S / den if den else 0.0

    ctrl = {v: f1_video(v, {v: 0.5}) for v in vids}

    def const_arms(med_fn, label):
        meds = {v: med_fn(v) for v in vids}
        for b in (0.37, 0.5, 1.0):
            cm = {v: 0.5 + b * (meds[v] - 0.5) for v in vids}
            yield ("%s  const β=%.2f" % (label, b),
                   {v: f1_video(v, cm) for v in vids}, None)

    def frame_arm(traj_fn, label, alpha):
        per = {}
        for v in vids:
            d = D[v]
            f0, f1 = YE.window(d["n"], args.drop_tail)
            per[v] = traj_fn(v, f1 - f0)
        return ("%s  frame α=%.2f" % (label, alpha),
                {v: f1_video(v, None, per) for v in vids}, None)

    # ---- the new arms: main object by duration x size
    print("\n--- main-object criterion: what does it pick? ---")
    for rule in args.rules:
        nobj, cls_cnt = [], {}
        pres, areas = [], []
        for v in vids:
            rec = D[v]["rec"]
            tid = pick_main(rec["tracks"], len(rec["sampled"]), rule)
            if tid is None:
                continue
            e = rec["tracks"][tid]
            p, a, _, _ = track_stats(e, len(rec["sampled"]))
            pres.append(p); areas.append(a)
            nobj.append(len(rec["tracks"]))
            c = e[0][5]
            cls_cnt[c] = cls_cnt.get(c, 0) + 1
        top = ", ".join("%s %d" % (k, v) for k, v in
                        sorted(cls_cnt.items(), key=lambda t: -t[1])[:5])
        print("  %-16s tracks/video %.1f | chosen presence %.2f meanarea %.3f | %s"
              % (rule, np.mean(nobj), np.mean(pres), np.mean(areas), top))

    arms = []
    for rule in args.rules:
        def med_fn(v, rule=rule):
            rec = D[v]["rec"]
            tid = pick_main(rec["tracks"], len(rec["sampled"]), rule)
            if tid is None:
                return 0.5
            fi, ce = track_centres(rec["tracks"][tid], D[v]["ax"])
            return float(np.median(ce))

        def traj_fn(v, win, rule=rule):
            rec = D[v]["rec"]
            tid = pick_main(rec["tracks"], len(rec["sampled"]), rule)
            if tid is None:
                return np.full(win, 0.5)
            fi, ce = track_centres(rec["tracks"][tid], D[v]["ax"])
            return YE.densify(fi, ce, D[v]["n"], win)

        arms += list(const_arms(med_fn, rule))
        arms.append(frame_arm(traj_fn, rule, 1.0))

    # ---- references: control, current argmax track, weighted-mean-person
    def ref_argmax_med(v):
        import yolo_experiment as YE2
        rec = cache[v]
        fidx, ce, sc, nm = YE2.centres_from_cache(rec, D[v]["ax"], False)
        return float(np.median(ce))
    arms.append(("REF control (centred)", ctrl, None))

    print("\n--- val F1 (frames fixed at 0..n-round(%.2f n)-1) ---" % args.drop_tail)
    print("  %-40s %8s %10s" % ("arm", "F1", "Δ vs ctrl"))
    rows = {}
    for name, per, _ in arms:
        f1 = float(np.mean(list(per.values())))
        rows[name] = per
        print("  %-40s %.4f %+10.4f" % (name, f1, f1 - np.mean(list(ctrl.values()))))

    # ---- held-out: β fitted on half, evaluated on the other half
    print("\n--- held-out: beta/alpha fitted on half the videos, scored on the other ---")
    rng = np.random.default_rng(0)
    ctrl_arr = np.array([ctrl[v] for v in vids])
    for rule in args.rules:
        rec_of = {}
        for v in vids:
            c = D[v]["rec"]
            tid = pick_main(c["tracks"], len(c["sampled"]), rule)
            if tid is None:
                rec_of[v] = None
                continue
            fi, ce = track_centres(c["tracks"][tid], D[v]["ax"])
            f0, f1 = YE.window(D[v]["n"], args.drop_tail)
            rec_of[v] = (fi, ce, YE.densify(fi, ce, D[v]["n"], f1 - f0))
        ds = []
        for _ in range(200):
            idx = rng.permutation(len(vids))
            te = [vids[i] for i in idx[:len(vids) // 2]]
            tr = [vids[i] for i in idx[len(vids) // 2:]]
            tr = [v for v in tr if rec_of[v] is not None]
            te = [v for v in te if rec_of[v] is not None]
            if len(tr) < 5 or not te:
                continue
            yy = np.array([float(np.median(rec_of[u][1])) for u in tr])
            gg = np.array([D[u]["mu_gt"] for u in tr])
            if yy.std() < 1e-9:
                continue
            b = float(np.cov(gg, yy, ddof=1)[0, 1] / np.cov(gg, yy, ddof=1)[1, 1])
            cm = {v: 0.5 + b * (float(np.median(rec_of[v][1])) - 0.5) for v in te}
            ds.append(np.mean([f1_video(v, cm) - ctrl[v] for v in te]))
        ds = np.array(ds)
        if len(ds):
            print("  %-40s held-out Δ %+.4f  wins %.1f%%"
                  % (rule + "  const β=LOO", ds.mean(), 100 * (ds > 0).mean()))

    if args.dump_best:
        best = max(args.rules, key=lambda r: np.mean(
            [v for v in rows.get(r + "  const β=0.50", {}).values()] or [0]))
        print("\nbest rule by unshrunk const F1: %s" % best)
    return 0


def main():
    ap = argparse.ArgumentParser(description="main-object selection for the crop")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--tracker", default=DEFAULT_TRACKER)
    ap.add_argument("--index", default=os.path.join(HERE, "val_index.json"))
    ap.add_argument("--gt", default=os.path.join(HERE, "val_gt.jsonl"))
    ap.add_argument("--video-dir", default=os.path.join(HERE, "val_video"))
    ap.add_argument("--metadata", default=os.path.join(HERE, "val_metadata.json"))
    ap.add_argument("--cache", default=os.path.join(HERE, "yolo_track_cache.json"))
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--drop-tail", type=float, default=0.08)
    ap.add_argument("--rules", default="pres*area,integrated,presence,meanarea,pres*maxarea")
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--eval", dest="do_eval", action="store_true")
    ap.add_argument("--dump-best", action="store_true")
    args = ap.parse_args()
    args.rules = [s.strip() for s in args.rules.split(",") if s.strip()]
    if args.build_cache:
        return build_cache(args)
    return evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
