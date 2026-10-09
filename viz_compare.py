#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Side-by-side visual check of the crop box, per the request:

    left   = the source clip, untouched
    middle = source + the GT crop box
    right  = source + the strategy's box, WITH the GT box drawn too
             so the two can be compared frame by frame

Colours (BGR):  GT = green, strategy prediction = red,
                the shipped centred box = thin blue reference.

"the strategy" defaults to `gated` -- the arm actually shipped in
submission_gated.zip: pool every person detection per frame (weight conf*area),
take the per-video median and emit it as a per-video constant, then push the
whole video back to the centred box when it looks cluttered (mean person
detections per sampled frame > --threshold). val F1 0.4302 vs the centred
champion's 0.3688, and 32.85 vs 31.05 on the real leaderboard.

Other arms: `flat_person` = the same red box without the gate (0.4214),
`flat_person_frame` = per-frame trajectory (0.3910), `centred` = the previous
champion (0.3688).

The default cache is the DETECTION cache, because that is what the shipped
pipeline (emit_gated.py) consumes -- plain model() output, no tracker. The
older ByteTrack cache drops 19% of the person detections and measures 0.008
worse, so it is only kept for reproducing old numbers.

Videos are picked to cover the three cases that matter: GT far off centre
(strategy should move), GT centred (strategy should stay put), and the videos
where the strategy loses the most.

Usage
    python viz_compare.py                 # ~6 videos, mp4 + one contact sheet
    python viz_compare.py --n 8 --scale 2
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
from yolo_gate import frame_counts                                 # noqa: E402


def cache_series(rec, ax):
    """Per video: (sampled frames with a person, pooled person centre,
    mean person detections per sampled frame).

    Handles both cache formats. `dets` is the detection cache the shipped
    pipeline uses; `tracks` is the older ByteTrack one, kept only so old
    commands still reproduce (it drops 19% of person detections -- see RUNBOOK
    4.10 -- so its numbers are lower).
    """
    if "dets" in rec:
        cts, cnt = person_series(rec, ax)
        fs = sorted(cts)
        return (np.array(fs, dtype=np.float64),
                np.array([cts[f] for f in fs], dtype=np.float64),
                float(np.mean(list(cnt.values()))) if cnt else 0.0)
    fi, ce, cn = frame_counts(rec["tracks"], ax)
    return fi, ce, (float(np.mean(cn)) if len(cn) else 0.0)


def gt_map(g, W, H, tw, th, ax):
    """{frame: (free-axis centre, exact xyxy box)} -- centre for comparing
    against the strategy, box for drawing and scoring exactly as the judge does."""
    out = {}
    for e in g["predictions"]:
        x, y, w = e["bboxes"]
        f = int(e["frame"])
        if f in out:
            continue
        c = ((x + w / 2.0) / W) if ax == "x" \
            else ((y + (w * float(th) / float(tw)) / 2.0) / H)
        out[f] = (c, box_from_triplet(x, y, w, tw, th))
    return out


def draw(img, box, colour, thickness=2):
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    import cv2
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, thickness)


def label(img, text, colour):
    import cv2
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                colour, 1, cv2.LINE_AA)


def main():
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=os.path.join(HERE, "val_gt.jsonl"))
    ap.add_argument("--metadata", default=os.path.join(HERE, "val_metadata.json"))
    ap.add_argument("--video-dir", default=os.path.join(HERE, "val_video"))
    ap.add_argument("--cache", default=os.path.join(HERE, "yolo_val_cache.json"),
                    help="detection cache (default, = what emit_gated.py ships) "
                         "or the older yolo_track_cache.json")
    ap.add_argument("--out", default=os.path.join(HERE, "viz"))
    ap.add_argument("--drop-tail", type=float, default=0.08)
    ap.add_argument("--strategy", default="gated",
                    choices=["gated", "flat_person", "flat_person_frame",
                             "centred"],
                    help="gated = shipped (red box, clutter fallback to "
                         "centred, 0.4302); flat_person = red without the gate "
                         "(0.4214); flat_person_frame = per-frame trajectory "
                         "(0.3910); centred = the previous champion (0.3688)")
    ap.add_argument("--threshold", type=float, default=2.0,
                    help="clutter cut for --strategy gated: mean person "
                         "detections per sampled frame above which the video "
                         "falls back to the centred box (RUNBOOK 4.10)")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--scale", type=float, default=2.0)
    args = ap.parse_args()

    def strategy_centres(d, a):
        """The per-frame centre array a given arm would emit for one video."""
        f0, f1 = YE.window(d["n"], a.drop_tail)
        if a.strategy == "centred":
            return np.full(f1 - f0, 0.5)
        if not len(d["fi"]) or (a.strategy == "gated" and d["gated"]):
            # cluttered, or no person found: the gate says fall back to centred
            return np.full(f1 - f0, 0.5)
        if a.strategy in ("flat_person", "gated"):
            # per-video constant: the measured best (jitter is pure loss)
            return np.full(f1 - f0, float(np.median(d["ce"])))
        return YE.densify(d["fi"], d["ce"], d["n"], f1 - f0)

    cache = json.load(open(args.cache, "r", encoding="utf-8"))
    meta = json.load(open(args.metadata, "r", encoding="utf-8"))
    gt = {r["video_id"]: r for r in
          (json.loads(l) for l in open(args.gt, "r", encoding="utf-8") if l.strip())}
    os.makedirs(args.out, exist_ok=True)

    # --- rank the videos so the picked set covers the interesting cases
    rows = []
    for vid, g in gt.items():
        if vid not in meta or vid not in cache:
            continue
        W, H, n = meta[vid]["W"], meta[vid]["H"], int(meta[vid]["n_frames"])
        tw, th = g["targetRatioWH"]
        ax = free_axis(W, H, tw, th)
        gcs = gt_map(g, W, H, tw, th, ax)
        if not gcs:
            continue
        fi, ce, stat = cache_series(cache[vid], ax)
        d = {"vid": vid, "W": W, "H": H, "n": n, "tw": tw, "th": th, "ax": ax,
             "gcs": gcs, "mu_gt": float(np.median([c for c, _ in gcs.values()])),
             "n_gt": len(gcs), "fi": fi, "ce": ce, "stat": stat}
        d["pred"] = float(np.median(ce)) if len(ce) else 0.5
        # the gate is a per-video decision: too cluttered to trust the pooled
        # centre, or no person at all -> keep the centred box
        d["gated"] = (not len(ce)) or (stat > args.threshold)
        rows.append(d)
    # F1 of the strategy vs GT for the *whole* video, to find its worst cases
    for d in rows:
        cw, ch = crop_size(d["W"], d["H"], d["tw"], d["th"])
        f0, f1 = YE.window(d["n"], args.drop_tail)
        centres = strategy_centres(d, args)
        p = YE.boxes_for({"n_frames": d["n"], "_drop_tail": args.drop_tail},
                         d["ax"], cw, ch, d["W"], d["H"], centres)
        S = 0.0
        for e in p:
            f = int(e["frame"])
            if f not in d["gcs"]:
                continue
            S += iou_xyxy(box_from_triplet(*e["bboxes"], d["tw"], d["th"]),
                          d["gcs"][f][1])
        den = len(p) + d["n_gt"]
        d["f1"] = 2 * S / den if den else 0.0
        d["sm"] = centres
        # the shipped centred box on the same frames, for a like-for-like delta
        d["f1_ctrl"] = 0.0
        pc = YE.boxes_for({"n_frames": d["n"], "_drop_tail": args.drop_tail},
                          d["ax"], cw, ch, d["W"], d["H"],
                          np.full(f1 - f0, 0.5))
        Sc = sum(iou_xyxy(box_from_triplet(*e["bboxes"], d["tw"], d["th"]),
                          d["gcs"][int(e["frame"])][1])
                 for e in pc if int(e["frame"]) in d["gcs"])
        d["f1_ctrl"] = 2 * Sc / den if den else 0.0
        # what the red box would have scored on this video WITH the gate open,
        # so the gate's effect is visible per video (non-gated videos: same)
        pr = (YE.boxes_for({"n_frames": d["n"], "_drop_tail": args.drop_tail},
                           d["ax"], cw, ch, d["W"], d["H"],
                           np.full(f1 - f0, float(np.median(d["ce"]))))
              if len(d["ce"]) else pc)
        Sr = sum(iou_xyxy(box_from_triplet(*e["bboxes"], d["tw"], d["th"]),
                          d["gcs"][int(e["frame"])][1])
                 for e in pr if int(e["frame"]) in d["gcs"])
        d["f1_red"] = 2 * Sr / den if den else 0.0
        d["saved"] = d["f1"] - d["f1_red"]
    if rows:
        print("arm %-18s mean F1 over %d val videos = %.4f   (centred = %.4f)"
              % (args.strategy, len(rows),
                 float(np.mean([d["f1"] for d in rows])),
                 float(np.mean([d["f1_ctrl"] for d in rows]))))

    by_off = sorted(rows, key=lambda d: -abs(d["mu_gt"] - 0.5))
    by_f1 = sorted(rows, key=lambda d: d["f1"])
    # the most cluttered videos come first: under --strategy gated those are the
    # ones where the red box is refused, which is the whole point of the arm
    # ordered by how much the gate SAVED (not by clutter): a gated video the red
    # box would have handled fine shows nothing, one it would have botched is
    # the whole argument for the gate
    by_gate = sorted([d for d in rows if d["gated"]], key=lambda d: -d["saved"])
    picked, seen = [], set()
    pool = (by_gate[:3] + by_off[:args.n // 2 + 2]
            + by_f1[:args.n - args.n // 2 + 2] + by_off + by_f1 + by_gate)
    for d in pool:
        if len(picked) >= args.n:
            break
        if d["vid"] not in seen:
            seen.add(d["vid"])
            picked.append(d)
    picked.sort(key=lambda d: -(abs(d["mu_gt"] - 0.5)))
    print("strategy = %s ; picked %d videos" % (args.strategy, len(picked)))

    sc = args.scale
    sheet = []
    for d in picked:
        path = os.path.join(args.video_dir, d["vid"] + ".mp4")
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, bgr = cap.read()
            if not ok or bgr is None:
                break
            frames.append(bgr)
        cap.release()
        if not frames:
            print("  [warn] cannot read %s" % path)
            continue
        W, H = d["W"], d["H"]
        cw, ch = crop_size(W, H, d["tw"], d["th"])
        f0, f1 = YE.window(d["n"], args.drop_tail)
        cx0 = int(round((W - cw) / 2.0))
        cy0 = int(round((H - ch) / 2.0))
        out_path = os.path.join(args.out, "%s_%s.mp4" % (args.strategy, d["vid"]))
        vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             float(meta[d["vid"]].get("fps") or 30.0),
                             (int(3 * W * sc), int(H * sc)))
        mid_frame = None
        for f, bgr in enumerate(frames):
            # the strategy is only defined on the kept window; the tail keeps the
            # last window value so the drawing never runs off the array
            fi_ = min(f, len(d["sm"]) - 1)
            # round exactly like emit_gated.boxes_from_offsets does, otherwise a
            # gated video (which should BE the centred box) draws 0.5px off and
            # shows a sliver of blue that is not in the submitted file
            if d["ax"] == "x":
                ox = int(round(float(np.clip(d["sm"][fi_] * W - cw / 2.0,
                                             0, W - cw))))
                pbox = box_from_triplet(ox, 0, cw, d["tw"], d["th"])
            elif d["ax"] == "y":
                oy = int(round(float(np.clip(d["sm"][fi_] * H - ch / 2.0,
                                             0, H - ch))))
                pbox = box_from_triplet(0, oy, cw, d["tw"], d["th"])
            else:
                pbox = box_from_triplet(cx0, cy0, cw, d["tw"], d["th"])
            cbox = box_from_triplet(cx0, cy0, cw, d["tw"], d["th"])
            gbox = None
            if f in d["gcs"]:
                gbox = d["gcs"][f][1]
            L, M, R = bgr.copy(), bgr.copy(), bgr.copy()
            label(L, "1 source", (255, 255, 255))
            label(M, "2 source + GT box (green)", (0, 255, 0))
            if args.strategy == "gated":
                tag = ("3 GATED -> centred box (clutter %.2f > %.1f)"
                       % (d["stat"], args.threshold)) if d["gated"] else \
                      ("3 red box, gate open (clutter %.2f <= %.1f)"
                       % (d["stat"], args.threshold))
            else:
                tag = "3 %s (red) + GT (green)" % args.strategy
            label(R, tag + ("  [+ blue = centred]" if args.strategy != "centred"
                            else ""), (0, 0, 255))
            if gbox is not None:
                draw(M, gbox, (0, 255, 0), 2)
                draw(R, gbox, (0, 255, 0), 2)
            if args.strategy != "centred":
                draw(R, cbox, (255, 120, 0), 1)
            draw(R, pbox, (0, 0, 255), 2)
            strip = np.hstack([L, M, R])
            if sc != 1.0:
                strip = cv2.resize(strip, None, fx=sc, fy=sc,
                                   interpolation=cv2.INTER_NEAREST)
            vw.write(strip)
            if f == min(len(frames) - 1, f0 + (f1 - f0) // 2):
                mid_frame = strip.copy()
        vw.release()
        if mid_frame is not None:
            sheet.append((mid_frame, d))
        print("  %s  n=%d  |GT-0.5|=%.3f  clutter=%.2f%s  red-would-be=%.3f  "
              "F1=%.3f (centred %.3f, %+.3f)  gate saves %+.3f  -> %s"
              % (d["vid"], len(frames), abs(d["mu_gt"] - 0.5), d["stat"],
                 " GATED" if d["gated"] else "      ", d["f1_red"],
                 d["f1"], d["f1_ctrl"], d["f1"] - d["f1_ctrl"], d["saved"],
                 os.path.basename(out_path)))

    if sheet:
        rows_img = []
        wmax = max(s.shape[1] for s, _ in sheet)
        for s, d in sheet:
            pad = np.zeros((s.shape[0], wmax - s.shape[1], 3), dtype=s.dtype)
            rows_img.append(np.hstack([s, pad]))
        cv2.imwrite(os.path.join(args.out, "sheet_%s.png" % args.strategy),
                    np.vstack(rows_img))
        print("wrote %s"
              % os.path.join(args.out, "sheet_%s.png" % args.strategy))
    return 0


if __name__ == "__main__":
    sys.exit(main())
