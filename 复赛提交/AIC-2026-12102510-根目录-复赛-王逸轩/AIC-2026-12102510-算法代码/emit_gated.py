#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Overlay the red (per-video constant) centre onto an existing submission.

Why an overlay and not a rewrite of baseline_cv.py
-------------------------------------------------
`baseline_cv.py` decides the kept-frame set and the crop size *inside* its
per-video loop, and both of those were verified byte-for-byte against the file
we already shipped (`--keep-ratio 1.0 --drop-tail 0.08` == `YE.window()`).
Changing that loop to also carry a *global* rank gate would mean a two-pass
driver and put the verified frame set back on the table. This script instead
takes the incumbent predictions file as an input: every video keeps the exact
frame list and the exact `w` it has today, and the only thing that ever changes
is the free-axis offset. Single-variable diff, trivially auditable:

    python emit_gated.py --base predictions_hedge.jsonl \\
        --cache yolo_val_cache.json --metadata val_metadata.json \\
        --out predictions_gated_val.jsonl --q 0.30

Then `compare_runs.py base.jsonl out.jsonl` reports exactly the videos that
moved and by how many pixels.

The arm
-------
Per video, pool every `person` detection with weight `conf * area` (this is the
FLAT arm -- track identity is deliberately thrown away, it only added noise),
along the free axis. Take the per-video MEDIAN of that per-frame pooled centre
and emit it as a per-video constant. This is worth +0.0443 val F1 over the
centred-box champion on its own.

The gate
--------
That gain is not free: it is negative on videos whose ground-truth framing is
already near the centre. So videos that look cluttered -- many people on screen,
where a pooled centre is meaningless -- fall back to the incumbent's centred
box. Clutter is the mean number of person detections per sampled frame, and the
gate is an ABSOLUTE cut (--threshold), not a top-q rank. Two reasons:

  * val says the harm is a *tail* phenomenon -- red delta by clutter bin is
    +0.187 at 1.0-1.5 but -0.030 at 3-6 and -0.084 at 6+, i.e. the good videos
    live at moderate clutter. Cutting a fixed fraction is not the same thing as
    cutting the harmful tail.
  * the test set is only half as crowded as val (mean 0.95 vs 2.11 person
    detections per sampled frame). val's own q=0.30 rank cut sits at 2.014 and
    would gate ~10% of the test set, while the rank form would still gate 30% --
    slicing straight into the band that pays. Transferring val's bin->delta onto
    the test distribution: rank q=0.30 +0.027, no gate +0.050, abs 2.0 +0.051.

The fallback is what the file already contains, so a gated video is *exactly*
the incumbent submission -- the downside of this whole change is bounded by the
champion, and cluttered videos are where the arm actually loses
(corr(clutter, red delta) = -0.32).
"""
import argparse
import json
import os
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from stage_data import crop_size, free_axis                       # noqa: E402

POWER = 1.0

# Weight files the inference pipeline actually loads. Mirrors baseline_cv.py's
# own fallback list, for the same reason: the first path that exists (or that
# has no "/" and so is fetched by ultralytics) is the one that gets loaded, so
# that is the only file `model_size_mb` may count.
DEFAULT_WEIGHTS = [
    "/mnt/data/ML/模型实践与复现/YOLO/yolo11n.pt",
    "/mnt/data/ML/模型实践与复现/YOLO/yolov8n.pt",
    "yolo11n.pt",
]

MB = 1024 * 1024


def resolve_weights():
    """First entry of DEFAULT_WEIGHTS that would actually be loaded."""
    for c in DEFAULT_WEIGHTS:
        if os.path.exists(c) or "/" not in c:
            return c
    return None


def weight_size(path):
    """(decompressed bytes, on-disk bytes) of one weight file.

    The submission spec asks for the weights' *decompressed* total. A `.pt`
    checkpoint is itself a zip container, so its size on disk is not the size of
    the weights inside it; when the file is a zip we sum the uncompressed entry
    sizes, which is what "解压后" means here.
    """
    raw = os.path.getsize(path)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            return sum(i.file_size for i in z.infolist()), raw
    return raw, raw


def measure_model_size_mb(paths):
    """(decompressed MB, per-file breakdown lines) for the given weights."""
    total = 0
    lines = []
    for p in paths:
        if not os.path.exists(p):
            raise SystemExit("weight file not found: %s" % p)
        dec, raw = weight_size(p)
        total += dec
        lines.append("    %s\n      %d B decompressed (on disk %d B)"
                     % (p, dec, raw))
    return round(total / MB, 2), lines


def person_series(rec, ax):
    """Per sampled frame: (pooled person centre, n person detections)."""
    key = "cx" if ax == "x" else "cy"
    centres, counts = {}, {}
    for fi, dets in zip(rec["frames"], rec["dets"]):
        num = den = 0.0
        n = 0
        for d in dets:
            if d["cls"] != "person":
                continue
            n += 1
            w = float(d["conf"]) * max(float(d["area"]), 1e-9) ** POWER
            if w <= 0:
                continue
            num += w * float(d[key])
            den += w
        if den > 0:
            centres[fi] = min(1.0, max(0.0, num / den))
        counts[fi] = n
    return centres, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="incumbent predictions jsonl")
    ap.add_argument("--cache", required=True, help="detection cache json")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--q", type=float, default=0.30,
                    help="top fraction of videos by clutter that keep the "
                         "incumbent centred box; 0 disables the gate")
    ap.add_argument("--threshold", type=float, default=None,
                    help="ABSOLUTE clutter cut (mean person dets per sampled "
                         "frame) instead of the --q rank. Prefer this: the harm "
                         "is concentrated in the far tail of the clutter "
                         "distribution, not in a fixed top fraction of it, and "
                         "the test set is only half as crowded as val (see "
                         "RUNBOOK 4.10).")
    ap.add_argument("--model-size-mb", type=float, default=None,
                    help="value written to every line's model_size_mb field. "
                         "Default: measured from --weights (decompressed total).")
    ap.add_argument("--weights", action="append", default=None,
                    help="a weight file the inference pipeline loads; "
                         "repeatable. Default: the same single file "
                         "baseline_cv.py would pick.")
    args = ap.parse_args()

    wpaths = args.weights
    if wpaths is None:
        auto = resolve_weights()
        wpaths = [auto] if auto else []
    if args.model_size_mb is None:
        if not wpaths:
            raise SystemExit("cannot measure model_size_mb: no weight file "
                             "found, pass --weights or --model-size-mb")
        size_mb, breakdown = measure_model_size_mb(wpaths)
    else:
        size_mb, breakdown = args.model_size_mb, []

    base = [json.loads(l) for l in open(args.base, "r", encoding="utf-8")
            if l.strip()]
    cache = json.load(open(args.cache, "r", encoding="utf-8"))
    meta = json.load(open(args.metadata, "r", encoding="utf-8"))

    rows = []
    for rec in base:
        vid = rec["video_id"]
        if vid not in cache:
            raise SystemExit("no detections cached for %s" % vid)
        m = meta.get(vid) or cache[vid]
        W, H = int(m["W"]), int(m["H"])
        tw, th = rec["targetRatioWH"]
        ax = free_axis(W, H, tw, th)
        centres, counts = person_series(cache[vid], ax)
        stat = (float(np.mean([counts[f] for f in sorted(counts)]))
                if counts else 0.0)
        cs = np.array(sorted(centres.values()), dtype=np.float64)
        rows.append({"rec": rec, "ax": ax, "W": W, "H": H,
                     "cw": crop_size(W, H, tw, th)[0],
                     "ch": crop_size(W, H, tw, th)[1],
                     "c": float(np.median(cs)) if len(cs) else None,
                     "stat": stat, "n_person_frames": len(cs)})

    stats = np.array([r["stat"] for r in rows], dtype=np.float64)
    if args.threshold is not None:
        thr = args.threshold
    else:
        thr = np.quantile(stats, 1 - args.q) if args.q > 0 else np.inf
    # rank, not value: ties at the threshold are kept, same as hard_vec in
    # yolo_gate.py, so the val numbers transfer exactly.
    for r in rows:
        r["gated"] = not (r["c"] is not None and r["stat"] <= thr)

    out = []
    moved = 0
    for r in rows:
        rec, preds = r["rec"], r["rec"]["predictions"]
        boxes = [p["bboxes"] for p in preds]
        cw, ch = r["cw"], r["ch"]
        if boxes[0][2] != cw:
            raise SystemExit("%s: base w=%d but crop_size says %d -- the base "
                             "file was not produced by this crop_size"
                             % (rec["video_id"], boxes[0][2], cw))
        if r["gated"] or r["ax"] is None:
            new = boxes
        elif r["ax"] == "x":
            travel = max(0, r["W"] - cw)
            o = int(round(float(np.clip(r["c"] * r["W"] - cw / 2.0, 0, travel))))
            new = [[o, 0, cw] for _ in boxes]
        else:
            travel = max(0, r["H"] - ch)
            o = int(round(float(np.clip(r["c"] * r["H"] - ch / 2.0, 0, travel))))
            new = [[0, o, cw] for _ in boxes]
        moved += sum(1 for a, b in zip(boxes, new) if a != b)
        out.append({"video_id": rec["video_id"],
                    "targetRatioWH": rec["targetRatioWH"],
                    "model_size_mb": size_mb,
                    "predictions": [{"frame": p["frame"], "bboxes": b}
                                    for p, b in zip(preds, new)]})

    with open(args.out, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")

    ng = sum(1 for r in rows if r["gated"])
    print("wrote %s" % args.out)
    print("  videos=%d  gated(=incumbent box)=%d  red=%d  frames changed=%d/%d"
          % (len(rows), ng, len(rows) - ng, moved,
             sum(len(r["rec"]["predictions"]) for r in rows)))
    print("  model_size_mb = %s" % size_mb)
    for line in breakdown:
        print(line)
    print("  clutter cut (mean person dets / sampled frame) = %.3f  [%s]"
          % (thr, "absolute --threshold" if args.threshold is not None
             else "rank --q=%.2f" % args.q))
    return 0


if __name__ == "__main__":
    sys.exit(main())
