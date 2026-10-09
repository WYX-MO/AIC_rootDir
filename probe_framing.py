#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The cheap decisive test before any training:

    is the training set's framing (cropRois) reproducible from what YOLO can
    see, plus a CONSTANT offset?

If yes, the 0.0606 pool bias we measured on val is a calibration constant --
subtract it and collect the +0.028 online for free, no model, no GPU.

If no (the framing is a slow drift YOLO cannot recover), then only a trained
estimator can help, and the ceiling is still +0.028.

Stage 1 (this script, no GPU): extract a few val-split clips out of the 33 zips
using the `<id>_<start>_<end>.mp4` naming + the id's first char, then report
the cropRois trajectory in the 300-high canonical space against YOLO person
boxes on the same frames.

Usage
    python probe_framing.py --recon          # shards + candidate rows, no extract
    python probe_framing.py --extract --n 3
    python probe_framing.py --compare        # needs ultralytics + the clips
"""

import os
import io
import re
import sys
import json
import glob
import zipfile
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = "/media/iizom/E/BaiduNetdiskDownload/高光剪辑训练集/qvhighlights-videos"
CLIPS = os.path.join(HERE, "probe_video")
YOLO_W = "/mnt/data/ML/模型实践与复现/YOLO/yolo11n.pt"


def load_rows():
    rows = []
    with io.open(os.path.join(TRAIN, "train.jsonl"), "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


def shards():
    return sorted(glob.glob(os.path.join(TRAIN, "qvhighlights-videos-*.zip")))


def vid_of(rec):
    c = rec.get("clip", {})
    return c.get("source_vid") or rec.get("video_id") or rec.get("id")


def clip_key(rec):
    """the mp4 basename is just source_vid + '.mp4'.

    Verified by listing a shard: entries look like `1/1-mmR6VTu7U_210.0_360.0.mp4`,
    i.e. the numbers are the 150 s QVHighlights window already baked into
    source_vid -- NOT start_sec/end_sec (a common wrong guess).
    """
    v = vid_of(rec)
    return (v, v + ".mp4") if v else (None, None)


def recon(rows):
    sh = shards()
    print("shards found: %d" % len(sh))
    for z in sh[:5]:
        print("   ", os.path.basename(z))
    have = [r for r in rows if r.get("cropRois")]
    vals = [r for r in have if r.get("dataset_split") == "val"]
    print("rows %d, with cropRois %d, val-with-rois %d" % (len(rows), len(have), len(vals)))
    # a row's own fields, to pin down the naming
    r = vals[0]
    print("\nsample row keys:", sorted(r.keys()))
    print("  clip:", json.dumps(r.get("clip", {}), ensure_ascii=False)[:300])
    print("  cropRois[:3]:", json.dumps(r["cropRois"][:3])[:300])
    print("  n cropRois:", len(r["cropRois"]))
    print("  free_axis:", r["clip"].get("free_axis"))
    # how many cropRois frames, and are they contiguous?
    fr = [int(x[0]) for x in r["cropRois"]]
    print("  frames:", fr[:8], "...", fr[-3:], " step~", (fr[1] - fr[0]) if len(fr) > 1 else None)
    # length distribution + naming cross-check
    print("\nval rows, cropRois frame counts:")
    for v in vals[:10]:
        vv, key = clip_key(v)
        print("   %-14s %-30s n=%4d axis=%s" % (vv, key, len(v["cropRois"]),
                                                 v["clip"].get("free_axis")))
    return vals


def extract(vals, n, stride):
    os.makedirs(CLIPS, exist_ok=True)
    sh = shards()
    picked = 0
    for v in vals[::max(1, len(vals) // (n * 4))]:
        if picked >= n:
            break
        vid, key = clip_key(v)
        if not vid:
            continue
        zpath = os.path.join(TRAIN, "qvhighlights-videos-%s.zip" % vid[0].lower())
        if not os.path.exists(zpath):
            print("  no shard for %s" % vid)
            continue
        out = os.path.join(CLIPS, key)
        if os.path.exists(out):
            print("  have %s" % key)
            picked += 1
            continue
        try:
            with zipfile.ZipFile(zpath) as z:
                names = [nm for nm in z.namelist() if nm.endswith(key)]
                if not names:
                    print("  %s not in %s" % (key, os.path.basename(zpath)))
                    continue
                with z.open(names[0]) as src, open(out, "wb") as dst:
                    dst.write(src.read())
            print("  extracted %-34s -> %.1f MB" % (key, os.path.getsize(out) / 1e6))
            picked += 1
        except Exception as ex:                                    # noqa: BLE001
            print("  FAIL %s: %s" % (key, ex))
    print("extracted %d" % picked)
    return picked


def compare(vals, stride, imgsz):
    """cropRois centre vs YOLO person-pool centre, canonical 300-high space."""
    try:
        from ultralytics import YOLO
    except Exception as ex:                                        # noqa: BLE001
        raise SystemExit("ultralytics unavailable: %s" % ex)
    if not os.path.exists(YOLO_W):
        raise SystemExit("no weights at %s" % YOLO_W)
    import cv2
    model = YOLO(YOLO_W)

    print("%-34s %5s %8s %8s %8s %8s" % ("clip", "nfr", "medC", "medY", "dC", "corr"))
    for v in vals:
        vid, key = clip_key(v)
        path = os.path.join(CLIPS, key) if key else None
        if not path or not os.path.exists(path):
            continue
        cap = cv2.VideoCapture(path)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rois = {int(x[0]): x[1] for x in v["cropRois"]}
        sc = 300.0 / H                                  # cropRois canvas -> src
        cs, ys = [], []
        f = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if f % stride == 0 and f in rois:
                x, y, w, h = rois[f]
                cx = (x + w / 2.0) * sc / W             # normalised, src width
                cs.append(cx)
                r = model.predict(img, imgsz=imgsz, verbose=False, classes=[0])[0]
                if r.boxes is not None and len(r.boxes):
                    b = r.boxes.xyxy.cpu().numpy()
                    a = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))
                    ccx = ((b[:, 0] + b[:, 2]) / 2.0) / W
                    ys.append(float((ccx * a).sum() / a.sum()))
                else:
                    ys.append(np.nan)
            f += 1
        cap.release()
        cs, ys = np.array(cs), np.array(ys)
        m = ~np.isnan(ys)
        if m.sum() < 3:
            print("%-34s %5d  no dets" % (key, len(cs)))
            continue
        r = float(np.corrcoef(cs[m], ys[m])[0, 1]) if cs[m].std() > 0 and ys[m].std() > 0 else float("nan")
        print("%-34s %5d %8.4f %8.4f %8.4f %8.3f"
              % (key, len(cs), float(np.median(cs)), float(np.median(ys[m])),
                 float(np.median(ys[m]) - np.median(cs)), r))
    print("\n  medC = cropRois centre (canonical->src, normalised by W)")
    print("  medY = YOLO person conf*area pool centre, same normalisation")
    print("  dC   = how far off the framing is from the YOLO pool (the bias)")
    print("  corr = does the framing MOVE with the YOLO pool? (>0.6 => yes)")


def series(vals, stride, imgsz):
    """per-frame pairs, so a median cannot hide an anti-correlation."""
    from ultralytics import YOLO
    import cv2
    model = YOLO(YOLO_W)
    for v in vals:
        vid, key = clip_key(v)
        path = os.path.join(CLIPS, key) if key else None
        if not path or not os.path.exists(path):
            continue
        cap = cv2.VideoCapture(path)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        rois = {int(x[0]): x[1] for x in v["cropRois"]}
        sc = 300.0 / H
        print("\n== %s  %dx%d  rois=%d frames %s" %
              (key, W, H, len(rois), (min(rois), max(rois))))
        print("   %5s %8s %8s %6s %8s %8s" %
              ("f", "cropCX", "poolCX", "ndet", "topCX", "topA/W2"))
        f = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if f % stride == 0 and f in rois:
                x, y, w, h = rois[f]
                cx = (x + w / 2.0) * sc / W
                r = model.predict(img, imgsz=imgsz, verbose=False, classes=[0])[0]
                b = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
                if len(b):
                    a = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
                    ccx = ((b[:, 0] + b[:, 2]) / 2.0) / W
                    j = int(np.argmax(a))
                    print("   %5d %8.4f %8.4f %6d %8.4f %8.3f"
                          % (f, cx, float((ccx * a).sum() / a.sum()), len(b),
                             float(ccx[j]), float(a[j]) / (W * H)))
                else:
                    print("   %5d %8.4f %8s %6d" % (f, cx, "-", 0))
            f += 1
        cap.release()


def viz(vals, stride, imgsz, nshow):
    """draw cropRois (green) vs YOLO persons (red) so the box convention is visible."""
    from ultralytics import YOLO
    import cv2
    model = YOLO(YOLO_W)
    for v in vals:
        vid, key = clip_key(v)
        path = os.path.join(CLIPS, key) if key else None
        if not path or not os.path.exists(path):
            continue
        cap = cv2.VideoCapture(path)
        rois = {int(x[0]): x[1] for x in v["cropRois"]}
        want = sorted(rois)[::max(1, len(rois) // nshow)][:nshow]
        f = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if f in want:
                x, y, w, h = rois[f]
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                r = model.predict(img, imgsz=imgsz, verbose=False, classes=[0])[0]
                if r.boxes is not None:
                    for bb in r.boxes.xyxy.cpu().numpy().astype(int):
                        cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]),
                                      (0, 0, 255), 1)
                out = os.path.join(CLIPS, "viz_%s_%04d.jpg" % (key[:-4], f))
                cv2.imwrite(out, img)
                print("  ", os.path.basename(out))
            f += 1
        cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", action="store_true")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--series", action="store_true")
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--nshow", type=int, default=3)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    rows = load_rows()
    vals = [r for r in rows if r.get("cropRois") and r.get("dataset_split") == "val"]
    if a.recon:
        recon(rows)
    if a.extract:
        extract(vals, a.n, a.stride)
    if a.compare:
        compare(vals, a.stride, a.imgsz)
    if a.series:
        series(vals, a.stride, a.imgsz)
    if a.viz:
        viz(vals, a.stride, a.imgsz, a.nshow)
    if not (a.recon or a.extract or a.compare or a.series or a.viz):
        recon(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
