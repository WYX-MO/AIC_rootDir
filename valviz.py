#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Do the GT crops really hold nobody, or could YOLO just not see them?

label_probe.py bucketed videos by "does any cached YOLO person centre fall
inside the GT crop's x-range". That is only as good as the detector -- and the
clips are 534x300, where a far person is ~17x17 px, right at YOLO11n's floor.

A visual check has to use the RIGHT file: cropRois frames index the trimmed
`val_video/<id>.mp4` (e.g. 424 frames), NOT the 150 s `val_cache/<src>.mp4`
(4500 frames). Getting that wrong makes every drawn box meaningless.

So: open val_video/<id>.mp4, draw the GT crop (green) and fresh detections
(red), and re-measure coverage at two imgsz. If coverage jumps from 640 to
1280, the "no person" bucket was a detection miss, not an empty crop.

Usage
    python valviz.py                        # the 7 broken + 2 borderline
    python valviz.py --ids qvh_000106_9x16 --imgsz 640,1280 --viz
"""

import os
import sys
import json
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VALVID = os.path.join(HERE, "val_video")
YOLO_W = "/mnt/data/ML/模型实践与复现/YOLO/yolo11n.pt"
DEFAULT_IDS = ["qvh_000586_9x16", "qvh_000801_9x16", "qvh_000693_9x16",
               "qvh_000020_9x16", "qvh_000258_9x16", "qvh_000106_9x16",
               "qvh_000197_9x16", "qvh_000641_9x16", "qvh_000872_9x16"]


def load_gt():
    gt = {}
    with open(os.path.join(HERE, "val_gt.jsonl"), "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                r = json.loads(ln)
                gt[r["video_id"]] = r
    meta = json.load(open(os.path.join(HERE, "val_metadata.json"), "r", encoding="utf-8"))
    return gt, meta


def boxes_of(rec, meta):
    """val_gt prediction triplets -> {frame: (x1,y1,x2,y2)} in clip pixels."""
    m = meta[rec["video_id"]]
    W, H = m["W"], m["H"]
    tw, th = rec["targetRatioWH"]
    out = {}
    for e in rec["predictions"]:
        x, y, w = e["bboxes"][0], e["bboxes"][1], e["bboxes"][2]
        h = w * th / tw
        f = int(e["frame"])
        if f not in out:
            out[f] = (x, y, x + w, y + h)
    return out, W, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default=",".join(DEFAULT_IDS))
    ap.add_argument("--imgsz", default="640,1280")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--nshow", type=int, default=2)
    a = ap.parse_args()
    sizes = [int(s) for s in a.imgsz.split(",")]

    from ultralytics import YOLO
    import cv2
    model = YOLO(YOLO_W)
    gt, meta = load_gt()
    os.makedirs(os.path.join(HERE, "probe_video"), exist_ok=True)

    print("%-18s %5s %6s %8s %8s %8s   %s"
          % ("video", "frms", "W", "cov640", "cov1280", "cov_all", "note"))
    for vid in a.ids.split(","):
        vid = vid.strip()
        rec = gt.get(vid)
        if not rec:
            print("  %s: not in val_gt" % vid)
            continue
        gb, W, H = boxes_of(rec, meta)
        path = os.path.join(VALVID, vid + ".mp4")
        if not os.path.exists(path):
            print("  %s: no clip at %s" % (vid, path))
            continue
        cap = cv2.VideoCapture(path)
        cw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ch = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (cw, ch) != (W, H):
            print("  %s: clip %dx%d but metadata says %dx%d -- WRONG FILE?"
                  % (vid, cw, ch, W, H))
        want = sorted(gb)[::a.stride]
        pick = set(want)
        shots = set(want[:: max(1, len(want) // max(a.nshow, 1))][:a.nshow])
        hit = {s: 0 for s in sizes}
        tot, shown = 0, 0
        f = 0
        maxf = max(want) if want else -1
        while f <= maxf:
            ok, img = cap.read()
            if not ok:
                break
            if f in pick:
                x1, y1, x2, y2 = gb[f]
                tot += 1
                for s in sizes:
                    r = model.predict(img, imgsz=s, verbose=False, classes=[0])[0]
                    b = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
                    if len(b):
                        cx = (b[:, 0] + b[:, 2]) / 2.0
                        if ((cx >= x1) & (cx <= x2)).any():
                            hit[s] += 1
                    if a.viz and s == sizes[-1] and shown < a.nshow and f in shots:
                        r2 = model.predict(img, imgsz=s, verbose=False)[0]
                        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                                      (0, 255, 0), 2)
                        if r2.boxes is not None:
                            for bb, cl in zip(r2.boxes.xyxy.cpu().numpy(),
                                              r2.boxes.cls.cpu().numpy()):
                                col = (0, 0, 255) if int(cl) == 0 else (255, 128, 0)
                                cv2.rectangle(img, (int(bb[0]), int(bb[1])),
                                              (int(bb[2]), int(bb[3])), col, 1)
                        o = os.path.join(HERE, "probe_video",
                                         "vv_%s_%04d_s%d.jpg" % (vid, f, s))
                        cv2.imwrite(o, img)
                        print("     wrote", os.path.basename(o))
                        shown += 1
            f += 1
        cap.release()
        cov = {s: (hit[s] / tot if tot else 0.0) for s in sizes}
        note = ""
        if cov[sizes[0]] < 0.2 <= cov[sizes[-1]]:
            note = "<- WAS A DETECTION MISS"
        elif cov[sizes[-1]] < 0.2:
            note = "genuinely no person in the crop"
        print("%-18s %5d %6d %8.3f %8.3f %8s   %s"
              % (vid, tot, W, cov[sizes[0]], cov[sizes[-1]], "-", note))
    print("\n  cov = share of GT frames where a detected person's centre is inside")
    print("        the GT crop's horizontal span (imgsz as labelled)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
