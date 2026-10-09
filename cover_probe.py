#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A robust "is the GT crop framing a detectable thing?" measure.

The centre-in-window test in label_probe.py is knife-edge: the crop is 0.32
wide, so a person near its edge flips the count, and it moved from 0.24 to 0.02
for the same video between imgsz 640 and 1280. Area coverage does not flip.

For each GT frame, take the cached detections (all 13 classes are in
yolo_val_cache.json, 13889 boxes) and measure what fraction of the crop's area
is covered by boxes:

    frac_person  union of person boxes   n crop
    frac_any     union of any-class boxes n crop

No inference, no GPU -- runs on the cache. The eye check on the worst videos
found milk jugs, a newspaper front page and a POV road: real non-person shots,
so frac_any should be high where frac_person is ~0 if the framing follows
"salient content of the shot" rather than "a person".

Usage
    python cover_probe.py
"""

import os
import sys
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import select_probe as SP                                          # noqa: E402


def box_of(det, W, H):
    """cx/cy/area (normalised) + assumed square-ish -> xyxy in pixels."""
    a = max(det["area"], 0.0) ** 0.5
    return (max(det["cx"] - a / 2, 0.0) * W, max(det["cy"] - a / 2, 0.0) * H,
            min(det["cx"] + a / 2, 1.0) * W, min(det["cy"] + a / 2, 1.0) * H)


def cover(d, drop_tail):
    """mean crop-area covered by person / any detections, over GT frames."""
    a, b = SP.YE.window(d["n"], drop_tail)
    fs = np.array(d["frames"], dtype=np.int64)
    fp, fa, n = [], [], 0
    for f, gb in sorted(d["gcs"].items()):
        if not (a <= f < b):
            continue
        j = int(np.argmin(np.abs(fs - f)))
        W, H = d["W"], d["H"]
        cw, ch = max(gb[2] - gb[0], 1e-6), max(gb[3] - gb[1], 1e-6)
        # rasterise the crop once, then paint the boxes into it
        gx = np.linspace(gb[0], gb[2], 24)[None, :]
        gy = np.linspace(gb[1], gb[3], 24)[:, None]
        n += 1
        for want_person in (True, False):
            m = np.zeros((24, 24), dtype=bool)
            for x in d["dets"][j]:
                if want_person and x["cls"] != "person":
                    continue
                x1, y1, x2, y2 = box_of(x, W, H)
                m |= (gx >= x1) & (gx <= x2) & (gy >= y1) & (gy <= y2)
            (fp if want_person else fa).append(m.mean())
    return {"fp": float(np.mean(fp)) if fp else 0.0,
            "fa": float(np.mean(fa)) if fa else 0.0, "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=os.path.join(HERE, "val_gt.jsonl"))
    ap.add_argument("--metadata", default=os.path.join(HERE, "val_metadata.json"))
    ap.add_argument("--cache", default=os.path.join(HERE, "yolo_val_cache.json"))
    ap.add_argument("--drop-tail", type=float, default=0.08)
    a = ap.parse_args()
    rows = SP.load(a)
    for d in rows:
        d["c"] = cover(d, a.drop_tail)
        f0, f1 = SP.YE.window(d["n"], a.drop_tail)
        pm = [SP.pick([x for x in dd if x["cls"] == "person"], d["ax"], "pool", None)
              for dd in d["dets"]]
        pm = [x for x in pm if x is not None]
        d["pool"] = SP.f1(d, np.full(f1 - f0, float(np.median(pm)) if pm else 0.5),
                          a.drop_tail)
        d["orac"] = SP.f1(d, np.full(f1 - f0, float(np.median(
            list(d["gctr"].values())))), a.drop_tail)

    fa = np.array([d["c"]["fa"] for d in rows])
    fp = np.array([d["c"]["fp"] for d in rows])
    print("videos %d   mean crop-area covered by cached detections" % len(rows))
    print("   frac_any    mean %.3f  median %.3f" % (fa.mean(), np.median(fa)))
    print("   frac_person mean %.3f  median %.3f" % (fp.mean(), np.median(fp)))
    print()
    print("   %-10s %5s %10s %10s %8s %8s" % ("frac_any", "n", "frac_any", "frac_pers", "pool", "oracle"))
    for lo in [0.0, 0.2, 0.4, 0.6, 0.8]:
        m = (fa >= lo) & (fa < lo + 0.2)
        if m.sum():
            print("   %.1f-%.1f    %5d %10.3f %10.3f %8.4f %8.4f"
                  % (lo, lo + 0.2, m.sum(), fa[m].mean(), fp[m].mean(),
                     np.mean([d["pool"] for d, k in zip(rows, m) if k]),
                     np.mean([d["orac"] for d, k in zip(rows, m) if k])))
    print()
    for cut in [0.3, 0.5]:
        g = fa >= cut
        po = np.mean([d["pool"] for d, k in zip(rows, g) if k])
        orr = np.mean([d["orac"] for d, k in zip(rows, g) if k])
        print("   frac_any >= %.1f : %d/%d videos, pool %.4f oracle %.4f"
              " -> headroom %+.4f val %+.4f online"
              % (cut, g.sum(), len(rows), po, orr, orr - po, (orr - po) / 3.4))
    print()
    print("   videos whose crop is LEAST covered by anything (framing != detectable object):")
    for d in sorted(rows, key=lambda x: x["c"]["fa"])[:8]:
        print("     %-18s frac_any %.3f frac_person %.3f  pool %.4f oracle %.4f"
              % (d["vid"], d["c"]["fa"], d["c"]["fp"], d["pool"], d["orac"]))
    print()
    print("   videos whose crop holds a DETECTABLE thing but no person (object framing):")
    for d in [x for x in rows if x["c"]["fa"] > 0.5 and x["c"]["fp"] < 0.15]:
        print("     %-18s frac_any %.3f frac_person %.3f  pool %.4f oracle %.4f"
              % (d["vid"], d["c"]["fa"], d["c"]["fp"], d["pool"], d["orac"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
