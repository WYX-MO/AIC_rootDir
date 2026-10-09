#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""How much of the +0.03 online is REAL signal and how much is a broken label?

probe_framing.py showed one training clip whose cropRois sits on a letterboxed
news caption while the people are in the right-hand content window. If labels
like that are common, part of the oracle headroom (val 0.4214 -> 0.5315) is
unreachable by ANY estimator, and the +0.03 is partly a mirage.

Free: runs entirely on yolo_val_cache.json (person cx/cy/area already there) +
val_gt.jsonl. No download, no GPU.

Per video, over the GT frames inside the scoring window, ask of each GT crop:
does ANY person detection's centre fall inside the crop's horizontal span?
Then bucket the 57 videos and recompute the ceiling on the healthy ones.

Usage
    python label_probe.py
"""

import os
import sys
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import select_probe as SP                                          # noqa: E402


def analyse(d, drop_tail):
    """-> coverage, free-axis offsets, and the raw pieces for reporting."""
    a, b = SP.YE.window(d["n"], drop_tail)
    fs = np.array(d["frames"], dtype=np.int64)
    hits, offs, empty, ndet = 0, [], 0, []
    for f, gb in sorted(d["gcs"].items()):
        if not (a <= f < b):
            continue
        j = int(np.argmin(np.abs(fs - f)))
        ps = [x for x in d["dets"][j] if x["cls"] == "person"]
        lo, hi = gb[0] / d["W"], gb[2] / d["W"]
        gc = (lo + hi) / 2.0
        ndet.append(len(ps))
        if not ps:
            empty += 1
            continue
        inside = [p for p in ps if lo <= (p["cx"] if d["ax"] == "x" else p["cy"]) <= hi]
        if inside:
            hits += 1
            c = np.array([p["cx"] if d["ax"] == "x" else p["cy"] for p in inside],
                         dtype=np.float64)
            offs.append(gc - float(np.median(c)))
    tot = len(ndet)
    return {"cov": (hits / tot) if tot else 0.0,
            "n": tot, "hits": hits, "empty": empty,
            "off": float(np.median(offs)) if offs else None,
            "ndet": float(np.mean(ndet)) if ndet else 0.0,
            "w_gt": gb_width(d)}


def gb_width(d):
    k = next(iter(d["gcs"]))
    return (d["gcs"][k][2] - d["gcs"][k][0]) / d["W"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=os.path.join(HERE, "val_gt.jsonl"))
    ap.add_argument("--metadata", default=os.path.join(HERE, "val_metadata.json"))
    ap.add_argument("--cache", default=os.path.join(HERE, "yolo_val_cache.json"))
    ap.add_argument("--drop-tail", type=float, default=0.08)
    a = ap.parse_args()

    rows = SP.load(a)
    for d in rows:
        f0, f1 = SP.YE.window(d["n"], a.drop_tail)
        d["an"] = analyse(d, a.drop_tail)
        d["f1_pool"] = SP.f1(d, np.full(f1 - f0, _pool_med(d)), a.drop_tail)
        d["f1_orac"] = SP.f1(d, np.full(f1 - f0, float(np.median(
            list(d["gctr"].values())))), a.drop_tail)

    cov = np.array([d["an"]["cov"] for d in rows])
    grp = np.where(cov >= 0.8, "clean", np.where(cov >= 0.2, "partial", "broken"))
    print("videos %d   GT-crop covers a person in the window:" % len(rows))
    print("   %-8s %5s %10s %10s %10s" % ("bucket", "n", "mean cov", "pool", "oracle"))
    for g in ["clean", "partial", "broken"]:
        m = grp == g
        if m.sum() == 0:
            continue
        print("   %-8s %5d %10.3f %10.4f %10.4f"
              % (g, m.sum(), cov[m].mean(),
                 np.mean([d["f1_pool"] for d, k in zip(rows, m) if k]),
                 np.mean([d["f1_orac"] for d, k in zip(rows, m) if k])))
    print("   %-8s %5d            %10.4f %10.4f"
          % ("ALL", len(rows), np.mean([d["f1_pool"] for d in rows]),
             np.mean([d["f1_orac"] for d in rows])))

    print()
    print("   coverage histogram (fraction of GT frames whose crop holds a person):")
    for lo in np.arange(0.0, 1.01, 0.1):
        m = (cov >= lo) & (cov < lo + 0.1)
        if m.sum():
            print("     %.1f-%.1f  %s" % (lo, lo + 0.1, "#" * int(m.sum())))
    print()
    good = cov >= 0.2
    print("   drop the %d broken videos -> oracle ceiling %.4f (was %.4f),"
          % ((~good).sum(), np.mean([d["f1_orac"] for d, k in zip(rows, good) if k]),
             np.mean([d["f1_orac"] for d in rows])))
    print("      pool %.4f on the same subset -> real headroom %+.4f val -> %+.4f online"
          % (np.mean([d["f1_pool"] for d, k in zip(rows, good) if k]),
             np.mean([d["f1_orac"] for d, k in zip(rows, good) if k])
             - np.mean([d["f1_pool"] for d, k in zip(rows, good) if k]),
             (np.mean([d["f1_orac"] for d, k in zip(rows, good) if k])
              - np.mean([d["f1_pool"] for d, k in zip(rows, good) if k])) / 3.4))

    print()
    print("   lead-room offset (GT centre - nearest covered person centre), clean+partial:")
    offs = np.array([d["an"]["off"] for d, k in zip(rows, good) if k
                     and d["an"]["off"] is not None])
    if len(offs):
        print("     n %d  median %+.4f  mean %+.4f  mean|.| %.4f  p10 %+.4f p90 %+.4f"
              % (len(offs), float(np.median(offs)), float(offs.mean()),
                 float(np.abs(offs).mean()), float(np.percentile(offs, 10)),
                 float(np.percentile(offs, 90))))
        print("     share with |offset| < 0.05: %.0f%%   (those are the 'framing==person' ones)"
              % (100.0 * (np.abs(offs) < 0.05).mean()))

    print()
    print("   %-24s %6s %6s %6s %8s %8s" % ("worst offenders", "cov", "hits", "n", "pool", "oracle"))
    for d, g in sorted(zip(rows, grp), key=lambda t: t[0]["an"]["cov"])[:12]:
        print("   %-24s %6.2f %6d %6d %8.4f %8.4f"
              % (d["vid"][:24], d["an"]["cov"], d["an"]["hits"], d["an"]["n"],
                 d["f1_pool"], d["f1_orac"]))
    return 0


def _pool_med(d):
    v = [SP.pick([x for x in dd if x["cls"] == "person"], d["ax"], "pool", None)
         for dd in d["dets"]]
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else 0.5


if __name__ == "__main__":
    sys.exit(main())
