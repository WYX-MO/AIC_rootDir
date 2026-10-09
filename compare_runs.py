#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GT-free A/B comparison of two submissions.
无真值的两次提交对比。

The test set ships no ground truth, so on the 174 real videos we cannot compute
F1 at all -- the leaderboard is the only judge. This script answers the next
best question: *how different are two runs?* Use it to sanity-check a change
before spending a submission, and to spot a run that silently degenerated.

Reports per-video and aggregate:
  * frame-set Jaccard       |A and B| / |A or B|     1.0 == identical frame sets
  * frame count delta       N_pred(B) - N_pred(A)
  * box centre offset along the free axis, in source pixels (mean abs / p90)
  * videos present in only one file, videos with empty predictions
  * a verdict: label-noise level (<5% frames differ) vs a real behavioural change

A change that moves Jaccard below ~0.9 is a genuine algorithmic change and
deserves a leaderboard probe; a change above ~0.98 is not worth submitting.

Dependencies: none beyond the stdlib.
"""

import os
import sys
import json
import argparse
import statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_pred(path):
    """submission jsonl -> {video_id: {"ratio":(tw,th), "preds":{frame:(x,y,w)}}}"""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                print("  [warn] %s:%d malformed, skipped" % (path, lineno))
                continue
            vid = str(o.get("video_id"))
            tr = o.get("targetRatioWH", [16, 9])
            try:
                ratio = (float(tr[0]), float(tr[1]))
            except (TypeError, ValueError, IndexError):
                ratio = (16.0, 9.0)
            preds = {}
            for p in o.get("predictions", []) or []:
                if not isinstance(p, dict) or "frame" not in p:
                    continue
                bb = p.get("bboxes") or []
                if len(bb) != 3:
                    continue
                try:
                    f = int(p["frame"])
                    preds[f] = (int(bb[0]), int(bb[1]), int(bb[2]))
                except (TypeError, ValueError):
                    continue
            out[vid] = {"ratio": ratio, "preds": preds}
    return out


def free_axis_of(tw, th):
    return "x" if th > tw else "y"


def main():
    ap = argparse.ArgumentParser(description="GT-free A/B comparison of two runs")
    ap.add_argument("--a", required=True, help="baseline submission jsonl")
    ap.add_argument("--b", required=True, help="candidate submission jsonl")
    ap.add_argument("--per-video", action="store_true")
    ap.add_argument("--max-show", type=int, default=20)
    ap.add_argument("--out", default=None, help="write the per-video table as json")
    args = ap.parse_args()

    A = load_pred(args.a)
    B = load_pred(args.b)

    only_a = sorted(set(A) - set(B))
    only_b = sorted(set(B) - set(A))
    common = sorted(set(A) & set(B))

    rows = []
    for vid in common:
        ta, tb = A[vid]["ratio"], B[vid]["ratio"]
        fa, fb = A[vid]["preds"], B[vid]["preds"]
        sa, sb = set(fa), set(fb)
        union = len(sa | sb)
        jac = (len(sa & sb) / union) if union else 1.0
        ax = free_axis_of(*ta)
        offs = []
        for f in sa & sb:
            if ax == "x":
                offs.append(abs(fa[f][0] - fb[f][0]))
            else:
                offs.append(abs(fa[f][1] - fb[f][1]))
        rows.append({
            "video_id": vid, "n_a": len(fa), "n_b": len(fb),
            "jaccard": jac, "shared": len(sa & sb),
            "mean_abs_off": statistics.mean(offs) if offs else 0.0,
            "ratio_mismatch": (ta != tb),
            # crop width, so a box-only change can be reported on a real scale
            # (a shift of 60px means nothing without knowing the crop is 169 wide)
            "w": (fa[next(iter(fa))][2] if fa else 1),
        })

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    jacs = [r["jaccard"] for r in rows]
    offs = [r["mean_abs_off"] for r in rows]
    dna = [r["n_a"] for r in rows]
    dnb = [r["n_b"] for r in rows]

    print("=" * 74)
    print("COMPARE   A=%s" % args.a)
    print("          B=%s" % args.b)
    print("=" * 74)
    print("videos: A=%d  B=%d  common=%d  only-A=%d  only-B=%d"
          % (len(A), len(B), len(common), len(only_a), len(only_b)))
    if only_a:
        print("  only in A: %s" % ", ".join(only_a[:args.max_show]))
    if only_b:
        print("  only in B: %s" % ", ".join(only_b[:args.max_show]))
    empty_a = [v for v in A if not A[v]["preds"]]
    empty_b = [v for v in B if not B[v]["preds"]]
    print("empty predictions: A=%d  B=%d" % (len(empty_a), len(empty_b)))
    print()
    print("--- frame budget ---")
    print("  N_pred total   : A=%d  B=%d   (%+d, %+.1f%%)"
          % (sum(dna), sum(dnb), sum(dnb) - sum(dna),
             100.0 * (sum(dnb) - sum(dna)) / max(1, sum(dna))))
    print("  N_pred per video: A mean %.1f median %.1f | B mean %.1f median %.1f"
          % (mean(dna), statistics.median(dna) if dna else 0,
             mean(dnb), statistics.median(dnb) if dnb else 0))
    print()
    print("--- agreement ---")
    if jacs:
        print("  frame-set Jaccard: mean %.4f  median %.4f  min %.4f"
              % (mean(jacs), statistics.median(jacs), min(jacs)))
        print("  near-identical (>=0.98): %.1f%% of videos" % (100 * mean([j >= 0.98 for j in jacs])))
        print("  substantially changed  (<0.90): %.1f%% of videos" % (100 * mean([j < 0.90 for j in jacs])))
        print("  box centre |offset| px: mean %.2f  median %.2f  p90 %.2f"
              % (mean(offs), statistics.median(offs),
                 sorted(offs)[int(0.9 * len(offs))] if offs else 0))
    mism = [r["video_id"] for r in rows if r["ratio_mismatch"]]
    if mism:
        print("  WARNING ratio mismatch on %d videos: %s" % (len(mism), ", ".join(mism[:10])))
    print()

    if jacs:
        j = mean(jacs)
        moff = statistics.median(offs) if offs else 0.0
        ws = [r["w"] for r in rows if r["w"]]
        rel = moff / statistics.median(ws) if ws else 0.0
        if j < 0.90:
            verdict = ("REAL CHANGE -- %d%% of frames moved; this is a genuine "
                       "behavioural difference" % int(round(100 * (1 - j))))
        elif j < 0.98:
            verdict = ("MINOR CHANGE -- %d%% of frames moved; worth one probe "
                       "only if the change was meant to be small"
                       % int(round(100 * (1 - j))))
        elif rel >= 0.05:
            # frame sets identical but the boxes moved: the F1 metric sees this
            # (it is IoU), this script's Jaccard does NOT. Saying "label noise"
            # here would be actively wrong.
            verdict = ("BOX-ONLY CHANGE -- frame sets identical (Jaccard %.3f) "
                       "but boxes moved a median of %.0f px = %.0f%% of the crop "
                       "width. Frame Jaccard is blind to this; IoU is not."
                       % (j, moff, 100 * rel))
        else:
            verdict = ("LABEL NOISE -- frame sets identical and boxes moved "
                       "<5%% of the crop width (median %.0f px)" % moff)
        print("VERDICT: %s" % verdict)

    if args.per_video:
        print("\n%-24s %7s %7s %9s %10s" % ("video_id", "N_A", "N_B", "jaccard", "off_px"))
        for r in sorted(rows, key=lambda r: r["jaccard"]):
            print("%-24s %7d %7d %9.4f %10.2f"
                  % (r["video_id"], r["n_a"], r["n_b"], r["jaccard"], r["mean_abs_off"]))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"summary": {
                "n_common": len(common), "n_only_a": len(only_a), "n_only_b": len(only_b),
                "jaccard_mean": mean(jacs), "jaccard_median": statistics.median(jacs) if jacs else 0,
                "npred_a": sum(dna), "npred_b": sum(dnb),
                "mean_abs_offset_px": mean(offs)},
                "per_video": rows}, f, ensure_ascii=False, indent=1)
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
