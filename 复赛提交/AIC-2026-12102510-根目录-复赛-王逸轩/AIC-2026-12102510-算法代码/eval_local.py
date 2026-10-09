#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local evaluator / submission validator for the AIC highlight re-framing task.
本地评测器 / 提交校验器。

Two modes / 两种模式:

  validate --index test_index.json --pred preds.jsonl --video-dir video
      Strict format check with NO ground truth required. This is the mode you
      run before every leaderboard submission: it enforces exactly the field
      constraints the official judge documents, so a submission that passes
      here cannot be rejected for being unparseable.

  score --gt gt.jsonl --pred preds.jsonl --video-dir val_video
      Computes the official metric against ground truth.

Official metric / 官方指标
--------------------------
Per video, with S = sum of IoU over valid same-frame matched pairs,
N_pred = number of valid predictions, N_gt = number of GT boxes:

    F1 = 2 * S / (N_pred + N_gt)

then averaged over videos. This is the standard generalized (continuous)
IoU-weighted F-score: FP = N_pred - n_matched, FN = N_gt - n_matched, so
F1 = 2n_matched / (2n_matched + FP + FN) = 2S / (N_pred + N_gt).

Matching is per (video_id, frame). A prediction only earns IoU if it lands on
a frame that carries a GT box -- frame agreement is a hard binary constraint,
which is why "keep how many frames" matters as much as "where is the box".

Invalid entries (frame out of range, bbox out of frame, NaN, missing fields,
duplicate frame, unmatched video_id, malformed JSONL) earn nothing. We take the
FIRST valid prediction per (video, frame); later duplicates are false
positives. The baselines clamp before writing so this never triggers.

Note on 0/0: a video whose GT is empty and which we left empty has an undefined
F1. We report all three conventions (skip / count as 0 / count as 1) because the
official judge's choice is not documented -- see --zero-gt.

Dependencies: none beyond the stdlib (metadata via stage_data.probe_one).
"""

import os
import sys
import json
import math
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 模型权重解压后总大小上限。官方写的是 "9 GB"; 本工具包全程按 1 MB = 1024*1024
# 换算, 所以取 9 GiB = 9216 MB。若赛方按 SI 的 9000 MB 卡, 这里的判定会偏松
# ~2%, 但我们也只在明显越界时报 FATAL, 不会替你放行一个 9.x GB 的提交。
MODEL_SIZE_LIMIT_MB = 9216


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------
def load_index(path):
    """test_index.json -> [(video_id, (tw, th)), ...]"""
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    out = []
    for it in items:
        out.append((str(it["video_id"]), _ratio(it.get("targetRatioWH", [16, 9]))))
    return out


def load_pred_jsonl(path):
    """submission jsonl -> ({video_id: (targetRatioWH, [pred, ...])}, errors)

    Kept tolerant on read so that `validate` can report *what* is wrong rather
    than dying on the first malformed line.
    """
    recs = {}
    errors = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(("json", lineno, None, "line %d: invalid JSON (%s)" % (lineno, e.msg)))
                continue
            if not isinstance(obj, dict):
                errors.append(("json", lineno, None, "line %d: not a JSON object" % lineno))
                continue
            vid = obj.get("video_id")
            if vid is None:
                errors.append(("json", lineno, None, "line %d: missing video_id" % lineno))
                continue
            vid = str(vid)
            if vid in recs:
                errors.append(("json", lineno, vid, "line %d: duplicate video_id %s" % (lineno, vid)))
                continue
            preds = obj.get("predictions", [])
            if not isinstance(preds, list):
                errors.append(("json", lineno, vid, "line %d: predictions is not a list" % lineno))
                preds = []
            recs[vid] = (_ratio(obj.get("targetRatioWH")), preds, lineno,
                         obj.get("model_size_mb"))
    return recs, errors


def load_meta(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ratio(tr, default=(16.0, 9.0)):
    try:
        tw, th = float(tr[0]), float(tr[1])
        if tw > 0 and th > 0:
            return (tw, th)
    except (TypeError, ValueError, IndexError):
        pass
    return default


def _finite_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def box_from_triplet(x, y, w, tw, th):
    """[x, y, w] -> (x, y, x+w, y+h), h derived as w * th / tw."""
    h = w * float(th) / float(tw)
    return (x, y, x + w, y + h)


def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------------
# per-prediction validation
# --------------------------------------------------------------------------
def check_prediction(entry, vid, ratio, meta_entry, index_in_list, tol=1.0):
    """Return (ok, frame, box_xyxy, code, message).

    meta_entry: {"W","H","n_frames","fps"} or None when unknown.

    tol: pixel slack on the derived-height bound check. The submission carries
    only `w` and the judge derives `h = w*th/tw`, which is generally fractional.
    The official GT itself ships such boxes (a 300-tall source with 9:16 target
    carries w=169 -> derived h=300.44 > 300), so the judge must tolerate the
    sub-pixel overshoot; a hard float-exact check would reject the ground truth.
    Entries that violate the float-exact rule but fall inside `tol` are counted
    separately (see `n_soft`) so real overflows stay visible.
    """
    def bad(code, msg):
        return False, None, None, code, msg, False

    if not isinstance(entry, dict):
        return bad("shape", "prediction is not an object")
    if "frame" not in entry:
        return bad("missing", "missing 'frame'")
    f = entry["frame"]
    if isinstance(f, bool) or not isinstance(f, (int, float)) or not math.isfinite(float(f)):
        return bad("frame", "'frame' is not a finite number: %r" % (f,))
    if float(f) != int(f):
        return bad("frame", "'frame' is not an integer: %r" % (f,))
    f = int(f)
    if f < 0:
        return bad("frame", "'frame' is negative: %d" % f)

    bb = entry.get("bboxes")
    if bb is None:
        return bad("missing", "missing 'bboxes'")
    if not isinstance(bb, (list, tuple)):
        return bad("bbox", "'bboxes' is not a list: %r" % (bb,))
    if len(bb) != 3:
        return bad("bbox", "'bboxes' must be a triplet [x,y,w], got %d values" % len(bb))
    if not all(_finite_num(v) for v in bb):
        return bad("bbox", "'bboxes' contains a non-finite value: %r" % (bb,))
    x, y, w = (float(bb[0]), float(bb[1]), float(bb[2]))
    if x < 0 or y < 0:
        return bad("bbox", "'x'/'y' must be >= 0, got [%.3f, %.3f]" % (x, y))
    if w <= 0:
        return bad("bbox", "'w' must be > 0, got %.3f" % w)

    tw, th = ratio
    bx1, by1, bx2, by2 = box_from_triplet(x, y, w, tw, th)

    soft = False
    if meta_entry:
        W, H = meta_entry.get("W"), meta_entry.get("H")
        n = meta_entry.get("n_frames")
        if n is not None and f >= n:
            return False, None, None, "frame", "'frame' %d >= video frame count %d" % (f, n), False
        if W and bx2 > W + tol:
            return False, None, None, "bounds", "box right edge %.3f exceeds width %d (x=%.3f w=%.3f)" % (bx2, W, x, w), False
        if H and by2 > H + tol:
            return False, None, None, "bounds", "box bottom edge %.3f exceeds height %d (y=%.3f derived h=%.3f)" % (by2, H, y, by2 - by1), False
        if (W and bx2 > W) or (H and by2 > H):
            soft = True     # within tol: judge must be rounding the derived h

    return True, f, (bx1, by1, bx2, by2), None, None, soft


# --------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------
def cmd_validate(args):
    index = load_index(args.index)
    index_ids = [v for v, _ in index]
    meta = load_meta(args.metadata) if os.path.exists(args.metadata) else {}

    if not os.path.exists(args.pred):
        print("ERROR: prediction file not found: %s" % args.pred)
        return 2
    recs, json_errors = load_pred_jsonl(args.pred)

    violations = list(json_errors)
    per_video_ok = {}

    # ---- model_size_mb: 初赛可选, 复赛/半决赛必填且各行一致 ----
    seen_sizes = {}
    n_with_size = 0
    for vid, (_r, _p, lineno, size) in recs.items():
        if size is None:
            continue
        if not _finite_num(size) or float(size) <= 0:
            violations.append(("size", lineno, vid,
                               "model_size_mb is not a positive finite number: %r" % (size,)))
            continue
        n_with_size += 1
        seen_sizes.setdefault(float(size), []).append(vid)

    if recs and 0 < n_with_size < len(recs):
        violations.append(("size", 0, None,
                           "model_size_mb present on only %d/%d lines; 复赛要求每行都填"
                           % (n_with_size, len(recs))))
    if seen_sizes:
        vals = sorted(seen_sizes)
        if len(vals) > 1:
            violations.append(("size", 0, None,
                               "model_size_mb must be identical on every line, found %d distinct "
                               "values: %s" % (len(vals), ", ".join("%g" % v for v in vals[:6]))))
        if vals[-1] > MODEL_SIZE_LIMIT_MB:
            violations.append(("size", 0, None,
                               "model_size_mb %g exceeds the 9 GB limit (%d MB)"
                               % (vals[-1], MODEL_SIZE_LIMIT_MB)))
    soft_bounds = {}
    total_preds = 0

    missing_videos = [v for v in index_ids if v not in recs]
    extra_videos = [v for v in recs if v not in set(index_ids)]

    for vid, (tw, th) in index:
        if vid not in recs:
            continue
        ratio, preds, lineno, _size = recs[vid]
        if (tw, th) != ratio and args.strict_ratio:
            violations.append(("ratio", lineno, vid,
                               "targetRatioWH mismatch: file has %s, index has [%g,%g]"
                               % (list(ratio), tw, th)))
        m = meta.get(vid)
        if m is None:
            violations.append(("meta", lineno, vid,
                               "no video metadata (video-dir has it? run stage_data.py probe)"))
        seen = {}
        n_ok = 0
        for i, entry in enumerate(preds):
            total_preds += 1
            ok, f, box, code, msg, soft = check_prediction(entry, vid, (tw, th), m, i, args.bounds_tol)
            if not ok:
                violations.append((code, lineno, vid, "prediction[%d]: %s" % (i, msg)))
                continue
            if soft:
                soft_bounds[vid] = soft_bounds.get(vid, 0) + 1
            if f in seen:
                violations.append(("dup", lineno, vid,
                                   "prediction[%d]: duplicate frame %d (first at prediction[%d])"
                                   % (i, f, seen[f])))
                continue
            seen[f] = i
            n_ok += 1
        if args.strict_order and preds:
            frames = [p.get("frame") for p in preds if isinstance(p, dict)]
            fr = [x for x in frames if isinstance(x, (int, float)) and not isinstance(x, bool)]
            if fr != sorted(fr):
                violations.append(("order", lineno, vid, "predictions are not sorted by frame (recommended)"))
        per_video_ok[vid] = (n_ok, len(preds))

    # ---- report ----
    by_code = {}
    for code, lineno, vid, msg in violations:
        by_code.setdefault(code, []).append((lineno, vid, msg))

    print("=" * 72)
    print("VALIDATE  %s" % args.pred)
    print("=" * 72)
    print("index videos            : %d" % len(index_ids))
    print("prediction lines        : %d" % len(recs))
    print("total prediction entries : %d" % total_preds)
    print("valid entries           : %d" % sum(v[0] for v in per_video_ok.values()))
    print("videos missing from file : %d" % len(missing_videos))
    print("unknown video_id in file : %d" % len(extra_videos))
    if seen_sizes:
        print("model_size_mb           : %s  (%d/%d lines)"
              % (", ".join("%g" % v for v in sorted(seen_sizes)), n_with_size, len(recs)))
    else:
        print("model_size_mb           : absent (初赛可选, 复赛/半决赛必填)")
    print("violations              : %d" % len(violations))

    if missing_videos:
        print("\n[videos missing] %s" % ", ".join(missing_videos[:20]) +
              (" ..." if len(missing_videos) > 20 else ""))
    if extra_videos:
        print("\n[unknown video_id] %s" % ", ".join(extra_videos[:20]))
    if violations:
        fatal = [c for c in ("json", "frame", "bbox", "bounds", "dup", "missing", "shape", "ratio", "meta", "size")
                 if c in by_code]
        print("\n[violations by kind] %s"
              % ", ".join("%s=%d" % (c, len(by_code[c])) for c in sorted(by_code, key=lambda k: -len(by_code[k]))))
        for code in fatal:
            if code not in by_code:
                continue
            print("\n--- %s (%d) ---" % (code, len(by_code[code])))
            for lineno, vid, msg in by_code[code][:args.max_show]:
                print("  line %-6s vid %-8s %s" % (lineno, vid, msg))
            if len(by_code[code]) > args.max_show:
                print("  ... and %d more" % (len(by_code[code]) - args.max_show))

    # non-fatal advisory
    empty = [v for v, (n_ok, n) in per_video_ok.items() if n == 0]
    if empty:
        print("\n[videos with empty predictions] %d (allowed: means 'no highlight')" % len(empty))

    if soft_bounds:
        tot_soft = sum(soft_bounds.values())
        print("\n[advisory: sub-pixel bound overshoot] %d entries in %d videos" % (tot_soft, len(soft_bounds)))
        print("  These exceed W/H by < %.1f px only because the derived h = w*th/tw is" % args.bounds_tol)
        print("  fractional (the official GT itself does this). Accepted here; tolerated by the")
        print("  judge if it rounds the derived h. Videos: %s"
              % ", ".join(sorted(soft_bounds)[:15]) + (" ..." if len(soft_bounds) > 15 else ""))

    ok = not violations and not missing_videos and not extra_videos
    print("\nRESULT: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


# --------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------
def load_gt(path):
    """gt jsonl -> {video_id: (ratio, [pred, ...])}. Same schema as a submission."""
    recs, errors = load_pred_jsonl(path)
    out = {}
    for vid, (ratio, preds, _, _) in recs.items():
        out[vid] = (ratio, preds)
    if errors:
        print("WARNING: %d malformed GT line(s) skipped" % len(errors))
    return out


def cmd_score(args):
    gt = load_gt(args.gt)
    pred, _ = load_pred_jsonl(args.pred)
    meta = load_meta(args.metadata) if os.path.exists(args.metadata) else {}

    rows = []
    for vid, (ratio, gt_preds) in gt.items():
        tw, th = ratio
        m = meta.get(vid)

        gt_frames = {}
        n_gt_bad = 0
        for e in gt_preds:
            ok, f, box, code, _, _ = check_prediction(e, vid, ratio, m, 0, args.bounds_tol)
            if not ok:
                n_gt_bad += 1
                continue
            gt_frames[f] = box

        pr_frames = {}
        n_pred_valid = 0
        n_pred_invalid = 0
        if vid in pred:
            pr_ratio, pr_preds, _, _ = pred[vid]
            for e in pr_preds:
                ok, f, box, code, _, _ = check_prediction(e, vid, ratio, m, 0, args.bounds_tol)
                if not ok:
                    n_pred_invalid += 1
                    continue
                n_pred_valid += 1
                if f not in pr_frames:      # first valid wins; rest are FP
                    pr_frames[f] = box

        S = 0.0
        n_matched = 0
        for f, pb in pr_frames.items():
            gb = gt_frames.get(f)
            if gb is None:
                continue
            S += iou_xyxy(pb, gb)
            n_matched += 1

        N_pred = len(pr_frames)
        N_gt = len(gt_frames)
        f1 = (2.0 * S / (N_pred + N_gt)) if (N_pred + N_gt) > 0 else None
        rows.append({
            "video_id": vid, "N_pred": N_pred, "N_gt": N_gt, "S": S,
            "matched": n_matched, "F1": f1,
            "mean_iou_matched": (S / n_matched) if n_matched else 0.0,
            "pred_invalid": n_pred_invalid, "gt_invalid": n_gt_bad,
        })

    # ---- aggregate ----
    defined = [r for r in rows if r["F1"] is not None]
    undef = [r for r in rows if r["F1"] is None]

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    f1_skip = mean([r["F1"] for r in defined])
    f1_zero = mean([r["F1"] if r["F1"] is not None else 0.0 for r in rows])
    f1_one = mean([r["F1"] if r["F1"] is not None else 1.0 for r in rows])

    print("=" * 72)
    print("SCORE  pred=%s" % args.pred)
    print("        gt  =%s" % args.gt)
    print("=" * 72)
    print("videos scored           : %d" % len(rows))
    print("  with defined F1 (N_pred+N_gt>0) : %d" % len(defined))
    print("  degenerate 0/0                  : %d" % len(undef))
    print()
    print("--- OFFICIAL METRIC: mean over videos of 2*S/(N_pred+N_gt) ---")
    print("  F1 (skip 0/0 videos)    : %.4f" % f1_skip)
    print("  F1 (0/0 counted as 0)   : %.4f" % f1_zero)
    print("  F1 (0/0 counted as 1)   : %.4f" % f1_one)
    print()
    if defined:
        f1s = sorted(r["F1"] for r in defined)
        print("  distribution: min=%.4f p25=%.4f median=%.4f p75=%.4f max=%.4f"
              % (f1s[0], f1s[len(f1s)//4], f1s[len(f1s)//2], f1s[3*len(f1s)//4], f1s[-1]))
    print()
    tot_gt = sum(r["N_gt"] for r in rows)
    tot_pr = sum(r["N_pred"] for r in rows)
    tot_match = sum(r["matched"] for r in rows)
    tot_s = sum(r["S"] for r in rows)
    print("--- diagnostics ---")
    print("  total GT boxes          : %d" % tot_gt)
    print("  total valid predictions : %d" % tot_pr)
    print("  matched pairs           : %d" % tot_match)
    print("  keep ratio k=N_pred/N_gt: %.3f   (optimal ~1.0)" % (tot_pr / tot_gt if tot_gt else 0))
    print("  frame recall  matched/N_gt      : %.4f" % (tot_match / tot_gt if tot_gt else 0))
    print("  frame precision matched/N_pred  : %.4f" % (tot_match / tot_pr if tot_pr else 0))
    print("  mean IoU on matched pairs       : %.4f" % (tot_s / tot_match if tot_match else 0))
    print("  invalid pred entries            : %d" % sum(r["pred_invalid"] for r in rows))
    if tot_pr and tot_gt:
        a = tot_s / tot_match if tot_match else 0.0
        print("  => implied ceiling if k were 1  : %.4f  (2a/(1+k) with k=1)" % (2 * a / (1 + tot_pr / tot_gt) * (1 + tot_pr / tot_gt) / 2))

    if args.per_video:
        print("\n--- per video ---")
        print("%-24s %7s %7s %8s %6s %8s" % ("video_id", "N_pred", "N_gt", "matched", "S", "F1"))
        for r in sorted(rows, key=lambda r: (r["F1"] is None, r["F1"] if r["F1"] is not None else 0)):
            print("%-24s %7d %7d %8d %6.2f %8s"
                  % (r["video_id"], r["N_pred"], r["N_gt"], r["matched"], r["S"],
                     ("%.4f" % r["F1"]) if r["F1"] is not None else "undef"))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"summary": {"f1_skip_zerogt": f1_skip, "f1_zero": f1_zero, "f1_one": f1_one,
                                   "n_videos": len(rows), "n_defined": len(defined)},
                       "per_video": rows}, f, ensure_ascii=False, indent=1)
        print("\nwrote %s" % args.out)
    return 0


# --------------------------------------------------------------------------
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="local validator + scorer")
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="strict format check, no GT needed")
    v.add_argument("--index", default=os.path.join(here, "test_index.json"))
    v.add_argument("--pred", required=True)
    v.add_argument("--video-dir", default=os.path.join(here, "video"))
    v.add_argument("--metadata", default=os.path.join(here, "metadata.json"))
    v.add_argument("--strict-ratio", action="store_true", default=True)
    v.add_argument("--no-strict-ratio", dest="strict_ratio", action="store_false")
    v.add_argument("--strict-order", action="store_true", default=False,
                   help="also require predictions sorted by frame (advisory per spec)")
    v.add_argument("--bounds-tol", type=float, default=1.0,
                   help="pixel slack on the derived-height bound check (default 1.0)")
    v.add_argument("--max-show", type=int, default=25)
    v.set_defaults(func=cmd_validate)

    s = sub.add_parser("score", help="score against GT")
    s.add_argument("--gt", required=True)
    s.add_argument("--pred", required=True)
    s.add_argument("--video-dir", default=os.path.join(here, "val_video"))
    s.add_argument("--metadata", default=os.path.join(here, "val_metadata.json"))
    s.add_argument("--bounds-tol", type=float, default=1.0,
                   help="pixel slack on the derived-height bound check (default 1.0)")
    s.add_argument("--per-video", action="store_true")
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_score)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
