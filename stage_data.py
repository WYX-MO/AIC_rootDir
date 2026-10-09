#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data staging / metadata probing for the AIC highlight re-framing task.
数据落地与元数据探测。

Two jobs:
  1) probe   -- read every *.mp4 in a directory and dump width/height/fps/frame
                count/duration to metadata.json.  The evaluator needs the same
                numbers to validate `frame` bounds and `bboxes` bounds, so this
                is the single source of truth for "what is a legal prediction".
  2) val     -- build a local validation set (videos + gt jsonl) out of the
                QVHighlights weak-label training annotations, so we can compute
                the official metric locally instead of flying blind.

Why ffprobe and not cv2 CAP_PROP_FRAME_COUNT: the latter is unreliable on
VFR/H.264 and silently returns 0 for some files.  ffprobe reports nb_frames and
duration authoritatively; we fall back to cv2 only when ffprobe is missing.

probe 读取目录下所有 mp4 的宽/高/帧率/总帧数/时长写入 metadata.json。评测与校验
都依赖同一组数字（frame 上界、bbox 边界），因此这里必须是唯一权威来源。
val 从 QVHighlights 弱标注训练集构造本地验证集（视频 + gt jsonl），使本地可算官方指标。
"""

import os
import sys
import json
import glob
import argparse
import subprocess


# --------------------------------------------------------------------------
# probing
# --------------------------------------------------------------------------
def _fps_from_ratio(ratio):
    """'30000/1001' -> 29.97 ; '25/1' -> 25.0 ; None on failure."""
    if not ratio:
        return None
    try:
        if "/" in str(ratio):
            num, den = str(ratio).split("/")
            den = float(den)
            return float(num) / den if den else None
        return float(ratio)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def probe_ffprobe(path):
    """Authoritative probe via ffprobe. Returns dict or None."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-show_entries", "format=duration",
        "-of", "json", path,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=60)
        if out.returncode != 0:
            return None
        data = json.loads(out.stdout.decode("utf-8", "replace") or "{}")
    except Exception:
        return None

    streams = data.get("streams") or []
    if not streams:
        return None
    st = streams[0]
    W = int(st.get("width") or 0)
    H = int(st.get("height") or 0)
    fps = _fps_from_ratio(st.get("avg_frame_rate")) or _fps_from_ratio(st.get("r_frame_rate"))
    n = st.get("nb_frames")
    n = int(n) if n not in (None, "", "N/A") else None
    dur = st.get("duration")
    if dur in (None, "", "N/A"):
        dur = (data.get("format") or {}).get("duration")
    try:
        dur = float(dur) if dur not in (None, "", "N/A") else None
    except (TypeError, ValueError):
        dur = None

    if n is None and dur is not None and fps:
        n = int(round(dur * fps))
    if dur is None and n is not None and fps:
        dur = n / fps
    if not (W and H):
        return None
    return {"W": W, "H": H, "fps": fps, "n_frames": n, "duration": dur, "probe": "ffprobe"}


def probe_cv2(path):
    """Fallback probe. Less trustworthy (VFR / truncated index)."""
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return None
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    cap.release()
    if not (W and H):
        return None
    dur = (n / fps) if (n and fps) else None
    return {"W": W, "H": H, "fps": fps, "n_frames": n, "duration": dur, "probe": "cv2"}


def probe_one(path):
    return probe_ffprobe(path) or probe_cv2(path)


# --------------------------------------------------------------------------
# geometry helpers shared with the baselines / evaluator
# --------------------------------------------------------------------------
def target_aspect(tw, th):
    return float(tw) / float(th) if th else 0.0


def crop_size(W, H, tw, th):
    """Largest target-ratio rectangle that fits inside W x H.

    Returns (cw, ch). For a 9:16 target on a landscape source this is
    (round(H*9/16), H) -- full height, i.e. only the free axis moves.

    ROUNDING: `round`, NOT `floor`
    ------------------------------
    The submission carries only `w`; the judge derives `h = w * th / tw`, which
    is generally fractional. Rounding to nearest therefore lets the derived
    height overshoot H by up to 0.5 * th/tw px -- 0.889px at 9:16. That is
    bounded and is NOT a bug, because the ground truth does exactly the same:
    all 57 local val videos carry w = round(H*9/16) = 169 for a 300-tall source,
    i.e. derived h = 300.44 > 300. The upstream baseline_qwen.compute_crop_size
    rounds the same way. The metric compares our boxes against those GT boxes,
    so matching their integer width is worth ~0.6% mean IoU on EVERY matched
    pair: measured IoU 0.6526 -> 0.6568, val F1 0.3561 -> 0.3584.

    An earlier revision floored here to guarantee the box stayed inside the
    frame. That was wrong -- it optimised for a validator constraint the data
    does not impose and paid for it on every pair.

    Free axis 'y' (16:9 target on a portrait source) never overshoots: there
    w = W exactly, so the derived h = W*9/16 is pinned by the source width.
    提交只带 w，评测按 h = w*th/tw 补算高度。就近取整最多让 h 溢出 H
    0.5*th/tw 像素（9:16 时 0.889px），而 GT 自己就是这样（val 全部 57 条
    w = round(H*9/16) = 169、h = 300.44 > 300），官方 baseline_qwen 亦然。
    指标是拿我们的框跟这些 GT 框算 IoU，对齐它们的整数宽度值约 0.6% 平均 IoU；
    早先"向下取整以保证不越界"的版本反而白丢了这部分。
    """
    if tw <= 0 or th <= 0 or W <= 0 or H <= 0:
        return W, H
    ta = target_aspect(tw, th)
    if W / float(H) >= ta:          # source wider than target -> full height
        cw = min(int(round(H * ta)), W)
        ch = H
    else:                           # source taller than target -> full width
        ch = min(int(round(W / ta)), H)
        cw = W
    return max(1, cw), max(1, ch)


def free_axis(W, H, tw, th):
    """Which axis the crop window can travel along.
    'x' when the crop spans the full height (landscape source / portrait
    target), 'y' when it spans the full width (portrait source / landscape
    target). Matches the training annotations, which are all free_axis='x'."""
    cw, ch = crop_size(W, H, tw, th)
    if ch == H and cw < W:
        return "x"
    if cw == W and ch < H:
        return "y"
    return "none"


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def cmd_probe(args):
    files = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")))
    if not files:
        print("no *.mp4 under %s" % args.video_dir)
        return 1
    meta = {}
    failed = []
    for i, p in enumerate(files, 1):
        vid = os.path.splitext(os.path.basename(p))[0]
        m = probe_one(p)
        if m is None:
            failed.append(vid)
            continue
        meta[vid] = m
        if i % 25 == 0 or i == len(files):
            print("  probed %d/%d" % (i, len(files)))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1, sort_keys=True)

    print("\nwrote %s (%d videos, %d failed)" % (args.out, len(meta), len(failed)))
    if failed:
        print("FAILED: %s" % ", ".join(failed[:20]))
    return 0


# --------------------------------------------------------------------------
# local validation set from the QVHighlights weak-label annotations
# --------------------------------------------------------------------------
# The training annotations are weak labels produced by a teacher VLM on
# QVHighlights clips. Useful properties we rely on here:
#   * cropRois are `[[frame, [x, y, w, h]], ...]` already in SOURCE PIXELS
#     (crop size == crop_size(W,H,tw,th); e.g. 534x300 source, 9:16 target ->
#     169x300, exactly what the annotations carry).
#   * `frame` is relative to the annotated WINDOW, not to the 150s source file
#     (verified: span == clip.end_sec - clip.start_sec for every record).
#   * `quality.selected_ratio` == len(cropRois) / n_frames, i.e. the tau we need.
#   * `spatial_source == 'dropped_center_default'` marks records where the
#     spatial pass produced nothing -- those have no cropRois and are unusable
#     as GT (they are an annotation gap, NOT a "no highlight" label).
#
# So we cut the annotated window out of the source file and emit GT whose frame
# indices line up with the cut clip. That makes each val sample look like a real
# (short) test video.

def build_zip_manifest(zip_dir, pattern="qvhighlights-videos-*.zip"):
    """basename -> (zip_path, inner_path). One pass over central directories."""
    manifest = {}
    zips = sorted(glob.glob(os.path.join(zip_dir, pattern)))
    if not zips:
        return manifest
    for i, z in enumerate(zips, 1):
        try:
            out = subprocess.run(["unzip", "-Z1", z], capture_output=True, timeout=600)
            if out.returncode != 0:
                continue
            for name in out.stdout.decode("utf-8", "replace").splitlines():
                name = name.strip()
                if name.lower().endswith(".mp4"):
                    manifest[os.path.basename(name)] = (z, name)
        except Exception as e:
            print("  [warn] cannot list %s: %s" % (z, e))
        if i % 8 == 0:
            print("  indexed %d/%d zips (%d files)" % (i, len(zips), len(manifest)))
    return manifest


def cmd_val(args):
    here = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(args.annotations):
        print("annotations not found: %s" % args.annotations)
        return 1
    recs = []
    with open(args.annotations, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))

    want_split = set(args.splits.split(","))
    sel = [r for r in recs
           if r.get("dataset_split") in want_split and (r.get("cropRois") or [])]
    print("records: %d total, %d selected (splits=%s, have cropRois)"
          % (len(recs), len(sel), args.splits))
    if args.limit:
        sel = sel[:args.limit]

    print("indexing video zips under %s ..." % args.zip_dir)
    manifest = build_zip_manifest(args.zip_dir)
    print("  manifest: %d mp4 files across zips" % len(manifest))

    os.makedirs(args.video_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    gt_lines = []
    skipped = []
    for i, r in enumerate(sel, 1):
        vid = str(r["video_id"])
        src = r["clip"]["source_vid"]
        fname = src if src.lower().endswith(".mp4") else src + ".mp4"
        out_video = os.path.join(args.video_dir, vid + ".mp4")

        if not (os.path.exists(out_video) and not args.force):
            hit = manifest.get(fname)
            if hit is None:
                skipped.append((vid, fname, "not found in zips"))
                continue
            zip_path, inner = hit
            raw = os.path.join(args.cache_dir, fname)
            try:
                if not os.path.exists(raw):
                    subprocess.run(["unzip", "-j", "-o", zip_path, inner, "-d", args.cache_dir],
                                   capture_output=True, timeout=600, check=True)
                start = float(r["clip"]["start_sec"])
                end = float(r["clip"]["end_sec"])
                dur = max(0.05, end - start)
                # input seeking (-ss before -i) lands on the exact frame; re-encode
                # so output frame 0 == window start, matching the GT frame indices.
                cmd = ["ffmpeg", "-y", "-v", "error",
                       "-ss", "%.3f" % start, "-i", raw, "-t", "%.3f" % dur,
                       "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                       "-c:a", "aac", "-b:a", "96k", out_video]
                subprocess.run(cmd, capture_output=True, timeout=600, check=True)
            except subprocess.CalledProcessError as e:
                skipped.append((vid, fname, "ffmpeg/unzip failed: %s" % (e.stderr or b"")[:120]))
                continue
            except Exception as e:
                skipped.append((vid, fname, "error: %s" % e))
                continue
            finally:
                if args.keep_raw is False and os.path.exists(raw):
                    pass  # keep the raw cache; re-runs are much cheaper

        m = probe_one(out_video)
        if m is None or not m.get("n_frames"):
            skipped.append((vid, fname, "probe failed on cut clip"))
            continue
        n = m["n_frames"]

        preds = []
        for frame, box in r["cropRois"]:
            fi = int(frame)
            if fi < 0 or fi >= n:
                continue
            x, y, w, h = (int(round(float(v))) for v in box)
            preds.append({"frame": fi, "bboxes": [x, y, w]})
        if not preds:
            skipped.append((vid, fname, "all GT frames outside cut clip"))
            continue
        seen = set()
        uniq = []
        for p in sorted(preds, key=lambda p: p["frame"]):
            if p["frame"] in seen:
                continue
            seen.add(p["frame"])
            uniq.append(p)
        gt_lines.append({"video_id": vid, "targetRatioWH": list(r["targetRatioWH"]),
                         "predictions": uniq})
        if i % 10 == 0 or i == len(sel):
            print("  %d/%d" % (i, len(sel)))

    with open(args.out, "w", encoding="utf-8") as f:
        for rec in gt_lines:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("\nwrote %s  (%d videos)" % (args.out, len(gt_lines)))

    # val_index.json mirrors test_index.json so eval_local.py validate can be run
    # against the val set (same interface as the real submission).
    index = [{"video_id": rec["video_id"], "targetRatioWH": rec["targetRatioWH"]}
             for rec in gt_lines]
    with open(args.index, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)
    print("wrote %s  (%d videos)" % (args.index, len(index)))
    if skipped:
        print("skipped %d:" % len(skipped))
        for s in skipped[:15]:
            print("   %s (%s): %s" % s)

    # probe the cut clips for the evaluator
    meta = {}
    for rec in gt_lines:
        p = os.path.join(args.video_dir, rec["video_id"] + ".mp4")
        m = probe_one(p)
        if m:
            meta[rec["video_id"]] = m
    with open(args.metadata, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1, sort_keys=True)
    print("wrote %s  (%d videos)" % (args.metadata, len(meta)))

    # report the tau prior we end up with
    taus = []
    for rec in gt_lines:
        m = meta.get(rec["video_id"])
        if m and m.get("n_frames"):
            taus.append(len(rec["predictions"]) / m["n_frames"])
    if taus:
        taus.sort()
        print("val tau = N_gt/n_frames: min=%.3f p25=%.3f median=%.3f p75=%.3f max=%.3f"
              % (taus[0], taus[len(taus)//4], taus[len(taus)//2], taus[3*len(taus)//4], taus[-1]))
    return 0


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="stage data / probe video metadata")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="probe all *.mp4 in a dir -> metadata.json")
    p.add_argument("--video-dir", default=os.path.join(here, "video"))
    p.add_argument("--out", default=os.path.join(here, "metadata.json"))
    p.set_defaults(func=cmd_probe)

    v = sub.add_parser("val", help="build local val set from QVHighlights annotations")
    v.add_argument("--annotations", default="/media/iizom/E/BaiduNetdiskDownload/"
                                             "高光剪辑训练集/qvhighlights-videos/train.jsonl")
    v.add_argument("--zip-dir", default="/media/iizom/E/BaiduNetdiskDownload/"
                                        "高光剪辑训练集/qvhighlights-videos")
    v.add_argument("--splits", default="val", help="comma list of dataset_split values")
    v.add_argument("--limit", type=int, default=0)
    v.add_argument("--video-dir", default=os.path.join(here, "val_video"))
    v.add_argument("--cache-dir", default=os.path.join(here, "val_cache"))
    v.add_argument("--out", default=os.path.join(here, "val_gt.jsonl"))
    v.add_argument("--index", default=os.path.join(here, "val_index.json"))
    v.add_argument("--metadata", default=os.path.join(here, "val_metadata.json"))
    v.add_argument("--force", action="store_true", help="re-cut even if output exists")
    v.add_argument("--keep-raw", dest="keep_raw", action="store_true", default=True)
    v.set_defaults(func=cmd_val)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
