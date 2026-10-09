#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen2.5-VL baseline: semantic highlight localisation + subject reframe.
Qwen2.5-VL 基线：语义级高光定位 + 主体重构。

WHY THIS EXISTS
---------------
The CV baseline (baseline_cv.py) measured F1 = 0.3561 on the local val set, and
that was with tau=1.0 -- i.e. it predicted EVERY frame with a centred box and
made no temporal decision at all. Every low-level signal we tried was at chance
for localisation (per-signal AUC 0.480-0.586), because "which moment is the
highlight" is a SEMANTIC judgement (the training teacher's summaries read like
"commercial rocket ignites and lifts off, then the shot cuts to a ground
tracking antenna"). Meanwhile the ceiling with perfect localisation but the same
centred box is 0.6268. That gap -- 0.357 -> 0.627 -- is what a video LLM is for.

TWO STAGES
----------
  1) TEMPORAL. N evenly spaced frames are shown to the model with their
     timestamps, and it returns the highlight interval as JSON
     {"start_sec", "end_sec"} plus a short subject phrase in the same reply
     (one call does both). Seconds are converted to frames via the probed
     frame count, never via fps*duration, because those disagree on VFR files.
  2) SPATIAL. K frames inside that interval are shown and the model returns a
     normalised focus point per frame; the point is mixed toward the frame
     centre by --alpha, clamped, median+EMA smoothed, then interpolated to
     every kept frame. The crop size is fixed by the ratio rule, so only one
     axis can move (stage_data.free_axis decides which).

MEASURED ON VAL (57 videos) -- READ THIS BEFORE TRUSTING THE MODEL
------------------------------------------------------------------
  VLM, as first written (alpha=1)                F1 = 0.3229
  same intervals, every box forced to centre     F1 = 0.3561
  plain tau=1 centred hedge (no model at all)    F1 = 0.3561

Two facts, both measured:

  * STAGE 1 WAS A NO-OP. With the prompt escape hatch ("if the whole clip is
    good, return the full range") the model answered "the whole clip is the
    highlight" for 57/57 videos, so the kept interval was the video and the
    temporal stage carried zero information. The escape hatch is now removed.
  * STAGE 2 WAS ACTIVELY HARMFUL. It moved 11,017 of 19,459 boxes off centre
    and dropped mean IoU on matched pairs from 0.6526 to 0.5796, because GT
    crop boxes are essentially centred (median offset -2px). Hence --alpha now
    defaults to 0 and --no-stage2 exists.

So the honest state of this route is: the plumbing works end to end (real
inference, 6.7 s/video, 57/57 videos emitted, 0 violations), but the 3B model
does not yet deliver the 0.357 -> 0.627 that semantic localisation is worth.
Stage 1 is the only part that could: a correct interval with the SAME centred
box is worth 0.6268 (perfect-interval ceiling). Stage 2 cannot pay for itself
while the ground truth is centred.

FAIL-SAFE, DELIBERATELY
-----------------------
Any of these -- model unavailable, unparseable JSON, empty/degenerate interval,
zero kept frames -- falls back to tau=1.0 with a centred box, which is the
val-optimal CV strategy (F1 0.3561). An empty list would score 0 for the video;
the hedge scores ~0.356. The script counts how many videos it had to emit as
empty and warns, because that is the only unrecoverable failure mode.
That matters because the local val set cannot tell us whether an aggressive
interval is right -- see RUNBOOK.md "The tau bet".

WIDENING
--------
The metric F1 = 2S/(N_pred+N_gt) punishes missing GT frames far more than it
punishes extra predictions (a superset scores 2a/(1+k), a subset 2ak/(1+k), and
recall is the dominant term at high k). So a VLM interval that is too tight is
costly. --widen expands the interval symmetrically about its centre, and
--min-keep-ratio / --max-keep-ratio clamp the result. widen=1.0 trusts the model
exactly; widen>1 hedges. Both are worth a val sweep before a leaderboard probe.

Dependencies: torch, transformers>=4.49, qwen_vl_utils, accelerate,
bitsandbytes (only for --load-4bit), opencv, numpy. No decord needed: frames are
sampled with OpenCV and handed to the processor as images, which sidesteps the
qwen_vl_utils video-reader path entirely.
"""

import os
import sys
import json
import math
import time
import argparse

os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "av")

# huggingface.co is unreachable on this machine; the weights live in a local
# cache at /mnt/data/hf. Without these two, from_pretrained() blocks for a very
# long time on network timeouts and the GPU stays idle (observed: 394 MiB used,
# no output, several minutes). Set both before importing transformers.
os.environ.setdefault("HF_HOME", "/mnt/data/hf")
if os.path.isdir(os.path.join(os.environ["HF_HOME"], "hub")):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 8 GB of VRAM and a 3B model: the default caching allocator fragments badly
# across 174 videos (load, generate, free, repeat) and fragments into an OOM.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage_data import crop_size, free_axis, probe_one       # noqa: E402
from baseline_cv import median_filter, ema, offset_from_center  # noqa: E402

try:
    import cv2
except ImportError:
    sys.stderr.write("opencv-python-headless is required\n")
    raise


# --------------------------------------------------------------------------
# frame sampling
# --------------------------------------------------------------------------
def grab_frames(path, idxs, max_side=512):
    """Read the given absolute frame indices. Returns [(idx, PIL.Image), ...].

    Sequential single pass with a set of wanted indices, so cost is one decode
    of the file regardless of how many frames are wanted. Missing indices are
    simply absent from the result -- callers must tolerate a short list.
    """
    from PIL import Image
    want = sorted(set(int(i) for i in idxs))
    onepass = {i: None for i in want}
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    cur = 0
    remaining = len(want)
    wi = 0
    while remaining > 0:
        ok, fr = cap.read()
        if not ok:
            break
        if wi < len(want) and cur == want[wi]:
            h, w = fr.shape[:2]
            s = min(1.0, float(max_side) / max(h, w))
            if s < 1.0:
                fr = cv2.resize(fr, (max(1, int(round(w * s))),
                                     max(1, int(round(h * s)))),
                                interpolation=cv2.INTER_AREA)
            onepass[cur] = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            wi += 1
            remaining -= 1
            while wi < len(want) and want[wi] <= cur:
                wi += 1
        cur += 1
    cap.release()
    return [(i, onepass[i]) for i in want if onepass[i] is not None]


def even_indices(n_frames, k):
    if k <= 1:
        return [0]
    return sorted(set(int(round(i * (n_frames - 1) / float(k - 1)))
                      for i in range(k)))


# --------------------------------------------------------------------------
# model wrapper
# --------------------------------------------------------------------------
class QwenVL(object):
    def __init__(self, model_id, device="cuda", load_4bit=False,
                 max_pixels=512 * 28 * 28, verbose=False):
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as Cls
        except ImportError:                       # older naming
            from transformers import Qwen2VLForConditionalGeneration as Cls

        self.torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.verbose = verbose
        self.processor = AutoProcessor.from_pretrained(
            model_id, max_pixels=max_pixels, min_pixels=256 * 28 * 28)
        kw = {}
        if load_4bit and self.device == "cuda":
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        else:
            kw["torch_dtype"] = torch.bfloat16
        self.model = Cls.from_pretrained(model_id, device_map=self.device, **kw)
        self.model.eval()
        if self.verbose:
            print("  [vlm] %s on %s%s" % (model_id, self.device,
                                          " (4bit)" if load_4bit else " (bf16)"))

    def ask(self, images, prompt, max_new_tokens=160):
        """images: list of PIL.Image. Returns the decoded text."""
        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, return_tensors="pt")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                      do_sample=False)
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]


# --------------------------------------------------------------------------
# output parsing -- tolerant, never raises
# --------------------------------------------------------------------------
def extract_json(text):
    """Pull the first JSON object/array out of a reply. None if there is none.

    The opener that appears EARLIEST wins, not the object one. Stage 2 asks for
    an array of points; if we tried "{" first we would slice the first inner
    element out of the array and silently return a single point instead of the
    list. Both candidates are tried in document order.
    """
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1] if len(t.split("```")) > 1 else t
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
    cands = []
    for opener, closer in (("{", "}"), ("[", "]")):
        a = t.find(opener)
        b = t.rfind(closer)
        if a != -1 and b > a:
            cands.append((a, t[a:b + 1]))
    cands.sort()                                  # earliest opener first
    for _a, cand in cands:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            try:                                  # trailing commas
                return json.loads(cand.replace(",]", "]").replace(",}", "}"))
            except json.JSONDecodeError:
                pass
    return None


def _to_point(e, scale):
    """One stage-2 element -> (x, y) in 0..1, or None.

    Qwen2.5-VL answers grounding questions in whichever of its several
    conventions it prefers, so accept all of them: [x,y], a dict with x/y (or
    cx/cy), a 4-number bbox (either [x0,y0,x1,y1] or its {"bbox_2d": ...} form)
    -> the bbox centre. Coordinates may be 0..1, 0..100 or 0..1000; `scale` is
    derived once for the whole array by the caller.
    """
    def num(v):
        try:
            return float(v) / scale
        except (TypeError, ValueError):
            return None

    if isinstance(e, dict):
        if "bbox_2d" in e or "bbox" in e:
            e = e.get("bbox_2d", e.get("bbox"))
        else:
            x = _first_num(e, ("x", "cx", "center_x"))
            y = _first_num(e, ("y", "cy", "center_y"))
            if x is None or y is None:
                return None
            return (num(x), num(y))
    if isinstance(e, (list, tuple)):
        vals = [num(v) for v in e]
        if any(v is None for v in vals):
            return None
        if len(vals) == 4:                      # bbox -> centre
            vals = [0.5 * (vals[0] + vals[2]), 0.5 * (vals[1] + vals[3])]
        elif len(vals) == 3:                    # [idx, x, y]
            vals = vals[1:]
        if len(vals) != 2:
            return None
        return (vals[0], vals[1])
    return None


def _coord_scale(flat):
    """Guess whether the numbers are 0..1, 0..100 or 0..1000."""
    mx = max([abs(v) for v in flat if v is not None] or [0.0])
    if mx <= 1.5:
        return 1.0
    if mx <= 150.0:
        return 100.0
    return 1000.0


def _first_num(d, keys):
    for k in keys:
        if isinstance(d, dict) and k in d:
            try:
                return float(d[k])
            except (TypeError, ValueError):
                pass
    return None


PROMPT_TEMPORAL = (
    "下面是从一个视频中等间隔抽取的 {k} 帧图片，按时间先后排列。"
    "第 1 帧对应 {t0:.2f} 秒，最后一帧对应 {t1:.2f} 秒，整段视频长 {dur:.2f} 秒。\n"
    "请找出这段视频里最精彩、最值得保留的**一段连续画面**，"
    "并指出该片段的画面主体是什么。\n"
    "这一段必须明显短于整段视频（通常占全片 20%~60%）。"
    "不要返回整段视频；即使全片都不错，也要指出其中最突出的那一段。\n"
    "只输出一个 JSON 对象，不要任何其他文字，格式：\n"
    '{{"start_sec": <片段开始的秒数>, "end_sec": <片段结束的秒数>, '
    '"subject": "<画面主体，10字以内>"}}\n'
    "只输出 JSON，不要解释。"
)

PROMPT_SPATIAL = (
    "下面按顺序给出同一个视频精彩片段的 {k} 张图片（第 1 张到第 {k} 张）。"
    "请为**每一张**图片指出最值得保留的主体在画面中的中心点，"
    "用归一化坐标 [x, y] 表示（左上角 [0,0]，右下角 [1,1]，x 为水平方向）。\n"
    "只输出一个 JSON 数组，长度必须正好是 {k}，每个元素是一张图片的 [x, y]。"
    "例如 {k}=3 时输出：[[0.5,0.5],[0.4,0.6],[0.45,0.55]]\n"
    "不要输出 bbox 或任何其他文字。"
)


# --------------------------------------------------------------------------
# per-video driver
# --------------------------------------------------------------------------
def vlm_interval(vlm, path, n_frames, dur, w_idx, args, log):
    """Return (f0, f1, subject) or None when the model gave nothing usable."""
    if not w_idx:
        return None
    imgs = [im for _, im in w_idx]
    times = [i / float(max(1, n_frames - 1)) * dur for i, _ in w_idx]
    prompt = PROMPT_TEMPORAL.format(k=len(imgs), t0=times[0], t1=times[-1],
                                    dur=dur)
    txt = vlm.ask(imgs, prompt, max_new_tokens=args.max_new_tokens)
    log("stage1 raw: %s" % txt.strip()[:200])
    obj = extract_json(txt)
    if obj is None:
        log("stage1: no JSON")
        return None
    if isinstance(obj, list) and obj:
        obj = obj[0]
    if not isinstance(obj, dict):
        log("stage1: not an object")
        return None
    a = _first_num(obj, ("start_sec", "start", "begin_sec", "from_sec"))
    b = _first_num(obj, ("end_sec", "end", "stop_sec", "to_sec"))
    subj = obj.get("subject") or obj.get("subjects") or obj.get("主体") or ""
    if a is None or b is None:
        log("stage1: missing start/end")
        return None
    if b < a:
        a, b = b, a
    if b - a < 1e-3:
        log("stage1: degenerate interval %.3f..%.3f" % (a, b))
        return None

    # widen about the centre, then clamp to the video
    c = 0.5 * (a + b)
    half = 0.5 * (b - a) * args.widen
    a, b = max(0.0, c - half), min(dur, c + half)
    if b - a < 1e-3:
        return None

    # seconds -> frames using the PROBED count, not fps*duration (VFR)
    f0 = int(math.floor(a / dur * n_frames))
    f1 = int(math.ceil(b / dur * n_frames))
    f0 = max(0, min(n_frames - 1, f0))
    f1 = max(f0 + 1, min(n_frames, f1))
    log("stage1: %.2f..%.2f s -> frames [%d, %d)  subject=%r"
        % (a, b, f0, f1, subj))
    return f0, f1, str(subj)


def vlm_centers(vlm, path, f0, f1, n_frames, k, args, log):
    """Normalised free-axis centres per kept frame. None when unusable."""
    idxs = even_indices(f1 - f0, k)
    abs_idx = [f0 + i for i in idxs]
    got = grab_frames(path, abs_idx, args.max_side)
    if len(got) < max(2, k // 2):
        log("stage2: only %d/%d frames read" % (len(got), len(abs_idx)))
        return None
    imgs = [im for _, im in got]
    prompt = PROMPT_SPATIAL.format(k=len(imgs))
    txt = vlm.ask(imgs, prompt, max_new_tokens=args.max_new_tokens)
    log("stage2 raw: %s" % txt.strip()[:200])
    arr = extract_json(txt)
    if isinstance(arr, dict):                    # {"points": [...]} and friends
        for k in ("points", "centers", "centres", "result", "results", "data"):
            if isinstance(arr.get(k), list):
                arr = arr[k]
                break
        else:
            # A bare point/bbox for the whole strip: one point, broadcast below.
            if any(k in arr for k in ("bbox_2d", "bbox", "x", "cx", "center_x")):
                arr = [arr]
    if isinstance(arr, list) and len(arr) == 1 and isinstance(arr[0], list) \
            and arr[0] and isinstance(arr[0][0], (list, tuple, dict)):
        arr = arr[0]                             # model wrapped the array once
    if not isinstance(arr, list) or not arr or len(arr) > 4 * len(imgs):
        log("stage2: unusable array (len=%s want=%d)"
            % (len(arr) if isinstance(arr, list) else None, len(imgs)))
        return None

    scale = _coord_scale([v for e in arr
                          for v in (e if isinstance(e, (list, tuple)) else [])])
    pts = []
    for e in arr:
        p = _to_point(e, scale)
        if p is None:
            log("stage2: unparseable element %r" % (e,))
            return None
        pts.append((min(1.0, max(0.0, p[0])), min(1.0, max(0.0, p[1]))))

    # The model does not always honour "exactly k entries": it may answer once
    # for the whole strip (1 point) or emit a few. Resample whatever we got onto
    # the k sampled frames rather than throwing the answer away.
    if len(pts) != len(idxs):
        log("stage2: %d points for %d images -> resampled"
            % (len(pts), len(imgs)))
    xs = np.array([p[0] for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64)
    if len(pts) == 1:
        xs = np.repeat(xs, len(idxs))
        ys = np.repeat(ys, len(idxs))
    else:
        srcp = np.linspace(0.0, 1.0, len(pts))
        dstp = np.linspace(0.0, 1.0, len(idxs))
        xs = np.interp(dstp, srcp, xs)
        ys = np.interp(dstp, srcp, ys)

    # scatter the k sampled points back onto their frames, interp the rest
    n = f1 - f0
    src = np.array([i for i in idxs], dtype=np.float64)
    grid = np.arange(n, dtype=np.float64)
    fx = np.interp(grid, src, xs)
    fy = np.interp(grid, src, ys)
    return fx, fy


def hedge_preds(W, H, n_frames, tw, th):
    """The val-optimal no-model strategy: EVERY frame, centred box.

    This is the safety net. It must never be replaced by an empty list: an empty
    submission scores F1 = 0 for that video (2*0/(0+N_gt)), whereas the hedge
    scores ~0.356 on average. Losing a video to a crash is the single most
    expensive thing this script can do, so every failure path lands here.
    """
    cw, ch = crop_size(W, H, tw, th)
    ax = free_axis(W, H, tw, th)
    x = int((W - cw) // 2)
    y = int((H - ch) // 2)
    if ax == "x":
        x, y = x, 0
    elif ax == "y":
        x, y = 0, y
    return [{"frame": int(f), "bboxes": [int(x), int(y), int(cw)]}
            for f in range(n_frames)]


def process_video(vid, tw, th, args, vlm, log):
    path = os.path.join(args.video_dir, vid + ".mp4")
    if not os.path.exists(path):
        return None, "no video file"
    m = probe_one(path)
    if not m or not m.get("n_frames"):
        return None, "probe failed"
    W, H = m["W"], m["H"]
    n_frames = int(m["n_frames"])
    fps = float(m["fps"] or 25.0)
    dur = float(m["duration"] or (n_frames / fps if fps else 0.0))
    if dur <= 0:
        return None, "unknown duration"

    if vlm is None:
        return hedge_preds(W, H, n_frames, tw, th), None
    try:
        return _process_with_vlm(vid, path, W, H, n_frames, dur, tw, th,
                                 args, vlm, log)
    except Exception as e:
        # OOM, a processor quirk, a codec hiccup -- degrade to the hedge, never
        # to an empty list (empty == F1 0 for this video).
        try:
            import torch
            if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        log("VLM stage failed (%s: %s) -> hedge" % (type(e).__name__, e))
        return hedge_preds(W, H, n_frames, tw, th), "vlm failed: %s" % type(e).__name__


def _process_with_vlm(vid, path, W, H, n_frames, dur, tw, th, args, vlm, log):
    """The real path. Exceptions here are caught by the caller, which keeps the
    hedge, so nothing in this function needs to be defensive about failure."""
    cw, ch = crop_size(W, H, tw, th)
    ax = free_axis(W, H, tw, th)
    cx_c, cy_c = (W - cw) // 2, (H - ch) // 2

    # default plan == the val-optimal hedge: every frame, centred box
    f0, f1 = 0, n_frames
    centers = np.full(n_frames, 0.5, dtype=np.float64)
    used = "hedge(tau=1,centre)"
    subject = ""

    k1 = min(args.stage1_frames, n_frames)
    w_idx = grab_frames(path, even_indices(n_frames, k1), args.max_side)
    iv = vlm_interval(vlm, path, n_frames, dur, w_idx, args, log)
    if iv is not None:
        f0, f1, subject = iv
        # clamp the kept ratio
        lo = int(args.min_keep_ratio * n_frames)
        hi = int(args.max_keep_ratio * n_frames)
        cur = f1 - f0
        if lo > 0 and cur < lo:
            grow = lo - cur
            f0 = max(0, f0 - grow // 2)
            f1 = min(n_frames, f1 + grow - grow // 2)
        elif hi > 0 and cur > hi:
            drop = cur - hi
            f0 = f0 + drop // 2
            f1 = f1 - (drop - drop // 2)
        f1 = max(f0 + 1, min(n_frames, f1))
        f0 = max(0, min(n_frames - 1, f0))
        used = "vlm(%.2f of video)" % ((f1 - f0) / float(n_frames))
        setc = None if args.no_stage2 else vlm_centers(
            vlm, path, f0, f1, n_frames,
            min(args.stage2_frames, max(2, f1 - f0)), args, log)
        if setc is not None:
            axis_track = setc[0] if ax == "x" else setc[1]
            full = np.full(n_frames, 0.5, dtype=np.float64)
            full[f0:f1] = axis_track
            centers = full
            used += "+vlm_centre"
        else:
            used += "+centre"

    offs, _travel = offset_from_center(centers, W, H, cw, ch, ax, args.alpha)
    offs = ema(median_filter(offs, args.smooth_win), args.ema)

    # emit only the kept window; re-clamp after smoothing (interpolate-then-clamp)
    preds = []
    for f in range(f0, f1):
        o = int(round(float(offs[f])))
        if ax == "x":
            x, y = max(0, min(max(0, W - cw), o)), 0
        elif ax == "y":
            x, y = 0, max(0, min(max(0, H - ch), o))
        else:
            x, y = cx_c, cy_c
        preds.append({"frame": int(f), "bboxes": [int(x), int(y), int(cw)]})
    log("used: %s  subject=%r  frames=[%d,%d)  %d preds"
        % (used, subject[:20], f0, f1, len(preds)))
    return preds, None


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Qwen2.5-VL highlight baseline")
    ap.add_argument("--index", default=os.path.join(here, "test_index.json"))
    ap.add_argument("--video-dir", default=os.path.join(here, "video"))
    ap.add_argument("--out", default=os.path.join(here, "predictions_vlm.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--load-4bit", dest="load_4bit", action="store_true",
                    default=True,
                    help="quantize weights (default: on; bf16 3B OOMs on 8GB)")
    ap.add_argument("--no-4bit", dest="load_4bit", action="store_false",
                    help="full bf16 weights -- only fits on a bigger card")
    ap.add_argument("--num-videos", type=int, default=0)
    ap.add_argument("--stage1-frames", type=int, default=12)
    ap.add_argument("--stage2-frames", type=int, default=8)
    ap.add_argument("--max-side", type=int, default=512,
                    help="frames are resized so the long side is at most this")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--widen", type=float, default=1.0,
                    help=">1 expands the model interval about its centre (hedge)")
    ap.add_argument("--min-keep-ratio", type=float, default=0.0)
    ap.add_argument("--max-keep-ratio", type=float, default=1.0)
    ap.add_argument("--no-stage2", action="store_true",
                    help="skip the spatial stage (measured harmful on val: "
                         "mean IoU 0.653 centred -> 0.580 with the model box)")
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="1=trust the model's focus point, 0=frame centre. "
                         "Default 0: on val the model box cost 0.033 F1")
    ap.add_argument("--ema", type=float, default=0.35)
    ap.add_argument("--smooth-win", type=int, default=3)
    ap.add_argument("--no-vlm", action="store_true",
                    help="skip the model entirely: emit the tau=1 hedge")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    vlm = None
    if not args.no_vlm:
        try:
            vlm = QwenVL(args.model, args.device, args.load_4bit, verbose=True)
        except Exception as e:
            print("[warn] VLM unavailable (%s: %s)" % (type(e).__name__, e))
            print("[warn] every video falls back to the tau=1 centred hedge")

    with open(args.index, "r", encoding="utf-8") as f:
        index = json.load(f)
    if args.num_videos > 0:
        index = index[:args.num_videos]

    t0 = time.time()
    n_lines = n_entries = n_empty = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for vi, it in enumerate(index, 1):
            vid = str(it["video_id"])
            tr = it.get("targetRatioWH", [16, 9])
            tw, th = float(tr[0]), float(tr[1])
            logs = []
            log = (lambda s: (print("      %s" % s), logs.append(s))) \
                if args.verbose else (lambda s: None)
            try:
                preds, err = process_video(vid, tw, th, args, vlm, log)
            except Exception as e:
                # Last resort. An empty list scores 0 for this video, so probe
                # the video and emit the hedge; only give up on an empty list if
                # even the probe fails.
                preds, err = None, "exception: %s: %s" % (type(e).__name__, e)
                try:
                    m = probe_one(os.path.join(args.video_dir, vid + ".mp4"))
                    if m and m.get("n_frames"):
                        preds = hedge_preds(m["W"], m["H"], int(m["n_frames"]),
                                            tw, th)
                except Exception:
                    pass
            if preds is None:
                preds = []
                n_empty += 1
            if err:
                print("  [warn] vid %s: %s" % (vid, err))
            fout.write(json.dumps(
                {"video_id": vid, "targetRatioWH": [int(tw), int(th)],
                 "predictions": preds}, ensure_ascii=False) + "\n")
            fout.flush()
            n_lines += 1
            n_entries += len(preds)
            if vi % 5 == 0 or vi == len(index):
                el = time.time() - t0
                print("  %d/%d  (%d entries, %.1fs, %.1fs/video)"
                      % (vi, len(index), n_entries, el, el / vi))
    print("Done: %d videos / %d predictions -> %s" % (n_lines, n_entries, args.out))
    if n_empty:
        print("[warn] %d videos produced NO predictions (F1=0 for those); "
              "the hedge should have covered them" % n_empty)


if __name__ == "__main__":
    main()
