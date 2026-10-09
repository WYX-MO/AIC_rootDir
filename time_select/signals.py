#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""逐视频抽"时间维"信号并缓存。

三条信号（全部对齐到「逐帧」网格，方便选择器直接切）：
  audio   —— ffmpeg 解 s16le 单声道 PCM -> 逐帧 RMS，归一化到 0..1
  motion  —— cv2 按 yolo 缓存同一批抽样帧解码，相邻抽样帧灰度差均值 -> 插值到逐帧
  person  —— 抽样帧里 cls=='person' 的检出数 -> 逐帧
  ndet    —— 抽样帧总检出数 -> 逐帧
  conf    —— 抽样帧 person 置信和 -> 逐帧

缓存格式（JSON，单文件一数据集）：
  { "<vid>": {"W":..,"H":..,"n_frames":..,"fps":..,
              "audio":[...], "motion":[...], "person":[...], "ndet":[...], "conf":[...]}, ... }

用法：
  python3 signals.py --dataset val
  python3 signals.py --dataset test             # 全部 426（只需帧网格已知的视频）
  python3 signals.py --dataset test --ids 0,174,175
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE)

from stage_data import crop_size, free_axis, probe_ffprobe   # noqa: E402

DATASETS = {
    "val": dict(
        video_dir=os.path.join(BASE, "val_video"),
        metadata=os.path.join(BASE, "val_metadata.json"),
        cache=os.path.join(BASE, "yolo_val_cache.json"),
        base=os.path.join(BASE, "predictions_val_center.jsonl"),
        suffix=".mp4",
    ),
    "test": dict(
        video_dir=os.path.join(BASE, "fu_testset", "video"),
        metadata=os.path.join(BASE, "fu_metadata.json"),
        cache=os.path.join(BASE, "fu_yolo_test_cache.json"),
        base=os.path.join(BASE, "fu_base.jsonl"),
        suffix=".mp4",
    ),
}


# ---------------------------------------------------------------- audio
def audio_rms(path, n_frames, fps, sr=16000):
    """逐帧 RMS 包络，归一化 0..1；无音轨/失败返回 None。"""
    try:
        out = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-f", "s16le",
             "-ac", "1", "-ar", str(sr), "-"],
            capture_output=True, timeout=180)
        if out.returncode != 0 or not out.stdout:
            return None
        a = np.frombuffer(out.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception:
        return None
    if a.size == 0:
        return None
    spf = float(sr) / float(fps or 25.0)
    env = np.zeros(max(1, n_frames), dtype=np.float32)
    for i in range(len(env)):
        seg = a[int(i * spf):int((i + 1) * spf)]
        if seg.size:
            env[i] = float(np.sqrt((seg.astype(np.float64) ** 2).mean()))
    m = float(env.max())
    if m > 0:
        env /= m
    return env


# ---------------------------------------------------------------- motion
def motion_series(path, samp_idx, width=160):
    """相邻抽样帧的灰度差均值（下采样到 width 宽后），长度 = len(samp_idx)。

    顺序 grab 跳帧（比逐帧 POS_FRAMES seek 快得多，1080p 上也够用）。
    """
    try:
        import cv2
    except Exception:
        return None
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    targets = sorted(int(f) for f in samp_idx)
    tset = set(targets)
    tmax = targets[-1] if targets else -1
    vals = []
    prev = None
    cur = -1
    while True:
        if not cap.grab():
            break
        cur += 1
        if cur in tset:
            ok, img = cap.retrieve()
            if not ok or img is None:
                vals.append(float("nan")); prev = None
            else:
                g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                h = max(1, int(g.shape[0] * width / g.shape[1]))
                g = cv2.resize(g, (width, h)).astype(np.float32)
                vals.append(float(np.abs(g - prev).mean()) if prev is not None else 0.0)
                prev = g
        if cur >= tmax:
            break
    cap.release()
    if len(vals) < len(targets):          # 视频短于预期，补齐
        vals += [float("nan")] * (len(targets) - len(vals))
    return np.array(vals[:len(targets)], dtype=np.float64)


# ---------------------------------------------------------------- yolo
def yolo_series(rec):
    """从 yolo 缓存记录取逐抽样帧 (person数, 总检出数, person置信和)。"""
    fr = rec["frames"]
    n = len(fr)
    person = np.zeros(n); ndet = np.zeros(n); conf = np.zeros(n)
    for i, dets in enumerate(rec["dets"]):
        ndet[i] = len(dets)
        for d in dets:
            if d.get("cls") == "person":
                person[i] += 1
                conf[i] += float(d.get("conf", 0.0))
    return person, ndet, conf


# ---------------------------------------------------------------- helpers
def _interp(vals_samp, samp_idx, n_frames):
    """抽样网格 -> 逐帧；全 NaN 则返回全 0。"""
    v = np.asarray(vals_samp, dtype=np.float64)
    s = np.asarray(samp_idx, dtype=np.float64)
    good = ~np.isnan(v)
    if good.sum() == 0:
        return np.zeros(n_frames, dtype=np.float64)
    if good.sum() == 1:
        return np.full(n_frames, float(v[good][0]))
    return np.interp(np.arange(n_frames, dtype=np.float64), s[good], v[good])


def _norm(a):
    a = np.asarray(a, dtype=np.float64)
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def extract_one(vid, path, meta_rec, yolo_rec):
    n = int(meta_rec["n_frames"]); fps = float(meta_rec.get("fps") or 25.0)
    samp = list(yolo_rec["frames"]) if yolo_rec else list(range(0, n, 5))
    audio = audio_rms(path, n, fps)
    if audio is None:
        audio = np.zeros(n, dtype=np.float64)
    motion_s = motion_series(path, samp)
    person_s, ndet_s, conf_s = yolo_series(yolo_rec) if yolo_rec else (
        np.zeros(len(samp)), np.zeros(len(samp)), np.zeros(len(samp)))
    return {
        "W": int(meta_rec["W"]), "H": int(meta_rec["H"]),
        "n_frames": n, "fps": fps,
        "audio": [round(float(x), 5) for x in audio],
        "motion": [round(float(x), 5) for x in _interp(motion_s, samp, n)],
        "person": [round(float(x), 5) for x in _interp(person_s, samp, n)],
        "ndet": [round(float(x), 5) for x in _interp(ndet_s, samp, n)],
        "conf": [round(float(x), 5) for x in _interp(conf_s, samp, n)],
    }


def load_base_ids(cfg):
    ids = []
    for line in open(cfg["base"], encoding="utf-8"):
        if line.strip():
            ids.append(json.loads(line)["video_id"])
    return ids


def main():
    ap = argparse.ArgumentParser(description="时间维信号抽取")
    ap.add_argument("--dataset", choices=list(DATASETS), required=True)
    ap.add_argument("--ids", default=None,
                    help="逗号分隔；默认全部（以 base 预测文件里的 id 为准）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    meta = json.load(open(cfg["metadata"], encoding="utf-8"))
    cache = json.load(open(cfg["cache"], encoding="utf-8"))
    out = args.out or os.path.join(HERE, "cache", "signals_%s.json" % args.dataset)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    ids = args.ids.split(",") if args.ids else load_base_ids(cfg)
    if args.limit:
        ids = ids[:args.limit]

    done = {}
    if os.path.exists(out):
        done = json.load(open(out, encoding="utf-8"))

    for k, vid in enumerate(ids):
        if vid in done:
            continue
        m = meta.get(vid) or cache.get(vid)
        if m is None:
            print("  skip (no meta): %s" % vid); continue
        path = os.path.join(cfg["video_dir"], vid + cfg["suffix"])
        if not os.path.exists(path):
            print("  skip (no video): %s" % vid); continue
        done[vid] = extract_one(vid, path, m, cache.get(vid))
        if (k + 1) % 10 == 0:
            print("  %d/%d" % (k + 1, len(ids)))
            json.dump(done, open(out, "w", encoding="utf-8"))

    json.dump(done, open(out, "w", encoding="utf-8"))
    print("wrote %s  (%d videos)" % (out, len(done)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
