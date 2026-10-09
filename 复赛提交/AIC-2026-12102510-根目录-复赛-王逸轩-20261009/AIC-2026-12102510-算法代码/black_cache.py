#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建「首/尾黑屏片头片尾」检测用的逐帧亮度缓存。

为什么
------
有些视频开头或结尾带着短视频软件的自动片头/片尾（如 VIDEVO / 抖音 的 logo
卡），画面大部分是黑或深灰、且**没有主体**。这类帧若照常输出裁剪框，会以
"我们预测了、GT 没有" 的形式拉低 F1（F1 = 2|P∩G|/(|P|+|G|)）。所以在**开头或
结尾**遇到"大部分黑灰且无目标"的帧，应当**不框选**（= 把这些帧从帧集里去掉）。

本脚本只负责把"每抽样帧有多黑"量出来存成缓存；"丢不丢、丢多少"交给
emit_trace.py 的 `--black-cache` 分支决定——阈值可调，不必重新解码。

产物
----
{vid: {"W","H","n_frames","frames":[抽样帧号...],
       "mean":[...每抽样帧灰度均值...], "dark":[...低于 dark-pixel 的像素占比...]}}
`frames` 与 YOLO 主缓存同 stride=5，可直接按帧号对齐。mean/dark 存原始值，
下游换阈值无需重跑。

用法
    python3 black_cache.py --video-dir fu_testset/video --metadata fu_metadata.json \
        --out fu_black_cache.json --stride 5 --dark-pixel 60
    python3 black_cache.py --video-dir val_video --metadata val_metadata.json \
        --out val_black_cache.json
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def build(args):
    import cv2

    meta = json.load(open(args.metadata, encoding="utf-8"))
    ids = list(meta)
    ids.sort(key=lambda s: (int(s) if s.isdigit() else 10 ** 9, s))
    if args.limit:
        ids = ids[:args.limit]
    print("亮度缓存：%d 视频（stride=%d, dark-pixel=%.0f）"
          % (len(ids), args.stride, args.dark_pixel))

    cache = {}
    t0 = time.time()
    for k, vid in enumerate(ids):
        m = meta.get(vid) or {}
        n = int(m.get("n_frames") or 0)
        W, H = int(m.get("W") or 0), int(m.get("H") or 0)
        path = os.path.join(args.video_dir, vid + ".mp4")
        if not (n and W and H) or not os.path.exists(path):
            print("  [warn] skip %s" % vid)
            continue
        cap = cv2.VideoCapture(path)
        frames, means, darks = [], [], []
        # 顺序解码：抽样帧 read()（解码），其余 grab()（只推进）。逐帧 seek 会在
        # 长视频上反复回退到关键帧，慢十几倍（见 pet_cache.py 的注释）。
        i = 0
        while i < n:
            if i % args.stride == 0:
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    break
                g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                frames.append(i)
                means.append(round(float(g.mean()), 2))
                darks.append(round(float((g < args.dark_pixel).mean()), 4))
            else:
                if not cap.grab():
                    break
            i += 1
        cap.release()
        cache[vid] = {"W": W, "H": H, "n_frames": n, "frames": frames,
                      "mean": means, "dark": darks}
        if (k + 1) % 25 == 0 or k + 1 == len(ids):
            print("  ... %d/%d (%.0fs)" % (k + 1, len(ids), time.time() - t0))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    print("\nwrote %s (%d 视频, %.1f MB)"
          % (args.out, len(cache), os.path.getsize(args.out) / 1e6))
    return 0


def main():
    ap = argparse.ArgumentParser(description="首/尾黑屏检测的逐帧亮度缓存")
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--dark-pixel", type=float, default=60.0,
                    help="单像素灰度低于该值算 '黑'（0-255）")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    return build(args)


if __name__ == "__main__":
    sys.exit(main())
