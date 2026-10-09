#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""yolo+gated+trace —— 单主角视频改成"追踪"框，其余沿用 yolo+gated。

策略（在 emit_gated.py 之上加一条分支）
-------------------------------------
对每条视频，沿自由轴决定输出框：

  1. 门控回退（同 emit_gated）：杂乱度 > --threshold 或 无 person 检出
     → 沿用居中基线框（下界锚定 = 上一版冠军）。
  2. **单主角 + 未门控** → 逐帧追踪：把每抽样帧的 person 池化中心插值到"逐帧"、
     再做时序平滑，得到逐帧移动的裁剪框。这就是 yolo+gated+trace 新增的部分。
  3. 其余（多主体但不挤）→ 沿用 emit_gated 的**每视频常数**红框。

"单主角"判据（dets 无 track id，用逐帧 person 计数近似）
  单帧恰好 1 个 person 的抽样帧占比 ≥ --single-thr（默认 0.70）。

为什么单独拎出"单主角"
  emit_gated 的常数框胜过逐帧跟踪（val 上跟踪 Δ−0.0099），但那个负结果是在**全部
  视频**上测的，其中多主体视频的抖动占了主导。单主角视频没有"多个 near-tie 物体
  来回翻"的问题，逐帧信息或许是干净的——这条分支就是去验证这个假设。

用法
    python emit_trace.py --base predictions_val_center.jsonl \
        --cache yolo_val_cache.json --metadata val_metadata.json \
        --out predictions_val_trace.jsonl --threshold 2.0
    eval_local.py score --gt val_gt.jsonl --pred predictions_val_trace.jsonl \
        --video-dir val_video --metadata val_metadata.json
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from stage_data import crop_size, free_axis                       # noqa: E402
from emit_gated import person_series, resolve_weights, measure_model_size_mb  # noqa: E402

STRATEGY = "yolo+gated+trace"


def smooth(cs, k):
    """长度不变的滑动平均，edge 补齐。k=偶数时取 k|1。"""
    k = int(k)
    if k <= 1 or cs.size < 3:
        return cs
    k |= 1
    pad = k // 2
    ker = np.ones(k) / k
    return np.convolve(np.pad(cs, (pad, pad), mode="edge"), ker, mode="valid")


def frame_centers(centres, counts, frames, smooth_k):
    """逐帧中心：抽样帧中心 -> 插值到 base 的每一帧 -> 时序平滑。"""
    samp = np.array(sorted(counts), dtype=np.float64)
    cs = np.array([centres.get(int(f), np.nan) for f in samp], dtype=np.float64)
    good = ~np.isnan(cs)
    if good.sum() == 0:
        return None
    cs = np.interp(np.arange(len(cs)), np.arange(len(cs))[good], cs[good])
    cs = smooth(cs, smooth_k)
    fr = np.array(sorted(int(f) for f in frames), dtype=np.float64)
    return np.interp(fr, samp, cs)          # 逐帧中心，与 fr 对齐


def main():
    ap = argparse.ArgumentParser(description=STRATEGY)
    ap.add_argument("--base", required=True, help="居中基线 jsonl")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=2.0,
                    help="绝对杂乱度门（mean person/sampled frame），> 则回退居中")
    ap.add_argument("--single-thr", type=float, default=0.70,
                    help="抽样帧中'恰好 1 个 person'的占比阈值，≥ 判为单主角")
    ap.add_argument("--smooth", type=int, default=9,
                    help="时序平滑窗口（抽样帧数）；1=不平滑")
    ap.add_argument("--model-size-mb", type=float, default=None)
    ap.add_argument("--weights", action="append", default=None)
    args = ap.parse_args()

    wpaths = args.weights
    if wpaths is None:
        auto = resolve_weights()
        wpaths = [auto] if auto else []
    if args.model_size_mb is None:
        if not wpaths:
            raise SystemExit("cannot measure model_size_mb")
        size_mb, _ = measure_model_size_mb(wpaths)
    else:
        size_mb = args.model_size_mb

    base = [json.loads(l) for l in open(args.base, encoding="utf-8") if l.strip()]
    cache = json.load(open(args.cache, encoding="utf-8"))
    meta = json.load(open(args.metadata, encoding="utf-8"))

    out = []
    n_gated = n_single = n_red = 0
    for rec in base:
        vid = rec["video_id"]
        m = meta.get(vid) or cache.get(vid)
        W, H = int(m["W"]), int(m["H"])
        tw, th = rec["targetRatioWH"]
        cw, ch = crop_size(W, H, tw, th)
        ax = free_axis(W, H, tw, th)
        centres, counts = person_series(cache[vid], ax)

        nfr = len(counts)
        n1 = sum(1 for f in counts if counts[f] == 1)
        clash = (float(np.mean([counts[f] for f in sorted(counts)])) if counts else 0.0)
        gated = not (centres and clash <= args.threshold)
        single = (not gated) and ax is not None and nfr > 0 and (n1 / nfr) >= args.single_thr

        preds = rec["predictions"]
        boxes = [p["bboxes"] for p in preds]
        if boxes and boxes[0][2] != cw:
            raise SystemExit("%s: base w=%d but crop_size=%d" % (vid, boxes[0][2], cw))

        if gated or ax is None:
            new = boxes                                   # 回退：居中
            n_gated += 1
        elif single:
            fr = [p["frame"] for p in preds]
            cf = frame_centers(centres, counts, fr, args.smooth)
            if cf is None:
                new = boxes
                n_gated += 1
            else:
                if ax == "x":
                    travel = max(0, W - cw)
                    new = [[int(round(float(np.clip(c * W - cw / 2.0, 0, travel)))), 0, cw]
                           for c in cf]
                else:
                    travel = max(0, H - ch)
                    new = [[0, int(round(float(np.clip(c * H - ch / 2.0, 0, travel)))), cw]
                           for c in cf]
                n_single += 1
        else:
            cs = np.array(sorted(centres.values()), dtype=np.float64)
            c = float(np.median(cs))
            if ax == "x":
                travel = max(0, W - cw)
                o = int(round(float(np.clip(c * W - cw / 2.0, 0, travel))))
                new = [[o, 0, cw] for _ in boxes]
            else:
                travel = max(0, H - ch)
                o = int(round(float(np.clip(c * H - ch / 2.0, 0, travel))))
                new = [[0, o, cw] for _ in boxes]
            n_red += 1

        out.append({"video_id": vid, "targetRatioWH": rec["targetRatioWH"],
                    "model_size_mb": size_mb,
                    "predictions": [{"frame": p["frame"], "bboxes": b}
                                    for p, b in zip(preds, new)]})

    with open(args.out, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")

    print("strategy=%s -> %s" % (STRATEGY, args.out))
    print("  videos=%d  回退居中=%d  追踪(单主角)=%d  常数红框=%d"
          % (len(out), n_gated, n_single, n_red))
    print("  gate: clutter > %.2f 退回居中 | single: 单person帧占比 >= %.2f | smooth=%d"
          % (args.threshold, args.single_thr, args.smooth))
    print("  model_size_mb = %s" % size_mb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
