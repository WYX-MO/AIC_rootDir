#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为「VLM 摘要主语是动物」的视频单独跑一遍低阈值、只检 cat/dog 的 YOLO 缓存。

为什么单独做
------------
主缓存 fu_yolo_test_cache.json 是 YOLO 默认 conf=0.25 全类检出的。宠物（尤其中远景
里奔跑的猫狗）又小又糊，conf 常低于 0.25 直接丢检 → 逐帧追踪无中心可插值。所以给
动物主语视频单独一遍：**只检 cat/dog（YOLO 会跳过 person，省得被主人抢中心）+ 更低
的 conf 阈值**，把远处的小宠物也捞回来。

产物格式与主缓存完全一致：{vid: {"W","H","n_frames","frames","dets"}}，
dets 元素 {cls,conf,cx,cy,area}，cx/cy 已按 W/H 归一化。emit_trace.py 直接读。

用法：
  python3 pet_cache.py --summaries vlm_summaries_test.json \
      --video-dir fu_testset/video --metadata fu_metadata.json \
      --out fu_pet_cache.json --conf 0.05
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DEFAULT_WEIGHTS = "/mnt/data/ML/模型实践与复现/YOLO/yolo11n.pt"

# 与 vlm_summary.py / emit_trace.py 同口径的「动物做主语」判据
ANIMALS = ["猫", "狗", "柯基", "犬", "猫咪", "小狗", "猫猫", "狗狗", "宠物",
           "金毛", "泰迪", "柴犬", "哈士奇", "边牧", "萨摩", "田园犬"]
ANIMAL_RE = re.compile("|".join(ANIMALS))


def animal_videos(summaries):
    """{vid: summary} -> 摘要主语像动物的视频 id 列表。"""
    return [v for v, s in summaries.items() if s and ANIMAL_RE.search(s)]


def build(args):
    from ultralytics import YOLO

    summaries = json.load(open(args.summaries, encoding="utf-8"))
    meta = json.load(open(args.metadata, encoding="utf-8"))
    vids = animal_videos(summaries)
    vids.sort(key=lambda s: (int(s) if s.isdigit() else 10 ** 9, s))
    if args.limit:
        vids = vids[:args.limit]
    print("动物主语视频 %d 条（conf=%.2f, classes=cat/dog, stride=%d）"
          % (len(vids), args.conf, args.stride))

    model = YOLO(args.weights)
    names = model.names
    pet_ids = [i for i, n in names.items() if n in ("cat", "dog")]
    print("cat/dog 类别 id:", pet_ids)

    import cv2
    cache = {}
    t0 = time.time()
    for k, vid in enumerate(vids):
        path = os.path.join(args.video_dir, vid + ".mp4")
        m = meta.get(vid) or {}
        W, H = int(m.get("W") or 0), int(m.get("H") or 0)
        n = int(m.get("n_frames") or 0)
        if not (W and H and n):
            print("  [warn] no metadata %s" % vid)
            continue
        cap = cv2.VideoCapture(path)
        frames, keep = [], []
        # 顺序解码：只在需要的那一帧 read()（解码），其余 grab()（只推进、不解码）。
        # 与逐帧 seek 相比，抽样下标完全一致（0,stride,2stride,...），但长视频快得多
        # ——一帧一帧 seek 会反复回退到关键帧，1 万帧的视频能慢十几倍。
        i = 0
        while i < n:
            if i % args.stride == 0:
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    break
                frames.append(bgr)
                keep.append(i)
            else:
                if not cap.grab():
                    break
            i += 1
        cap.release()

        dets = []
        for s in range(0, len(frames), args.batch):
            res = model(frames[s:s + args.batch], imgsz=args.imgsz, conf=args.conf,
                        classes=pet_ids, verbose=False, device=args.device)
            for r in res:
                per = []
                for b in r.boxes:
                    xyxy = [float(v) for v in b.xyxy[0].tolist()]
                    bw, bh = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
                    per.append({
                        "cls": names.get(int(b.cls[0]), str(int(b.cls[0]))),
                        "conf": round(float(b.conf[0]), 4),
                        "cx": round((xyxy[0] + xyxy[2]) / 2.0 / W, 5),
                        "cy": round((xyxy[1] + xyxy[3]) / 2.0 / H, 5),
                        "area": round((bw * bh) / float(W * H), 6),
                    })
                dets.append(per)
        cache[vid] = {"W": W, "H": H, "n_frames": n, "frames": keep, "dets": dets}
        nd = sum(len(d) for d in dets)
        nfr = sum(1 for d in dets if d)
        print("  %-6s %4d 抽样帧, 有宠物检出 %3d 帧, 共 %4d 框" % (vid, len(keep), nfr, nd))
        if (k + 1) % 10 == 0 or k + 1 == len(vids):
            print("  ... %d/%d (%.0fs)" % (k + 1, len(vids), time.time() - t0))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    empt = sum(1 for v in cache.values() if not any(v["dets"]))
    print("\nwrote %s  (%d 视频, %d 条零检出, %.1f MB)"
          % (args.out, len(cache), empt, os.path.getsize(args.out) / 1e6))
    return 0


def main():
    ap = argparse.ArgumentParser(description="动物主语视频的低阈值宠物缓存")
    ap.add_argument("--summaries", required=True)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    return build(args)


if __name__ == "__main__":
    sys.exit(main())
