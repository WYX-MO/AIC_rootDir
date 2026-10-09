#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为「竖屏源(free_axis='y')」视频单独跑一遍 person 检测，**存整框几何**（顶边+高）。

为什么要单独做
------------
竖屏源 → 目标 16:9 时裁剪窗只能**上下平移**（`free_axis='y'`），而主缓存只存了 person
框的 `cx/cy/area`，**没有框高**，无法知道头在哪。用户诉求：竖屏人物视频现在把窗压在
"胸和腰"（= person 框中心），希望焦点落到**人脸**。人脸≈person 框顶部往下 ~10% 框高，
所以必须知道 `y1` 和 `bh`。

产物 dets 元素在原有基础上多两个字段：
  top = y1/H（框顶归一化，= 头顶附近）、bh = (y2-y1)/H（框高归一化）。
emit_trace.py 用 `head_cy = top + head_frac*bh` 把主体中心从"躯干"抬到"头/脸"。

用法：
  python3 face_cache.py --video-dir fu_testset/video --metadata fu_metadata.json \
      --base fu_base.jsonl --out fu_person_box_cache.json --conf 0.25
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from stage_data import free_axis                                    # noqa: E402

DEFAULT_WEIGHTS = "/mnt/data/ML/模型实践与复现/YOLO/yolo11n.pt"


def portrait_videos(base, meta):
    """源为竖屏且裁剪窗只能上下平移（free_axis=='y'）的视频 id。"""
    out = []
    for rec in base:
        v = rec["video_id"]
        m = meta.get(v) or {}
        W, H = int(m.get("W") or 0), int(m.get("H") or 0)
        if not (W and H):
            continue
        if free_axis(W, H, *rec["targetRatioWH"]) == "y":
            out.append(v)
    return out


def build(args):
    import cv2
    from ultralytics import YOLO

    base = [json.loads(l) for l in open(args.base, encoding="utf-8") if l.strip()]
    meta = json.load(open(args.metadata, encoding="utf-8"))
    vids = portrait_videos(base, meta)
    vids.sort(key=lambda s: (int(s) if s.isdigit() else 10 ** 9, s))
    print("竖屏(free_axis=y)视频 %d 条（classes=person, conf=%.2f, stride=%d）"
          % (len(vids), args.conf, args.stride))

    model = YOLO(args.weights)
    names = model.names
    pid = [i for i, n in names.items() if n == "person"]

    cache = {}
    t0 = time.time()
    for k, vid in enumerate(vids):
        m = meta[vid]
        W, H, n = int(m["W"]), int(m["H"]), int(m["n_frames"])
        cap = cv2.VideoCapture(os.path.join(args.video_dir, vid + ".mp4"))
        frames, keep = [], []
        i = 0
        while i < n:
            if i % args.stride == 0:
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    break
                frames.append(bgr)
                keep.append(i)
            elif not cap.grab():
                break
            i += 1
        cap.release()

        dets = []
        for s in range(0, len(frames), args.batch):
            res = model(frames[s:s + args.batch], imgsz=args.imgsz, conf=args.conf,
                        classes=pid, verbose=False, device=args.device)
            for r in res:
                per = []
                for b in r.boxes:
                    xyxy = [float(v) for v in b.xyxy[0].tolist()]
                    bw, bh = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
                    per.append({
                        "cls": "person",
                        "conf": round(float(b.conf[0]), 4),
                        "cx": round((xyxy[0] + xyxy[2]) / 2.0 / W, 5),
                        "cy": round((xyxy[1] + xyxy[3]) / 2.0 / H, 5),
                        "top": round(xyxy[1] / H, 5),      # 框顶（≈头顶）
                        "bh": round(bh / H, 5),            # 框高
                        "area": round((bw * bh) / float(W * H), 6),
                    })
                dets.append(per)
        cache[vid] = {"W": W, "H": H, "n_frames": n, "frames": keep, "dets": dets}
        nd = sum(len(d) for d in dets)
        nfr = sum(1 for d in dets if d)
        print("  %-5s %4d 抽样帧, 有人 %3d 帧, 共 %4d 框" % (vid, len(keep), nfr, nd))
        if (k + 1) % 10 == 0 or k + 1 == len(vids):
            print("  ... %d/%d (%.0fs)" % (k + 1, len(vids), time.time() - t0))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    print("\nwrote %s  (%d 视频, %.1f MB)"
          % (args.out, len(cache), os.path.getsize(args.out) / 1e6))
    return 0


def main():
    ap = argparse.ArgumentParser(description="竖屏视频 person 整框缓存（含头顶/框高）")
    ap.add_argument("--base", required=True)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    return build(args)


if __name__ == "__main__":
    sys.exit(main())
