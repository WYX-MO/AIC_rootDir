#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用多模态模型（Qwen2.5-VL）只做「时间区间」选择，产出区间缓存。

复用 baseline_vlm.py 的 stage1（vlm_interval）。不碰空间：框臂由 run_experiments
从现成预测里沿用，这样对比只反映"时间选择"这一维。

产物：cache/vlm_intervals_<dataset>.json  = {vid: [f0, f1]}（帧区间，[f0,f1)）

用法：
  python3 vlm_select.py --dataset val
  python3 vlm_select.py --dataset test --ids fullframe
"""
import argparse
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE)
sys.path.insert(0, HERE)

os.environ.setdefault("HF_HOME", "/mnt/data/hf")
import baseline_vlm as BV                                    # noqa: E402
from stage_data import probe_one                             # noqa: E402
from run_experiments import DATASETS, load_pred_map, fullframe_ids   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="VLM 时间区间选择")
    ap.add_argument("--dataset", choices=list(DATASETS), required=True)
    ap.add_argument("--ids", default=None, help="test: 'fullframe' | 'all' | 逗号列表")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--load-4bit", dest="load_4bit", action="store_true", default=True)
    ap.add_argument("--no-4bit", dest="load_4bit", action="store_false")
    ap.add_argument("--stage1-frames", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--widen", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    meta = json.load(open(cfg["meta"], encoding="utf-8"))
    vsig = os.path.join(HERE, "cache", "signals_%s.json" % args.dataset)
    vtup = os.path.join(BASE, "val_video") if args.dataset == "val" else \
        os.path.join(BASE, "fu_testset", "video")
    suffix = ".mp4"

    ids = None
    if args.ids and args.ids != "all":
        box = load_pred_map(cfg["box_default"])
        ids = (fullframe_ids(meta, box) if args.ids == "fullframe"
               else args.ids.split(","))
    if ids is None:
        ids = list(json.load(open(vsig, encoding="utf-8")))   # 以信号缓存的 id 为全集
    if args.limit:
        ids = ids[:args.limit]

    out = os.path.join(HERE, "cache", "vlm_intervals_%s.json" % args.dataset)
    done = json.load(open(out, encoding="utf-8")) if os.path.exists(out) else {}

    print("加载模型 %s (4bit=%s) ..." % (args.model, args.load_4bit))
    vlm = BV.QwenVL(args.model, device=args.device, load_4bit=args.load_4bit, verbose=True)

    def log(_):
        pass

    for k, vid in enumerate(ids):
        if vid in done:
            continue
        m = meta.get(vid)
        if not m:
            continue
        path = os.path.join(vtup, vid + suffix)
        if not os.path.exists(path):
            continue
        n = int(m["n_frames"]); dur = float(m.get("duration") or 0.0)
        w_idx = BV.grab_frames(path, BV.even_indices(n, args.stage1_frames))
        if dur <= 0:
            dur = n / float(m.get("fps") or 25.0)
        iv = BV.vlm_interval(vlm, path, n, dur, w_idx, args, lambda s: None)
        done[vid] = [iv[0], iv[1]] if iv else None
        if (k + 1) % 5 == 0:
            print("  %d/%d" % (k + 1, len(ids)))
            json.dump(done, open(out, "w", encoding="utf-8"))

    json.dump(done, open(out, "w", encoding="utf-8"))
    ok = sum(1 for v in done.values() if v)
    print("wrote %s  (%d videos, %d 有区间)" % (out, len(done), ok))
    return 0


if __name__ == "__main__":
    sys.exit(main())
