#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用多模态模型（Qwen2.5-VL）给每个视频出一句 ≤15 字的摘要，缓存给可视化软件读。

复用 baseline_vlm.py 的模型封装（QwenVL / grab_frames / even_indices）。
不看框、不看时间，只让模型"看一眼说这是啥"——纯内容摘要，供 view_result.py 显示。

产物：vlm_summaries_<dataset>.json = {video_id: "摘要（≤15字）"}
断点续跑：已缓存的 id 直接跳过。

用法：
  python3 vlm_summary.py --dataset test          # 复赛 426 条
  python3 vlm_summary.py --dataset val           # 本地 val 57 条
  python3 vlm_summary.py --dataset test --limit 8
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

os.environ.setdefault("HF_HOME", "/mnt/data/hf")
import baseline_vlm as BV                                    # noqa: E402

DATASETS = {
    "test": dict(video_dir=os.path.join(HERE, "fu_testset", "video"),
                 meta=os.path.join(HERE, "fu_metadata.json")),
    "val":  dict(video_dir=os.path.join(HERE, "val_video"),
                 meta=os.path.join(HERE, "val_metadata.json")),
}

PROMPT_SUMMARY = (
    "下面是从一个视频中等间隔抽取的 {k} 帧画面，按时间先后排列。\n"
    "请用**不超过 15 个汉字**的一句话，概括这段视频拍的是什么（主体 + 动作/场景）。\n"
    "只输出这句话本身，不要引号、不要标点以外的任何多余文字、不要解释。"
)

MAX_CHARS = 15


def clean_summary(text):
    """模型输出 -> 单行 ≤15 字的摘要（去掉引号/换行/前后缀）。"""
    if not text:
        return ""
    s = text.strip().splitlines()[0] if text.strip() else ""
    s = s.strip().strip('"').strip("'").strip("“”‘’").strip()
    # 去掉常见的客套前缀
    for pre in ("这段视频", "这个视频", "视频中", "视频里", "画面中", "画面里", "这是"):
        if s.startswith(pre):
            s = s[len(pre):]
            break
    s = re.sub(r"\s+", "", s)
    return s[:MAX_CHARS]


def main():
    ap = argparse.ArgumentParser(description="VLM 每视频一句话摘要")
    ap.add_argument("--dataset", choices=list(DATASETS), required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--load-4bit", dest="load_4bit", action="store_true", default=True)
    ap.add_argument("--no-4bit", dest="load_4bit", action="store_false")
    ap.add_argument("--frames", type=int, default=8, help="抽样帧数")
    ap.add_argument("--max-side", type=int, default=448)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    meta = json.load(open(cfg["meta"], encoding="utf-8"))

    out = os.path.join(HERE, "vlm_summaries_%s.json" % args.dataset)
    done = json.load(open(out, encoding="utf-8")) if os.path.exists(out) else {}

    vdir = cfg["video_dir"]
    ids = [v for v in meta if os.path.exists(os.path.join(vdir, v + ".mp4"))]
    ids.sort(key=lambda s: (int(s) if s.isdigit() else 10 ** 9, s))
    todo = [v for v in ids if v not in done]
    if args.limit:
        todo = todo[:args.limit]
    print("dataset=%s  共 %d 条, 待处理 %d 条 (已缓存 %d)"
          % (args.dataset, len(ids), len(todo), len(done)))
    if not todo:
        print("无待处理，退出。")
        return 0

    print("加载模型 %s (4bit=%s) ..." % (args.model, args.load_4bit))
    vlm = BV.QwenVL(args.model, device=args.device, load_4bit=args.load_4bit,
                    verbose=True)

    for k, vid in enumerate(todo):
        m = meta.get(vid) or {}
        n = int(m.get("n_frames") or 0)
        path = os.path.join(vdir, vid + ".mp4")
        if n <= 0:
            done[vid] = ""
            continue
        try:
            got = BV.grab_frames(path, BV.even_indices(n, args.frames), args.max_side)
            imgs = [im for _, im in got]
            if not imgs:
                done[vid] = ""
            else:
                txt = vlm.ask(imgs, PROMPT_SUMMARY.format(k=len(imgs)),
                              max_new_tokens=args.max_new_tokens)
                done[vid] = clean_summary(txt)
        except Exception as e:                      # 单条失败不中断
            print("  [warn] %s: %s: %s" % (vid, type(e).__name__, e))
            done[vid] = ""
        if (k + 1) % 5 == 0 or (k + 1) == len(todo):
            print("  %d/%d  last=%r" % (k + 1, len(todo), done[vid]))
            json.dump(done, open(out, "w", encoding="utf-8"), ensure_ascii=False)

    json.dump(done, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    ok = sum(1 for v in done.values() if v)
    print("wrote %s  (%d 条, %d 有摘要)" % (out, len(done), ok))
    return 0


if __name__ == "__main__":
    sys.exit(main())
