#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把某个时间选择策略的产物（只含 159 条"整幅"视频）并回完整 426 条提交文件。

除这 159 条外，其余视频沿用当前最优的空间臂（predictions_trace.jsonl 的框）。
输出可直接提交：out/test/predictions_timesel_<sel>.jsonl（426 行）。

用法：
  python3 assemble_test.py --sel keep_all
  python3 assemble_test.py --sel vlm_1.25 --spatial predictions_trace.jsonl
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser(description="并回完整提交文件")
    ap.add_argument("--sel", required=True, help="实验名，如 keep_all / vlm_1.25")
    ap.add_argument("--spatial", default=os.path.join(BASE, "predictions_trace.jsonl"),
                    help="空间臂底盘（426 条）")
    ap.add_argument("--temporal-dir", default=os.path.join(HERE, "out", "test"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tp = os.path.join(args.temporal_dir, "predictions_test_%s.jsonl" % args.sel)
    if not os.path.exists(tp):
        raise SystemExit("缺时间臂产物 %s（先跑 run_experiments.py --dataset test）" % tp)
    spatial = read_jsonl(args.spatial)
    temporal = {r["video_id"]: r for r in read_jsonl(tp)}

    out = args.out or os.path.join(HERE, "out", "test",
                                   "predictions_timesel_%s.jsonl" % args.sel)
    n_swapped = 0
    with open(out, "w", encoding="utf-8") as f:
        for r in spatial:
            vid = r["video_id"]
            if vid in temporal:
                t = temporal[vid]
                r = dict(r)
                r["predictions"] = t["predictions"]      # 只换帧集，框沿用空间臂
                n_swapped += 1
            f.write(json.dumps(r) + "\n")
    print("wrote %s  (426 行, 其中 %d 条换用时间策略 %s)"
          % (out, n_swapped, args.sel))
    return 0


if __name__ == "__main__":
    sys.exit(main())
