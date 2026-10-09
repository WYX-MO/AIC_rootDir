#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""时间维选择可视化：逐视频画信号曲线 + 各方案保留区间。

上排：音频 RMS / 运动 / person 三条逐帧信号（各自归一化，仅示意形状）。
下排：帧集区间条——GT 真值窗（仅 val）/ keep_all / VLM 区间 / 位置先验，
      以及对应方案的 frame-Dice（val 才有）。

产物：out/viz/<dataset>_<vid>.png

用法：
  python3 viz_time.py --dataset val --ids qvh_000020_9x16,qvh_000032_9x16
  python3 viz_time.py --dataset test --ids 174,175,176
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE)
sys.path.insert(0, HERE)

from run_experiments import DATASETS, load_pred_map, fullframe_ids   # noqa: E402

for f in ("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
          "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
    if os.path.exists(f):
        try:
            font_manager.fontManager.addfont(f)
        except Exception:
            pass
plt.rcParams["font.family"] = ["Droid Sans Fallback", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _norm(a):
    a = [float(x) for x in a]
    lo, hi = min(a), max(a)
    return [(x - lo) / (hi - lo) if hi > lo else 0.0 for x in a]


def _dice(P, G):
    P, G = set(P), set(G)
    if not P and not G:
        return None
    return 2.0 * len(P & G) / (len(P) + len(G))


def load_gt_frames(cfg):
    if not cfg.get("gt"):
        return {}
    out = {}
    for line in open(cfg["gt"], encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out[r["video_id"]] = sorted({p["frame"] for p in r["predictions"]})
    return out


def main():
    ap = argparse.ArgumentParser(description="时间维选择可视化")
    ap.add_argument("--dataset", choices=list(DATASETS), required=True)
    ap.add_argument("--ids", required=True, help="逗号列表 | fullframe")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    meta = json.load(open(cfg["meta"], encoding="utf-8"))
    sig = json.load(open(cfg["sig"], encoding="utf-8"))
    box = load_pred_map(cfg["box_default"])
    gt = load_gt_frames(cfg)
    vip = os.path.join(HERE, "cache", "vlm_intervals_%s.json" % args.dataset)
    vlm = json.load(open(vip, encoding="utf-8")) if os.path.exists(vip) else {}

    if args.ids == "fullframe":
        ids = fullframe_ids(meta, box)
    else:
        ids = args.ids.split(",")
    if args.limit:
        ids = ids[:args.limit]

    odir = os.path.join(HERE, "out", "viz")
    os.makedirs(odir, exist_ok=True)

    for vid in ids:
        if vid not in sig:
            print("skip (no signal): %s" % vid); continue
        n = int(sig[vid]["n_frames"])
        grid = [int(f) for f, _ in box.get(vid, (None, [], None))[1]]
        keep_all = [f for f in grid if f < 0.92 * n]
        pos = [f for f in grid if 0.16 * n <= f < 0.88 * n]
        iv = vlm.get(vid)
        vlmset = [f for f in grid if iv and iv[0] <= f < iv[1]]
        G = gt.get(vid, [])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6),
                                       gridspec_kw={"height_ratios": [3, 1]})
        x = list(range(n))
        for key, col in (("audio", "#1f77b4"), ("motion", "#ff7f0e"),
                         ("person", "#2ca02c")):
            if key in sig[vid]:
                ax1.plot(x, _norm(sig[vid][key]), lw=0.8, color=col, label=key)
        ax1.set_xlim(0, n)
        ax1.legend(loc="upper right", fontsize=8)
        ax1.set_title("%s  n=%d frames" % (vid, n), fontsize=10)
        if G:                                    # GT 窗高亮
            ax1.axvspan(min(G), max(G) + 1, color="k", alpha=0.08)
        ax1.set_ylabel("signal (norm)")

        def band(y, P, color, label, alpha=0.5):
            ax2.broken_barh([(min(P), max(P) - min(P) + 1)] if P else [],
                            (y - 0.3, 0.6), facecolors=color, alpha=alpha)
            ax2.text(-0.01 * n, y, label, ha="right", va="center", fontsize=8)

        rows = [("GT", G, "black"), ("keep_all", keep_all, "#888888"),
                ("pos[.16,.88]", pos, "#c7a2d9"), ("VLM", vlmset, "#d62728")]
        for i, (lab, P, col) in enumerate(rows):
            band(i, P, col, lab)
        ax2.set_ylim(-0.6, len(rows) - 0.4)
        ax2.set_xlim(0, n)
        ax2.set_yticks([])
        ax2.set_xlabel("frame index")
        dice = {"GT": _dice(G, G) if G else None,
                "keep_all": _dice(keep_all, G) if G else None,
                "VLM": _dice(vlmset, G) if G else None}
        ax2.set_title("frame-Dice  keep_all=%s vlm=%s" % (
            "%.3f" % dice["keep_all"] if dice["keep_all"] is not None else "-",
            "%.3f" % dice["VLM"] if dice["VLM"] is not None else "-"), fontsize=9)
        fig.tight_layout()
        p = os.path.join(odir, "%s_%s.png" % (args.dataset, vid))
        fig.savefig(p, dpi=110)
        plt.close(fig)
        print("wrote %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
