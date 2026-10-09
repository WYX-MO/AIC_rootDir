#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""跑时间维选择实验：固定框臂，只改帧集，对比 val 分数 + 出测试集预测。

  框臂固定 = 用一份现成预测（--box-source）里的逐帧框，选择器只在它的帧网格上取子集。
  val   : 对 val_gt.jsonl 打分（唯一能出分的地方）。
  test  : 对 159 条"源=目标画幅"的条目出预测（无 GT，供人工看）。

用法：
  python3 run_experiments.py --dataset val
  python3 run_experiments.py --dataset test --ids fullframe
  python3 run_experiments.py --dataset test --ids 0,174,175 --experiments best_window_combined_0.6
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE)
sys.path.insert(0, HERE)

import eval_local as ev                                     # noqa: E402
import frame_selectors as S                                 # noqa: E402
from stage_data import crop_size                            # noqa: E402

DATASETS = {
    "val": dict(meta=os.path.join(BASE, "val_metadata.json"),
                sig=os.path.join(HERE, "cache", "signals_val.json"),
                gt=os.path.join(BASE, "val_gt.jsonl"),
                box_default=os.path.join(BASE, "predictions_val_trace.jsonl")),
    "test": dict(meta=os.path.join(BASE, "fu_metadata.json"),
                 sig=os.path.join(HERE, "cache", "signals_test.json"),
                 gt=None,
                 box_default=os.path.join(BASE, "predictions_trace.jsonl")),
}

# 实验矩阵：name -> selector + 参数（+ 可选 score 合成）
EXPERIMENTS = [
    dict(name="keep_all",                 sel="keep_all"),
    dict(name="position_0.16_0.88",       sel="position_window", p=dict(lo=0.16, hi=0.88)),
    dict(name="window_audio_0.6",         sel="best_window", p=dict(frac=0.6), keys=("audio",),     w=(1.0,)),
    dict(name="window_audio_0.5",         sel="best_window", p=dict(frac=0.5), keys=("audio",),     w=(1.0,)),
    dict(name="window_motion_0.6",        sel="best_window", p=dict(frac=0.6), keys=("motion",),    w=(1.0,)),
    dict(name="window_person_0.6",        sel="best_window", p=dict(frac=0.6), keys=("person",),    w=(1.0,)),
    dict(name="window_comb_0.5",          sel="best_window", p=dict(frac=0.5), keys=("audio", "motion", "person"), w=(1.0, 1.0, 1.0)),
    dict(name="window_comb_0.6",          sel="best_window", p=dict(frac=0.6), keys=("audio", "motion", "person"), w=(1.0, 1.0, 1.0)),
    dict(name="window_comb_0.7",          sel="best_window", p=dict(frac=0.7), keys=("audio", "motion", "person"), w=(1.0, 1.0, 1.0)),
    dict(name="window_audioMotion_0.6",   sel="best_window", p=dict(frac=0.6), keys=("audio", "motion"),            w=(1.0, 1.0)),
    dict(name="top_comb_0.6",             sel="top_frac",    p=dict(frac=0.6), keys=("audio", "motion", "person"), w=(1.0, 1.0, 1.0)),
    dict(name="multi_comb_0.6_k2",        sel="multi_window", p=dict(frac=0.6, k=2), keys=("audio", "motion", "person"), w=(1.0, 1.0, 1.0)),
    dict(name="multi_comb_0.7_k3",        sel="multi_window", p=dict(frac=0.7, k=3), keys=("audio", "motion", "person"), w=(1.0, 1.0, 1.0)),
    # VLM 时间区间臂（由 vlm_select.py 产出 cache/vlm_intervals_<dataset>.json）
    dict(name="vlm_1.0",                  sel="interval", ext="vlm_intervals", p=dict(widen=1.0)),
    dict(name="vlm_1.25",                 sel="interval", ext="vlm_intervals", p=dict(widen=1.25)),
    dict(name="vlm_1.5",                  sel="interval", ext="vlm_intervals", p=dict(widen=1.5)),
]


def load_pred_map(path):
    """predictions jsonl -> {vid: (ratio, [(frame, box), ...])}，保序。"""
    out = {}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        out[r["video_id"]] = (r["targetRatioWH"],
                              [(p["frame"], p["bboxes"]) for p in r["predictions"]],
                              r.get("model_size_mb"))
    return out


def fullframe_ids(meta, box):
    ids = []
    for vid, (ratio, preds, _) in box.items():
        m = meta.get(vid)
        if not m:
            continue
        W, H = int(m["W"]), int(m["H"])
        if crop_size(W, H, ratio[0], ratio[1]) == (W, H):
            ids.append(vid)
    return ids


def build_run(exp, sig_all, box, ext_map=None):
    """按实验对一个视频算保留帧。"""
    out = {}
    ext_map = ext_map or {}
    for vid, (ratio, preds, msize) in box.items():
        if vid not in sig_all or not preds:
            out[vid] = (ratio, preds, msize)
            continue
        sig = sig_all[vid]
        grid = [int(f) for f, _ in preds]
        score = None
        if exp.get("keys"):
            score = S.combined_score(sig, exp["keys"], exp.get("w", (1,) * len(exp["keys"])))
        kw = dict(exp.get("p", {}))
        if exp.get("ext"):
            kw["iv"] = ext_map.get(vid)
        keep = set(S.REGISTRY[exp["sel"]](sig, grid, score=score, **kw))
        newpreds = [(f, b) for f, b in preds if f in keep]
        if not newpreds:                      # 兜底：选择器为空则退回整格
            newpreds = preds
        out[vid] = (ratio, newpreds, msize)
    return out


def write_pred(path, predmap):
    with open(path, "w", encoding="utf-8") as f:
        for vid, (ratio, preds, msize) in predmap.items():
            f.write(json.dumps({
                "video_id": vid, "targetRatioWH": ratio,
                "model_size_mb": msize,
                "predictions": [{"frame": f, "bboxes": b} for f, b in preds],
            }) + "\n")


def score_val(pred_path, cfg):
    """用 eval_local 的部件算官方 mean-over-videos F1（skip 0/0）。"""
    gt = ev.load_gt(cfg["gt"])
    pred, _ = ev.load_pred_jsonl(pred_path)
    meta = ev.load_meta(cfg["meta"])
    f1s = []
    for vid, (ratio, gt_preds) in gt.items():
        tw, th = ratio
        m = meta.get(vid)
        gtf = {}
        for e in gt_preds:
            ok, f, box, _, _, _ = ev.check_prediction(e, vid, ratio, m, 0, 1.0)
            if ok:
                gtf[f] = box
        prf = {}
        if vid in pred:
            pr = pred[vid][1]
            for e in pr:
                ok, f, box, _, _, _ = ev.check_prediction(e, vid, ratio, m, 0, 1.0)
                if ok and f not in prf:
                    prf[f] = box
        Ssum = 0.0
        for f, pb in prf.items():
            gb = gtf.get(f)
            if gb is not None:
                Ssum += ev.iou_xyxy(pb, gb)
        Np, Ng = len(prf), len(gtf)
        if Np + Ng:
            f1s.append(2.0 * Ssum / (Np + Ng))
    return (sum(f1s) / len(f1s)) if f1s else 0.0


def main():
    ap = argparse.ArgumentParser(description="时间维选择实验")
    ap.add_argument("--dataset", choices=list(DATASETS), required=True)
    ap.add_argument("--box-source", default=None)
    ap.add_argument("--ids", default=None, help="test: 'fullframe' | 'all' | 逗号列表")
    ap.add_argument("--experiments", default="all", help="逗号分隔的实验名，或 all")
    ap.add_argument("--outdir", default=os.path.join(HERE, "out"))
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    box_path = args.box_source or cfg["box_default"]
    if not os.path.exists(cfg["sig"]):
        raise SystemExit("缺信号缓存 %s，先跑 signals.py --dataset %s" % (cfg["sig"], args.dataset))

    meta = json.load(open(cfg["meta"], encoding="utf-8"))
    sig_all = json.load(open(cfg["sig"], encoding="utf-8"))
    box = load_pred_map(box_path)

    if args.dataset == "test" and args.ids and args.ids != "all":
        keep_ids = (set(fullframe_ids(meta, box)) if args.ids == "fullframe"
                    else set(args.ids.split(",")))
        box = {v: x for v, x in box.items() if v in keep_ids}

    exps = EXPERIMENTS if args.experiments == "all" else [
        e for e in EXPERIMENTS if e["name"] in args.experiments.split(",")]
    if not exps:
        raise SystemExit("没有匹配的实验名")

    odir = os.path.join(args.outdir, args.dataset)
    os.makedirs(odir, exist_ok=True)

    print("dataset=%s  box=%s  videos=%d  experiments=%d"
          % (args.dataset, os.path.basename(box_path), len(box), len(exps)))

    rows = []
    for exp in exps:
        ext_map = None
        if exp.get("ext"):
            ep = os.path.join(HERE, "cache", "%s_%s.json" % (exp["ext"], args.dataset))
            if not os.path.exists(ep):
                print("  skip %s: 缺 %s（先跑 vlm_select.py）" % (exp["name"], ep))
                continue
            ext_map = json.load(open(ep, encoding="utf-8"))
        run = build_run(exp, sig_all, box, ext_map)
        outp = os.path.join(odir, "predictions_%s_%s.jsonl" % (args.dataset, exp["name"]))
        write_pred(outp, run)
        n_tot = sum(len(p) for _, p, _ in run.values())
        n_base = sum(len(p) for _, p, _ in box.values())
        rec = dict(name=exp["name"], out=os.path.basename(outp),
                   n_pred=n_tot, keep_ratio=round(n_tot / max(1, n_base), 3))
        if args.dataset == "val":
            rec["F1"] = round(score_val(outp, cfg), 5)
            print("  %-26s F1=%.5f  keep=%.3f" % (exp["name"], rec["F1"], rec["keep_ratio"]))
        else:
            print("  %-26s keep=%.3f  -> %s" % (exp["name"], rec["keep_ratio"], rec["out"]))
        rows.append(rec)

    if args.dataset == "val":
        rows.sort(key=lambda r: -r["F1"])
        csvp = os.path.join(odir, "summary_val.csv")
        with open(csvp, "w", newline="", encoding="utf-8") as f:
            wtr = csv.DictWriter(f, fieldnames=["name", "F1", "keep_ratio", "n_pred", "out"])
            wtr.writeheader(); wtr.writerows(rows)
        print("\n--- val 排名 ---")
        for r in rows:
            print("  %.5f  %-26s keep=%.3f" % (r["F1"], r["name"], r["keep_ratio"]))
        print("wrote %s" % csvp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
