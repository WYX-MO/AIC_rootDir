#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""yolo+gated+trace+pet —— 单主角视频改成"追踪"框，动物主语的视频追踪宠物。

策略（在 emit_gated.py 之上加一条分支）
-------------------------------------
对每条视频，沿自由轴决定输出框：

  1. 门控回退（同 emit_gated）：杂乱度 > --threshold 或 无主体检出
     → 沿用居中基线框（下界锚定 = 上一版冠军）。
  2. **单主角 + 未门控** → 逐帧追踪：把每抽样帧的主体（person，或宠物）池化中心
     插值到"逐帧"、再做时序平滑，得到逐帧移动的裁剪框。
  3. 其余（多主体但不挤）→ 沿用 emit_gated 的**每视频常数**红框。

动物主语优先（pet 分支）
  当**VLM 摘要的主语是动物**（vlm_summaries_*.json 命中猫/狗/宠物等词条），且视频里
  确有小宠物检出时，主体一律判为宠物——**即使画面里同时有 person**。否则
  subject_series 会因"person 存在"把中心让给主人，宠物视频（如"小狗在主人后面跑"）
  就聚焦错人。宠物视频一律逐帧追踪，不走单主角占比门槛。

宠物用小阈值单独检一遍（--pet-cache）
  主缓存是 conf=0.25 全类检出的；中远景奔跑的小猫小狗常低于该阈值直接丢检。
  pet_cache.py 对动物主语视频单独跑一遍 `classes=cat/dog, conf≈0.05` 的低阈值缓存，
  emit_trace 对这类视频改读它——"宠物比较小，阈值低一点"。

为什么单独拎出"单主角"
  emit_gated 的常数框胜过逐帧跟踪（val 上跟踪 Δ−0.0099），但那个负结果是在**全部
  视频**上测的，其中多主体视频的抖动占了主导。单主角视频没有"多个 near-tie 物体
  来回翻"的问题，逐帧信息或许是干净的——这条分支就是去验证这个假设。

用法
    python emit_trace.py --base fu_base.jsonl \
        --cache fu_yolo_test_cache.json --metadata fu_metadata.json \
        --summaries vlm_summaries_test.json --pet-cache fu_pet_cache.json \
        --out predictions_trace_pet.jsonl --threshold 2.0
"""
import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from stage_data import crop_size, free_axis                       # noqa: E402
from emit_gated import person_series, resolve_weights, measure_model_size_mb  # noqa: E402

STRATEGY = "yolo+gated+trace+pet"

# COCO 里就有的宠物类。测试集里 cat/dog 很多（val 几乎没有），而 old 逻辑在
# "检测不到 person" 时一律退回居中框，把整片让给了宠物视频。这里让无人但有宠物的
# 视频改聚焦猫狗——判据只换主体、下游逻辑（单主体追踪 / 常数框）完全复用。
PETS = ("cat", "dog")

# VLM 摘要主语像动物 -> 判为"动物主角"视频（与 pet_cache.py 同口径）。
ANIMALS = ["猫", "狗", "柯基", "犬", "猫咪", "小狗", "猫猫", "狗狗", "宠物",
           "金毛", "泰迪", "柴犬", "哈士奇", "边牧", "萨摩", "田园犬"]
ANIMAL_RE = re.compile("|".join(ANIMALS))


def is_animal_subject(summary):
    return bool(summary) and bool(ANIMAL_RE.search(summary))


def class_series(rec, ax, classes):
    """逐抽样帧：指定类别按 conf*area 池化 -> (中心, 计数)。与 person_series 同口径。"""
    key = "cx" if ax == "x" else "cy"
    centres, counts = {}, {}
    for fi, dets in zip(rec["frames"], rec["dets"]):
        num = den = 0.0
        n = 0
        for d in dets:
            if d["cls"] not in classes:
                continue
            n += 1
            w = float(d["conf"]) * max(float(d["area"]), 1e-9)
            if w <= 0:
                continue
            num += w * float(d[key])
            den += w
        if den > 0:
            centres[fi] = min(1.0, max(0.0, num / den))
        counts[fi] = n
    return centres, counts


def subject_series(rec, ax, prefer_pet=False):
    """选主体：有人 -> person（原行为）；无人有宠物 -> cat/dog；都无 -> 空。

    prefer_pet=True（VLM 摘要主语是动物）时**优先宠物**——即便画面里也有 person，
    也把中心给宠物（"小狗在主人后面跑"要盯狗，不是盯主人）。

    返回 (centres, counts, kind)，kind ∈ {"person","pet",None}（只用于统计打印）。
    """
    cp, np_ = class_series(rec, ax, PETS)
    if prefer_pet and cp:
        return cp, np_, "pet"
    c, n = person_series(rec, ax)
    if c:
        return c, n, "person"
    if cp:
        return cp, np_, "pet"
    return {}, {}, None


def head_series(rec, ax, head_frac):
    """把 person 框的中心从"躯干"抬到"头/脸"再池化：cy → top + head_frac*bh。

    竖屏源(ax=='y')的裁剪窗只能上下平移，旧口径 person_series 取框中心=胸腰，
    用户希望焦点落到人脸。头/脸 ≈ 框顶往下 head_frac(默认 0.20，偏脸/胸口；
    0.10 会偏到额头) 个框高。需要 face_cache.py 存的 `top`/`bh` 字段；缺字段的 det 原样用 cy。
    """
    r = dict(rec)
    dets = []
    for per in rec["dets"]:
        new = []
        for d in per:
            if d["cls"] == "person" and "top" in d and "bh" in d:
                d = dict(d)
                d["cy"] = min(1.0, max(0.0, float(d["top"]) + head_frac * float(d["bh"])))
            new.append(d)
        dets.append(new)
    r["dets"] = dets
    return person_series(r, ax)


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


# --------------------------------------------------------------------------
# 首/尾黑屏片头片尾 -> 不框选
# --------------------------------------------------------------------------
# 短视频软件常给视频自动加片头/片尾卡（VIDEVO / 抖音 logo 之类），画面大部分是
# 黑或深灰、且没有主体。这类帧若照常输出框，等于"预测了 GT 没有的帧"，按
# F1 = 2|P∩G|/(|P|+|G|) 直接拉低分数。判据只用"大部分黑灰 AND 无 person/pet 目标"
# —— 二者缺一不可：纯夜戏（如 vid7，mean≈8 但确有 person）不会被误丢。
def _head_run(mask, gap):
    """从 0 号抽样帧起的最大 slate 前缀的最后下标；无则 -1。允许 gap 个夹杂。"""
    n = len(mask)
    i, end = 0, -1
    while i < n:
        if mask[i]:
            end = i
            i += 1
            continue
        j = i
        while j < n and not mask[j]:
            j += 1
        if (j - i) > gap or j >= n:
            break
        i = j
    return end


def _tail_run(mask, gap):
    """到末号抽样帧止的最大 slate 后缀的起始下标；无则 len(mask)。"""
    n = len(mask)
    i, start = n - 1, n
    while i >= 0:
        if mask[i]:
            start = i
            i -= 1
            continue
        j = i
        while j >= 0 and not mask[j]:
            j -= 1
        if (i - j) > gap or j < 0:
            break
        i = j
    return start


def target_frames(dets, frames, pet_rec, tconf):
    """抽样帧号 -> 该帧是否有 person/pet 目标（conf>=tconf）。缺帧默认 True（保守保留）。"""
    have = {}
    for fi, per in zip(frames, dets):
        have[fi] = any(d["cls"] in ("person", "cat", "dog")
                       and float(d["conf"]) >= tconf for d in per)
    if pet_rec:                             # 动物主语视频的低阈值 cat/dog 缓存
        for fi, per in zip(pet_rec["frames"], pet_rec["dets"]):
            if have.get(fi):
                continue
            if any(d["cls"] in ("cat", "dog") and float(d["conf"]) >= tconf
                   for d in per):
                have[fi] = True
    return have


def slate_drop(pred_frames, blk, dets, dets_frames, pet_rec, args):
    """返回 (drop_idx_set, info) 或 None。

    在首/尾丢弃"大部分黑灰且无目标"的帧。drop_idx 是相对 pred_frames 的下标集合。
    """
    if not blk or not blk.get("frames"):
        return None
    bf = blk["frames"]
    dark = [(dk >= args.dark_frac and mn <= args.dark_mean)
            for mn, dk in zip(blk["mean"], blk["dark"])]
    have = target_frames(dets, dets_frames, pet_rec, args.target_conf)
    slate = [d and not have.get(f, False) for d, f in zip(dark, bf)]
    head_end = _head_run(slate, args.slate_gap)
    tail_start = _tail_run(slate, args.slate_gap)
    if head_end < 0 and tail_start >= len(slate):
        return None
    import bisect
    drop = set()
    for idx, f in enumerate(pred_frames):
        j = bisect.bisect_right(bf, f) - 1      # 最近的 <= f 的抽样帧
        if j < 0:
            j = 0
        if j <= head_end or j >= tail_start:
            drop.add(idx)
    if not drop or len(drop) > args.max_drop * max(1, len(pred_frames)):
        return None
    info = {"head_end": head_end, "tail_start": tail_start,
            "n_samp": len(slate), "n_slate_samp": int(sum(slate)), "bf": bf}
    return drop, info


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
    ap.add_argument("--summaries", default=None,
                    help="vlm_summaries_*.json；给了则'摘要主语是动物'的视频优先追踪宠物")
    ap.add_argument("--pet-cache", default=None,
                    help="pet_cache.py 的低阈值 cat/dog 缓存；动物主语视频改读它")
    ap.add_argument("--person-box-cache", default=None,
                    help="face_cache.py 的竖屏 person 整框缓存；竖屏人物视频把焦点从胸腰抬到人脸")
    ap.add_argument("--head-frac", type=float, default=0.24,
                    help="头/脸中心 = person 框顶 + head_frac*框高（竖屏聚焦人脸用；0.10 额头 / 0.20 脸 / 0.24 脸-胸口 / >0.30 切头）")
    ap.add_argument("--black-cache", default=None,
                    help="black_cache.py 的逐帧亮度缓存；给了则丢弃首/尾'大部分黑灰且无目标'的片头片尾帧")
    ap.add_argument("--dark-frac", type=float, default=0.80,
                    help="低于 --dark-pixel 的像素占比 >= 该值才算'大部分黑灰'")
    ap.add_argument("--dark-mean", type=float, default=60.0,
                    help="且该抽样帧灰度均值 <= 该值")
    ap.add_argument("--slate-gap", type=int, default=0,
                    help="黑屏 run 中允许夹杂的非黑屏抽样帧数（0=严格连续）")
    ap.add_argument("--target-conf", type=float, default=0.30,
                    help="person/pet 检出 conf 低于该值不算'有目标'")
    ap.add_argument("--max-drop", type=float, default=0.5,
                    help="单视频最多丢弃的帧比例上限（防止把整条丢空）")
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
    summaries = (json.load(open(args.summaries, encoding="utf-8"))
                 if args.summaries else {})
    pet_cache = (json.load(open(args.pet_cache, encoding="utf-8"))
                 if args.pet_cache else {})
    person_box_cache = (json.load(open(args.person_box_cache, encoding="utf-8"))
                        if args.person_box_cache else {})
    black = (json.load(open(args.black_cache, encoding="utf-8"))
             if args.black_cache else {})

    out = []
    n_gated = n_single = n_red = 0
    n_person = n_pet = 0
    n_pet_trace = 0
    n_animal_vids = 0
    n_head = 0
    n_slate_vids = 0
    n_slate_frames = 0
    for rec in base:
        vid = rec["video_id"]
        m = meta.get(vid) or cache.get(vid)
        W, H = int(m["W"]), int(m["H"])
        tw, th = rec["targetRatioWH"]
        cw, ch = crop_size(W, H, tw, th)
        ax = free_axis(W, H, tw, th)
        # VLM 摘要主语是动物 -> 优先宠物；有低阈值宠物缓存则改读它（小宠物捞回）。
        animal = is_animal_subject(summaries.get(vid, ""))
        n_animal_vids += animal
        src = pet_cache.get(vid, cache[vid]) if animal else cache[vid]
        centres, counts, kind = subject_series(src, ax, prefer_pet=animal)
        n_person += (kind == "person")
        n_pet += (kind == "pet")

        # 竖屏人物视频：把焦点从"胸腰"(person 框中心)抬到"人脸"(框顶+head_frac*框高)。
        portrait_person = (ax == "y" and kind == "person")
        if portrait_person and person_box_cache.get(vid):
            hc, hcnt = head_series(person_box_cache[vid], ax, args.head_frac)
            if hc:
                centres, counts = hc, hcnt
                n_head += 1

        nfr = len(counts)
        n1 = sum(1 for f in counts if counts[f] == 1)
        clash = (float(np.mean([counts[f] for f in sorted(counts)])) if counts else 0.0)
        # 宠物直接逐帧追踪：不受杂乱度门限制（宠物小、框低阈值；多宠也照追）。
        # 竖屏人物同理：用户要聚焦人脸，不走回退居中的门。
        gated = (not centres) or (kind != "pet" and not portrait_person
                                  and clash > args.threshold)
        # 单主角判据：person 视频看"恰好 1 个 person"的抽样帧占比；
        # 宠物视频本就单主体 → 直接追踪，不设占比门槛（用户要求跟踪宠物）。
        is_single = nfr > 0 and (n1 / nfr) >= args.single_thr
        single = (not gated) and ax is not None and (kind == "pet" or is_single)

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
                n_pet_trace += (kind == "pet")
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

        # 首/尾黑屏片头片尾：大部分黑灰且无目标 -> 不框选（丢帧）。
        if black.get(vid):
            res = slate_drop([p["frame"] for p in preds], black[vid],
                             cache[vid]["dets"], cache[vid]["frames"],
                             pet_cache.get(vid), args)
            if res:
                drop, _info = res
                preds = [p for i, p in enumerate(preds) if i not in drop]
                new = [b for i, b in enumerate(new) if i not in drop]
                n_slate_vids += 1
                n_slate_frames += len(drop)

        out.append({"video_id": vid, "targetRatioWH": rec["targetRatioWH"],
                    "model_size_mb": size_mb,
                    "predictions": [{"frame": p["frame"], "bboxes": b}
                                    for p, b in zip(preds, new)]})

    with open(args.out, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")

    print("strategy=%s -> %s" % (STRATEGY, args.out))
    print("  动物主语视频(VLM 摘要命中) = %d" % n_animal_vids)
    print("  竖屏人物聚焦人脸 = %d" % n_head)
    print("  videos=%d  回退居中=%d  追踪=%d  常数红框=%d"
          % (len(out), n_gated, n_single, n_red))
    print("  其中追踪的宠物视频=%d" % n_pet_trace)
    print("  主体: person=%d  pet(cat/dog)=%d  无主体=%d"
          % (n_person, n_pet, len(out) - n_person - n_pet))
    print("  gate: clutter > %.2f 退回居中(宠物除外) | single: 单person帧占比 >= %.2f | smooth=%d"
          % (args.threshold, args.single_thr, args.smooth))
    if black:
        print("  首/尾黑屏丢帧: %d 视频, 共丢 %d 帧 (dark_frac>=%.2f 且 mean<=%.0f 且 无目标)"
              % (n_slate_vids, n_slate_frames, args.dark_frac, args.dark_mean))
    if summaries:
        print("  summaries=%s" % args.summaries)
    if pet_cache:
        print("  pet-cache=%s (%d 条)" % (args.pet_cache, len(pet_cache)))
    if person_box_cache:
        print("  person-box-cache=%s (%d 条) head-frac=%.2f"
              % (args.person_box_cache, len(person_box_cache), args.head_frac))
    print("  model_size_mb = %s" % size_mb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
