#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does a COCO detector earn its keep? One inference pass, three roles.

`baseline_cv.py` already showed that per-frame YOLO *tracking* is worth ~+0.001
(noise) on the old configuration. But the configuration has since changed
(`round` + drop-tail, val F1 0.3561 -> 0.3688) and only ONE of YOLO's three
possible roles was ever measured. This script measures all three in a single
pass so the answer is not a stale number.

The three roles
---------------
  A  CONSTANT per-video offset. Not tracking: take the median subject centre
     over the whole video and shift the crop by that one amount, all frames.
     Motivated by the GT boxes being off-centre per video (median -10.7px,
     |mean| 44.0px, range -160..+135) rather than off-centre per frame.
  B  PER-FRAME tracking. median_filter + EMA on the sampled centres, linearly
     interpolated to every frame -- i.e. what baseline_cv --alpha>0 does.
  C  TEMPORAL score. YOLO `conf * sqrt(area)` over time as a highlight signal,
     scored by AUC against the GT frame labels. Role of the detector as a
     *localizer* rather than a *framer*.

Protocol
--------
The frame set is held FIXED at the current champion's
`0 .. n - round(0.08*n) - 1` for every arm, and the box is centred except where
the arm says otherwise. That isolates the box contribution from the temporal
one, and makes every number directly comparable to 0.3688 (the shipped hedge
with the tail drop, whose box is exactly centred and which this script must
reproduce as a control before any delta is believed).

Frames are sampled every --stride; detections are cached to JSON so the JSON
can be re-scored without touching the GPU again.

Usage
-----
    python yolo_experiment.py --build-cache            # ~one GPU pass, cached
    python yolo_experiment.py                          # evaluate roles A/B/C
    python yolo_experiment.py --weights .../yolo11n.pt
"""

import os
import sys
import json
import time
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from stage_data import crop_size, free_axis, probe_one          # noqa: E402
from eval_local import iou_xyxy, box_from_triplet               # noqa: E402
from baseline_cv import offset_from_center, median_filter, ema  # noqa: E402

DEFAULT_WEIGHTS = "/mnt/data/ML/模型实践与复现/YOLO/yolo11n.pt"
ALPHAS = (0.25, 0.50, 0.75)


# --------------------------------------------------------------------------
# pass 1 -- inference
# --------------------------------------------------------------------------
def build_cache(args):
    import cv2
    from ultralytics import YOLO

    with open(args.index, "r", encoding="utf-8") as f:
        index = json.load(f)
    meta = json.load(open(args.metadata, "r", encoding="utf-8"))

    model = YOLO(args.weights)
    names = model.names
    cache = {}
    t0 = time.time()
    for k, rec in enumerate(index):
        vid = rec["video_id"]
        m = meta.get(vid) or probe_one(os.path.join(args.video_dir, vid + ".mp4"))
        if not m or not m.get("n_frames"):
            print("  [warn] no metadata for %s" % vid)
            continue
        W, H, n = m["W"], m["H"], int(m["n_frames"])
        idxs = list(range(0, n, args.stride))
        cap = cv2.VideoCapture(os.path.join(args.video_dir, vid + ".mp4"))
        frames, keep_idx = [], []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, bgr = cap.read()
            if not ok or bgr is None:
                continue
            frames.append(bgr)
            keep_idx.append(i)
        cap.release()
        if not frames:
            print("  [warn] no frames decoded for %s" % vid)
            continue

        dets = []
        for s in range(0, len(frames), args.batch):
            chunk = frames[s:s + args.batch]
            res = model(chunk, imgsz=args.imgsz, verbose=False, device=args.device)
            for r in res:
                per = []
                for b in r.boxes:
                    xyxy = [float(v) for v in b.xyxy[0].tolist()]
                    cf = float(b.conf[0])
                    cl = int(b.cls[0])
                    bw, bh = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
                    per.append({
                        "cls": names.get(cl, str(cl)),
                        "conf": round(cf, 4),
                        "cx": round((xyxy[0] + xyxy[2]) / 2.0 / W, 5),
                        "cy": round((xyxy[1] + xyxy[3]) / 2.0 / H, 5),
                        "area": round((bw * bh) / float(W * H), 6),
                    })
                dets.append(per)
        cache[vid] = {"W": W, "H": H, "n_frames": n,
                      "frames": keep_idx, "dets": dets}
        nd = sum(len(d) for d in dets)
        print("  %s  %d sampled frames, %d detections" % (vid, len(keep_idx), nd))
        if (k + 1) % 10 == 0 or k + 1 == len(index):
            print("  ... %d/%d videos (%.0fs)" % (k + 1, len(index), time.time() - t0))

    with open(args.cache, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    print("\nwrote %s  (%d videos, %.1f MB)"
          % (args.cache, len(cache), os.path.getsize(args.cache) / 1e6))
    return 0


# --------------------------------------------------------------------------
# pass 2 -- evaluation
# --------------------------------------------------------------------------
def subject_of(dets, cx_weight=0.0):
    """Pick the frame's subject. conf * sqrt(area), centricity tie-break."""
    best, best_s = None, -1.0
    for d in dets:
        cent = 1.0 - (abs(d["cx"] - 0.5) + abs(d["cy"] - 0.5))
        s = d["conf"] * (d["area"] ** 0.5) + cx_weight * cent
        if s > best_s:
            best, best_s = d, s
    return best


def centres_from_cache(rec, ax, want_all):
    """Sampled per-frame subject centres along the free axis.

    Returns (frame_idx[], centre[], score[], top1_name[]) -- the score is
    conf*sqrt(area) of the subject, 0 when nothing was detected.
    """
    fi, ce, sc, nm = [], [], [], []
    for f, dets in zip(rec["frames"], rec["dets"]):
        d = subject_of(dets)
        if d is None:
            fi.append(f); ce.append(0.5); sc.append(0.0); nm.append("-")
            continue
        fi.append(f)
        ce.append(d["cx"] if ax == "x" else d["cy"])
        sc.append(d["conf"] * (d["area"] ** 0.5))
        nm.append(d["cls"])
    return np.array(fi), np.array(ce), np.array(sc), nm


def window(n_frames, drop_tail):
    f1 = n_frames - int(round(drop_tail * n_frames))
    if f1 < 1:
        f1 = max(1, n_frames - int(round(drop_tail * n_frames)))
    return 0, f1


def boxes_from_offsets(rec, ax, cw, ch, W, H, offs_px):
    """Whole-window prediction from free-axis PIXEL offsets (already scaled)."""
    f0, f1 = window(rec["n_frames"], rec["_drop_tail"])
    n = f1 - f0
    travel = max(0, (W - cw) if ax == "x" else (H - ch) if ax == "y" else 0)
    offs = np.clip(np.asarray(offs_px, dtype=np.float64)[:n], 0, travel)
    preds = []
    for i in range(n):
        o = int(round(float(offs[i])))
        if ax == "x":
            x, y = max(0, min(W - cw, o)), 0
        elif ax == "y":
            x, y = 0, max(0, min(H - ch, o))
        else:
            x, y = int(round((W - cw) / 2.0)), int(round((H - ch) / 2.0))
        preds.append({"frame": int(f0 + i), "bboxes": [int(x), int(y), int(cw)]})
    return preds


def boxes_for(rec, ax, cw, ch, W, H, centre_norm):
    """Whole-window prediction from normalized free-axis CENTRES."""
    offs, _ = offset_from_center(np.asarray(centre_norm, dtype=np.float64),
                                 W, H, cw, ch, ax, 1.0)
    return boxes_from_offsets(rec, ax, cw, ch, W, H, offs)


def densify(fidx, vals, n_frames, window_n):
    """Sampled (frame, value) -> per-frame value over [0, window_n)."""
    if len(fidx) == 0:
        return np.full(window_n, 0.5)
    grid = np.arange(window_n, dtype=np.float64)
    if len(fidx) == 1:
        return np.full(window_n, float(vals[0]))
    return np.interp(grid, fidx.astype(np.float64), vals.astype(np.float64),
                     left=float(vals[0]), right=float(vals[-1]))


def score(gt_by_vid, preds_by_vid, meta):
    """Official metric: mean over videos of 2*S/(N_pred+N_gt)."""
    rows, tot_s, tot_m = [], 0.0, 0
    for vid, gt in gt_by_vid.items():
        m = meta[vid]
        W, H, n = m["W"], m["H"], int(m["n_frames"])
        tw, th = gt["targetRatioWH"]
        gt_map = {}
        for e in gt["predictions"]:
            if int(e["frame"]) not in gt_map:
                gt_map[int(e["frame"])] = box_from_triplet(
                    e["bboxes"][0], e["bboxes"][1], e["bboxes"][2], tw, th)
        N_gt = len(gt_map)
        S, nm = 0.0, 0
        for e in preds_by_vid.get(vid, []):
            f = int(e["frame"])
            if f not in gt_map:
                continue
            a = box_from_triplet(e["bboxes"][0], e["bboxes"][1], e["bboxes"][2], tw, th)
            b = gt_map[f]
            S += iou_xyxy(a, b)
            nm += 1
        N_pred = len(preds_by_vid.get(vid, []))
        den = N_pred + N_gt
        F1 = (2.0 * S / den) if den else None
        rows.append({"vid": vid, "F1": F1, "S": S, "match": nm,
                     "N_pred": N_pred, "N_gt": N_gt})
        tot_s += S
        tot_m += nm
    defined = [r for r in rows if r["F1"] is not None]
    f1 = float(np.mean([r["F1"] for r in defined])) if defined else 0.0
    return f1, rows, (tot_s / tot_m if tot_m else 0.0)


def auc(pos, neg):
    """Rank AUC via Mann-Whitney. pos/neg are score arrays."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1, dtype=np.float64)
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    for j, c in enumerate(cnt):
        if c > 1:
            sel = inv == j
            ranks[sel] = ranks[sel].mean()
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def evaluate(args):
    cache = json.load(open(args.cache, "r", encoding="utf-8"))
    meta = json.load(open(args.metadata, "r", encoding="utf-8"))
    gt_lines = [json.loads(l) for l in open(args.gt, "r", encoding="utf-8") if l.strip()]
    gt_by_vid = {r["video_id"]: r for r in gt_lines}

    ctrl, roleA, roleB, roleC = {}, {}, {}, {}
    sweep = {}
    auc_pos, auc_neg = [], []
    mae_c, mae_med = [], []
    detected = 0
    total_sampled = 0
    n_gt_frames_hit_by_det = 0
    n_gt_frames = 0

    for vid, gt in gt_by_vid.items():
        if vid not in cache:
            print("  [warn] %s missing from cache" % vid)
            continue
        rec = cache[vid]
        rec["_drop_tail"] = args.drop_tail
        W, H, n = rec["W"], rec["H"], rec["n_frames"]
        tw, th = gt["targetRatioWH"]
        cw, ch = crop_size(W, H, tw, th)
        ax = free_axis(W, H, tw, th)
        f0, f1 = window(n, args.drop_tail)
        win_n = f1 - f0
        fidx, ce, sc, nm = centres_from_cache(rec, ax, False)
        total_sampled += len(fidx)
        detected += int((sc > 0).sum())

        # ---- control: pure centre crop (must reproduce the shipped 0.3688)
        ctrl[vid] = boxes_for(rec, ax, cw, ch, W, H,
                              np.full(win_n, 0.5))

        # ---- Role A: one constant offset from the median subject centre
        a_centre = float(np.median(ce)) if len(ce) else 0.5
        roleA[vid] = boxes_for(rec, ax, cw, ch, W, H,
                               np.full(win_n, a_centre))

        # ---- Role B: per-frame tracking (median + EMA), dense
        # same order as baseline_cv: scale -> median -> ema -> clip
        dense = densify(fidx, ce, n, win_n)
        offs, travel = offset_from_center(dense, W, H, cw, ch, ax, 1.0)
        offs = median_filter(offs, 3)
        offs = ema(offs, args.ema)
        roleB[vid] = boxes_from_offsets(rec, ax, cw, ch, W, H, offs)

        # ---- Role B with a partial pull (alpha < 1), same as baseline_cv does
        for a in ALPHAS:
            op, _ = offset_from_center(dense, W, H, cw, ch, ax, a)
            op = ema(median_filter(op, 3), args.ema)
            sweep.setdefault(a, {})[vid] = boxes_from_offsets(
                rec, ax, cw, ch, W, H, op)

        # ---- Role C: temporal score AUC against GT frame labels
        dense_sc = densify(fidx, sc, n, win_n)
        gtf = set(int(e["frame"]) for e in gt["predictions"] if int(e["frame"]) < win_n)
        for i in range(win_n):
            if i in gtf:
                auc_pos.append(dense_sc[i])
            else:
                auc_neg.append(dense_sc[i])
        # Coverage must be measured on the SAME footing as the detection rate:
        # only GT frames that actually fall on a sampled index can be covered.
        # (Dividing by all GT frames while counting only sampled ones invents a
        # chance-level 1/stride number that says nothing about the detector.)
        det_at = {int(f): bool(s > 0) for f, s in zip(fidx, sc)}
        for f in gtf:
            if f in det_at:
                n_gt_frames += 1
                n_gt_frames_hit_by_det += int(det_at[f])

        # diagnostics: how far is the GT box centre from the frame centre?
        gt_c = []
        for e in gt["predictions"]:
            x, y, w = e["bboxes"]
            if ax == "x":
                gt_c.append((x + w / 2.0) / W)
            else:
                hh = w * float(th) / float(tw)
                gt_c.append((y + hh / 2.0) / H)
        if gt_c:
            mae_c.append(abs(np.mean(gt_c) - 0.5))
            mae_med.append(abs(np.median(gt_c) - 0.5))

    print("\n--- YOLO subject detectability ---")
    print("  sampled frames            : %d" % total_sampled)
    print("  with >=1 detection        : %d (%.1f%%)"
          % (detected, 100.0 * detected / max(1, total_sampled)))
    print("  GT frames covered by a det: %d/%d (%.1f%%)  [sampled GT frames only]"
          % (n_gt_frames_hit_by_det, n_gt_frames,
             100.0 * n_gt_frames_hit_by_det / max(1, n_gt_frames)))
    if auc_pos:
        print("  Role C temporal AUC       : %.4f  (chance 0.5000)"
              % auc(np.array(auc_pos), np.array(auc_neg)))
    if mae_c:
        print("  |GT centre - 0.5| mean    : %.4f  (median %.4f)"
              % (np.mean(mae_c), np.mean(mae_med)))

    print("\n--- val F1 (frames fixed at 0..n-round(%.2f*n)-1, 57 videos) ---" % args.drop_tail)
    base = None
    for name, preds in (("control  centred box (== shipped)", ctrl),
                        ("Role A   constant YOLO offset", roleA),
                        ("Role B   per-frame YOLO track", roleB)):
        f1, rows, mi = score(gt_by_vid, preds, meta)
        if base is None:
            base = f1
        print("  %-36s %.4f   (delta %+.4f, mean IoU %.4f)"
              % (name, f1, f1 - base, mi))

    # alpha sweep: does a PARTIAL pull toward the YOLO subject do better than
    # the full track (role B) or none at all (control)?
    print("\n--- Role B alpha sweep (0 = centred box, 1 = full track) ---")
    f1c, rows_c, _ = score(gt_by_vid, ctrl, meta)
    dC = {r["vid"]: r["F1"] for r in rows_c}
    print("  %-30s %.4f" % ("alpha 0.00 (control)", f1c))
    for a in sorted(sweep):
        f1, rows_a, mi = score(gt_by_vid, sweep[a], meta)
        dA = {r["vid"]: r["F1"] for r in rows_a}
        dl = [dA[v] - dC[v] for v in dA if dC[v] is not None and dA[v] is not None]
        dl = np.array(dl)
        win, loss = int((dl > 1e-9).sum()), int((dl < -1e-9).sum())
        se = dl.std(ddof=1) / max(1, np.sqrt(len(dl)))
        print("  %-30s %.4f   (delta %+.4f +/-%.4f se, IoU %.4f, W/L %d/%d)"
              % ("alpha %.2f" % a, f1, f1 - f1c, se, mi, win, loss))

    # does YOLO help on the videos whose GT is genuinely off-centre?
    _, rowsB, _ = score(gt_by_vid, roleB, meta)
    dB = {r["vid"]: r for r in rowsB}
    dC = {r["vid"]: r for r in rows_c}
    offs = []
    for vid in gt_by_vid:
        gt = gt_by_vid[vid]
        W = meta[vid]["W"]; H = meta[vid]["H"]
        tw, th = gt["targetRatioWH"]
        ax = free_axis(W, H, tw, th)
        cs = []
        for e in gt["predictions"]:
            x, y, w = e["bboxes"]
            cs.append(((x + w / 2.0) / W) if ax == "x"
                      else ((y + (w * float(th) / float(tw)) / 2.0) / H))
        if cs:
            offs.append((vid, abs(float(np.mean(cs)) - 0.5)))
    offs.sort(key=lambda t: -t[1])
    if offs:
        gto = np.array([o for _, o in offs])

        def _ymed(v):
            """Per-video median YOLO centre, over frames WITH a detection only.
            Including the 0.5 fallback for undetected frames would dilute the
            median and report a weaker correlation than the detector's real
            per-video bias -- keep this identical to --diagnose."""
            _, ce, sc, _ = centres_from_cache(cache[v], free_axis(
                meta[v]["W"], meta[v]["H"], *gt_by_vid[v]["targetRatioWH"]), False)
            ok = sc > 0
            return float(np.median(ce[ok])) if ok.any() else 0.5

        ysg = np.array([_ymed(v) for v, _ in offs])
        gsg = np.array([float(np.median([
            ((e["bboxes"][0] + e["bboxes"][2] / 2.0) / meta[v]["W"])
            if free_axis(meta[v]["W"], meta[v]["H"],
                         *gt_by_vid[v]["targetRatioWH"]) == "x" else
            ((e["bboxes"][1] + (e["bboxes"][2] * float(gt_by_vid[v]["targetRatioWH"][1])
                                / float(gt_by_vid[v]["targetRatioWH"][0])) / 2.0)
             / meta[v]["H"])
            for e in gt_by_vid[v]["predictions"]])) for v, _ in offs])
        ymo = np.abs(ysg - 0.5)
        dl = np.array([dB[v]["F1"] - dC[v]["F1"] if
                       (dB[v]["F1"] is not None and dC[v]["F1"] is not None) else 0.0
                       for v, _ in offs])
        print("  corr(|yolo_centre-0.5|,|gt-0.5|) = %+.3f  <- MISLEADING: abs drops the sign"
              % np.corrcoef(ymo, gto)[0, 1])
        print("  corr(signed yolo_centre, signed gt_centre) = %+.3f  <- the real test"
              % np.corrcoef(ysg, gsg)[0, 1])
        print("  corr(|yolo_centre-0.5|, delta)   = %+.3f  <- gating signal for Role B"
              % np.corrcoef(ymo, dl)[0, 1])
        print("  corr(|gt_centre-0.5|, delta)     = %+.3f  (look-ahead, upper bound only)"
              % np.corrcoef(gto, dl)[0, 1])
        print("  run --diagnose for bias/jitter split, optimal shrink and bootstrap CI")
    for frac, label in ((0.25, "most off-centre 25%"), (0.50, "most off-centre 50%")):
        k = max(1, int(round(frac * len(offs))))
        sel = [v for v, _ in offs[:k]]
        bc = float(np.mean([dC[v]["F1"] for v in sel if dC[v]["F1"] is not None]))
        bb = float(np.mean([dB[v]["F1"] for v in sel if dB[v]["F1"] is not None]))
        print("  %-28s control %.4f  vs  Role B %.4f  (%+.4f, n=%d)"
              % (label, bc, bb, bb - bc, k))

    if args.dump:
        for name, preds in (("ctrl", ctrl), ("roleA", roleA), ("roleB", roleB)):
            with open(os.path.join(HERE, "yolo_preds_%s.jsonl" % name), "w",
                      encoding="utf-8") as f:
                for vid in preds:
                    f.write(json.dumps({"video_id": vid,
                                        "targetRatioWH": gt_by_vid[vid]["targetRatioWH"],
                                        "predictions": preds[vid]},
                                       ensure_ascii=False) + "\n")
    return 0


def diagnose(args):
    """WHY the detector loses: decompose its centre displacement into a usable
    per-video bias and unusable frame-to-frame jitter, then shrink optimally.

    This is the correction of an earlier, too-quick reading. `|yolo-0.5|` vs
    `|gt-0.5|` correlates only +0.09, which looks like "the detector knows
    nothing". That test is WRONG: taking absolute values throws away the sign,
    which is precisely the information. The SIGNED correlation between the
    per-video median YOLO centre and the per-video median GT centre is +0.49.
    So the detector is not uninformed -- it is weak, and it is buried under its
    own jitter. That distinction decides how to use it, and it is also the
    reason an unshrunk offset (baseline_cv --alpha 1) loses.
    """
    cache = json.load(open(args.cache, "r", encoding="utf-8"))
    meta = json.load(open(args.metadata, "r", encoding="utf-8"))
    gt = {r["video_id"]: r for r in
          (json.loads(l) for l in open(args.gt, "r", encoding="utf-8") if l.strip())}

    D, within, cls_count, cls_switch = {}, [], {}, []
    for vid, g in gt.items():
        if vid not in cache:
            continue
        m = meta[vid]
        W, H = m["W"], m["H"]
        tw, th = g["targetRatioWH"]
        ax = free_axis(W, H, tw, th)
        fidx, ce, sc, nm = centres_from_cache(cache[vid], ax, False)
        ok = sc > 0
        if ok.sum() >= 3:
            D[vid] = {"y_med": float(np.median(ce[ok])),
                      "y_sd": float((ce[ok] - ce[ok].mean()).std()),
                      "W": W, "H": H, "n": int(m["n_frames"]),
                      "ax": ax}
            within.append(D[vid]["y_sd"])
            names = [nm[i] for i in range(len(nm)) if ok[i]]
            cls_switch += [names[i] != names[i - 1] for i in range(1, len(names))]
            for x in names:
                cls_count[x] = cls_count.get(x, 0) + 1
        cs = []
        for e in g["predictions"]:
            x, y, w = e["bboxes"]
            cs.append(((x + w / 2.0) / W) if ax == "x"
                      else ((y + (w * float(th) / float(tw)) / 2.0) / H))
        if cs and vid in D:
            D[vid]["mu_gt"] = float(np.median(cs))
            D[vid]["_tw"], D[vid]["_th"] = tw, th
    D = {v: d for v, d in D.items() if "mu_gt" in d}
    vids = sorted(D)
    g = np.array([D[v]["mu_gt"] for v in vids])
    y = np.array([D[v]["y_med"] for v in vids])
    # between-video component: the part a constant offset could use
    b_sd = float(np.std(y))
    w_sd = float(np.mean(within)) if within else 0.0
    cov = np.cov(g, y, ddof=1)
    beta = float(cov[0, 1] / cov[1, 1])
    err0, errb = float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[0, 0] - cov[0, 1] ** 2 / cov[1, 1]))

    print("--- displacement decomposition (%d videos) ---" % len(vids))
    print("  YOLO centre: between-video bias sd %.4f, within-video jitter sd %.4f"
          % (b_sd, w_sd))
    print("  -> a per-video CONSTANT can only use the bias; jitter is pure loss")
    print("  YOLO picks: %s" % ", ".join("%s %d" % (k, v) for k, v
          in sorted(cls_count.items(), key=lambda t: -t[1])[:5]))
    print("  picked subject CHANGES between consecutive sampled frames: %.1f%%"
          % (100 * np.mean(cls_switch) if cls_switch else 0.0))
    print("--- per-video bias, signed ---")
    print("  corr(YOLO median centre, GT median centre) = %+.3f" % np.corrcoef(g, y)[0, 1])
    print("  |devi| vs |devi|  = %+.3f   <- the misleading test (drops the sign)"
          % np.corrcoef(np.abs(y - 0.5), np.abs(g - 0.5))[0, 1])
    print("  optimal shrink beta* = %.3f   error sd %.4f -> %.4f (-%.0f%%)"
          % (beta, err0, errb, 100 * (1 - errb / err0)))

    def preds_with(centre_fn):
        out = {}
        for vid, d in D.items():
            cw, ch = crop_size(d["W"], d["H"], d["_tw"], d["_th"])
            f0, f1 = window(d["n"], args.drop_tail)
            out[vid] = boxes_for({"n_frames": d["n"], "_drop_tail": args.drop_tail},
                                 d["ax"], cw, ch, d["W"], d["H"],
                                 np.full(f1 - f0, centre_fn(vid, d)))
        return out

    def f1_per(preds):
        vals = []
        for vid, p in preds.items():
            d = D[vid]
            gm = {}
            for e in gt[vid]["predictions"]:
                fi = int(e["frame"])
                if fi not in gm:
                    gm[fi] = box_from_triplet(*e["bboxes"], d["_tw"], d["_th"])
            S = 0.0
            for e in p:
                fi = int(e["frame"])
                if fi in gm:
                    S += iou_xyxy(box_from_triplet(*e["bboxes"], d["_tw"], d["_th"]), gm[fi])
            den = len(p) + len(gm)
            vals.append(2 * S / den if den else 0.0)
        return np.array(vals)

    def beta_loo(vid, d):
        o = [u for u in vids if u != vid]
        c = np.cov([D[u]["mu_gt"] for u in o], [D[u]["y_med"] for u in o], ddof=1)
        return 0.5 + (c[0, 1] / c[1, 1]) * (d["y_med"] - 0.5)

    print("\n--- constant per-video offset (bias only, no jitter) ---")
    print("  (control here is lower than the headline 0.3688: this mode drops")
    print("   videos with <3 detections, so only the DELTAS below are comparable)")
    arms = [("centred (control)", lambda v, d: 0.5),
            ("yolo median, alpha=1 (no shrink)", lambda v, d: d["y_med"]),
            ("yolo median, beta*=%.2f" % beta, lambda v, d: 0.5 + beta * (d["y_med"] - 0.5)),
            ("yolo median, LOO-beta", beta_loo)]
    rng = np.random.default_rng(0)
    per = {}
    for name, fn in arms:
        p = f1_per(preds_with(fn))
        per[name] = p
        print("  %-34s %.4f" % (name, p.mean()))
    base = per["centred (control)"]
    print("\n--- paired bootstrap over videos (10000x) ---")
    for name, _ in arms[1:]:
        d = per[name] - base
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print("  %-34s delta %+.4f  95%%CI [%+.4f, %+.4f]  %s"
              % (name, d.mean(), lo, hi, "SIGNIFICANT" if (lo > 0 or hi < 0) else "spans 0"))
    return 0


def main():
    ap = argparse.ArgumentParser(description="YOLO: is it worth using?")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--index", default=os.path.join(HERE, "val_index.json"))
    ap.add_argument("--gt", default=os.path.join(HERE, "val_gt.jsonl"))
    ap.add_argument("--video-dir", default=os.path.join(HERE, "val_video"))
    ap.add_argument("--metadata", default=os.path.join(HERE, "val_metadata.json"))
    ap.add_argument("--cache", default=os.path.join(HERE, "yolo_val_cache.json"))
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--ema", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="partial pull for the alpha-sweep arm (role B)")
    ap.add_argument("--drop-tail", type=float, default=0.08)
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--diagnose", action="store_true",
                    help="why the detector loses: bias vs jitter, optimal shrink, "
                         "paired bootstrap CI")
    args = ap.parse_args()
    if args.build_cache:
        return build_cache(args)
    if args.diagnose:
        return diagnose(args)
    return evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
