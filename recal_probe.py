#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Is the framing bias a FUNCTION of things we can already see?

gate_probe measured the pool's systematic bias at 0.0606 (mean |bias| 0.0697)
and priced its removal at +0.028 online. The obvious question is whether that
bias is a constant we can just subtract, or a predictable function of the
scene -- either way it is free (no training, no GPU).

This fits, leave-one-out,   GT_med ~ [1, pool_med, |pool_med-.5|, ndet, spread]
on the 57 val videos whose YOLO detections are already in yolo_val_cache.json,
then re-scores F1 per video with the fitted centre. Three references:

    pool            0.4214  (shipped arm, no recalibration)
    oracle de-bias  0.5176  (centre := GT_med, knows the answer)
    regression      ?       (must beat 0.4214 to be worth anything)

If the regression lands near 0.4214, the bias is NOT a function of observables
and a trained estimator is the only route; if it lands near 0.5176, we collect
the whole +0.028 for free.

Usage
    python recal_probe.py
"""

import os
import sys
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import select_probe as SP                                          # noqa: E402


def features(d):
    """per-video observables: the pool centre and how the detections sit."""
    mus, nd, sp = [], [], []
    for fi, dd in zip(d["frames"], d["dets"]):
        ps = [x for x in dd if x["cls"] == "person"]
        v = SP.pick(ps, d["ax"], "pool", None)
        if v is None:
            continue
        mus.append(v)
        nd.append(len(ps))
        c = np.array([x["cx"] if d["ax"] == "x" else x["cy"] for x in ps])
        if len(c) > 1:
            sp.append(float((c.max() - c.min()) / 2.0))
    mus = np.array(mus, dtype=np.float64)
    gts = np.array(list(d["gctr"].values()), dtype=np.float64)
    return {"mu": float(np.median(mus)) if len(mus) else 0.5,
            "sd": float(mus.std()) if len(mus) else 0.0,
            "nd": float(np.mean(nd)) if nd else 0.0,
            "sp": float(np.mean(sp)) if sp else 0.0,
            "gt": float(np.median(gts)) if len(gts) else 0.5}


def design(F, names):
    return np.array([[{"mu": f["mu"], "dm": abs(f["mu"] - 0.5),
                       "nd": np.log1p(f["nd"]), "sp": f["sp"], "sd": f["sd"],
                       "one": 1.0}[n] for n in names] for f in F])


def f1_const(d, c, drop_tail):
    a, b = SP.YE.window(d["n"], drop_tail)
    return SP.f1(d, np.full(b - a, float(c)), drop_tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=os.path.join(HERE, "val_gt.jsonl"))
    ap.add_argument("--metadata", default=os.path.join(HERE, "val_metadata.json"))
    ap.add_argument("--cache", default=os.path.join(HERE, "yolo_val_cache.json"))
    ap.add_argument("--drop-tail", type=float, default=0.08)
    a = ap.parse_args()

    rows = SP.load(a)
    for d in rows:
        d["f"] = features(d)
    print("videos %d" % len(rows))

    F = [d["f"] for d in rows]
    y = np.array([f["gt"] for f in F])
    pool = np.array([f["mu"] for f in F])
    print("  gt_med      mean %.4f sd %.4f" % (y.mean(), y.std()))
    print("  pool_med    mean %.4f sd %.4f" % (pool.mean(), pool.std()))
    print("  bias (pool-gt) mean %+.4f  mean|.| %.4f  sd %.4f"
          % ((pool - y).mean(), np.abs(pool - y).mean(), (pool - y).std()))
    print("  corr(pool,gt) = %+.3f" % np.corrcoef(pool, y)[0, 1])
    print()

    base = float(np.mean([f1_const(d, d["f"]["mu"], a.drop_tail) for d in rows]))
    oracle = float(np.mean([f1_const(d, d["f"]["gt"], a.drop_tail) for d in rows]))
    zero = float(np.mean([f1_const(d, 0.5, a.drop_tail) for d in rows]))
    print("  references: centred %.4f   pool %.4f   oracle de-bias %.4f"
          % (zero, base, oracle))
    print()

    models = [
        ("const (subtract mean bias)", ["one"]),
        ("+pool", ["one", "mu"]),
        ("+pool +|d|", ["one", "mu", "dm"]),
        ("+pool +nd +sp", ["one", "mu", "nd", "sp"]),
        ("all", ["one", "mu", "dm", "nd", "sp", "sd"]),
    ]
    print("  %-28s %8s %8s %10s %10s" % ("model (LOO)", "R2", "rmse", "F1", "d pool"))
    for name, names in models:
        X = design(F, names)
        pred = np.zeros(len(F))
        for i in range(len(F)):
            m = np.ones(len(F), dtype=bool)
            m[i] = False
            w, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
            pred[i] = float(X[i] @ w)
        w, *_ = np.linalg.lstsq(X, y, rcond=None)
        r2 = 1.0 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
        rmse = float(np.sqrt(((y - pred) ** 2).mean()))
        f1 = float(np.mean([f1_const(d, p, a.drop_tail)
                            for d, p in zip(rows, np.clip(pred, 0.0, 1.0))]))
        print("  %-28s %8.3f %8.4f %10.4f %+10.4f" % (name, r2, rmse, f1, f1 - base))
    print()
    print("  how much of the 0.0606 bias survives a LOO recalibration?     ")
    X = np.array([[f["mu"], 1.0] for f in F])
    pred = np.zeros(len(F))
    for i in range(len(F)):
        m = np.ones(len(F), dtype=bool)
        m[i] = False
        w, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
        pred[i] = float(X[i] @ w)
    print("     residual |bias| mean %.4f  (raw %.4f) -> credit %.0f%%"
          % (np.abs(pred - y).mean(), np.abs(pool - y).mean(),
             100.0 * (1.0 - np.abs(pred - y).mean() / np.abs(pool - y).mean())))
    print("     per-video gain: %d/%d videos improve, %d worse"
          % (sum(1 for d, p in zip(rows, pred)
                 if f1_const(d, p, a.drop_tail) > f1_const(d, d["f"]["mu"], a.drop_tail)),
             len(rows),
             sum(1 for d, p in zip(rows, pred)
                 if f1_const(d, p, a.drop_tail) < f1_const(d, d["f"]["mu"], a.drop_tail))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
