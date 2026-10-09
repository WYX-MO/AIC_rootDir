#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""帧集选择器：把逐帧信号 -> 保留的帧号集合（提交帧集的子集）。

统一接口：  fn(sig, grid, **params) -> list[int]（绝对帧号，升序）

  sig   : signals.py 抽的单视频信号字典（audio/motion/person/ndet/conf 都是逐帧）
  grid  : 允许保留的候选帧（= base 预测里已有的帧号，一般是 0..n-round(0.08n)-1）

所有选择器只做"取子集"，不新增帧，因此框臂与 base 完全对齐。
"""
import numpy as np


# ----------------------------------------------------------------- 工具
def _z(a):
    a = np.asarray(a, dtype=np.float64)
    s = a.std()
    return (a - a.mean()) / s if s > 1e-9 else np.zeros_like(a)


def _zkey(sig, key, eps=1e-9):
    a = np.asarray(sig[key], dtype=np.float64)
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo + eps else np.zeros_like(a)


def combined_score(sig, keys=("audio", "motion", "person"),
                   weights=(1.0, 1.0, 1.0)):
    """把若干信号归一化后加权求和，得到逐帧分数（长度 = n_frames）。"""
    sc = np.zeros(len(sig["audio"]), dtype=np.float64)
    for k, w in zip(keys, weights):
        if k in sig and w:
            sc += float(w) * _z(_zkey(sig, k))
    return sc


def _grid_idx(grid):
    g = np.asarray(grid, dtype=int)
    return g


def _top_mask(scores_on_grid, frac):
    """在 grid 上按分数取前 frac 比例（并列用帧号小者优先）。"""
    n = len(scores_on_grid)
    k = max(1, int(round(frac * n)))
    order = np.lexsort((np.arange(n), -scores_on_grid))  # 分数降序，帧号次之
    keep = np.zeros(n, dtype=bool)
    keep[order[:k]] = True
    return keep


# ----------------------------------------------------------------- 选择器
def keep_all(sig, grid, **kw):
    """现行基线：candidate grid 全保留（grid 本身已含"丢末 8%"，见 run 侧）。"""
    return list(grid)


def drop_tail(sig, grid, frac=0.08, **kw):
    g = np.asarray(grid, dtype=int)
    n = int(sig["n_frames"])
    cut = int(round((1.0 - frac) * n))
    return sorted(int(f) for f in g if f < cut)


def top_frac(sig, grid, key="audio", frac=0.6, score=None, **kw):
    """保留分数最高的 frac 比例帧（可散点，不要求连续）。"""
    g = _grid_idx(grid)
    sc = combined_score(sig) if score is None else np.asarray(score, dtype=np.float64)
    s = sc[g]
    keep = _top_mask(s, frac)
    return sorted(int(f) for f in g[keep])


def best_window(sig, grid, key="audio", frac=0.6, score=None, **kw):
    """保留"最佳连续窗"：长度 = 舍入(frac*n)，使窗内分数和最大。"""
    g = _grid_idx(grid)
    if len(g) == 0:
        return []
    sc = combined_score(sig) if score is None else np.asarray(score, dtype=np.float64)
    s = sc[g]
    n = len(g)
    L = max(1, int(round(frac * n)))
    if L >= n:
        return sorted(int(f) for f in g)
    c = np.concatenate([[0.0], np.cumsum(s)])
    win = c[L:] - c[:-L]
    j = int(np.argmax(win))
    return sorted(int(f) for f in g[j:j + L])


def multi_window(sig, grid, key="audio", frac=0.6, k=2, merge_gap=15,
                 score=None, **kw):
    """分数阈值 + 合并成至多 k 段连续窗，总长≈frac*n（阈值二分逼近）。"""
    g = _grid_idx(grid)
    if len(g) == 0:
        return []
    sc = combined_score(sig) if score is None else np.asarray(score, dtype=np.float64)
    s = sc[g]
    n = len(g)
    target = max(1, int(round(frac * n)))

    def runs_at(th):
        m = s >= th
        runs = []
        i = 0
        while i < n:
            if m[i]:
                j = i
                while j + 1 < n and m[j + 1]:
                    j += 1
                runs.append((i, j)); i = j + 1
            else:
                i += 1
        # 合并间隔 < merge_gap 的段
        merged = []
        for a, b in runs:
            if merged and (g[a] - g[merged[-1][1]]) <= merge_gap:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        lengths = sorted(((g[b] - g[a] + 1), a, b) for a, b in merged)
        lengths.reverse()
        keep = []
        for _, a, b in lengths[:k]:
            keep.extend(g[a:b + 1])
        return sorted(set(int(f) for f in keep)), merged

    lo, hi = float(s.min()), float(s.max())
    best = None
    for _ in range(40):
        th = 0.5 * (lo + hi)
        keep, _ = runs_at(th)
        if best is None or abs(len(keep) - target) < abs(len(best) - target):
            best = keep
        if len(keep) > target:
            lo = th
        else:
            hi = th
    return best if best is not None else sorted(int(f) for f in g)


def position_window(sig, grid, lo=0.16, hi=0.88, **kw):
    """纯位置先验：只保留 [lo*n, hi*n) 区间内的候选帧（无内容信号）。"""
    g = np.asarray(grid, dtype=int)
    n = int(sig["n_frames"])
    a, b = lo * n, hi * n
    return sorted(int(f) for f in g if a <= f < b)


def interval(sig, grid, iv=None, widen=1.0, **kw):
    """保留外部给定的帧区间 [f0, f1)（如 VLM 截取的区间）。

    iv=None（该视频没有区间）-> 退回整格（=keep_all），保证不劣于基线。
    widen：以区间中点为中心按比例放宽（>1 变宽，<1 变窄）。
    """
    g = np.asarray(grid, dtype=int)
    if not iv:
        return sorted(int(f) for f in g)
    f0, f1 = float(iv[0]), float(iv[1])
    if f1 <= f0:
        return sorted(int(f) for f in g)
    c = 0.5 * (f0 + f1)
    half = 0.5 * (f1 - f0) * float(widen)
    a, b = c - half, c + half
    out = [int(f) for f in g if a <= f < b]
    return out if out else sorted(int(f) for f in g)


REGISTRY = {
    "keep_all": keep_all,
    "drop_tail": drop_tail,
    "top_frac": top_frac,
    "best_window": best_window,
    "multi_window": multi_window,
    "position_window": position_window,
    "interval": interval,
}
