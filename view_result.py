#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""view_result.py -- 提交结果对比可视化（参考程序 / reference viewer）

用途
----
同一个视频，左右并排看：
  左 = 原视频（不画框）
  右 = 预测框(红, 虚线) + GT框(绿, 实线) 叠加
顶部显示该视频的 F1 / meanIoU(matched) / N_pred / N_gt，播放条显示当前帧 IoU。

数据来源（顶部「数据集」下拉切换）
  复赛test : fu_testset/video/  426 条，无 GT（右侧只画预测框，不计分）
  本地val  : val_video/          57 条，GT = val_gt.jsonl（可算分）

「结果文件」是提交格式的 jsonl：
  {"video_id":.., "targetRatioWH":[tw,th], "predictions":[{"frame":.., "bboxes":[x,y,w]}]}
分数口径与 eval_local.py 完全一致（同帧匹配、IoU 加权 F1）。

快捷键：空格 播放/暂停 · ←/→ 上一/下一帧 · ↑/↓ 上一/下一视频 · Home/End 首/末帧
"""
import os
import sys
import json
import bisect
import argparse
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_local as ev                                    # noqa: E402

# --------------------------------------------------------------------------
# 数据集预设：视频目录 / metadata / GT / 默认结果文件
# --------------------------------------------------------------------------
DATASETS = {
    "复赛test": dict(video_dir="fu_testset/video",
                     metadata="fu_metadata.json",
                     gt=None,
                     cache="fu_yolo_test_cache.json",
                     result="fu_predictions_gated.jsonl"),
    "本地val": dict(video_dir="val_video",
                    metadata="val_metadata.json",
                    gt="val_gt.jsonl",
                    cache="yolo_val_cache.json",
                    result="predictions_val_gated.jsonl"),
}

RED = (60, 60, 235)      # 预测框（BGR），虚线
GREEN = (70, 200, 70)    # GT 框（BGR），实线
ORANGE = (0, 145, 255)   # YOLO person 检测
GREY = (170, 170, 170)   # YOLO 其他类别检测
PANEL_H = 460            # 每侧显示高度（px）


# --------------------------------------------------------------------------
# 数据加载
# --------------------------------------------------------------------------
def _p(path):
    return path if os.path.isabs(path) else os.path.join(HERE, path)


def load_pred_map(path):
    """jsonl -> {vid: (ratio(tw,th), {frame:(x,y,w)})}. 首个合法帧胜出。"""
    recs, _ = ev.load_pred_jsonl(path)
    out = {}
    for vid, (ratio, preds, _ln, _sz) in recs.items():
        m = {}
        for e in preds:
            if not isinstance(e, dict):
                continue
            f = e.get("frame")
            if isinstance(f, bool) or not isinstance(f, (int, float)):
                continue
            f = int(f)
            if f < 0 or f in m:
                continue
            bb = e.get("bboxes")
            if not (isinstance(bb, (list, tuple)) and len(bb) == 3):
                continue
            try:
                m[f] = tuple(float(v) for v in bb)
            except (TypeError, ValueError):
                continue
        out[str(vid)] = (ratio, m)
    return out


def load_meta(path):
    if not path or not os.path.exists(_p(path)):
        return {}
    with open(_p(path), "r", encoding="utf-8") as f:
        return json.load(f)


def score_video(predm, gtm, tw, th):
    """与 eval_local 同口径。返回 dict。"""
    S = 0.0
    matched = 0
    for f, (x, y, w) in predm.items():
        g = gtm.get(f)
        if g is None:
            continue
        a = ev.box_from_triplet(x, y, w, tw, th)
        b = ev.box_from_triplet(g[0], g[1], g[2], tw, th)
        S += ev.iou_xyxy(a, b)
        matched += 1
    N_pred, N_gt = len(predm), len(gtm)
    f1 = (2.0 * S / (N_pred + N_gt)) if (N_pred + N_gt) > 0 else None
    return {"F1": f1, "S": S, "matched": matched, "N_pred": N_pred, "N_gt": N_gt,
            "mean_iou": (S / matched) if matched else 0.0}


# --------------------------------------------------------------------------
# 绘图
# --------------------------------------------------------------------------
def dashed_rect(img, x1, y1, x2, y2, color, thickness=2, dash=14):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    def seg_h(y):
        x = x1
        while x < x2:
            cv2.line(img, (x, y), (min(x + dash, x2), y), color, thickness)
            x += 2 * dash
    def seg_v(x):
        y = y1
        while y < y2:
            cv2.line(img, (x, y), (x, min(y + dash, y2)), color, thickness)
            y += 2 * dash
    seg_h(y1); seg_h(y2); seg_v(x1); seg_v(x2)


def yolo_box_of(W, H, d):
    """检测 -> 像素 xyxy。与 cover_probe.box_of 同约定：a=√area，盒=a·W × a·H。"""
    a = max(float(d.get("area", 0.0)), 0.0) ** 0.5
    x1 = (float(d["cx"]) - a / 2.0) * W
    y1 = (float(d["cy"]) - a / 2.0) * H
    return x1, y1, x1 + a * W, y1 + a * H


def draw_boxes(frame, pred, gt, tw, th, yolo=None):
    """在 BGR 帧上画 YOLO 检测(橙person/灰其他) + 预测(红虚线) + GT(绿实线)。"""
    h_img, w_img = frame.shape[:2]
    if yolo:
        for d in yolo:
            x1, y1, x2, y2 = yolo_box_of(w_img, h_img, d)
            is_p = d.get("cls") == "person"
            col = ORANGE if is_p else GREY
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), col,
                          2 if is_p else 1)
            cv2.putText(frame, "%s %.2f" % (d.get("cls", "?"), float(d.get("conf", 0))),
                        (int(x1), max(12, int(y1) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    if gt is not None:
        x, y, w = gt
        h = w * th / tw
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), GREEN, 3)
        cv2.putText(frame, "GT", (int(x) + 4, int(y) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2, cv2.LINE_AA)
    if pred is not None:
        x, y, w = pred
        h = w * th / tw
        dashed_rect(frame, x, y, x + w, y + h, RED, 2)
        cv2.putText(frame, "PRED", (int(x) + 4, int(y) + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, cv2.LINE_AA)
    return frame


# --------------------------------------------------------------------------
# 音频（cv2 只解视频帧、不带音轨，所以用 ffplay 单独播音轨）
# --------------------------------------------------------------------------
class AudioPlayer:
    """按帧号换算成秒，用 ffplay 播该视频的音轨；换视频/暂停/重定位即时停。"""

    def __init__(self):
        self.proc = None
        self.path = None
        self.fps = 25.0
        self.enabled = bool(self._which("ffplay"))

    @staticmethod
    def _which(exe):
        for d in os.environ.get("PATH", "").split(os.pathsep):
            if d and os.path.exists(os.path.join(d, exe)):
                return True
        return False

    def set_video(self, path, fps):
        self.stop()
        self.path = path
        self.fps = fps or 25.0

    def start(self, frame):
        self.stop()
        if not (self.enabled and self.path and os.path.exists(self.path)):
            return
        t = max(0.0, float(frame) / self.fps)
        try:
            self.proc = subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
                 "-ss", "%.3f" % t, "-i", self.path],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            self.proc = None

    def stop(self):
        if self.proc:
            try:
                self.proc.terminate()
            except Exception:
                pass
            self.proc = None


def audio_envelope(path, n_frames, fps, sr=16000):
    """逐帧音量包络（RMS，归一化到 0..1）。用 ffmpeg 解一段单声道 PCM。

    返回长度 n_frames 的 np.float32，或 None（无音轨/失败）。
    """
    try:
        out = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-f", "s16le",
             "-ac", "1", "-ar", str(sr), "-"],
            capture_output=True, timeout=120)
        if out.returncode != 0 or not out.stdout:
            return None
        a = np.frombuffer(out.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception:
        return None
    if a.size == 0:
        return None
    spf = float(sr) / float(fps or 25.0)
    env = np.zeros(max(1, n_frames), dtype=np.float32)
    for i in range(len(env)):
        s, e = int(i * spf), int((i + 1) * spf)
        seg = a[s:e]
        if seg.size:
            env[i] = float(np.sqrt((seg.astype(np.float64) ** 2).mean()))
    m = float(env.max())
    if m > 0:
        env /= m
    return env


# --------------------------------------------------------------------------
# 应用
# --------------------------------------------------------------------------
class Viewer:
    def __init__(self, root):
        self.root = root
        root.title("AIC 结果对比 viewer")

        self.video_dir = None
        self.meta = {}
        self.gt = {}                 # {vid: (ratio, {frame:(x,y,w)})}
        self.pred = {}               # {vid: (ratio, {frame:(x,y,w)})}
        self.vids = []
        self.cur = 0                 # 当前视频在 self.vids 中的下标
        self.cap = None
        self.n_frames = 0
        self.fps = 25.0
        self.W = self.H = 0
        self.cur_frame = 0
        self.playing = False
        self.after_id = None
        self._cache = {}
        self.cache = {}          # YOLO 检测缓存 {vid: {"frames":..,"dets":..}}
        self.ymap = {}           # 当前视频 {frame: [det,...]}
        self.yframes = []        # 当前视频的抽样帧（升序）
        self.audio = AudioPlayer()
        self.env = None          # 当前视频的逐帧音量包络
        self.env_cache = {}      # {vid: env}

        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        top = ttk.Frame(self.root, padding=6)
        top.pack(fill="x")

        ttk.Label(top, text="结果文件").grid(row=0, column=0, sticky="e")
        self.e_result = ttk.Entry(top, width=46)
        self.e_result.grid(row=0, column=1, columnspan=3, sticky="we", padx=4)
        ttk.Button(top, text="浏览…", command=self._browse_result).grid(row=0, column=4)
        ttk.Button(top, text="载入", command=self.reload).grid(row=0, column=5)

        ttk.Label(top, text="GT文件").grid(row=1, column=0, sticky="e")
        self.e_gt = ttk.Entry(top, width=46)
        self.e_gt.grid(row=1, column=1, columnspan=3, sticky="we", padx=4)
        ttk.Button(top, text="浏览…", command=self._browse_gt).grid(row=1, column=4)

        ttk.Label(top, text="数据集").grid(row=1, column=5, sticky="e")
        self.cb_ds = ttk.Combobox(top, values=list(DATASETS), width=12, state="readonly")
        self.cb_ds.set("复赛test")
        self.cb_ds.grid(row=1, column=6, padx=(4, 0))
        self.cb_ds.bind("<<ComboboxSelected>>", lambda e: self._apply_dataset())

        sel = ttk.Frame(self.root, padding=(6, 0))
        sel.pack(fill="x")
        ttk.Button(sel, text="◀ 上一视频", command=lambda: self.step_video(-1)).pack(side="left")
        self.cb_vid = ttk.Combobox(sel, width=40, state="readonly")
        self.cb_vid.pack(side="left", padx=6)
        self.cb_vid.bind("<<ComboboxSelected>>", lambda e: self.select_video(self.cb_vid.current()))
        ttk.Button(sel, text="下一视频 ▶", command=lambda: self.step_video(1)).pack(side="left")
        self.lbl_score = ttk.Label(sel, text="", font=("TkDefaultFont", 10, "bold"))
        self.lbl_score.pack(side="left", padx=12)
        self.var_yolo = tk.BooleanVar(value=True)
        ttk.Checkbutton(sel, text="显示YOLO检测(橙=person)",
                        variable=self.var_yolo, command=self._redraw).pack(side="left", padx=6)
        self.audio_on = tk.BooleanVar(value=self.audio.enabled)
        ttk.Checkbutton(sel, text="🔊 声音", variable=self.audio_on,
                        command=self._on_audio_toggle).pack(side="left", padx=6)
        self.lbl_file = ttk.Label(sel, text="", foreground="#666")
        self.lbl_file.pack(side="right", padx=8)

        mid = ttk.Frame(self.root, padding=6)
        mid.pack()
        self.lbl_left = tk.Label(mid, bg="#111", width=PANEL_H, height=PANEL_H)
        self.lbl_left.grid(row=0, column=0, padx=4)
        self.lbl_right = tk.Label(mid, bg="#111", width=PANEL_H, height=PANEL_H)
        self.lbl_right.grid(row=0, column=1, padx=4)
        ttk.Label(mid, text="原视频（无框）").grid(row=1, column=0, pady=(2, 0))
        ttk.Label(mid, text="预测(红虚线) + GT(绿实线)").grid(row=1, column=1, pady=(2, 0))

        wf = ttk.Frame(self.root, padding=(6, 0))
        wf.pack(fill="x")
        ttk.Label(wf, text="音量包络(每帧RMS)").pack(side="left")
        self.wave = tk.Canvas(wf, height=72, bg="#f7f7f7", highlightthickness=1,
                              highlightbackground="#ccc")
        self.wave.pack(side="left", fill="x", expand=True, padx=(6, 0))
        self.wave.bind("<Configure>", lambda e: self._draw_wave())
        self.wave.bind("<Button-1>", self._on_wave_click)
        self.wave.bind("<B1-Motion>", self._on_wave_click)

        bot = ttk.Frame(self.root, padding=6)
        bot.pack(fill="x")
        self.btn_play = ttk.Button(bot, text="▶ 播放", width=8, command=self.toggle_play)
        self.btn_play.pack(side="left")
        ttk.Button(bot, text="⏮", width=3, command=lambda: self.seek(0)).pack(side="left")
        ttk.Button(bot, text="◀", width=3, command=lambda: self.seek(self.cur_frame - 1)).pack(side="left")
        ttk.Button(bot, text="▶", width=3, command=lambda: self.seek(self.cur_frame + 1)).pack(side="left")
        ttk.Button(bot, text="⏭", width=3,
                   command=lambda: self.seek(self.n_frames - 1)).pack(side="left")
        self.scale = ttk.Scale(bot, from_=0, to=1, orient="horizontal", command=self._on_scale)
        self.scale.pack(side="left", fill="x", expand=True, padx=8)
        self.lbl_frame = ttk.Label(bot, text="—", width=24)
        self.lbl_frame.pack(side="left")

        top.columnconfigure(1, weight=1)

        self.root.bind("<space>", lambda e: self.toggle_play())
        self.root.bind("<Left>", lambda e: self.seek(self.cur_frame - 1))
        self.root.bind("<Right>", lambda e: self.seek(self.cur_frame + 1))
        self.root.bind("<Up>", lambda e: self.step_video(-1))
        self.root.bind("<Down>", lambda e: self.step_video(1))
        self.root.bind("<Home>", lambda e: self.seek(0))
        self.root.bind("<End>", lambda e: self.seek(self.n_frames - 1))
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._apply_dataset()
        self.reload()

    def _browse_result(self):
        p = filedialog.askopenfilename(initialdir=HERE, title="选择结果 jsonl",
                                       filetypes=[("jsonl/json", "*.jsonl *.json"), ("all", "*")])
        if p:
            self.e_result.delete(0, "end"); self.e_result.insert(0, p)

    def _browse_gt(self):
        p = filedialog.askopenfilename(initialdir=HERE, title="选择 GT jsonl",
                                       filetypes=[("jsonl/json", "*.jsonl *.json"), ("all", "*")])
        if p:
            self.e_gt.delete(0, "end"); self.e_gt.insert(0, p)

    def _apply_dataset(self):
        ds = DATASETS[self.cb_ds.get()]
        self.video_dir = _p(ds["video_dir"])
        self.meta_path = ds["metadata"]
        self.cache_path = _p(ds["cache"])
        self.e_result.delete(0, "end"); self.e_result.insert(0, _p(ds["result"]))
        self.e_gt.delete(0, "end"); self.e_gt.insert(0, _p(ds["gt"]) if ds["gt"] else "")

    # ---------------- 载入 ----------------
    def reload(self):
        rpath = self.e_result.get().strip()
        if not rpath or not os.path.exists(_p(rpath)):
            messagebox.showerror("载入失败", "结果文件不存在：\n%s" % rpath)
            return
        try:
            self.pred = load_pred_map(_p(rpath))
        except Exception as e:
            messagebox.showerror("载入失败", "解析结果文件出错：%s" % e)
            return
        self.meta = load_meta(self.meta_path)

        gpath = self.e_gt.get().strip()
        self.gt = load_pred_map(_p(gpath)) if (gpath and os.path.exists(_p(gpath))) else {}

        # YOLO 检测缓存（画出检测框与 conf；抽样帧来自 baseline 的 yolo cache）
        self.cache = {}
        cp = getattr(self, "cache_path", None)
        if cp and os.path.exists(cp):
            try:
                with open(cp, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}

        # 只保留有视频文件的条目，按 video_id / 序号排序
        all_ids = list(self.pred.keys())
        self.vids = [v for v in all_ids if os.path.exists(os.path.join(self.video_dir, v + ".mp4"))]
        self.vids.sort(key=lambda s: (int(s) if s.isdigit() else 10**9, s))
        miss = len(all_ids) - len(self.vids)

        # 全文件平均 F1（仅对有 GT 的视频）
        f1s = []
        for v in self.vids:
            if v not in self.gt:
                continue
            tw, th = self._ratio(v)
            r = score_video(self.pred[v][1], self.gt[v][1], tw, th)
            if r["F1"] is not None:
                f1s.append(r["F1"])
        mean_f1 = (sum(f1s) / len(f1s)) if f1s else None
        gt_tag = ("无 GT" if not self.gt else
                  "有GT %d/%d" % (sum(1 for v in self.vids if v in self.gt), len(self.vids)))
        note = "  (缺视频文件 %d 条)" % miss if miss else ""
        self.lbl_file.config(text="文件均分 F1 = %s  |  %s%s"
                             % (("%.4f" % mean_f1) if mean_f1 is not None else "—", gt_tag, note))

        self.cb_vid["values"] = ["%3d  %s" % (i, v) for i, v in enumerate(self.vids)]
        if self.vids:
            self.cb_vid.current(0)
            self.select_video(0)

    def _ratio(self, v):
        rec = self.pred.get(v) or self.gt.get(v)
        tw, th = rec[0] if rec else (16.0, 9.0)
        return float(tw), float(th)

    # ---------------- 视频 ----------------
    def select_video(self, idx):
        if not self.vids:
            return
        idx = max(0, min(idx, len(self.vids) - 1))
        self.cur = idx
        self.cb_vid.current(idx)
        vid = self.vids[idx]
        path = os.path.join(self.video_dir, vid + ".mp4")

        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        m = self.meta.get(vid, {})
        self.W = int(m.get("W") or self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.H = int(m.get("H") or self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.n_frames = int(m.get("n_frames") or self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.fps = float(m.get("fps") or self.cap.get(cv2.CAP_PROP_FPS) or 25.0) or 25.0
        self._cache.clear()
        self.scale.config(to=max(1, self.n_frames - 1))

        ys = self.cache.get(vid)
        self.ymap = {}
        if ys:
            self.ymap = {int(f): d for f, d in
                         zip(ys.get("frames", []), ys.get("dets", []))}
        self.yframes = sorted(self.ymap)
        self.audio.set_video(path, self.fps)
        if vid in self.env_cache:
            self.env = self.env_cache[vid]
        else:
            self.env = audio_envelope(path, self.n_frames, self.fps)
            self.env_cache[vid] = self.env
        self._draw_wave()

        # 该视频分数
        tw, th = self._ratio(vid)
        pm = self.pred.get(vid, (None, {}))[1]
        gm = self.gt.get(vid, (None, {}))[1]
        self.score = score_video(pm, gm, tw, th)
        s = self.score
        if gm:
            f1 = "%.4f" % s["F1"] if s["F1"] is not None else "undef"
            self.lbl_score.config(text="%s   F1=%s  meanIoU=%.3f  N_pred=%d  N_gt=%d  匹配=%d"
                                  % (vid, f1, s["mean_iou"], s["N_pred"], s["N_gt"], s["matched"]))
        else:
            self.lbl_score.config(text="%s   [无GT]  N_pred=%d" % (vid, s["N_pred"]))

        self.seek(0)

    def step_video(self, d):
        if self.vids:
            self.select_video(self.cur + d)

    def _frame(self, i):
        if i in self._cache:
            return self._cache[i]
        if self.cap is None:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, img = self.cap.read()
        if not ok:
            return None
        if i in self._cache:
            pass
        self._cache[i] = img
        if len(self._cache) > 240:
            self._cache.pop(next(iter(self._cache)))
        return img

    def render(self):
        i = self.cur_frame
        img = self._frame(i)
        if img is None:
            return
        vid = self.vids[self.cur]
        tw, th = self._ratio(vid)
        pm = self.pred.get(vid, (None, {}))[1]
        gm = self.gt.get(vid, (None, {}))[1]
        pr = pm.get(i)
        gt = gm.get(i)

        # YOLO 检测：取"最近一个抽样帧 ≤ 当前帧"，抽样帧本身用实值
        yolo, ysrc = None, None
        if self.var_yolo.get() and self.yframes:
            j = bisect.bisect_right(self.yframes, i) - 1
            if j >= 0:
                ysrc = self.yframes[j]
                yolo = self.ymap[ysrc]

        left = img.copy()
        right = draw_boxes(img.copy(), pr, gt, tw, th, yolo)

        # 当前帧 IoU
        cur_iou = "—"
        if pr is not None and gt is not None:
            cur_iou = "%.3f" % ev.iou_xyxy(ev.box_from_triplet(*pr, tw, th),
                                           ev.box_from_triplet(*gt, tw, th))
        elif gt is not None and pr is None:
            cur_iou = "漏(有GT无预测)"
        elif pr is not None and gt is None:
            cur_iou = "多(有预测无GT)"

        if yolo is not None:
            np_ = sum(1 for d in yolo if d.get("cls") == "person")
            ytag = "  YOLO:%dperson/%ddet@%d%s" % (np_, len(yolo), ysrc,
                                                   "" if ysrc == i else "(沿用)")
        else:
            ytag = "  YOLO:—"
        self.lbl_frame.config(text="帧 %d/%d  此帧IoU=%s%s"
                              % (i, self.n_frames - 1, cur_iou, ytag))

        self._show(left, self.lbl_left)
        self._show(right, self.lbl_right)
        self._wave_playhead()

    def _redraw(self):
        if self.vids:
            self.render()

    # ---------------- 音量包络 ----------------
    def _draw_wave(self):
        c = self.wave
        c.delete("bars")
        W = c.winfo_width()
        H = int(c["height"])
        if self.env is None or W < 4 or not len(self.env):
            c.delete("ph")
            return
        n = len(self.env)
        mid = H / 2.0
        for x in range(W):
            i0 = int(x / W * n)
            i1 = max(i0 + 1, int((x + 1) / W * n))
            seg = self.env[i0:min(i1, n)]
            v = float(seg.max()) if seg.size else 0.0
            h = v * (mid - 3)
            c.create_line(x, mid - h, x, mid + h, fill="#4a90d9", tags="bars")
        self._wave_playhead()

    def _wave_playhead(self):
        c = self.wave
        c.delete("ph")
        if self.env is None or not len(self.env):
            return
        W = c.winfo_width()
        H = int(c["height"])
        n = len(self.env)
        x = int(self.cur_frame / max(1, n - 1) * (W - 1))
        c.create_line(x, 0, x, H, fill="#e05555", width=2, tags="ph")

    def _on_wave_click(self, ev):
        if self.env is None or not len(self.env):
            return
        W = self.wave.winfo_width()
        self.seek(int(ev.x / max(1, W) * len(self.env)))

    def _show(self, bgr, label):
        h, w = bgr.shape[:2]
        s = PANEL_H / float(h)
        disp = cv2.resize(bgr, (max(1, int(w * s)), PANEL_H), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        im = ImageTk.PhotoImage(Image.fromarray(rgb))
        label.configure(image=im, width=im.width(), height=im.height())
        label.image = im

    # ---------------- 播放 ----------------
    def _set_frame(self, i):
        if not self.vids:
            return
        i = max(0, min(int(i), max(0, self.n_frames - 1)))
        self.cur_frame = i
        self.scale.set(i)
        self.render()

    def seek(self, i):
        self._set_frame(i)
        self._audio_restart()

    def _audio_restart(self):
        if self.playing and self.audio_on.get():
            self.audio.start(self.cur_frame)

    def _on_audio_toggle(self):
        if not self.audio_on.get():
            self.audio.stop()
        else:
            self._audio_restart()

    def _on_scale(self, val):
        if abs(float(val) - self.cur_frame) >= 1:
            self.seek(int(float(val)))

    def toggle_play(self):
        self.playing = not self.playing
        self.btn_play.config(text="⏸ 暂停" if self.playing else "▶ 播放")
        if self.playing:
            self._audio_restart()
            self._tick()
        else:
            self.audio.stop()

    def _tick(self):
        if not self.playing:
            return
        nxt = self.cur_frame + 1
        if nxt >= self.n_frames:
            self.playing = False
            self.btn_play.config(text="▶ 播放")
            self.audio.stop()
            return
        self._set_frame(nxt)
        self.after_id = self.root.after(max(10, int(1000.0 / self.fps)), self._tick)

    def on_close(self):
        self.playing = False
        self.audio.stop()
        if self.cap:
            self.cap.release()
        self.root.destroy()


def main():
    ap = argparse.ArgumentParser(description="AIC result viewer")
    ap.add_argument("--result", default=None, help="预填结果 jsonl 路径")
    ap.add_argument("--dataset", default="复赛test", choices=list(DATASETS))
    args = ap.parse_args()

    root = tk.Tk()
    app = Viewer(root)
    if args.dataset in DATASETS:
        app.cb_ds.set(args.dataset)
        app._apply_dataset()
    if args.result:
        app.e_result.delete(0, "end"); app.e_result.insert(0, _p(args.result))
    app.reload()
    root.mainloop()


if __name__ == "__main__":
    sys.exit(main())
