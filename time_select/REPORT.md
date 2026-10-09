# 时间维高光选择实验（time_select）

针对复赛 426 条里 **159 条「源画幅 == 目标画幅」** 的条目单独做一条策略线：
这些条目裁剪框恒等于整幅，**得分 100% 由「选哪些帧」决定**。

试了 **3 类共 16 种**时间选择策略（低层信号 / 位置先验 / 多模态 VLM VLM），
在本地 val 上打分并对比。

---

## 0. 结论速览（TL;DR）

> **没有任何一种时间选择策略能超过基线 `keep_all`（保留前 92% 帧）。**

| 口径 | keep_all（基线） | 最优替代 | VLM 臂 |
| --- | --- | --- | --- |
| 官方 val F1（含裁剪框 IoU） | **0.4431** | 0.4255（位置先验） | 0.337–0.370 |
| 纯帧集 Dice（= 159 条测试的计分方式） | **0.5848** | 0.5543（位置先验） | 0.4435–0.4827 |

因此对这 159 条，**现行提交（keep_all）在时间维上已经是最优**，无需改动。

---

## 1. 为什么这 159 条只看「时间」

- 159 条全部落在 id `174–425`，且源宽高比 == 目标宽高比：
  `crop_size(W,H,tw,th) == (W,H)` → 裁剪框 = 整幅 → 命中帧 IoU = 1。
- 于是单条得分退化为**纯帧集合 Dice**：
  `F1 = 2·|P∩G| / (|P|+|G|)`，其中 P = 提交帧集，G = GT 帧集。
- 现状：`fu_base.jsonl` / `predictions_trace.jsonl` 对这 159 条已经是
  **保留 `0 .. 0.92·n` 的帧（keep_all）**，框 = 整幅。

## 2. 数据事实（val 57 条 GT）

- GT 是**单段连续窗**的占 39/57（68%）；窗长中位 **0.37n**，起始中位 **0.16n**，结束中位 **0.87n**。
- 窗位置**方差很大**：对 `[a,b]` 做 0.02 步长网格搜索，最优固定窗 = `[0.00, 0.91]`，
  Dice = **0.5850**，与 keep_all（0.5848）**完全打平** → 没有可用的位置先验。
- **长度条件窗**（按 n_frames 短/长分桶各取最优窗）同样无增益：
  短视频 keep_all 0.6461 vs 条件窗 0.6478（+0.002，噪声）；长视频两者都 0.5214。

## 3. 信号诊断：为什么低层信号不行

逐帧信号对「该帧是否属于 GT 窗」的判别 AUC（0.5 = 纯随机）：

| 信号 | AUC |
| --- | --- |
| audio（音频 RMS 包络） | **0.497**（≈随机） |
| motion（相邻帧灰度差） | 0.562 |
| person（person 检出数） | 0.525 |
| ndet（总检出数） | 0.451 |
| conf（person 置信和） | 0.519 |

信号的形状**不携带**「哪里是高光」的信息。因此任何"掐头去尾"式裁剪，
都是把正确帧和错误帧**等比例**丢掉 → Dice 单调下降。

## 4. 实验矩阵与结果（`run_experiments.py --dataset val`）

`out/val/summary_val.csv`：

| 策略 | val F1 | 纯帧 Dice | 帧保留率 | 说明 |
| --- | --- | --- | --- | --- |
| **keep_all** | **0.4431** | **0.5848** | 1.000 | 基线（对照） |
| position_0.16_0.88 | 0.4255 | 0.5543 | 0.782 | 纯位置先验 |
| multi_comb_0.7_k3 | 0.4052 | 0.5391 | 0.700 | 三信号融合·多窗 |
| window_comb_0.7 | 0.3986 | 0.5222 | 0.700 | 三信号融合·最佳连续窗 |
| window_comb_0.6 | 0.3822 | 0.5107 | 0.600 | 同上 frac=0.6 |
| top_comb_0.6 | 0.3748 | 0.4993 | 0.600 | 融合 top-p 散点 |
| vlm_1.5 | 0.3701 | 0.4827 | 0.519 | VLM 区间 ×1.5 |
| multi_comb_0.6_k2 | 0.3696 | 0.4931 | 0.609 | 多窗 k=2 |
| window_person_0.6 | 0.3676 | 0.4826 | 0.600 | 仅 person |
| window_motion_0.6 | 0.3664 | 0.4879 | 0.600 | 仅 motion |
| window_audioMotion_0.6 | 0.3592 | 0.4797 | 0.600 | audio+motion |
| vlm_1.25 | 0.3554 | 0.4662 | 0.470 | VLM ×1.25 |
| window_audio_0.6 | 0.3478 | 0.4531 | 0.600 | 仅 audio |
| vlm_1.0 | 0.3370 | 0.4435 | 0.422 | VLM ×1.0 |
| window_comb_0.5 | 0.3321 | 0.4466 | 0.500 | 融合 frac=0.5 |
| window_audio_0.5 | 0.3223 | 0.4267 | 0.500 | audio frac=0.5 |

**全部低于 keep_all**，且保留率越低掉得越多。

## 5. 多模态 VLM 臂（`vlm_select.py`）

- 模型 Qwen2.5-VL-3B-Instruct（4bit），stage1 抽 12 帧，问 model「高光的 start/end 秒」。
- 关掉空间 stage2（本实验只关心时间），产出 `cache/vlm_intervals_<dataset>.json`。
- 结果关键现象：**widen 越大分越高**（1.0 → 1.5 时 val F1 0.337 → 0.370）。
  这是「区间无定位力」的典型特征—— 区间越宽越接近 keep_all。
- 抽看 3 条 val：1 条起点猜对但严重超长，2 条明显错位（见 `out/viz/`）。
- **关键：VLM 区间的「统计分布」其实和 GT 很像**（val：start 中位 0.13 vs GT 0.16，
  长度中位 0.40 vs GT 0.37），但**逐视频的位置是错的** —— Dice 是逐视频算的，
  分布对、个体错并不能加分。这也是它涨不过 keep_all 的原因。

### 5.1 复赛 159 条上的 VLM 产物

- 159 条里 149 条给出区间，10 条为空（空 → 自动退回 keep_all）。
- 帧保留率 0.36–0.42（widen 1.0–1.5），即 VLM 把这批视频剪到约 40%。

## 5.5 复赛（159 条）产物

`run_experiments.py --dataset test --ids fullframe` 为 **16 种策略各出**一份
`out/test/predictions_test_<策略>.jsonl`（只含这 159 条）。列表：

```
predictions_test_keep_all.jsonl            predictions_test_top_comb_0.6.jsonl
predictions_test_position_0.16_0.88.jsonl  predictions_test_multi_comb_0.6_k2.jsonl
predictions_test_window_audio_0.5/0.6 …    predictions_test_multi_comb_0.7_k3.jsonl
predictions_test_window_motion_0.6 …       predictions_test_vlm_1.0/1.25/1.5.jsonl
predictions_test_window_person_0.6 …
predictions_test_window_comb_0.5/0.6/0.7 …
predictions_test_window_audioMotion_0.6 …
```

并回完整 426 条提交文件（159 条换帧集、其余沿用空间臂）：

```
out/test/predictions_timesel_<策略>.jsonl
```

>`eval_local.py validate`：426 行、`violations: 0`、`model_size_mb 5.26`。
>（`RESULT: FAIL` 只是环境性——本地 `video/` 只有初赛 174 条，故 174–425 报
> `unknown video_id`；判分以 violations=0 与行数为准。）

## 6. 产物（都在本文件夹）

```
time_select/
  signals.py            # 逐视频抽信号（audio/motion/person/ndet/conf），缓存 cache/signals_*.json
  frame_selectors.py    # 选择器注册表（keep_all / best_window / multi_window / position_window / interval …）
  run_experiments.py    # 遍历选择器：出预测 + 打分 + 写文件
  vlm_select.py         # 多模态时间区间臂（Qwen2.5-VL-3B）
  viz_time.py           # 可视化：信号曲线 + 各方案保留区间 + frame-Dice
  assemble_test.py      # 把时间臂产物并回完整 426 条提交文件
  cache/                # signals_val/test.json, vlm_intervals_val/test.json
  out/val/              # 每个策略一份 predictions_val_*.jsonl + summary_val.csv
  out/test/             # 159 条测试的各策略 predictions_test_*.jsonl
  out/viz/              # 波形/区间对比图
  REPORT.md
```

- 完整 426 条提交文件（159 条换用某时间策略）：
  `python3 assemble_test.py --sel keep_all` → `out/test/predictions_timesel_keep_all.jsonl`。

## 7. 建议

1. **这 159 条保持 keep_all**：时间维已无空间可压。现行提交即最优。
2. 若仍要压时间维，需要**真正有定位力的信号**，候选方向：
   - 更强/更大的多模态模型，并把 prompt 改成**逐帧打分**（对每帧打 0–10，取连续高分段），
     而非现在的「直接报 start/end」（3B 模型现在倾向报过宽区间）；
   - 若赛方提供 GT 或允许训练，用训练集 987 条的 `segments` 监督一个小打分器。
3. 精力应优先投在**其余 267 条**（需要挪框）的空间臂上。
