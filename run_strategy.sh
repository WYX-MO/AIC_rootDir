#!/usr/bin/env bash
# 每个策略一次性出两份结果：本地 val + 复赛 426 条测试版。
#
#   ./run_strategy.sh emit_trace.py [额外的 emit_*.py 参数...]
#   ./run_strategy.sh emit_gated.py --threshold 2.0
#
# 约定产物（都在 baseline/ 下）：
#   predictions_val_<tag>.jsonl   -> 对 val_gt.jsonl 评分（有真值）
#   predictions_<tag>.jsonl       -> 复赛 426 条（fu_*），跑 validate
# tag 默认取脚本名去掉 emit_/ 后缀，如 emit_trace.py -> trace；可用 --tag 覆盖。
set -euo pipefail
cd "$(dirname "$0")"

if [ $# -lt 1 ]; then
  echo "用法: $0 <emit_脚本.py> [--tag 名字] [其他参数...]" >&2
  exit 2
fi

SCRIPT="$1"; shift
TAG="${SCRIPT#emit_}"; TAG="${TAG%.py}"
ARGS=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--tag" ]; then TAG="$2"; shift 2; else ARGS+=("$1"); shift; fi
done

VAL_OUT="predictions_val_${TAG}.jsonl"
TEST_OUT="predictions_${TAG}.jsonl"

echo "=================================================="
echo " 策略 ${SCRIPT}   tag=${TAG}"
echo "=================================================="

echo "[1/4] 生成 val 版 -> ${VAL_OUT}"
python3 "$SCRIPT" --base predictions_val_center.jsonl \
  --cache yolo_val_cache.json --metadata val_metadata.json \
  --out "$VAL_OUT" "${ARGS[@]}"

echo
echo "[2/4] val 评分"
python3 eval_local.py score --gt val_gt.jsonl --pred "$VAL_OUT" \
  --video-dir val_video --metadata val_metadata.json 2>&1 | \
  grep -E "F1 \(skip|mean IoU|frame recall|frame precision"

echo
echo "[3/4] 生成复赛 426 条 -> ${TEST_OUT}"
python3 "$SCRIPT" --base fu_base.jsonl \
  --cache fu_yolo_test_cache.json --metadata fu_metadata.json \
  --out "$TEST_OUT" "${ARGS[@]}"

echo
echo "[4/4] 复赛 validate（本地 video/ 只有 174 条，unknown video_id 是环境性的）"
python3 eval_local.py validate --pred "$TEST_OUT" \
  --video-dir video --metadata fu_metadata.json 2>&1 | \
  grep -E "prediction lines|total prediction|model_size_mb|violations|unknown video_id in|RESULT"
echo
echo "完成: ${VAL_OUT}  ${TEST_OUT}"
