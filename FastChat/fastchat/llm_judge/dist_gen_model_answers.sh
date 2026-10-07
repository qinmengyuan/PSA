#!/bin/bash

set -ex

# 配置参数
TOTAL_QUESTIONS=2037
MAX_RETRIES=2
MODEL_PATH="/rds/projects/f/fengyv-proactive-ds/projects/ActorAttack/ft/work_dirs/llama3_8b_instruct_qlora-pinjie1_2/merged"
ANSWER_DIR="/rds/projects/f/fengyv-proactive-ds/projects/SafeDialBench-Dataset/FastChat/fastchat/llm_judge/data/SafeDial_bench/model_answer/fy_actor_wupinjie"
ANSWER_PREFIX="V84_part"
FINAL_OUTPUT="V84.jsonl"
MODEL_ID="fy_actor_wupinjie"
# 获取GPU数量
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
# NUM_GPUS=2
echo "Detected $NUM_GPUS GPUs."

# 均分问题
QUESTIONS_PER_TASK=$((TOTAL_QUESTIONS / NUM_GPUS))

# 启动每个子任务
for ((i=0; i<NUM_GPUS; i++)); do
  BEGIN=$((i * QUESTIONS_PER_TASK))
  END=$(((i + 1) * QUESTIONS_PER_TASK))
  [[ $i -eq $((NUM_GPUS - 1)) ]] && END=$TOTAL_QUESTIONS

  LOG_FILE="${ANSWER_PREFIX}${i}.log"
  OUTPUT_FILE="${ANSWER_DIR}/${ANSWER_PREFIX}${i}.jsonl"
  DONE_FILE="${ANSWER_PREFIX}${i}.done"

  (
    ATTEMPT=0
    while [[ $ATTEMPT -le $MAX_RETRIES ]]; do
      echo "[GPU $i][Attempt $ATTEMPT] Running questions $BEGIN to $END..." | tee "$LOG_FILE"
      # CUDA_VISIBLE_DEVICES=$((i + 2)) python gen_model_answer.py \
      CUDA_VISIBLE_DEVICES=$i python gen_model_answer.py \
        --model-path "$MODEL_PATH" \
        --model-id "$MODEL_ID" \
        --language en \
        --question-begin $BEGIN \
        --question-end $END \
        --answer-file "$OUTPUT_FILE" \
        >> "$LOG_FILE" 2>&1

      if [[ $? -eq 0 ]]; then
        echo "[GPU $i] Finished successfully." >> "$LOG_FILE"
        touch "$DONE_FILE"
        break
      else
        echo "[GPU $i] Failed on attempt $ATTEMPT." >> "$LOG_FILE"
        ((ATTEMPT++))
        sleep 5
      fi
    done

    if [[ ! -f "$DONE_FILE" ]]; then
      echo "[GPU $i] Failed after $MAX_RETRIES retries." >> "$LOG_FILE"
    fi
  ) &
done

# 等待所有子任务结束
wait
echo "All subprocesses finished."

# 检查所有任务是否成功完成
ALL_DONE=true
for ((i=0; i<NUM_GPUS; i++)); do
  if [[ ! -f "${ANSWER_PREFIX}${i}.done" ]]; then
    echo "❌ Task $i did not complete successfully. Check ${ANSWER_PREFIX}${i}.log"
    ALL_DONE=false
  fi
done

# 合并输出文件
if $ALL_DONE; then
  echo "✅ All tasks done. Merging output to ${FINAL_OUTPUT}..."
  cat "${ANSWER_DIR}/${ANSWER_PREFIX}"*.jsonl > "${ANSWER_DIR}/${FINAL_OUTPUT}"
  echo "✅ Merge complete: ${FINAL_OUTPUT}"
else
  echo "⚠️ Some tasks failed. Skipping merge."
fi
# 收到 Ctrl+C (SIGINT) 时终止所有子进程
trap "echo 'Interrupted. Killing subprocesses...'; pkill -P $$; exit 1" SIGINT
