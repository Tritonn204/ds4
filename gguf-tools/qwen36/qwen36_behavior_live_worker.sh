#!/usr/bin/env bash
set -euo pipefail

HF_PATH="/mnt/e/tensors/Qwen3.6-35B-A3B"
GGUF_PATH="/home/tritonn/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-Q8_0.gguf"
PROMPT_FILE="${2:-/mnt/f/git/ds4/gguf-tools/qwen36/qwen36_oracle_prompts/live_runtime_probe.txt}"

OUT_DIR="${1:-/tmp/qwen36_behavior_live_worker}"
N_PREDICT="${N_PREDICT:-32}"
COOLDOWN_SEC="${COOLDOWN_SEC:-15}"
TOP_K="${TOP_K:-8}"
BASELINE_LAYER_PROGRESS="${BASELINE_LAYER_PROGRESS:-0}"
PATCHED_LAYER_PROGRESS="${PATCHED_LAYER_PROGRESS:-0}"

mkdir -p "$OUT_DIR"

BASELINE_JSON="$OUT_DIR/baseline.json"
PATCHED_JSON="$OUT_DIR/patched.json"
COMPARE_JSON="$OUT_DIR/compare.json"

BASELINE_EXTRA_ARGS=()
PATCHED_EXTRA_ARGS=()

if [[ "$BASELINE_LAYER_PROGRESS" == "1" ]]; then
  BASELINE_EXTRA_ARGS+=(--hf-baseline-layer-progress)
fi

if [[ "$PATCHED_LAYER_PROGRESS" == "1" ]]; then
  PATCHED_EXTRA_ARGS+=(--hf-patched-layer-progress)
fi

echo "[behavior-live] baseline run"
python3 -u /mnt/f/git/ds4/gguf-tools/qwen36/qwen36_behavior_oracle.py \
  --mode baseline \
  --hf "$HF_PATH" \
  --gguf "$GGUF_PATH" \
  --prompt-file "$PROMPT_FILE" \
  --n-predict "$N_PREDICT" \
  --top-k "$TOP_K" \
  "${BASELINE_EXTRA_ARGS[@]}" \
  --json-out "$BASELINE_JSON"

echo "[behavior-live] cooldown ${COOLDOWN_SEC}s"
sleep "$COOLDOWN_SEC"

echo "[behavior-live] patched run"
python3 -u /mnt/f/git/ds4/gguf-tools/qwen36/qwen36_behavior_oracle.py \
  --mode patched-session \
  --hf "$HF_PATH" \
  --gguf "$GGUF_PATH" \
  --c-bin /mnt/f/git/ds4/qwen36-c-prefix-q8-chain-live \
  --c-bin-worker-bin /mnt/f/git/ds4/qwen36-live-contract-worker \
  --full-layer-worker-bin /mnt/f/git/ds4/qwen36-gpu-full-layer-worker \
  --fixture /tmp/qwen36_live_contracts/blk0.live.bin \
  --fixture /tmp/qwen36_live_contracts/blk1.live.bin \
  --fixture /tmp/qwen36_live_contracts/blk2.live.bin \
  --fixture /tmp/qwen36_live_contracts/blk4.live.bin \
  --fixture /tmp/qwen36_live_contracts/blk5.live.bin \
  --fixture /tmp/qwen36_live_contracts/blk6.live.bin \
  --fixture /tmp/qwen36_live_contracts/blk8.live.bin \
  --fixture /tmp/qwen36_live_contracts/blk9.live.bin \
  --fixture /tmp/qwen36_live_contracts/blk10.live.bin \
  --fixture /tmp/qwen36_live_contracts/blk12.live.bin \
  --fixture /tmp/qwen36_live_contracts/blk13.live.bin \
  --fixture /tmp/qwen36_live_contracts/blk14.live.bin \
  --fixture /tmp/qwen36_live_contracts/blk16.live.bin \
  --fixture /tmp/qwen36_live_contracts/blk17.live.bin \
  --fixture /tmp/qwen36_live_contracts/blk18.live.bin \
  --fixture /tmp/qwen36_live_contracts/blk20.live.bin \
  --fixture /tmp/qwen36_live_contracts/blk21.live.bin \
  --fixture /tmp/qwen36_live_contracts/blk22.live.bin \
  --full-layer 3 \
  --full-layer 7 \
  --full-layer 11 \
  --full-layer 15 \
  --full-layer 19 \
  --full-layer 23 \
  --splice-layer 23 \
  --prompt-file "$PROMPT_FILE" \
  --n-predict "$N_PREDICT" \
  --top-k "$TOP_K" \
  "${PATCHED_EXTRA_ARGS[@]}" \
  --json-out "$PATCHED_JSON"

echo "[behavior-live] compare"
python3 -u /mnt/f/git/ds4/gguf-tools/qwen36/qwen36_behavior_compare.py \
  --baseline-json "$BASELINE_JSON" \
  --patched-json "$PATCHED_JSON" \
  --json-out "$COMPARE_JSON"

echo "[behavior-live] done"
echo "[behavior-live] baseline_json=$BASELINE_JSON"
echo "[behavior-live] patched_json=$PATCHED_JSON"
echo "[behavior-live] compare_json=$COMPARE_JSON"
