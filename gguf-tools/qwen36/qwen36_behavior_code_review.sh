#!/usr/bin/env bash
set -euo pipefail

HF_PATH="/mnt/e/tensors/Qwen3.6-35B-A3B"
GGUF_PATH="/home/tritonn/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-Q8_0.gguf"
PROMPT_FILE="/mnt/f/git/ds4/gguf-tools/qwen36/qwen36_oracle_prompts/code_review.txt"

OUT_DIR="${1:-/tmp/qwen36_behavior_code_review}"
N_PREDICT="${N_PREDICT:-32}"
COOLDOWN_SEC="${COOLDOWN_SEC:-15}"
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

echo "[behavior-sh] baseline run"
python3 -u /mnt/f/git/ds4/gguf-tools/qwen36/qwen36_behavior_oracle.py \
  --mode baseline \
  --hf "$HF_PATH" \
  --gguf "$GGUF_PATH" \
  --prompt-file "$PROMPT_FILE" \
  --n-predict "$N_PREDICT" \
  --top-k 8 \
  "${BASELINE_EXTRA_ARGS[@]}" \
  --json-out "$BASELINE_JSON"

echo "[behavior-sh] cooldown ${COOLDOWN_SEC}s"
sleep "$COOLDOWN_SEC"

echo "[behavior-sh] patched run"
python3 -u /mnt/f/git/ds4/gguf-tools/qwen36/qwen36_same_process_owned_tail_probe.py \
  --hf "$HF_PATH" \
  --gguf "$GGUF_PATH" \
  --prefix-seq-worker-bin /mnt/f/git/ds4/qwen36-fixture-blk0-worker \
  --prefix-seq-fixture /tmp/qwen36_decoder_layer_weight_blk0_code_review.bin \
  --c-bin /mnt/f/git/ds4/qwen36-c-prefix-q8-chain-dynamic \
  --c-bin-worker-bin /mnt/f/git/ds4/qwen36-fixture-blk0-worker \
  --full-layer-worker-bin /mnt/f/git/ds4/qwen36-gpu-full-layer-worker \
  --full-layer 3 \
  --full-layer 7 \
  --full-layer 11 \
  --full-layer 15 \
  --fixture /tmp/qwen36_decoder_layer_weight_blk1_code_review.bin \
  --fixture /tmp/qwen36_decoder_layer_weight_blk2_code_review.bin \
  --fixture /tmp/qwen36_decoder_layer_weight_blk4_code_review.bin \
  --fixture /tmp/qwen36_decoder_layer_weight_blk5_code_review.bin \
  --fixture /tmp/qwen36_decoder_layer_weight_blk6_code_review.bin \
  --fixture /tmp/qwen36_decoder_layer_weight_blk8_code_review.bin \
  --fixture /tmp/qwen36_decoder_layer_weight_blk9_code_review.bin \
  --fixture /tmp/qwen36_decoder_layer_weight_blk10_code_review.bin \
  --fixture /tmp/qwen36_decoder_layer_weight_blk12_code_review.bin \
  --fixture /tmp/qwen36_decoder_layer_weight_blk13_code_review.bin \
  --fixture /tmp/qwen36_decoder_layer_weight_blk14_code_review.bin \
  --prompt-file "$PROMPT_FILE" \
  --splice-layer 15 \
  --n-predict "$N_PREDICT" \
  --top-k 8 \
  "${PATCHED_EXTRA_ARGS[@]}" \
  --json-out "$PATCHED_JSON"

echo "[behavior-sh] compare"
python3 -u /mnt/f/git/ds4/gguf-tools/qwen36/qwen36_behavior_compare.py \
  --baseline-json "$BASELINE_JSON" \
  --patched-json "$PATCHED_JSON" \
  --json-out "$COMPARE_JSON"

echo "[behavior-sh] done"
echo "[behavior-sh] baseline_json=$BASELINE_JSON"
echo "[behavior-sh] patched_json=$PATCHED_JSON"
echo "[behavior-sh] compare_json=$COMPARE_JSON"
