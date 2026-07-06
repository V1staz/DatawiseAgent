#!/usr/bin/env bash
set -Eeuo pipefail

# Runs qwen3-vl-8b-thinking first through the quality-gated DataModeling agent
# harness. Only if 8B reaches the configured normalized-performance threshold
# does this script launch qwen3-vl-32b-thinking. No API key is stored here.

ROOT="${DWA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
if [[ -n "${DWA_PY:-}" ]]; then
  PY="$DWA_PY"
elif [[ -x /home/yu/miniconda3/envs/DWA/bin/python ]]; then
  PY=/home/yu/miniconda3/envs/DWA/bin/python
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi
WORKERS="${DWA_DM_WORKERS:-2}"
CHAT_TIMEOUT="${DWA_DM_CHAT_TIMEOUT:-1200}"
EARLY_STOP_MIN="${DWA_DM_EARLY_STOP_MIN:-60}"
EARLY_STOP_POLL="${DWA_DM_EARLY_STOP_POLL:-10}"
PERF_THRESHOLD="${DWA_DM_8B_GATE_THRESHOLD:-0.4290}"
MIN_SCORE_OK="${DWA_DM_MIN_SCORE_OK:-70}"
MIN_QUALITY_PASSED="${DWA_DM_MIN_QUALITY_PASSED:-0}"
SILICONFLOW_BASE_URL="${SILICONFLOW_BASE_URL:-https://api.siliconflow.cn/v1}"

if [[ -z "${SILICONFLOW_API_KEY:-}" ]]; then
  echo "SILICONFLOW_API_KEY is not set. Export it before running this script." >&2
  exit 2
fi

SECRET_ENV_FILE="$(mktemp -t dwa_siliconflow_env.XXXXXX)"
chmod 600 "$SECRET_ENV_FILE"
{
  printf 'export OPENAI_BASE_URL=%q\n' "$SILICONFLOW_BASE_URL"
  printf 'export OPENAI_API_KEY=%q\n' "$SILICONFLOW_API_KEY"
} > "$SECRET_ENV_FILE"
cleanup_secret_env(){ rm -f "$SECRET_ENV_FILE"; }
trap cleanup_secret_env EXIT

cd "$ROOT"
LOG_DIR="evaluation/experimental_results/Datamodeling-DSBench/run_logs"
mkdir -p "$LOG_DIR"

log(){ printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "$LOG_DIR/qwen3vl_quality_gate_driver.log"; }

ensure_server(){
  local session="$1" config="$2" logfile="$3"
  log "starting server $session with config $config via OPENAI_BASE_URL=$SILICONFLOW_BASE_URL"
  tmux kill-session -t "$session" 2>/dev/null || true
  if ss -ltn '( sport = :8000 )' | grep -q ':8000'; then
    log "killing existing process on port 8000 before $session"
    fuser -k 8000/tcp || true
    sleep 3
  fi
  tmux new-session -d -s "$session" "cd '$ROOT'; source '$SECRET_ENV_FILE'; CUSTOM_CONFIG='$config' '$PY' main.py 2>&1 | tee -a '$logfile'"
  for _ in $(seq 1 90); do
    if curl -fsS http://127.0.0.1:8000/healthz >/dev/null 2>&1; then
      log "server $session healthy"
      return 0
    fi
    sleep 1
  done
  log "server $session failed health check"
  tmux capture-pane -pt "$session" -S -120 || true
  return 1
}

run_full_model(){
  local model="$1" config="$2" session="$3" out="$4" logfile="$5"
  ensure_server "$session" "$config" "$LOG_DIR/server_${model}.log"
  mkdir -p "$out"
  local start end rc
  start=$(date +%s)
  echo "{\"model\":\"$model\",\"started_at\":$start,\"num_workers\":$WORKERS,\"quality_gate\":true,\"provider\":\"siliconflow\"}" > "$out/run_meta_start.json"
  "$PY" evaluation/run_agent_harness_datamodeling.py \
    --model-name "$model" \
    --output-dir "$out" \
    --num-workers "$WORKERS" \
    --max-repair-attempts 2 \
    --chat-timeout-seconds "$CHAT_TIMEOUT" \
    --early-stop-min-seconds "$EARLY_STOP_MIN" \
    --early-stop-poll-seconds "$EARLY_STOP_POLL" \
    --score 2>&1 | tee -a "$LOG_DIR/$logfile"
  rc=${PIPESTATUS[0]}
  end=$(date +%s)
  echo "{\"model\":\"$model\",\"finished_at\":$end,\"seconds\":$((end-start)),\"rc\":$rc}" > "$out/run_meta_finish.json"
  return "$rc"
}

compare_models(){
  local output_json="$1"; shift
  "$PY" evaluation/DataModeling/scripts/compare_agent_harness_results.py \
    --models "$@" \
    --output-json "$output_json"
}

EIGHT_MODEL="qwen3-vl-8b-thinking-agent-harness-quality-full"
EIGHT_OUT="evaluation/experimental_results/Datamodeling-DSBench/$EIGHT_MODEL"
THIRTYTWO_MODEL="qwen3-vl-32b-thinking-agent-harness-quality-full"
THIRTYTWO_OUT="evaluation/experimental_results/Datamodeling-DSBench/$THIRTYTWO_MODEL"
COMPARE_JSON="evaluation/experimental_results/Datamodeling-DSBench/qwen3vl_quality_gate_compare.json"
GATE_JSON="evaluation/experimental_results/Datamodeling-DSBench/qwen3vl_quality_gate_decision.json"

run_full_model "$EIGHT_MODEL" configs/setting_model_siliconflow_qwen3_vl_8b_thinking.yaml dwa-qwen8b-quality "$EIGHT_OUT" "$EIGHT_MODEL.log" || true
compare_models "$COMPARE_JSON" "$EIGHT_MODEL" qwen25 qwen35b-a3b 2>/dev/null || compare_models "$COMPARE_JSON" "$EIGHT_MODEL"

if "$PY" evaluation/DataModeling/scripts/check_qwen3vl_gate.py \
  --compare-json "$COMPARE_JSON" \
  --model "$EIGHT_MODEL" \
  --threshold "$PERF_THRESHOLD" \
  --min-quality-passed "$MIN_QUALITY_PASSED" \
  --min-score-ok "$MIN_SCORE_OK" | tee "$GATE_JSON"; then
  log "8B passed gate; launching 32B"
  tmux kill-session -t dwa-qwen8b-quality 2>/dev/null || true
  fuser -k 8000/tcp >/dev/null 2>&1 || true
  sleep 3
  run_full_model "$THIRTYTWO_MODEL" configs/setting_model_siliconflow_qwen3_vl_32b_thinking.yaml dwa-qwen32b-quality "$THIRTYTWO_OUT" "$THIRTYTWO_MODEL.log" || true
  compare_models "$COMPARE_JSON" "$EIGHT_MODEL" "$THIRTYTWO_MODEL" qwen25 qwen35b-a3b 2>/dev/null || compare_models "$COMPARE_JSON" "$EIGHT_MODEL" "$THIRTYTWO_MODEL"
else
  log "8B did not pass gate; 32B run skipped"
fi

log "qwen3-vl quality-gated pipeline complete"
