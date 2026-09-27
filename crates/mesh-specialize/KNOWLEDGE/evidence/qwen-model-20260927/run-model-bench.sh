#!/usr/bin/env bash
set -euo pipefail
export GIT_PAGER=cat PAGER=cat TERM=xterm
cd /home/ndizazzo/dev/mesh/mesh-llm
tokens=$1
trial_dir=$2
source_head=$3
ptx=target/specialize/qwen-model-20260927/attention-wide.ptx
test "$(sha256sum "$ptx" | cut -d ' ' -f 1)" = fa04eb2e19c22bcd47fc657c9adb6d8e079349719d31f7bbb213fe85a8a70ab6
test "$(git rev-parse HEAD)" = "$source_head"
test -z "$(git status --porcelain)"
mkdir "$trial_dir"
printf '%s\n' "$source_head" > "$trial_dir/source-head.txt"
printf '%s\n' "$tokens" > "$trial_dir/prompt-token-ids.txt"
systemctl --user is-active ninfer-qwen38.service > "$trial_dir/service-before.txt"
systemctl --user show ninfer-qwen38.service -p MainPID -p ActiveEnterTimestamp >> "$trial_dir/service-before.txt"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader > "$trial_dir/processes-before.csv"
date --iso-8601=seconds > "$trial_dir/start-time.txt"
cleanup() {
  result=$?
  trap - EXIT
  set +e
  systemctl --user start ninfer-qwen38.service
  healthy=0
  for attempt in $(seq 1 45); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 http://127.0.0.1:1235/health)
    if test "$code" = 200; then healthy=1; break; fi
    sleep 1
  done
  systemctl --user show ninfer-qwen38.service -p ActiveState -p MainPID -p ActiveEnterTimestamp > "$trial_dir/service-after.txt"
  printf '%s\n' "$code" > "$trial_dir/health-after.txt"
  journalctl --user -u ninfer-qwen38.service --since "$(cat "$trial_dir/start-time.txt")" --no-pager > "$trial_dir/service-journal.log"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader > "$trial_dir/processes-after.csv"
  date --iso-8601=seconds > "$trial_dir/end-time.txt"
  cat "$trial_dir/service-after.txt" "$trial_dir/health-after.txt" "$trial_dir/processes-after.csv"
  if test "$healthy" != 1; then exit 99; fi
  exit "$result"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
systemctl --user stop ninfer-qwen38.service
nvidia-smi --query-gpu=uuid,name,driver_version,memory.total,memory.free,pstate,clocks.gr,clocks.mem,power.draw,power.limit --format=csv > "$trial_dir/gpu-before.csv"
sha256sum target/release/xtask "$ptx" > "$trial_dir/trial-hashes.txt"
command=(target/release/xtask specialize qwen-model-bench --artifact /data/ai/models/mesh-specialize/qwen3.8-27b-f0b7c9e7-raw-v1.mspec --tokens "$tokens" --output-tokens 8 --repetitions 1 --ptx "$ptx" --device 0 --output "$trial_dir/report.json")
systemd-run --user --scope --quiet --unit "qwen-model-bench-$(date +%s)" -p MemoryMax=8G -p MemorySwapMax=0 timeout 240 "${command[@]}" > "$trial_dir/run.log" 2>&1
jq -M '{completed,error,harness_elapsed_seconds,repetitions,allocation_bytes,memory:{minimum_sampled_free_bytes:.memory.minimum_sampled_free_bytes,arena_memory_release_observed:.memory.arena_memory_release_observed},configured_capacity_tokens,requested_final_cursor_past}' "$trial_dir/report.json"
