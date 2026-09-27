#!/usr/bin/env bash
set -euo pipefail
export GIT_PAGER=cat PAGER=cat TERM=xterm
cd /home/ndizazzo/dev/mesh/mesh-llm
reference=$1
trial_dir=$2
mode=$3
source_head=$4
ptx=target/specialize/qwen-model-20260927/bf16-refined.ptx
test "$(sha256sum "$ptx" | cut -d ' ' -f 1)" = 4a5b55a1d4a29325b1b357f4f822e815caab9f9f1648c05b4632d898ba40ad77
test "$(git rev-parse HEAD)" = "$source_head"
test -z "$(git status --porcelain)"
test -f "$reference"
mkdir "$trial_dir"
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
sha256sum target/release/xtask "$ptx" "$reference" > "$trial_dir/trial-hashes.txt"
command=(target/release/xtask specialize qwen-model-check --artifact /data/ai/models/mesh-specialize/qwen3.8-27b-f0b7c9e7-raw-v1.mspec --reference "$reference" --ptx "$ptx" --device 0 --output "$trial_dir/report.json")
if test "$mode" != normal; then
  command=(/opt/cuda/bin/compute-sanitizer --tool "$mode" --error-exitcode 97 --log-file "$trial_dir/sanitizer.log" "${command[@]}")
fi
systemd-run --user --scope --quiet --unit "qwen-model-$mode-$(date +%s)" -p MemoryMax=8G -p MemorySwapMax=0 timeout 240 "${command[@]}" > "$trial_dir/run.log" 2>&1
jq -M '{all_passed,error,elapsed_seconds,activation_probe,full_model_executed,greedy_token_exact,selected_token,reference_token,whole_vs_token_logits_and_state_bit_exact,logits:.logits.aggregate,first_nonexact:([.layers[]?|select(.bf16_differences>0)][0]),stages:[.stages[]?|{name,bf16_differences}]}' "$trial_dir/report.json"
