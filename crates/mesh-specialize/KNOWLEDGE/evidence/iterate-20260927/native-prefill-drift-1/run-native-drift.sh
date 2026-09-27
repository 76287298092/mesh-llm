#!/usr/bin/env bash
set -euo pipefail
export GIT_PAGER=cat PAGER=cat TERM=xterm
cd /home/ndizazzo/dev/mesh/mesh-llm
trial_dir=$1
source_head=$2
ptx=$3
ptx_hash=$4
suite=${5:-normal}
tokens=248044,271
test "$(sha256sum "$ptx" | cut -d ' ' -f 1)" = "$ptx_hash"
test "$(git rev-parse HEAD)" = "$source_head"
test -z "$(git status --porcelain)"
mkdir "$trial_dir"
printf '%s\n' "$source_head" > "$trial_dir/source-head.txt"
printf '%s\n' "$tokens" > "$trial_dir/prompt-token-ids.txt"
initial_active=$(systemctl --user is-active ninfer-qwen38.service || true)
printf '%s\n' "$initial_active" > "$trial_dir/service-before.txt"
systemctl --user show ninfer-qwen38.service -p MainPID -p ActiveEnterTimestamp >> "$trial_dir/service-before.txt"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader > "$trial_dir/processes-before.csv"
date --iso-8601=seconds > "$trial_dir/start-time.txt"
sampler_pid=
cleanup() {
  result=$?
  trap - EXIT
  set +e
  if test -n "$sampler_pid"; then kill "$sampler_pid" 2>/dev/null; wait "$sampler_pid" 2>/dev/null; fi
  healthy=1
  if test "$initial_active" = active; then
  systemctl --user start ninfer-qwen38.service
  healthy=0
  for attempt in $(seq 1 45); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 http://127.0.0.1:1235/health)
    if test "$code" = 200; then healthy=1; break; fi
    sleep 1
  done
  else
    code=not-restored-initially-inactive
  fi
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
if test "$initial_active" = active; then systemctl --user stop ninfer-qwen38.service; fi
(while true; do date --iso-8601=seconds; nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader; sleep 1; done) > "$trial_dir/process-samples.log" 2>&1 &
sampler_pid=$!
nvidia-smi --query-gpu=uuid,name,driver_version,memory.total,memory.free,pstate,clocks.gr,clocks.mem,power.draw,power.limit --format=csv > "$trial_dir/gpu-before.csv"
sha256sum target/release/xtask "$ptx" > "$trial_dir/trial-hashes.txt"

artifact=/data/ai/models/mesh-specialize/qwen3.8-27b-f0b7c9e7-raw-v1.mspec


tokens=$(python3 -c 'print(",".join(map(str,[248044]+[300+(i*7919)%100000 for i in range(1,128)])))')
command=(env MESH_SPECIALIZE_FP8_PROFILE=native-prefill target/release/xtask specialize qwen-model-profile --artifact "$artifact" --tokens "$tokens" --ptx "$ptx" --device 0 --output "$trial_dir/profile-native-prefill.json")
set +e
systemd-run --user --scope --quiet --unit "native-drift-$(date +%s)" -p MemoryMax=8G -p MemorySwapMax=0 timeout --kill-after=15s 240 "${command[@]}" > "$trial_dir/profile-native-prefill.log" 2>&1
code=$?
set -e
printf '%s\n' "$code" > "$trial_dir/profile-exit.txt"
jq -e '.completed == true and .exact_output_and_state == true' "$trial_dir/profile-native-prefill.json" > /dev/null
jq -M '{arithmetic_profile,all_passed,partition:.whole_vs_token_partition}' "$trial_dir/profile-native-prefill.json"
