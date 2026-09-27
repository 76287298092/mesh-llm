#!/usr/bin/env bash
set -euo pipefail
export GIT_PAGER=cat PAGER=cat TERM=xterm
cd /home/ndizazzo/dev/mesh/mesh-llm
trial_dir=$1
source_head=$2
ptx=$3
ptx_hash=$4
suite=${5:-normal}
test "$(sha256sum "$ptx" | cut -d ' ' -f 1)" = "$ptx_hash"
test "$(git rev-parse HEAD)" = "$source_head"
test -z "$(git status --porcelain)"
mkdir "$trial_dir"
printf '%s\n' "$source_head" > "$trial_dir/source-head.txt"
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
cp target/specialize/iterate-20260927/text-prompts.json "$trial_dir/prompts.json"
for name in python explain; do
 tokens=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(",".join(map(str,next(p["tokens"] for p in d["prompts"] if p["name"]==sys.argv[2]))))' "$trial_dir/prompts.json" "$name")
 for mode in exact a16-decode; do
 command=(env MESH_SPECIALIZE_FP8_PROFILE="$mode" target/release/xtask specialize qwen-model-bench --artifact "$artifact" --tokens "$tokens" --output-tokens 32 --repetitions 3 --ptx "$ptx" --device 0 --output "$trial_dir/bench-$name-$mode.json")
 systemd-run --user --scope --quiet --unit "a16-bench-$name-$mode-$(date +%s)" -p MemoryMax=8G -p MemorySwapMax=0 timeout --kill-after=15s 300 "${command[@]}" > "$trial_dir/bench-$name-$mode.log" 2>&1
 jq -e '.completed == true' "$trial_dir/bench-$name-$mode.json" >/dev/null
 teacher=$(jq -r '.repetitions[0].generated_token_ids[0]' "$trial_dir/bench-$name-exact.json")
 command=(env MESH_SPECIALIZE_FP8_PROFILE="$mode" MESH_SPECIALIZE_LOGIT_DUMP_DIR="$trial_dir/logits-$name-$mode" target/release/xtask specialize qwen-model-profile --artifact "$artifact" --tokens "$tokens" --ptx "$ptx" --device 0 --output "$trial_dir/profile-$name-$mode.json" --teacher-token "$teacher")
 set +e
 systemd-run --user --scope --quiet --unit "a16-profile-$name-$mode-$(date +%s)" -p MemoryMax=8G -p MemorySwapMax=0 timeout --kill-after=15s 300 "${command[@]}" > "$trial_dir/profile-$name-$mode.log" 2>&1
 status=$?
 set -e
 printf '%s\n' "$status" > "$trial_dir/profile-$name-$mode.exit"
 if test "$mode" = exact; then test "$status" = 0; else test "$status" = 0 || test "$status" = 1; fi
 jq -e '.completed == true and .exact_output_and_state == true and .exact_prefill_logits == true and .memory_released == true' "$trial_dir/profile-$name-$mode.json" >/dev/null
 done
done
for tool in memcheck racecheck synccheck; do
 command=(env MESH_SPECIALIZE_FP8_PROFILE=a16-decode compute-sanitizer --tool "$tool" --error-exitcode 97 target/release/xtask specialize qwen-model-profile --artifact "$artifact" --tokens 248044 --ptx "$ptx" --device 0 --output "$trial_dir/$tool.json" --teacher-token 271)
 systemd-run --user --scope --quiet --unit "a16-check-$tool-$(date +%s)" -p MemoryMax=8G -p MemorySwapMax=0 timeout --kill-after=15s 420 "${command[@]}" > "$trial_dir/$tool.log" 2>&1
 jq -e '.all_passed == true' "$trial_dir/$tool.json" >/dev/null
done
