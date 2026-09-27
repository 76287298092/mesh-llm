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

artifact=/data/ai/models/mesh-specialize/qwen3.8-27b-f0b7c9e7-raw-v1.mspec
reference=target/specialize/qwen-model-20260927/reference-two.json
modes=(normal)
if test "$suite" = all; then modes=(normal memcheck racecheck synccheck); fi
for mode in "${modes[@]}"; do
  command=(target/release/xtask specialize qwen-model-check --artifact "$artifact" --reference "$reference" --ptx "$ptx" --device 0 --output "$trial_dir/$mode.json")
  if test "$mode" != normal; then command=(/opt/cuda/bin/compute-sanitizer --tool "$mode" --error-exitcode 97 --log-file "$trial_dir/$mode-sanitizer.log" "${command[@]}"); fi
  systemd-run --user --scope --quiet --unit "perf-check-$mode-$(date +%s)" -p MemoryMax=8G -p MemorySwapMax=0 timeout 240 "${command[@]}" > "$trial_dir/$mode.log" 2>&1
  jq -e '.all_passed == true and ([.layers[].bf16_differences]|add)==0 and .logits.aggregate.max_abs_error==0 and .whole_vs_token_logits_and_state_bit_exact==true' "$trial_dir/$mode.json" > /dev/null
  jq -Me '{all_passed,fp8_exact_passed:.fp8_exact_probe.all_passed,logit_max_error:.logits.aggregate.max_abs_error,nonexact_layers:[.layers[]?|select(.bf16_differences>0)],whole_vs_token_logits_and_state_bit_exact}' "$trial_dir/$mode.json"
done
if test "${6:-skip}" = before; then
  old_binary=target/specialize/perf-20260927/xtask-before
  old_ptx=target/specialize/qwen-model-20260927/attention-wide.ptx
  test "$(sha256sum "$old_binary" | cut -d ' ' -f 1)" = 67f9f807c1a544ff7cf2fb3d5b4b2c4dd5d8f4395b0c70644178a42fdb8099fc
  test "$(sha256sum "$old_ptx" | cut -d ' ' -f 1)" = fa04eb2e19c22bcd47fc657c9adb6d8e079349719d31f7bbb213fe85a8a70ab6
  sha256sum "$old_binary" "$old_ptx" > "$trial_dir/before-control-hashes.txt"
  for size in two 128; do
    if test "$size" = two; then tokens=248044,271; else tokens=$(python3 -c 'print(",".join(map(str,[248044]+[300+(i*7919)%100000 for i in range(1,128)])))'); fi
    command=("$old_binary" specialize qwen-model-bench --artifact "$artifact" --tokens "$tokens" --output-tokens 8 --repetitions 3 --ptx "$old_ptx" --device 0 --output "$trial_dir/before-bench-$size.json")
    systemd-run --user --scope --quiet --unit "perf-before-$size-$(date +%s)" -p MemoryMax=8G -p MemorySwapMax=0 timeout 240 "${command[@]}" > "$trial_dir/before-bench-$size.log" 2>&1
    jq -M '{completed,samples:[.repetitions[]|{prefill_input_tokens_per_second,decode_tokens_per_second}]}' "$trial_dir/before-bench-$size.json"
  done
fi
for size in two 128; do
  if test "$size" = two; then tokens=248044,271; else tokens=$(python3 -c 'print(",".join(map(str,[248044]+[300+(i*7919)%100000 for i in range(1,128)])))'); fi
  command=(target/release/xtask specialize qwen-model-bench --artifact "$artifact" --tokens "$tokens" --output-tokens 8 --repetitions 3 --ptx "$ptx" --device 0 --output "$trial_dir/bench-$size.json")
  systemd-run --user --scope --quiet --unit "perf-bench-$size-$(date +%s)" -p MemoryMax=8G -p MemorySwapMax=0 timeout 240 "${command[@]}" > "$trial_dir/bench-$size.log" 2>&1
  jq -M '{completed,samples:[.repetitions[]|{prefill_input_tokens_per_second,decode_tokens_per_second}],allocation_bytes,configured_capacity}' "$trial_dir/bench-$size.json"
done
command=(target/release/xtask specialize qwen-model-profile --artifact "$artifact" --tokens 248044,271 --ptx "$ptx" --device 0 --output "$trial_dir/profile.json")
systemd-run --user --scope --quiet --unit "perf-profile-$(date +%s)" -p MemoryMax=8G -p MemorySwapMax=0 timeout 240 "${command[@]}" > "$trial_dir/profile.log" 2>&1
jq -M '{all_passed,unprofiled_decode_wall_seconds,kernel_profile:(.kernel_profile|{launch_count,total_gpu_ms,groups:(.groups[:4])})}' "$trial_dir/profile.json"

command=(target/release/xtask specialize qwen-model-profile --artifact "$artifact" --tokens "$tokens" --ptx "$ptx" --device 0 --output "$trial_dir/profile-long.json")
systemd-run --user --scope --quiet --unit "perf-profile-long-$(date +%s)" -p MemoryMax=8G -p MemorySwapMax=0 timeout 240 "${command[@]}" > "$trial_dir/profile-long.log" 2>&1
jq -M '{all_passed,unprofiled_decode_wall_seconds,kernel_profile:(.kernel_profile|{launch_count,total_gpu_ms,groups:(.groups[:3])})}' "$trial_dir/profile-long.json"
