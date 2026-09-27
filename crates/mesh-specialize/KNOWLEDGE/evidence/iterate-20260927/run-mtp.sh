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
reference=target/specialize/iterate-20260927/mtp-reference-two.json
command=(target/release/xtask specialize qwen-model-check --artifact "$artifact" --reference target/specialize/qwen-model-20260927/reference-two.json --ptx "$ptx" --device 0 --output "$trial_dir/target-check.json")
systemd-run --user --scope --quiet --unit "mtp-target-$(date +%s)" -p MemoryMax=8G -p MemorySwapMax=0 timeout 240 "${command[@]}" > "$trial_dir/target-check.log" 2>&1
jq -e '.all_passed == true and ([.layers[].bf16_differences]|add)==0 and .logits.aggregate.max_abs_error==0 and .whole_vs_token_logits_and_state_bit_exact==true' "$trial_dir/target-check.json" > /dev/null
modes=(normal)
if test "$suite" = all; then modes=(normal memcheck racecheck synccheck); fi
for mode in "${modes[@]}"; do
 command=(target/release/xtask specialize qwen-mtp-check --artifact "$artifact" --reference "$reference" --tokens 248044,271 --output-tokens 8 --depth 4 --repetitions 1 --ptx "$ptx" --device 0 --output "$trial_dir/$mode.json")
 if test "$mode" != normal; then command=(/opt/cuda/bin/compute-sanitizer --tool "$mode" --error-exitcode 97 --log-file "$trial_dir/$mode-sanitizer.log" "${command[@]}"); fi
 systemd-run --user --scope --quiet --unit "mtp-check-$mode-$(date +%s)" -p MemoryMax=8G -p MemorySwapMax=0 timeout 300 "${command[@]}" > "$trial_dir/$mode.log" 2>&1
 jq -e '.all_passed == true' "$trial_dir/$mode.json" > /dev/null
 jq -M '{all_passed,head:(.head|{all_passed,whole_token_partition_bit_exact,greedy_matches_reference}),control,forced_rejection:(.forced_rejection|{all_passed,replay_rows}),trials,memory}' "$trial_dir/$mode.json"
done
if test "$suite" = all; then
 for name in count python explain; do
  tokens=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(",".join(map(str,next(p["tokens"] for p in d["prompts"] if p["name"]==sys.argv[2]))))' target/specialize/iterate-20260927/text-prompts.json "$name")
  for depth in 1 4; do
   command=(target/release/xtask specialize qwen-mtp-check --artifact "$artifact" --reference "$reference" --tokens "$tokens" --output-tokens 32 --depth "$depth" --repetitions 3 --ptx "$ptx" --device 0 --output "$trial_dir/$name-depth$depth.json")
   systemd-run --user --scope --quiet --unit "mtp-bench-$name-$depth-$(date +%s)" -p MemoryMax=8G -p MemorySwapMax=0 timeout 300 "${command[@]}" > "$trial_dir/$name-depth$depth.log" 2>&1
   jq -e '.all_passed == true' "$trial_dir/$name-depth$depth.json" > /dev/null
   jq -M '{all_passed,depth,control_tps:.control.decode_tokens_per_second,trials:[.trials[]|{decode_tokens_per_second,accepted_fraction,rounds,replay_rows,total_generation_speedup}],memory}' "$trial_dir/$name-depth$depth.json"
  done
 done
fi
