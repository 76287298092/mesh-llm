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
reference=target/specialize/iterate-20260927/mtp-reference-two.json
cp target/specialize/iterate-20260927/text-prompts.json "$trial_dir/prompts.json"
for name in python; do
 tokens=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(",".join(map(str,next(p["tokens"] for p in d["prompts"] if p["name"]==sys.argv[2]))))' "$trial_dir/prompts.json" "$name")
 for mode in off 2 4 8 16; do
 command=(env MESH_SPECIALIZE_FP8_SPLIT_K="$mode" MESH_SPECIALIZE_MTP_RECOVERY=full-forward target/release/xtask specialize qwen-mtp-check --artifact "$artifact" --reference "$reference" --tokens "$tokens" --output-tokens 8 --depth 4 --repetitions 1 --ptx "$ptx" --device 0 --output "$trial_dir/$name-$mode.json")
 systemd-run --user --scope --quiet --unit "splitk-sweep-$name-$mode-$(date +%s)" -p MemoryMax=8G -p MemorySwapMax=0 timeout --kill-after=15s 420 "${command[@]}" > "$trial_dir/$name-$mode.log" 2>&1
 jq -e '.all_passed == true and (.forced_rejections|length)==4' "$trial_dir/$name-$mode.json" >/dev/null
 done
done
