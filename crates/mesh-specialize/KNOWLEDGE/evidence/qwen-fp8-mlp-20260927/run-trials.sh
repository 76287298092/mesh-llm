#!/usr/bin/env bash
set -euo pipefail
cd /home/ndizazzo/dev/mesh/mesh-llm
trial_dir=target/specialize/qwen-fp8-mlp-20260927
test "$(git rev-parse HEAD)" = baf1cd881b6cbe754ef50cb861c827d5f8654344
test -z "$(git status --porcelain)"
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
sha256sum target/release/xtask "$trial_dir/probes.ptx" > "$trial_dir/trial-hashes.txt"
for mode in normal memcheck racecheck synccheck; do
  command=(target/release/xtask specialize qwen-fp8-mlp-check --artifact /data/ai/models/mesh-specialize/qwen3.8-27b-f0b7c9e7-raw-v1.mspec --ptx "$trial_dir/probes.ptx" --device 0 --output "$trial_dir/$mode.json")
  if test "$mode" != normal; then
    command=(/opt/cuda/bin/compute-sanitizer --tool "$mode" --error-exitcode 97 --log-file "$trial_dir/$mode-sanitizer.log" "${command[@]}")
  fi
  systemd-run --user --scope --quiet --unit "qwen-fp8-mlp-$mode-$(date +%s)" -p MemoryMax=8G -p MemorySwapMax=0 timeout 240 "${command[@]}" > "$trial_dir/$mode.log" 2>&1
  jq -M '{all_passed,elapsed_seconds,foreign_context_rejections,cases:[.cases[]|{prefix,rows,all_passed,down_differences:.down.bf16_differences}]}' "$trial_dir/$mode.json"
done
