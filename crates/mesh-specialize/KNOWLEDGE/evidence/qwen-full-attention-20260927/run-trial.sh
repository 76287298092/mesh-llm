#!/usr/bin/env bash
set -euo pipefail
cd /home/ndizazzo/dev/mesh/mesh-llm
trial_dir=target/specialize/qwen-full-attention-20260927
test "$(git rev-parse HEAD)" = cffc88757e5c928af7ece865ca00db63f1b36b48
test -z "$(git status --porcelain)"
test "$(sha256sum "$trial_dir/probes.ptx" | cut -d' ' -f1)" = 18b5ad2b5ff4834d374f8b0eb94e3dc68e41eb245890b7f7116249e4205d9e88
test "$(sha256sum target/release/xtask | cut -d' ' -f1)" = 6b5d7cad494165ef1ccedbdfcaba99a718e186193439940f8fe29ddb5bb7676d
restore_ninfer() {
  rc=$?
  trap - EXIT
  systemctl --user start ninfer-qwen38.service
  systemctl --user show ninfer-qwen38.service -p ActiveState -p MainPID -p ActiveEnterTimestamp > "$trial_dir/service-after.txt"
  exit "$rc"
}
trap restore_ninfer EXIT
systemctl --user stop ninfer-qwen38.service
test "$(systemctl --user show ninfer-qwen38.service -p ActiveState --value)" = inactive
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "$trial_dir/gpu-stopped.csv"
args=(specialize qwen-attention-check --artifact /data/ai/models/mesh-specialize/qwen3.8-27b-f0b7c9e7-raw-v1.mspec --ptx "$trial_dir/probes.ptx" --device 0)
systemd-run --user --scope --quiet --unit "mesh-full-attention-normal-$(date +%s)" -p MemoryMax=8G -p MemorySwapMax=0 timeout 240 target/release/xtask "${args[@]}" --output "$trial_dir/normal.json" > "$trial_dir/normal.log" 2>&1
jq -Mc '{all_passed,elapsed_seconds,kind,gate_resources}' "$trial_dir/normal.json"
for mode in memcheck racecheck synccheck; do
  systemd-run --user --scope --quiet --unit "mesh-full-attention-$mode-$(date +%s)" -p MemoryMax=8G -p MemorySwapMax=0 timeout 240 /opt/cuda/bin/compute-sanitizer --tool "$mode" --error-exitcode 97 --log-file "$trial_dir/$mode.log" target/release/xtask "${args[@]}" --output "$trial_dir/$mode.json" > "$trial_dir/$mode-stdout.log" 2>&1
  jq -Mc --arg mode "$mode" '{mode:$mode,all_passed,elapsed_seconds}' "$trial_dir/$mode.json"
  tail -2 "$trial_dir/$mode.log"
done

