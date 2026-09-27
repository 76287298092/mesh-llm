set -e
cd /home/ndizazzo/dev/mesh/mesh-llm
trial_dir=target/specialize/qwen-full-attention-20260927
test -z "$(git status --porcelain)"
git fetch --quiet origin codex/issue-1393-feasibility
git merge --quiet --ff-only origin/codex/issue-1393-feasibility
test "$(git rev-parse HEAD)" = 8d5528f046d6535346e7dffc1c48ede01fa1671b
just with-lld cargo test -p mesh-specialize --all-features > "$trial_dir/linux-tests-refined.log" 2>&1
just with-lld cargo clippy -p mesh-specialize --all-targets --all-features -- -D warnings > "$trial_dir/linux-clippy-refined.log" 2>&1
just specialize-tools-build > "$trial_dir/linux-build-refined.log" 2>&1
PATH="/opt/cuda/bin:$PATH" just specialize-sass "$trial_dir/probes-refined.ptx" > "$trial_dir/ptxas-refined.log" 2>&1
cp target/specialize/probes.cubin "$trial_dir/probes-refined.cubin"
{ git rev-parse HEAD; sha256sum target/release/xtask "$trial_dir/probes-refined.ptx"; } > "$trial_dir/source-refined.txt"
restore_ninfer() {
rc=$?
trap - EXIT
systemctl --user start ninfer-qwen38.service
systemctl --user show ninfer-qwen38.service -p ActiveState -p MainPID -p ActiveEnterTimestamp > "$trial_dir/service-after-refined.txt"
exit "$rc"
}
trap restore_ninfer EXIT
systemctl --user stop ninfer-qwen38.service
test "$(systemctl --user show ninfer-qwen38.service -p ActiveState --value)" = inactive
args=(specialize qwen-attention-check --artifact /data/ai/models/mesh-specialize/qwen3.8-27b-f0b7c9e7-raw-v1.mspec --ptx "$trial_dir/probes-refined.ptx" --device 0)
systemd-run --user --scope --quiet --unit "mesh-attention-refined-$(date +%s)" -p MemoryMax=8G -p MemorySwapMax=0 timeout 240 target/release/xtask "${args[@]}" --output "$trial_dir/refined.json" > "$trial_dir/refined.log" 2>&1
jq -Mc '{all_passed,elapsed_seconds,projection_resources,gate_resources,cases:[.cases[]|{rows:(.tokens|length),all_passed,l2:.finish.whole_layer_reference.hidden.aggregate.normalized_l2,max_token_l2:(.finish.whole_layer_reference.hidden.partitions|map(.normalized_l2)|max)}]}' "$trial_dir/refined.json"
for mode in memcheck racecheck synccheck; do
systemd-run --user --scope --quiet --unit "mesh-attention-refined-$mode-$(date +%s)" -p MemoryMax=8G -p MemorySwapMax=0 timeout 240 /opt/cuda/bin/compute-sanitizer --tool "$mode" --error-exitcode 97 --log-file "$trial_dir/refined-$mode.log" target/release/xtask "${args[@]}" --output "$trial_dir/refined-$mode.json" > "$trial_dir/refined-$mode-stdout.log" 2>&1
jq -Mc --arg mode "$mode" '{mode:$mode,all_passed,elapsed_seconds}' "$trial_dir/refined-$mode.json"
tail -2 "$trial_dir/refined-$mode.log"
done

