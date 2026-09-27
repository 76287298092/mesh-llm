set -e
cd /home/ndizazzo/dev/mesh/mesh-llm
trial_dir=target/specialize/qwen-full-attention-20260927
test -z "$(git status --porcelain)"
git fetch --quiet origin codex/issue-1393-feasibility
git merge --quiet --ff-only origin/codex/issue-1393-feasibility
test "$(git rev-parse HEAD)" = 4ad9df826cf8edcf328135a23fb27b0f634d4b36
just with-lld cargo test -p mesh-specialize --all-features > "$trial_dir/linux-tests-wide.log" 2>&1
just with-lld cargo clippy -p mesh-specialize --all-targets --all-features -- -D warnings > "$trial_dir/linux-clippy-wide.log" 2>&1
just specialize-tools-build > "$trial_dir/linux-build-wide.log" 2>&1
PATH="/opt/cuda/bin:$PATH" just specialize-sass "$trial_dir/probes-wide.ptx" > "$trial_dir/ptxas-wide.log" 2>&1
cp target/specialize/probes.cubin "$trial_dir/probes-wide.cubin"
{ git rev-parse HEAD; sha256sum target/release/xtask "$trial_dir/probes-wide.ptx"; } > "$trial_dir/source-wide.txt"
restore_ninfer() {
rc=$?
trap - EXIT
systemctl --user start ninfer-qwen38.service
systemctl --user show ninfer-qwen38.service -p ActiveState -p MainPID -p ActiveEnterTimestamp > "$trial_dir/service-after-wide.txt"
exit "$rc"
}
trap restore_ninfer EXIT
systemctl --user stop ninfer-qwen38.service
test "$(systemctl --user show ninfer-qwen38.service -p ActiveState --value)" = inactive
args=(specialize qwen-attention-check --artifact /data/ai/models/mesh-specialize/qwen3.8-27b-f0b7c9e7-raw-v1.mspec --ptx "$trial_dir/probes-wide.ptx" --device 0)
systemd-run --user --scope --quiet --unit "mesh-attention-wide-$(date +%s)" -p MemoryMax=8G -p MemorySwapMax=0 timeout 240 target/release/xtask "${args[@]}" --output "$trial_dir/wide.json" > "$trial_dir/wide.log" 2>&1
jq -Mc '{all_passed,elapsed_seconds,projection_resources,gate_resources,cases:[.cases[]|{rows:(.tokens|length),all_passed,l2:.finish.whole_layer_reference.hidden.aggregate.normalized_l2,max_token_l2:(.finish.whole_layer_reference.hidden.partitions|map(.normalized_l2)|max)}]}' "$trial_dir/wide.json"
for mode in memcheck racecheck synccheck; do
systemd-run --user --scope --quiet --unit "mesh-attention-wide-$mode-$(date +%s)" -p MemoryMax=8G -p MemorySwapMax=0 timeout 240 /opt/cuda/bin/compute-sanitizer --tool "$mode" --error-exitcode 97 --log-file "$trial_dir/wide-$mode.log" target/release/xtask "${args[@]}" --output "$trial_dir/wide-$mode.json" > "$trial_dir/wide-$mode-stdout.log" 2>&1
jq -Mc --arg mode "$mode" '{mode:$mode,all_passed,elapsed_seconds}' "$trial_dir/wide-$mode.json"
tail -2 "$trial_dir/wide-$mode.log"
done

