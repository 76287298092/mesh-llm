#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
umask 077
if test "$#" != 7; then
  printf 'usage: bash %s ABSOLUTE_NEW_TRIAL_DIR ABSOLUTE_FULL_PTX ABSOLUTE_FIXTURE ARTIFACT_SHA256 PTX_SHA256 EXECUTABLE_SHA256 FIXTURE_SHA256\n' "$0" >&2
  exit 2
fi
trial=$1 ptx=$2 fixture=$3 artifact_sha=$4 ptx_sha=$5 executable_sha=$6 fixture_sha=$7
for path in "$trial" "$ptx" "$fixture"; do
  [[ "$path" == /* && "$path" != *$'\n'* && "$path" != *$'\r'* ]] || exit 2
done
for hash in "$artifact_sha" "$ptx_sha" "$executable_sha" "$fixture_sha"; do
  [[ "$hash" =~ ^[0-9a-f]{64}$ ]] || exit 2
done
snapshot=/home/ndizazzo/dev/mesh/issue1393-q4-snapshot-20260930
xtask=$snapshot/target/release/xtask
artifact=/data/ai/ninfer/models/qwen3_8_27b_nvfp4.ninfer
exclusive=/home/ndizazzo/dev/mesh/mesh-llm/target/specialize/reassess-20260927/gpu-exclusive.sh
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
filter=$script_dir/target-batch-selected-20260930.jq
test -r "$ptx" && test -r "$fixture" && test -r "$artifact" && test -x "$xtask"
test -r "$exclusive" && test -r "$filter"
for tool in sha256sum jq nvidia-smi compute-sanitizer systemd-run systemctl timeout rg curl; do
  command -v "$tool" >/dev/null
done
mkdir -- "$trial"
cd -- "$trial"
trial=$PWD
exec > >(tee runner.log) 2>&1
printf '%s\n' "snapshot=$snapshot" "ptx=$ptx" "fixture=$fixture" "artifact=$artifact" \
  'scope=target-batch-decode-correctness-only' 'source-provenance=parent-owned pending snapshot' \
  'native-mtp-admitted=false' 'model-executable=false' 'timing-claim=false' \
  'device=0' 'prefix=32 verification=2..5 continuation=3 capacity=40' \
  'passes=racecheck' 'MemoryMax=32G MemorySwapMax=0 timeout=2400 kill-after=30' \
  'original-trialB-timeout=1200s preserved; retry uses a distinct new trial' \
  'gate=full exact words, all 64 layers, all 128 state regions, expected cursors; selected tokens insufficient' \
  'schema-assumption=20260930 PhaseReport/WordComparison/StateComparison; fail closed on changes' > scope.txt
printf '%s  %s\n' "$artifact_sha" "$artifact" "$ptx_sha" "$ptx" \
  "$executable_sha" "$xtask" "$fixture_sha" "$fixture" > expected-hashes.txt
sha256sum -- "$artifact" "$ptx" "$xtask" "$fixture" "$exclusive" "$filter" > hashes.txt
sha256sum --check --strict expected-hashes.txt > hash-check.log 2>&1
jq -e '.schema_version == 1 and .capacity == 40 and .target_vocabulary == 248320 and
  .native_mtp_admitted == false and .timing_claim == false and
  (.prefix | type == "array" and length == 32) and
  (.target_tokens | type == "array" and length == 5) and
  (.continuation | type == "array" and length == 3) and
  all(.prefix[], .target_tokens[], .continuation[]; type == "number" and . == floor and . >= 0 and . < 248320)' \
  "$fixture" > fixture-check.log
expected=$(jq -n --arg artifact_path "$artifact" --arg fixture_path "$fixture" --arg ptx_path "$ptx" \
  --arg artifact_sha256 "$artifact_sha" --arg fixture_sha256 "$fixture_sha" --arg ptx_sha256 "$ptx_sha" \
  '{artifact_path:$artifact_path,fixture_path:$fixture_path,ptx_path:$ptx_path,
    artifact_sha256:$artifact_sha256,fixture_sha256:$fixture_sha256,ptx_sha256:$ptx_sha256,device:0}')
for name in ${!MESH_SPECIALIZE_@}; do unset "$name"; done
{
  date --iso-8601=seconds
  uname -a
  compute-sanitizer --version
  systemd-run --version
  jq --version
  timeout --version
} > versions.log 2>&1
gpu0_uuid=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)
test "$gpu0_uuid" = GPU-80ded6bd-1a89-2628-3d94-902187dbab1d
source "$exclusive"
gpu_exclusive_begin > exclusive-begin.log 2>&1
gpu_idle() {
  local name=$1 code=0
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader > "$name-processes.csv"
  rg --fixed-strings "$gpu0_uuid" "$name-processes.csv" && return 1
  code=$?
  test "$code" = 1
}
gpu_idle admission
operation=("$xtask" specialize target-batch-decode-check --artifact "$artifact" --fixture "$fixture" --ptx "$ptx" --device 0)
failed=0
for rows in 2 3 4 5; do
  selected_expected=$(jq -n --argjson expected "$expected" --argjson rows "$rows" '$expected + {selected_rows:$rows}')
  for pass in racecheck; do
    run=rows-$rows-$pass
    code=0
    if ! gpu_idle "$run"; then
      printf '%s refused: GPU0 compute not idle\n' "$run" >> results.txt
      failed=1
      continue
    fi
    sha256sum --check --strict expected-hashes.txt > "$run-hash-check.log" 2>&1 || { failed=1; continue; }
    test ! -e "$trial/$run.json" || { failed=1; continue; }
    command=("${operation[@]}")
    if test "$pass" != normal; then
      flags=()
      if test "$pass" = racecheck; then
        flags=(--racecheck-num-workers 1 --force-synchronization-limit 1)
      fi
      command=(compute-sanitizer --tool "$pass" "${flags[@]}" --error-exitcode 97 "${operation[@]}")
    fi
    printf '%s start %s\n' "$(date --iso-8601=seconds)" "$run" >> runs.log
    printf '%q ' "${command[@]}" --output "$trial/$run.json" --rows "$rows" >> commands.log
    printf '\n' >> commands.log
    systemd-run --user --scope --quiet --unit "target-batch-$run-$$-$(date +%s)" \
      -p MemoryMax=32G -p MemorySwapMax=0 \
      timeout --kill-after=30s 2400 "${command[@]}" --output "$trial/$run.json" --rows "$rows" > "$run.log" 2>&1 || code=$?
    printf '%s end %s exit=%s\n' "$(date --iso-8601=seconds)" "$run" "$code" >> runs.log
    pass_failed=0
    test "$code" = 0 || pass_failed=1
    if test "$pass" != normal; then
      summary='^========= (ERROR|RACECHECK|SYNCCHECK) SUMMARY: [0-9]+ (errors?|hazards?)(.*)$'
      rg "$summary" "$run.log" > "$run-sanitizer-summary.log" || pass_failed=1
      rg -q '^========= (ERROR|RACECHECK|SYNCCHECK) SUMMARY: 0 (errors?|hazards?)(.*)$' "$run-sanitizer-summary.log" || pass_failed=1
      if rg -q '^========= (ERROR|RACECHECK|SYNCCHECK) SUMMARY: [1-9][0-9]* ' "$run-sanitizer-summary.log"; then pass_failed=1; fi
    fi
    jq -e --slurpfile fixture "$fixture" --argjson expected "$selected_expected" --argjson selected_rows "$rows" \
      -f "$filter" "$trial/$run.json" > "$run-assert.log" 2>&1 || pass_failed=1
    printf '%s exit=%s gate_failed=%s\n' "$run" "$code" "$pass_failed" >> results.txt
    if test "$pass_failed" != 0; then failed=1; fi
  done
done
printf 'final_failed=%s; evidence preserved in %s\n' "$failed" "$trial" >> results.txt
exit "$failed"
