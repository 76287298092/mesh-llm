def sha: type == "string" and test("^[0-9a-f]{64}$");
def words($count):
  .passed == true and .expected_words == $count and
  .left_words == $count and .right_words == $count and .compared_words == $count and
  .differing_words == 0 and .first_mismatch_index == null and
  .first_left_word == null and .first_right_word == null and
  has("first_mismatch_index") and has("first_left_word") and has("first_right_word");
def region_manifest:
  [range(0;64) as $layer |
    ("layers." + (if $layer < 10 then "0" else "" end) + ($layer|tostring)) as $base |
    if $layer % 4 == 3 then
      {name: ($base + ".attention.k"), length: 81920},
      {name: ($base + ".attention.v"), length: 81920}
    else
      {name: ($base + ".gdn.history"), length: 61440},
      {name: ($base + ".gdn.recurrent"), length: 3145728}
    end] | sort_by(.name);
def state:
  .equal == true and .first_differing_region == null and has("first_differing_region") and
  (.left_aggregate_sha256 | sha) and .left_aggregate_sha256 == .right_aggregate_sha256 and
  (.regions | type == "array" and length == 128) and
  ([.regions[] | {name,length}] | sort_by(.name)) == region_manifest and
  all(.regions[]; .identical == true and .differing_bytes == 0 and
    has("first_difference") and .first_difference == null and
    (.left_sha256 | sha) and .left_sha256 == .right_sha256);
def session($past): . == {past: $past, capacity: 40, poisoned: false};
def phase($name; $rows; $past; $path):
  .phase == $name and .batch_path == $path and
  .ordinary_decode_path == "forward_detailed_decode_record_false_decode_true" and
  .passed == true and .batch_completed == true and .ordinary_decode_completed == true and
  (.batch_recovery_records | type == "number" and . == floor and . >= 0) and
  (.ordinary_decode_recovery_records | type == "number" and . == floor and . >= 0) and
  has("batch_error") and .batch_error == null and
  has("ordinary_decode_error") and .ordinary_decode_error == null and
  has("compare_error") and .compare_error == null and
  .compared_layers == 64 and has("first_differing_layer") and .first_differing_layer == null and
  (.layers | type == "array" and length == 64) and
  [.layers[].layer] == [range(0;64)] and
  all(.layers[]; .words | words($rows * 5120)) and
  (.hidden | words($rows * 5120)) and (.logits | words($rows * 248320)) and
  (.state | state) and .cursor.equal == true and
  (.cursor.left | session($past)) and (.cursor.right | session($past)) and
  (.batch_session | session($past)) and (.ordinary_decode_session | session($past));
($fixture | length) == 1 and
($selected_rows | type == "number" and . == floor and . >= 1 and . <= 5) and
.schema_version == 1 and .kind == "resident-target-batch-decode-correctness" and
.all_passed == true and .native_mtp_admitted == false and .timing_claim == false and
(if has("model_executable") then .model_executable == false else true end) and
.batch_path == "forward_recorded_record_true_decode_false" and
.ordinary_decode_path == "forward_detailed_decode_record_false_decode_true" and
.prefix_tokens == 32 and .target_tokens == $fixture[0].target_tokens and
.continuation_tokens == $fixture[0].continuation and .request == $expected and
.selected_rows == $selected_rows and .request.selected_rows == $selected_rows and
all(.. | objects | to_entries[] | select(.key == "error" or (.key | endswith("_error"))); .value == null) and
(.cases | type == "array" and length == 1) and [.cases[].rows] == [$selected_rows] and
all(.cases[]; .rows as $rows | .passed == true and
  (.verification | phase("verification"; $rows; 32 + $rows; "forward_recorded_record_true_decode_false")) and
  (.continuation | phase("continuation"; 3; 35 + $rows; "forward_detailed_decode_record_false_decode_true")))
