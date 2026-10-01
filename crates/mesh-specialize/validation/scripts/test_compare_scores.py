"""Host tests for score comparison validity and gate scope (stdlib only)."""
import copy
import math
import json
import tempfile
import unittest
from pathlib import Path

import compare_scores as scores


def record(first=0):
    return (first, math.log(0.01), 5.0, *range(first, first + 64), *([math.log(0.01)] * 64))


def stream(mode='prefill'):
    return {
        "id": "s", "domain": "test", "input_tokens": 5, "scored_tokens": 4,
        "windows": [{"input_begin": 0, "input_end": 5, "target_begin": 1,
                     "target_end": 5, "scored_tokens": 4,
                     "prefill_context_rows": {"begin": 0, "end": 1 if mode == 'decode' else 5},
                     "prefill_scored_input_rows": {"begin": 0, "end": 1 if mode == 'decode' else 4},
                     "decode_scored_hidden_rows": {"begin": 1, "end": 4} if mode == 'decode' else {"begin": 0, "end": 0},
                     "decode_input_rows": {"begin": 1, "end": 4} if mode == 'decode' else {"begin": 5, "end": 5}}],
    }


def write_score_dir(path, extra=None):
    path.mkdir()
    item = dict(stream((extra or {}).get('score_mode', 'prefill')), records='s.scores.bin', input_tokens_sha256='input',
                full_logits_sha256='logits')
    manifest = {'all_passed': True, 'streams': [item],
                'corpus_id': 'test', 'context_tokens': 5, 'stride_tokens': 2,
                'artifact_sha256': 'artifact', 'identity': {'weights': 'same'},
                'profiles': {}, 'ptx_sha256': 'ptx', 'full_logit_hash': {'enabled': True}}
    manifest.update(extra or {})
    (path / 'manifest.json').write_text(json.dumps(manifest))
    (path / 's.scores.bin').write_bytes(scores.RECORD.pack(*record()) * 4)
    return path


class ScoreComparisonTests(unittest.TestCase):
    def test_score_mode_defaults_to_prefill_and_rejects_invalid_modes(self):
        self.assertEqual(scores.score_mode({}), 'prefill')
        self.assertEqual(scores.score_mode({'score_mode': 'decode', 'streams': [
            stream('decode'),
        ]}), 'decode')
        for value in (None, True, 'Decode', 'auto', 1):
            with self.subTest(value=value), self.assertRaisesRegex(SystemExit, 'score_mode'):
                scores.score_mode({'score_mode': value})

    def test_decode_score_mode_requires_reported_decode_rows(self):
        manifest = {'score_mode': 'decode', 'streams': [
            stream('decode'),
        ]}
        self.assertEqual(scores.score_mode(manifest), 'decode')
        with self.assertRaisesRegex(SystemExit, 'no decode-scored'):
            scores.score_mode({'score_mode': 'decode', 'streams': []})

    def test_execution_ranges_reject_malformed_bounds(self):
        for mode in ('prefill', 'decode'):
            for key in scores.EXECUTION_RANGES:
                for bad in (None, [], {}, {'begin': -1, 'end': 1},
                            {'begin': 2, 'end': 1}, {'begin': 0, 'end': 6},
                            {'begin': False, 'end': 1}, {'begin': 0, 'end': 1.0},
                            {'begin': '0', 'end': 1}):
                    with self.subTest(mode=mode, key=key, bad=bad):
                        item = stream(mode)
                        item['windows'][0][key] = bad
                        with self.assertRaisesRegex(SystemExit, key):
                            scores.score_mode({'score_mode': mode, 'streams': [item]})

    def test_execution_ranges_reject_bounded_but_incorrect_plan(self):
        for mode in ('prefill', 'decode'):
            for key in scores.EXECUTION_RANGES:
                with self.subTest(mode=mode, key=key):
                    item = stream(mode)
                    item['windows'][0][key] = {'begin': 0, 'end': 1}
                    if item['windows'][0][key] == stream(mode)['windows'][0][key]:
                        item['windows'][0][key] = {'begin': 0, 'end': 0}
                    with self.assertRaisesRegex(SystemExit, key):
                        scores.score_mode({'score_mode': mode, 'streams': [item]})

    def test_decode_ranges_accept_one_row_window_before_eligible_window(self):
        item = stream('decode')
        one_row = {'input_begin': 0, 'input_end': 2, 'target_begin': 1,
                   'target_end': 2, 'scored_tokens': 1,
                   'prefill_context_rows': {'begin': 0, 'end': 1},
                   'prefill_scored_input_rows': {'begin': 0, 'end': 1},
                   'decode_scored_hidden_rows': {'begin': 1, 'end': 1},
                   'decode_input_rows': {'begin': 1, 'end': 1}}
        item['windows'].insert(0, one_row)
        self.assertEqual(scores.score_mode({'score_mode': 'decode', 'streams': [item]}), 'decode')

    def test_internal_and_repeat_checks_reject_execution_range_mismatch(self):
        for key in scores.EXECUTION_RANGES:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                candidate_dir = write_score_dir(Path(tmp) / 'candidate', {'score_mode': 'decode'})
                other_dir = write_score_dir(Path(tmp) / 'other', {'score_mode': 'decode'})
                manifest, candidate = scores.load_dir(candidate_dir)
                other = json.loads((other_dir / 'manifest.json').read_text())
                other['streams'][0]['windows'][0][key]['end'] -= 1
                (other_dir / 'manifest.json').write_text(json.dumps(other))
                with self.assertRaisesRegex(SystemExit, key):
                    scores.compare_internal(candidate_dir, other_dir)
                with self.assertRaisesRegex(SystemExit, key):
                    scores.determinism(candidate, other_dir, manifest)
                with self.assertRaisesRegex(SystemExit, key):
                    scores.full_logit_determinism(candidate_dir, other_dir)

    def test_stream_protocol_requires_identical_reported_execution_ranges(self):
        for key in scores.EXECUTION_RANGES:
            with self.subTest(key=key):
                other = stream('decode')
                other['windows'][0][key]['end'] -= 1
                with self.assertRaisesRegex(SystemExit, 'execution ranges differ'):
                    scores.require_stream_protocol(stream('decode'), other)

    def test_forward_rows_defaults_to_legacy_schedule(self):
        self.assertEqual(scores.forward_rows({}), 512)
        scores.require_forward_schedule({}, {'forward_rows': 512})
        scores.require_forward_schedule({'forward_rows': 1}, {'forward_rows': 1})
        for value in range(1, 513):
            self.assertEqual(scores.forward_rows({'forward_rows': value}), value)

    def test_forward_rows_rejects_invalid_manifest_values(self):
        for value in (None, True, False, 0, -1, 513, 1.0, '1', [], {}):
            with self.subTest(value=value), self.assertRaisesRegex(SystemExit, 'forward_rows'):
                scores.forward_rows({'forward_rows': value})

    def test_internal_comparison_requires_same_schedule(self):
        for left, right, valid in (
            ({}, {}, True), ({}, {'forward_rows': 512}, True),
            ({'forward_rows': 1}, {'forward_rows': 1}, True),
            ({}, {'forward_rows': 1}, False),
            ({'forward_rows': 8}, {'forward_rows': 7}, False),
            ({'forward_rows': 1}, {'forward_rows': None}, False),
        ):
            with self.subTest(left=left, right=right), tempfile.TemporaryDirectory() as tmp:
                a = write_score_dir(Path(tmp) / 'a', left)
                b = write_score_dir(Path(tmp) / 'b', right)
                if valid:
                    overall, _, _, _ = scores.compare_internal(a, b)
                    self.assertEqual(overall['top1_agreement'], 1)
                else:
                    with self.assertRaisesRegex(SystemExit, 'forward_rows'):
                        scores.compare_internal(a, b)

    def test_internal_comparison_requires_same_score_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefill = write_score_dir(Path(tmp) / 'prefill')
            decode = write_score_dir(Path(tmp) / 'decode', {'score_mode': 'decode'})
            with self.assertRaisesRegex(SystemExit, 'score_mode'):
                scores.compare_internal(prefill, decode)

    def test_repeat_checks_reject_mismatched_schedule_even_with_equal_bytes_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = write_score_dir(Path(tmp) / 'a', {'forward_rows': 1})
            b = write_score_dir(Path(tmp) / 'b', {'forward_rows': 512})
            manifest, candidate = scores.load_dir(a)
            with self.assertRaisesRegex(SystemExit, 'forward_rows'):
                scores.determinism(candidate, b, manifest)
            with self.assertRaisesRegex(SystemExit, 'forward_rows'):
                scores.full_logit_determinism(a, b)

    def test_repeat_checks_reject_mismatched_score_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefill = write_score_dir(Path(tmp) / 'prefill')
            decode = write_score_dir(Path(tmp) / 'decode', {'score_mode': 'decode'})
            manifest, candidate = scores.load_dir(decode)
            with self.assertRaisesRegex(SystemExit, 'score_mode'):
                scores.determinism(candidate, prefill, manifest)
            with self.assertRaisesRegex(SystemExit, 'score_mode'):
                scores.full_logit_determinism(decode, prefill)

    def test_repeat_accepts_old_missing_schedule_as_512(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = write_score_dir(Path(tmp) / 'a')
            b = write_score_dir(Path(tmp) / 'b', {'forward_rows': 512})
            manifest, candidate = scores.load_dir(a)
            self.assertIs(scores.determinism(candidate, b, manifest), True)
            self.assertIs(scores.full_logit_determinism(a, b), True)


    def test_same_distribution_has_zero_kl(self):
        union, coarse = scores.position_kl(record(), record())
        self.assertAlmostEqual(union, 0.0)
        self.assertAlmostEqual(coarse, 0.0)

    def test_disjoint_support_exposes_coarse_bound_limitation(self):
        union, coarse = scores.position_kl(record(), record(64))
        self.assertGreater(union, 0.0)
        self.assertEqual(coarse, 0.0)
        # Top-1 is separate; a zero lower bound cannot establish full KL agreement.
        self.assertNotEqual(record()[3], record(64)[3])

    def test_invalid_probability_records_are_rejected(self):
        for bad in (float('nan'), float('inf'), 0.1):
            r = list(record())
            r[-1] = bad
            with self.assertRaises(SystemExit):
                scores.validate_record(r, 'fixture')

    def test_duplicate_ids_are_rejected(self):
        r = list(record())
        r[4] = r[3]
        with self.assertRaises(SystemExit):
            scores.validate_record(r, 'fixture')

    def test_probability_mass_above_one_is_rejected(self):
        r = list(record())
        r[67:] = [math.log(0.1)] * 64
        with self.assertRaises(SystemExit):
            scores.validate_record(r, 'fixture')

    def test_matching_protocol_passes(self):
        scores.require_stream_protocol(stream(), stream())

    def test_equal_counts_do_not_hide_changed_window(self):
        other = copy.deepcopy(stream())
        other['windows'][0]['input_begin'] = 1
        with self.assertRaises(SystemExit):
            scores.require_stream_protocol(stream(), other)

    def test_changed_domain_is_rejected(self):
        other = stream()
        other['domain'] = 'another'
        with self.assertRaises(SystemExit):
            scores.require_stream_protocol(stream(), other)

    def test_record_repeat_cannot_satisfy_full_logit_gate(self):
        summary = {'nll_relative_increase': 0, 'top1_agreement': 1,
                   'mean_kl': 0, 'p999_kl': 0}
        gates = scores.gate_rows(summary, {'test': summary}, True)
        self.assertEqual(next(g for g in gates if g['gate'] == 'Score-record repeatability')['status'], 'PASS')
        self.assertEqual(next(g for g in gates if g['gate'] == 'Full-logit determinism')['status'], 'NOT RUN')

    def test_full_hash_evidence_requires_equal_raw_logits(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / 'a', Path(tmp) / 'b'
            a.mkdir()
            b.mkdir()
            item = dict(stream(), input_tokens_sha256='input', full_logits_sha256='logits')
            manifest = {'streams': [item], 'full_logit_hash': {'enabled': True},
                        'corpus_id': 'test', 'context_tokens': 5, 'stride_tokens': 2,
                        'profiles': {}, 'artifact_sha256': 'artifact', 'ptx_sha256': 'ptx'}
            (a / 'manifest.json').write_text(json.dumps(manifest))
            (b / 'manifest.json').write_text(json.dumps(manifest))
            self.assertIs(scores.full_logit_determinism(a, b), True)
            item['full_logits_sha256'] = 'different'
            (b / 'manifest.json').write_text(json.dumps(manifest))
            self.assertIs(scores.full_logit_determinism(a, b), False)
            del item['full_logits_sha256']
            (b / 'manifest.json').write_text(json.dumps(manifest))
            self.assertIsNone(scores.full_logit_determinism(a, b))

    def test_input_hash_checks_unscored_first_token_identity(self):
        a = dict(stream(), input_tokens_sha256='first')
        b = dict(stream(), input_tokens_sha256='second')
        with self.assertRaises(SystemExit):
            scores.require_stream_protocol(a, b)

    def test_percentile_uses_nearest_rank(self):
        self.assertEqual(scores.nearest_rank(list(range(1000)), 0.999), 998)


if __name__ == '__main__':
    unittest.main()
