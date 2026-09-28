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


def stream():
    return {
        "id": "s", "domain": "test", "input_tokens": 5, "scored_tokens": 4,
        "windows": [{"input_begin": 0, "input_end": 5, "target_begin": 1,
                     "target_end": 5, "scored_tokens": 4}],
    }


class ScoreComparisonTests(unittest.TestCase):
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
