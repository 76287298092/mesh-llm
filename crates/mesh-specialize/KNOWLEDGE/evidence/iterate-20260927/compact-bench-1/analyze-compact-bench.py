import json, statistics, sys
from pathlib import Path
root = Path(sys.argv[1])
summary = {}
for prompt in ['python', 'explain']:
    summary[prompt] = {}
    for mode in ['full-forward', 'compact']:
        data = json.loads((root / f'{prompt}-{mode}.json').read_text())
        assert data['all_passed'] and len(data['trials']) == 3
        assert len(data['forced_rejections']) == 4
        trials = data['trials']
        summary[prompt][mode] = {
            'decode_tokens_per_second_median': statistics.median(t['decode_tokens_per_second'] for t in trials),
            'decode_tokens_per_second_range': [min(t['decode_tokens_per_second'] for t in trials), max(t['decode_tokens_per_second'] for t in trials)],
            'accepted_fraction': [t['accepted_fraction'] for t in trials],
            'phase_seconds_median': {phase: statistics.median(t['phase_seconds'][phase] for t in trials) for phase in ['draft','verification','replay','teacher']},
            'tokens': trials[0]['tokens'],
            'target_state_sha256': trials[0]['target_state_sha256'],
        }
    assert summary[prompt]['compact']['tokens'] == summary[prompt]['full-forward']['tokens']
    assert summary[prompt]['compact']['target_state_sha256'] == summary[prompt]['full-forward']['target_state_sha256']
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
