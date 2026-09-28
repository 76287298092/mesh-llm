import json,hashlib,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]); result={}
for name in ['128','512']:
    labels=['baseline','tiled-original','tiled-fixed']
    reports={label:json.loads((root/f'profile-{name}-{label}.json').read_text()) for label in labels}
    base=reports['baseline']; record={}
    for label,report in reports.items():
        assert all(report[key] is True for key in ['completed','all_passed','exact_output_and_state','exact_prefill_logits','whole_vs_token_partition_exact','memory_released']), (name,label,'profile gate')
        for key in ['identity','artifact_path','prefix_token_ids','teacher_forced_token','decode_input_token','attention_profile','arithmetic_profile','mlp_workspace']:
            assert base[key]==report[key],(name,label,key)
        assert base['control_state_sha256']==report['control_state_sha256'], (name,label,'state')
        path=root/f'logits-{name}-{label}'; manifest=json.loads((path/'manifest.json').read_text())
        expected=json.loads((root/f'logits-{name}-baseline'/'manifest.json').read_text())
        checks={}
        for f in manifest['files']:
            data=(path/f['file']).read_bytes()
            assert hashlib.sha256(data).hexdigest()==f['sha256']
            control=next(v for v in expected['files'] if v['file']==f['file'])
            checks[f['file']]=f['sha256']==control['sha256']
        assert all(checks.values()),(name,label,'logit bytes')
        bench=json.loads((root/f'bench-{name}-{label}.json').read_text())
        control_bench=json.loads((root/f'bench-{name}-baseline.json').read_text())
        assert bench['completed'] and bench['memory']['arena_memory_release_observed']
        assert all(r['generated_token_ids']==control_bench['repetitions'][0]['generated_token_ids'] for r in bench['repetitions'])
        record[label]={'prefill_median':statistics.median(r['prefill_input_tokens_per_second'] for r in bench['repetitions']),'prefill_samples':[r['prefill_input_tokens_per_second'] for r in bench['repetitions']],'decode_median':statistics.median(r['decode_tokens_per_second'] for r in bench['repetitions']),'logit_bytes_exact':checks,'same_input_state_exact':True,'generated_tokens_exact':True,'ptx_sha256':bench['ptx_sha256']}
    result[name]=record
(root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
