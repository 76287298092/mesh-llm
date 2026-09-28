import hashlib, json, math, statistics, struct, sys
from pathlib import Path

def compare(control, candidate):
    assert len(control) == len(candidate) and control
    assert all(math.isfinite(x) for x in control + candidate)
    def distribution(values):
        maximum = max(values)
        exps = [math.exp(x-maximum) for x in values]
        total = math.fsum(exps)
        return [x/total for x in exps], maximum + math.log(total)
    p, logz = distribution(control)
    q, other_logz = distribution(candidate)
    top = control.index(max(control)); other_top = candidate.index(max(candidate))
    numerator = math.fsum((a-b)**2 for a,b in zip(control,candidate))
    denominator = math.fsum(a*a for a in control)
    return {
        'relative_l2':math.sqrt(numerator/denominator) if denominator else None,
        'max_absolute_error':max(abs(a-b) for a,b in zip(control,candidate)),
        'differing_values':sum(a!=b for a,b in zip(control,candidate)),
        'kl_control_to_candidate':math.fsum(prob*(a-b+other_logz-logz) for prob,a,b in zip(p,control,candidate)),
        'total_variation':math.fsum(abs(a-b) for a,b in zip(p,q))/2,
        'greedy_agreement':top==other_top, 'control_token':top, 'candidate_token':other_top,
        'control_top_probability':p[top], 'candidate_probability_of_control_top':q[top],
    }
assert compare([1.,2.],[4.,5.])['total_variation'] < 1e-14
assert compare([0.,1.],[1.,0.])['greedy_agreement'] is False
assert abs(compare([0.,1.],[1.,0.])['total_variation'] - math.tanh(.5)) < 1e-14

def load(root, name, mode, label):
    directory = root / f'logits-{name}-{mode}'
    manifest = json.loads((directory/'manifest.json').read_text())
    filename = label+'.bf16le'
    metadata = next(x for x in manifest['files'] if x['file']==filename)
    data=(directory/filename).read_bytes()
    assert hashlib.sha256(data).hexdigest()==metadata['sha256']
    words = [v[0] for v in struct.iter_unpack('<H',data)]
    assert len(words)==metadata['elements']
    return [struct.unpack('<f',struct.pack('<I',w<<16))[0] for w in words],manifest

root=Path(sys.argv[1]); summaries={}
for name in (sys.argv[2:] or ['python','explain']):
    control=json.loads((root/f'profile-{name}-exact.json').read_text())
    candidate=json.loads((root/f'profile-{name}-online.json').read_text())
    for key in ['identity','artifact_path','ptx_sha256','prefix_token_ids','teacher_forced_token','decode_input_token']:
        assert control[key]==candidate[key], (name,key)
    assert control['teacher_forced_token'] is not None
    for report in (control,candidate):
        assert all(report[k] is True for k in ('completed','all_passed','exact_output_and_state','exact_prefill_logits','memory_released'))
    assert control['attention_profile'] != candidate['attention_profile']
    summary={'identity':control['identity'],'teacher_token':control['teacher_forced_token'],'positions':{}}
    for label in ['whole-prefill','whole-decode']:
        exact,em=load(root,name,'exact',label); other,om=load(root,name,'online',label)
        assert em['prefix_token_ids']==om['prefix_token_ids'] and em['teacher_token']==om['teacher_token']
        summary['positions'][label]=compare(exact,other)
    for mode in ['exact','online']:
        bench=json.loads((root/f'bench-{name}-{mode}.json').read_text())
        assert bench['completed'] is True
        assert all(r['generated_token_ids']==bench['repetitions'][0]['generated_token_ids'] for r in bench['repetitions'])
        summary[mode]={
            'decode_median':statistics.median(r['decode_tokens_per_second'] for r in bench['repetitions']),
            'prefill_median':statistics.median(r['prefill_input_tokens_per_second'] for r in bench['repetitions']),
            'tokens':bench['repetitions'][0]['generated_token_ids'],
        }
    summary['generated_tokens_equal'] = summary['exact']['tokens']==summary['online']['tokens']
    summaries[name]=summary
(root/'comparison.json').write_text(json.dumps(summaries,indent=2)+'\n')
print(json.dumps(summaries,indent=2))
