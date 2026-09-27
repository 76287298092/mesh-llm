from pathlib import Path
import json, statistics
root=Path('target/specialize/perf-20260927')
rows=[]
for name in ['fp8-exact-round','bf16-round','word-round','warp4-round','reuse4-round','inline-round']:
 r=root/name
 if not (r/'bench-128.json').exists() or not (r/'bench-128.json').stat().st_size:continue
 row={'trial':name, 'source':(r/'source-head.txt').read_text().strip()}
 for size in ['two','128']:
  j=json.loads((r/f'bench-{size}.json').read_text())
  assert j['completed'] and len(j['repetitions'])==3
  expected=[271]*8 if size=='two' else [98094,6013]*4
  assert all(s['generated_token_ids']==expected for s in j['repetitions'])
  row[size]={k:{'median':statistics.median(s[k] for s in j['repetitions']), 'min':min(s[k] for s in j['repetitions']), 'max':max(s[k] for s in j['repetitions'])} for k in ['prefill_input_tokens_per_second','decode_tokens_per_second']}
 rows.append(row)
final=root/'reuse4-round'
before={}
for size in ['two','128']:
 p=final/f'before-bench-{size}.json'
 if p.exists() and p.stat().st_size:
  j=json.loads(p.read_text()); assert j['completed']
  before[size]={k:statistics.median(s[k] for s in j['repetitions']) for k in ['prefill_input_tokens_per_second','decode_tokens_per_second']}
  expected=[271]*8 if size=='two' else [98094,6013]*4
  assert all(s['generated_token_ids']==expected for s in j['repetitions'])
result={'before_medians':before, 'iterations':rows}
(root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
