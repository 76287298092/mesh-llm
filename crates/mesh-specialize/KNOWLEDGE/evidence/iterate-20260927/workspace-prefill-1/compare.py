import json, statistics, sys
from pathlib import Path
r = Path(sys.argv[1])
rows=[]
for name in ["128","512"]:
    bench={mode:json.loads((r/f"bench-{name}-{mode}.json").read_text()) for mode in ["off","on"]}
    profiles={mode:json.loads((r/f"profile-{name}-{mode}.json").read_text()) for mode in ["off","on"]}
    assert all(p["all_passed"] for p in profiles.values())
    assert all(p["prefix_token_ids"] == profiles["off"]["prefix_token_ids"] and p["decode_input_token"] == profiles["off"]["decode_input_token"] for p in profiles.values())
    assert profiles["off"]["control_state_sha256"] == profiles["on"]["control_state_sha256"]
    assert profiles["off"]["logit_dump"]["manifest"]["files"] == profiles["on"]["logit_dump"]["manifest"]["files"]
    tokens=[p["generated_token_ids"] for p in bench["off"]["repetitions"]]
    assert all(t==tokens[0] for t in tokens)
    assert tokens == [p["generated_token_ids"] for p in bench["on"]["repetitions"]]
    rates={mode:statistics.median(p["prefill_input_tokens_per_second"] for p in report["repetitions"]) for mode,report in bench.items()}
    rows.append(dict(prompt=name,prefill_input_tokens_per_second=rates,improvement_percent=(rates["on"]/rates["off"]-1)*100,generated_tokens_exact=True,logits_exact=True,state_exact=True))
print(json.dumps(dict(cases=rows,scope="Shared GPU, 8 output tokens, three repetitions per mode; fixed off/on order. Retained raw-token prefill fixtures, not Ninfer parity."),indent=2))
