import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
reports={p.stem:json.loads(p.read_text()) for p in root.glob("python-*.json") if p.stat().st_size}
control=reports.get("python-off")
for name,d in sorted(reports.items()):
 trials=d["trials"]
 cross=all(t["tokens"]==control["control"]["tokens"] and t["target_state_sha256"]==control["control"]["state_sha256"] for t in trials) if control else None
 print(json.dumps({"name":name,"all_passed":d["all_passed"],"splits":d.get("fp8_split_k"),"cross_control_exact":cross,"tps":[t["decode_tokens_per_second"] for t in trials],"phase_seconds":[t["phase_seconds"] for t in trials],"profile":d["verification_profile"]}))
