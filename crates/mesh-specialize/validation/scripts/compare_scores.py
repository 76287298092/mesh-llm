#!/usr/bin/env python3
"""Compare teacher-forced score directories under KNOWLEDGE/findings/quality-gates.md.

Inputs are directories written by `xtask specialize qwen-model-score`: a
`manifest.json` plus one `<stream>.scores.bin` per stream of fixed 524-byte
little-endian records (u32 target, f32 target logprob, f32 logsumexp,
u32[64] top ids, f32[64] top logprobs).

Internal gate (control = exact profile P, candidate = fast profile Q):
  * mean NLL relative increase, overall <= 0.5%, each domain <= 1.0%
  * top-1 agreement over all scored positions >= 98.0%
  * mean KL(P || Q) <= 0.02 nats, 99.9th-percentile KL <= 1.0 nat
    (overall and per domain)
  * same-profile determinism: `--repeat DIR` must be byte-identical to the
    candidate (records are derived from logits; without it the row is NOT RUN)

Gated KL uses a common partition (quality-gates.md, clarified 2026-09-28): each
id in both top-64 lists is its own cell and one bucket holds every other id,
with each side's bucket mass equal to one minus its mass on the shared ids. By
the data-processing inequality this is a lower bound on full-vocabulary KL.
An informational union estimate is also reported: a union id outside a
profile's own top-64 is assigned min(p64, raw_tail / (n_unknown + 1)), the rest
of that profile's tail stays in the bucket. Percentiles use the
nearest-rank method. Probabilities are floored at 1e-30 inside logarithms.

`--ninfer report.json` compares per-stream and per-window scored-token counts
and NLL with a `ninfer-perplexity` schema-v2 report. It is reported, never a
gate; mismatched scored-token counts mark the comparison INVALID.

Standard library only.
"""

import argparse
import json
import math
import struct
import sys
from pathlib import Path

TOP_K = 64
RECORD = struct.Struct("<Iff64I64f")
FLOOR = 1e-30
THRESHOLDS = {
    "nll_overall": 0.005,
    "nll_domain": 0.010,
    "top1": 0.980,
    "mean_kl": 0.02,
    "p999_kl": 1.0,
}


def load_dir(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("all_passed") is not True:
        sys.exit(f"{path}: manifest is not all_passed")
    streams = []
    for stream in manifest["streams"]:
        raw = (path / stream["records"]).read_bytes()
        if len(raw) % RECORD.size:
            sys.exit(f"{path}: {stream['records']} is not a whole number of records")
        records = list(RECORD.iter_unpack(raw))
        if len(records) != stream["scored_tokens"]:
            sys.exit(f"{path}: {stream['id']} record count disagrees with manifest")
        streams.append((stream, records, raw))
    return manifest, streams


def distribution(record):
    ids = record[3:3 + TOP_K]
    probs = [math.exp(v) for v in record[3 + TOP_K:3 + 2 * TOP_K]]
    return dict(zip(ids, probs)), max(0.0, 1.0 - sum(probs)), min(probs)


def union_side(known, raw_tail, p64, support):
    unknown = [i for i in support if i not in known]
    fill = min(p64, raw_tail / (len(unknown) + 1)) if unknown else 0.0
    probs = [known.get(i, fill) for i in support]
    return probs, max(0.0, raw_tail - fill * len(unknown))


def kl_terms(p, q):
    return sum(pi * (math.log(max(pi, FLOOR)) - math.log(max(qi, FLOOR))) for pi, qi in zip(p, q) if pi > 0)


def position_kl(control, candidate):
    p_known, p_tail, p64 = distribution(control)
    q_known, q_tail, q64 = distribution(candidate)
    support = sorted(set(p_known) | set(q_known))
    p, p_rest = union_side(p_known, p_tail, p64, support)
    q, q_rest = union_side(q_known, q_tail, q64, support)
    union = kl_terms(p + [p_rest], q + [q_rest])
    common = sorted(set(p_known) & set(q_known))
    pc = [p_known[i] for i in common]
    qc = [q_known[i] for i in common]
    coarse = kl_terms(pc + [max(0.0, 1.0 - sum(pc))], qc + [max(0.0, 1.0 - sum(qc))])
    return max(0.0, union), max(0.0, coarse)


def nearest_rank(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def summarize(rows):
    n = len(rows)
    control_nll = sum(r["control_nll"] for r in rows) / n
    candidate_nll = sum(r["candidate_nll"] for r in rows) / n
    kls = [r["coarse_kl"] for r in rows]
    union = [r["kl"] for r in rows]
    return {
        "scored_tokens": n,
        "control_mean_nll": control_nll,
        "candidate_mean_nll": candidate_nll,
        "nll_relative_increase": (candidate_nll - control_nll) / control_nll,
        "top1_agreement": sum(r["top1"] for r in rows) / n,
        "mean_kl": sum(kls) / n,
        "p999_kl": nearest_rank(kls, 0.999),
        "max_kl": max(kls),
        "mean_union_estimate_kl": sum(union) / n,
        "p999_union_estimate_kl": nearest_rank(union, 0.999),
    }


def compare_internal(control_dir, candidate_dir):
    control_manifest, control = load_dir(control_dir)
    candidate_manifest, candidate = load_dir(candidate_dir)
    for key in ("corpus_id", "context_tokens", "stride_tokens"):
        if control_manifest[key] != candidate_manifest[key]:
            sys.exit(f"{key} differs between control and candidate")
    if [s["id"] for s, _, _ in control] != [s["id"] for s, _, _ in candidate]:
        sys.exit("stream lists differ between control and candidate")
    rows_by_domain = {}
    for (stream, p_records, _), (_, q_records, _) in zip(control, candidate):
        if len(p_records) != len(q_records):
            sys.exit(f"{stream['id']}: scored counts differ")
        rows = rows_by_domain.setdefault(stream["domain"], [])
        for position, (p, q) in enumerate(zip(p_records, q_records)):
            if p[0] != q[0]:
                sys.exit(f"{stream['id']} position {position}: target ids differ")
            kl, coarse = position_kl(p, q)
            rows.append({
                "control_nll": -p[1],
                "candidate_nll": -q[1],
                "top1": p[3] == q[3],
                "kl": kl,
                "coarse_kl": coarse,
            })
    everything = [row for rows in rows_by_domain.values() for row in rows]
    return (
        summarize(everything),
        {domain: summarize(rows) for domain, rows in sorted(rows_by_domain.items())},
        {"control": control_manifest, "candidate": candidate_manifest},
        candidate,
    )


def determinism(candidate, repeat_dir):
    if repeat_dir is None:
        return None
    _, repeat = load_dir(repeat_dir)
    if [s["id"] for s, _, _ in candidate] != [s["id"] for s, _, _ in repeat]:
        return False
    return all(a == b for (_, _, a), (_, _, b) in zip(candidate, repeat))


def gate_rows(overall, domains, identical):
    rows = [("NLL increase overall", overall["nll_relative_increase"], "<=", THRESHOLDS["nll_overall"])]
    rows += [(f"NLL increase {d}", s["nll_relative_increase"], "<=", THRESHOLDS["nll_domain"]) for d, s in domains.items()]
    rows.append(("Top-1 agreement overall", overall["top1_agreement"], ">=", THRESHOLDS["top1"]))
    for scope, summary in [("overall", overall)] + list(domains.items()):
        rows.append((f"Mean KL {scope}", summary["mean_kl"], "<=", THRESHOLDS["mean_kl"]))
        rows.append((f"99.9th pct KL {scope}", summary["p999_kl"], "<=", THRESHOLDS["p999_kl"]))
    result = []
    for name, value, op, limit in rows:
        passed = value <= limit if op == "<=" else value >= limit
        result.append({"gate": name, "value": value, "op": op, "threshold": limit,
                       "status": "PASS" if passed else "FAIL"})
    result.append({"gate": "Same-profile determinism", "value": identical, "op": "==",
                   "threshold": True,
                   "status": "NOT RUN" if identical is None else ("PASS" if identical else "FAIL")})
    return result


def compare_ninfer(manifest, report_path):
    report = json.loads(Path(report_path).read_text())
    theirs = {s["id"]: s for s in report["streams"]}
    execution = report.get("execution", {})
    protocol_match = (execution.get("context_tokens") == manifest["context_tokens"]
                      and execution.get("stride_tokens") == manifest["stride_tokens"])
    streams, valid = [], protocol_match
    for ours in manifest["streams"]:
        other = theirs.get(ours["id"])
        if other is None:
            streams.append({"id": ours["id"], "status": "MISSING"})
            valid = False
            continue
        counts_match = ours["scored_tokens"] == other["scored_tokens"]
        windows = []
        for mine, their in zip(ours["windows"], other["windows"]):
            same = all(mine[k] == their[k] for k in ("input_begin", "input_end", "target_begin", "target_end"))
            windows.append({"index": mine["index"], "bounds_match": same,
                            "scored_match": mine["scored_tokens"] == their["scored_tokens"],
                            "nll_difference": mine["mean_nll"] - their["mean_nll"]})
        windows_match = len(ours["windows"]) == len(other["windows"]) and all(
            w["bounds_match"] and w["scored_match"] for w in windows)
        valid = valid and counts_match and windows_match
        streams.append({
            "id": ours["id"], "domain": ours["domain"],
            "scored_tokens": ours["scored_tokens"], "ninfer_scored_tokens": other["scored_tokens"],
            "mean_nll": ours["mean_nll"], "ninfer_mean_nll": other["mean_nll"],
            "nll_difference": ours["mean_nll"] - other["mean_nll"],
            "max_abs_window_nll_difference": max((abs(w["nll_difference"]) for w in windows), default=None),
            "windows_match": windows_match, "windows": windows,
        })
    ours_total = manifest["overall"]
    return {
        "status": "VALID" if valid else "INVALID",
        "protocol_match": protocol_match,
        "ninfer_kv_dtype": execution.get("kv_dtype"),
        "overall": {"mean_nll": ours_total["mean_nll"], "ninfer_mean_nll": report["overall"]["mean_nll"],
                    "nll_difference": ours_total["mean_nll"] - report["overall"]["mean_nll"]},
        "streams": streams,
        "note": "Different NVFP4 checkpoints and runtimes; reported, not a gate.",
    }


def print_report(result):
    if "gates" in result:
        print(f"{'gate':<40}{'value':>16}  {'threshold':>14}  status")
        for row in result["gates"]:
            value = row["value"]
            text = f"{value:.6g}" if isinstance(value, float) else str(value)
            print(f"{row['gate']:<40}{text:>16}  {row['op']} {row['threshold']!s:>11}  {row['status']}")
        print(f"verdict: {result['verdict']}")
    for label, ninfer in result.get("ninfer", {}).items():
        print(f"\nninfer vs {label}: {ninfer['status']} (kv {ninfer['ninfer_kv_dtype']})")
        print(f"{'stream':<16}{'tokens':>8}{'ninfer':>8}{'nll':>12}{'ninfer':>12}{'diff':>12}{'max win':>12}")
        for s in ninfer["streams"]:
            if s.get("status") == "MISSING":
                print(f"{s['id']:<16} MISSING")
                continue
            print(f"{s['id']:<16}{s['scored_tokens']:>8}{s['ninfer_scored_tokens']:>8}"
                  f"{s['mean_nll']:>12.6f}{s['ninfer_mean_nll']:>12.6f}{s['nll_difference']:>12.6f}"
                  f"{s['max_abs_window_nll_difference'] or 0.0:>12.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--control", required=True, help="exact-profile score directory")
    parser.add_argument("--candidate", help="fast-profile score directory")
    parser.add_argument("--repeat", help="second run of the candidate for the determinism row")
    parser.add_argument("--ninfer", help="ninfer-perplexity report.json at matched context/stride")
    parser.add_argument("--output", required=True, help="new JSON result file")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        sys.exit(f"refusing to overwrite {output}")
    result = {"schema_version": 1, "thresholds": THRESHOLDS}
    control_manifest, _ = load_dir(args.control)
    if args.candidate:
        overall, domains, manifests, candidate = compare_internal(args.control, args.candidate)
        identical = determinism(candidate, args.repeat)
        gates = gate_rows(overall, domains, identical)
        statuses = {row["status"] for row in gates}
        result.update({
            "control": args.control, "candidate": args.candidate, "repeat": args.repeat,
            "control_profiles": manifests["control"].get("profiles"),
            "candidate_profiles": manifests["candidate"].get("profiles"),
            "overall": overall, "domains": domains, "gates": gates,
            "verdict": "FAIL" if "FAIL" in statuses else ("INCOMPLETE" if "NOT RUN" in statuses else "PASS"),
        })
    if args.ninfer:
        result["ninfer"] = {"control": compare_ninfer(control_manifest, args.ninfer)}
        if args.candidate:
            result["ninfer"]["candidate"] = compare_ninfer(manifests["candidate"], args.ninfer)
    print_report(result)
    with output.open("x") as handle:
        json.dump(result, handle, indent=1)
        handle.write("\n")


if __name__ == "__main__":
    main()
