#!/usr/bin/env python3
"""Write STREAMS_JSON for `xtask specialize qwen-model-score` from a corpus manifest.

The manifest is Ninfer-style (`{"corpus_id", "streams": [{"id", "domain",
"path"}], "modes": {...}}`, paths relative to the manifest directory), for
example `target/specialize/reassess-20260927/ppl-slice/manifest.json`. Text is
encoded with the artifact's `tokenizer.json` and `add_special_tokens=False`,
matching how the slice token counts were produced. `--expect-counts` checks
the result against `token-counts.json`.

Requires the Python `tokenizers` package (the same one used to build the slice).
"""

import argparse
import json
import sys
from pathlib import Path

import tokenizers


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="corpus manifest.json")
    parser.add_argument("--tokenizer", required=True, help="artifact tokenizer.json")
    parser.add_argument("--mode", default="full", help="manifest mode selecting streams (default full)")
    parser.add_argument("--expect-counts", help="token-counts.json to verify against")
    parser.add_argument("--output", required=True, help="new STREAMS_JSON file")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        sys.exit(f"refusing to overwrite {output}")
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    selected = manifest.get("modes", {}).get(args.mode)
    if selected is None:
        sys.exit(f"manifest has no mode {args.mode!r}")
    by_id = {stream["id"]: stream for stream in manifest["streams"]}
    tokenizer = tokenizers.Tokenizer.from_file(args.tokenizer)
    streams, counts = [], {}
    for stream_id in selected:
        source = by_id[stream_id]
        text = (manifest_path.parent / source["path"]).read_text(encoding="utf-8")
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if len(ids) < 2:
            sys.exit(f"{stream_id} tokenized to fewer than two tokens")
        streams.append({"id": stream_id, "domain": source["domain"], "tokens": ids})
        counts[stream_id] = len(ids)
    if args.expect_counts:
        expected = json.loads(Path(args.expect_counts).read_text())
        mismatched = {k: (counts.get(k), v) for k, v in expected.items() if counts.get(k) != v}
        if mismatched:
            sys.exit(f"token counts differ from {args.expect_counts}: {mismatched}")
    with output.open("x") as handle:
        json.dump({"corpus_id": manifest["corpus_id"], "streams": streams}, handle)
        handle.write("\n")
    print(json.dumps({"output": str(output), "corpus_id": manifest["corpus_id"], "token_counts": counts}))


if __name__ == "__main__":
    main()
