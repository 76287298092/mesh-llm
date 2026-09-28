"""Compare extracted Ninfer tokenizer/template assets on this campaign's inputs.

No container parsing: ninfer-resources was extracted by Ninfer's own reader.
This checks asset-level rendering/encoding, not the server's internal token buffer.
"""
import hashlib
import json
from pathlib import Path

import jinja2
import tokenizers

ROOT = Path(__file__).resolve().parent


def digest(ids):
    return hashlib.sha256(b''.join(t.to_bytes(4, 'little') for t in ids)).hexdigest()


def fail(message):
    raise ValueError(message)


def main():
    old = tokenizers.Tokenizer.from_file(str(ROOT.parent / 'tokenizer.json'))
    new = tokenizers.Tokenizer.from_file(str(ROOT / 'ninfer-resources/tokenizer.json'))
    env = jinja2.Environment()
    env.globals['raise_exception'] = fail
    template = env.from_string((ROOT / 'ninfer-resources/chat_template.jinja').read_text())
    fixtures = json.loads((ROOT / 'matched-prompts.json').read_text())['prompts']
    results = []
    for p in fixtures:
        rendered = template.render(messages=[{'role': 'user', 'content': p['user_text']}],
                                   add_generation_prompt=True, enable_thinking=False)
        ids = new.encode(rendered, add_special_tokens=False).ids
        results.append({'id': p['name'], 'rendered_text_equal': rendered == p['rendered_prompt'],
                        'token_ids_equal': ids == p['tokens'], 'ninfer_tokens': len(ids),
                        'mesh_tokens': len(p['tokens']), 'ninfer_ids_sha256': digest(ids),
                        'mesh_ids_sha256': digest(p['tokens'])})
    manifest = json.loads((ROOT / 'ppl-slice/manifest.json').read_text())
    for s in manifest['streams']:
        text = (ROOT / 'ppl-slice' / s['path']).read_text()
        a = old.encode(text, add_special_tokens=False).ids
        b = new.encode(text, add_special_tokens=False).ids
        results.append({'id': s['id'], 'token_ids_equal': a == b,
                        'mesh_tokens': len(a), 'ninfer_tokens': len(b),
                        'mesh_ids_sha256': digest(a), 'ninfer_ids_sha256': digest(b)})
    report = {'scope': 'Asset-level token IDs; server-internal IDs not observed',
              'tokenizers_version': tokenizers.__version__, 'jinja_version': jinja2.__version__,
              'all_ids_equal': all(r['token_ids_equal'] for r in results), 'cases': results}
    with (ROOT / 'token-assets-comparison.json').open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
