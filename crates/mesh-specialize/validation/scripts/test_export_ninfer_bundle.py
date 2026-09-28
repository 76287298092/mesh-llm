"""Exporter tests use a fake read-only Artifact API, not a container parser."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import export_ninfer_bundle as export


class Object:
    def __init__(self, name, data):
        self.id, self.data, self.bytes = name, data, len(data)

    def to_json(self):
        return {'id': self.id, 'kind': 'tensor', 'format': 'bf16',
                'layout': 'contiguous_le_v1', 'shape': [self.bytes // 2],
                'offset': 9000, 'bytes': self.bytes}


class Artifact:
    def __init__(self):
        objects = [Object('weight/a', b'abcdef'), Object('weight/b', b'ghij'),
                   Object('auxiliary/a', b'1234'), Object('resource/tokenizer', b'{}'),
                   Object('vision/unused', b'xx')]
        self.by_id = {o.id: o for o in objects}
        self.reads = []
        self.directory = SimpleNamespace(
            components={'text': {'resources': {'tokenizer.json': 'resource/tokenizer'}},
                        'mtp': {}, 'vision': {}}, objects=objects,
            bindings={'text/token_embedding': {'object': 'weight/a'},
                      'text/output_head': {'object': 'weight/a'},
                      'proposal/head': {'object': 'weight/b'},
                      'proposal/token_ids': {'object': 'weight/b'},
                      'vision/unused': {'object': 'vision/unused'}},
            uses=[{'parameter': 'text/output_head', 'input': 'text/hidden',
                   'auxiliaries': {'divisor': {'object': 'auxiliary/a'}}},
                  {'parameter': 'text/output_head', 'input': 'dflash2/hidden',
                   'auxiliaries': {'unused': {'object': 'vision/unused'}}}],
        )

    def iter_object(self, name):
        self.reads.append(name)
        data = self.by_id[name].data
        yield data[:2]
        yield data[2:]


class ExportTests(unittest.TestCase):
    def test_reference_closure_includes_auxiliaries_and_resources(self):
        a = Artifact()
        components, _, uses, objects = export.export_selection(a.directory)
        self.assertEqual(set(components), {'text', 'mtp'})
        self.assertEqual(len(uses), 1)
        self.assertEqual({o.id for o in objects}, {'weight/a', 'weight/b', 'auxiliary/a', 'resource/tokenizer'})

    def test_preserves_parent_bytes_once_and_zero_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Artifact()
            path = Path(tmp) / 'export'
            m = export.write_bundle(a, path, export.SOURCE_SHA256, export.SOURCE_BYTES, export.READER_REVISION)
            payload = (path / 'payload.bin').read_bytes()
            previous = 0
            for item in m['objects']:
                self.assertEqual(item['offset'] % 256, 0)
                self.assertEqual(payload[previous:item['offset']], bytes(item['offset'] - previous))
                raw = payload[item['offset']:item['offset'] + item['bytes']]
                self.assertEqual(raw, a.by_id[item['source_id']].data)
                self.assertEqual(hashlib.sha256(raw).hexdigest(), item['sha256'])
                previous = item['offset'] + item['bytes']
            self.assertEqual(m['payload']['sha256'], hashlib.sha256(payload).hexdigest())
            self.assertEqual(len(a.reads), len(set(a.reads)))
            self.assertNotIn('vision/unused', a.reads)
            self.assertFalse(m['preservation']['requantized'])
            self.assertFalse(m['preservation']['runtime_executable'])

    def test_missing_required_component_is_rejected(self):
        a = Artifact()
        del a.directory.components['mtp']
        with self.assertRaises(ValueError):
            export.export_selection(a.directory)

    def test_missing_reference_is_rejected(self):
        a = Artifact()
        a.directory.bindings['text/new'] = {'parts': [{'object': 'absent', 'range': [0, 1]}]}
        with self.assertRaises(ValueError):
            export.export_selection(a.directory)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                export.write_bundle(Artifact(), Path(tmp), 'hash', 1, 'revision')

    def test_truncated_object_does_not_commit_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Artifact()
            a.by_id['weight/a'].bytes += 1
            path = Path(tmp) / 'export'
            with self.assertRaises(ValueError):
                export.write_bundle(a, path, 'hash', 1, 'revision')
            self.assertFalse((path / 'manifest.json').exists())

    def test_non_string_reference_rejected(self):
        with self.assertRaises(ValueError):
            list(export.object_refs({'object': 7}))


if __name__ == '__main__':
    unittest.main()
