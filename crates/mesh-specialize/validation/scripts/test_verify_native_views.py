"""CPU-only, tagged-byte oracle tests. No Ninfer file parser or runtime required."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import verify_native_views as verifier


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'KNOWLEDGE/evidence/reassess-20260928/ninfer-identity/ninfer-artifact-inspect.json'


def tensor(name, shape, fmt, layout, size):
    return {'id': name, 'kind': 'tensor', 'shape': shape,
            'format': fmt, 'layout': layout, 'bytes': size}


def child(name, start, end):
    return {'parts': [{'object': name, 'range': [start, end]}]}


def observed_directory():
    with FIXTURE.open() as stream:
        data = json.load(stream)
    return SimpleNamespace(components=data['components'], bindings=data['binding_records'],
                           uses=data['use_records'], files=[None],
                           objects=[SimpleNamespace(to_json=lambda obj=obj: obj)
                                    for obj in data['object_records']])


class ChunkReader:
    def __init__(self, payloads, chunks=3):
        self.payloads, self.chunks, self.reads = payloads, chunks, []

    def iter_object(self, name):
        self.reads.append(name)
        raw = self.payloads[name]
        for start in range(0, len(raw), self.chunks):
            yield memoryview(raw)[start:start + self.chunks]


class TransformTests(unittest.TestCase):
    def test_nvfp4_all_positions_against_coordinate_tags(self):
        # Independently tag all five source axes. Four byte passes cover a
        # unique uint32 source-position tag, avoiding modulo-256 aliasing.
        rows, width = 384, 192
        a, b, c, d, e = np.indices((rows // 128, width // 64, 32, 4, 4))
        natural_row = a * 128 + d * 32 + c
        natural_group = b * 4 + e
        tags = natural_row * (width // 16) + natural_group
        natural_tags = np.arange(rows * (width // 16)).reshape(rows, width // 16)
        for shift in (0, 8, 16, 24):
            raw = ((tags >> shift) & 255).astype(np.uint8).tobytes()
            result = verifier.nvfp4_scales(raw, rows, width)
            np.testing.assert_array_equal(result, (natural_tags >> shift) & 255)
            # Inverse axis operation proves a bijection and preserves every byte.
            inverse = result.reshape(rows // 128, 4, 32, width // 64, 4).transpose(0, 3, 2, 1, 4)
            self.assertEqual(inverse.tobytes(), raw)

    def test_nvfp4_rejects_partial_tiles_and_lengths(self):
        for rows, width, count in [(127, 64, 508), (128, 63, 504), (128, 64, 511), (0, 64, 0)]:
            with self.subTest(rows=rows, width=width, count=count), self.assertRaises(ValueError):
                verifier.nvfp4_scales(bytes(count), rows, width)

    def test_q_gate_head_order_code_bytes_and_scale_signed_zero(self):
        # Different byte lanes tag head, channel, and Q/G identity. Never float.
        heads, channels = 24, 256
        h, c = np.indices((heads, channels))
        q = np.stack((h, c, h ^ c, np.full_like(h, 0x19)), axis=-1).astype(np.uint8)
        g = np.stack((h, c, h ^ c, np.full_like(h, 0xE7)), axis=-1).astype(np.uint8)
        output = verifier.q_gate_rows(q, g)
        for head in range(heads):
            np.testing.assert_array_equal(output[head * 512:head * 512 + 256], q[head])
            np.testing.assert_array_equal(output[head * 512 + 256:(head + 1) * 512], g[head])
        qwords = np.arange(6144, dtype='<u2')
        gwords = np.arange(6144, dtype='<u2') ^ np.uint16(0x8000)
        scales = verifier.q_gate_rows(qwords, gwords)
        self.assertEqual(scales[0].tobytes(), b'\x00\x00')
        self.assertEqual(scales[256].tobytes(), b'\x00\x80')
        restored = scales.reshape(24, 2, 256, 2)
        self.assertEqual(restored[:, 0].tobytes(), qwords.tobytes())
        self.assertEqual(restored[:, 1].tobytes(), gwords.tobytes())

    def test_q_gate_rejects_partial_rows(self):
        with self.assertRaises(ValueError):
            verifier.q_gate_rows(bytes(6143), bytes(6143))
        with self.assertRaises(ValueError):
            verifier.q_gate_rows(bytes(6144), bytes(6145))

    def test_convolution_transposes_words_without_tap_reversal(self):
        words = np.array([[0, 0x8000, 0x3F80], [0x7FC1, 0xFFC1, 1],
                          [0x1234, 0xABCD, 0x7777], [0x4321, 0xDCBA, 0xFFFF]], dtype='<u2')
        result = verifier.convolution_words(words, 3)
        expected = np.array([[[0, 0x7FC1, 0x1234, 0x4321]],
                             [[0x8000, 0xFFC1, 0xABCD, 0xDCBA]],
                             [[0x3F80, 1, 0x7777, 0xFFFF]]], dtype='<u2')
        self.assertEqual(result.tobytes(), expected.tobytes())
        self.assertEqual(result.transpose(2, 0, 1).tobytes(), words.tobytes())
        with self.assertRaises(ValueError):
            verifier.convolution_words(words.tobytes()[:-1], 3)


class PlaneTests(unittest.TestCase):
    def test_fp8_child_uses_parent_scale_base_and_preserves_words(self):
        rows, width = 6, 7  # padding is nontrivial; child base is deliberately wrong.
        obj = tensor('fp8', [rows, width], 'fp8_e4m3fn_row_bf16', 'row_scale_v1', 268)
        raw = bytearray((i * 37 + 19) % 256 for i in range(obj['bytes']))
        raw[256:268] = np.array([0x1111, 0, 0x8000, 0x7FC1, 0xFFFF, 0x2222], dtype='<u2').tobytes()
        p = verifier.Planner([obj], {'a': child('fp8', 7, 21), 'b': child('fp8', 21, 35)}, [])
        p.fp8(['a', 'b'], 'projection', 4, width)
        codes, scales = [v.materialize(raw) for v in p.views]
        self.assertIs(codes.obj, raw)
        self.assertIs(scales.obj, raw)
        self.assertEqual(codes.tobytes(), raw[7:35])
        self.assertEqual(scales.tobytes(), raw[258:266])
        self.assertEqual(p.views[1].dtype, 'bf16')
        self.assertEqual(p.views[1].shape, (4, 1))

    def test_fp8_q_gate_skips_key_and_value_planes(self):
        rows, width = 14336, 3
        size = verifier.align256(rows * width) + rows * 2
        obj = tensor('parent', [rows, width], 'fp8_e4m3fn_row_bf16', 'row_scale_v1', size)
        raw = bytearray(size)
        codes = np.frombuffer(raw, dtype=np.uint8, count=rows * width).reshape(rows, width)
        codes[:6144] = [1, 2, 3]
        codes[6144:7168] = [71, 72, 73]
        codes[7168:13312] = [4, 5, 6]
        codes[13312:] = [81, 82, 83]
        scale_base = verifier.align256(rows * width)
        scales = np.frombuffer(raw, dtype='<u2', offset=scale_base)
        scales[:6144] = 0
        scales[6144:7168] = 0x7FC1
        scales[7168:13312] = 0x8000
        scales[13312:] = 0xFFFF
        p = verifier.Planner([obj], {'q': child('parent', 0, 6144 * width),
                                     'g': child('parent', 7168 * width, 13312 * width)}, [])
        p.fp8(['q', 'g'], 'q_proj', 12288, width, interleave=True)
        code_output, scale_output = [v.materialize(raw).tobytes() for v in p.views]
        self.assertEqual(code_output, (bytes([1, 2, 3]) * 256 + bytes([4, 5, 6]) * 256) * 24)
        self.assertEqual(scale_output, (b'\x00\x00' * 256 + b'\x00\x80' * 256) * 24)

    def test_nvfp4_parent_planes_child_tiles_and_bound_scalar_chunking(self):
        rows, width = 256, 128
        scale_base = verifier.align256(rows * width // 2)
        divisor_base = scale_base + rows * width // 16
        obj = tensor('packed', [rows, width], 'nvfp4', 'block_scale_k16_m128x4_v1', divisor_base + 4)
        aux = tensor('input', [], 'fp32', 'contiguous_le_v1', 4)
        uses = [{'parameter': 'up', 'input': 'ffn', 'activation_policy': 'AllowA4',
                 'auxiliaries': {'activation_input_divisor': {'object': 'input'}}}]
        p = verifier.Planner([obj, aux], {'up': child('packed', 128 * width, rows * width)}, uses)
        p.nvfp4('up', 'up_proj', 128, width, 'ffn')
        raw = bytearray((i * 13 + i // 17) % 256 for i in range(obj['bytes']))
        raw[divisor_base:] = b'\x00\x00\x00\x80'  # negative FP32 zero, not rewritten
        input_bits = b'\x23\x01\xc0\x7f'  # opaque payload also survives one-byte chunks
        reader = ChunkReader({'packed': raw, 'input': input_bits}, chunks=1)
        parent_records = verifier.hash_parent(reader, obj, p.views[:3])
        aux_record, = verifier.hash_parent(reader, aux, p.views[3:])
        self.assertEqual(parent_records[0]['sha256'], hashlib.sha256(raw[8192:16384]).hexdigest())
        self.assertEqual(parent_records[2]['sha256'], hashlib.sha256(raw[-4:]).hexdigest())
        self.assertEqual(aux_record['sha256'], hashlib.sha256(input_bits).hexdigest())
        self.assertIn('use:up@ffn', aux_record['source_bindings'][0]['binding'])
        self.assertEqual(aux_record['source_bindings'][0]['object'], 'input')
        expected_scales = verifier.nvfp4_scales(memoryview(raw)[scale_base:divisor_base], rows, width)[128:]
        self.assertEqual(p.views[1].materialize(raw).tobytes(), expected_scales.tobytes())
        self.assertEqual(reader.reads, ['packed', 'input'])
        # Changing this use's scalar changes exactly its canonical input divisor.
        reader.payloads['input'] = b'\x00\x00\x80\x3f'
        changed, = verifier.hash_parent(reader, aux, p.views[3:])
        self.assertNotEqual(aux_record['sha256'], changed['sha256'])

    def test_direct_f32_preserves_signed_zero_and_nan_payloads(self):
        obj = tensor('decay', [4], 'fp32', 'contiguous_le_v1', 16)
        p = verifier.Planner([obj], {'a': child('decay', 1, 3)}, [])
        p.direct('a', 'A_log', 'f32', (2,))
        raw = bytes.fromhex('00000000 00000080 2301c07f 0000803f')
        result = p.views[0].materialize(raw)
        self.assertIs(result.obj, raw)
        self.assertEqual(result.tobytes(), raw[4:12])

    def test_rejects_bad_ranges_truncation_and_oversized_chunks(self):
        obj = tensor('direct', [4], 'bf16', 'contiguous_le_v1', 8)
        for bounds in [(-1, 1), (0, 5), (1, 1), (True, 2)]:
            p = verifier.Planner([obj], {'x': child('direct', *bounds)}, [])
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                p.span('x')
        p = verifier.Planner([obj], {'x': {'object': 'direct'}}, [])
        p.direct('x', 'norm.weight', 'bf16', (4,))
        for raw in [bytes(7), bytes(9)]:
            with self.subTest(length=len(raw)), self.assertRaises(ValueError):
                verifier.hash_parent(ChunkReader({'direct': raw}), obj, p.views)
        with self.assertRaises(ValueError):
            p.views[0].materialize(bytes(7))


class InventoryTests(unittest.TestCase):
    def test_real_metadata_exact_coverage_count_bytes_and_types(self):
        p = verifier.plan(observed_directory())
        self.assertEqual(len(p.used), 963)
        self.assertEqual(len(p.views), 1589)
        self.assertEqual(sum(v.length for v in p.views), 20_375_588_160)
        by_name = {v.name: v for v in p.views}
        embedding = by_name['tensors/model.language_model.embed_tokens.weight']
        self.assertEqual((embedding.dtype, embedding.shape), ('fp8_e4m3', (248320, 5120)))
        self.assertEqual(sum(v.name.endswith(('.A_log', '.dt_bias')) for v in p.views), 96)
        self.assertTrue(all(v.dtype == 'f32' for v in p.views if v.name.endswith(('.A_log', '.dt_bias'))))
        self.assertFalse(any(v.name.endswith(('.k_scale', '.v_scale')) for v in p.views))
        self.assertFalse(any('mtp' in v.name or 'proposal' in v.name for v in p.views))
        self.assertEqual(max(p.objects[v.storage]['bytes'] for v in p.views), verifier.MAX_PARENT_BYTES)

    def test_inventory_rejects_missing_extra_partial_and_bad_use(self):
        for mutation in ('missing', 'extra', 'row', 'tile', 'aux', 'duplicate_use', 'config'):
            directory = observed_directory()
            binding = directory.bindings
            with self.subTest(mutation=mutation):
                if mutation == 'missing':
                    del binding['text/final_norm']
                elif mutation == 'extra':
                    binding['text/new'] = binding['text/final_norm']
                elif mutation == 'row':
                    binding['text/layers/3/attention/query']['parts'][0]['range'][0] = 1
                elif mutation == 'tile':
                    span = binding['text/layers/0/mlp/up']['parts'][0]['range']
                    span[0] -= 5120
                    span[1] -= 5120
                elif mutation in ('aux', 'duplicate_use'):
                    use = next(u for u in directory.uses if u['parameter'] == 'text/layers/0/mlp/up')
                    if mutation == 'aux':
                        use['auxiliaries']['activation_input_divisor'] = binding['text/final_norm']
                    else:
                        directory.uses.append(copy.deepcopy(use))
                else:
                    directory.components['text']['config']['num_attention_heads'] = 23
                with self.assertRaises(ValueError):
                    verifier.plan(directory)

    def test_grouped_hashing_reads_each_parent_once_and_reports_no_weights(self):
        obj = tensor('norms', [4], 'bf16', 'contiguous_le_v1', 8)
        p = verifier.Planner([obj], {'a': child('norms', 0, 2), 'b': child('norms', 2, 4)}, [])
        p.direct('a', 'a', 'bf16', (2,))
        p.direct('b', 'b', 'bf16', (2,))
        artifact = ChunkReader({'norms': bytes.fromhex('0000 0080 803f 81ff')})
        artifact.directory = SimpleNamespace(files=[None])
        with patch.object(verifier, 'plan', return_value=p):
            report = verifier.verify(artifact)
        self.assertEqual(artifact.reads, ['norms'])
        self.assertEqual(report['count'], 2)
        self.assertEqual(report['bytes'], 8)
        json.dumps(report)  # No retained arrays, bytes, or memoryviews.
        self.assertEqual(report['source']['reader_revision'], verifier.READER_REVISION)


class ReportTests(unittest.TestCase):
    def test_comparison_reports_all_fields_duplicates_missing_and_extra(self):
        obj = {'name': 'a', 'dtype': 'bf16', 'shape': [2], 'length': 4, 'sha256': 'abc'}
        expected = {'objects': [obj, dict(obj, name='b')]}
        self.assertEqual(verifier.compare_reports(expected, copy.deepcopy(expected)), [])
        actual = {'objects': [{'name': 'a', 'dtype': 'f32', 'shape': [1], 'length': 9},
                              obj, dict(obj, name='extra'), None]}
        mismatches = verifier.compare_reports(expected, actual)
        self.assertEqual(len(mismatches), 8)
        self.assertEqual({m['field'] for m in mismatches}, {'name', 'dtype', 'shape', 'length', 'sha256'})
        with self.assertRaises(ValueError):
            verifier.compare_reports(expected, {'objects': {}})

    def test_never_overwrites_existing_path_or_dangling_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'report.json'
            verifier.write_report(path, {'objects': []})
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                verifier.write_report(path, {'objects': ['replacement']})
            self.assertEqual(path.read_bytes(), original)
            link = Path(tmp) / 'link.json'
            link.symlink_to(Path(tmp) / 'absent.json')
            with self.assertRaises(FileExistsError):
                verifier.write_report(link, {})

    def test_cli_refuses_wrong_source_without_importing_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / 'source.ninfer'
            artifact.write_bytes(b'wrong')
            output = Path(tmp) / 'report.json'
            with patch.object(verifier, 'checked_reader') as reader, self.assertRaises(ValueError):
                verifier.main(['--ninfer-source', tmp, '--artifact', str(artifact), '--output', str(output)])
            reader.assert_not_called()
            self.assertFalse(output.exists())

    def test_cli_comparison_writes_all_mismatches_and_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output, comparison = [Path(tmp) / n for n in ('source', 'output', 'rust')]
            source.write_bytes(b'fixture')
            digest = hashlib.sha256(b'fixture').hexdigest()
            record = {'name': 'a', 'dtype': 'bf16', 'shape': [2], 'length': 4, 'sha256': 'expected'}
            report = {'count': 1, 'bytes': 4, 'objects': [record]}
            comparison.write_text(json.dumps({'objects': [dict(record, dtype='f32', sha256='bad')]}))
            with patch.object(verifier, 'SOURCE_BYTES', 7), patch.object(verifier, 'SOURCE_SHA256', digest), \
                    patch.object(verifier, 'checked_reader') as reader, \
                    patch.object(verifier, 'verify', return_value=report), patch('builtins.print'):
                code = verifier.main(['--ninfer-source', tmp, '--artifact', str(source),
                                      '--output', str(output), '--compare', str(comparison)])
            self.assertEqual(code, 1)
            result = json.loads(output.read_text())
            self.assertEqual(result['comparison']['mismatch_count'], 2)
            self.assertFalse(result['comparison']['matched'])
            reader.assert_called_once()

    def test_cli_same_length_bad_hash_and_source_drift_never_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / 'source', Path(tmp) / 'output'
            source.write_bytes(b'fixture')
            args = ['--ninfer-source', tmp, '--artifact', str(source), '--output', str(output)]
            with patch.object(verifier, 'SOURCE_BYTES', 7), \
                    patch.object(verifier, 'checked_reader') as reader, self.assertRaises(ValueError):
                verifier.main(args)
            reader.assert_not_called()
            digest = hashlib.sha256(b'fixture').hexdigest()
            with patch.object(verifier, 'SOURCE_BYTES', 7), patch.object(verifier, 'SOURCE_SHA256', digest), \
                    patch.object(verifier, 'checked_reader'), patch.object(verifier, 'verify', return_value={}), \
                    patch.object(verifier, 'sha256_file', side_effect=[digest, 'changed']), \
                    self.assertRaises(ValueError):
                verifier.main(args)
            self.assertFalse(output.exists())

    def test_reader_rejects_wrong_pin_dirty_tree_and_import_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            for outputs, module in [(['wrong-pin'], None),
                                    ([verifier.READER_REVISION, ' M tools/artifact/reader.py'], None),
                                    ([verifier.READER_REVISION, ''], SimpleNamespace(__file__='/elsewhere/reader.py'))]:
                with self.subTest(outputs=outputs), \
                        patch.object(verifier.subprocess, 'check_output', side_effect=outputs), \
                        patch.object(verifier.importlib, 'import_module', return_value=module), \
                        patch.object(verifier.sys, 'path', list(verifier.sys.path)), \
                        self.assertRaises(ValueError):
                    verifier.checked_reader(Path(tmp))


if __name__ == '__main__':
    unittest.main()
