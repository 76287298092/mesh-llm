#!/usr/bin/env python3
"""Independent offline hashes of canonical text views from the pinned .ninfer.

Framing and object reads belong exclusively to the official read-only Artifact
API in --ninfer-source (the same checkout pin/path as export_ninfer_bundle.py).
Only encoded planes are interpreted here. NumPy permutes opaque bytes/words;
there is no dequantization, numerical cast, model execution, or .mspec export.

Reads are grouped by physical parent: one raw object lives at a time (at most
1,271,895,040 bytes), plus reader chunks and one small transformed projection.
Contiguous planes hash through memoryviews; reports retain no weight data.
"""
import argparse
from dataclasses import dataclass
import hashlib
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np

SOURCE_SHA256 = '74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82'
SOURCE_BYTES = 23_719_715_844
READER_REVISION = 'e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d'
PROFILE = 'qwen3.8-27b:text:ninfer-v3-control-v1'
TEXT_BINDINGS = 963
TEXT_TENSORS = 1589
TEXT_BYTES = 20_375_588_160
MAX_PARENT_BYTES = 1_271_895_040
WORD_BYTES = {'f32': 4, 'bf16': 2, 'fp8_e4m3': 1, 'u8': 1}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def align256(size):
    return (size + 255) // 256 * 256


def byte_view(data):
    return memoryview(data).cast('B')


def checked_slice(data, first, end):
    view = byte_view(data)
    require(0 <= first < end <= len(view), 'source byte range out of bounds')
    return view[first:end]


def nvfp4_scales(data, rows, width):
    """Stored [tileN,tileK,lane,quarter,group] -> [N,K/16].

    Swapping the axes, rather than implementing the Rust per-element address
    calculation, independently assigns each stored byte its natural row/group.
    Whole 128-row child tiles can be passed without materializing parent scales.
    """
    require(rows > 0 and rows % 128 == 0 and width > 0 and width % 64 == 0,
            'NVFP4 scale geometry must contain complete tiles')
    source = np.frombuffer(data, dtype=np.uint8)
    require(source.size == rows * width // 16, 'NVFP4 scale length mismatch')
    return source.reshape(rows // 128, width // 64, 32, 4, 4).transpose(
        0, 3, 2, 1, 4).reshape(rows, width // 16)


def q_gate_rows(query, gate, heads=24, head_rows=256):
    """Stack whole opaque rows as [head,Q-or-gate,channel,byte]."""
    q = np.frombuffer(query, dtype=np.uint8)
    g = np.frombuffer(gate, dtype=np.uint8)
    require(heads > 0 and head_rows > 0 and q.size == g.size and q.size > 0,
            'Q/gate geometry mismatch')
    require(q.size % (heads * head_rows) == 0, 'partial Q/gate row')
    q = q.reshape(heads, head_rows, -1)
    g = g.reshape(heads, head_rows, -1)
    return np.stack((q, g), axis=1).reshape(heads * 2 * head_rows, -1)


def convolution_words(data, channels):
    require(channels > 0 and len(byte_view(data)) == channels * 4 * 2,
            'convolution length mismatch')
    # uint16 is only an opaque two-byte word, never a BF16 numerical conversion.
    words = np.frombuffer(data, dtype='<u2').reshape(4, channels)
    return np.ascontiguousarray(words.T.reshape(channels, 1, 4))


@dataclass(frozen=True)
class Span:
    storage: dict
    first: int
    end: int
    binding: str

    def provenance(self):
        return {'binding': self.binding, 'object': self.storage['id'],
                'element_range': [self.first, self.end]}


@dataclass(frozen=True)
class View:
    name: str
    dtype: str
    shape: tuple
    storage: str
    ranges: tuple
    transform: str
    source_bindings: tuple
    geometry: tuple = ()

    @property
    def length(self):
        return math.prod(self.shape) * WORD_BYTES[self.dtype]

    def materialize(self, parent):
        parts = [checked_slice(parent, begin, end) for begin, end in self.ranges]
        if self.transform == 'copy':
            require(len(parts) == 1, 'copy must be contiguous')
            result = parts[0]
        elif self.transform == 'q_gate_stack':
            require(len(parts) == 2, 'Q/gate must have two source spans')
            result = q_gate_rows(*parts)
        elif self.transform == 'nvfp4_scale_axes':
            result = nvfp4_scales(parts[0], *self.geometry)
        elif self.transform == 'conv_transpose':
            result = convolution_words(parts[0], *self.geometry)
        else:
            raise ValueError(f'unknown transform {self.transform}')
        result = byte_view(result)
        require(len(result) == self.length, f'{self.name}: canonical length mismatch')
        return result

    def record(self, parent):
        # No buffer or NumPy object escapes this method into the report.
        return {'name': self.name, 'dtype': self.dtype, 'shape': list(self.shape),
                'length': self.length,
                'sha256': hashlib.sha256(self.materialize(parent)).hexdigest(),
                'source_bindings': list(self.source_bindings),
                'source_byte_ranges': [list(r) for r in self.ranges],
                'transform_tags': [self.transform, 'source_bits_unchanged']}


class Planner:
    """Metadata-only specification, independent of Rust object/index machinery."""
    def __init__(self, objects, bindings, uses):
        self.objects = {}
        for obj in objects:
            if obj.get('kind') != 'tensor':
                continue
            require(obj['id'] not in self.objects, 'duplicate tensor storage')
            require(isinstance(obj['shape'], (list, tuple))
                    and all(type(x) is int and x > 0 for x in obj['shape']),
                    'invalid parent shape')
            require(type(obj['bytes']) is int and 0 < obj['bytes'] <= MAX_PARENT_BYTES,
                    'invalid or oversized parent bytes')
            self.objects[obj['id']] = obj
        self.bindings, self.uses = bindings, uses
        self.used = set()
        self.views = []

    def binding_span(self, binding, label):
        if set(binding) == {'object'}:
            object_id, bounds = binding['object'], None
        else:
            require(set(binding) == {'parts'} and len(binding['parts']) == 1,
                    f'{label}: expected one physical parent')
            part = binding['parts'][0]
            require(set(part) == {'object', 'range'}, f'{label}: invalid binding part')
            object_id, bounds = part['object'], part['range']
        require(object_id in self.objects, f'{label}: absent tensor {object_id}')
        obj = self.objects[object_id]
        elements = math.prod(obj['shape'])
        if bounds is None:
            bounds = [0, elements]
        require(isinstance(bounds, (tuple, list)) and len(bounds) == 2
                and all(type(x) is int for x in bounds)
                and 0 <= bounds[0] < bounds[1] <= elements,
                f'{label}: logical range out of bounds')
        return Span(obj, *bounds, label)

    def span(self, name):
        require(name in self.bindings, f'missing binding {name}')
        self.used.add(name)
        return self.binding_span(self.bindings[name], name)

    def add(self, name, dtype, shape, spans, ranges, transform='copy', geometry=()):
        obj = spans[0].storage
        require(all(s.storage['id'] == obj['id'] for s in spans), 'multiple parents')
        require(all(0 <= lo < hi <= obj['bytes'] for lo, hi in ranges),
                f'{name}: source plane outside parent')
        self.views.append(View('tensors/' + name, dtype, tuple(shape), obj['id'],
                               tuple(ranges), transform,
                               tuple(s.provenance() for s in spans), tuple(geometry)))

    def direct(self, source, name, dtype, shape):
        span = self.span(source)
        obj, word = span.storage, WORD_BYTES[dtype]
        require(obj['format'] == {'bf16': 'bf16', 'f32': 'fp32'}[dtype]
                and obj['layout'] == 'contiguous_le_v1', f'{source}: direct encoding')
        require(span.end - span.first == math.prod(shape), f'{source}: direct shape')
        require(obj['bytes'] == math.prod(obj['shape']) * word, 'direct parent size')
        self.add(name, dtype, shape, [span], [(span.first * word, span.end * word)])

    def matrix_spans(self, sources, rows, width, fmt, layout):
        spans = [self.span(s) for s in sources]
        obj = spans[0].storage
        require(obj['format'] == fmt and obj['layout'] == layout, 'matrix encoding')
        require(len(obj['shape']) == 2 and obj['shape'][1] == width, 'parent geometry')
        require(all(s.storage['id'] == obj['id'] and s.first % width == 0
                    and s.end % width == 0 for s in spans), 'partial rows or mixed parents')
        require(sum(s.end - s.first for s in spans) == rows * width, 'child row count')
        ordered = sorted((s.first, s.end) for s in spans)
        require(all(a[1] <= b[0] for a, b in zip(ordered, ordered[1:])), 'overlapping rows')
        return spans

    def fp8(self, sources, name, rows, width, interleave=False):
        spans = self.matrix_spans(sources, rows, width, 'fp8_e4m3fn_row_bf16', 'row_scale_v1')
        obj = spans[0].storage
        scale_base = align256(math.prod(obj['shape']))
        require(obj['bytes'] == scale_base + obj['shape'][0] * 2, 'FP8 parent extent')
        if interleave:
            require(len(spans) == 2 and rows == 12288
                    and all(s.end - s.first == 6144 * width for s in spans), 'Q/gate rows')
            transform = 'q_gate_stack'
        else:
            require(all(a.end == b.first for a, b in zip(spans, spans[1:])),
                    'noncontiguous plain FP8 rows')
            transform = 'copy'
        for suffix, dtype, shape, base, row_bytes in (
                ('weight', 'fp8_e4m3', (rows, width), 0, width),
                ('weight_scale', 'bf16', (rows, 1), scale_base, 2)):
            ranges = [(base + s.first // width * row_bytes,
                       base + s.end // width * row_bytes) for s in spans]
            if not interleave:
                ranges = [(ranges[0][0], ranges[-1][1])]
            self.add(f'{name}.{suffix}', dtype, shape, spans, ranges, transform)

    def nvfp4(self, source, name, rows, width, input_name):
        span, = self.matrix_spans([source], rows, width, 'nvfp4', 'block_scale_k16_m128x4_v1')
        obj = span.storage
        require(obj['shape'][0] % 128 == 0 and width % 64 == 0
                and (span.first // width) % 128 == 0 and rows % 128 == 0,
                'partial NVFP4 scale tiles')
        scale_base = align256(math.prod(obj['shape']) // 2)
        divisor_base = scale_base + math.prod(obj['shape']) // 16
        require(obj['bytes'] == divisor_base + 4, 'NVFP4 parent extent')
        self.add(name + '.weight_packed', 'u8', (rows, width // 2), [span],
                 [(span.first // 2, span.end // 2)])
        self.add(name + '.weight_scale', 'fp8_e4m3', (rows, width // 16), [span],
                 [(scale_base + span.first // 16, scale_base + span.end // 16)],
                 'nvfp4_scale_axes', (rows, width))
        self.add(name + '.weight_global_scale', 'f32', (1,), [span],
                 [(divisor_base, divisor_base + 4)])
        uses = [u for u in self.uses if u['parameter'] == source and u['input'] == input_name]
        require(len(uses) == 1 and uses[0]['activation_policy'] == 'AllowA4',
                f'{source}: missing or ambiguous NVFP4 use')
        aux = self.binding_span(uses[0]['auxiliaries']['activation_input_divisor'],
                                f'use:{source}@{input_name}/activation_input_divisor')
        require(aux.storage['format'] == 'fp32' and aux.storage['layout'] == 'contiguous_le_v1'
                and aux.storage['bytes'] == 4 and aux.first == 0 and aux.end == 1,
                'input divisor must be a source-bound raw FP32 scalar')
        self.add(name + '.input_global_scale', 'f32', (1,), [aux], [(0, 4)])

    def convolution(self, source, name):
        span = self.span(source)
        obj = span.storage
        require(obj['format'] == 'bf16' and obj['layout'] == 'contiguous_le_v1'
                and obj['shape'] == [4, 10240] and obj['bytes'] == 81920
                and span.first == 0 and span.end == 40960, 'convolution geometry')
        self.add(name, 'bf16', (10240, 1, 4), [span], [(0, 81920)],
                 'conv_transpose', (10240,))


def validate_config(config):
    expected = {'hidden_size': 5120, 'vocab_size': 248320, 'num_hidden_layers': 64,
                'num_attention_heads': 24, 'num_key_value_heads': 4, 'head_dim': 256,
                'linear_num_key_heads': 16, 'linear_key_head_dim': 128,
                'linear_num_value_heads': 48, 'linear_value_head_dim': 128,
                'linear_conv_kernel_dim': 4, 'intermediate_size': 17408}
    for field, value in expected.items():
        require(config.get(field) == value, f'unsupported config {field}')
    require(config.get('layer_types') == [
        'full_attention' if i % 4 == 3 else 'linear_attention' for i in range(64)],
        'unsupported layer schedule')
    require(struct.pack('<f', config['rms_norm_eps']) == struct.pack('<f', 1e-6), 'norm epsilon')
    rope = config['rope_parameters']
    require(rope['rope_theta'] == 10_000_000 and rope['partial_rotary_factor'] == 0.25, 'RoPE')


def map_attention(p, source, dest):
    s, d = source + '/attention/', dest + '.self_attn.'
    for src, dst in [('query_norm', 'q_norm'), ('key_norm', 'k_norm')]:
        p.direct(s + src, d + dst + '.weight', 'bf16', (256,))
    p.fp8([s + 'query', s + 'gate'], d + 'q_proj', 12288, 5120, interleave=True)
    for src, dst, rows, width in [('key', 'k_proj', 1024, 5120),
                                   ('value', 'v_proj', 1024, 5120),
                                   ('output', 'o_proj', 5120, 6144)]:
        p.fp8([s + src], d + dst, rows, width)


def map_gdn(p, source, dest):
    s, d = source + '/gdn/', dest + '.linear_attn.'
    for src, dst in [('a_log', 'A_log'), ('dt_bias', 'dt_bias')]:
        p.direct(s + src, d + dst, 'f32', (48,))
    for src, dst in [('a_projection', 'in_proj_a'), ('b_projection', 'in_proj_b')]:
        p.direct(s + src, d + dst + '.weight', 'bf16', (48, 5120))
    p.direct(s + 'norm', d + 'norm.weight', 'bf16', (128,))
    p.convolution(s + 'convolution', d + 'conv1d.weight')
    p.fp8([s + 'query', s + 'key', s + 'value'], d + 'in_proj_qkv', 10240, 5120)
    p.fp8([s + 'z'], d + 'in_proj_z', 6144, 5120)
    p.fp8([s + 'output'], d + 'out_proj', 5120, 6144)


def plan(directory):
    validate_config(directory.components['text']['config'])
    p = Planner([o.to_json() for o in directory.objects], directory.bindings, directory.uses)
    p.fp8(['text/token_embedding'], 'model.language_model.embed_tokens', 248320, 5120)
    p.fp8(['text/output_head'], 'lm_head', 248320, 5120)
    p.direct('text/final_norm', 'model.language_model.norm.weight', 'bf16', (5120,))
    for layer in range(64):
        source, dest = f'text/layers/{layer}', f'model.language_model.layers.{layer}'
        for src, dst in [('input_norm', 'input_layernorm'),
                         ('post_attention_norm', 'post_attention_layernorm')]:
            p.direct(source + '/' + src, dest + '.' + dst + '.weight', 'bf16', (5120,))
        (map_attention if layer % 4 == 3 else map_gdn)(p, source, dest)
        for op, rows, width, input_name in [('gate', 17408, 5120, 'ffn_input'),
                                            ('up', 17408, 5120, 'ffn_input'),
                                            ('down', 5120, 17408, 'mlp/product')]:
            src, dst = source + '/mlp/' + op, dest + '.mlp.' + op + '_proj'
            if layer < 56:
                p.nvfp4(src, dst, rows, width, source + '/' + input_name)
            else:
                p.fp8([src], dst, rows, width)
    selected = {n for n in directory.bindings if n.startswith('text/')}
    require(p.used == selected and len(selected) == TEXT_BINDINGS,
            f'text binding coverage mismatch: missing={sorted(selected - p.used)}, '
            f'extra={sorted(p.used - selected)}, count={len(selected)}')
    require(len(p.views) == TEXT_TENSORS and len({v.name for v in p.views}) == TEXT_TENSORS,
            'canonical view count or name uniqueness mismatch')
    require(sum(v.length for v in p.views) == TEXT_BYTES, 'canonical byte total mismatch')
    return p


def hash_parent(artifact, storage, views):
    """One-entry grouped cache; release the previous parent before the next read."""
    data = bytearray(storage['bytes'])
    target = memoryview(data)
    offset = 0
    for chunk in artifact.iter_object(storage['id']):
        chunk = byte_view(chunk)
        end = offset + len(chunk)
        require(end <= len(target), f'{storage["id"]}: oversized object read')
        target[offset:end] = chunk
        offset = end
    require(offset == len(target), f'{storage["id"]}: truncated object read')
    return [v.record(target) for v in views]


def verify(artifact):
    require(len(artifact.directory.files) == 1, 'only the pinned single-file artifact is supported')
    p = plan(artifact.directory)
    records = []
    for storage_id, views in itertools.groupby(sorted(p.views, key=lambda v: v.storage),
                                                key=lambda v: v.storage):
        records.extend(hash_parent(artifact, p.objects[storage_id], views))
    return {'schema_version': 1, 'profile': PROFILE,
            'source': {'artifact_sha256': SOURCE_SHA256, 'artifact_bytes': SOURCE_BYTES,
                       'reader_revision': READER_REVISION},
            'text_bindings': len(p.used), 'count': len(records),
            'bytes': sum(v['length'] for v in records),
            'objects': sorted(records, key=lambda v: v['name']),
            'preservation': {'requantized': False, 'numerically_cast': False,
                             'runtime_executable': False, 'synthesized_tensors': False,
                             'excluded_components': ['vision', 'dflash2', 'mtp', 'proposal']}}


def compare_reports(expected, actual):
    """Return every metadata/hash/set mismatch, including duplicates/missing fields."""
    require(isinstance(actual, dict) and isinstance(actual.get('objects'), list),
            'comparison report must contain an objects array')
    want = {v['name']: v for v in expected['objects']}
    seen, mismatches = set(), []
    for index, item in enumerate(actual['objects']):
        if not isinstance(item, dict) or not isinstance(item.get('name'), str):
            mismatches.append({'index': index, 'field': 'name', 'error': 'invalid object name'})
            continue
        name = item['name']
        if name in seen:
            mismatches.append({'name': name, 'field': 'name', 'error': 'duplicate object'})
        seen.add(name)
        if name not in want:
            mismatches.append({'name': name, 'field': 'name', 'error': 'unexpected object'})
            continue
        for field in ('dtype', 'shape', 'length', 'sha256'):
            if item.get(field) != want[name][field]:
                mismatches.append({'name': name, 'field': field, 'expected': want[name][field],
                                   'actual': item.get(field)})
    mismatches.extend({'name': name, 'field': 'name', 'error': 'missing object'}
                      for name in sorted(want.keys() - seen))
    return mismatches


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def checked_reader(source):
    source = source.resolve(strict=True)
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    require(revision == READER_REVISION, 'Ninfer reader checkout does not match the pinned revision')
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--',
                                     'tools/artifact'], text=True)
    require(not dirty.strip(), 'Ninfer artifact reader has local changes')
    # Do not silently use an already imported package from a different checkout.
    sys.path.insert(0, str(source))
    module = importlib.import_module('tools.artifact.reader')
    require(Path(module.__file__).resolve() == source / 'tools/artifact/reader.py',
            'reader import escaped the requested source checkout')
    return module.Artifact


def write_report(path, report):
    # Exclusive creation also refuses dangling symlinks and races after preflight.
    with path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ninfer-source', type=Path, required=True)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, metavar='NEW_JSON')
    parser.add_argument('--compare', type=Path, metavar='RUST_REPORT')
    args = parser.parse_args(argv)
    require(not os.path.lexists(args.output), 'refusing to overwrite output')
    # Hash the exact source before opening it through the official reader.
    before = args.artifact.stat()
    require(before.st_size == SOURCE_BYTES and sha256_file(args.artifact) == SOURCE_SHA256,
            'artifact does not match the pinned source bytes')
    reader = checked_reader(args.ninfer_source)
    with reader(args.artifact) as artifact:
        report = verify(artifact)
    after = args.artifact.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            'source changed during verification')
    require(sha256_file(args.artifact) == SOURCE_SHA256, 'source hash changed during verification')
    mismatches = []
    if args.compare:
        with args.compare.open(encoding='utf-8') as stream:
            mismatches = compare_reports(report, json.load(stream))
        report['comparison'] = {'report': str(args.compare), 'matched': not mismatches,
                                'mismatch_count': len(mismatches), 'mismatches': mismatches}
    write_report(args.output, report)
    print(json.dumps({'count': report['count'], 'bytes': report['bytes'],
                      'source_sha256': SOURCE_SHA256, 'source_pin': READER_REVISION,
                      'mismatches': len(mismatches), 'output': str(args.output)}))
    return 1 if mismatches else 0


if __name__ == '__main__':
    sys.exit(main())
