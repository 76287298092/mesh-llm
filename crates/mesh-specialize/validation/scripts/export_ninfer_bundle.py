#!/usr/bin/env python3
"""Offline, byte-preserving export via an explicitly supplied Ninfer checkout.

This script does not parse .ninfer framing or implement its codecs. It invokes
Ninfer's existing read-only Artifact API, and exports a versioned intermediate
bundle for the independent Rust .mspec assembler. No model math runs here.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

SOURCE_SHA256 = '74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82'
SOURCE_BYTES = 23_719_715_844
READER_REVISION = 'e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d'
PROFILE = 'qwen3.8-27b:ninfer-preserved-v1'
ALIGNMENT = 256


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def selected_name(name):
    return name.startswith(('text/', 'mtp/', 'proposal/'))


def object_refs(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == 'object':
                if not isinstance(item, str):
                    raise ValueError('non-string object reference')
                yield item
            else:
                yield from object_refs(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from object_refs(item)


def export_selection(directory):
    """Reference closure includes use auxiliaries, not just primary weights."""
    components = {k: v for k, v in directory.components.items() if k in ('text', 'mtp')}
    if set(components) != {'text', 'mtp'}:
        raise ValueError('expected text and mtp components')
    bindings = {k: v for k, v in directory.bindings.items() if selected_name(k)}
    uses = [u for u in directory.uses
            if selected_name(u['parameter']) and selected_name(u['input'])]
    refs = set(object_refs(bindings)) | set(object_refs(uses))
    for component in components.values():
        refs.update(component.get('resources', {}).values())
    by_id = {o.id: o for o in directory.objects}
    if refs - by_id.keys():
        raise ValueError(f'missing referenced objects: {sorted(refs - by_id.keys())}')
    if not {'text/token_embedding', 'text/output_head', 'proposal/head', 'proposal/token_ids'} <= bindings.keys():
        raise ValueError('missing required text/proposal bindings')
    return components, bindings, uses, [by_id[name] for name in sorted(refs)]


def write_bundle(artifact, output, source_hash, source_bytes, reader_revision):
    output.mkdir(exist_ok=False)
    components, bindings, uses, objects = export_selection(artifact.directory)
    entries = []
    storage_metadata = []
    packed_sha = hashlib.sha256()
    offset = 0
    with (output / 'payload.bin').open('xb') as payload:
        for index, obj in enumerate(objects):
            aligned = (offset + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
            padding = bytes(aligned - offset)
            payload.write(padding)
            packed_sha.update(padding)
            offset = aligned
            name = f'storage/{index:06d}'
            digest = hashlib.sha256()
            count = 0
            for chunk in artifact.iter_object(obj.id):
                count += len(chunk)
                if count > obj.bytes:
                    raise ValueError(f'object {obj.id} exceeds declared extent')
                digest.update(chunk)
                packed_sha.update(chunk)
                payload.write(chunk)
            if count != obj.bytes or count == 0:
                raise ValueError(f'object {obj.id} length mismatch')
            entries.append({'name': name, 'offset': offset, 'bytes': count,
                            'sha256': digest.hexdigest(), 'source_id': obj.id})
            metadata = obj.to_json()
            metadata.pop('offset', None)
            metadata['storage'] = name
            storage_metadata.append(metadata)
            offset += count
        payload.flush()
        import os
        os.fsync(payload.fileno())
    manifest = {
        'schema_version': 1, 'profile': PROFILE,
        'source': {'artifact_sha256': source_hash, 'artifact_bytes': source_bytes,
                   'reader_revision': reader_revision},
        'payload': {'path': 'payload.bin', 'bytes': offset, 'sha256': packed_sha.hexdigest()},
        'objects': entries,
        'model': {'components': components, 'bindings': bindings, 'uses': uses,
                  'storages': storage_metadata},
        'preservation': {'requantized': False, 'layout_transformed': False,
                         'all_selected_object_bytes_copied': True,
                         'excluded_components': ['vision', 'dflash2'],
                         'runtime_executable': False},
    }
    return manifest


def checked_reader(source):
    source = source.resolve(strict=True)
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != READER_REVISION:
        raise ValueError('Ninfer reader checkout revision does not match the pinned revision')
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--', 'tools/artifact'], text=True)
    if dirty.strip():
        raise ValueError('Ninfer artifact reader has local changes')
    sys.path.insert(0, str(source))
    module = importlib.import_module('tools.artifact.reader')
    if not Path(module.__file__).resolve().is_relative_to(source):
        raise ValueError('reader import escaped the requested source checkout')
    return module.Artifact, revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ninfer-source', type=Path, required=True)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    if args.output_directory.exists():
        raise ValueError('refusing to overwrite export directory')
    before = args.artifact.stat()
    if before.st_size != SOURCE_BYTES or sha256_file(args.artifact) != SOURCE_SHA256:
        raise ValueError('artifact does not match the benchmarked source bytes')
    reader, revision = checked_reader(args.ninfer_source)
    with reader(args.artifact) as artifact:
        manifest = write_bundle(artifact, args.output_directory, SOURCE_SHA256, before.st_size, revision)
    after = args.artifact.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('source changed during export; incomplete bundle retained')
    if sha256_file(args.artifact) != SOURCE_SHA256:
        raise ValueError('source hash changed during export; incomplete bundle retained')
    # Commit marker appears only after byte copying and source re-verification.
    with (args.output_directory / 'manifest.json').open('x') as out:
        json.dump(manifest, out, indent=2, ensure_ascii=False)
        out.write('\n')
    print(json.dumps({'objects': len(manifest['objects']), 'payload_bytes': manifest['payload']['bytes'],
                      'source_sha256': SOURCE_SHA256, 'requantized': False,
                      'runtime_executable': False, 'output': str(args.output_directory)}))


if __name__ == '__main__':
    main()
