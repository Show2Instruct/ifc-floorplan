#!/usr/bin/env python
"""Check a recorded floorplan run without altering it or its dataset inputs.

Verifies archived sources, input/output fingerprints, model coverage, provenance,
polygon validity, section ownership and connection endpoints. This is an artifact
consistency check, not a measurement of architectural accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import zipfile

from shapely.geometry import MultiPolygon, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fp import runs


def verify(run_dir, *, root=runs.ROOT):
    run_dir, root = Path(run_dir).resolve(), Path(root).resolve()
    record = json.loads((run_dir / 'run.json').read_text())
    if record.get('operation') != 'floorplans' or record.get('format_version') != 1:
        raise ValueError('requires a format-1 floorplans run')
    errors, warnings, counts = [], [], Counter()

    def require(condition, message):
        if not condition:
            errors.append(message)

    def file_check(base, name, expected):
        path = (base / name).resolve()
        ok = path.is_relative_to(base) and path.is_file()
        require(ok, f'missing or unsafe file: {name}')
        if ok:
            require(runs.sha256(path) == expected, f'fingerprint mismatch: {name}')
        return ok

    require(record['status'] == 'completed', f"run status: {record['status']}")
    require(not record['failures'], 'run contains failed models')
    fingerprints = record['source_files']
    code_id = hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest()
    require(code_id == record['code_sha256'], 'source fingerprint map contradicts code id')
    if file_check(run_dir, 'source.zip', record['source_archive_sha256']):
        with zipfile.ZipFile(run_dir / 'source.zip') as archive:
            require(sorted(archive.namelist()) == sorted(fingerprints), 'archive source list mismatch')
            for name, digest in fingerprints.items():
                require(name in archive.namelist() and
                        hashlib.sha256(archive.read(name)).hexdigest() == digest,
                        f'archived source mismatch: {name}')
    changed = [name for name, digest in fingerprints.items()
               if not (root / name).is_file() or runs.sha256(root / name) != digest]
    if changed:
        warnings.append({'live_sources_differ_from_archive': changed})
    for name, digest in record['output_files'].items():
        file_check(run_dir, name, digest)
        counts['recorded_output_files'] += 1
    actual = {str(p.relative_to(run_dir)) for p in run_dir.rglob('*')
              if p.is_file() and p != run_dir / 'run.json'}
    require(actual == set(record['output_files']), 'recorded output file set differs from disk')
    inputs = {row['model']: row for row in record['inputs']}
    models = {row['model']: row for row in record['models']}
    require(len(inputs) == len(record['inputs']), 'duplicate input model')
    require(len(models) == len(record['models']), 'duplicate model result')
    require(inputs.keys() == models.keys(), 'model result set differs from selected inputs')
    for model, entry in inputs.items():
        path = (root / entry['path']).resolve()
        require(path.is_relative_to(root / 'data'), f'{model}: input outside data/')
        file_check(root, entry['path'], entry['ifc_sha256'])
        counts['inputs'] += 1

    def geometry(value, label, *, required=False):
        if not value:
            require(not required, f'{label}: missing geometry')
            return
        try:
            poly = MultiPolygon([Polygon(part[0], part[1:]) for part in value])
            require(poly.is_valid and not poly.is_empty and poly.area > 0,
                    f'{label}: invalid or empty polygon')
            counts['polygons_checked'] += 1
        except (ValueError, TypeError, IndexError) as exc:
            errors.append(f'{label}: unreadable polygon: {exc}')

    for model, result in models.items():
        counts[f"models_{result['status']}"] += 1
        if result['status'] == 'skipped':
            require(bool(result.get('reason')), f'{model}: skip without reason')
            continue
        if result['status'] != 'completed':
            continue
        folder = run_dir / 'models' / model
        try:
            building = json.loads((folder / 'building.json').read_text())
            require(building['provenance'] == runs.model_provenance(record, inputs[model]),
                    f'{model}: building provenance contradicts run')
            owned, rooms, storeys = {}, {}, []
            indices = [s['storey_index'] for s in building['storeys']]
            require(len(indices) == len(set(indices)), f'{model}: duplicate storey index')
            for summary in building['storeys']:
                i = summary['storey_index']
                storey = json.loads((folder / f'storey_{i:02d}.json').read_text())
                storeys.append(storey)
                label = f'{model}/storey_{i:02d}'
                require(all(storey.get(k) == v for k, v in summary.items()),
                        f'{label}: building summary contradicts storey')
                for field, records in [('n_rooms', 'rooms'), ('n_elements', 'elements'),
                                       ('n_visible_elements', 'projections')]:
                    require(storey[field] == len(storey[records]), f'{label}: {field} mismatch')
                counts['storeys'] += 1
                for kind in ('rooms', 'outdoor_spaces'):
                    for room in storey.get(kind, []):
                        require(room['id'] not in rooms, f'{label}: duplicate room {room["id"]}')
                        rooms[room['id']] = i
                        geometry(room['geometry'], f'{label}/{room["id"]}', required=True)
                        counts[kind] += 1
                for element in storey['elements']:
                    sid = element['step_id']
                    require(sid not in owned, f'{label}: duplicate owned element #{sid}')
                    owned[sid] = (i, element['ifc_class'])
                    geometry(element['footprint_geometry'], f'{label}/element#{sid}')
                status = storey.get('room_inference', {}).get('status', 'missing')
                counts[f'room_inference_{status}'] += 1
                section = storey.get('section_projection', {})
                counts['section_fallbacks'] += len(section.get('fallbacks', []))
                counts['section_exclusions'] += len(section.get('excluded', []))
            for storey in storeys:
                i = storey['storey_index']
                label = f'{model}/storey_{i:02d}'
                seen = set()
                for section in storey['projections']:
                    sid = section['step_id']
                    require(sid not in seen, f'{label}: duplicate section #{sid}')
                    seen.add(sid)
                    require(owned.get(sid) == (section['owner_storey_index'], section['ifc_class']),
                            f'{label}: section ownership/class mismatch #{sid}')
                    require(section['room'] is None or rooms.get(section['room']) == i,
                            f'{label}: section room missing or on another storey #{sid}')
                    geometry(section['geometry'], f'{label}/section#{sid}', required=True)
                for edge in storey.get('room_connections', []):
                    for key in ('a', 'b'):
                        endpoint = edge[key]
                        outside = isinstance(endpoint, str) and endpoint.startswith('EXT-')
                        require(rooms.get(endpoint) == i or (outside and edge.get('exterior')),
                                f'{label}: connection endpoint missing/wrong storey: {endpoint}')
                    counts['room_connections'] += 1
            for edge in building.get('vertical_connections', []):
                require(edge['a'] in rooms and edge['b'] in rooms,
                        f'{model}: vertical connection has missing room')
                require(rooms.get(edge['a']) != rooms.get(edge['b']),
                        f'{model}: vertical connection on same storey')
                counts['vertical_connections'] += 1
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f'{model}: cannot verify model: {exc}')
    return {'run': str(run_dir), 'run_sha256': runs.sha256(run_dir / 'run.json'),
            'verifier_sha256': runs.sha256(Path(__file__)),
            'checked_utc': runs.utc(), 'passed': not errors, 'counts': dict(counts),
            'errors': errors, 'warnings': warnings,
            'scope': 'Artifact integrity and structural consistency; not architectural accuracy.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    out = runs.safe_output(args.out)
    if out.exists() or out.is_relative_to(args.run.resolve()):
        parser.error('choose a new report outside the source run')
    report = verify(args.run)
    runs.atomic_json(out, report)
    print(json.dumps({k: report[k] for k in ('passed', 'counts', 'errors')}, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
