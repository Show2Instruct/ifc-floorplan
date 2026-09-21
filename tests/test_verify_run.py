"""Integrity failures must remain visible even when a run says it completed."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fp import runs
from fp.verify_run import verify


class RunIntegrity(unittest.TestCase):
    def fixture(self, root):
        run = root / 'results/run'
        folder = run / 'models/x'
        folder.mkdir(parents=True)
        (root / 'data').mkdir()
        (root / 'data/x.ifc').write_text('input fixture')
        source = b'archived source fixture'
        hashes = {'scripts/example.py': hashlib.sha256(source).hexdigest()}
        with zipfile.ZipFile(run / 'source.zip', 'w') as z:
            z.writestr('scripts/example.py', source)
        record = dict(format_version=1, operation='floorplans', status='completed', failures=[],
                      source_files=hashes, source_archive_sha256=runs.sha256(run/'source.zip'),
                      code_sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
                      settings={}, dataset='fixture',
                      inputs=[dict(model='x', path='data/x.ifc', ifc_sha256=runs.sha256(root/'data/x.ifc'))],
                      models=[dict(model='x', status='completed')])
        geometry = [[[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]]]
        summary = dict(storey_index=0, n_rooms=1, n_elements=0, n_visible_elements=0)
        storey = dict(summary, rooms=[dict(id='r', geometry=geometry)], elements=[], projections=[],
                      room_connections=[])
        building = dict(storeys=[summary], provenance=runs.model_provenance(record, record['inputs'][0]))
        (folder/'building.json').write_text(json.dumps(building))
        (folder/'storey_00.json').write_text(json.dumps(storey))
        self.refresh(run, record)
        return run

    def refresh(self, run, record=None):
        record = record or json.loads((run/'run.json').read_text())
        record['output_files'] = {str(p.relative_to(run)): runs.sha256(p)
                                  for p in run.rglob('*') if p.is_file() and p.name != 'run.json'}
        (run/'run.json').write_text(json.dumps(record))

    def test_intact_run_passes_even_when_live_source_has_changed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = verify(self.fixture(root), root=root)
            self.assertTrue(report['passed'], report['errors'])
            self.assertTrue(report['warnings'])

    def test_modified_input_or_output_cannot_pass(self):
        for target in ('data/x.ifc', 'results/run/models/x/storey_00.json'):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run = self.fixture(root)
                path = root / target
                path.write_text(path.read_text() + '\n')
                report = verify(run, root=root)
                self.assertFalse(report['passed'])
                self.assertTrue(any('fingerprint mismatch' in e for e in report['errors']))

    def test_rehashed_broken_connection_still_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self.fixture(root)
            path = run/'models/x/storey_00.json'
            storey = json.loads(path.read_text())
            storey['room_connections'] = [dict(a='r', b='missing')]
            path.write_text(json.dumps(storey))
            self.refresh(run)
            report = verify(run, root=root)
            self.assertFalse(report['passed'])
            self.assertTrue(any('connection endpoint' in e for e in report['errors']))

    def test_rehashed_invalid_polygon_still_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self.fixture(root)
            path = run/'models/x/storey_00.json'
            storey = json.loads(path.read_text())
            storey['rooms'][0]['geometry'] = [[[[0, 0], [2, 2], [0, 2], [2, 0], [0, 0]]]]
            path.write_text(json.dumps(storey))
            self.refresh(run)
            report = verify(run, root=root)
            self.assertFalse(report['passed'])
            self.assertTrue(any('invalid or empty polygon' in e for e in report['errors']))


if __name__ == '__main__':
    unittest.main()
