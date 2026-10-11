#!/usr/bin/env python
"""Checks for run_pipeline.py: explicit inputs, run records, failures and reproducibility.

Runs in a scratch workspace holding a copy of one small ResBIM IFC and a deliberately broken
file, so it takes seconds and never writes near the real data or results.

    python tests/test_runs.py
"""
from __future__ import annotations

import contextlib
import inspect
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fp import run_pipeline, runs
from fp.build import build
from fp.extract import infer_rooms

SAMPLE = runs.SOURCE_ROOT / "data" / "ResBIM-IFC" / "data" / "0" / "0.ifc"
SCRIPTS = Path(__file__).resolve().parent.parent

# A child interpreter running the entry point against the scratch workspace. Output that
# depends on set iteration order only shows up across processes, because string hashing
# is seeded per process; two runs in one process always agree.
CHILD = """
import sys
from pathlib import Path
sys.path.insert(0, {scripts!r})
from fp import run_pipeline, runs
runs.ROOT = Path({ws!r})
sys.exit(run_pipeline.main({argv!r}))
"""


def tree(directory):
    """Relative path -> sha256 for every file below a directory."""
    d = Path(directory)
    return {str(p.relative_to(d)): runs.sha256(p) for p in sorted(d.rglob("*")) if p.is_file()}


@unittest.skipUnless(SAMPLE.is_file(), f"needs {SAMPLE}")
class RunPipeline(unittest.TestCase):
    def setUp(self):
        self.ws = Path(tempfile.mkdtemp(prefix="fp-runs-"))
        self.ifc = self.ws / "data" / "set" / "0" / "0.ifc"
        self.ifc.parent.mkdir(parents=True)
        shutil.copyfile(SAMPLE, self.ifc)
        self.bad = self.ws / "data" / "set" / "bad" / "bad.ifc"
        self.bad.parent.mkdir(parents=True)
        self.bad.write_text("this is not an IFC file\n")
        self.saved_root, runs.ROOT = runs.ROOT, self.ws

    def tearDown(self):
        runs.ROOT = self.saved_root
        shutil.rmtree(self.ws, ignore_errors=True)

    def manifest(self, rows, name="manifest.json"):
        path = self.ws / "results" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"path_root": "..", "pairs": rows}))
        return path

    def row(self, model, ifc, **extra):
        return {"model": model, "ifc": str(ifc.relative_to(self.ws)), **extra}

    def run_cli(self, *argv):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return run_pipeline.main([str(a) for a in argv])

    def test_records_completed_failed_and_skipped_models(self):
        m = self.manifest([
            self.row("0", self.ifc, ifc_sha256=runs.sha256(self.ifc),
                     eligible_for_structural_pipeline_pilot=True, quality_flags=[]),
            self.row("bad", self.bad, eligible_for_structural_pipeline_pilot=True),
            self.row("empty", self.ifc, eligible_for_structural_pipeline_pilot=False,
                     quality_flags=["no_building_elements_in_canonical_ifc"])])
        out = self.ws / "results" / "runs" / "r1"
        self.assertEqual(self.run_cli("--manifest", m, "--dataset", "t", "--out", out), 1)

        rec = json.loads((out / "run.json").read_text())
        self.assertEqual(rec["status"], "completed_with_failures")
        self.assertEqual({r["model"]: r["status"] for r in rec["models"]},
                         {"0": "completed", "bad": "failed", "empty": "skipped"})
        self.assertEqual([f["model"] for f in rec["failures"]], ["bad"])
        self.assertTrue(rec["failures"][0]["error"])
        skipped = next(r for r in rec["models"] if r["model"] == "empty")
        self.assertEqual(skipped["quality_flags"], ["no_building_elements_in_canonical_ifc"])
        self.assertFalse((out / "models" / "empty").exists())

        prov = json.loads((out / "models" / "0" / "building.json").read_text())["provenance"]
        self.assertEqual(prov["input_ifc"], "data/set/0/0.ifc")
        self.assertEqual(prov["input_ifc_sha256"], runs.sha256(self.ifc))
        self.assertEqual(prov["code_sha256"], rec["code_sha256"])
        self.assertEqual(prov["settings"], rec["settings"])
        self.assertEqual(prov["dataset"], "t")
        self.assertNotIn("run_id", prov)

        self.assertEqual(runs.sha256(out / "source.zip"), rec["source_archive_sha256"])
        with zipfile.ZipFile(out / "source.zip") as z:
            self.assertIn("fp/build.py", z.namelist())
        self.assertEqual(rec["output_files"]["models/0/building.json"],
                         runs.sha256(out / "models" / "0" / "building.json"))
        self.assertTrue((out / "environment-freeze.txt").read_text().strip())

    def test_separate_processes_reproduce_every_output_file(self):
        m = self.manifest([self.row("0", self.ifc, ifc_sha256=runs.sha256(self.ifc))])
        trees = []
        for seed in (1, 2):
            out = self.ws / "results" / "runs" / f"seed{seed}"
            argv = [str(a) for a in ("--manifest", m, "--dataset", "t", "--out", out)]
            proc = subprocess.run(
                [sys.executable, "-c", CHILD.format(scripts=str(SCRIPTS), ws=str(self.ws), argv=argv)],
                env=dict(os.environ, PYTHONHASHSEED=str(seed)), capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            trees.append(tree(out / "models"))
        self.assertTrue(trees[0])
        self.assertEqual(trees[0], trees[1])

    def test_entry_point_matches_a_direct_build(self):
        m = self.manifest([self.row("0", self.ifc)])
        out = self.ws / "results" / "runs" / "cli"
        self.assertEqual(self.run_cli("--manifest", m, "--dataset", "t", "--out", out), 0)
        direct = self.ws / "results" / "direct"
        with contextlib.redirect_stdout(io.StringIO()):
            build("0", ifc_path=str(self.ifc), out_dir=str(direct), verbose=False)

        via_cli, plain = tree(out / "models" / "0"), tree(direct)
        self.assertEqual(sorted(via_cli), sorted(plain))
        for name in plain:
            if name != "building.json":
                self.assertEqual(via_cli[name], plain[name], name)
        with_provenance = json.loads((out / "models" / "0" / "building.json").read_text())
        with_provenance.pop("provenance")
        self.assertEqual(with_provenance, json.loads((direct / "building.json").read_text()))

    def test_parallel_execution_preserves_artifacts_and_failure_accounting(self):
        manifest = self.manifest([self.row('0', self.ifc), self.row('bad', self.bad),
            self.row('skip', self.ifc, eligible_for_structural_pipeline_pilot=False)])
        results = []
        for workers in (1, 2):
            out = self.ws/'results'/f'workers{workers}'
            self.assertEqual(self.run_cli('--manifest', manifest, '--dataset', 't', '--threads', 1,
                                          '--workers', workers, '--out', out), 1)
            record = json.loads((out/'run.json').read_text())
            self.assertEqual(record['execution_workers'], workers)
            self.assertEqual([(r['model'],r['status']) for r in record['models']],
                             [('0','completed'),('bad','failed'),('skip','skipped')])
            results.append(tree(out/'models'))
        self.assertEqual(results[0], results[1])

    def test_refuses_unsafe_or_inconsistent_requests(self):
        runs_dir = self.ws / "results" / "runs"
        good = self.manifest([self.row("0", self.ifc, ifc_sha256=runs.sha256(self.ifc))])
        wrong = self.manifest([self.row("0", self.ifc, ifc_sha256="0" * 64)], "wrong.json")
        twice = self.manifest([self.row("0", self.ifc), self.row("0", self.ifc)], "twice.json")
        outside = self.ws / "results" / "copy.ifc"
        shutil.copyfile(self.ifc, outside)
        (runs_dir / "full").mkdir(parents=True)
        (runs_dir / "full" / "keep.txt").write_text("existing output")

        cases = {
            "fingerprint differs": ["--manifest", wrong, "--dataset", "t", "--out", runs_dir / "x1"],
            "output under data/": ["--manifest", good, "--dataset", "t", "--out", self.ws / "data" / "o"],
            "output not empty": ["--manifest", good, "--dataset", "t", "--out", runs_dir / "full"],
            "input outside data/": ["--ifc", outside, "--dataset", "t", "--out", runs_dir / "x2"],
            "unknown model": ["--manifest", good, "--dataset", "t", "--models", "nope",
                              "--out", runs_dir / "x3"],
            "no dataset label": ["--manifest", good, "--out", runs_dir / "x4"],
            "duplicate identity": ["--manifest", twice, "--dataset", "t", "--out", runs_dir / "x5"],
            "bad setting": ["--manifest", good, "--dataset", "t", "--cell", "0",
                            "--out", runs_dir / "x6"],
        }
        for name, argv in cases.items():
            with self.subTest(name), self.assertRaises(SystemExit):
                self.run_cli(*argv)
        for name in ("x1", "x2", "x3", "x4", "x5", "x6"):
            self.assertFalse((runs_dir / name).exists(), name)
        self.assertEqual((runs_dir / "full" / "keep.txt").read_text(), "existing output")
        self.assertEqual(sorted(p.name for p in (self.ws / "data").iterdir()), ["set"])


class ClassCounts(unittest.TestCase):
    def test_tied_counts_are_ordered_the_same_in_every_process(self):
        classes = ["IfcWall", "IfcOpeningElement", "IfcDoor", "IfcWindow", "IfcSlab", "IfcBeam"]
        code = ("import json, sys; sys.path.insert(0, %r); from fp.build import class_counts; "
                "print(json.dumps(list(class_counts([{'ifc_class': c} for c in %r * 3]))))"
                % (str(SCRIPTS), classes))
        orders = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                 env=dict(os.environ, PYTHONHASHSEED=str(seed)), check=True).stdout
                  for seed in range(4)}
        self.assertEqual(len(orders), 1, orders)
        self.assertEqual(json.loads(orders.pop()), sorted(classes))


class Defaults(unittest.TestCase):
    def test_nonfinite_settings_are_rejected_before_starting_a_run(self):
        for key in ("cell", "min_room_area"):
            for value in (float("nan"), float("inf"), float("-inf")):
                cfg = dict(threads=1, cell=0.12, min_room_area=1.5, erode=0)
                cfg[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    runs.settings(Namespace(**cfg))

    def test_default_inputs_are_the_gni_corpus(self):
        blank = Namespace(dataset=None, manifest=None, input_dir=None, ifc=None)
        self.assertEqual(runs.dataset_label(blank), "gni")
        with self.assertRaises(ValueError):
            runs.dataset_label(Namespace(dataset=None, manifest=Path("m.json"),
                                         input_dir=None, ifc=None))

    def test_build_defaults_match_room_inference_defaults(self):
        rooms, built = inspect.signature(infer_rooms).parameters, inspect.signature(build).parameters
        self.assertEqual(built["cell"].default, rooms["cell"].default)
        self.assertEqual(built["min_room_area"].default, rooms["min_area"].default)
        self.assertIsNone(built["ifc_path"].default)
        self.assertIsNone(built["out_dir"].default)


if __name__ == "__main__":
    unittest.main(verbosity=2)
