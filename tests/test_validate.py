#!/usr/bin/env python
"""Checks for the validation measures: room matching, pooling, the elevation rule and a run.

The run checks use a scratch workspace holding a copy of one small ResBIM IFC, so they take
seconds and never write near the real data or results.

    python tests/test_validate.py
"""
from __future__ import annotations

import contextlib
import copy
import io
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fp import runs
from fp import validate as V
from fp.build import assign_storeys, storey_bands

SCRIPTS = Path(__file__).resolve().parent.parent
SAMPLE = runs.SOURCE_ROOT / "data" / "ResBIM-IFC" / "data" / "0" / "0.ifc"

CHILD = """
import sys
from pathlib import Path
sys.path.insert(0, {scripts!r})
from fp import runs, validate
runs.ROOT = Path({ws!r})
sys.exit(validate.main({argv!r}))
"""


class RoomMatching(unittest.TestCase):
    # Two truth rooms that both overlap the same small inferred room strongly and a large
    # inferred room weakly. Which one gets the small room decides the second match.
    T1, T2 = box(0, 0, 10, 6.2), box(0, 0, 10, 5.8)
    LARGE, SMALL = box(0, 0, 10, 10), box(0, 0, 10, 6)

    def test_iou_values(self):
        self.assertAlmostEqual(V.iou(self.T1, self.LARGE), 0.62)
        self.assertAlmostEqual(V.iou(self.T1, self.SMALL), 60 / 62)
        self.assertAlmostEqual(V.iou(self.T2, self.LARGE), 0.58)
        self.assertAlmostEqual(V.iou(self.T2, self.SMALL), 58 / 60)

    def test_matching_does_not_depend_on_truth_order(self):
        inferred = [self.LARGE, self.SMALL]
        forward = V.match_rooms(V.iou_matrix([self.T1, self.T2], inferred), 0.6)
        backward = V.match_rooms(V.iou_matrix([self.T2, self.T1], inferred), 0.6)
        self.assertEqual(len(forward), len(backward))
        self.assertEqual(sorted(round(v, 9) for *_, v in forward),
                         sorted(round(v, 9) for *_, v in backward))

    def test_old_rule_depends_on_truth_order(self):
        inferred = [self.LARGE, self.SMALL]
        forward = V.legacy_match(V.iou_matrix([self.T1, self.T2], inferred), 0.6)
        backward = V.legacy_match(V.iou_matrix([self.T2, self.T1], inferred), 0.6)
        self.assertEqual((len(forward), len(backward)), (1, 2))

    def test_exact_ties_do_not_change_scores_under_any_permutation(self):
        matrix = [[0.6, 0.6, None], [0.6, 0.0, 0.0], [0.0, 0.6, 0.9]]
        for rows in itertools.permutations(range(3)):
            for cols in itertools.permutations(range(3)):
                shuffled = [[matrix[r][c] for c in cols] for r in rows]
                matches = V.match_rooms(shuffled, 0.5)
                self.assertEqual(len(matches), 3, (rows, cols, matches))
                self.assertEqual(math.fsum(value for *_, value in matches), 2.1)

    def test_matching_maximises_count_before_iou_and_iou_breaks_count_ties(self):
        # The best single overlap blocks two eligible matches; recall has priority.
        matches = V.match_rooms([[0.9, 0.6], [0.6, 0.0]], 0.5)
        self.assertEqual(matches, [(0, 1, 0.6), (1, 0, 0.6)])
        # Both complete assignments are eligible, but one has higher total IoU.
        matches = V.match_rooms([[0.9, 0.8], [0.85, 0.6]], 0.5)
        self.assertEqual(matches, [(0, 1, 0.8), (1, 0, 0.85)])

    def test_bad_iou_inputs_are_rejected(self):
        for matrix in ([[float("nan")]], [[float("inf")]], [[1.1]], [[0.5], [0.4, 0.3]]):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                V.match_rooms(matrix, 0.5)
        self.assertEqual(V.match_rooms([[None, 0]], 0), [])

    def test_area_totals_are_not_rounded_before_pooling(self):
        result = V.room_test([box(0, 0, 1, 0.1234567)], [box(0, 0, 1, 0.2345678)])
        self.assertAlmostEqual(result["area_truth_m2"], 0.1234567, places=12)
        self.assertAlmostEqual(result["area_inferred_m2"], 0.2345678, places=12)

    def test_threshold_is_inclusive_and_matches_are_one_to_one(self):
        matrix = V.iou_matrix([box(0, 0, 10, 10)], [box(0, 0, 10, 5), box(0, 0, 10, 5)])
        matches = V.match_rooms(matrix, 0.5)
        self.assertEqual(len(matches), 1)
        self.assertAlmostEqual(matches[0][2], 0.5)
        self.assertEqual(V.match_rooms(matrix, 0.51), [])

    def test_room_test_counts_unmatched_rooms_and_overlapping_truth(self):
        truth = [box(0, 0, 5, 5), box(5, 0, 10, 5), box(4, 0, 6, 5)]
        inferred = [box(0, 0, 5, 5), box(5, 0, 10, 5), box(20, 20, 22, 22), box(30, 30, 32, 32)]
        rec = V.room_test(truth, inferred)
        self.assertEqual((rec["n_truth"], rec["n_inferred"]), (3, 4))
        for threshold in ("0.1", "0.5"):
            self.assertEqual(rec["by_iou_threshold"][threshold]["matched"], 2)
        self.assertAlmostEqual(rec["by_iou_threshold"]["0.5"]["iou_sum"], 2.0)
        self.assertAlmostEqual(rec["truth_overlap_m2"], 10.0)
        self.assertEqual(rec["iou_failures"], 0)

    def test_no_inferred_rooms(self):
        rec = V.room_test([box(0, 0, 5, 5), box(5, 0, 10, 5)], [])
        self.assertEqual(rec["by_iou_threshold"]["0.5"]["matched"], 0)
        self.assertEqual(rec["legacy_1_sep"], {"matched": 0, "mean_iou": 0.0,
                                               "area_truth": 50.0, "area_inferred": 0.0})


class Pooling(unittest.TestCase):
    def test_rates_carry_their_counts(self):
        self.assertEqual(V.ratio(3, 4), {"numerator": 3, "denominator": 4, "value": 0.75})
        self.assertEqual(V.ratio(0, 0), {"numerator": 0, "denominator": 0, "value": None})

    def test_pooled_median_is_not_the_median_of_model_medians(self):
        model_a, model_b = [0.0, 0.0, 0.0], [0.5]
        self.assertEqual(V.error_summary(model_a + model_b)["median"], 0.0)
        medians = [V.error_summary(model_a)["median"], V.error_summary(model_b)["median"]]
        self.assertEqual(V.median_or_none(medians), 0.25)

    def test_thresholds_are_inclusive(self):
        s = V.error_summary([0.01, 0.05, 0.07, 0.2])
        self.assertEqual(s["within_5pct"], {"numerator": 2, "denominator": 4, "value": 0.5})
        self.assertEqual(s["within_10pct"]["numerator"], 3)
        self.assertIsNone(V.error_summary([])["median"])


class InferenceFailures(unittest.TestCase):
    truth = [box(0, 0, 5, 5), box(5, 0, 10, 5)]
    barriers = [box(0, 0, 10, 0.1)]
    options = {"cell": 0.12, "min_area": 1.5, "erode": 0}

    def test_exception_keeps_truth_rooms_in_the_sample(self):
        with patch.object(V, "infer_rooms", side_effect=RuntimeError("test failure")):
            result = V.infer_and_test_rooms(self.truth, self.barriers, self.options)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["n_truth"], 2)
        self.assertEqual(result["n_inferred"], 0)
        self.assertEqual(result["by_iou_threshold"]["0.5"]["matched"], 0)
        self.assertIn("test failure", result["inference_diagnostics"]["error"])

    def test_grid_limit_is_a_failure_with_no_allocation(self):
        result = V.infer_and_test_rooms(self.truth, [box(0, 0, 1000, 1000)], self.options)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["inference_diagnostics"]["status"], "grid_limit")
        self.assertEqual(result["n_truth"], 2)

    def test_an_empty_but_completed_inference_is_not_a_failure(self):
        result = V.infer_and_test_rooms(self.truth, self.barriers, self.options)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["n_inferred"], 0)


class ElevationRule(unittest.TestCase):
    def test_band_or_nearest_end(self):
        bands = [(-1.35, 2.65), (2.65, 5.65)]
        self.assertEqual(V.elevation_guess(0.0, bands), (0, False))
        self.assertEqual(V.elevation_guess(2.65, bands), (1, False))
        self.assertEqual(V.elevation_guess(9.0, bands), (1, True))
        self.assertEqual(V.elevation_guess(-5.0, bands), (0, True))

    def test_same_rule_as_the_pipeline(self):
        storeys = [{"step_id": 1, "elevation_m": 0.0}, {"step_id": 2, "elevation_m": 3.0},
                   {"step_id": 3, "elevation_m": 6.5}]
        heights = [-4.0, -0.36, -0.35, 0.0, 2.64, 2.65, 3.5, 6.14, 6.15, 12.0]
        data = {"storeys": storeys,
                "elements": [{"step_id": 100 + i, "storey_step_id": None, "bbox_min": [0.0, 0.0, z]}
                             for i, z in enumerate(heights)]}
        assigned = assign_storeys(data)
        bands = storey_bands(storeys, min(heights), max(heights))
        for i, z in enumerate(heights):
            index, how = assigned[100 + i]
            self.assertEqual((index, how == "inferred_clamped"), V.elevation_guess(z, bands), z)


class ProjectionMeasures(unittest.TestCase):
    """Analytic meshes exercise the validator, projection and inference together.

    Only IFC extraction and declared-quantity lookup are replaced. These fixtures make
    ownership, height and expected projected area independently specified inputs.
    """
    settings = {"threads": 1, "cell": 0.12, "min_room_area": 1.5, "erode": 0,
                "cut_height": 1.2, "cut_thickness": 0.01}

    @staticmethod
    def cuboid(step, cls, bounds, owner=100):
        x0, y0, z0, x1, y1, z1 = bounds
        return {"step_id": step, "ifc_class": cls, "has_geometry": True,
                "storey_step_id": owner, "storey_link": "IfcRelContainedInSpatialStructure",
                "bbox_min": [x0, y0, z0], "bbox_max": [x1, y1, z1],
                "_verts": np.array([(x0,y0,z0), (x1,y0,z0), (x1,y1,z0), (x0,y1,z0),
                                    (x0,y0,z1), (x1,y0,z1), (x1,y1,z1), (x0,y1,z1)]),
                "_faces": [(0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),
                           (1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7)]}

    def fixture(self, barriers=True):
        # Both rooms belong upstairs. Their enclosing walls belong downstairs but
        # extend through both floors. Room inference must still see them upstairs.
        elements = [self.cuboid(1, "IfcSpace", (.2,.2,3.,3.9,3.8,5.8), owner=200),
                    self.cuboid(2, "IfcSpace", (4.1,.2,3.,7.8,3.8,5.8), owner=200)]
        if barriers:
            walls = [(0,0,0,8,.2,6), (0,3.8,0,8,4,6), (0,0,0,.2,4,6),
                     (7.8,0,0,8,4,6), (3.9,0,0,4.1,4,6)]
            elements += [self.cuboid(10+i, "IfcWall", bounds) for i, bounds in enumerate(walls)]
        return {"schema": "IFC4", "storey_source": "IfcBuildingStorey", "elements": elements,
                "storeys": [{"step_id": 100, "name": "Ground", "elevation_m": 0.},
                            {"step_id": 200, "name": "Upper", "elevation_m": 3.}]}

    def validate_fixture(self, data, settings=None, area=None, volume=None):
        class Element:
            Representation = True
            ObjectPlacement = None

            def __init__(self, step):
                self.step = step

        class File:
            def by_id(self, step):
                return Element(step)

            def by_type(self, name):
                return []  # No independent placement/direct-relationship evidence in this fixture.

        def lookup(values):
            return lambda e: ({"status": "ok", "value_si": values[e.step]} if e.step in values
                              else {"status": "missing"})

        with patch.object(V, "extract", return_value=(copy.deepcopy(data), File())), \
                patch.object(V, "declared_footprint_area", side_effect=lookup(area or {})), \
                patch.object(V, "declared_volume", side_effect=lookup(volume or {})):
            return V.validate(Path("analytic.ifc"), settings or self.settings)

    def test_rooms_use_visible_barriers_across_storey_ownership(self):
        row = self.validate_fixture(self.fixture())
        tests = row["room_inference"]["storeys"]
        self.assertEqual(len(tests), 1)
        self.assertEqual(tests[0]["storey_index"], 1)
        self.assertEqual(tests[0]["status"], "completed")
        self.assertEqual(tests[0]["n_inferred"], 2)
        self.assertEqual(tests[0]["by_iou_threshold"]["0.5"]["matched"], 2)
        self.assertEqual(tests[0]["section_projection"]["band_m"], [4.2, 4.21])

    def test_no_visible_barriers_keeps_reference_eligible(self):
        row = self.validate_fixture(self.fixture(barriers=False))
        summary = V.summarise([row], row.pop("_area_errors"), row.pop("_volume_errors"))
        rooms = summary["room_inference"]
        self.assertEqual(rooms["storeys_tested"], 1)
        self.assertEqual(rooms["completion"], V.ratio(1, 1))
        self.assertEqual(rooms["by_iou_threshold"]["0.5"]["recall"], V.ratio(0, 2))
        self.assertEqual(rooms["by_iou_threshold"]["0.5"]["precision"], V.ratio(0, 0))
        self.assertEqual(row["room_inference"]["storeys"][0]["inference_diagnostics"]["status"],
                         "no_barriers")

    def test_a_wall_whose_cut_fails_stands_in_with_its_outline(self):
        # the walls belong downstairs; one fails the upstairs cut, and its plan outline stands in
        data = self.fixture()
        data["elements"][-1]["_faces"] = data["elements"][-1]["_faces"][:-2]
        row = self.validate_fixture(data)
        rooms = row["room_inference"]
        self.assertEqual(rooms["completion"], V.ratio(1, 1))
        section = rooms["storeys"][0]["section_projection"]
        self.assertEqual(section["excluded"], [])
        self.assertEqual([f["step_id"] for f in section["fallbacks"]], [14])

    def test_a_barrier_without_a_mesh_fails_storey_without_dropping_references(self):
        # model_129's curtain walls have no mesh at all, so nothing can stand in for them: the
        # storey fails, and its reference rooms stay in the denominator
        data = self.fixture()
        wall = data["elements"][-1]
        del wall["_verts"], wall["_faces"]
        wall["storey_step_id"] = 200
        row = self.validate_fixture(data)
        rooms = row["room_inference"]
        self.assertEqual(rooms["completion"], V.ratio(0, 1))
        test = rooms["storeys"][0]
        self.assertEqual((test["n_truth"], test["n_inferred"]), (2, 0))
        self.assertEqual(test["inference_diagnostics"]["status"], "incomplete_section_barriers")
        self.assertEqual(test["section_projection"]["excluded"][0]["step_id"], 14)
        summary = V.summarise([row], row.pop("_area_errors"), row.pop("_volume_errors"))
        self.assertEqual(summary["room_inference"]["truth_rooms_on_failed_storeys"], 2)
        self.assertEqual(summary["room_inference"]["by_iou_threshold"]["0.5"]["recall"], V.ratio(0, 2))

    def test_projection_exception_fails_storey_without_discarding_model(self):
        with patch.object(V, "section_elements", side_effect=RuntimeError("bad mesh")):
            row = self.validate_fixture(self.fixture())
        self.assertEqual(row["room_inference"]["completion"], V.ratio(0, 1))
        test = row["room_inference"]["storeys"][0]
        self.assertEqual(test["n_truth"], 2)
        self.assertEqual(test["inference_diagnostics"]["status"], "section_projection_error")
        self.assertIn("bad mesh", test["section_projection"]["error"])
        self.assertEqual(row["footprint_extraction"], {**V.ratio(7, 7), "missing_by_class": {}})

    def test_quantity_silhouette_is_complete_and_independent_of_cut_height(self):
        data = self.fixture(barriers=False)
        wedge = self.cuboid(20, "IfcSlab", (0,0,0,2,1,2))
        wedge["_verts"] = np.array([(0,0,0),(2,0,0),(0,0,2),(0,1,0),(2,1,0),(0,1,2)])
        wedge["_faces"] = [(0,1,2),(3,5,4),(0,3,4),(0,4,1),(1,4,5),(1,5,2),(2,5,3),(2,3,0)]
        data["elements"].append(wedge)
        for height in (0.25, 1.25, 8.):
            with self.subTest(height=height):
                row = self.validate_fixture(data, {**self.settings, "cut_height": height},
                                            area={20: 2.}, volume={20: 2.})
                self.assertEqual(row["_area_errors"], [0.])
                # The wedge's actual volume is 2 m3, whereas its silhouette x full
                # height is 4 m3. Keeping this mismatch makes the proxy's limit visible.
                self.assertEqual(row["_volume_errors"], [1.])


@unittest.skipUnless(SAMPLE.is_file(), f"needs {SAMPLE}")
class ValidationRun(unittest.TestCase):
    def setUp(self):
        self.ws = Path(tempfile.mkdtemp(prefix="fp-validate-"))
        self.ifc = self.ws / "data" / "set" / "0" / "0.ifc"
        self.ifc.parent.mkdir(parents=True)
        shutil.copyfile(SAMPLE, self.ifc)
        self.saved_root, runs.ROOT = runs.ROOT, self.ws

    def tearDown(self):
        runs.ROOT = self.saved_root
        shutil.rmtree(self.ws, ignore_errors=True)

    def argv(self, out):
        return [str(a) for a in ("--ifc", self.ifc, "--dataset", "t", "--out", out)]

    def test_run_record_and_internally_consistent_counts(self):
        out = self.ws / "results" / "runs" / "v1"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(V.main(self.argv(out)), 0)
        record = json.loads((out / "run.json").read_text())
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["output_files"]["validation.json"], runs.sha256(out / "validation.json"))

        result = json.loads((out / "validation.json").read_text())
        self.assertEqual(result["definitions_version"], V.DEFINITIONS_VERSION)
        self.assertEqual(result["code_sha256"], record["code_sha256"])
        model, summary = result["per_model"][0], result["summary"]
        self.assertEqual(model["input_ifc_sha256"], runs.sha256(self.ifc))

        links = model["storey_links"]
        self.assertEqual(sum(links["by_route"].values()), links["all_elements"]["denominator"])
        self.assertEqual(links["all_elements"]["numerator"],
                         links["all_elements"]["denominator"] - links["by_route"].get("none", 0))
        assignment = model["storey_assignment"]
        self.assertEqual(sum(assignment["by_method"].values()), assignment["denominator"])
        self.assertEqual(assignment["denominator"], links["elements_with_geometry"]["denominator"])
        self.assertLessEqual(model["elevation_fallback"]["denominator"],
                             links["elements_with_geometry"]["numerator"])
        self.assertLessEqual(model["placement_agreement"]["denominator"],
                             links["all_elements"]["denominator"])
        extraction = model["geometry_extraction"]
        self.assertEqual(extraction["denominator"] - extraction["numerator"],
                         sum(extraction["missing_by_class"].values()))

        self.assertEqual(model["room_inference"]["storeys_tested"], 0)   # ResBIM has no IfcSpace
        self.assertEqual(model["footprint_area"]["n"], 0)                # nor declared quantities
        self.assertEqual(summary["models"], 1)
        self.assertEqual(summary["model_execution"]["completion"], V.ratio(1, 1))
        self.assertEqual(summary["footprint_extraction"], model["footprint_extraction"])
        for name in ("footprint_area", "volume"):
            self.assertEqual(sum(model[name]["comparison_status"].values()),
                             sum(model[name]["quantity_status"].values()))
            self.assertEqual(model[name]["comparison_status"].get("compared", 0), model[name]["n"])
        self.assertEqual(summary["storey_links"]["all_elements"], links["all_elements"])
        self.assertEqual(summary["legacy_1_sep"]["completeness"]["placed"], assignment["numerator"])

    def test_separate_processes_write_identical_results(self):
        results = []
        for seed in (1, 2):
            out = self.ws / "results" / "runs" / f"seed{seed}"
            code = CHILD.format(scripts=str(SCRIPTS), ws=str(self.ws), argv=self.argv(out))
            proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                  env=dict(os.environ, PYTHONHASHSEED=str(seed)))
            self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            results.append((out / "validation.json").read_bytes())
        self.assertEqual(results[0], results[1])

    def test_parallel_validation_preserves_results(self):
        results = []
        for workers in (1, 2):
            out = self.ws/'results'/f'workers{workers}'
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(V.main(self.argv(out)+['--workers',str(workers)]), 0)
            results.append((out/'validation.json').read_bytes())
        self.assertEqual(results[0], results[1])

    def test_failed_model_is_explicit_in_execution_denominator(self):
        out = self.ws / "results" / "runs" / "failure"
        with patch.object(V, "validate", side_effect=RuntimeError("model failed")), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(V.main(self.argv(out)), 1)
        result = json.loads((out / "validation.json").read_text())
        self.assertEqual(result["summary"]["model_execution"]["completion"], V.ratio(0, 1))
        self.assertEqual(result["summary"]["room_inference"]["completion"], V.ratio(0, 0))
        self.assertEqual(result["per_model"], [])

    def test_failed_storey_remains_in_pooled_recall(self):
        row = V.validate(self.ifc, {"threads": 1, "cell": 0.12, "min_room_area": 1.5, "erode": 0})
        truth = InferenceFailures.truth
        successful = {**V.room_test(truth, truth), "status": "completed"}
        with patch.object(V, "infer_rooms", side_effect=RuntimeError("test failure")):
            failed = V.infer_and_test_rooms(truth, InferenceFailures.barriers,
                                           InferenceFailures.options)
        row["room_inference"] = {"storeys_tested": 2, "inference_failures": 1,
                                 "completion": V.ratio(1, 2), "storeys": [successful, failed]}
        summary = V.summarise([row], row.pop("_area_errors"), row.pop("_volume_errors"))
        rooms = summary["room_inference"]
        self.assertEqual(rooms["truth_rooms"], 4)
        self.assertEqual(rooms["truth_rooms_on_failed_storeys"], 2)
        self.assertEqual(rooms["by_iou_threshold"]["0.5"]["recall"], V.ratio(2, 4))
        self.assertEqual(rooms["by_iou_threshold"]["0.5"]["precision"], V.ratio(2, 2))
        self.assertEqual(rooms["completion"], V.ratio(1, 2))
        self.assertEqual(summary["legacy_1_sep"]["room_inference"]["truth_rooms"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
