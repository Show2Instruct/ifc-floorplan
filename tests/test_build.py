"""Regression tests for graph identity and silently omitted inference limits."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from shapely.geometry import box, Polygon
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fp.build import build, reconcile_rooms
from fp.extract import footprint, infer_rooms, poly_geometry
from fp.projections import section_band
from fp import runs


def element(step, cls, x):
    # Coplanar triangles suffice to isolate the identity bug from IFC meshing.
    return {"step_id": step, "ifc_class": cls, "label": str(step), "has_geometry": True,
            "storey_step_id": 100, "storey_link": "IfcRelContainedInSpatialStructure",
            "bbox_min": [x, 0., 0.], "bbox_max": [x+1, 1., 0.],
            "_verts": np.array([[x, 0., 0.], [x+1, 0., 0.], [x+1, 1., 0.], [x, 1., 0.]]),
            "_faces": [(0, 1, 2), (0, 2, 3)]}


def prism(step, x0, y0, x1, y1, z0=0., z1=3., cls="IfcWall", owner=100):
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    verts = np.array([(x, y, z0) for x, y in corners] + [(x, y, z1) for x, y in corners])
    faces = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6)]
    for i in range(4):
        j = (i + 1) % 4
        faces += [(i, j, j + 4), (i, j + 4, i + 4)]
    return {"step_id": step, "ifc_class": cls, "label": str(step), "has_geometry": True,
            "storey_step_id": owner, "storey_link": "IfcRelContainedInSpatialStructure",
            "bbox_min": [x0, y0, z0], "bbox_max": [x1, y1, z1],
            "_verts": verts, "_faces": faces}


def room_walls(owner=100):
    """Four walls around a 3 x 2.8 m room."""
    return [prism(1, 0, 0, 3.2, .2, owner=owner), prism(2, 0, 3.0, 3.2, 3.2, owner=owner),
            prism(3, 0, 0, .2, 3.2, owner=owner), prism(4, 3.0, 0, 3.2, 3.2, owner=owner)]


def fixture(elements, elevations):
    return {"file": "/fixture.ifc", "schema": "IFC4", "unit_scale_to_m": 1.,
            "storey_source": "IfcBuildingStorey", "spaces": [], "relations": [],
            "elements": elements,
            "storeys": [{"step_id": 100 + 100 * i, "name": f"L{i}", "elevation_m": z,
                         "source": "IfcBuildingStorey"} for i, z in enumerate(elevations)]}


class BuildGraph(unittest.TestCase):
    def test_visible_geometry_is_separate_from_unique_storey_ownership(self):
        elements = []
        for sid, x, owner in [(1,0.,100),(2,1.,200)]:
            e = element(sid, 'IfcWall', x)
            e['_verts'] = np.vstack([e['_verts'], e['_verts'] + np.array([0.,0.,6.])])
            e['_faces'] += [(4,5,6),(4,6,7),(0,1,5),(0,5,4),(1,2,6),(1,6,5),
                            (2,3,7),(2,7,6),(3,0,4),(3,4,7)]
            e['bbox_max'][2] = 6.
            e['storey_step_id'] = owner
            elements.append(e)
        data = {'file':'/fixture.ifc','schema':'IFC4','unit_scale_to_m':1.,
                'storey_source':'IfcBuildingStorey','spaces':[],'relations':[], 'elements':elements,
                'storeys':[{'step_id':100,'name':'Ground','elevation_m':0.,'source':'IfcBuildingStorey'},
                           {'step_id':200,'name':'Upper','elevation_m':3.,'source':'IfcBuildingStorey'}]}
        with tempfile.TemporaryDirectory(dir=runs.ROOT) as directory:
            with patch('fp.build.extract', return_value=(data,None)):
                building = build('fixture', out_dir=directory, verbose=False)
            self.assertEqual(building['n_elements'], 2)
            for index in range(2):
                plan = json.loads((Path(directory)/f'storey_{index:02d}.json').read_text())
                self.assertEqual(plan['format_version'],3)
                self.assertEqual(plan['n_elements'],1)
                self.assertEqual(plan['n_visible_elements'],2)
                self.assertEqual({p['step_id']:p['owner_storey_index'] for p in plan['projections']}, {1:0,2:1})
                self.assertEqual(plan['elements'][0]['footprint_projection']['height_scope'],'full_height')
                self.assertEqual({tuple(sorted((r['source'],r['target']))) for r in plan['spatial_relations']
                                  if r['type']=='adjacent_to'}, {(1,2)})

    def test_cut_is_lowered_to_stay_below_the_next_declared_level(self):
        storeys = [{"elevation_m": 0.}, {"elevation_m": .6}, {"elevation_m": 4.}]
        low, high, height, note = section_band(storeys, 0, 1.2, .01)
        self.assertAlmostEqual(low, .3)
        self.assertAlmostEqual(height, .3)
        self.assertIn("cut_lowered", note)
        # a storey with room for the full cut keeps it
        low, high, height, note = section_band(storeys, 1, 1.2, .01)
        self.assertAlmostEqual(low, 1.8)
        self.assertAlmostEqual(high, 1.81)
        self.assertIsNone(note)

    def test_band_base_spans_from_above_the_floor_to_the_cut(self):
        # a plan shows what stands below its cut plane: model_100's 0.00-0.90 m walls vanish
        # from a thin slice at 1.2 m and the rooms they separate merge.
        storeys = [{"elevation_m": 3.}, {"elevation_m": 6.}]
        low, high, height, note = section_band(storeys, 0, 1.2, .01, band_base=.05)
        self.assertAlmostEqual(low, 3.05)
        self.assertAlmostEqual(high, 4.2)
        self.assertAlmostEqual(height, 1.2)
        # a base at or above the cut leaves nothing to project
        low, high, height, note = section_band(storeys, 0, 1.2, .01, band_base=1.199)
        self.assertIsNone(low)
        self.assertIn("leaves no height", note)

    def test_levels_too_close_get_no_section_rather_than_the_level_above(self):
        # model_103's GL sits 4 cm under EG; a 1.2 m cut there reported EG's rooms on a
        # level that holds only columns.
        data = fixture(room_walls(owner=200), [0., .04])
        with tempfile.TemporaryDirectory(dir=runs.ROOT) as directory:
            with patch("fp.build.extract", return_value=(data, None)):
                build("fixture", out_dir=directory, verbose=False)
            lower = json.loads((Path(directory) / "storey_00.json").read_text())
            upper = json.loads((Path(directory) / "storey_01.json").read_text())
        self.assertEqual(lower["n_rooms"], 0)
        self.assertEqual(lower["room_inference"]["status"], "no_section_band")
        self.assertIn("levels_too_close", lower["room_inference"]["reason"])
        self.assertIsNone(lower["section_projection"]["band_m"])
        self.assertEqual(upper["n_rooms"], 1)

    def test_barrier_whose_section_fails_falls_back_to_its_silhouette(self):
        # model_27's windows failed with inconsistent cut orientation, leaving a hole in the
        # wall at every opening, and its rooms drained into one another.
        data = fixture(room_walls(), [0.])
        with tempfile.TemporaryDirectory(dir=runs.ROOT) as directory:
            with patch("fp.build.extract", return_value=(data, None)), \
                 patch("fp.projections.project_band",
                       return_value=(None, {"status": "inconsistent_cut_orientation"})):
                build("fixture", out_dir=directory, verbose=False)
            plan = json.loads((Path(directory) / "storey_00.json").read_text())
        self.assertEqual(plan["n_rooms"], 1)
        section = plan["section_projection"]
        self.assertEqual(len(section["fallbacks"]), 4)
        self.assertTrue(section["barriers_complete"])
        self.assertFalse(section["barriers_all_sectioned"])
        self.assertEqual(plan["room_inference"]["barriers_from_full_height_silhouette"], 4)
        self.assertEqual({p["projection"]["fallback"] for p in plan["projections"]},
                         {"full_height_silhouette"})

    def test_a_column_encloses_a_room_like_a_wall(self):
        # model_175's frame defines its rooms with columns, and model_14's facade leaks at
        # every gap between curtain walls where a column stands.
        from fp.projections import BARRIERS
        self.assertIn("IfcColumn", BARRIERS)
        gap = [prism(1, 0, 0, 3.2, .2), prism(2, 0, 3.0, 3.2, 3.2),
               prism(3, 0, 0, .2, 3.2), prism(4, 3.0, 0, 3.2, 1.4),
               prism(5, 3.0, 1.8, 3.2, 3.2)]
        column = prism(6, 3.0, 1.4, 3.2, 1.8, cls="IfcColumn")
        for elements, expected in ((gap, 0), (gap + [column], 1)):
            data = fixture([copy.deepcopy(e) for e in elements], [0.])
            with tempfile.TemporaryDirectory(dir=runs.ROOT) as directory:
                with patch("fp.build.extract", return_value=(data, None)):
                    build("fixture", out_dir=directory, verbose=False)
                plan = json.loads((Path(directory) / "storey_00.json").read_text())
            self.assertEqual(plan["n_rooms"], expected)

    def test_a_region_inside_a_declared_space_is_that_space_read_twice(self):
        declared = [{"id": "space_1", "step_id": 1, "name": "Lobby", "source": "IfcSpace",
                     "_poly": box(0, 0, 4, 4)}]
        inferred = [{"id": "room_0_0", "name": "Room 1", "source": "inferred_from_wall_geometry",
                     "_poly": box(.1, .1, 3.9, 3.9)}]
        rooms, notes = reconcile_rooms(declared, inferred)
        self.assertEqual([r["id"] for r in rooms], ["space_1"])
        self.assertEqual(notes["absorbed_inferred_duplicate"], 1)

    def test_walls_dividing_one_declared_space_produce_separate_rooms(self):
        # The file names one space where the plan shows two. Absorbing the parts would erase a
        # wall that is really there; this is what a "split" verdict marks during annotation.
        declared = [{"id": "space_1", "step_id": 1, "name": "Office", "source": "IfcSpace",
                     "_poly": box(0, 0, 4, 4)}]
        inferred = [{"id": "room_0_0", "name": "Room 1", "source": "inferred_from_wall_geometry",
                     "area_m2": 7.6, "_poly": box(.1, .1, 1.9, 3.9)},
                    {"id": "room_0_1", "name": "Room 2", "source": "inferred_from_wall_geometry",
                     "area_m2": 7.6, "_poly": box(2.1, .1, 3.9, 3.9)}]
        rooms, notes = reconcile_rooms(declared, inferred)
        self.assertEqual(len(rooms), 2)
        self.assertNotIn("space_1", [r["id"] for r in rooms])
        self.assertTrue(all(r["subdivides_declared_space"] == 1 for r in rooms))
        self.assertEqual(notes["declared_subdivided"], 1)
        # the parts keep their own geometry, not the declared space's
        self.assertAlmostEqual(sum(r["_poly"].area for r in rooms), 2 * 1.8 * 3.8)

    def test_a_room_the_file_never_declared_is_kept_whole(self):
        declared = [{"id": "space_1", "step_id": 1, "name": "Lobby", "source": "IfcSpace",
                     "_poly": box(0, 0, 4, 4)}]
        inferred = [{"id": "room_0_0", "name": "Room 1", "source": "inferred_from_wall_geometry",
                     "_poly": box(5, 0, 9, 4)}]
        rooms, notes = reconcile_rooms(declared, inferred)
        self.assertEqual(sorted(r["id"] for r in rooms), ["room_0_0", "space_1"])
        kept = next(r for r in rooms if r["id"] == "room_0_0")
        self.assertEqual(kept["overlap_with_declared_space"], 0.0)
        self.assertAlmostEqual(kept["_poly"].area, 16.)
        self.assertEqual(notes["kept_outside_declared_spaces"], 1)

    def test_a_region_only_grazing_a_declared_space_stays_a_separate_room(self):
        # a declared space bleeding over a wall centreline must not swallow the room next door
        declared = [{"id": "space_1", "step_id": 1, "name": "Hall", "source": "IfcSpace",
                     "_poly": box(0, 0, 4, 4)}]
        inferred = [{"id": "room_0_0", "name": "Room 1", "source": "inferred_from_wall_geometry",
                     "_poly": box(3.8, 0, 7.8, 4)}]
        rooms, _ = reconcile_rooms(declared, inferred)
        self.assertEqual(len(rooms), 2)
        self.assertEqual(next(r for r in rooms if r["id"] == "room_0_0")["overlap_with_declared_space"], .05)

    def test_room_areas_sit_on_the_wall_faces_not_the_raster_margin(self):
        # The raster grows barriers by half a cell and blocks a cell that merely touches one,
        # which cost about 0.15 m off every side: a 3x4 m room measured 10.27 and a 1 m corridor
        # 7.00. A corridor is mostly edge, so the error hit hardest where it mattered most.
        def walls(w, h, t=.2):
            return [box(-t, -t, w + t, 0), box(-t, h, w + t, h + t),
                    box(-t, -t, 0, h + t), box(w, -t, w + t, h + t)]
        for w, h in ((3, 4), (2, 2), (1., 10.), (.9, 6.)):
            with self.subTest(room=f"{w}x{h}"):
                barriers = walls(w, h)
                rooms = infer_rooms(barriers, unary_union(barriers).bounds)
                self.assertEqual(len(rooms), 1)
                self.assertAlmostEqual(rooms[0]["area_m2"], w * h, delta=.05 * w * h * .2 + .02)
                # the cell count is kept, and is the smaller, pessimistic figure
                self.assertLess(rooms[0]["raster_area_m2"], rooms[0]["area_m2"])

    def test_growing_back_to_the_walls_does_not_merge_rooms_across_a_thin_partition(self):
        barriers = [box(-.2, -.2, 6.2, 0), box(-.2, 4, 6.2, 4.2), box(-.2, -.2, 0, 4.2),
                    box(6, -.2, 6.2, 4.2), box(2.9, -.2, 3.1, 4.2)]
        rooms = infer_rooms(barriers, unary_union(barriers).bounds)
        self.assertEqual(len(rooms), 2)
        for room in rooms:
            self.assertAlmostEqual(room["area_m2"], 2.9 * 4, delta=.1)

    def test_a_railed_balcony_is_recorded_and_drawn_although_it_is_below_the_cut(self):
        # The railings (0-1.1 m) and the slab never reach the 1.2 m section, so the plan showed no
        # balcony even though the IFC held every part of it.
        elements = room_walls() + [
            prism(5, -.2, -.2, 3.4, 5.2, z0=-.3, z1=0., cls="IfcSlab"),
            prism(6, -.2, 5.1, 3.4, 5.2, z0=0., z1=1.1, cls="IfcRailing"),
            prism(7, -.2, 3.2, -.1, 5.2, z0=0., z1=1.1, cls="IfcRailing"),
            prism(8, 3.3, 3.2, 3.4, 5.2, z0=0., z1=1.1, cls="IfcRailing")]
        data = fixture(elements, [0.])
        with tempfile.TemporaryDirectory(dir=runs.ROOT) as directory:
            with patch("fp.build.extract", return_value=(data, None)):
                building = build("fixture", out_dir=directory, verbose=False)
            plan = json.loads((Path(directory) / "storey_00.json").read_text())
            drawing = (Path(directory) / "storey_00.svg").read_text()
        self.assertEqual(plan["n_rooms"], 1)
        self.assertEqual(len(plan["outdoor_spaces"]), 1)
        self.assertGreater(plan["outdoor_spaces"][0]["area_m2"], 5.)
        self.assertEqual(building["graph_counts"]["outdoor_spaces"], 1)
        self.assertIn("Outdoor space 1", drawing)
        self.assertIn('stroke-dasharray="4 3"', drawing)      # the slab below the cut is drawn

    def test_sloping_roof_silhouette_is_not_truncated_to_the_highest_element_base(self):
        e = element(1,'IfcSlab',0.)
        e['_verts'] = np.array([(0,0,0),(2,0,0),(0,0,2),(0,1,0),(2,1,0),(0,1,2)])
        e['_faces'] = [(0,1,2),(3,5,4),(0,3,4),(0,4,1),(1,4,5),(1,5,2),(2,5,3),(2,3,0)]
        e['bbox_max'] = [2.,1.,2.]
        data = {'file':'/fixture.ifc','schema':'IFC4','unit_scale_to_m':1.,
                'storey_source':'IfcBuildingStorey','spaces':[],'relations':[], 'elements':[e],
                'storeys':[{'step_id':100,'name':'Roof','elevation_m':0.,'source':'IfcBuildingStorey'}]}
        with tempfile.TemporaryDirectory(dir=runs.ROOT) as directory:
            with patch('fp.build.extract', return_value=(data,None)):
                build('fixture', out_dir=directory, verbose=False)
            plan = json.loads((Path(directory)/'storey_00.json').read_text())
            geometry = plan['elements'][0]['footprint_geometry'][0]
            self.assertAlmostEqual(Polygon(geometry[0],geometry[1:]).area, 2.)
            self.assertAlmostEqual(plan['projections'][0]['area_m2'], .8, places=5)

    def test_interleaved_spaces_do_not_change_element_adjacency(self):
        # Walls 1 and 3 touch. Wall 2 is distant. A space placed between them
        # used to shift the footprint associated with wall 2 and drop wall 3's.
        elements = [element(1, "IfcWall", 0), element(10, "IfcSpace", 0),
                    element(2, "IfcWall", 10), element(3, "IfcWall", 1)]
        data = {"file": "/fixture.ifc", "schema": "IFC4", "unit_scale_to_m": 1.,
                "storey_source": "IfcBuildingStorey", "spaces": [10], "relations": [],
                "storeys": [{"step_id": 100, "name": "Ground", "elevation_m": 0.,
                             "source": "IfcBuildingStorey"}], "elements": elements}
        with tempfile.TemporaryDirectory(dir=runs.ROOT) as directory:
            for order in (elements, list(reversed(elements))):
                out = Path(directory) / ("a" if order[0]["step_id"] == 1 else "b")
                with patch("fp.build.extract", return_value=(copy.deepcopy(dict(data, elements=order)), None)):
                    build("fixture", out_dir=str(out), verbose=False, cut_height=0.)
                storey = json.loads((out / "storey_00.json").read_text())
                edges = {tuple(sorted((r["source"], r["target"]))) for r in storey["spatial_relations"]
                         if r["type"] == "adjacent_to"}
                self.assertEqual(edges, {(1, 3)})

    def test_direct_build_does_not_overwrite_existing_outputs(self):
        with tempfile.TemporaryDirectory(dir=runs.ROOT) as directory:
            sentinel = Path(directory) / "building.json"
            sentinel.write_text("existing evidence")
            with self.assertRaisesRegex(ValueError, "already contains"):
                build("fixture", out_dir=directory, verbose=False)
            self.assertEqual(sentinel.read_text(), "existing evidence")


class GeometryDiagnostics(unittest.TestCase):
    def test_authoritative_serialization_keeps_submillimetre_gaps(self):
        # Rounding both x values to 1.0 pinches the ring closed at its notch.
        polygon = Polygon([(0, 0), (2, 0), (2, 2), (1.0004, 2), (1.0004, 1),
                           (1.0002, 1), (1.0002, 2), (0, 2)])
        self.assertTrue(polygon.is_valid)
        rings = poly_geometry(polygon)[0]
        restored = Polygon(rings[0], rings[1:])
        self.assertTrue(restored.is_valid)
        self.assertTrue(restored.equals_exact(polygon, 0))
        rounded = poly_geometry(polygon, ndigits=3)[0]
        self.assertFalse(Polygon(rounded[0], rounded[1:]).is_valid)

    def test_full_height_fallback_is_explicit(self):
        e = element(1, "IfcWall", 0)
        diagnostics = {}
        polygon, _ = footprint(e["_verts"], e["_faces"], z_lo=5., z_hi=8., diagnostics=diagnostics)
        self.assertEqual(polygon.area, 1.)
        self.assertEqual(diagnostics["height_scope"], "full_height_fallback")

    def test_oversize_grid_is_not_an_unexplained_zero_room_result(self):
        diagnostics = {}
        rooms = infer_rooms([box(0, 0, 1000, 1000)], (0, 0, 1000, 1000), diagnostics=diagnostics)
        self.assertEqual(rooms, [])
        self.assertEqual(diagnostics["status"], "grid_limit")
        self.assertGreater(diagnostics["grid_cells"], 6_000_000)

    def test_room_area_describes_the_polygon_and_raster_area_is_separate(self):
        diagnostics = {}
        barriers = [box(0, 0, 5, .2), box(0, 4.8, 5, 5), box(0, 0, .2, 5), box(4.8, 0, 5, 5)]
        rooms = infer_rooms(barriers, (0, 0, 5, 5), diagnostics=diagnostics)
        self.assertEqual(len(rooms), 1)
        self.assertEqual(rooms[0]["area_m2"], round(rooms[0]["polygon"].area, 2))
        self.assertIn("raster_area_m2", rooms[0])
        self.assertEqual(diagnostics["status"], "completed")

    def test_nonfinite_input_is_rejected(self):
        e = element(1, "IfcWall", 0)
        e["_verts"][0, 0] = float("nan")
        with self.assertRaises(ValueError):
            footprint(e["_verts"], e["_faces"])
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                infer_rooms([], (0, 0, 1, 1), cell=value)


if __name__ == "__main__":
    unittest.main()
