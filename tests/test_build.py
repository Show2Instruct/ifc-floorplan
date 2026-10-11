"""Regression tests for graph identity and silently omitted inference limits."""
import copy
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from shapely.geometry import box, Point, Polygon
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fp.build import build, disjoint_declared, disjoint_rooms, reconcile_rooms
from fp.connectivity import door_connections
from fp.extract import footprint, infer_rooms, poly_geometry
from fp.projections import level_clusters, section_band
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
                self.assertEqual(plan['format_version'], 4)
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

    def test_levels_too_close_do_not_both_report_the_rooms(self):
        # model_103's GL sits 4 cm under EG. A 1.2 m cut on GL reported EG's rooms on a level
        # holding only columns. The two are one physical level, and its rooms belong to the storey
        # that owns what stands on it.
        data = fixture(room_walls(owner=200), [0., .04])
        with tempfile.TemporaryDirectory(dir=runs.ROOT) as directory:
            with patch("fp.build.extract", return_value=(data, None)):
                build("fixture", out_dir=directory, verbose=False)
            lower = json.loads((Path(directory) / "storey_00.json").read_text())
            upper = json.loads((Path(directory) / "storey_01.json").read_text())
        self.assertEqual(lower["n_rooms"], 0)
        self.assertEqual(lower["room_inference"]["status"], "level_inferred_on_another_storey")
        self.assertEqual(upper["n_rooms"], 1)

    def test_a_floor_with_a_beam_level_just_above_keeps_its_own_cut(self):
        # model_192 puts "L beam" 40 cm above L1. Measured to that record, L1's cut fell below
        # the minimum and L1 got no section: its eight doors joined nothing while the beam level
        # took the rooms. The cut is measured to the next physical level instead.
        space = prism(20, .3, .3, 2.9, 2.9, cls="IfcSpace", owner=100)
        data = fixture(room_walls(owner=100) + [space], [0., .4, 5.])
        with tempfile.TemporaryDirectory(dir=runs.ROOT) as directory:
            with patch("fp.build.extract", return_value=(data, None)):
                build("fixture", out_dir=directory, verbose=False)
            floor = json.loads((Path(directory) / "storey_00.json").read_text())
            beams = json.loads((Path(directory) / "storey_01.json").read_text())
        self.assertEqual(floor["section_projection"]["status"], "sectioned")
        self.assertAlmostEqual(floor["section_projection"]["band_m"][0], 1.2)
        self.assertGreaterEqual(floor["n_rooms"], 1)
        self.assertEqual(beams["n_rooms"], 0)

    def test_the_floor_of_a_level_is_the_storey_owning_its_walls_doors_and_windows(self):
        # model_190 puts Level 4 at -0.45 m under Level 1. Level 4 owned more elements, but 16 of
        # them were opening voids; Level 1 owns the doors and windows, and the rooms stand on it.
        walls = room_walls(owner=100)[:2] + room_walls(owner=200)[2:]
        doors = [prism(30 + k, .2, 1., .3, 2., z1=2.1, cls="IfcDoor", owner=200) for k in range(3)]
        voids = [prism(40 + k, .2, 1., .3, 2., cls="IfcOpeningElement", owner=100) for k in range(6)]
        data = fixture(walls + doors + voids, [-.45, 0., 4.])
        with tempfile.TemporaryDirectory(dir=runs.ROOT) as directory:
            with patch("fp.build.extract", return_value=(data, None)):
                build("fixture", out_dir=directory, verbose=False)
            lower = json.loads((Path(directory) / "storey_00.json").read_text())
            upper = json.loads((Path(directory) / "storey_01.json").read_text())
        self.assertEqual(lower["room_inference"]["status"], "level_inferred_on_another_storey")
        self.assertEqual(lower["room_inference"]["storey_index"], 1)
        self.assertGreaterEqual(upper["n_rooms"], 1)

    def test_furniture_below_the_cut_is_placed_in_its_room_and_walls_are_not(self):
        # A table 0.75 m high never reaches the 1.2 m cut, so it was placed in no room and a scene
        # graph hung it from the storey. Its footprint says which room it stands in.
        table = prism(50, 1., 1., 2., 2., z1=.75, cls="IfcFurniture", owner=100)
        data = fixture(room_walls(owner=100) + [table], [0., 3.])
        with tempfile.TemporaryDirectory(dir=runs.ROOT) as directory:
            with patch("fp.build.extract", return_value=(data, None)):
                build("fixture", out_dir=directory, verbose=False)
            storey = json.loads((Path(directory) / "storey_00.json").read_text())
        by_id = {e["step_id"]: e for e in storey["elements"]}
        self.assertEqual(by_id[50]["room"], storey["rooms"][0]["id"])
        self.assertEqual(by_id[50]["room_by"], "footprint")
        self.assertIsNone(by_id[1]["room"])            # a wall bounds the room, it is not in it

    def test_the_ground_level_is_the_one_at_the_datum_and_upper_floor_files_have_none(self):
        def grades(elevations):
            data = fixture(room_walls(owner=100), elevations)
            with tempfile.TemporaryDirectory(dir=runs.ROOT) as directory:
                with patch("fp.build.extract", return_value=(data, None)):
                    build("fixture", out_dir=directory, verbose=False)
                return [json.loads((Path(directory) / f"storey_{i:02d}.json").read_text())["at_grade"]
                        for i in range(len(elevations))]
        self.assertEqual(grades([-3., 0., 3.5]), [False, True, False])
        self.assertEqual(grades([1.35, 5.45]), [True, False])       # a raised ground floor
        self.assertEqual(grades([12., 16.]), [False, False])        # only the upper floors modelled

    def test_an_upper_door_leads_out_only_onto_a_surface_on_its_own_level(self):
        # model_191 Level 3: a lobby door opens onto the roof of the wing below, whose build-up
        # runs from the floor to 0.51 m above it. Onto nothing, or onto a roof a storey down, the
        # same door leads nowhere.
        def exits(beyond):
            z0, z1 = 10.2, 13.2
            walls = [prism(1, 0, 0, 3.2, .2, z0, z1, owner=200), prism(3, 0, 0, .2, 3.2, z0, z1, owner=200),
                     prism(4, 3.0, 0, 3.2, 3.2, z0, z1, owner=200),
                     prism(2, 0, 3.0, 1.2, 3.2, z0, z1, owner=200), prism(5, 2.1, 3.0, 3.2, 3.2, z0, z1, owner=200)]
            leaf = prism(6, 1.2, 3.05, 2.1, 3.15, z0, z0 + 2.1, cls="IfcDoor", owner=200)
            data = fixture(walls + [leaf] + beyond, [0., z0])
            with tempfile.TemporaryDirectory(dir=runs.ROOT) as directory:
                with patch("fp.build.extract", return_value=(data, None)):
                    build("fixture", out_dir=directory, verbose=False)
                storey = json.loads((Path(directory) / "storey_01.json").read_text())
            self.assertFalse(storey["at_grade"])
            kinds = [c["type"] for c in storey["room_connections"]]
            return kinds.count("exterior_door"), storey["connection_counts"].get("door_onto_a_void", 0)

        self.assertEqual(exits([]), (0, 1))
        low_roof = prism(7, -1., 3.2, 4.2, 7., 10.2, 10.71, cls="IfcSlab", owner=200)
        self.assertEqual(exits([low_roof]), (1, 0))
        roof_a_storey_down = prism(7, -1., 3.2, 4.2, 7., 6.9, 7.2, cls="IfcSlab", owner=200)
        self.assertEqual(exits([roof_a_storey_down]), (0, 1))
        # model_192: no slab is modelled beyond the door, but tables stand there at floor height
        table = prism(8, 0., 3.25, 4., 5.5, 10.2, 10.95, cls="IfcFurniture", owner=200)
        self.assertEqual(exits([table]), (1, 0))
        # a table a storey down stands on another floor
        table_below = prism(8, 0., 3.25, 4., 5.5, 6.9, 7.65, cls="IfcFurniture", owner=200)
        self.assertEqual(exits([table_below]), (0, 1))

    def test_floors_the_file_never_declares_get_storeys_of_their_own(self):
        # model_172 declares one storey, Ebene 3 at 10.5 m, and puts on it the walls and spaces of
        # floors at 0, 3.5 and 7 m: the cut at 10.5 + 1.2 m saw one floor and every space was stacked.
        def ring(start, z, owner=100):
            walls = []
            for k in range(3):
                walls += [prism(start + 4 * k, 2 * k, 0, 2 * k + 2, .2, z, z + 3, owner=owner),
                          prism(start + 4 * k + 1, 2 * k, 5.8, 2 * k + 2, 6, z, z + 3, owner=owner),
                          prism(start + 4 * k + 2, 0, 2 * k, .2, 2 * k + 2, z, z + 3, owner=owner),
                          prism(start + 4 * k + 3, 5.8, 2 * k, 6, 2 * k + 2, z, z + 3, owner=owner)]
            return walls
        elements = []
        for n, z in enumerate((0., 3.5, 7., 10.5)):
            elements += ring(1000 + 100 * n, z)
            elements.append(prism(2000 + n, .3, .3, 5.7, 5.7, z, z + 2.8, cls="IfcSpace"))
        data = fixture(elements, [10.5])
        with tempfile.TemporaryDirectory(dir=runs.ROOT) as directory:
            with patch("fp.build.extract", return_value=(data, None)):
                building = build("fixture", out_dir=directory, verbose=False)
            levels = [json.loads(p.read_text()) for p in sorted(Path(directory).glob("storey_*.json"))]
        self.assertEqual([round(s["elevation_m"], 2) for s in levels], [0., 3.5, 7., 10.5])
        self.assertEqual([s["n_rooms"] for s in levels], [1, 1, 1, 1])
        self.assertEqual([lvl["elevation_m"] for lvl in building["undeclared_levels"]], [0., 3.5, 7.])
        self.assertEqual([lvl["moved_elements"] for lvl in building["undeclared_levels"]], [13, 13, 13])
        moved = [e for e in levels[0]["elements"] if e["storey_link"].startswith("moved")]
        self.assertEqual(len(moved), 12)          # the walls; the space is no element record
        # model_74 declares no storeys at all, and the fallback put a level between its floors
        elements = []
        for n, z in enumerate((0., 3.5)):
            elements += ring(1000 + 100 * n, z, owner=None)
            elements.append(prism(2000 + n, .3, .3, 5.7, 5.7, z, z + 2.8, cls="IfcSpace", owner=None))
        data = fixture(elements, [1.75])
        data["storey_source"] = "inferred_from_geometry"
        data["storeys"][0].update(step_id=None, source="inferred_from_geometry")
        with tempfile.TemporaryDirectory(dir=runs.ROOT) as directory:
            with patch("fp.build.extract", return_value=(data, None)):
                building = build("fixture", out_dir=directory, verbose=False)
            levels = [json.loads(p.read_text()) for p in sorted(Path(directory).glob("storey_*.json"))]
        self.assertEqual(building["storey_source"], "wall_bases")
        self.assertEqual([(s["elevation_m"], s["n_rooms"]) for s in levels], [(0., 1), (3.5, 1)])
        # model_182: three floors' spaces on no storey and almost no IfcWall to show the floors
        elements = []
        for n, z in enumerate((0., 3.6, 7.2)):
            elements += [prism(3000 + 10 * n + k, 4 * k, 0, 4 * k + 3.8, 3.8, z, z + 3., cls="IfcSpace", owner=None)
                         for k in range(2)]
        elements += [prism(3100 + k, 4 * k - .2, -.2, 4 * k + 4, 0., 10.8, 13.8, owner=100) for k in range(2)]
        data = fixture(elements, [10.8])
        with tempfile.TemporaryDirectory(dir=runs.ROOT) as directory:
            with patch("fp.build.extract", return_value=(data, None)):
                building = build("fixture", out_dir=directory, verbose=False)
            levels = [json.loads(p.read_text()) for p in sorted(Path(directory).glob("storey_*.json"))]
        self.assertEqual([lvl["elevation_m"] for lvl in building["undeclared_levels"]], [0., 3.6, 7.2])
        self.assertEqual([(s["elevation_m"], s["n_rooms"]) for s in levels][:3], [(0., 2), (3.6, 2), (7.2, 2)])
        # model_6 states a hall standing on its 4th storey on its 3rd, over the 3rd storey's hall
        halls = [prism(4001, 0, 0, 8, 6, 13.3, 17.4, cls="IfcSpace", owner=100),
                 prism(4002, 0, 0, 8, 6, 17.4, 21.5, cls="IfcSpace", owner=100)]
        data = fixture(halls, [13.2, 17.6])
        with tempfile.TemporaryDirectory(dir=runs.ROOT) as directory:
            with patch("fp.build.extract", return_value=(data, None)):
                build("fixture", out_dir=directory, verbose=False)
            levels = [json.loads(p.read_text()) for p in sorted(Path(directory).glob("storey_*.json"))]
        self.assertEqual([[r["id"] for r in s["rooms"]] for s in levels], [["space_4001"], ["space_4002"]])
        self.assertEqual([(r["storey_assignment"], r.get("storey_stated")) for s in levels for r in s["rooms"]],
                         [("IfcRelContainedInSpatialStructure", None), ("moved to the storey its base stands on", 100)])
        # model_172's facades: a curtain wall per floor with no shape of its own, only its parts, all
        # named on the one declared storey
        elements = []
        for n, z in enumerate((0., 3.5)):
            elements += ring(1000 + 100 * n, z)
        facade = {"step_id": 5000, "ifc_class": "IfcCurtainWall", "label": "facade", "has_geometry": False,
                  "storey_step_id": 100, "storey_link": "IfcRelContainedInSpatialStructure"}
        mullions = [prism(5001 + k, 2 * k, -.3, 2 * k + .1, -.2, 0., 3., cls="IfcMember") for k in range(3)]
        data = fixture(elements + [facade] + mullions, [3.5])
        data["relations"] = [{"type": "part_of", "source": m["step_id"], "target": 5000} for m in mullions]
        with tempfile.TemporaryDirectory(dir=runs.ROOT) as directory:
            with patch("fp.build.extract", return_value=(data, None)):
                build("fixture", out_dir=directory, verbose=False)
            levels = [json.loads(p.read_text()) for p in sorted(Path(directory).glob("storey_*.json"))]
        placed = {s["elevation_m"]: [e["step_id"] for e in s["elements"] if e["ifc_class"] == "IfcCurtainWall"] for s in levels}
        self.assertEqual(placed, {0.: [5000], 3.5: []})
        # model_80: one space on a floor with no storey and too few walls to show it is a floor of its own
        lone = [prism(4101, 0, 0, 6, 6, 0., 2.4, cls="IfcSpace", owner=None),
                prism(4102, 0, 0, 6, 6, 4.0, 6.4, cls="IfcSpace", owner=None)]
        data = fixture(lone, [0.])
        with tempfile.TemporaryDirectory(dir=runs.ROOT) as directory:
            with patch("fp.build.extract", return_value=(data, None)):
                building = build("fixture", out_dir=directory, verbose=False)
            levels = [json.loads(p.read_text()) for p in sorted(Path(directory).glob("storey_*.json"))]
        self.assertEqual([lvl["elevation_m"] for lvl in building["undeclared_levels"]], [4.])
        self.assertEqual([(s["elevation_m"], [r["id"] for r in s["rooms"]]) for s in levels],
                         [(0., ["space_4101"]), (4., ["space_4102"])])
        self.assertEqual([(r["storey_assignment"], "storey_stated" in r, r.get("storey_stated")) for s in levels for r in s["rooms"]],
                         [("inferred_from_elevation", False, None),
                          ("moved to an undeclared level by the height of its base", True, None)])
        # a storey's own walls reaching up through the floors above stay where the file puts them
        data = fixture(room_walls(owner=100) * 1, [0., 3.])
        for w in data["elements"]:
            w["bbox_max"][2] = 9.
        with tempfile.TemporaryDirectory(dir=runs.ROOT) as directory:
            with patch("fp.build.extract", return_value=(data, None)):
                building = build("fixture", out_dir=directory, verbose=False)
        self.assertEqual(building["undeclared_levels"], [])

    def test_physical_levels_group_records_nearer_than_the_minimum_cut(self):
        storeys = [{"elevation_m": z} for z in (0., .4, 5., 5.04, 10.)]
        self.assertEqual(level_clusters(storeys), [[0, 1], [2, 3], [4]])
        low, high, height, note = section_band(storeys, 0, 1.2, .01)
        self.assertAlmostEqual(low, 1.2)

    def test_one_level_listed_twice_is_cut_once_and_inferred_once(self):
        # model_194 lists Ground Floor and OK OG5 both at 0.00 m. Measuring the cut to the next
        # record left the floor that holds the spaces with no section at all, so its doors joined
        # nothing, while its twin carried the rooms. 83 rooms and 91 doors sat on such storeys.
        space = prism(20, .3, .3, 2.9, 2.9, cls="IfcSpace", owner=100)
        leaf = prism(30, 1.2, 0., 2.0, .2, z1=2.1, cls="IfcDoor", owner=200)
        data = fixture(room_walls(owner=100) + [space, leaf], [0., 0., 3.])
        with tempfile.TemporaryDirectory(dir=runs.ROOT) as directory:
            with patch("fp.build.extract", return_value=(data, None)):
                build("fixture", out_dir=directory, verbose=False)
            first = json.loads((Path(directory) / "storey_00.json").read_text())
            twin = json.loads((Path(directory) / "storey_01.json").read_text())
        # the cut is measured to the next level, 3 m up, not to the twin at the same height
        self.assertEqual(first["section_projection"]["status"], "sectioned")
        self.assertAlmostEqual(first["section_projection"]["band_m"][0], 1.2)
        self.assertGreaterEqual(first["n_rooms"], 1)
        # the twin sees the same geometry, so it must not repeat the level's rooms
        self.assertEqual(twin["room_inference"]["status"], "level_inferred_on_another_storey")
        self.assertEqual(twin["room_inference"]["storey_index"], 0)
        self.assertEqual(twin["n_rooms"], 0)
        # nor count the level's doors as leading nowhere: the first record connects them
        self.assertEqual(twin["connection_counts"], {"doors_on_a_storey_without_rooms": 1})
        self.assertNotIn("door_between_no_rooms", first["connection_counts"])

    def test_a_storey_without_a_section_asserts_no_open_passages(self):
        # Declared spaces survive without a section, but nothing is known about what stands
        # between them. Across the corpus this claimed 50 ways through on evidence of nothing.
        spaces = [prism(10, 0, 0, 3, 3, cls="IfcSpace", owner=100),
                  prism(11, 3, 0, 6, 3, cls="IfcSpace", owner=100)]
        data = fixture(spaces, [0., 3.])
        with tempfile.TemporaryDirectory(dir=runs.ROOT) as directory:
            with patch("fp.build.extract", return_value=(data, None)):
                # a band base above the cut leaves no height to section
                build("fixture", out_dir=directory, verbose=False, band_base=1.5)
            lower = json.loads((Path(directory) / "storey_00.json").read_text())
        self.assertEqual(lower["section_projection"]["status"], "no_section_band")
        self.assertEqual(lower["n_rooms"], 2)
        self.assertEqual([l for l in lower["room_connections"] if l["type"] == "open_passage"], [])
        self.assertEqual(lower["connection_counts"]["open_passages_without_a_section"], 2)

    def test_a_declared_room_keeps_the_global_id_other_tools_know_it_by(self):
        # other tools reading the same file name a room by its IfcSpace GlobalId
        space = dict(prism(20, .3, .3, 2.9, 2.9, cls="IfcSpace", owner=100), global_id="3xSpace")
        data = fixture(room_walls(owner=100) + [space], [0., 3.])
        with tempfile.TemporaryDirectory(dir=runs.ROOT) as directory:
            with patch("fp.build.extract", return_value=(data, None)):
                build("fixture", out_dir=directory, verbose=False)
            storey = json.loads((Path(directory) / "storey_00.json").read_text())
        self.assertEqual([(r["id"], r["global_id"]) for r in storey["rooms"]], [("space_20", "3xSpace")])

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

    def test_a_wall_of_the_floor_below_that_fails_the_cut_still_encloses_this_floor(self):
        # model_17, model_47 and model_83: walls owned by the ground floor stand through the
        # storey above, and their meshes do not close at its cut. Excluded, they left its rooms
        # open; their plan outline is the same at every height they reach.
        walls = [dict(w, bbox_max=[*w["bbox_max"][:2], 6.], _verts=w["_verts"] * [1, 1, 2]) for w in room_walls(owner=100)]
        data = fixture(walls, [0., 3.])
        with tempfile.TemporaryDirectory(dir=runs.ROOT) as directory:
            with patch("fp.build.extract", return_value=(data, None)), \
                 patch("fp.projections.project_band",
                       return_value=(None, {"status": "incomplete_cut_contours"})):
                build("fixture", out_dir=directory, verbose=False)
            upper = json.loads((Path(directory) / "storey_01.json").read_text())
        section = upper["section_projection"]
        self.assertEqual(section["excluded"], [])
        self.assertEqual([f["owner_storey_index"] for f in section["fallbacks"]], [0, 0, 0, 0])
        self.assertTrue(section["barriers_complete"])
        self.assertEqual(upper["n_rooms"], 1)

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

    def test_an_opening_left_unfilled_divides_rooms_and_an_archway_joins_them(self):
        # 1,202 GNI openings reach the floor with nothing in them and 915 more are hatches; at the
        # cut each was a hole in its wall, and the rooms either side flooded into one.
        def merged(step, boxes, owner=100):
            pieces = [prism(step, *b, owner=owner) for b in boxes]
            verts, faces, offset = [], [], 0
            for piece in pieces:
                verts.extend(piece["_verts"].tolist())
                faces.extend([(a + offset, b + offset, c + offset) for a, b, c in piece["_faces"]])
                offset += len(piece["_verts"])
            lows = np.min([piece["bbox_min"] for piece in pieces], axis=0).tolist()
            highs = np.max([piece["bbox_max"] for piece in pieces], axis=0).tolist()
            return dict(pieces[0], _verts=np.array(verts), _faces=faces, bbox_min=lows, bbox_max=highs)

        outer = [prism(1, 0, 0, 6, .2), prism(2, 0, 2.8, 6, 3), prism(3, 0, 0, .2, 3), prism(4, 5.8, 0, 6, 3)]
        partition = merged(5, [(2.95, .2, 3.15, 1.0, 0., 3.), (2.95, 2.0, 3.15, 2.8, 0., 3.),
                               (2.95, 1.0, 3.15, 2.0, 2.1, 3.)])

        def plan(opening, relations):
            data = fixture(outer + [copy.deepcopy(partition)] + opening, [0.])
            data["relations"] = relations
            with tempfile.TemporaryDirectory(dir=runs.ROOT) as directory:
                with patch("fp.build.extract", return_value=(data, None)):
                    build("fixture", out_dir=directory, verbose=False)
                return json.loads((Path(directory) / "storey_00.json").read_text())

        archway = [prism(50, 2.9, 1.0, 3.2, 2.0, 0., 2.1, cls="IfcOpeningElement")]
        voids = [{"type": "voids", "source": 5, "target": 50}]
        self.assertEqual(plan(archway, [])["n_rooms"], 1)           # without the relation, one room
        storey = plan(archway, voids)
        self.assertEqual(storey["n_rooms"], 2)
        ways = [c for c in storey["room_connections"] if c["type"] == "open_passage"]
        self.assertEqual([(c.get("opening"), round(c["width"], 1)) for c in ways], [(50, 1.0)])
        self.assertEqual([(h["walkable"], h["filled_by"]) for h in storey["opening_holes"]], [(True, None)])
        # an archway at the partition's end clips the corner of the outer wall the file names
        end_partition = merged(5, [(2.95, 1.1, 3.15, 2.8, 0., 3.), (2.95, .2, 3.15, 1.1, 2.1, 3.)])
        data = fixture(outer + [end_partition, prism(50, 2.9, .1, 3.2, 1.1, 0., 2.1, cls="IfcOpeningElement")], [0.])
        data["relations"] = [{"type": "voids", "source": 1, "target": 50}]
        with tempfile.TemporaryDirectory(dir=runs.ROOT) as directory:
            with patch("fp.build.extract", return_value=(data, None)):
                build("fixture", out_dir=directory, verbose=False)
            corner = json.loads((Path(directory) / "storey_00.json").read_text())
        self.assertEqual(corner["n_rooms"], 2)
        self.assertEqual([(h["host"], h.get("host_from")) for h in corner["opening_holes"]], [(5, "geometry")])
        self.assertAlmostEqual(corner["opening_holes"][0]["width_m"], .9, places=2)
        # credited to the outer wall the partition joins, the archway is still cut into the partition
        storey = plan(archway, [{"type": "voids", "source": 1, "target": 50}])
        self.assertEqual(storey["n_rooms"], 2)
        self.assertEqual([(h["host"], h.get("host_from"), h.get("host_stated")) for h in storey["opening_holes"]],
                         [(5, "geometry", 1)])
        self.assertEqual(len([c for c in storey["room_connections"] if c["type"] == "open_passage"]), 1)
        # ResBIM 10 lists one archway three times, once per wall meeting there: one way through
        triple = archway + [prism(51, 2.9, 1.0, 3.2, 2.0, 0., 2.1, cls="IfcOpeningElement"),
                            prism(52, 2.9, 1.0, 3.2, 2.0, 0., 2.1, cls="IfcOpeningElement")]
        storey = plan(triple, voids + [{"type": "voids", "source": 5, "target": 51}, {"type": "voids", "source": 5, "target": 52}])
        self.assertEqual(len([c for c in storey["room_connections"] if c["type"] == "open_passage"]), 1)
        self.assertEqual(Counter(h.get("same_as") for h in storey["opening_holes"]), Counter({None: 1, 50: 2}))
        # and the void of a joining wall at a door's opening is that door's hole, not an archway
        leaf_shut = prism(60, 2.95, 1.0, 3.0, 2.0, 0., 2.1, cls="IfcDoor")
        door_void = prism(53, 2.9, 1.0, 3.2, 2.0, 0., 2.1, cls="IfcOpeningElement")
        storey = plan([leaf_shut, door_void] + archway, voids + [{"type": "voids", "source": 5, "target": 53},
                      {"type": "hosts", "source": 5, "target": 60, "via_opening": 53}])
        self.assertEqual(sorted(c["type"] for c in storey["room_connections"]), ["connected_by_door"])
        # one space declared across the archway is one room: the archway does not divide it
        storey = plan(archway + [prism(70, .2, .2, 5.8, 2.8, 0., 3., cls="IfcSpace")], voids)
        self.assertEqual([(r["id"], r["source"]) for r in storey["rooms"]], [("space_70", "IfcSpace")])
        # a hatch closes the wall but is no way through
        hatch = [prism(50, 2.9, 1.0, 3.2, 2.0, .9, 1.8, cls="IfcOpeningElement")]
        storey = plan(hatch, voids)
        self.assertEqual(storey["n_rooms"], 2)
        self.assertEqual([c for c in storey["room_connections"] if c["type"] == "open_passage"], [])
        # a door whose leaf is drawn open: its doorway still divides the rooms, and the door joins them
        leaf = prism(60, 3.15, 1.0, 3.95, 1.05, 0., 2.1, cls="IfcDoor")
        storey = plan(archway + [leaf], voids + [{"type": "hosts", "source": 5, "target": 60, "via_opening": 50}])
        self.assertEqual(storey["n_rooms"], 2)
        self.assertEqual(sorted(c["type"] for c in storey["room_connections"]), ["connected_by_door"])

    def test_an_opening_closes_its_wall_as_thick_as_the_cut_finds_it(self):
        # model_203: a wall thicker at its base than at the cut gave its window a hole 0.18 m into
        # the room, and the room lost that strip.
        def merged(step, boxes, owner=100):
            pieces = [prism(step, *b, owner=owner) for b in boxes]
            verts, faces, offset = [], [], 0
            for piece in pieces:
                verts.extend(piece["_verts"].tolist())
                faces.extend([(a + offset, b + offset, c + offset) for a, b, c in piece["_faces"]])
                offset += len(piece["_verts"])
            return dict(pieces[0], _verts=np.array(verts), _faces=faces,
                        bbox_min=np.min([p["bbox_min"] for p in pieces], axis=0).tolist(),
                        bbox_max=np.max([p["bbox_max"] for p in pieces], axis=0).tolist())
        walls = [prism(1, 0, 0, 6, .2), merged(2, [(0, 2.8, 6, 3, 0., 3.), (0, 2.6, 6, 3, 0., .3)]),
                 prism(3, 0, 0, .2, 3), prism(4, 5.8, 0, 6, 3)]
        window = prism(50, 2, 2.5, 3, 3.1, .85, 2.2, cls="IfcOpeningElement")

        def area(relations):
            data = fixture(copy.deepcopy(walls) + [copy.deepcopy(window)], [0.])
            data["relations"] = relations
            with tempfile.TemporaryDirectory(dir=runs.ROOT) as directory:
                with patch("fp.build.extract", return_value=(data, None)):
                    build("fixture", out_dir=directory, verbose=False)
                return json.loads((Path(directory) / "storey_00.json").read_text())["rooms"][0]["area_m2"]
        self.assertAlmostEqual(area([{"type": "voids", "source": 2, "target": 50}]), area([]), places=2)

    def test_a_door_through_a_thick_outer_wall_still_leads_outside(self):
        # the door's own hole reaches past where the search for open ground starts
        pieces = [prism(5, *b) for b in ((0, -.8, 2.5, 0, 0., 3.), (3.5, -.8, 6, 0, 0., 3.), (2.5, -.8, 3.5, 0, 2.1, 3.))]
        verts, faces = [], []
        for k, piece in enumerate(pieces):
            verts.extend(piece["_verts"].tolist())
            faces.extend([(a + 8 * k, b + 8 * k, c + 8 * k) for a, b, c in piece["_faces"]])
        outer = dict(pieces[0], _verts=np.array(verts), _faces=faces, bbox_min=[0, -.8, 0.], bbox_max=[6, 0, 3.])
        walls = [outer, prism(2, 0, 3, 6, 3.2), prism(3, -.2, -.8, 0, 3.2), prism(4, 6, -.8, 6.2, 3.2)]
        leaf = prism(60, 2.5, -.42, 3.5, -.38, 0., 2.1, cls="IfcDoor")
        void = prism(50, 2.5, -.8, 3.5, 0., 0., 2.1, cls="IfcOpeningElement")
        relations = [{"type": "voids", "source": 5, "target": 50},
                     {"type": "hosts", "source": 5, "target": 60, "via_opening": 50}]
        data = fixture(walls + [leaf, void], [0.])
        data["relations"] = relations
        with tempfile.TemporaryDirectory(dir=runs.ROOT) as directory:
            with patch("fp.build.extract", return_value=(data, None)):
                build("fixture", out_dir=directory, verbose=False)
            storey = json.loads((Path(directory) / "storey_00.json").read_text())
        self.assertEqual([c["type"] for c in storey["room_connections"]], ["exterior_door"])

    def test_whatever_fills_an_opening_closes_it_like_a_door(self):
        # 64 GNI openings are filled by proxies, lift doors among them; not being barriers, they
        # left a hole in the wall at the cut and the room drained through it.
        walls = [prism(1, 0, 0, 3.2, .2), prism(2, 0, 3.0, 3.2, 3.2),
                 prism(3, 0, 0, .2, 3.2), prism(4, 3.0, 0, 3.2, 1.4), prism(5, 3.0, 1.8, 3.2, 3.2)]
        proxy = prism(6, 3.0, 1.4, 3.2, 1.8, z1=2.1, cls="IfcBuildingElementProxy")
        for relations, expected in (([], 0), ([{"type": "hosts", "source": 4, "target": 6, "via_opening": 7}], 1)):
            data = fixture([copy.deepcopy(e) for e in walls + [proxy]], [0.])
            data["relations"] = relations
            with tempfile.TemporaryDirectory(dir=runs.ROOT) as directory:
                with patch("fp.build.extract", return_value=(data, None)):
                    build("fixture", out_dir=directory, verbose=False)
                plan = json.loads((Path(directory) / "storey_00.json").read_text())
            self.assertEqual(plan["n_rooms"], expected)

    def test_declared_spaces_that_overlap_are_made_disjoint(self):
        def space(step, x0, y0, x1, y1, stated=True, name=None):
            return {"id": f"space_{step}", "step_id": step, "name": name or f"S{step}", "source": "IfcSpace",
                    "_poly": box(x0, y0, x1, y1), "_stated": stated}
        # model_46 lists each hall twice; the repeat goes
        spaces, aside = disjoint_declared([space(1, 0, 0, 10, 10), space(2, 0, 0, 10, 10.1)])
        self.assertEqual([s["id"] for s in spaces], ["space_1"])
        self.assertEqual(aside[0]["reason"], "repeats")
        # model_118 holds the spaces of floors it does not contain, on no storey, over its own
        spaces, aside = disjoint_declared([space(1, 0, 0, 10, 10), space(2, 0, 0, 6, 10, stated=False)])
        self.assertEqual([s["id"] for s in spaces], ["space_1"])
        self.assertIn("placed on no storey", aside[0]["reason"])
        # model_203 declares "Flaeche" over the hallway it also declares; the hallway keeps its floor
        spaces, aside = disjoint_declared([space(1, 0, 0, 20, 10, name="Flaeche"), space(2, 2, 2, 5, 9, name="Hallway")])
        whole = {s["name"]: s for s in spaces}
        self.assertAlmostEqual(whole["Hallway"]["_poly"].area, 21.)
        self.assertAlmostEqual(whole["Flaeche"]["_poly"].area, 200. - 21.)
        self.assertEqual(whole["Flaeche"]["carved_by"], ["space_2"])
        self.assertFalse(whole["Flaeche"]["_poly"].intersects(box(2.1, 2.1, 4.9, 8.9)))
        # a catch-all the specific spaces cover completely has nothing left
        spaces, aside = disjoint_declared([space(1, 0, 0, 4, 4), space(2, 0, 0, 2, 4), space(3, 2, 0, 4, 4.0)])
        self.assertEqual(sorted(s["id"] for s in spaces), ["space_2", "space_3"])
        self.assertEqual(aside[0]["reason"], "covered by the spaces it overlaps")
        # model_139: an office the walls enclose, and a smaller zone declared inside it; the walls
        # say the office is the room, so the zone is set aside and the office stays whole
        office, zone = space(1, 0, 0, 3, 2.6, name="Office"), space(2, .2, .2, 2.2, 2.4, name="Office zone")
        spaces, aside = disjoint_declared([office, zone], regions=[box(.02, .02, 2.98, 2.58)])
        self.assertEqual([s["id"] for s in spaces], ["space_1"])
        self.assertAlmostEqual(spaces[0]["_poly"].area, 7.8)
        self.assertEqual((aside[0]["reason"], aside[0]["in"]), ("a zone inside the room it lies in", "space_1"))
        # model_203 again, with walls: they enclose the hallway, not "Flaeche", so the hallway keeps its floor
        spaces, aside = disjoint_declared([space(1, 0, 0, 20, 10, name="Flaeche"), space(2, 2, 2, 5, 9, name="Hallway")],
                                          regions=[box(2.05, 2.05, 4.95, 8.95), box(5.2, 0, 20, 10)])
        self.assertEqual(sorted(s["name"] for s in spaces), ["Flaeche", "Hallway"])
        self.assertEqual(next(s for s in spaces if s["name"] == "Flaeche")["carved_by"], ["space_2"])
        # model_182: the same outline on two floors is two rooms, not one listed twice
        lower, upper = space(1, 0, 0, 10, 10), space(2, 0, 0, 10, 10)
        lower["_z"], upper["_z"] = (0., 3.), (3.6, 6.6)
        spaces, aside = disjoint_declared([lower, upper])
        self.assertEqual((len(spaces), aside), (2, []))
        # spaces drawn a few centimetres over a shared wall line keep their outlines
        spaces, aside = disjoint_declared([space(1, 0, 0, 4.05, 4), space(2, 3.95, 0, 8, 4)])
        self.assertEqual(len(spaces), 2)
        self.assertNotIn("carved_by", spaces[0])

    def test_rooms_that_still_overlap_after_reconciliation_are_made_disjoint(self):
        def room(rid, poly, source="inferred_from_wall_geometry"):
            return {"id": rid, "name": rid, "source": source, "_poly": poly}
        # model_74: a part of a divided corridor reached 1.6 m2 into the office beside it
        rooms, dropped = disjoint_rooms([room("room_0_0", box(0, 0, 10, 2.2)),
                                         room("space_9", box(4, 1.8, 7, 6), source="IfcSpace")])
        whole = {r["id"]: r for r in rooms}
        self.assertAlmostEqual(whole["space_9"]["_poly"].area, 12.6)
        self.assertAlmostEqual(whole["room_0_0"]["_poly"].area, 22. - 1.2)
        self.assertEqual(whole["room_0_0"]["trimmed_by_rooms"], ["space_9"])
        # model_197: two inferred rooms both grown into the seam between them; the smaller keeps it
        rooms, _ = disjoint_rooms([room("room_4_0", box(0, 0, 10, 10)), room("room_4_1", box(9.4, 0, 14, 10))])
        whole = {r["id"]: r for r in rooms}
        self.assertAlmostEqual(whole["room_4_0"]["_poly"].area, 94.)
        self.assertNotIn("trimmed_by_rooms", whole["room_4_1"])
        # a sliver along a shared wall line is left alone
        rooms, _ = disjoint_rooms([room("a", box(0, 0, 4.05, 4)), room("b", box(3.95, 0, 8, 4))])
        self.assertFalse(any(r.get("trimmed_by_rooms") for r in rooms))

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
        declared = [{"id": "space_1", "step_id": 1, "global_id": "G1", "name": "Office", "source": "IfcSpace",
                     "_poly": box(0, 0, 4, 4)}]
        inferred = [{"id": "room_0_0", "name": "Room 1", "source": "inferred_from_wall_geometry",
                     "area_m2": 7.6, "_poly": box(.1, .1, 1.9, 3.9)},
                    {"id": "room_0_1", "name": "Room 2", "source": "inferred_from_wall_geometry",
                     "area_m2": 7.6, "_poly": box(2.1, .1, 3.9, 3.9)}]
        rooms, notes = reconcile_rooms(declared, inferred)
        self.assertEqual(len(rooms), 2)
        self.assertNotIn("space_1", [r["id"] for r in rooms])
        self.assertTrue(all(r["subdivides_declared_space"] == 1 for r in rooms))
        self.assertTrue(all(r["declared_space_global_id"] == "G1" for r in rooms))
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

    def test_a_region_spanning_declared_spaces_is_cut_back_to_what_they_do_not_cover(self):
        # A barrier missing at the cut lets free space flood across two spaces the file states.
        # Keeping the flood whole would put two rooms on the same floor and a door leaving its
        # wall would land in whichever came first.
        declared = [{"id": "space_1", "step_id": 1, "name": "Office", "source": "IfcSpace",
                     "_poly": box(0, 0, 4, 4)},
                    {"id": "space_2", "step_id": 2, "name": "Store", "source": "IfcSpace",
                     "_poly": box(6, 0, 10, 4)}]
        inferred = [{"id": "room_0_0", "name": "Room 1", "source": "inferred_from_wall_geometry",
                     "_poly": box(0, 0, 10, 4)}]
        rooms, notes = reconcile_rooms(declared, inferred)
        self.assertEqual(sorted(r["id"] for r in rooms), ["room_0_0", "space_1", "space_2"])
        corridor = next(r for r in rooms if r["id"] == "room_0_0")
        self.assertAlmostEqual(corridor["_poly"].area, 8.)
        self.assertTrue(corridor["trimmed_by_declared_spaces"])
        self.assertEqual(notes["trimmed_by_declared_spaces"], 1)
        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                self.assertLess(rooms[i]["_poly"].intersection(rooms[j]["_poly"]).area, .01)

    def test_a_region_the_declared_spaces_already_cover_is_dropped(self):
        declared = [{"id": "space_1", "step_id": 1, "name": "A", "source": "IfcSpace",
                     "_poly": box(0, 0, 4, 4)},
                    {"id": "space_2", "step_id": 2, "name": "B", "source": "IfcSpace",
                     "_poly": box(4, 0, 8, 4)}]
        inferred = [{"id": "room_0_0", "name": "Room 1", "source": "inferred_from_wall_geometry",
                     "_poly": box(0, 0, 8, 4)}]
        rooms, notes = reconcile_rooms(declared, inferred)
        self.assertEqual(sorted(r["id"] for r in rooms), ["space_1", "space_2"])
        self.assertEqual(notes["dropped_covered_by_declared_spaces"], 1)

    def test_a_remainder_in_two_separate_pieces_becomes_two_rooms(self):
        declared = [{"id": "space_1", "step_id": 1, "name": "A", "source": "IfcSpace",
                     "_poly": box(0, 2, 4, 6)}]
        inferred = [{"id": "room_0_0", "name": "Room 1", "source": "inferred_from_wall_geometry",
                     "_poly": box(0, 0, 4, 8)}]
        rooms, notes = reconcile_rooms(declared, inferred)
        self.assertEqual(sorted(r["id"] for r in rooms), ["room_0_0_0", "room_0_0_1", "space_1"])
        self.assertEqual(notes["split_by_declared_spaces"], 1)
        for room in rooms:
            if room["id"].startswith("room_"):
                self.assertAlmostEqual(room["_poly"].area, 8.)

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

    def test_a_room_does_not_grow_through_the_joint_round_a_door_leaf(self):
        # model_3: a 38 mm leaf flush with one face of a 125 mm partition, 3 mm short of its
        # frame at each end. The west room grew through those gaps into the recess behind the
        # leaf, so both rays from the door met it and the door was filed as inside one room.
        outer = [box(0, 0, 6, .2), box(0, 2.8, 6, 3), box(0, 0, .2, 3), box(5.8, 0, 6, 3)]
        partition = [box(2.95, .2, 3.075, 1.), box(2.95, 1.9, 3.075, 2.8)]
        frame = [box(2.93, .98, 3.095, 1.02), box(2.93, 1.88, 3.095, 1.92)]
        leaf = box(2.95, 1.023, 2.988, 1.877)
        barriers = outer + partition + frame + [leaf]
        rooms = infer_rooms(barriers, (0, 0, 6, 3))
        self.assertEqual(len(rooms), 2)
        west = next(r["polygon"] for r in rooms if r["polygon"].contains(Point(1.5, 1.5)))
        east = next(r["polygon"] for r in rooms if r["polygon"].contains(Point(4.5, 1.5)))
        recess = Point(3.03, 1.45)
        self.assertFalse(west.contains(recess))
        self.assertTrue(east.contains(recess))
        self.assertAlmostEqual(west.bounds[2], 2.95, places=2)       # still on the wall face
        section = unary_union(frame + [leaf])
        edges, _ = door_connections([{"id": "west", "_poly": west}, {"id": "east", "_poly": east}],
                                    [{"global_id": "d", "_poly": section}],
                                    blocked=unary_union(barriers).buffer(.06))
        self.assertEqual([(e["type"], {e["a"], e["b"]}) for e in edges],
                         [("connected_by_door", {"west", "east"})])

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
        # railed all round, a terrace at ground level is a dead end like a balcony (ResBIM's 723
        # balconies sit at elevation 0)
        out = plan["outdoor_spaces"][0]["id"]
        self.assertEqual([c for c in plan["room_connections"] if c["a"] == out], [])
        self.assertEqual(plan["connection_counts"].get("terrace_railed_all_round"), 1)
        # with a gap in its railing, steps or a gate, it leads onto the ground (model_171's veranda)
        gap = [e for e in elements if e["step_id"] != 6] + [prism(9, -.2, 5.1, 1.0, 5.2, z0=0., z1=1.1, cls="IfcRailing"),
                                                            prism(10, 2.2, 5.1, 3.4, 5.2, z0=0., z1=1.1, cls="IfcRailing")]
        data = fixture(gap, [0.])
        with tempfile.TemporaryDirectory(dir=runs.ROOT) as directory:
            with patch("fp.build.extract", return_value=(data, None)):
                build("fixture", out_dir=directory, verbose=False)
            gated = json.loads((Path(directory) / "storey_00.json").read_text())
        out = gated["outdoor_spaces"][0]["id"]
        ways = [c for c in gated["room_connections"] if c["a"] == out]
        self.assertEqual([(c["type"], c.get("exterior")) for c in ways], [("open_passage", True)])
        self.assertAlmostEqual(ways[0]["width"], 1.2, delta=.1)        # the gate at its width
        # one floor up the same balcony is a dead end
        raised = [dict(e, bbox_min=[*e["bbox_min"][:2], e["bbox_min"][2] + 4.], bbox_max=[*e["bbox_max"][:2], e["bbox_max"][2] + 4.],
                       _verts=e["_verts"] + [0, 0, 4.], storey_step_id=200) for e in elements]
        data = fixture(raised, [0., 4.])
        with tempfile.TemporaryDirectory(dir=runs.ROOT) as directory:
            with patch("fp.build.extract", return_value=(data, None)):
                build("fixture", out_dir=directory, verbose=False)
            upper = json.loads((Path(directory) / "storey_01.json").read_text())
        self.assertEqual(len(upper["outdoor_spaces"]), 1)
        self.assertFalse(any(c["a"].startswith("OUT-") or str(c.get("b")).startswith("OUT-") and c.get("exterior")
                             for c in upper["room_connections"] if c.get("exterior")))

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
