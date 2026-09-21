#!/usr/bin/env python
"""Checks for storey membership and outline serialisation.

The storey an element belongs to is read from the IFC relationship tables first, with the
placement chain only as a fallback: in 19 GNI models the placement chain alone put 226 doors
and windows one or more storeys too low. Outlines are also serialised with their holes, which
the older ring format dropped.

    python tests/test_storeys.py
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import ifcopenshell
import ifcopenshell.guid
from shapely.geometry import MultiPolygon, Polygon, box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fp.build import _path, assign_storeys
from fp.extract import (extract, poly_coords, poly_geometry, storey_from_placement,
                        storey_from_relationships)

MODEL_103 = Path(__file__).resolve().parents[2] / "data/gni-bim/2025_BIMfundamentals/model_103.ifc"
RESBIM_99 = Path(__file__).resolve().parents[2] / "data/ResBIM-IFC/data/99/99.ifc"
CONTAINED, AGGREGATED = "IfcRelContainedInSpatialStructure", "IfcRelAggregates"
VOIDS, FILLS = "IfcRelVoidsElement", "IfcRelFillsElement"


class Building:
    """A minimal IFC4 building with a ground floor and a first floor 4.8 m up."""

    def __init__(self):
        f = self.f = ifcopenshell.file(schema="IFC4")
        metre = f.create_entity("IfcSIUnit", UnitType="LENGTHUNIT", Name="METRE")
        project = self.rooted("IfcProject", Name="p",
                              UnitsInContext=f.create_entity("IfcUnitAssignment", Units=[metre]))
        self.site = self.rooted("IfcSite", Name="site", ObjectPlacement=self.placement(None))
        self.building = self.rooted("IfcBuilding", Name="b",
                                    ObjectPlacement=self.placement(self.site.ObjectPlacement))
        self.gf = self.rooted("IfcBuildingStorey", Name="GF", Elevation=0.0,
                              ObjectPlacement=self.placement(self.building.ObjectPlacement))
        self.l1 = self.rooted("IfcBuildingStorey", Name="L1", Elevation=4.8,
                              ObjectPlacement=self.placement(self.building.ObjectPlacement, 4.8))
        self.aggregate(project, [self.site])
        self.aggregate(self.site, [self.building])
        self.aggregate(self.building, [self.gf, self.l1])

    def rooted(self, cls, **attrs):
        return self.f.create_entity(cls, GlobalId=ifcopenshell.guid.new(), **attrs)

    def placement(self, relative_to, z=0.0):
        point = self.f.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, z))
        return self.f.create_entity("IfcLocalPlacement", PlacementRelTo=relative_to,
                                    RelativePlacement=self.f.create_entity("IfcAxis2Placement3D",
                                                                           Location=point))

    def aggregate(self, whole, parts):
        self.rooted("IfcRelAggregates", RelatingObject=whole, RelatedObjects=parts)

    def contain(self, structure, elements):
        self.rooted("IfcRelContainedInSpatialStructure", RelatingStructure=structure,
                    RelatedElements=elements)

    def element(self, cls, relative_to, z=0.0, **attrs):
        return self.rooted(cls, ObjectPlacement=self.placement(relative_to, z), **attrs)


class StoreyMembership(unittest.TestCase):
    def test_containment_wins_over_a_placement_chain_reaching_a_lower_storey(self):
        b = Building()
        door = b.element("IfcDoor", b.gf.ObjectPlacement, 4.8)
        b.contain(b.l1, [door])
        self.assertEqual(storey_from_placement(door), b.gf.id())
        self.assertEqual(storey_from_relationships(door), (b.l1.id(), CONTAINED))

    def test_indirect_routes_name_every_relationship_followed(self):
        b = Building()
        space = b.element("IfcSpace", b.l1.ObjectPlacement)
        b.aggregate(b.l1, [space])
        chair = b.element("IfcFurniture", space.ObjectPlacement)
        b.contain(space, [chair])
        wall = b.element("IfcWall", b.l1.ObjectPlacement)
        b.contain(b.l1, [wall])
        panel = b.element("IfcPlate", wall.ObjectPlacement)
        b.aggregate(wall, [panel])
        opening = b.element("IfcOpeningElement", wall.ObjectPlacement)
        b.rooted("IfcRelVoidsElement", RelatingBuildingElement=wall, RelatedOpeningElement=opening)
        window = b.element("IfcWindow", opening.ObjectPlacement)
        b.rooted("IfcRelFillsElement", RelatingOpeningElement=opening, RelatedBuildingElement=window)

        expected = {
            space: AGGREGATED,
            chair: f"{CONTAINED} > {AGGREGATED}",
            panel: f"{AGGREGATED} > {CONTAINED}",
            opening: f"{VOIDS} > {CONTAINED}",
            window: f"{FILLS} > {VOIDS} > {CONTAINED}",
        }
        for element, route in expected.items():
            with self.subTest(element.is_a()):
                self.assertEqual(storey_from_relationships(element), (b.l1.id(), route))

    def test_relationships_above_storey_level_or_absent_give_no_storey(self):
        b = Building()
        post = b.element("IfcColumn", b.site.ObjectPlacement)
        b.contain(b.site, [post])
        storey, reason = storey_from_relationships(post)
        self.assertIsNone(storey)
        self.assertIn("IfcSite", reason)
        self.assertIsNone(storey_from_placement(post))

        note = b.element("IfcAnnotation", b.gf.ObjectPlacement)
        self.assertEqual(storey_from_relationships(note), (None, "no relationship"))
        self.assertEqual(storey_from_placement(note), b.gf.id())

    def test_extract_records_the_route_and_falls_back_to_placement(self):
        b = Building()
        door = b.element("IfcDoor", b.gf.ObjectPlacement, 4.8)
        b.contain(b.l1, [door])
        note = b.element("IfcAnnotation", b.gf.ObjectPlacement)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "building.ifc")
            b.f.write(path)
            data, _ = extract(path, threads=1)
        storey_name = {s["step_id"]: s["name"] for s in data["storeys"]}
        by_guid = {e["global_id"]: e for e in data["elements"]}
        self.assertEqual((storey_name[by_guid[door.GlobalId]["storey_step_id"]],
                          by_guid[door.GlobalId]["storey_link"]), ("L1", CONTAINED))
        self.assertEqual((storey_name[by_guid[note.GlobalId]["storey_step_id"]],
                          by_guid[note.GlobalId]["storey_link"]), ("GF", "ObjectPlacement"))

    def test_assignment_is_labelled_with_the_route_not_always_containment(self):
        route = f"{VOIDS} > {CONTAINED}"
        data = {"storeys": [{"step_id": 1, "elevation_m": 0.0}, {"step_id": 2, "elevation_m": 4.8}],
                "elements": [
                    {"step_id": 10, "storey_step_id": 2, "storey_link": route, "bbox_min": [0.0, 0.0, 0.0]},
                    {"step_id": 11, "storey_step_id": 1, "storey_link": "ObjectPlacement", "bbox_min": [0.0, 0.0, 0.0]},
                    {"step_id": 12, "storey_step_id": None, "storey_link": None, "bbox_min": [0.0, 0.0, 5.0]}]}
        assigned = assign_storeys(data)
        self.assertEqual(assigned[10], (1, route))
        self.assertEqual(assigned[11], (0, "ObjectPlacement"))
        self.assertEqual(assigned[12], (1, "inferred_from_elevation"))


@unittest.skipUnless(MODEL_103.is_file(), f"needs {MODEL_103}")
class Model103(unittest.TestCase):
    """26 windows and doors whose placement chain names the storey 4 cm below their own."""

    def test_doors_and_windows_follow_their_containment(self):
        f = ifcopenshell.open(str(MODEL_103))
        corrected = 0
        for e in f.by_type("IfcProduct"):
            rels = getattr(e, "ContainedInStructure", None) or ()
            if not rels or not rels[0].RelatingStructure.is_a("IfcBuildingStorey"):
                continue
            contained = rels[0].RelatingStructure.id()
            placed = storey_from_placement(e)
            if placed is None or placed == contained:
                continue
            corrected += 1
            self.assertEqual(storey_from_relationships(e), (contained, CONTAINED))
        self.assertEqual(corrected, 26)


class Outlines(unittest.TestCase):
    HOLED = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], holes=[[(4, 4), (6, 4), (6, 6), (4, 6)]])

    def test_geometry_keeps_the_hole_that_ring_format_drops(self):
        self.assertEqual(Polygon(poly_coords(self.HOLED)[0]).area, 100.0)
        geometry = poly_geometry(self.HOLED)
        self.assertEqual((len(geometry), len(geometry[0])), (1, 2))
        self.assertAlmostEqual(Polygon(geometry[0][0], geometry[0][1:]).area, 96.0)

    def test_multipolygon_and_nothing(self):
        self.assertEqual(len(poly_geometry(MultiPolygon([box(0, 0, 1, 1), box(2, 0, 3, 1)]))), 2)
        self.assertEqual(poly_geometry(None), [])

    def test_plan_path_draws_the_hole(self):
        frame = (0.0, 0.0, 10.0, 10.0, 1.0, 10, 10)
        self.assertEqual(_path(self.HOLED, frame).count("M"), 2)
        self.assertEqual(_path(box(0, 0, 10, 10), frame).count("M"), 1)


@unittest.skipUnless(RESBIM_99.is_file(), f"needs {RESBIM_99}")
class DenseMeshUnion(unittest.TestCase):
    """ResBIM 99 has two door leaves whose 2,636-face meshes break a plain GEOS union."""

    def test_footprint_survives_and_says_how(self):
        from fp.extract import footprint
        data, _ = extract(str(RESBIM_99), threads=1)
        by_step = {e["step_id"]: e for e in data["elements"]}
        for step in (3013, 3041):
            door = by_step[step]
            poly, method = footprint(door["_verts"], door["_faces"])
            self.assertEqual(method, "union_snapped")
            self.assertAlmostEqual(poly.area, 0.1175, places=3)
        ordinary = next(e for e in data["elements"] if e["ifc_class"] == "IfcWall" and "_verts" in e)
        self.assertEqual(footprint(ordinary["_verts"], ordinary["_faces"])[1], "union")


if __name__ == "__main__":
    unittest.main(verbosity=2)
