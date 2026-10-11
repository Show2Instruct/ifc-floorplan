"""Walls, openings and slabs read back as parameters a BIM tool can rebuild from."""
import sys
from pathlib import Path
import unittest

from shapely.geometry import Polygon, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fp.vectorize import opening_on_wall, rebuilt_wall, storey_vector, wall_axis, wall_extensions
from fp.verify_vector import verify


def element(step, cls, poly, z0=0., z1=3., gid=None):
    return {"step_id": step, "ifc_class": cls, "global_id": gid or f"G{step}", "_poly": poly,
            "bbox_min": [*poly.bounds[:2], z0], "bbox_max": [*poly.bounds[2:], z1]}


class Walls(unittest.TestCase):
    def test_a_straight_wall_gives_its_axis_and_thickness(self):
        start, end, thickness, fill = wall_axis(box(0, 0, 4, .2))
        self.assertAlmostEqual(abs(end[0] - start[0]), 4.)
        self.assertAlmostEqual(start[1], .1)
        self.assertAlmostEqual(thickness, .2)
        self.assertAlmostEqual(fill, 1.)

    def test_the_axis_direction_does_not_depend_on_vertex_order(self):
        a = wall_axis(box(0, 0, 4, .2))
        b = wall_axis(Polygon([(4, .2), (0, .2), (0, 0), (4, 0)]))
        self.assertEqual([round(v, 6) for v in (*a[0], *a[1])], [round(v, 6) for v in (*b[0], *b[1])])

    def test_a_wall_that_is_not_straight_says_so(self):
        l_shape = Polygon([(0, 0), (4, 0), (4, .2), (.2, .2), (.2, 3), (0, 3)])
        vector = storey_vector([element(1, "IfcWall", l_shape)], [], [], 0., 3.)
        self.assertFalse(vector["walls"][0]["straight"])
        self.assertEqual(vector["counts"]["walls_not_straight"], 1)

    def test_a_cavity_wall_is_straight_and_reports_its_cavity(self):
        # two leaves with a gap between them: straight, 0.30 thick, a third of it air
        from shapely.geometry import MultiPolygon
        leaves = MultiPolygon([box(0, 0, 5, .1), box(0, .2, 5, .3)])
        wall = storey_vector([element(1, "IfcWall", leaves)], [], [], 0., 3.)["walls"][0]
        self.assertTrue(wall["straight"])
        self.assertAlmostEqual(wall["thickness_m"], .3)
        self.assertAlmostEqual(wall["solid_fill"], .667, places=3)

    def test_a_wall_built_from_parts_is_read_from_its_parts_and_hosts_its_door(self):
        # the IfcWall carries no shape; its layer parts do
        wall = {"step_id": 1, "ifc_class": "IfcWall", "global_id": "W1"}
        inner = element(11, "IfcBuildingElementPart", box(0, 0, 6, .1))
        outer = element(12, "IfcBuildingElementPart", box(0, .1, 6, .25))
        door = element(2, "IfcDoor", box(3, 0, 4, .25), 0., 2.1)
        relations = [{"type": "part_of", "source": 11, "target": 1},
                     {"type": "part_of", "source": 12, "target": 1},
                     {"type": "hosts", "source": 1, "target": 2}]
        vector = storey_vector([wall, door], relations, [], 0., 3., by_id={11: inner, 12: outer, 1: wall})
        self.assertEqual(vector["walls"][0]["composed_from_parts"], 2)
        self.assertAlmostEqual(vector["walls"][0]["thickness_m"], .25)
        self.assertAlmostEqual(vector["openings"][0]["offset_m"], 3.5)

    def test_parameters_rebuild_the_footprint_they_came_from(self):
        footprint = Polygon([(0, 0), (3, 3), (2.86, 3.14), (-.14, .14)])      # rotated, 0.2 thick
        vector = storey_vector([element(1, "IfcWall", footprint)], [], [], 0., 3.)
        rebuilt = rebuilt_wall(vector["walls"][0])
        self.assertGreater(rebuilt.intersection(footprint).area / rebuilt.union(footprint).area, .99)


class Openings(unittest.TestCase):
    def test_a_door_is_placed_along_its_host_wall(self):
        wall = element(1, "IfcWall", box(0, 0, 6, .2), 0., 3.)
        door = element(2, "IfcDoor", box(2, 0, 2.9, .2), 0., 2.1, gid="D2")
        relations = [{"type": "hosts", "source": 1, "target": 2}]
        links = [{"a": "r1", "b": "r2", "type": "connected_by_door", "door": "D2"}]
        vector = storey_vector([wall, door], relations, links, 0., 3.)
        opening = vector["openings"][0]
        self.assertEqual(opening["host_wall"], "wall_1")
        self.assertAlmostEqual(opening["offset_m"], 2.45)
        self.assertAlmostEqual(opening["width_m"], .9)
        self.assertAlmostEqual(opening["off_axis_m"], 0.)
        self.assertEqual((opening["sill_m"], opening["head_m"]), (0., 2.1))
        self.assertEqual(opening["rooms"], [["r1", "r2"]])

    def test_a_window_keeps_its_sill(self):
        wall = element(1, "IfcWall", box(0, 0, 6, .2), 3., 6.)
        window = element(2, "IfcWindow", box(1, 0, 2.2, .2), 3.9, 5.3)
        vector = storey_vector([wall, window], [{"type": "hosts", "source": 1, "target": 2}], [], 3., 6.)
        self.assertAlmostEqual(vector["openings"][0]["sill_m"], .9)
        self.assertEqual(vector["openings"][0]["kind"], "window")

    def test_an_opening_whose_host_is_on_another_storey_is_still_placed(self):
        wall = element(1, "IfcWall", box(0, 0, 6, .2))
        door = element(2, "IfcDoor", box(4, 0, 5, .2))
        vector = storey_vector([door], [{"type": "hosts", "source": 1, "target": 2}], [], 0., 3.,
                               by_id={1: wall})
        self.assertAlmostEqual(vector["openings"][0]["offset_m"], 4.5)

    def test_a_curtain_panel_holding_a_door_is_read_as_its_wall(self):
        # model_133: the door's host is an IfcPlate, a glass panel standing in for a wall
        panel = element(1, "IfcPlate", box(0, 0, 3, .05))
        door = element(2, "IfcDoor", box(1, 0, 2, .05), 0., 2.1)
        vector = storey_vector([panel, door], [{"type": "hosts", "source": 1, "target": 2}], [], 0., 3.)
        self.assertEqual([(w["id"], w["ifc_class"]) for w in vector["walls"]], [("wall_1", "IfcPlate")])
        self.assertEqual(vector["openings"][0]["host_wall"], "wall_1")
        self.assertAlmostEqual(vector["openings"][0]["offset_m"], 1.5)
        # a panel that holds nothing stays out of the wall list
        vector = storey_vector([panel], [], [], 0., 3.)
        self.assertEqual(vector["walls"], [])

    def test_an_opening_left_unfilled_is_cut_into_its_wall_too(self):
        # an archway has no door to place, only its void; a rebuilt wall needs the hole all the same
        wall = element(1, "IfcWall", box(0, 0, 6, .2))
        void = element(2, "IfcOpeningElement", box(2, -.5, 3.2, .7), 0., 2.3)
        hosted_door = element(3, "IfcDoor", box(4, 0, 4.9, .2), 0., 2.1)
        door_void = element(4, "IfcOpeningElement", box(4, 0, 4.9, .2), 0., 2.1)
        relations = [{"type": "voids", "source": 1, "target": 2}, {"type": "voids", "source": 1, "target": 4},
                     {"type": "hosts", "source": 1, "target": 3, "via_opening": 4}]
        links = [{"a": "r1", "b": "r2", "type": "open_passage", "opening": "G2"}]
        vector = storey_vector([wall, void, hosted_door, door_void], relations, links, 0., 3.,
                               by_id={4: door_void})
        kinds = {o["id"]: o for o in vector["openings"]}
        self.assertEqual(sorted(o["kind"] for o in vector["openings"]), ["door", "opening"])
        arch = kinds["opening_2"]
        self.assertEqual(arch["host_wall"], "wall_1")
        self.assertAlmostEqual(arch["offset_m"], 2.6)
        self.assertAlmostEqual(arch["width_m"], 1.2)
        self.assertEqual((arch["sill_m"], arch["head_m"]), (0., 2.3))
        self.assertEqual(arch["rooms"], [["r1", "r2"]])

    def test_an_opening_credited_to_the_wrong_wall_at_a_join_goes_on_the_wall_it_lies_in(self):
        # ResBIM 1014: the void sits in the horizontal wall, but the file names the vertical wall
        # that joins it at that end
        across = element(1, "IfcWall", box(0, 0, 6, .2))
        up = element(2, "IfcWall", box(5.8, .2, 6, 6))
        void = element(3, "IfcOpeningElement", box(2, -.3, 3, .5), 0., 2.1)
        vector = storey_vector([across, up, void], [{"type": "voids", "source": 2, "target": 3}], [], 0., 3.)
        arch = vector["openings"][0]
        self.assertEqual((arch["host_wall"], arch["host_from"], arch["host_stated"]), ("wall_1", "geometry", "wall_2"))
        self.assertAlmostEqual(arch["offset_m"], 2.5)
        self.assertAlmostEqual(arch["off_axis_m"], 0.)
        # named rightly, it stays where the file puts it
        vector = storey_vector([across, up, void], [{"type": "voids", "source": 1, "target": 3}], [], 0., 3.)
        self.assertNotIn("host_from", vector["openings"][0])
        # ResBIM's commoner case: next to the join the void also clips the corner of the wall named,
        # and the wall it overlaps far more holds it
        corner = element(4, "IfcOpeningElement", box(5.0, -.3, 5.95, .5), 0., 2.1)
        vector = storey_vector([across, up, corner], [{"type": "voids", "source": 2, "target": 4}], [], 0., 3.)
        arch = vector["openings"][0]
        self.assertEqual((arch["host_wall"], arch.get("host_from")), ("wall_1", "geometry"))
        self.assertAlmostEqual(arch["width_m"], .95)

    def test_a_void_in_a_gap_of_its_wall_stays_there_whatever_stands_in_it(self):
        # model_59: a full-height void the length of a facade leaves a gap in the concrete wall named,
        # a curtain wall stands in it, and a partition's end just reaches into it
        from shapely.geometry import MultiPolygon
        wall = element(1, "IfcWall", MultiPolygon([box(0, 0, 3, .25), box(8, 0, 10, .25)]), 0., 3.5)
        void = element(2, "IfcOpeningElement", box(3, -.15, 8, .4), 0., 7.4)
        facade = element(3, "IfcCurtainWall", box(3, .05, 8, .2), 0., 7.4)
        partition = element(4, "IfcWall", box(7.9, .25, 8.05, 3), 0., 3.5)
        relations = [{"type": "voids", "source": 1, "target": 2}]
        vector = storey_vector([wall, void, facade, partition], relations, [], 0., 3.5)
        arch = vector["openings"][0]
        self.assertEqual((arch["host_wall"], arch.get("host_from")), ("wall_1", None))
        self.assertEqual((arch["offset_m"], arch["width_m"]), (5.5, 5.))

    def test_a_curtain_wall_does_not_take_an_opening_by_overlapping_it(self):
        # the void clips the corner of the wall named and lies along a curtain wall: the curtain wall
        # stands in it, so only an IfcWall could hold it instead of the wall named
        named = element(1, "IfcWall", box(0, 0, .2, 4), 0., 3.)
        facade = element(2, "IfcCurtainWall", box(.2, 0, 3, .1), 0., 3.)
        void = element(3, "IfcOpeningElement", box(.1, -.1, 3, .2), 0., 3.)
        vector = storey_vector([named, facade, void], [{"type": "voids", "source": 1, "target": 3}], [], 0., 3.)
        self.assertEqual(vector["openings"][0]["host_wall"], "wall_1")
        # an IfcWall there holds it, as at a join
        wall = element(2, "IfcWall", box(.2, 0, 3, .1), 0., 3.)
        vector = storey_vector([named, wall, void], [{"type": "voids", "source": 1, "target": 3}], [], 0., 3.)
        self.assertEqual((vector["openings"][0]["host_wall"], vector["openings"][0]["host_from"]), ("wall_2", "geometry"))
        # model_159: a door the file credits to a pier 19 m away goes in the curtain wall it stands in
        pier = element(1, "IfcWall", box(20, 5, 20.3, 5.3), 0., 3.)
        door = element(4, "IfcDoor", box(1, 0, 2, .1), 0., 2.1)
        vector = storey_vector([pier, facade, door], [{"type": "hosts", "source": 1, "target": 4}], [], 0., 3.)
        self.assertEqual((vector["openings"][0]["host_wall"], vector["openings"][0]["host_from"]), ("wall_2", "geometry"))

    def test_a_void_repeating_a_doors_opening_is_not_a_second_opening(self):
        wall = element(1, "IfcWall", box(0, 0, 6, .2))
        door = element(2, "IfcDoor", box(2, 0, 2.9, .2), 0., 2.1)
        door_void = element(3, "IfcOpeningElement", box(2, 0, 2.9, .2), 0., 2.1)
        repeat = element(4, "IfcOpeningElement", box(2, -.1, 2.9, .3), 0., 2.1)
        relations = [{"type": "voids", "source": 1, "target": 3}, {"type": "voids", "source": 1, "target": 4},
                     {"type": "hosts", "source": 1, "target": 2, "via_opening": 3}]
        vector = storey_vector([wall, door, door_void, repeat], relations, [], 0., 3., by_id={3: door_void})
        self.assertEqual([o["kind"] for o in vector["openings"]], ["door"])
        self.assertEqual(vector["counts"]["voids_repeating_another_opening"], 1)

    def test_a_void_above_a_door_on_the_floor_below_is_an_opening_of_its_own(self):
        # model_0: unglazed openings on the ground floor right below first-floor windows
        wall = element(1, "IfcWall", box(0, 0, 6, .2), 0., 7.)
        window = element(2, "IfcWindow", box(2, 0, 3, .2), 4.9, 7.2)
        window_void = element(3, "IfcOpeningElement", box(2, 0, 3, .2), 4.9, 7.2)
        below = element(4, "IfcOpeningElement", box(2, 0, 3, .2), .75, 3.25)
        relations = [{"type": "hosts", "source": 1, "target": 2, "via_opening": 3},
                     {"type": "voids", "source": 1, "target": 3}, {"type": "voids", "source": 1, "target": 4}]
        vector = storey_vector([wall, below], relations, [], 0., 3.5, by_id={3: window_void, 2: window})
        self.assertEqual([(o["id"], o["kind"]) for o in vector["openings"]], [("opening_4", "opening")])
        # the same void at the window's height is the window's hole cut again
        again = element(4, "IfcOpeningElement", box(2, -.1, 3, .3), 4.9, 7.2)
        vector = storey_vector([wall, again], relations, [], 0., 7.5, by_id={3: window_void, 2: window})
        self.assertEqual(vector["openings"], [])

    def test_only_a_wall_standing_where_the_void_cuts_its_wall_holds_it_instead(self):
        # model_38: a facade void two storeys high at the end of a first-floor wall, over a ground
        # floor wall that runs on below it; the void cuts nothing there
        upper = element(1, "IfcWall", box(0, 0, 6, .24), 4.8, 9.4)
        below = element(2, "IfcWall", box(6, 0, 13, .24), 0., 4.8)
        void = element(3, "IfcOpeningElement", box(6, -.15, 13, .39), .15, 9.6)
        extensions = wall_extensions([upper, below, void], [{"type": "voids", "source": 1, "target": 3}])
        self.assertEqual(extensions[1][:3], (0., 13., ["opening_3"]))

    def test_a_pier_is_run_over_a_void_along_either_of_its_sides(self):
        # model_159: a 0.3 m square pier at the end of a facade void; its axis is read one way or
        # the other, and the void lies along whichever way it is not
        pier = element(1, "IfcWall", box(0, 0, .3, .3), 0., 4.2)
        start, end = wall_axis(pier["_poly"])[:2]
        along_x = abs(end[0] - start[0]) > abs(end[1] - start[1])
        void = element(2, "IfcOpeningElement", box(0, .3, .3, 5.) if along_x else box(.3, 0, 5., .3), 0., 8.6)
        relations = [{"type": "voids", "source": 1, "target": 2}]
        extensions = wall_extensions([pier, void], relations)
        self.assertEqual(extensions[1][2:], (["opening_2"], True))
        vector = storey_vector([pier, void], relations, [], 0., 4.2, extensions=extensions)
        rebuilt, opening = vector["walls"][0], vector["openings"][0]
        self.assertEqual((rebuilt["length_m"], rebuilt["thickness_m"], rebuilt["body_m"]), (5., .3, [0., .3]))
        self.assertEqual((opening["offset_m"], opening["width_m"], opening["off_axis_m"]), (2.65, 4.7, 0.))
        # a wall that is no pier is never read across
        wall = element(1, "IfcWall", box(0, 0, 3, .3), 0., 4.2)
        self.assertEqual(wall_extensions([wall, element(2, "IfcOpeningElement", box(0, .3, .3, 5.), 0., 8.6)], relations), {})

    def test_a_void_repeating_another_extends_no_wall(self):
        # model_77 cuts one 36 m facade opening through the piers at both ends of it; the vector
        # layer keeps it once, on the first, so only the first runs over it
        first = element(1, "IfcWall", box(0, 0, .4, .3), 0., 3.6)
        second = element(2, "IfcWall", box(10.3, 0, 10.7, .3), 0., 3.6)
        facade = element(3, "IfcOpeningElement", box(.4, 0, 10.3, .3), 0., 6.95)
        again = element(4, "IfcOpeningElement", box(.4, 0, 10.3, .3), 0., 6.95)
        relations = [{"type": "voids", "source": 1, "target": 3}, {"type": "voids", "source": 2, "target": 4}]
        extensions = wall_extensions([first, second, facade, again], relations)
        self.assertEqual({k: v[2] for k, v in extensions.items()}, {1: ["opening_3"]})
        vector = storey_vector([first, second, facade, again], relations, [], 0., 3.6, extensions=extensions)
        self.assertEqual([o["id"] for o in vector["openings"]], ["opening_3"])
        # on different floors, both are openings, and each extends its wall
        upper = element(4, "IfcOpeningElement", box(.4, 0, 10.3, .3), 7., 10.6)
        lifted = element(2, "IfcWall", box(10.3, 0, 10.7, .3), 7., 10.6)
        extensions = wall_extensions([first, lifted, facade, upper], relations, {1: 0, 3: 0, 2: 1, 4: 1})
        self.assertEqual(sorted(extensions), [1, 2])

    def test_the_opening_void_places_the_door_rather_than_a_leaf_drawn_open(self):
        wall = element(1, "IfcWall", box(0, 0, 6, .2))
        leaf = element(2, "IfcDoor", box(2, 0, 3.8, 1.))            # leaf swung out into the room
        void = element(3, "IfcOpeningElement", box(2, 0, 2.9, .2))
        relations = [{"type": "hosts", "source": 1, "target": 2, "via_opening": 3}]
        opening = storey_vector([wall, leaf], relations, [], 0., 3., by_id={3: void})["openings"][0]
        self.assertAlmostEqual(opening["width_m"], .9)
        self.assertEqual(opening["placed_from"], "opening_void")

    def test_the_offset_is_measured_from_the_axis_start(self):
        axis = wall_axis(box(0, 0, 6, .2))
        offset, width, off_axis = opening_on_wall(box(5, 0, 6, .2), axis[:2])
        self.assertAlmostEqual(offset, 5.5)
        self.assertAlmostEqual(width, 1.)

    def test_a_wall_whose_end_an_opening_took_runs_over_the_opening(self):
        # model_172: a full-height archway at the end of its wall leaves the wall ending where the
        # archway begins, while the file still names that wall as its host; a glazed screen, a
        # curtain wall, stands in the archway
        wall = element(1, "IfcWall", box(0, 0, 2, .1), 0., 3.05)
        arch = element(2, "IfcOpeningElement", box(2, -.1, 3.7, .2), 0., 3.05)
        screen = element(3, "IfcCurtainWall", box(2.1, 0, 3.6, .1), 0., 3.05)
        relations = [{"type": "voids", "source": 1, "target": 2}]
        extensions = wall_extensions([wall, arch, screen], relations)
        self.assertEqual(list(extensions), [1])
        vector = storey_vector([wall, arch, screen], relations, [], 0., 3.05, extensions=extensions)
        rebuilt, opening = vector["walls"][0], vector["openings"][0]
        self.assertEqual((rebuilt["length_m"], rebuilt["body_m"], rebuilt["extended_over"]), (3.7, [0., 2.], ["opening_2"]))
        self.assertEqual((opening["host_wall"], opening.get("host_from")), ("wall_1", None))
        self.assertEqual((opening["offset_m"], opening["width_m"], opening["off_axis_m"]), (2.85, 1.7, 0.))
        self.assertAlmostEqual(rebuilt_wall(rebuilt, body=True).symmetric_difference(wall["_poly"]).area, 0., places=6)
        self.assertEqual(vector["counts"]["walls_extended_over_end_openings"], 1)
        # without the reading, the archway lies past the end of its wall
        plain = storey_vector([wall, arch], relations, [], 0., 3.05)
        self.assertGreater(plain["openings"][0]["offset_m"], plain["walls"][0]["length_m"])

    def test_a_wall_extended_at_its_start_shifts_every_opening_on_it(self):
        # a door taller than its wall at the wall's start (model_168), another door in the middle,
        # and a third on the storey above whose host is this wall
        wall = element(1, "IfcWall", box(1, 0, 5, .2), 0., 1.)
        void = element(2, "IfcOpeningElement", box(.1, 0, 1, .2), 0., 2.1)
        door = element(3, "IfcDoor", box(.1, 0, 1, .2), 0., 2.1)
        middle = element(4, "IfcDoor", box(3, 0, 3.9, .2), 0., .9)
        above = element(5, "IfcDoor", box(4, 0, 4.8, .2), 3., 5.)
        relations = [{"type": "voids", "source": 1, "target": 2},
                     {"type": "hosts", "source": 1, "target": 3, "via_opening": 2},
                     {"type": "hosts", "source": 1, "target": 4}, {"type": "hosts", "source": 1, "target": 5}]
        by_id = {e["step_id"]: e for e in (wall, void, door, middle, above)}
        extensions = wall_extensions(list(by_id.values()), relations)
        self.assertEqual(extensions[1][2], ["opening_3"])
        lower = storey_vector([wall, void, door, middle], relations, [], 0., 3., by_id, extensions)
        upper = storey_vector([above], relations, [], 3., 6., by_id, extensions)
        self.assertEqual(lower["walls"][0]["axis"], [[.1, .1], [5., .1]])
        self.assertEqual(lower["walls"][0]["body_m"], [.9, 4.9])
        offsets = {o["id"]: o["offset_m"] for o in lower["openings"] + upper["openings"]}
        self.assertEqual(offsets, {"opening_3": .45, "opening_4": 3.35, "opening_5": 4.3})

    def test_no_wall_is_extended_where_the_file_leaves_a_lintel_or_another_wall_holds_the_void(self):
        # model_21: a door at the end of a wall twice its height, with no wall above it
        wall = element(1, "IfcWall", box(0, 0, .2, 4), 0., 4.5)
        void = element(2, "IfcOpeningElement", box(0, -.85, .2, 0), 0., 2.1)
        relations = [{"type": "voids", "source": 1, "target": 2}]
        self.assertEqual(wall_extensions([wall, void], relations), {})
        # a full-height void past the end of one wall that stands in another is the other's opening
        end = element(3, "IfcWall", box(0, 0, 2, .2), 0., 3.)
        across = element(4, "IfcWall", box(2, -1, 2.3, 1), 0., 3.)
        hole = element(5, "IfcOpeningElement", box(2, 0, 2.3, .2), 0., 3.)
        self.assertEqual(wall_extensions([end, across, hole], [{"type": "voids", "source": 3, "target": 5}]), {})


class Storey(unittest.TestCase):
    def test_height_runs_to_the_next_level_and_is_read_from_the_walls_at_the_top(self):
        self.assertEqual(storey_vector([], [], [], 3., 6.3)["height_m"], 3.3)
        self.assertEqual(storey_vector([], [], [], 3., 6.3)["height_from"], "next_level")
        walls = [element(k, "IfcWall", box(0, k, 4, k + .2), 6.3, 6.3 + h) for k, h in enumerate((2.8, 2.9, 2.9, 1.1))]
        parapet = element(9, "IfcWall", box(0, 9, 4, 9.2), 9.2, 10.)       # on the roof, not this floor
        top = storey_vector(walls + [parapet], [], [], 6.3, None)
        self.assertEqual((top["height_m"], top["height_from"]), (2.9, "walls"))
        self.assertEqual(storey_vector([], [], [], 6.3, None)["height_m"], None)

    def test_a_slab_keeps_its_top_and_thickness(self):
        slab = element(5, "IfcSlab", box(0, 0, 10, 8), -.25, 0.)
        vector = storey_vector([slab], [], [], 0., 3.)
        self.assertEqual((vector["slabs"][0]["top_m"], vector["slabs"][0]["thickness_m"]), (0., .25))
        self.assertEqual(vector["slabs"][0]["area_m2"], 80.)


class Checker(unittest.TestCase):
    def test_a_run_whose_walls_and_openings_are_right_passes_every_check(self):
        import json, tempfile
        from fp.extract import poly_geometry
        wall = element(1, "IfcWall", box(0, 0, 6, .2))
        door = element(2, "IfcDoor", box(2, 0, 2.9, .2), 0., 2.1)
        vector = storey_vector([wall, door], [{"type": "hosts", "source": 1, "target": 2}], [], 0., 3.)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "models" / "m"
            folder.mkdir(parents=True)
            (folder / "building.json").write_text(json.dumps(dict(
                storeys=[dict(name="G", storey_index=0)], relations=[])))
            elements = [dict(step_id=e["step_id"], footprint_geometry=poly_geometry(e["_poly"]))
                        for e in (wall, door)]
            (folder / "storey_00.json").write_text(json.dumps(dict(
                height_m=3., elements=elements, vector=vector)))
            summary = verify(tmp)
        self.assertEqual(summary["straight_wall_rebuild_iou"]["within_0_9"], 1.)
        self.assertEqual(summary["share_of_openings_placed"], 1.)
        self.assertEqual(summary["share_of_placed_openings_on_their_host"], 1.)

    def test_an_opening_whose_wall_belongs_to_the_storey_below_is_checked_on_that_wall(self):
        import json, tempfile
        from fp.extract import poly_geometry
        wall = element(1, "IfcWall", box(0, 0, 6, .2))
        door = element(2, "IfcDoor", box(2, 0, 2.9, .2), 0., 2.1)
        stray = element(3, "IfcDoor", box(2, 3, 2.9, 3.2), 0., 2.1)
        hosts = [{"type": "hosts", "source": 1, "target": 2}, {"type": "hosts", "source": 1, "target": 3}]
        by_id = {1: wall}
        lower = storey_vector([wall], hosts, [], 0., 3., by_id)
        upper = storey_vector([door, stray], hosts, [], 0., 3., by_id)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "models" / "m"
            folder.mkdir(parents=True)
            (folder / "building.json").write_text(json.dumps(dict(
                storeys=[dict(name="G", storey_index=0), dict(name="1", storey_index=1)], relations=[])))
            for index, (vector, owned) in enumerate(((lower, [wall]), (upper, [door, stray]))):
                elements = [dict(step_id=e["step_id"], footprint_geometry=poly_geometry(e["_poly"])) for e in owned]
                (folder / f"storey_{index:02d}.json").write_text(json.dumps(dict(
                    height_m=3., elements=elements, vector=vector)))
            summary = verify(tmp)
        self.assertEqual(summary["counts"]["openings_on_a_host_from_another_storey"], 2)
        self.assertEqual(summary["counts"]["openings_on_their_host_wall"], 1)
        self.assertEqual(summary["counts"]["openings_off_their_host_wall"], 1)    # 3 m off its wall

    def test_a_wall_run_over_the_opening_at_its_end_is_checked_where_its_geometry_stands(self):
        import json, tempfile
        from fp.extract import poly_geometry
        wall = element(1, "IfcWall", box(0, 0, 2, .1), 0., 3.05)
        arch = element(2, "IfcOpeningElement", box(2, -.1, 3.7, .2), 0., 3.05)
        relations = [{"type": "voids", "source": 1, "target": 2}]
        vector = storey_vector([wall, arch], relations, [], 0., 3.05,
                               extensions=wall_extensions([wall, arch], relations))
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "models" / "m"
            folder.mkdir(parents=True)
            (folder / "building.json").write_text(json.dumps(dict(
                storeys=[dict(name="G", storey_index=0)], relations=relations)))
            elements = [dict(step_id=e["step_id"], footprint_geometry=poly_geometry(e["_poly"])) for e in (wall, arch)]
            (folder / "storey_00.json").write_text(json.dumps(dict(height_m=3.05, elements=elements, vector=vector)))
            summary = verify(tmp)
        self.assertEqual(summary["straight_wall_rebuild_iou"]["within_0_9"], 1.)
        self.assertEqual(summary["share_of_placed_openings_on_their_host"], 1.)
        self.assertEqual(summary["counts"]["walls_extended_over_end_openings"], 1)


if __name__ == "__main__":
    unittest.main()
