"""Room-to-room edges: doors, open passages and stairs."""
import sys
from pathlib import Path
import unittest

from shapely.geometry import box
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fp.connectivity import (door_connections, exterior_id, exterior_passages,
                             open_passages, outdoor_spaces, outside_region,
                             vertical_connections)


def room(rid, x0, y0, x1, y1, source="IfcSpace"):
    return {"id": rid, "source": source, "_poly": box(x0, y0, x1, y1)}


def door(step, x0, y0, x1, y1):
    return {"step_id": step, "global_id": f"G{step}", "ifc_class": "IfcDoor", "_poly": box(x0, y0, x1, y1)}


class Doors(unittest.TestCase):
    def test_a_door_joins_the_rooms_on_either_side(self):
        rooms = [room("a", 0, 0, 4, 4), room("b", 4.2, 0, 8, 4)]
        edges, counts = door_connections(rooms, [door(1, 4.0, 1.5, 4.2, 2.4)])
        self.assertEqual(len(edges), 1)
        self.assertEqual({edges[0]["a"], edges[0]["b"]}, {"a", "b"})
        self.assertEqual(edges[0]["type"], "connected_by_door")
        self.assertEqual(edges[0]["door"], "G1")
        self.assertAlmostEqual(edges[0]["width"], .9)
        self.assertEqual(counts["connected_by_door"], 1)

    def test_a_door_with_a_room_on_one_side_is_an_exterior_door(self):
        edges, counts = door_connections([room("a", 0, 0, 4, 4)], [door(1, 4.0, 1.5, 4.2, 2.4)])
        self.assertEqual(edges[0]["type"], "exterior_door")
        self.assertEqual(edges[0]["b"], None)
        self.assertEqual(counts["exterior_door"], 1)

    def test_a_thick_wall_still_reaches_the_room_behind_it(self):
        # the step-out has to clear the wall, not just the door panel
        rooms = [room("a", 0, 0, 4, 4), room("b", 4.6, 0, 8, 4)]
        edges, _ = door_connections(rooms, [door(1, 4.0, 1.5, 4.6, 2.4)])
        self.assertEqual(len(edges), 1)
        self.assertEqual({edges[0]["a"], edges[0]["b"]}, {"a", "b"})

    def test_a_door_inside_one_room_connects_nothing(self):
        edges, counts = door_connections([room("a", 0, 0, 8, 8)], [door(1, 4.0, 4.0, 4.2, 4.9)])
        self.assertEqual(edges, [])


class Passages(unittest.TestCase):
    def test_rooms_meeting_with_no_barrier_share_an_open_passage(self):
        rooms = [room("a", 0, 0, 4, 4), room("b", 4, 0, 8, 4)]
        edges, _ = open_passages(rooms, [])
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["type"], "open_passage")
        self.assertGreater(edges[0]["width"], 3.9)

    def test_a_wall_between_them_is_not_a_passage(self):
        rooms = [room("a", 0, 0, 4, 4), room("b", 4.2, 0, 8, 4)]
        edges, _ = open_passages(rooms, [box(4, -.1, 4.2, 4.1)])
        self.assertEqual(edges, [])

    def test_a_narrow_seam_is_not_a_way_through(self):
        rooms = [room("a", 0, 0, 4, 4), room("b", 4, 3.7, 8, 4)]
        edges, counts = open_passages(rooms, [])
        self.assertEqual(edges, [])
        self.assertEqual(counts.get("seam_too_narrow"), 1)


class Vertical(unittest.TestCase):
    def test_a_stair_joins_the_rooms_it_stands_in(self):
        stair = {"ifc_class": "IfcStairFlight", "bbox_min": [1, 1, 0], "bbox_max": [3, 3, 3.2],
                 "_poly": box(1, 1, 3, 3)}
        storeys = [{"step_id": 10, "elevation_m": 0.}, {"step_id": 20, "elevation_m": 3.}]
        rooms = {0: [room("g", 0, 0, 4, 4)], 1: [room("u", 0, 0, 4, 4)]}
        edges, counts = vertical_connections(rooms, [stair], storeys)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["type"], "vertically_connected")
        self.assertEqual(edges[0]["kind"], "stair")
        self.assertEqual((edges[0]["storey_a"], edges[0]["storey_b"]), (10, 20))

    def test_a_lift_through_four_levels_gives_three_edges_not_six(self):
        lift = {"ifc_class": "IfcTransportElement", "bbox_min": [1, 1, 0], "bbox_max": [2, 2, 9.5],
                "_poly": box(1, 1, 2, 2)}
        storeys = [{"step_id": i, "elevation_m": 3. * k} for k, i in enumerate((10, 20, 30, 40))]
        rooms = {k: [room(f"r{k}", 0, 0, 4, 4)] for k in range(4)}
        edges, _ = vertical_connections(rooms, [lift], storeys)
        self.assertEqual(len(edges), 3)
        self.assertEqual({e["kind"] for e in edges}, {"lift"})


class Outside(unittest.TestCase):
    def walls_around(self, x0, y0, x1, y1, gap=None):
        """Four walls, optionally leaving one side open for a given stretch."""
        t = .2
        pieces = [box(x0 - t, y0 - t, x1 + t, y0), box(x0 - t, y0 - t, x0, y1 + t),
                  box(x1, y0 - t, x1 + t, y1 + t)]
        if gap is None:
            pieces.append(box(x0 - t, y1, x1 + t, y1 + t))
        else:
            pieces += [box(x0 - t, y1, gap[0], y1 + t), box(gap[1], y1, x1 + t, y1 + t)]
        return pieces

    def test_a_room_open_along_one_side_reaches_the_outside(self):
        rooms = [room("a", 0, 0, 4, 4)]
        walls = self.walls_around(0, 0, 4, 4, gap=(1., 3.))
        edges, _ = exterior_passages(rooms, walls, storey_id="S1")
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["type"], "open_passage")
        self.assertEqual(edges[0]["b"], exterior_id("S1"))
        self.assertAlmostEqual(edges[0]["width"], 2., places=1)

    def test_a_room_walled_all_round_reaches_nothing(self):
        rooms = [room("a", 0, 0, 4, 4)]
        edges, _ = exterior_passages(rooms, self.walls_around(0, 0, 4, 4), storey_id="S1")
        self.assertEqual(edges, [])

    def test_a_storey_with_no_barriers_is_not_called_wide_open(self):
        # missing barriers mean we cannot see the enclosure, which is not the same as a room
        # standing in a field
        edges, counts = exterior_passages([room("a", 0, 0, 4, 4)], [], storey_id="S1")
        self.assertEqual(edges, [])
        self.assertEqual(counts["no_barriers_near_the_room"], 1)

    def test_a_neighbouring_room_beyond_the_gap_is_not_the_outside(self):
        rooms = [room("a", 0, 0, 4, 4), room("b", 0, 4.2, 4, 8)]
        walls = self.walls_around(0, 0, 4, 4, gap=(1., 3.))
        edges, _ = exterior_passages(rooms, walls, storey_id="S1")
        self.assertEqual([e for e in edges if e["a"] == "a"], [])

    def test_an_exterior_door_points_at_the_outside_node(self):
        edges, _ = door_connections([room("a", 0, 0, 4, 4)], [door(1, 4.0, 1.5, 4.2, 2.4)],
                                    storey_id="S1")
        self.assertEqual(edges[0]["b"], exterior_id("S1"))
        self.assertTrue(edges[0]["exterior"])

    def test_a_wall_below_the_cut_still_closes_the_room(self):
        # The section at 1.2 m sees nothing along that side, but a knee-high wall stands there,
        # so the room is not open to the outside.
        rooms = [room("a", 0, 0, 4, 4)]
        sectioned = [box(-.2, -.2, 4.2, 0), box(-.2, -.2, 0, 4.2), box(4, -.2, 4.2, 4.2)]
        low_wall = box(-.2, 4, 4.2, 4.2)
        open_edges, _ = exterior_passages(rooms, sectioned, storey_id="S1")
        self.assertEqual(len(open_edges), 1)
        closed, counts = exterior_passages(rooms, sectioned, storey_id="S1", at_any_height=[low_wall])
        self.assertEqual(closed, [])


class Tolerance(unittest.TestCase):
    def walls_with_hairline(self, gap):
        # a 4 x 4 room whose top wall is two pieces with `gap` metres between them
        return [box(-.2, -.2, 4.2, 0), box(-.2, -.2, 0, 4.2), box(4, -.2, 4.2, 4.2),
                box(-.2, 4, 2., 4.2), box(2. + gap, 4, 4.2, 4.2)]

    def test_a_hairline_joint_does_not_let_the_outside_into_a_room(self):
        # ResBIM model 9: exact geometry let open ground reach the whole Living Room
        walls = self.walls_with_hairline(.02)
        middle = box(1.5, 1.5, 2.5, 2.5)
        self.assertTrue(outside_region(walls, tolerance=0).intersects(middle))
        self.assertFalse(outside_region(walls).intersects(middle))

    def test_a_real_doorway_still_lets_it_in(self):
        walls = self.walls_with_hairline(.9)
        self.assertTrue(outside_region(walls).intersects(box(1.5, 1.5, 2.5, 2.5)))


class Outdoor(unittest.TestCase):
    def setUp(self):
        self.walls = [box(-.2, -.2, 4.2, 0), box(-.2, 4, 4.2, 4.2),
                      box(-.2, -.2, 0, 4.2), box(4, -.2, 4.2, 4.2)]
        self.rooms = [room("r", 0, 0, 4, 4)]
        self.slab = box(-.2, -.2, 4.2, 6.2)                     # runs 2 m past the top wall
        self.railings = [box(-.2, 6.1, 4.2, 6.2), box(-.2, 4.2, -.1, 6.2), box(4.1, 4.2, 4.2, 6.2)]

    def test_floor_past_the_walls_bordered_by_a_railing_is_an_outdoor_space(self):
        spaces, counts = outdoor_spaces([self.slab], self.railings, self.rooms,
                                        outside_region(self.walls), storey_id="S1")
        self.assertEqual(len(spaces), 1)
        self.assertEqual(spaces[0]["id"], "OUT-S1-1")
        self.assertGreater(spaces[0]["area_m2"], 8.)
        self.assertLess(spaces[0]["area_m2"], 9.)

    def test_without_a_railing_it_is_only_an_overhang(self):
        spaces, counts = outdoor_spaces([self.slab], [], self.rooms,
                                        outside_region(self.walls), storey_id="S1")
        self.assertEqual(spaces, [])
        self.assertEqual(counts["open_floor_without_railing"], 1)

    def test_a_railed_roof_with_no_room_beside_it_is_not_a_terrace(self):
        far = [room("r", 20, 20, 24, 24)]
        spaces, counts = outdoor_spaces([self.slab], self.railings, far,
                                        outside_region(self.walls), storey_id="S1")
        self.assertEqual(spaces, [])
        self.assertEqual(counts["open_floor_away_from_rooms"], 1)

    def test_a_door_onto_the_balcony_leads_to_the_balcony_not_the_street(self):
        outside = outside_region(self.walls)
        spaces, _ = outdoor_spaces([self.slab], self.railings, self.rooms, outside, storey_id="S1")
        leaf = door(1, 1.5, 4.0, 2.4, 4.2)
        edges, _ = door_connections(self.rooms + spaces, [leaf], storey_id="S1", outside=outside)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["type"], "connected_by_door")
        self.assertEqual({edges[0]["a"], edges[0]["b"]}, {"r", "OUT-S1-1"})


class VerticalRecall(unittest.TestCase):
    def test_a_stair_between_landings_is_placed_by_area_not_its_centre(self):
        # the stair's centre falls in the void between two rooms; most of it stands in one
        stair = {"ifc_class": "IfcStairFlight", "bbox_min": [0, 0, 0], "bbox_max": [4, 1, 3.2],
                 "_poly": box(0, 0, 4, 1)}
        storeys = [{"step_id": 10, "elevation_m": 0.}, {"step_id": 20, "elevation_m": 3.}]
        rooms = {0: [room("g", -.2, -.2, 3, 1.2)], 1: [room("u", -.2, -.2, 3, 1.2)]}
        edges, _ = vertical_connections(rooms, [stair], storeys)
        self.assertEqual(len(edges), 1)
        self.assertEqual({edges[0]["a"], edges[0]["b"]}, {"g", "u"})

    def test_four_flights_of_one_stair_are_one_way_up(self):
        storeys = [{"step_id": 10, "elevation_m": 0.}, {"step_id": 20, "elevation_m": 3.}]
        rooms = {0: [room("g", 0, 0, 4, 4)], 1: [room("u", 0, 0, 4, 4)]}
        flights = [{"ifc_class": "IfcStairFlight", "bbox_min": [1, 1, 0], "bbox_max": [3, 3, 3.2],
                    "_poly": box(1, 1, 3, 3)} for _ in range(4)]
        edges, _ = vertical_connections(rooms, flights, storeys)
        self.assertEqual(len(edges), 1)

class Envelope(unittest.TestCase):
    def test_a_door_into_a_space_we_missed_is_not_a_way_out(self):
        # The room beyond was never found, so the far side looks empty. It is still inside the
        # building, and calling it exterior asserts a way out that does not exist.
        walls = [box(-.2, -.2, 8.2, 0), box(-.2, 4, 8.2, 4.2), box(-.2, -.2, 0, 4.2),
                 box(8, -.2, 8.2, 4.2), box(4, -.2, 4.2, 4.2)]
        outside = outside_region(walls)
        rooms = [room("a", 0, 0, 4, 4)]
        leaf = door(1, 4.0, 1.5, 4.2, 2.4)
        edges, counts = door_connections(rooms, [leaf], storey_id="S1", outside=outside)
        self.assertEqual(edges, [])
        self.assertEqual(counts["door_to_a_space_we_did_not_find"], 1)
        # with no envelope to consult, the same door is reported as exterior
        edges, _ = door_connections(rooms, [leaf], storey_id="S1")
        self.assertEqual(edges[0]["type"], "exterior_door")

    def test_a_door_into_a_recess_is_still_a_way_out(self):
        # A U-shaped building. The notch between the two wings lies inside the building's
        # outline but is open to the sky, so a door into it does lead out. Asking whether the
        # far side is within the outline would get this wrong; asking whether it can be reached
        # from open ground gets it right.
        walls = [box(-.2, -.2, 8.2, 0),                       # bottom
                 box(-.2, -.2, 0, 6.2), box(8, -.2, 8.2, 6.2),  # the two outer sides
                 box(-.2, 6, 3.2, 6.2), box(4.8, 6, 8.2, 6.2),  # top, with the notch left open
                 box(3, 3, 3.2, 6.2), box(4.8, 3, 5, 6.2),      # the sides of the notch
                 box(3, 3, 5, 3.2)]                             # the back of the notch
        outside = outside_region(walls)
        edges, _ = door_connections([room("a", 0, 0, 3, 6)], [door(1, 3.0, 4., 3.2, 4.9)],
                                    storey_id="S1", outside=outside)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["type"], "exterior_door")

    def test_something_standing_beyond_the_door_does_not_hide_the_open_ground(self):
        # model_207: open ground just past the door, and a curtain-wall member 1.6 m out. A single
        # probe at that distance landed inside the member and the door was judged to lead nowhere.
        walls = [box(-.2, -.2, 4.2, 0), box(-.2, 4, 4.2, 4.2), box(-.2, -.2, 0, 4.2),
                 box(4, -.2, 4.2, 4.2)]
        member = box(5.65, 1., 5.85, 3.)                     # 1.55-1.75 m beyond the door's centre
        barriers = walls + [member]
        outside = outside_region(barriers)
        blocked = unary_union([b.buffer(.06) for b in barriers])
        edges, counts = door_connections([room("a", 0, 0, 4, 4)], [door(1, 4.0, 1.5, 4.2, 2.4)],
                                         storey_id="S1", outside=outside, blocked=blocked)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["type"], "exterior_door")

    def test_a_door_facing_only_barriers_is_not_called_a_way_out(self):
        walls = [box(-.2, -.2, 4.2, 0), box(-.2, 4, 4.2, 4.2), box(-.2, -.2, 0, 4.2),
                 box(4, -.2, 7., 4.2)]                         # solid beyond the door for 3 m
        outside = outside_region(walls)
        blocked = unary_union([b.buffer(.06) for b in walls])
        edges, counts = door_connections([room("a", 0, 0, 4, 4)], [door(1, 4.0, 1.5, 4.2, 2.4)],
                                         storey_id="S1", outside=outside, blocked=blocked)
        self.assertEqual(edges, [])
        self.assertEqual(counts["door_far_side_blocked"], 1)

    def test_a_courtyard_is_not_open_ground(self):
        # Fully enclosed by the building, so a door into it is not a way out of the building.
        walls = [box(-.2, -.2, 8.2, 0), box(-.2, 6, 8.2, 6.2),
                 box(-.2, -.2, 0, 6.2), box(8, -.2, 8.2, 6.2),
                 box(3, 2, 3.2, 4.2), box(5, 2, 5.2, 4.2),
                 box(3, 2, 5.2, 2.2), box(3, 4, 5.2, 4.2)]
        outside = outside_region(walls)
        edges, counts = door_connections([room("a", 0, 0, 3, 6)], [door(1, 3.0, 2.6, 3.2, 3.5)],
                                         storey_id="S1", outside=outside)
        self.assertEqual(edges, [])
        self.assertEqual(counts["door_to_a_space_we_did_not_find"], 1)

    def test_a_door_through_the_outer_wall_is_still_a_way_out(self):
        walls = [box(-.2, -.2, 4.2, 0), box(-.2, 4, 4.2, 4.2), box(-.2, -.2, 0, 4.2),
                 box(4, -.2, 4.2, 4.2)]
        outside = outside_region(walls)
        edges, _ = door_connections([room("a", 0, 0, 4, 4)], [door(1, 4.0, 1.5, 4.2, 2.4)],
                                    storey_id="S1", outside=outside)
        self.assertEqual(edges[0]["type"], "exterior_door")
        self.assertEqual(edges[0]["b"], exterior_id("S1"))


if __name__ == "__main__":
    unittest.main()