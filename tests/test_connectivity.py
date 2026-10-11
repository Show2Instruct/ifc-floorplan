"""Room-to-room edges: doors, open passages and stairs."""
import sys
from pathlib import Path
import unittest

from shapely.geometry import box
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fp.connectivity import (_ray_obstacle_limit, door_connections, exterior_id,
                             exterior_passages, open_passages, outdoor_spaces,
                             outside_region, room_boundaries, vertical_connections)


def room(rid, x0, y0, x1, y1, source="IfcSpace"):
    return {"id": rid, "source": source, "_poly": box(x0, y0, x1, y1)}


def door(step, x0, y0, x1, y1):
    return {"step_id": step, "global_id": f"G{step}", "ifc_class": "IfcDoor", "_poly": box(x0, y0, x1, y1)}


class Doors(unittest.TestCase):
    def test_search_cannot_jump_a_second_wall_to_reach_another_room(self):
        rooms = [room('a', 0, 0, 4, 4), room('b', 4.8, 0, 8, 4)]
        leaf = door(1, 4., 1.5, 4.2, 2.4)
        # A thin second wall lies between the doorway and b. It is thinner than
        # the old 5 cm point sampling and must still stop all three search rays.
        obstacles = unary_union([leaf['_poly'], box(4.621, 0, 4.629, 4)])
        edges, _ = door_connections(rooms, [leaf], blocked=obstacles)
        self.assertFalse(any(e['type'] == 'connected_by_door' for e in edges))

    def test_search_can_leave_the_door_and_its_initial_thick_wall(self):
        rooms = [room('a', 0, 0, 4, 4), room('b', 4.6, 0, 8, 4)]
        leaf = door(1, 4., 1.5, 4.6, 2.4)
        edges, _ = door_connections(rooms, [leaf], blocked=leaf['_poly'].buffer(.06))
        self.assertEqual({edges[0]['a'], edges[0]['b']}, {'a', 'b'})

    def test_a_door_set_back_from_its_wall_still_joins_both_rooms(self):
        # The wall's section fell back to its whole-element silhouette, which fills the opening,
        # and the leaf sits behind that. The ray starts clear of the wall, so a limit measured
        # from the ray origin stopped the search at the wall instead of crossing it.
        rooms = [room('a', 0, 0, 3.8, 4), room('b', 4.66, 0, 8, 4)]
        leaf = door(1, 3.8, 1.5, 4., 2.4)
        edges, _ = door_connections(rooms, [leaf], blocked=box(4., 0, 4.6, 4).buffer(.06))
        self.assertEqual({edges[0]['a'], edges[0]['b']}, {'a', 'b'})

    def test_a_wall_modelled_in_layers_is_one_wall_not_two(self):
        rooms = [room('a', 0, 0, 4, 4), room('b', 4.6, 0, 8, 4)]
        leaf = door(1, 4., 1.5, 4.2, 2.4)
        layers = unary_union([leaf['_poly'], box(4.4, 0, 4.45, 4)]).buffer(.06)
        edges, _ = door_connections(rooms, [leaf], blocked=layers)
        self.assertEqual({edges[0]['a'], edges[0]['b']}, {'a', 'b'})

    def test_the_ray_limit_crosses_the_first_wall_and_stops_at_the_second(self):
        blocked = unary_union([box(-3, 0, 3, .2), box(-3, 1., 3, 1.2)])
        # starting inside the first wall, and starting clear of it, must both reach past it
        self.assertAlmostEqual(_ray_obstacle_limit((0., .1), (0., 1.), 2., blocked), .9, places=2)
        self.assertAlmostEqual(_ray_obstacle_limit((0., -.05), (0., 1.), 2., blocked), 1.05, places=2)
        # with nothing beyond it, the whole reach is available
        self.assertEqual(_ray_obstacle_limit((0., -.05), (0., 1.), 2., box(-3, 0, 3, .2)), 2.)

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

    def test_a_leaf_drawn_open_is_crossed_on_the_axis_that_goes_through_it(self):
        # A leaf drawn open has a footprint about as deep as it is wide, so the long side is a
        # coin toss and the search ran along the wall: both rays met the hall and the door was
        # dismissed as standing inside one room.
        rooms = [room("hall", 0, 0, 8, 4), room("office", 0, 4.2, 8, 8)]
        leaf = door(1, 3.5, 3.4, 4.6, 4.56)           # 1.10 wide, 1.16 deep once swung open
        edges, counts = door_connections(rooms, [leaf])
        self.assertEqual(len(edges), 1)
        self.assertEqual({edges[0]["a"], edges[0]["b"]}, {"hall", "office"})
        self.assertEqual(counts.get("crossed_on_the_other_axis"), 1)

    def test_an_ordinary_door_is_still_crossed_on_its_short_axis(self):
        rooms = [room("a", 0, 0, 4, 4), room("b", 4.2, 0, 8, 4)]
        edges, counts = door_connections(rooms, [door(1, 4.0, 1.5, 4.2, 2.4)])
        self.assertEqual({edges[0]["a"], edges[0]["b"]}, {"a", "b"})
        self.assertNotIn("crossed_on_the_other_axis", counts)

    def test_a_door_inside_one_room_connects_nothing(self):
        edges, counts = door_connections([room("a", 0, 0, 8, 8)], [door(1, 4.0, 4.0, 4.2, 4.9)])
        self.assertEqual(edges, [])

    def test_a_space_drawn_into_the_doorway_still_leads_to_the_room_past_it(self):
        # Space a is declared to the far face of the 0.3 m wall, so it holds the opening and both
        # rays meet it first. Room b begins where the wall ends.
        rooms = [room("a", 0, 0, 4.3, 4), room("b", 4.3, 0, 8, 4)]
        leaf = door(1, 4.13, 1.5, 4.17, 2.4)
        edges, counts = door_connections(rooms, [leaf], blocked=leaf["_poly"].buffer(.06))
        self.assertEqual({edges[0]["a"], edges[0]["b"]}, {"a", "b"})
        self.assertEqual(counts["room_reaching_into_the_doorway"], 1)
        # a room further off than the door's own depth is not what the door leads to
        rooms = [room("a", 0, 0, 5, 4), room("b", 5, 0, 8, 4)]
        edges, counts = door_connections(rooms, [leaf], blocked=leaf["_poly"].buffer(.06))
        self.assertEqual(edges, [])
        self.assertEqual(counts["door_within_one_room"], 1)


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

    def test_rooms_a_little_apart_still_share_their_frontage(self):
        # model_2 Level 4: the stair landing faces the conference hall across a gap of a few
        # centimetres with nothing built in it. Matching outlines exactly found no frontage at all.
        for gap in (1e-6, .05, .2, .34):
            rooms = [room("a", 0, 0, 4, 4), room("b", 4 + gap, 0, 8, 4)]
            edges, _ = open_passages(rooms, [])
            self.assertEqual(len(edges), 1, gap)
            self.assertAlmostEqual(edges[0]["width"], 4., places=1)
        # beyond the gap a room is somewhere else
        edges, _ = open_passages([room("a", 0, 0, 4, 4), room("b", 4.5, 0, 8, 4)], [])
        self.assertEqual(edges, [])

    def test_what_stands_between_two_rooms_closes_the_frontage_in_front_of_it(self):
        rooms = [room("a", 0, 0, 4, 4), room("b", 4.3, 0, 8, 4)]
        wall = box(4, 0, 4.3, 4)
        self.assertEqual(open_passages(rooms, [wall])[0], [])
        # a pane standing in the middle of the gap touches neither room and still closes it
        pane = box(4.14, 0, 4.16, 4)
        self.assertEqual(open_passages(rooms, [pane])[0], [])
        # a wall with a 1.2 m opening leaves that opening
        edges, _ = open_passages(rooms, [box(4, 0, 4.3, 1.4), box(4, 2.6, 4.3, 4)])
        self.assertEqual(len(edges), 1)
        self.assertAlmostEqual(edges[0]["width"], 1.2, delta=.15)

    def test_rooms_meeting_only_at_a_corner_share_no_frontage(self):
        rooms = [room("a", 0, 0, 4, 4), room("b", 4.2, 4.2, 8, 8)]
        self.assertEqual(open_passages(rooms, [])[0], [])


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
        # a file with no IfcBuildingStorey has no storey ids, so the index says which storey
        self.assertEqual((edges[0]["storey_index_a"], edges[0]["storey_index_b"]), (0, 1))

    def test_a_stair_assembled_from_flights_joins_the_storeys_its_flights_cannot(self):
        # The IfcStair carries no shape and each flight climbs half a level, so on its own no
        # flight reaches a pair of storeys. Composed through aggregation, the stair does.
        stair = {"step_id": 1, "ifc_class": "IfcStair"}
        lower = {"step_id": 2, "ifc_class": "IfcStairFlight", "bbox_min": [1, 1, 0],
                 "bbox_max": [3, 2, 1.6], "_poly": box(1, 1, 3, 2)}
        upper = {"step_id": 3, "ifc_class": "IfcStairFlight", "bbox_min": [1, 2, 1.6],
                 "bbox_max": [3, 3, 3.2], "_poly": box(1, 2, 3, 3)}
        relations = [{"type": "part_of", "source": 2, "target": 1},
                     {"type": "part_of", "source": 3, "target": 1}]
        storeys = [{"step_id": 10, "elevation_m": 0.}, {"step_id": 20, "elevation_m": 3.}]
        rooms = {0: [room("g", 0, 0, 4, 4)], 1: [room("u", 0, 0, 4, 4)]}
        edges, counts = vertical_connections(rooms, [stair, lower, upper], storeys, relations=relations)
        self.assertEqual(len(edges), 1)
        self.assertEqual({edges[0]["a"], edges[0]["b"]}, {"g", "u"})
        self.assertEqual(counts["part_of_another_stair"], 2)
        self.assertEqual(vertical_connections(rooms, [lower, upper], storeys)[0], [])

    def test_a_stair_built_from_treads_is_composed_as_well(self):
        # Some files aggregate a stair from plain building element parts, one per tread.
        stair = {"step_id": 1, "ifc_class": "IfcStair"}
        treads = [{"step_id": 100 + i, "ifc_class": "IfcBuildingElementPart",
                   "bbox_min": [1, 1, .2 * i], "bbox_max": [3, 3, .2 * i + .2],
                   "_poly": box(1, 1, 3, 3)} for i in range(16)]
        relations = [{"type": "part_of", "source": t["step_id"], "target": 1} for t in treads]
        storeys = [{"step_id": 10, "elevation_m": 0.}, {"step_id": 20, "elevation_m": 3.2}]
        rooms = {0: [room("g", 0, 0, 4, 4)], 1: [room("u", 0, 0, 4, 4)]}
        edges, _ = vertical_connections(rooms, [stair] + treads, storeys, relations=relations)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["kind"], "stair")

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

    def test_above_the_ground_floor_an_opening_onto_a_void_leads_nowhere(self):
        # The courtyard of an E-shaped building seen from an upper storey: open air, no floor.
        rooms = [room("a", 0, 0, 4, 4)]
        walls = self.walls_around(0, 0, 4, 4, gap=(1., 3.))
        outside = outside_region(walls)
        edges, counts = exterior_passages(rooms, walls, storey_id="S1", outside=outside, at_grade=False)
        self.assertEqual(edges, [])
        self.assertEqual(counts["opening_onto_a_void"], 1)
        # a slab beyond the opening, a loggia onto a terrace, is somewhere to stand
        terrace = box(-.5, 4.2, 4.5, 7)
        edges, _ = exterior_passages(rooms, walls, storey_id="S1", outside=outside, at_grade=False,
                                     standable=terrace)
        self.assertEqual(len(edges), 1)

    def test_above_the_ground_floor_a_door_onto_a_void_is_no_way_out(self):
        rooms = [room("a", 0, 0, 4, 4)]
        walls = self.walls_around(0, 0, 4, 4, gap=(1.5, 2.4))
        leaf = door(1, 1.5, 4., 2.4, 4.2)
        blocked = unary_union([w.buffer(.06) for w in walls + [leaf["_poly"]]])
        outside = outside_region(walls + [leaf["_poly"]])
        edges, counts = door_connections(rooms, [leaf], storey_id="S1", outside=outside,
                                         blocked=blocked, at_grade=False)
        self.assertEqual(edges, [])
        self.assertEqual(counts["door_onto_a_void"], 1)
        # onto a balcony slab it is a way out
        edges, _ = door_connections(rooms, [leaf], storey_id="S1", outside=outside, blocked=blocked,
                                    at_grade=False, standable=box(0, 4.2, 4, 6))
        self.assertEqual(edges[0]["type"], "exterior_door")
        self.assertIn("a floor", edges[0]["source_of_truth"])
        # at ground level open ground is enough
        edges, _ = door_connections(rooms, [leaf], storey_id="S1", outside=outside, blocked=blocked)
        self.assertEqual(edges[0]["type"], "exterior_door")

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


class Boundaries(unittest.TestCase):
    def test_a_wall_between_two_rooms_bounds_both_of_them(self):
        rooms = [room("a", 0, 0, 4, 4), room("b", 4.2, 0, 8, 4)]
        wall = {"step_id": 1, "ifc_class": "IfcWall", "_poly": box(4, 0, 4.2, 4)}
        boundaries, _ = room_boundaries(rooms, [wall])
        self.assertEqual({b["room"] for b in boundaries}, {"a", "b"})
        self.assertTrue(all(abs(b["length_m"] - 4.) < .35 for b in boundaries))

    def test_touching_a_corner_is_not_a_boundary(self):
        rooms = [room("a", 0, 0, 4, 4)]
        column = {"step_id": 2, "ifc_class": "IfcColumn", "_poly": box(4, 4, 4.2, 4.2)}
        self.assertEqual(room_boundaries(rooms, [column])[0], [])

    def test_a_room_walled_all_round_is_fully_bounded(self):
        rooms = [room("a", 0, 0, 4, 4)]
        walls = [{"step_id": i, "ifc_class": "IfcWall", "_poly": g} for i, g in enumerate(
            [box(-.2, -.2, 4.2, 0), box(-.2, 4, 4.2, 4.2), box(-.2, 0, 0, 4), box(4, 0, 4.2, 4)])]
        room_boundaries(rooms, walls)
        self.assertGreaterEqual(rooms[0]["_bounded_share"], .99)


if __name__ == "__main__":
    unittest.main()
