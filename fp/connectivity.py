#!/usr/bin/env python
"""Which rooms connect to which.

An IFC file says which rooms exist. It does not say that a door in this wall joins the kitchen
to the hallway, and that is the edge a scene graph is built on. This module derives those edges
from geometry the pipeline already has, using the same vocabulary as the BIM-GNI plans so the
two can be compared directly:

  * `connected_by_door`   two rooms either side of a door
  * `exterior_door`       a door with a room on one side only
  * `open_passage`        two rooms sharing a stretch of boundary with no barrier across it
  * `vertically_connected` two rooms on different storeys joined by a stair, ramp or lift

Every edge records what produced it, so a wrong edge can be traced back to its geometry.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
from shapely.geometry import Point

REACH_M = 2.            # how far past a door to look for the room it opens into
SAME_WALL_M = .1        # solids nearer than this are one wall: a leaf, its reveal, its layers
FRONTAGE_TOLERANCE_M = .15   # how far a room outline may sit from the wall that bounds it
PASSAGE_GAP_M = .35     # how far apart two room outlines may sit and still share a boundary
MIN_PASSAGE_M = .6      # a shared opening narrower than this is a modelling seam, not a way through
FRONTAGE_SLACK_M = .02  # outline this much further from the other room than the gap still faces it
DOORWAY_M = .3          # how far past a door's centre the room it leads to must begin
VERTICAL = ("IfcStair", "IfcStairFlight", "IfcRamp", "IfcRampFlight", "IfcTransportElement")


def _axes(polygon):
    """The long and short axes of a footprint, as unit vectors, with the long side's length."""
    box = np.asarray(polygon.minimum_rotated_rectangle.exterior.coords)[:4]
    edges = np.roll(box, -1, axis=0) - box
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    long_i = int(np.argmax(lengths[:2])) if lengths[0] != lengths[1] else 0
    long_v = edges[long_i] / (lengths[long_i] or 1)
    return long_v, np.array([-long_v[1], long_v[0]]), float(lengths[long_i])


OUTSIDE_TOLERANCE_M = .06   # half the room-inference raster cell; closes hairline joints only
OUTDOOR_REACH_M = 1.        # how close an outdoor floor must be to a room to belong to it
RAILING_REACH_M = .3        # how close a railing must run to an outdoor floor to border it
MIN_OUTDOOR_M2 = 1.5


def outside_region(barriers, *, margin=2., tolerance=OUTSIDE_TOLERANCE_M):
    """The open ground around the building: everywhere reachable from beyond it.

        Needed to tell the outside from a room the pipeline missed. A door with no room on its far
        side is a way out only if that side is open ground. Asking whether a point lies inside the
        building's outline is a different question and gets a courtyard wrong, so the test is
        reachability: the barriers are cut out of a box drawn around them, and the piece touching the
        box edge is the outside.

        Barriers are thickened by a few centimetres first, the same margin room inference uses.
        Exact geometry leaves hairline joints between a door leaf and its wall that open ground
        floods through. A real doorway is an order of magnitude wider and stays open.
    """
    from shapely.geometry import box
    from shapely.ops import unary_union
    if not barriers:
        return None
    merged = unary_union([b.buffer(tolerance) for b in barriers] if tolerance > 0 else barriers)
    minx, miny, maxx, maxy = merged.bounds
    field = box(minx - margin, miny - margin, maxx + margin, maxy + margin)
    open_ground = field.difference(merged)
    parts = list(open_ground.geoms) if hasattr(open_ground, "geoms") else [open_ground]
    edge = field.boundary
    touching = [part for part in parts if part.distance(edge) < 1e-6]
    return unary_union(touching) if touching else None


def outdoor_spaces(slabs, railings, rooms, outside, *, storey_id, minimum=MIN_OUTDOOR_M2,
                   near_room=OUTDOOR_REACH_M, railing_reach=RAILING_REACH_M):
    """Balconies and terraces: floor outside the building that a railing borders.

        A file can carry a balcony without ever naming it: railings outside the walls and a slab
        running past them, with no IfcSpace. Nothing turns that into a space on its own, because the
        railings stand below the plan's cut and a railing does not enclose.

        So a balcony is the part of a storey's slab that open ground reaches, that a railing borders,
        and that lies next to a room. Requiring a room nearby keeps a roof with a safety railing from
        becoming a terrace. Returns (spaces, counts); each space keeps its outline and area.
    """
    from shapely.ops import unary_union
    from fp.extract import poly_coords, poly_geometry
    counts = Counter()
    if outside is None or not slabs:
        return [], dict(counts)
    open_floor = unary_union(slabs).intersection(outside)
    parts = list(open_floor.geoms) if hasattr(open_floor, "geoms") else [open_floor]
    enclosed = unary_union([r["_poly"] for r in rooms]) if rooms else None
    spaces = []
    for part in sorted((p for p in parts if p.geom_type == "Polygon"), key=lambda p: -p.area):
        if part.area < minimum:
            continue
        if enclosed is not None and part.intersection(enclosed).area > .2 * part.area:
            counts["floor_already_a_room"] += 1
            continue
        if enclosed is None or part.distance(enclosed) > near_room:
            counts["open_floor_away_from_rooms"] += 1
            continue
        if not any(r is not None and part.buffer(railing_reach).intersects(r) for r in railings):
            counts["open_floor_without_railing"] += 1
            continue
        k = len(spaces) + 1
        spaces.append({"id": f"OUT-{storey_id}-{k}", "step_id": None, "name": f"Outdoor space {k}",
                       "kind": "outdoor", "source": "slab_outside_walls_bordered_by_railing",
                       "area_m2": round(part.area, 2), "polygon": poly_coords(part),
                       "geometry": poly_geometry(part), "_poly": part})
        counts["outdoor_space"] += 1
    return spaces, dict(counts)


def terrace_ways_out(outdoor, railings, barriers, rooms, *, storey_id, reach=RAILING_REACH_M,
                     minimum=MIN_PASSAGE_M, step=.05):
    """Terraces at ground level that a person can step off onto the ground around them.

    A railing is there to stop people walking off, so a terrace railed all round is a dead end, as
    a balcony is. Where its outer edge has a stretch at least `minimum` long that no railing, wall
    or room borders, steps or a gate, it leads onto the ground. A ledge narrower than `minimum`,
    such as a slab running a few centimetres past the walls, is no way down.

    The edge is walked in short steps. A step is railed when a railing stands within `reach` across
    the edge from it; one merely past a railing's end is not, so a gate is measured at its width.
    """
    from shapely.geometry import LineString
    from shapely.ops import nearest_points, unary_union
    from shapely.prepared import prep
    edges, counts = [], Counter()
    fence = unary_union([r for r in railings if r is not None]) if railings else None
    # the outside region keeps a margin from the walls, so the terrace's edge along the building sits
    # a few centimetres off them
    walls = [b.buffer(FRONTAGE_TOLERANCE_M) for b in barriers if b is not None]
    walls += [room["_poly"].buffer(FRONTAGE_TOLERANCE_M) for room in rooms]
    walls = prep(unary_union(walls)) if walls else None
    for space in outdoor:
        body = space["_poly"].buffer(-minimum / 2, join_style=2).buffer(minimum / 2, join_style=2)
        if body.is_empty:
            counts["terrace_too_narrow_to_step_off"] += 1
            continue
        widest = 0.
        for ring in [g.exterior for g in getattr(body, "geoms", [body])]:
            coords = list(ring.coords)
            run, first_run, closed_before = 0., None, False
            for (x0, y0), (x1, y1) in zip(coords, coords[1:]):
                segment = LineString([(x0, y0), (x1, y1)])
                n = max(1, int(math.ceil(segment.length / step)))
                tx, ty = (x1 - x0) / segment.length, (y1 - y0) / segment.length
                for k in range(n):
                    here = segment.interpolate((k + .5) * segment.length / n)
                    closed = walls is not None and walls.contains(here)
                    if not closed and fence is not None:
                        there = nearest_points(fence, here)[0]
                        gap = here.distance(there)
                        if gap <= 1e-6:
                            closed = True
                        elif gap <= reach:
                            closed = abs(((there.x - here.x) * tx + (there.y - here.y) * ty) / gap) < .5
                    if closed:
                        if first_run is None:
                            first_run = run
                        widest, run = max(widest, run), 0.
                    else:
                        run += segment.length / n
            # the ring closes on itself, so the stretch before the first railed step continues the last
            widest = max(widest, run + (first_run or 0.) if first_run is not None else run)
        if widest < minimum:
            counts["terrace_railed_all_round"] += 1
            continue
        edges.append({"a": space["id"], "b": exterior_id(storey_id), "exterior": True, "type": "open_passage",
                      "door": None, "width": round(widest, 2), "kind": None,
                      "source_of_truth": "a gap in a ground-level terrace's railing, onto the ground"})
        counts["terrace_with_a_way_down"] += 1
    return edges, dict(counts)


def exterior_id(storey_id):
    """The outside, as a node.

        A door to the street and a terrace opening both lead somewhere, and a graph that drops them
        cannot answer how a building is entered.
    """
    return f"EXT-{storey_id}"


def _room_at(rooms, x, y):
    point = Point(x, y)
    for room in rooms:
        if room["_poly"].contains(point):
            return room
    return None


def _room_over(rooms, polygon):
    """The room a footprint mostly stands in. A stair between landings can have its centre
    outside every room, so area decides rather than a single point."""
    best, share = None, 0.
    for room in rooms:
        if not room["_poly"].intersects(polygon):
            continue
        overlap = room["_poly"].intersection(polygon).area
        if overlap > share:
            best, share = room, overlap
    return best if share > 0 else None


def _first_clear_point(centre, direction, blocked, reach, *, start=.3, step=.15, own=None):
    """The first point past a door that no barrier covers, nor the door's own hole, or None."""
    distance = start
    while distance <= reach:
        point = Point(centre.x + direction[0] * distance, centre.y + direction[1] * distance)
        if (blocked is None or not blocked.contains(point)) and (own is None or not own.contains(point)):
            return point
        distance += step
    return None


def _first_clear_point_is_outside(centre, direction, outside, blocked, reach, *, start=.3, step=.15, own=None):
    """Walk out from a door and let the first point clear of every barrier decide.

        A single point at a fixed distance is fragile: on one model ten doors had open ground at 0.3,
        0.6, 0.9 and 1.2 m, and only the point at 1.6 m landed inside a curtain-wall member beyond
        them, so each door was judged to lead nowhere. What matters is the space just past the door,
        so points inside a barrier are stepped over. Returns True, False, or None when nothing but
        barriers lies within reach.
    """
    distance = start
    while distance <= reach:
        point = Point(centre.x + direction[0] * distance, centre.y + direction[1] * distance)
        if (blocked is None or not blocked.contains(point)) and (own is None or not own.contains(point)):
            return outside.intersects(point.buffer(.02))
        distance += step
    return None


def _ray_obstacle_limit(base, direction, reach, blocked):
    """How far out the search may walk before a solid that is not the door's own wall.

    A door sits in a wall, so the first solid along the ray is that wall and has to be crossed.
    Any solid after it stops the search, which is what keeps a room behind a second wall from
    being read as the room behind this door. Solids nearer than SAME_WALL_M to each other are
    one wall, so a leaf set back in its reveal, or a wall modelled in layers, does not read as
    two. The whole ray is intersected, so a wall thinner than the sampling step cannot be
    stepped over.
    """
    if blocked is None or blocked.is_empty:
        return reach
    from shapely.geometry import LineString
    ray = LineString([base, (base[0] + direction[0] * reach, base[1] + direction[1] * reach)])
    pending, runs = [ray.intersection(blocked)], []
    while pending:
        part = pending.pop()
        if hasattr(part, 'geoms'):
            pending.extend(part.geoms)
        elif part.geom_type == 'LineString' and part.length > 1e-9:
            distances = [ray.project(Point(p)) for p in (part.coords[0], part.coords[-1])]
            runs.append((min(distances), max(distances)))
    walls = []
    for start, stop in sorted(runs):
        if walls and start <= walls[-1][1] + SAME_WALL_M:
            walls[-1][1] = max(walls[-1][1], stop)
        else:
            walls.append([start, stop])
    if walls and walls[0][0] <= SAME_WALL_M:
        walls.pop(0)
    return min(reach, walls[0][0]) if walls else reach


def _room_past_the_doorway(rooms, centre, across, room, door, blocked, reach):
    """The room a door leads to, when the room on its other side reaches into the doorway.

    A declared space drawn to the far face of the wall holds the opening itself, so both rays meet
    it first. The room beyond then begins within the door's own depth, on one side only. Found on
    both sides, or on neither, the door stays within one room.
    """
    corners = np.asarray(door.minimum_rotated_rectangle.exterior.coords)[:4] - (centre.x, centre.y)
    depth = corners @ across
    found = []
    for sign, half in ((1, depth.max()), (-1, -depth.min())):
        limit = min(_ray_obstacle_limit((centre.x, centre.y), sign * across, reach, blocked),
                    max(half + .15, DOORWAY_M))
        step, other = .05, None
        while step <= limit and other is None:
            hit = _room_at(rooms, centre.x + sign * across[0] * step, centre.y + sign * across[1] * step)
            if hit is not None and hit["id"] != room["id"]:
                other = hit
            step += .05
        if other is not None:
            found.append(other)
    return found[0] if len(found) == 1 else None


def door_connections(rooms, doors, *, reach=REACH_M, storey_id=None, outside=None, blocked=None,
                     at_grade=True, standable=None, noun="door"):
    """The rooms on either side of each door.

    A door fills an opening cut through a wall, so its own footprint sits inside the wall. The
    rooms it joins are found by stepping out from the door's centre along the wall's normal,
    in both directions, until a room is met. Stepping rather than taking a single offset lets
    one rule cover a thin partition and a half-metre external wall.

    An opening the file cuts through a wall and leaves unfilled is crossed the same way, with
    `noun="unfilled_opening"`: its edges are open passages that name the opening, and its counts
    are kept apart from the doors'.
    """
    edges, counts = [], Counter()
    is_door = noun == "door"
    field = "door" if is_door else "opening"
    renamed = {"crossed_on_the_other_axis": f"{noun}_crossed_on_the_other_axis",
               "no_geometry": f"{noun}_no_geometry",
               "room_reaching_into_the_doorway": f"room_reaching_into_the_{noun}"}

    def count(name):
        counts[name if is_door else renamed.get(name, name.replace("door", noun))] += 1

    what = "the door" if is_door else "an opening with nothing in it"
    for door in doors:
        poly = door.get("_poly")
        if poly is None or poly.is_empty:
            count("no_geometry")
            continue
        along, normal, width = _axes(poly)
        centre = poly.centroid

        def either_side(across, span, reach_from):
            """The rooms the door opens into, walking out both ways along `across`.

            Sampling only the centre line misses a door at a room's corner, or one whose leaf is
            modelled off-centre in its opening, so three lines across the leaf are tried.
            """
            sides = []
            for sign in (1, -1):
                hit = None
                for offset in (0., .25, -.25):
                    base = (centre.x + span[0] * reach_from * offset, centre.y + span[1] * reach_from * offset)
                    limit = _ray_obstacle_limit(base, sign * across, reach, blocked)
                    step = .05
                    while step <= limit and hit is None:
                        hit = _room_at(rooms, base[0] + sign * across[0] * step,
                                       base[1] + sign * across[1] * step)
                        step += .05
                    if hit is not None:
                        break
                sides.append(hit)
            return sides

        across = normal
        a, b = either_side(normal, along, width)
        if a is None and b is None or (a is not None and b is not None and a["id"] == b["id"]):
            # A leaf drawn open has a footprint about as deep as it is wide, so which side is the
            # door's width is a coin toss and the search can run along the wall instead of through
            # it. When that happens the two rays meet the same room, or none, and the other axis is
            # the one that crosses the opening.
            second = either_side(along, normal, width)
            if second[0] is not None and second[1] is not None and second[0]["id"] != second[1]["id"]:
                a, b, across = second[0], second[1], along
                count("crossed_on_the_other_axis")
        how = f"rooms either side of {what}, along its wall's normal"
        if a is not None and b is not None and a["id"] == b["id"]:
            beyond = _room_past_the_doorway(rooms, centre, across, a, poly, blocked, reach)
            if beyond is None:
                count("door_within_one_room")
                continue
            b = beyond
            how = "a room reaching into the doorway, and the room beginning past it"
            count("room_reaching_into_the_doorway")
        width = door.get("width_m", width)
        if a is not None and b is not None:
            edges.append({"a": a["id"], "b": b["id"], "type": "connected_by_door" if is_door else "open_passage",
                          field: door.get("global_id") or door.get("step_id"),
                          "width": round(width, 2), "kind": None,
                          "source_of_truth": how})
            count("connected_by_door")
        elif a is not None or b is not None:
            room = a or b
            blank = -across if b is None else across   # the side that found no room
            # the door's own hole in its wall is part of the door, neither room nor outside
            own = door.get("_hole")
            own = own.buffer(OUTSIDE_TOLERANCE_M) if own is not None else None
            if outside is not None:
                verdict = _first_clear_point_is_outside(centre, blank, outside, blocked, reach, own=own)
                if verdict is None:
                    count("door_far_side_blocked")
                    continue
                if not verdict:
                    # not open ground: a door to a space we failed to find, not a way out
                    count("door_to_a_space_we_did_not_find")
                    continue
            onto = "open ground"
            if not at_grade:
                # Above the ground floor, open air beyond a door is only a way out if there is
                # floor to stand on: a balcony, terrace, deck or walkway. Otherwise it is a void,
                # such as the courtyard of an E-shaped building seen from an upper storey.
                point = _first_clear_point(centre, blank, blocked, reach, own=own)
                if standable is None or point is None or not standable.intersects(point.buffer(.05)):
                    count("door_onto_a_void")
                    continue
                onto = "a floor"
            edges.append({"a": room["id"], "b": exterior_id(storey_id) if storey_id else None,
                          "exterior": True, "type": "exterior_door" if is_door else "open_passage",
                          field: door.get("global_id") or door.get("step_id"),
                          "width": round(width, 2), "kind": None,
                          "source_of_truth": f"a room on one side of {what}, and {onto} on the other"})
            count("exterior_door")
        else:
            count("door_between_no_rooms")
    return edges, dict(counts)


def _clear_frontage(frontage, other, built, minimum, *, step=.05):
    """How much of a frontage a person can cross to the other room, in runs at least `minimum` long.

    The frontage is walked in short steps, and from each step a line is drawn across the gap to the
    nearest point of the other room. Anything built on that line, touching either room or standing
    in the middle of the gap, closes that step. A gap between two columns narrower than `minimum`
    is a seam, so only runs that long count.
    """
    from shapely.geometry import LineString
    from shapely.ops import nearest_points
    total = 0.
    for line in getattr(frontage, "geoms", [frontage]):
        if line.geom_type != "LineString" or line.length <= 0:
            continue
        n = max(1, int(math.ceil(line.length / step)))
        seg, run = line.length / n, 0.
        for k in range(n):
            here = line.interpolate((k + .5) * seg)
            there = nearest_points(other, here)[0]
            path = LineString([here, there]) if here.distance(there) > 1e-6 else here.buffer(.02)
            if built is not None and built.intersects(path):
                total += run if run >= minimum - 1e-9 else 0.
                run = 0.
            else:
                run += seg
        total += run if run >= minimum - 1e-9 else 0.
    return total


def open_passages(rooms, barriers, *, gap=PASSAGE_GAP_M, minimum=MIN_PASSAGE_M):
    """Rooms that meet along a stretch of boundary with nothing built across it.

        Two declared spaces can abut with no wall between them: an archway, or a boundary the file
        draws but never builds. Inferred rooms rarely meet this way, since free space with no barrier
        between it floods into one region, so this mostly concerns IfcSpace. The shared frontage is
        measured and the parts covered by a barrier are removed; what is left is the way through.
    """
    from shapely.ops import unary_union
    from shapely.prepared import prep
    built = prep(unary_union(barriers)) if barriers else None
    edges, counts = [], Counter()
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            a, b = rooms[i], rooms[j]
            separation = a["_poly"].distance(b["_poly"])
            if separation > gap:
                continue
            # The frontage is the stretch of one room's outline that lies within the gap of the
            # other. Matching the two outlines exactly found it only where rooms touch, so rooms a
            # few centimetres apart had none. A corner touch still gives almost nothing: outline
            # within the gap of a corner is a short arc.
            reach = separation + FRONTAGE_SLACK_M
            frontage = a["_poly"].boundary.intersection(b["_poly"].buffer(reach, join_style=2))
            if frontage.is_empty:
                continue
            width = _clear_frontage(frontage, b["_poly"], built, minimum)
            if width < minimum:
                counts["seam_too_narrow"] += 1
                continue
            edges.append({"a": a["id"], "b": b["id"], "type": "open_passage", "door": None,
                          "width": round(width, 2), "kind": None,
                          "source_of_truth": "shared room boundary with no barrier across it"})
            counts["open_passage"] += 1
    return edges, dict(counts)


def exterior_passages(rooms, barriers, *, storey_id, tolerance=FRONTAGE_TOLERANCE_M,
                      minimum=MIN_PASSAGE_M, outside=None, at_any_height=(), at_grade=True,
                      standable=None):
    """Rooms open to the outside with nothing built across the opening.

        A terrace, a carport, a covered entrance: part of the room's outline runs along no wall and no
        other room lies beyond it. That stretch is the way out, and an ordinary interior room, covered
        by barriers all round, produces nothing.

        A room with no barrier anywhere near it is skipped rather than called wide open, because that
        means the storey's barriers are missing, not that the room is a field.
    """
    from shapely.ops import unary_union
    # A section shows only what stands at the cut. A wall below it, or one whose section could not
    # be rebuilt, leaves a room looking open to the sky: a bathroom came out with a 4.5 m opening
    # to the outside. An opening is real only where nothing is built at any height, so the
    # whole-element silhouettes veto the frontage too.
    solid = list(barriers) + [p for p in at_any_height if p is not None]
    exact = unary_union(solid) if solid else None
    built = exact.buffer(tolerance) if exact is not None else None
    edges, counts = [], Counter()
    for room in rooms:
        outline = room["_poly"].boundary
        if built is None or not built.intersects(outline):
            counts["no_barriers_near_the_room"] += 1
            continue
        free = outline.difference(built)
        # A neighbour across a wall is not the outside. The wall's own thickness decides how far
        # to reach for it, so a 0.2 m partition and a 0.5 m external wall are both covered.
        for other in rooms:
            if other is room:
                continue
            separation = room["_poly"].distance(other["_poly"])
            if separation < 1.:
                free = free.difference(other["_poly"].buffer(separation + tolerance))
            if free.is_empty:
                break
        if free.is_empty:
            continue
        parts = list(free.geoms) if hasattr(free, "geoms") else [free]
        # The tolerance decides which stretches count; the opening is then measured against the
        # barriers themselves, so the reported width is the gap rather than the gap less the
        # tolerance at each end.
        widest, measured = None, 0.
        for part in parts:
            near = outline.intersection(part.buffer(tolerance * 1.5))
            length = near.difference(exact).length if exact is not None else near.length
            if length > measured:
                widest, measured = part, length
        if measured < minimum:
            counts["frontage_too_narrow"] += 1
            continue
        if outside is not None and widest is not None:
            facing = widest.representative_point().buffer(tolerance * 4)
            if not outside.intersects(facing):
                counts["frontage_faces_a_space_we_did_not_find"] += 1
                continue
        if not at_grade:
            # the same rule as for doors: above the ground floor an opening leads somewhere only
            # onto a slab
            ahead = widest.buffer(.6).difference(room["_poly"])
            if outside is not None:
                ahead = ahead.intersection(outside)
            if standable is None or ahead.is_empty or ahead.intersection(standable).area < .5 * ahead.area:
                counts["opening_onto_a_void"] += 1
                continue
        widest_length = measured
        edges.append({"a": room["id"], "b": exterior_id(storey_id), "type": "open_passage",
                      "door": None, "width": round(widest_length, 2), "kind": None, "exterior": True,
                      "source_of_truth": "room outline running along no barrier, with no room beyond"})
        counts["open_passage_to_outside"] += 1
    return edges, dict(counts)


def _assembly_extent(element, by_id, children):
    """An element's geometry together with the parts it is made of.

    A stair usually carries no shape of its own. Its flights, landings or treads hold the
    geometry, and one flight spans half a level, so on its own it reaches neither of the storeys
    the stair serves. Composed through IFC aggregation, the stair covers its real height range.
    Returns (footprint, lowest z, highest z), or (None, None, None) without geometry.
    """
    from shapely.ops import unary_union
    polys, lows, highs = [], [], []
    stack, seen = [element], set()
    while stack:
        part = stack.pop()
        step_id = part.get("step_id")
        if step_id is not None:
            if step_id in seen:
                continue
            seen.add(step_id)
        if part.get("_poly") is not None and not part["_poly"].is_empty:
            polys.append(part["_poly"])
        low, high = part.get("bbox_min"), part.get("bbox_max")
        if low and high:
            lows.append(low[2])
            highs.append(high[2])
        for child in children.get(step_id, ()):
            if child in by_id:
                stack.append(by_id[child])
    if not polys or not lows:
        return None, None, None
    return unary_union(polys), min(lows), max(highs)


def vertical_connections(storey_rooms, elements, storeys, *, relations=()):
    """Rooms on different storeys joined by a stair, ramp or lift.

    The element's height range says which storeys it serves; the room it stands in on each of
    those storeys is the room that is connected. Consecutive storeys are joined in order, so a
    stair running through four levels gives three edges rather than six.

    Stairs are composed from their parts first. Where the file models a stair as an assembly,
    each part covers a fraction of the climb and reaches no pair of storeys by itself.
    """
    edges, counts, seen = [], Counter(), set()
    by_id = {e["step_id"]: e for e in elements if e.get("step_id") is not None}
    children, parent = {}, {}
    for relation in relations:
        if relation.get("type") == "part_of":
            children.setdefault(relation["target"], []).append(relation["source"])
            parent[relation["source"]] = relation["target"]

    def part_of_another(step_id):
        """A flight inside its stair is handled with the stair, not on its own."""
        walked = set()
        while step_id in parent and step_id not in walked:
            walked.add(step_id)
            step_id = parent[step_id]
            owner = by_id.get(step_id)
            if owner is not None and owner["ifc_class"].startswith(VERTICAL):
                return True
        return False

    for element in elements:
        if not element["ifc_class"].startswith(VERTICAL):
            continue
        if part_of_another(element.get("step_id")):
            counts["part_of_another_stair"] += 1
            continue
        poly, low_z, high_z = _assembly_extent(element, by_id, children)
        if poly is None:
            counts["no_geometry"] += 1
            continue
        served = []
        for index, storey in enumerate(storeys):
            elevation = storey["elevation_m"]
            if low_z - .6 <= elevation <= high_z + .6:
                room = _room_over(storey_rooms.get(index, []), poly)
                if room is not None:
                    served.append((index, room, storey))
        kind = "lift" if element["ifc_class"] == "IfcTransportElement" else (
            "ramp" if "Ramp" in element["ifc_class"] else "stair")
        for (ia, ra, sa), (ib, rb, sb) in zip(served, served[1:]):
            pair = tuple(sorted((ra["id"], rb["id"])))
            if ra["id"] == rb["id"] or pair in seen:
                continue                       # a stair of four flights is still one way up
            seen.add(pair)
            _, _, width = _axes(poly)
            edges.append({"a": ra["id"], "b": rb["id"], "type": "vertically_connected",
                          "door": None, "width": round(width, 2), "kind": kind,
                          "element": element.get("step_id"),
                          "storey_a": sa.get("step_id"), "storey_b": sb.get("step_id"),
                          "storey_index_a": ia, "storey_index_b": ib,
                          "source_of_truth": f"{kind} standing in both rooms"})
            counts[f"vertically_connected_{kind}"] += 1
    return edges, dict(counts)


BOUNDARY_REACH_M = .15      # how far an element may stand from a room's outline and still bound it
MIN_BOUNDARY_M = .3         # a shared stretch shorter than this is a corner touch, not a boundary


def room_boundaries(rooms, elements, *, reach=BOUNDARY_REACH_M, minimum=MIN_BOUNDARY_M):
    """Which elements bound each room, and along how much of its outline.

    The IFC equivalent is IfcRelSpaceBoundary, which few files carry. A wall stands between rooms,
    so the room its centre falls in says nothing about which rooms it bounds; what matters is how
    much of a room's outline runs along it. Returns (boundaries, counts), each boundary naming the
    room, the element and the length they share.
    """
    from shapely.strtree import STRtree
    shapes = [e["_poly"] for e in elements]
    if not shapes or not rooms:
        return [], {}
    tree = STRtree(shapes)
    boundaries, counts = [], Counter()
    for room in rooms:
        outline = room["_poly"].boundary
        perimeter = outline.length
        covered = []
        for index in tree.query(room["_poly"].buffer(reach)):
            element, shape = elements[int(index)], shapes[int(index)]
            if shape.distance(room["_poly"]) > reach:
                continue
            band = shape.buffer(reach)
            shared = outline.intersection(band).length
            if shared < minimum:
                continue
            covered.append(band)
            boundaries.append({"room": room["id"], "element": element["step_id"],
                               "ifc_class": element["ifc_class"], "length_m": round(shared, 2)})
            counts["boundaries"] += 1
        if perimeter > 0 and covered:
            from shapely.ops import unary_union
            share = outline.intersection(unary_union(covered)).length / perimeter
            room["_bounded_share"] = round(share, 3)
    return boundaries, dict(counts)
