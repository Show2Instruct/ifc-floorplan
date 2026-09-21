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
FRONTAGE_TOLERANCE_M = .15   # how far a room outline may sit from the wall that bounds it
PASSAGE_GAP_M = .35     # how far apart two room outlines may sit and still share a boundary
MIN_PASSAGE_M = .6      # a shared opening narrower than this is a modelling seam, not a way through
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


def _first_clear_point_is_outside(centre, direction, outside, blocked, reach, *, start=.3, step=.15):
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
        if blocked is None or not blocked.contains(point):
            return outside.intersects(point.buffer(.02))
        distance += step
    return None


def door_connections(rooms, doors, *, reach=REACH_M, storey_id=None, outside=None, blocked=None):
    """The rooms on either side of each door.

    A door fills an opening cut through a wall, so its own footprint sits inside the wall. The
    rooms it joins are found by stepping out from the door's centre along the wall's normal,
    in both directions, until a room is met. Stepping rather than taking a single offset lets
    one rule cover a thin partition and a half-metre external wall.
    """
    edges, counts = [], Counter()
    for door in doors:
        poly = door.get("_poly")
        if poly is None or poly.is_empty:
            counts["no_geometry"] += 1
            continue
        along, normal, width = _axes(poly)
        centre = poly.centroid
        # Sampling only the centre line misses a door at a room's corner, or one whose leaf is
        # modelled off-centre in its opening, so three lines across the leaf are tried.
        found = []
        for sign in (1, -1):
            hit = None
            for offset in (0., .25, -.25):
                base = (centre.x + along[0] * width * offset, centre.y + along[1] * width * offset)
                step = .05
                while step <= reach and hit is None:
                    hit = _room_at(rooms, base[0] + sign * normal[0] * step,
                                   base[1] + sign * normal[1] * step)
                    step += .05
                if hit is not None:
                    break
            found.append(hit)
        a, b = found
        if a is not None and b is not None and a["id"] == b["id"]:
            counts["door_within_one_room"] += 1
            continue
        if a is not None and b is not None:
            edges.append({"a": a["id"], "b": b["id"], "type": "connected_by_door",
                          "door": door.get("global_id") or door.get("step_id"),
                          "width": round(width, 2), "kind": None,
                          "source_of_truth": "rooms either side of the door's wall normal"})
            counts["connected_by_door"] += 1
        elif a is not None or b is not None:
            room = a or b
            blank = -normal if b is None else normal   # the side that found no room
            if outside is not None:
                verdict = _first_clear_point_is_outside(centre, blank, outside, blocked, reach)
                if verdict is None:
                    counts["door_far_side_blocked"] += 1
                    continue
                if not verdict:
                    # not open ground: a door to a space we failed to find, not a way out
                    counts["door_to_a_space_we_did_not_find"] += 1
                    continue
            edges.append({"a": room["id"], "b": exterior_id(storey_id) if storey_id else None,
                          "exterior": True, "type": "exterior_door",
                          "door": door.get("global_id") or door.get("step_id"),
                          "width": round(width, 2), "kind": None,
                          "source_of_truth": "a room on one side of the door only"})
            counts["exterior_door"] += 1
        else:
            counts["door_between_no_rooms"] += 1
    return edges, dict(counts)


def open_passages(rooms, barriers, *, gap=PASSAGE_GAP_M, minimum=MIN_PASSAGE_M):
    """Rooms that meet along a stretch of boundary with nothing built across it.

        Two declared spaces can abut with no wall between them: an archway, or a boundary the file
        draws but never builds. Inferred rooms rarely meet this way, since free space with no barrier
        between it floods into one region, so this mostly concerns IfcSpace. The shared frontage is
        measured and the parts covered by a barrier are removed; what is left is the way through.
    """
    from shapely.ops import unary_union
    built = unary_union(barriers).buffer(.02) if barriers else None
    edges, counts = [], Counter()
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            a, b = rooms[i], rooms[j]
            if a["_poly"].distance(b["_poly"]) > gap:
                continue
            # Distance-based frontage counts a corner touch as an opening, because boundary
            # near a corner is close to the other room without facing it. Growing the other
            # room by exactly the gap makes the two boundaries meet along the stretch they
            # share and nowhere else; corners then contribute points, which have no length.
            separation = a["_poly"].distance(b["_poly"])
            other = b["_poly"] if separation < 1e-9 else b["_poly"].buffer(separation + 1e-9, join_style=2)
            frontage = a["_poly"].boundary.intersection(other.boundary)
            if frontage.is_empty:
                continue
            if built is not None:
                frontage = frontage.difference(built)
            width = frontage.length
            if width < minimum:
                counts["seam_too_narrow"] += 1
                continue
            edges.append({"a": a["id"], "b": b["id"], "type": "open_passage", "door": None,
                          "width": round(width, 2), "kind": None,
                          "source_of_truth": "shared room boundary with no barrier across it"})
            counts["open_passage"] += 1
    return edges, dict(counts)


def exterior_passages(rooms, barriers, *, storey_id, tolerance=FRONTAGE_TOLERANCE_M,
                      minimum=MIN_PASSAGE_M, outside=None, at_any_height=()):
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
        widest_length = measured
        edges.append({"a": room["id"], "b": exterior_id(storey_id), "type": "open_passage",
                      "door": None, "width": round(widest_length, 2), "kind": None, "exterior": True,
                      "source_of_truth": "room outline running along no barrier, with no room beyond"})
        counts["open_passage_to_outside"] += 1
    return edges, dict(counts)


def vertical_connections(storey_rooms, elements, storeys):
    """Rooms on different storeys joined by a stair, ramp or lift.

    The element's own height range says which storeys it serves; the room it stands in on each
    of those storeys is the room that is connected. Consecutive storeys are joined in order,
    so a stair running through four levels gives three edges rather than six.
    """
    edges, counts, seen = [], Counter(), set()
    for element in elements:
        if not element["ifc_class"].startswith(VERTICAL):
            continue
        lo, hi = element.get("bbox_min"), element.get("bbox_max")
        poly = element.get("_poly")
        if poly is None or lo is None or hi is None:
            continue
        served = []
        for index, storey in enumerate(storeys):
            elevation = storey["elevation_m"]
            if lo[2] - .6 <= elevation <= hi[2] + .6:
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
                          "storey_a": sa.get("step_id"), "storey_b": sb.get("step_id"),
                          "source_of_truth": f"{kind} standing in both rooms"})
            counts[f"vertically_connected_{kind}"] += 1
    return edges, dict(counts)
