"""A storey as parametric parts: the form a floor plan takes when it is to become a model again.

A footprint polygon says where a wall stands but not how to rebuild it. A BIM authoring tool builds
a wall from an axis, a thickness and a height, and places a door by its host wall, its offset along
that wall, its width and its sill. This module reads those parameters back from the extracted
geometry, so a plan can be turned into IFC, and records how well each reading fits so nothing is
approximated silently.

Walls are read from the rectangle that best encloses their footprint: the axis runs along the
middle of its long sides, and the short side is the thickness. A wall that is not straight fills
that rectangle poorly, and its `fill` says so. Openings are placed by projecting their footprint
onto the axis of the wall that hosts them in the IFC.
"""
from __future__ import annotations

import math

import numpy as np

from fp.projections import SAME_LEVEL_M

WALLS = ("IfcWall", "IfcCurtainWall")
OPENINGS = ("IfcDoor", "IfcWindow")
STRAIGHT_FILL = .95     # a footprint filling less of its rectangle than this is not a straight wall
ON_AXIS_M = .05         # an opening may sit this much further from its wall's axis than half the wall
THROUGH_M = .05         # a void reaching this near a wall's base and top takes its whole height
PIER_RATIO = 1.5        # no longer than this times its thickness, a footprint is a pier


def _area_of(shape):
    """The parts of a shape that have area, or None: an intersection can also yield lines and points."""
    from shapely.ops import unary_union
    if shape is None or shape.is_empty:
        return None
    parts = [g for g in getattr(shape, "geoms", [shape]) if g.geom_type in ("Polygon", "MultiPolygon") and g.area > 1e-6]
    return unary_union(parts) if parts else None


def _rounded(value, digits=3):
    return None if value is None else round(float(value), digits)


def wall_axis(footprint):
    """(start, end, thickness, fill) for a wall footprint, or None without area.

    `fill` is how much of the enclosing rectangle the footprint's convex hull covers. A cavity
    wall's two leaves leave a gap inside the rectangle but still span it, so the hull, not the
    solid area, says whether the wall is straight.
    """
    if footprint is None or footprint.is_empty or footprint.area <= 0:
        return None
    rect = footprint.minimum_rotated_rectangle
    if rect.geom_type != "Polygon" or rect.area <= 0:
        return None
    p0, p1, p2, p3 = (np.asarray(c, dtype=float) for c in list(rect.exterior.coords)[:4])
    if np.linalg.norm(p1 - p0) >= np.linalg.norm(p2 - p1):
        start, end, thickness = (p0 + p3) / 2, (p1 + p2) / 2, np.linalg.norm(p2 - p1)
    else:
        start, end, thickness = (p0 + p1) / 2, (p3 + p2) / 2, np.linalg.norm(p1 - p0)
    # a stable direction, so the same wall always gets the same axis
    if (end[0], end[1]) < (start[0], start[1]):
        start, end = end, start
    return start, end, float(thickness), float(footprint.convex_hull.area / rect.area)


def _composed(element, by_id, children):
    """A wall's footprint and height range, from its own shape or from the parts it is made of.

    Some files model a wall as an assembly: the IfcWall carries no shape and its parts, one per
    layer or panel, hold the geometry.
    """
    from shapely.ops import unary_union
    if element.get("_poly") is not None:
        return element["_poly"], element.get("bbox_min"), element.get("bbox_max"), 0
    parts = [by_id[c] for c in children.get(element["step_id"], ()) if c in by_id]
    shapes = [p["_poly"] for p in parts if p.get("_poly") is not None]
    if not shapes:
        return None, None, None, 0
    lows = [p["bbox_min"] for p in parts if p.get("bbox_min")]
    highs = [p["bbox_max"] for p in parts if p.get("bbox_max")]
    low = [min(v[k] for v in lows) for k in range(3)] if lows else None
    high = [max(v[k] for v in highs) for k in range(3)] if highs else None
    return unary_union(shapes), low, high, len(shapes)


def opening_on_wall(opening, axis):
    """(offset along the axis to the opening's centre, width along the axis, distance off it)."""
    start, end = axis[0], axis[1]
    length = float(np.linalg.norm(end - start))
    if length <= 0 or opening is None or opening.is_empty:
        return None
    unit = (end - start) / length
    along = [float(np.dot(np.asarray(c) - start, unit)) for c in _coords(opening)]
    centre = np.asarray(opening.centroid.coords[0])
    offset = centre - start
    across = abs(float(unit[0] * offset[1] - unit[1] * offset[0]))
    return (min(along) + max(along)) / 2, max(along) - min(along), across


def _coords(geometry):
    for part in getattr(geometry, "geoms", [geometry]):
        yield from part.exterior.coords


def _taken(element):
    """A void as the repeat check compares it: its footprint and the heights it spans."""
    low, high = element.get("bbox_min"), element.get("bbox_max")
    return element["_poly"], low[2] if low else None, high[2] if high else None


def repeats(element, taken):
    """Whether a void is an opening already found, cut again.

    Where walls join, a file cuts one doorway through each of them, so a void covering half of a
    door's hole, or of an archway already found, is that opening listed again. It has to be at the
    same height as well: a void right above a door on the floor below is an opening of its own.
    """
    _, low, high = _taken(element)
    shape = element["_poly"]
    return any(g.intersects(shape) and g.intersection(shape).area >= .5 * min(g.area, shape.area)
               and (bottom is None or low is None or (bottom < high and top > low))
               for g, bottom, top in taken)


def wall_extensions(elements, relations, storey_of=None):
    """Walls drawn through an opening at their end, whose void took that end away again.

    A void at the end of a wall that takes the wall's whole height leaves the wall's geometry
    ending where the opening begins, while the file still names that wall as the host: model_172
    cuts archways at the ends of its walls, and model_168 stands doors taller than the low walls
    holding them. Read from what is left, the opening lies past its wall's end and could not be
    rebuilt in it. The wall was drawn through the opening, so it is read back that way, and the
    void takes the same stretch away again. A void cut through another wall is that wall's
    opening, and one lower than its wall would leave a lintel the file does not have, so neither
    counts. A curtain wall standing in the void fills it rather than holding it: model_172 sets
    glazed screens in its end archways.

    Only voids storey_vector keeps as openings count, so a void repeating another (see `repeats`)
    extends nothing. `storey_of` gives each element's storey index, the storey whose vector layer
    places it; without it, every element is taken to be on one storey.

    Returns {wall step_id: (start, end, opening ids, turned)}: the stretch of the wall's axis
    covering the wall and those openings, measured from the start of the axis its geometry gives,
    or with `turned`, of that axis read along the pier's other side (see `_turned`).
    """
    from shapely.strtree import STRtree
    by_id = {e["step_id"]: e for e in elements}
    children = {}
    for relation in relations:
        if relation.get("type") == "part_of":
            children.setdefault(relation["target"], []).append(relation["source"])
    hosts = [r for r in relations if r.get("type") == "hosts"
             and str((by_id.get(r.get("target")) or {}).get("ifc_class", "")).startswith(OPENINGS)]
    filled = {r.get("via_opening") for r in relations if r.get("type") == "hosts"}
    filled_by = {r.get("via_opening"): (r.get("source"), r.get("target")) for r in hosts}
    hosting = {r.get("source") for r in hosts}
    walls = {}
    for element in elements:
        cls = element["ifc_class"]
        if cls.startswith(WALLS) or element["step_id"] in hosting and not cls.startswith(("IfcSlab", "IfcRoof")):
            footprint, low, high, _ = _composed(element, by_id, children)
            if footprint is not None and not footprint.is_empty and low and high:
                walls[element["step_id"]] = (footprint, low[2], high[2])
    ids = [i for i in walls if by_id[i]["ifc_class"].startswith("IfcWall")]
    tree = STRtree([walls[i][0] for i in ids]) if ids else None
    # the voids kept as openings, as storey_vector finds them: the holes doors and windows fill,
    # then each storey's archways in turn, less those repeating one already found
    taken = [_taken(by_id[o]) for o in filled if o in by_id and by_id[o].get("_poly") is not None]
    found = {}
    out = {}
    for relation in relations:
        if relation.get("type") != "voids" or relation.get("target") not in by_id:
            continue
        void, host = by_id[relation["target"]], relation.get("source")
        if void["step_id"] in filled:
            if filled_by.get(void["step_id"], (None,))[0] != host:
                continue                    # the door it holds is placed on another wall
        else:
            if storey_of is not None and void["step_id"] not in storey_of:
                continue
            storey = found.setdefault(storey_of.get(void["step_id"]) if storey_of else None, [])
            if void.get("_poly") is not None and repeats(void, taken + storey):
                continue
            host_element = by_id.get(host)
            if host_element is None or not (host_element["ifc_class"].startswith(WALLS) or host in hosting):
                continue
            if void.get("_poly") is not None:
                storey.append(_taken(void))
        if host not in walls or void.get("_poly") is None or not void.get("bbox_min") or not void.get("bbox_max"):
            continue
        footprint, base, top = walls[host]
        if void["bbox_min"][2] > base + THROUGH_M or void["bbox_max"][2] < top - THROUGH_M:
            continue
        reading = wall_axis(footprint)
        if reading is None:
            continue
        # a pier is about as thick as it is long, so the way its axis runs is a guess: model_77's
        # and model_159's 0.3 m piers stand at the ends of facade voids, read either way
        fit = None
        for turned in (False, True):
            start, end, thickness = reading[0], reading[1], reading[2]
            if turned:
                if np.linalg.norm(end - start) > PIER_RATIO * thickness or host in out and not out[host][3]:
                    break
                start, end, thickness = _turned(start, end, thickness)
            elif host in out and out[host][3]:
                continue
            placed = opening_on_wall(void["_poly"], (start, end))
            if placed is None or placed[2] > thickness / 2 + ON_AXIS_M:
                continue
            length = float(np.linalg.norm(end - start))
            low, high = placed[0] - placed[1] / 2, placed[0] + placed[1] / 2
            if abs(low - length) <= ON_AXIS_M and high > length + ON_AXIS_M:
                fit = (0., high, length, turned)
            elif abs(high) <= ON_AXIS_M and low < -ON_AXIS_M:
                fit = (low, length, length, turned)
            if fit:
                break
        if fit is None:
            continue
        # another wall standing in the void where it cuts this one is the wall it opens
        if tree is not None and any(
                other != host and walls[other][0].intersection(void["_poly"]).area > 1e-6
                and walls[other][1] < top - THROUGH_M and walls[other][2] > base + THROUGH_M
                for other in (ids[k] for k in tree.query(void["_poly"]))):
            continue
        reach_start, reach_end, length, turned = fit
        start, end, opened, _ = out.get(host, (0., length, [], turned))
        opened.append(f"opening_{filled_by.get(void['step_id'], (None, void['step_id']))[1]}")
        out[host] = (min(start, reach_start), max(end, reach_end), opened, turned)
    return out


def _turned(start, end, thickness):
    """A wall's axis read along its other side, through the same centre: (start, end, thickness)."""
    centre, length = (start + end) / 2, float(np.linalg.norm(end - start))
    across = np.array([start[1] - end[1], end[0] - start[0]]) / length * thickness / 2
    start, end = centre - across, centre + across
    if (end[0], end[1]) < (start[0], start[1]):
        start, end = end, start
    return start, end, length


def storey_vector(elements, relations, connections, elevation, next_level, by_id=None, extensions=None):
    """Walls, openings and slabs of one storey in parametric form, with the storey's height.

    `elements` are the storey's own elements with their footprint in `_poly`. `by_id` resolves a
    host wall that belongs to another storey. `next_level` is the elevation of the next physical
    level above, or None at the top of the building, where the height is read from the walls.
    `extensions` are the walls drawn through openings at their ends, from wall_extensions over the
    whole building, so a wall standing through two storeys is read the same way on both.
    """
    by_id = by_id or {}
    extensions = extensions or {}
    own = {e["step_id"]: e for e in elements}
    children = {}
    for relation in relations:
        if relation.get("type") == "part_of":
            children.setdefault(relation["target"], []).append(relation["source"])
    walls, axes, readings, counts = [], {}, {}, {"walls": 0, "walls_not_straight": 0, "walls_composed_from_parts": 0,
                                                 "openings": 0, "openings_without_a_host_axis": 0}

    def read_wall(element):
        footprint, low, high, parts = _composed(element, by_id, children)
        reading = wall_axis(footprint)
        if reading is None:
            return None
        start, end, thickness, fill = reading
        body = None
        if element["step_id"] in extensions:
            # the axis runs over the openings at its ends; the wall's own geometry stands on `body`
            a, b, _, turned = extensions[element["step_id"]]
            if turned:
                start, end, thickness = _turned(start, end, thickness)
            length = float(np.linalg.norm(end - start))
            unit = (end - start) / length
            start, end, body = start + unit * a, start + unit * b, (0. - a, length - a)
        return start, end, thickness, fill, footprint, low, high, parts, body

    # whatever holds a door or window is read as a wall: a curtain panel standing in for one, say
    hosting = {r.get("source") for r in relations if r.get("type") == "hosts"
               and str((own.get(r.get("target")) or by_id.get(r.get("target")) or {}).get("ifc_class", "")).startswith(OPENINGS)}
    for element in elements:
        cls = element["ifc_class"]
        if not (cls.startswith(WALLS) or element["step_id"] in hosting and not cls.startswith(("IfcSlab", "IfcRoof"))):
            continue
        reading = read_wall(element)
        if reading is None:
            continue
        start, end, thickness, fill, footprint, low, high, parts, body = reading
        axes[element["step_id"]] = (start, end)
        readings[element["step_id"]] = reading
        counts["walls"] += 1
        counts["walls_composed_from_parts"] += parts > 0
        straight = fill >= STRAIGHT_FILL
        counts["walls_not_straight"] += not straight
        rect_area = (body[1] - body[0] if body else np.linalg.norm(end - start)) * thickness
        walls.append({"id": f"wall_{element['step_id']}", "step_id": element["step_id"],
                      "global_id": element.get("global_id"), "ifc_class": element["ifc_class"],
                      "axis": [[_rounded(start[0]), _rounded(start[1])], [_rounded(end[0]), _rounded(end[1])]],
                      "length_m": _rounded(np.linalg.norm(end - start)), "thickness_m": _rounded(thickness),
                      "base_m": _rounded(low[2] - elevation) if low else None,
                      "height_m": _rounded(high[2] - low[2]) if low and high else None,
                      "fill": _rounded(fill), "straight": straight,
                      # below 1 where the wall has a cavity or its leaves are modelled apart
                      "solid_fill": _rounded(footprint.area / rect_area) if rect_area > 0 else None,
                      "composed_from_parts": parts})
        if body:
            walls[-1].update(body_m=[_rounded(body[0]), _rounded(body[1])],
                             extended_over=sorted(extensions[element["step_id"]][2]))
            counts["walls_extended_over_end_openings"] = counts.get("walls_extended_over_end_openings", 0) + 1

    if next_level is not None:
        height, height_from = _rounded(next_level - elevation), "next_level"
    else:
        # nothing stands above the top storey, so the walls standing on its floor say how high it is
        tops = sorted(w["base_m"] + w["height_m"] for w in walls
                      if w["base_m"] is not None and w["height_m"] is not None and abs(w["base_m"]) < SAME_LEVEL_M)
        height, height_from = (_rounded(tops[len(tops) // 2]), "walls") if tops else (None, None)

    def reading_of(step_id):
        if step_id not in readings:
            element = own.get(step_id) or by_id.get(step_id)
            readings[step_id] = read_wall(element) if element is not None else None
        return readings[step_id]

    def fits(placed, reading):
        offset, _, off_axis = placed
        length = float(np.linalg.norm(reading[1] - reading[0]))
        return off_axis <= reading[2] / 2 + ON_AXIS_M and -ON_AXIS_M <= offset <= length + ON_AXIS_M

    def place(shape, host, opening_id=None):
        """(host, placement, how) for an opening's void or filling.

        The opening goes on the wall the file names, cut to where its void meets that wall, since
        a void is often extruded well past one face. Where two walls join, files credit an opening
        to the other wall, so when it does not lie in the named one it goes on the wall of this
        storey it does lie in, and `how` says so. That is an IfcWall: a curtain wall standing in a
        void fills it, and holds it only where no wall can. A wall drawn through the opening at its
        end keeps it, and so does a straight wall the void stands in a gap of, whatever stands in
        the void.
        """
        if shape is None:
            return host, None, "ifc"
        named = reading_of(host)
        named_area = named[4].intersection(shape).area if named is not None else 0.
        if named is not None:
            placed = opening_on_wall(shape, (named[0], named[1]))
            if opening_id in extensions.get(host, (0, 0, ()))[2]:
                return host, placed, "ifc"
            # A void taking a straight wall's whole height leaves a gap in it, and standing in that
            # gap on the wall's line it is that wall's opening: model_59 sets its facades, curtain
            # walls, in voids as long as the concrete walls they interrupt.
            if placed is not None and named[3] >= STRAIGHT_FILL and named_area <= 1e-6 and fits(placed, named):
                return host, placed, "ifc"
        candidates = sorted(((reading[4].intersection(shape).area, step_id) for step_id, reading in readings.items()
                             if step_id in axes and step_id != host and reading[4].intersects(shape)), reverse=True)
        candidates = [(area, step_id) for area, step_id in candidates if area > 1e-6]
        walls_of = lambda step_id: str((own.get(step_id) or by_id.get(step_id) or {}).get("ifc_class", "")).startswith("IfcWall")
        overlapping = [c for c in candidates if walls_of(c[1])]
        # a curtain wall the void lies in is the last place for it, where the wall named cannot hold it
        infill = [(step_id, "geometry") for _, step_id in candidates if not walls_of(step_id)]
        order = [(host, "ifc")] + [(step_id, "geometry") for _, step_id in overlapping]
        if overlapping and named_area < .5 * overlapping[0][0]:
            # a void at a join clips the corner of the wall named; the wall it overlaps holds it
            order = [(overlapping[0][1], "geometry"), (host, "ifc")] + [(step_id, "geometry") for _, step_id in overlapping[1:]]
        first = None
        for candidate, how in order + infill:
            reading = reading_of(candidate)
            if reading is None:
                continue
            hole = _area_of(shape.intersection(reading[4])) or _area_of(shape.intersection(reading[4].minimum_rotated_rectangle))
            placed = opening_on_wall(hole if hole is not None else shape, (reading[0], reading[1]))
            if placed is None:
                continue
            first = first or (candidate, placed, how)
            if fits(placed, reading):
                return candidate, placed, how
        return first or (host, None, "ifc")

    rooms_of = {}
    for link in connections:
        through = link.get("door") if link.get("door") is not None else link.get("opening")
        if through is not None:
            rooms_of.setdefault(through, []).append([link.get("a"), link.get("b")])

    openings = []
    for relation in relations:
        if relation.get("type") != "hosts" or relation.get("target") not in own:
            continue
        opening = own[relation["target"]]
        if not opening["ifc_class"].startswith(OPENINGS):
            continue
        counts["openings"] += 1
        stated = relation.get("source")
        # The void the IFC cuts for the opening is the hole itself. The filling element can be
        # wider, a leaf drawn open for instance, so the void is used wherever it has geometry.
        void = by_id.get(relation.get("via_opening"))
        shape, placed_from = (void["_poly"], "opening_void") if void is not None and void.get("_poly") is not None \
            else (opening.get("_poly"), "filling_element")
        host, placed, how = place(shape, stated, f"opening_{opening['step_id']}")
        if placed is None:
            counts["openings_without_a_host_axis"] += 1
        counts[f"placed_from_{placed_from}"] = counts.get(f"placed_from_{placed_from}", 0) + 1
        low, high = opening.get("bbox_min"), opening.get("bbox_max")
        key = opening.get("global_id") or opening.get("step_id")
        openings.append({
            "id": f"opening_{opening['step_id']}", "step_id": opening["step_id"],
            "global_id": opening.get("global_id"),
            "kind": "door" if opening["ifc_class"].startswith("IfcDoor") else "window",
            "host_wall": f"wall_{host}" if host is not None else None,
            "offset_m": _rounded(placed[0]) if placed else None,
            "width_m": _rounded(placed[1]) if placed else None,
            "off_axis_m": _rounded(placed[2]) if placed else None,
            "placed_from": placed_from if placed else None,
            "sill_m": _rounded(low[2] - elevation) if low else None,
            "head_m": _rounded(high[2] - elevation) if high else None,
            "rooms": rooms_of.get(key, [])})
        if how == "geometry":
            openings[-1].update(host_from="geometry", host_stated=f"wall_{stated}")
            counts["openings_hosted_by_geometry"] = counts.get("openings_hosted_by_geometry", 0) + 1

    # An opening the file cuts through a wall and leaves unfilled, an archway or a hatch, is cut
    # into the rebuilt wall as well; it has no door or window to place, only the void.
    filled = {r.get("via_opening") for r in relations if r.get("type") == "hosts"}
    taken = [_taken(e) for e in (own.get(o) or by_id.get(o) for o in filled) if e is not None and e.get("_poly") is not None]
    for relation in relations:
        if relation.get("type") != "voids" or relation.get("target") not in own or relation["target"] in filled:
            continue
        if own[relation["target"]].get("_poly") is not None and repeats(own[relation["target"]], taken):
            counts["voids_repeating_another_opening"] = counts.get("voids_repeating_another_opening", 0) + 1
            continue
        void, stated = own[relation["target"]], relation.get("source")
        host_element = own.get(stated) or by_id.get(stated)
        if host_element is None or not (host_element["ifc_class"].startswith(WALLS) or stated in hosting):
            continue                      # a hole in a slab or a beam is no opening in a wall
        counts["unfilled_openings"] = counts.get("unfilled_openings", 0) + 1
        host, placed, how = place(void.get("_poly"), stated, f"opening_{void['step_id']}")
        if void.get("_poly") is not None:
            taken.append(_taken(void))
        if placed is None:
            counts["openings_without_a_host_axis"] += 1
        low, high = void.get("bbox_min"), void.get("bbox_max")
        openings.append({
            "id": f"opening_{void['step_id']}", "step_id": void["step_id"],
            "global_id": void.get("global_id"), "kind": "opening",
            "host_wall": f"wall_{host}",
            "offset_m": _rounded(placed[0]) if placed else None,
            "width_m": _rounded(placed[1]) if placed else None,
            "off_axis_m": _rounded(placed[2]) if placed else None,
            "placed_from": "opening_void" if placed else None,
            "sill_m": _rounded(low[2] - elevation) if low else None,
            "head_m": _rounded(high[2] - elevation) if high else None,
            "rooms": rooms_of.get(void.get("global_id") or void["step_id"], [])})
        if how == "geometry":
            openings[-1].update(host_from="geometry", host_stated=f"wall_{stated}")
            counts["openings_hosted_by_geometry"] = counts.get("openings_hosted_by_geometry", 0) + 1

    slabs = []
    for element in elements:
        if not element["ifc_class"].startswith("IfcSlab") or element.get("_poly") is None:
            continue
        low, high = element.get("bbox_min"), element.get("bbox_max")
        slabs.append({"id": f"slab_{element['step_id']}", "step_id": element["step_id"],
                      "global_id": element.get("global_id"),
                      "top_m": _rounded(high[2] - elevation) if high else None,
                      "thickness_m": _rounded(high[2] - low[2]) if low and high else None,
                      "area_m2": _rounded(element["_poly"].area, 2)})
    return {"height_m": height, "height_from": height_from, "walls": walls, "openings": openings,
            "slabs": slabs, "counts": counts}


def rebuilt_wall(wall, body=False):
    """The footprint a wall's parameters describe, for checking them against the original.

    With `body`, only the stretch the wall's own geometry stands on: a wall drawn through openings
    at its ends runs over them, and their voids take that stretch away again.
    """
    from shapely.geometry import Polygon
    (x0, y0), (x1, y1) = wall["axis"]
    length = math.hypot(x1 - x0, y1 - y0)
    if length <= 0:
        return None
    if body and wall.get("body_m"):
        (a, b), ux, uy = wall["body_m"], (x1 - x0) / length, (y1 - y0) / length
        (x0, y0), (x1, y1), length = (x0 + ux * a, y0 + uy * a), (x0 + ux * b, y0 + uy * b), b - a
    nx, ny = -(y1 - y0) / length * wall["thickness_m"] / 2, (x1 - x0) / length * wall["thickness_m"] / 2
    return Polygon([(x0 + nx, y0 + ny), (x1 + nx, y1 + ny), (x1 - nx, y1 - ny), (x0 - nx, y0 - ny)])
