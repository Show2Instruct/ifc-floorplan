#!/usr/bin/env python
"""Assemble per-storey floor plans and a bird's-eye view from an IFC model.

Takes what fp.extract.extract() reads and writes, per model:

    <out>/<model>/
        building.json     the whole model: storeys, rooms, elements, relations, connections
        storey_00.json    one floor plan per storey
        storey_00.svg     the same floor plan drawn
        bev.svg           all storeys stacked, seen from above

The JSON is what another program consumes and the SVG is there so a person can check that the
JSON describes the building it claims to. Both come from the same geometry, so a visual error
and a data error are the same error. The hierarchy is storey to room to element, with each node
carrying its centre, dimensions and orientation.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union
from shapely.strtree import STRtree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fp.extract import extract, footprint, poly_coords, poly_geometry, infer_rooms, NON_FABRIC
from fp.projections import (barrier_polygons, barrier_roles, full_footprints, level_clusters,
                            section_elements, section_quality, BARRIERS, SAME_LEVEL_M)
from fp.vectorize import storey_vector, wall_axis, wall_extensions
from fp.connectivity import (MIN_PASSAGE_M, OUTSIDE_TOLERANCE_M, door_connections, exterior_passages,
                             open_passages, outdoor_spaces, outside_region, room_boundaries, terrace_ways_out,
                             vertical_connections)

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
IFC_DIR = os.path.join(ROOT, "data", "gni-bim", "2025_BIMfundamentals")
OUT_DIR = os.path.join(ROOT, "results", "floorplans")

# plan colours by IFC class: structure dark, openings saturated, contents pale
STYLE = {
    "IfcWall": ("#3A3733", "#3A3733", 0.0),
    "IfcWallStandardCase": ("#3A3733", "#3A3733", 0.0),
    "IfcCurtainWall": ("#5B7FA6", "#3F5C7C", 0.5),
    "IfcColumn": ("#3A3733", "#3A3733", 0.0),
    "IfcBeam": ("#8A8578", "#6E6A5F", 0.4),
    "IfcSlab": ("#EDEAE3", "#CFCabf", 0.5),
    "IfcRoof": ("#E4DFD4", "#C6C0B2", 0.5),
    "IfcDoor": ("#C2703A", "#8E4E24", 0.4),
    "IfcWindow": ("#4E86A8", "#2F6182", 0.4),
    "IfcStair": ("#9A9384", "#6E6A5F", 0.4),
    "IfcStairFlight": ("#9A9384", "#6E6A5F", 0.4),
    "IfcRailing": ("#B0A99A", "#8A8578", 0.3),
    "IfcFurniture": ("#CFC7B4", "#A9A090", 0.3),
    "IfcFlowTerminal": ("#9FB3A6", "#75897C", 0.3),
    "IfcPlate": ("#B7C6D2", "#8FA3B3", 0.3),
    "IfcMember": ("#A8A296", "#847E72", 0.3),
}
DEFAULT_STYLE = ("#C9C3B6", "#9A9384", 0.3)
DRAW_ORDER = ["IfcSlab", "IfcRoof", "IfcFurniture", "IfcFlowTerminal", "IfcPlate",
              "IfcMember", "IfcBeam", "IfcStair", "IfcStairFlight", "IfcRailing",
              "IfcCurtainWall", "IfcWall", "IfcWallStandardCase", "IfcColumn",
              "IfcWindow", "IfcDoor"]


def storey_bands(storeys, zmin, zmax):
    """Half-open [lo, hi) height band per storey, so every element lands in exactly one."""
    out = []
    for i, s in enumerate(storeys):
        lo = s["elevation_m"]
        hi = storeys[i + 1]["elevation_m"] if i + 1 < len(storeys) else zmax + 1.0
        out.append((lo - 0.35, hi - 0.35))       # a little below floor level catches slabs
    if out:
        out[0] = (min(zmin - 1.0, out[0][0]), out[0][1])
    return out


def assign_storeys(data):
    """Which storey each element belongs to.

        The storey the file states is used where present, labelled with the route extract.py found it
        by. Elements the file places on no storey are placed by the height of their base. The method
        is recorded per element so it can be checked later.
    """
    storeys = data["storeys"]
    if not storeys:
        return {}
    sid_index = {s["step_id"]: i for i, s in enumerate(storeys) if s["step_id"] is not None}
    zs = [e["bbox_min"][2] for e in data["elements"] if e.get("bbox_min")]
    bands = storey_bands(storeys, min(zs) if zs else 0.0, max(zs) if zs else 0.0)

    out = {}
    for e in data["elements"]:
        idx, how = None, None
        if e.get("storey_step_id") in sid_index:
            idx, how = sid_index[e["storey_step_id"]], e.get("storey_link") or "stated"
        elif e.get("bbox_min"):
            z = e["bbox_min"][2]
            for i, (lo, hi) in enumerate(bands):
                if lo <= z < hi:
                    idx, how = i, "inferred_from_elevation"
                    break
            if idx is None:
                idx, how = (0 if z < bands[0][0] else len(storeys) - 1), "inferred_clamped"
        if idx is not None:
            out[e["step_id"]] = (idx, how)
    return out


FLOOR_WALLS_M = 1.5      # wall bases nearer than this to one another stand on one floor
MIN_FLOOR_WALLS = 10     # fewer walls than this at one height is not a floor
MIN_FLOOR_SPACES = 1     # unless the file declares a space standing there
UNDECLARED_M = 2.0       # a floor this far from every storey the file declares is missing from it


def add_undeclared_levels(data):
    """Storeys for floors that hold walls but that the file never declares.

        Some files contain a whole building but declare one or two storeys, so every floor's walls
        and spaces sit on a storey metres above or below them. The plan is cut at the declared
        elevation and sees one floor, while the spaces of every floor are stacked on that storey.
        Wall bases show where the floors are: a cluster of them far from every declared storey is a
        floor of its own. Each gets a storey, and the elements standing on it move there from the
        storey the file named, with the move recorded on the element. Returns the added levels.
    """
    storeys = data["storeys"]
    if not storeys:
        return []
    # walls show where floors are, and so do the spaces the file declares on them, which matters
    # where a floor is enclosed by something other than IfcWall
    bases = sorted([(e["bbox_min"][2], "wall") for e in data["elements"]
                    if e["ifc_class"].startswith(("IfcWall", "IfcCurtainWall")) and e.get("bbox_min")]
                   + [(e["bbox_min"][2], "space") for e in data["elements"]
                      if e["ifc_class"] == "IfcSpace" and e.get("bbox_min")])
    groups = []
    for z, kind in bases:
        if groups and z - groups[-1][-1][0] <= FLOOR_WALLS_M:
            groups[-1].append((z, kind))
        else:
            groups.append([(z, kind)])
    floors = []
    for group in groups:
        walls = sum(1 for _, kind in group if kind == "wall")
        spaces = len(group) - walls
        if walls >= MIN_FLOOR_WALLS or spaces >= MIN_FLOOR_SPACES:
            floors.append((round(float(np.median([z for z, _ in group])), 3), walls))
    if data.get("storey_source") == "inferred_from_geometry":
        # The file declares no storeys at all, and the fallback clusters every element's base, so a
        # slab or a beam between floors pulls a level away from its floor. Where walls show the
        # floors, they are the storeys; everything is then placed by the height of its base.
        if floors:
            data["storeys"] = [{"step_id": None, "global_id": None, "name": f"Level {k} ({level:+.2f} m)",
                                "elevation_m": level, "source": "wall_bases", "walls": walls}
                               for k, (level, walls) in enumerate(floors)]
            data["storey_source"] = "wall_bases"
            return data["storeys"]
        return []
    declared = [s["elevation_m"] for s in storeys]
    added = []
    for level, walls in floors:
        if all(abs(level - z) > UNDECLARED_M for z in declared):
            added.append({"step_id": -(len(added) + 1), "global_id": None,
                          "name": f"Undeclared level {level:+.2f} m", "elevation_m": level,
                          "source": "wall_bases", "walls": walls})
    merged = sorted(storeys + added, key=lambda s: s["elevation_m"])
    zs = [e["bbox_min"][2] for e in data["elements"] if e.get("bbox_min")]
    if not zs:
        return added
    bands = storey_bands(merged, min(zs), max(zs))
    elevation_of = {s["step_id"]: s["elevation_m"] for s in storeys if s["step_id"] is not None}
    by_id = {e["step_id"]: e for e in data["elements"]}
    parts = {}
    for relation in data.get("relations") or []:
        if relation.get("type") == "part_of":
            parts.setdefault(relation["target"], []).append(relation["source"])
    for e in data["elements"]:
        # an element made of parts has no shape of its own and stands where they do: model_172's
        # facades are curtain walls of mullions and panels, one per floor, all named on one storey
        lows = [e["bbox_min"][2]] if e.get("bbox_min") else [
            by_id[p]["bbox_min"][2] for p in parts.get(e["step_id"], ()) if by_id.get(p, {}).get("bbox_min")]
        if not lows:
            continue
        base = min(lows)
        target = next((merged[i] for i, (lo, hi) in enumerate(bands) if lo <= base < hi), None)
        if target is None or target["step_id"] == e.get("storey_step_id"):
            continue
        stated = elevation_of.get(e.get("storey_step_id"))
        if stated is not None and abs(stated - base) <= UNDECLARED_M:
            continue                    # stands on the storey the file names, and reaches up
        if target.get("source") == "wall_bases":
            e["storey_stated"] = e.get("storey_step_id")
            e["storey_step_id"] = target["step_id"]
            e["storey_link"] = "moved to an undeclared level by the height of its base"
            target["moved_elements"] = target.get("moved_elements", 0) + 1
        elif e["ifc_class"] == "IfcSpace" and stated is not None:
            # A wall the file puts on the floor below can rise through this one, but a space is
            # the floor it stands on: model_6 puts a hall standing on its 4th storey on its 3rd.
            e["storey_stated"] = e.get("storey_step_id")
            e["storey_step_id"] = target["step_id"]
            e["storey_link"] = "moved to the storey its base stands on"
    data["storeys"] = merged
    return added


WALKABLE_SILL_M = .3     # an opening starting higher than a step above the floor is a hatch
WALKABLE_HEAD_M = 1.8    # and one lower than this cannot be walked through upright


def _polygonal(shape):
    """The area-bearing part of a shape, or None."""
    if shape is None or shape.is_empty:
        return None
    parts = [g for g in getattr(shape, "geoms", [shape]) if g.geom_type in ("Polygon", "MultiPolygon") and g.area > 1e-6]
    return unary_union(parts) if parts else None


def _outline(element, by_id, children):
    """An element's plan outline, or its parts' where the element itself carries no shape."""
    shape = element.get("_poly")
    if shape is None:
        parts = [by_id[c]["_poly"] for c in children.get(element["step_id"], ())
                 if c in by_id and by_id[c].get("_poly") is not None]
        shape = unary_union(parts) if parts else None
    return None if shape is None or shape.is_empty else shape


def _cut_strip(wall, cut):
    """The wall as thick as the cut finds it, along the whole length of its outline.

    The cut section stops where an opening begins, so at a wall's end it says nothing about the
    opening's stretch; its thickness holds for the whole wall all the same.
    """
    box_ = np.asarray(wall.minimum_rotated_rectangle.exterior.coords)[:4]
    first, second = box_[1] - box_[0], box_[2] - box_[1]
    along = first if np.linalg.norm(first) >= np.linalg.norm(second) else second
    if np.linalg.norm(along) <= 0:
        return None
    along = along / np.linalg.norm(along)
    normal = np.array([-along[1], along[0]])
    outline = np.asarray([c for g in getattr(wall, "geoms", [wall]) for c in g.exterior.coords])
    section = np.asarray([c for g in getattr(cut, "geoms", [cut]) if g.geom_type == "Polygon" for c in g.exterior.coords])
    if not len(section):
        return None
    a0, a1 = (outline @ along).min(), (outline @ along).max()
    n0, n1 = (section @ normal).min(), (section @ normal).max()
    return Polygon([a0 * along + n0 * normal, a1 * along + n0 * normal, a1 * along + n1 * normal, a0 * along + n1 * normal])


def _hole_in(opening, host, by_id, children, sections):
    """Where an opening meets a wall at the cut, or None."""
    wall = _outline(host, by_id, children)
    if wall is None:
        return None, None
    # The hole spans the wall as the cut finds it: a wall's whole outline can be thicker than the
    # wall at the cut, at a plinth say, and the hole would reach into the room beyond its face.
    cut = sections.get(host["step_id"])
    if cut is None:
        pieces = [sections[c] for c in children.get(host["step_id"], ()) if c in sections]
        cut = unary_union(pieces) if pieces else None
    strip = _cut_strip(wall, cut) if cut is not None and not cut.is_empty else None
    hole = _polygonal(opening["_poly"].intersection(wall))
    if hole is not None and strip is not None:
        hole = _polygonal(hole.intersection(strip))
    if hole is None:
        # an opening running the wall's full height splits its outline; its rectangle stands in
        hole = _polygonal(opening["_poly"].intersection(strip if strip is not None else wall.minimum_rotated_rectangle))
    return hole, wall


def opening_holes(data, roles, by_id, children, band, elevation, sections=None, barriers=None):
    """The holes that openings leave in their walls at this storey's cut, filled or not.

        A door or window fills its opening, but the cut through it can miss the doorway: a leaf
        drawn open covers its swing, not its frame. An opening left unfilled, an archway or a hatch,
        is a hole the cut cannot see at all. Either way the wall a room ends at runs across the
        hole, so room inference closes it. Only an unfilled opening a person can walk through, from
        the floor and at least door height, is a way between the rooms either side of it.

        Where two walls join, a file can credit an opening to the wrong one, so its void touches
        the wall it names only at an end. The opening is then cut into the barrier its void does
        overlap, from `barriers` (an STRtree and its elements), and the stated host is recorded.
    """
    if not band:
        return []
    low, high = band
    sections = sections or {}
    fills = {r.get("via_opening"): r["target"] for r in data.get("relations") or [] if r.get("type") == "hosts"}
    holes = []
    for relation in data.get("relations") or []:
        if relation.get("type") != "voids" or relation["source"] not in roles:
            continue
        opening, host = by_id.get(relation["target"]), by_id.get(relation["source"])
        if opening is None or host is None or opening.get("_poly") is None or not opening.get("bbox_min"):
            continue
        if opening["bbox_min"][2] > high or opening["bbox_max"][2] < low:
            continue
        # The opening belongs to the wall its void overlaps. The named wall keeps it unless another
        # overlaps it at least twice as much: a void at a join clips the corner of the wall named.
        stated_wall = _outline(host, by_id, children)
        stated_area = stated_wall.intersection(opening["_poly"]).area if stated_wall is not None else 0.
        others = []
        if barriers is not None:
            tree, candidates = barriers
            others = sorted(((candidates[int(k)]["_poly"].intersection(opening["_poly"]).area, candidates[int(k)])
                             for k in tree.query(opening["_poly"]) if candidates[int(k)]["step_id"] != host["step_id"]),
                            key=lambda pair: -pair[0])
            others = [(area, e) for area, e in others if area > 1e-6]
        order = [(host, "ifc")] + [(e, "geometry") for _, e in others]
        if others and stated_area < .5 * others[0][0]:
            order = [(others[0][1], "geometry"), (host, "ifc")] + [(e, "geometry") for _, e in others[1:]]
        hole = wall = None
        for candidate, host_from in order:
            hole, wall = _hole_in(opening, candidate, by_id, children, sections)
            if hole is not None:
                host = candidate
                break
        if hole is None:
            continue
        reading = wall_axis(wall)
        coords = np.asarray([c for g in getattr(hole, "geoms", [hole]) for c in g.exterior.coords])
        if reading is not None and np.linalg.norm(reading[1] - reading[0]) > 0:
            direction = (reading[1] - reading[0]) / np.linalg.norm(reading[1] - reading[0])
            along = coords @ direction
            width = float(along.max() - along.min())
        else:
            box_ = np.asarray(hole.minimum_rotated_rectangle.exterior.coords)
            width = float(max(np.linalg.norm(box_[1] - box_[0]), np.linalg.norm(box_[2] - box_[1])))
        sill, head = opening["bbox_min"][2] - elevation, opening["bbox_max"][2] - elevation
        filled_by = fills.get(relation["target"])
        record = {"step_id": opening["step_id"], "global_id": opening.get("global_id"),
                  "host": host["step_id"], "filled_by": filled_by,
                  "sill_m": round(sill, 3), "head_m": round(head, 3), "width_m": round(width, 3),
                  "walkable": filled_by is None and sill <= WALKABLE_SILL_M and head >= WALKABLE_HEAD_M
                              and width >= MIN_PASSAGE_M,
                  "_poly": hole}
        if host_from == "geometry":
            record.update(host_from="geometry", host_stated=relation["source"])
        holes.append(record)
    # Where walls join, a file can cut one doorway through each of them, so an unfilled void can be
    # the very hole a door fills, or one archway listed several times. Each counts once.
    def same(a, b, share):
        return a["_poly"].intersects(b["_poly"]) and \
            a["_poly"].intersection(b["_poly"]).area >= share * min(a["_poly"].area, b["_poly"].area)
    filled = [h for h in holes if h["filled_by"] is not None]
    unique = []
    for hole in holes:
        if hole["filled_by"] is not None:
            continue
        twin = next((f for f in filled if same(hole, f, .5)), None) or next((u for u in unique if same(hole, u, .9)), None)
        if twin is not None:
            hole.update(same_as=twin["step_id"], walkable=False)
        else:
            unique.append(hole)
    return holes


def class_counts(recs):
    """Element count per IFC class, largest first, ties broken by class name.

    Leaving ties to set iteration made the key order depend on Python's per-process
    string hashing, so two identical runs wrote different bytes."""
    return dict(sorted(Counter(r["ifc_class"] for r in recs).items(),
                       key=lambda kv: (-kv[1], kv[0])))


ROOM_DEFINING = ("IfcWall", "IfcCurtainWall", "IfcDoor", "IfcWindow")
# Classes that bound, open or carry rooms rather than stand in them.
NOT_CONTENTS = ("IfcWall", "IfcCurtainWall", "IfcDoor", "IfcWindow", "IfcPlate", "IfcOpeningElement",
                "IfcSlab", "IfcRoof", "IfcCovering", "IfcBeam", "IfcMember", "IfcBuildingElementPart",
                "IfcFooting", "IfcPile", "IfcSpace")
GRADE_TOLERANCE_M = 1.5    # how far from +-0.00 a ground floor may sit, for a raised one
FLOORS = ("IfcSlab", "IfcRoof")
STANDING_M = .3            # something whose base is this near a floor stands on it
CONTAINED_IN_SPACE = .8     # of an inferred region's own area, before it counts as that space
MIN_KEPT_AREA_M2 = 1.5      # a leftover smaller than this is modelling slack, not a room


REPEATED_IOU = .9           # two declared spaces this alike are one space listed twice


def _one_level(a, b):
    """Whether two rooms stand at one height: identical outlines on two floors are two rooms."""
    za, zb = a.get("_z"), b.get("_z")
    if not za or not zb:
        return True
    shared = min(za[1], zb[1]) - max(za[0], zb[0])
    return shared >= .5 * max(min(za[1] - za[0], zb[1] - zb[0]), 1e-6)
OVERLAP_M2 = .5             # declared spaces sharing more floor than this are made disjoint
WALLED_IOU = .7             # a declared space this close to a region the walls enclose is that room
ZONE_INSIDE = .8            # a space this much inside a walled room is a zone of it


def disjoint_declared(declared, *, regions=(), minimum=MIN_KEPT_AREA_M2):
    """Declared spaces that overlap, made disjoint, with every change recorded.

        Files list a space twice, put the spaces of floors they do not contain on one they do,
        declare a catch-all area over the rooms inside it, or a zone inside a room. Each leaves two
        rooms on the same floor, and an object standing there in both. A repeat is dropped. A space
        the file places on no storey, lying mostly over spaces it does place here, belongs elsewhere
        and is set aside. Where two spaces still overlap, the walls decide: `regions` are the
        regions the walls enclose, and the space matching one is the room. A smaller space inside a
        room the walls enclose is a zone of it and is set aside; a larger one spread over rooms the
        walls enclose is a catch-all, and they keep their floor. Without that evidence the smaller
        keeps the shared floor, as the more specific.

        Returns (spaces, set_aside).
    """
    kept, set_aside = [], []
    for space in sorted(declared, key=lambda d: (not d.get("_stated", True), d["step_id"])):
        shape = space["_poly"]
        repeat = next((k for k in kept if _one_level(k, space) and k["_poly"].intersects(shape)
                       and k["_poly"].intersection(shape).area >= REPEATED_IOU * k["_poly"].union(shape).area), None)
        if repeat is not None:
            set_aside.append({"id": space["id"], "name": space["name"], "reason": "repeats", "of": repeat["id"]})
            continue
        if not space.get("_stated", True):
            under = sum(k["_poly"].intersection(shape).area for k in kept
                        if k.get("_stated", True) and _one_level(k, space) and k["_poly"].intersects(shape))
            if under >= .5 * shape.area:
                set_aside.append({"id": space["id"], "name": space["name"],
                                  "reason": "placed on no storey, and lies over spaces placed on this one"})
                continue
        kept.append(space)

    def walled(space):
        """Whether the walls enclose this space: a region matches it closely."""
        shape = space["_poly"]
        return any(r.intersects(shape) and r.intersection(shape).area >= WALLED_IOU * r.union(shape).area
                   for r in regions)

    zones = []
    for small in sorted(kept, key=lambda d: d["_poly"].area):
        if walled(small):
            continue
        room = next((big for big in kept if big is not small and big["_poly"].area > small["_poly"].area
                     and _one_level(big, small) and walled(big)
                     and big["_poly"].intersection(small["_poly"]).area >= ZONE_INSIDE * small["_poly"].area), None)
        if room is not None and room not in zones:
            zones.append(small)
            set_aside.append({"id": small["id"], "name": small["name"], "reason": "a zone inside the room it lies in",
                              "in": room["id"]})
    kept = [k for k in kept if k not in zones]
    rank = {s["id"]: (s["_poly"].area, s["step_id"]) for s in kept}
    original = {s["id"]: s["_poly"] for s in kept}
    for space in kept:
        for other in kept:
            if other is space or rank[other["id"]] >= rank[space["id"]] or not _one_level(space, other):
                continue
            if not space["_poly"].intersects(original[other["id"]]):
                continue
            if space["_poly"].intersection(original[other["id"]]).area <= OVERLAP_M2:
                continue
            space["_poly"] = _polygonal(space["_poly"].difference(original[other["id"]])) or Polygon()
            space.setdefault("carved_by", []).append(other["id"])
    spaces = []
    for space in kept:
        if space.get("carved_by"):
            if space["_poly"].is_empty or space["_poly"].area < minimum:
                set_aside.append({"id": space["id"], "name": space["name"],
                                  "reason": "covered by the spaces it overlaps", "by": space["carved_by"]})
                continue
            space.update(area_m2=round(space["_poly"].area, 2), polygon=poly_coords(space["_poly"]),
                         geometry=poly_geometry(space["_poly"]))
        spaces.append(space)
    return spaces, set_aside


def disjoint_rooms(rooms, *, minimum=MIN_KEPT_AREA_M2):
    """The rooms of a storey, with any floor two of them still share given to one.

        Reconciliation keeps declared spaces and inferred regions apart, but two paths can still
        overlap: a part of a divided space can reach into the space next to it, and two inferred
        regions grown back to their walls can both claim a seam between them. A declared room keeps
        the shared floor over an inferred one, and otherwise the smaller, more specific room does.
        The other records what it gave way to. Returns (rooms, dropped ids).
    """
    rank = {r["id"]: (r["source"] != "IfcSpace", r["_poly"].area, str(r["id"])) for r in rooms}
    original = {r["id"]: r["_poly"] for r in rooms}
    for room in rooms:
        for other in rooms:
            if other is room or rank[other["id"]] >= rank[room["id"]] or not _one_level(room, other):
                continue
            if not room["_poly"].intersects(original[other["id"]]):
                continue
            if room["_poly"].intersection(original[other["id"]]).area <= OVERLAP_M2:
                continue
            room["_poly"] = _polygonal(room["_poly"].difference(original[other["id"]])) or Polygon()
            room.setdefault("trimmed_by_rooms", []).append(other["id"])
    kept, dropped = [], []
    for room in rooms:
        if room.get("trimmed_by_rooms"):
            if room["_poly"].is_empty or room["_poly"].area < minimum:
                dropped.append(room["id"])
                continue
            room.update(area_m2=round(room["_poly"].area, 2), polygon=poly_coords(room["_poly"]),
                        geometry=poly_geometry(room["_poly"]))
        kept.append(room)
    return kept, dropped


def _joined_through(parts, ways, reach=.1):
    """Parts of one declared space that a walkable opening joins, merged back into one piece.

    An archway closes the wall for room inference, so a space the file declares across one comes
    out as two regions. The file says it is one room, and an archway does not divide it.
    """
    group = list(range(len(parts)))

    def root(k):
        while group[k] != k:
            group[k] = group[group[k]]
            k = group[k]
        return k

    bridges = {}
    for way in ways:
        touching = [k for k, part in enumerate(parts) if part["_poly"].distance(way) <= reach]
        for k in touching[1:]:
            group[root(k)] = root(touching[0])
        if len(touching) > 1:
            bridges.setdefault(root(touching[0]), []).append(way)
    merged = {}
    for k, part in enumerate(parts):
        merged.setdefault(root(k), []).append(part)
    out = []
    for key, members in merged.items():
        if len(members) == 1:
            out.append(members[0])
            continue
        shape = unary_union([m["_poly"] for m in members] + bridges.get(key, []))
        whole = dict(members[0])
        whole.update(_poly=shape, area_m2=round(shape.area, 2), polygon=poly_coords(shape),
                     geometry=poly_geometry(shape))
        whole.pop("raster_area_m2", None)
        out.append(whole)
    return out


def reconcile_rooms(declared, inferred, *, contained=CONTAINED_IN_SPACE, minimum=MIN_KEPT_AREA_M2, ways=()):
    """Declared spaces and inferred regions on one storey, reconciled.

        A file that names some of its spaces but not all of them used to lose the rest, because
        inference only ran where no IfcSpace existed at all. On the models checked by hand that
        quietly dropped 83 real rooms.

        Running both raises the question of what to do where they overlap, and there are two answers:

          * one inferred region inside a declared space is the same room read twice, so the declared
            space is kept and the region absorbed;
          * several inferred regions inside one declared space mean walls divide it, so the parts
            become the rooms and each records the space it subdivides. Absorbing them would erase a
            wall that is really there.

        A region no declared space contains is a room the file never declared, and is kept with its
        own geometry and area. Containment is measured as a share of the inferred region's area, so a
        declared space bleeding a few centimetres over a wall centreline does not swallow the room
        next door.

        Where a region spans declared spaces instead of sitting inside one, a barrier is missing at
        the cut and free space has flooded across it. The declared spaces are what the file states, so
        they are cut out of the region and only the remainder is kept, as one room per separate piece
        above `minimum`. Keeping the region whole would leave two rooms covering the same floor, and a
        door stepping out of its wall would land in whichever of them came first.

        Returns (rooms, notes) with every room's origin recorded. Rooms do not overlap.
    """
    for room in declared:
        room['_parts'] = []
    kept, notes = [], Counter()
    for region in inferred:
        poly = region['_poly']
        area = poly.area
        if area <= 0:
            continue
        best, share = None, 0.
        for room in declared:
            overlap = poly.intersection(room['_poly']).area / area if poly.intersects(room['_poly']) else 0.
            if overlap > share:
                best, share = room, overlap
        if best is not None and share >= contained:
            best['_parts'].append(region)
            continue
        covered = [room['_poly'] for room in declared if poly.intersects(room['_poly'])]
        remainder = poly.difference(unary_union(covered)) if covered else poly
        pieces = [p for p in getattr(remainder, 'geoms', [remainder]) if p.area >= minimum]
        if not pieces:
            notes['dropped_covered_by_declared_spaces'] += 1
            continue
        pieces.sort(key=lambda p: -p.area)
        trimmed = sum(p.area for p in pieces) < area - 1e-9
        for n, piece in enumerate(pieces):
            part = dict(region)
            if len(pieces) > 1:
                part['id'] = f"{region['id']}_{n}"
                if region.get('name'):
                    part['name'] = f"{region['name']} part {n + 1}"
            if trimmed:
                part['trimmed_by_declared_spaces'] = True
                part.pop('raster_area_m2', None)
                part['_poly'] = piece
                part['area_m2'] = round(piece.area, 2)
                part['polygon'] = poly_coords(piece)
                part['geometry'] = poly_geometry(piece)
            part['overlap_with_declared_space'] = round(share, 3)
            kept.append(part)
            notes['kept_outside_declared_spaces'] += 1
        if trimmed:
            notes['trimmed_by_declared_spaces'] += 1
        if len(pieces) > 1:
            notes['split_by_declared_spaces'] += 1
    rooms = []
    for room in declared:
        parts = room.pop('_parts')
        if len(parts) > 1 and ways:
            parts = _joined_through(parts, ways)
        if len(parts) <= 1:
            rooms.append(room)
            notes['declared'] += 1
            notes['absorbed_inferred_duplicate'] += len(parts)
            continue
        # walls divide what the file called one space: the parts are the rooms
        for part in parts:
            part['subdivides_declared_space'] = room['step_id']
            part['declared_space_global_id'] = room.get('global_id')
            if 'storey_stated' in room:
                part.update(storey_assignment=room['storey_assignment'], storey_stated=room['storey_stated'])
            part['name'] = f"{room['name']} part {len(rooms) + 1}" if room.get('name') else part['name']
            rooms.append(part)
        notes['declared_subdivided'] += 1
        notes['subdivision_parts'] += len(parts)
    rooms.extend(kept)
    return rooms, dict(notes)


def section_adjacencies(visible, distance=.05):
    """Preserve the original strict distance rule without comparing every pair."""
    keyed = [(e['step_id'], e['_poly'], e['ifc_class']) for e in visible if e.get('_poly') is not None]
    if not keyed:
        return []
    polygons = [p for _, p, _ in keyed]
    candidates = STRtree(polygons).query(polygons, predicate='dwithin', distance=distance)
    pairs = sorted((int(a), int(b)) for a, b in zip(*candidates) if a < b)
    return [{'type': 'adjacent_to', 'source': keyed[a][0], 'target': keyed[b][0],
             'source_of_truth': 'section_footprint_proximity<5cm'}
            for a, b in pairs
            if keyed[a][2].startswith('IfcWall') == keyed[b][2].startswith('IfcWall')
            and polygons[a].distance(polygons[b]) < distance]


def orientation_deg(poly):
    """Dominant edge direction of a footprint, for LiteReality-style orientation."""
    try:
        r = poly.minimum_rotated_rectangle
        xs, ys = r.exterior.coords.xy
        edges = [((xs[i + 1] - xs[i]), (ys[i + 1] - ys[i])) for i in range(4)]
        dx, dy = max(edges, key=lambda e: e[0] ** 2 + e[1] ** 2)
        return round(float(np.degrees(np.arctan2(dy, dx))) % 180.0, 1)
    except Exception:
        return None


def build(model, *, ifc_path=None, out_dir=None, verbose=True, threads=4,
          cell=0.12, min_room_area=1.5, erode=None, provenance=None, cut_height=1.2,
          cut_thickness=.01, band_base=None, barrier_classes=BARRIERS):
    """Build one model's floor plans.

        ifc_path and out_dir default to the corpus layout, IFC_DIR/<model>.ifc and OUT_DIR/<model>/.
        run_pipeline.py passes both explicitly, along with the room-inference settings and a
        provenance block written into building.json, so the output names the code, input and settings
        behind it. erode=None keeps infer_rooms' own default.
    """
    out = out_dir or os.path.join(OUT_DIR, model)
    if os.path.exists(out) and os.listdir(out):
        raise ValueError(f"output already contains files; choose a new directory: {out}")
    data, f = extract(ifc_path or os.path.join(IFC_DIR, model + ".ifc"), threads=threads)
    undeclared = add_undeclared_levels(data)
    room_opts = {"cell": cell, "min_area": min_room_area}
    if erode is not None:
        room_opts["erode"] = erode
    assign = assign_storeys(data)
    storeys = data["storeys"]
    if not storeys:
        return None

    by_step = {e["step_id"]: e for e in data["elements"]}
    ifc_spaces = set(data["spaces"])

    full_footprints(data, assign)
    per_storey = defaultdict(list)
    for e in data["elements"]:
        if e["step_id"] not in assign:
            continue
        i, how = assign[e["step_id"]]
        e["_storey_index"], e["_storey_method"] = i, how
        per_storey[i].append(e)

    os.makedirs(out, exist_ok=True)

    building = {"format_version": 4, "model": model, "source_ifc": os.path.relpath(data["file"], ROOT),
                "representation": "owned elements have full-height silhouettes; projections and SVGs use height sections",
                "geometry_errors": data.get("geometry_errors", []),
                "schema": data["schema"], "unit_scale_to_m": data["unit_scale_to_m"],
                "storey_source": data["storey_source"],
                "n_elements": len(data["elements"]),
                "n_elements_with_geometry": sum(1 for e in data["elements"] if e["has_geometry"]),
                "storeys": [], "relations": data["relations"],
                "undeclared_levels": [{k: level.get(k) for k in ("name", "elevation_m", "walls", "moved_elements")}
                                      for level in undeclared]}
    if provenance:
        building["provenance"] = provenance

    # A file can list one physical level twice, an architectural storey beside a structural one.
    # Both cut the same geometry, so only one of them infers the level's rooms or the level would
    # hold each room twice. The file itself says which storey the level belongs to by putting its
    # spaces there, and the reference plans place their rooms the same way.
    infers_rooms = {}
    for group in level_clusters(storeys):
        def owned(index, test):
            return sum(1 for e in per_storey.get(index, []) if test(e))
        # Declared spaces decide first. Without them, the storey owning what defines rooms, walls,
        # doors and windows, is the floor; a structural or datum level owns beams, slabs and voids.
        owner = max(group, key=lambda index: (
            owned(index, lambda e: e["ifc_class"] == "IfcSpace" and e.get("_poly") is not None),
            owned(index, lambda e: e["ifc_class"].startswith(ROOM_DEFINING)),
            owned(index, lambda e: e["ifc_class"] not in NON_FABRIC),
            -index))
        for index in group:
            infers_rooms[index] = owner

    # Each storey's height runs to the next physical level above it, and the parametric layer
    # needs every element's footprint to place openings whose host wall sits on another storey.
    levels = level_clusters(storeys)
    next_level = {}
    for position, group in enumerate(levels):
        above = levels[position + 1] if position + 1 < len(levels) else None
        for index in group:
            next_level[index] = min(storeys[j]["elevation_m"] for j in above) if above else None
    elements_by_id = {e["step_id"]: e for e in data["elements"]}
    # walls drawn through openings at their ends, settled for the whole building before any storey
    # places an opening on them
    extensions = wall_extensions(data["elements"], data.get("relations") or [],
                                 {e["step_id"]: e["_storey_index"] for e in data["elements"] if "_storey_index" in e})

    # The ground level is the physical level at the ground-floor datum, +-0.00, allowing for a
    # raised ground floor. A file holding only the upper floors of a taller building has none.
    nearest = min(range(len(storeys)), key=lambda j: abs(storeys[j]["elevation_m"])) if storeys else None
    grade = set()
    if nearest is not None and abs(storeys[nearest]["elevation_m"]) <= GRADE_TOLERANCE_M:
        grade = set(next(group for group in levels if nearest in group))
    # Floors to stand on beyond an upper-storey door: slabs and roofs, including those built from parts.
    parents = {r["source"]: r["target"] for r in (data.get("relations") or []) if r.get("type") == "part_of"}

    def floor_like(element):
        step_id, cls, seen = element["step_id"], element["ifc_class"], set()
        while not cls.startswith(FLOORS):
            if step_id in seen or step_id not in parents:
                return False
            seen.add(step_id)
            step_id = parents[step_id]
            cls = elements_by_id.get(step_id, {}).get("ifc_class", "")
        return True

    floors = [e for e in data["elements"]
              if e.get("_poly") is not None and e.get("bbox_max") and floor_like(e)]

    roles, children = barrier_roles(data, barrier_classes)
    # barriers an opening can be cut into, should the file credit it to the wrong wall at a join
    cut_into = [e for e in data["elements"] if e["step_id"] in roles and e.get("_poly") is not None
                and not e["ifc_class"].startswith(("IfcDoor", "IfcWindow"))]
    cut_into = (STRtree([e["_poly"] for e in cut_into]), cut_into) if cut_into else None
    all_polys = []
    storey_rooms = {}
    storey_records = []
    for i, st in enumerate(storeys):
        elems = per_storey.get(i, [])
        visible, section_info = section_elements(data, assign, i, cut_height=cut_height,
                                                 cut_thickness=cut_thickness, band_base=band_base,
                                                 barrier_classes=barrier_classes)
        visible_by_id = {e['step_id']: e for e in visible}
        # Barriers for room inference: walls plus the elements that fill openings cut
        # through them. Without doors and windows the wall footprints have a hole at every
        # doorway and no room is ever enclosed.
        walls, _ = barrier_polygons(visible, data)
        sectioned = section_info.get("status") != "no_section_band"
        holes = opening_holes(data, roles, by_step, children,
                              section_info.get("band_m") if sectioned else None, st["elevation_m"],
                              sections={sid: e["_poly"] for sid, e in visible_by_id.items() if e.get("_poly") is not None},
                              barriers=cut_into)
        ways = [h for h in holes if h["walkable"] and not h.get("same_as")]
        # a room ends at its wall, across every opening in it; a way through stays open only where
        # nothing fills the opening and a person can walk through it
        enclosing = walls + [h["_poly"] for h in holes] if walls else []
        hatches = [h["_poly"] for h in holes if not h["walkable"] and h["filled_by"] is None and not h.get("same_as")]
        across = walls + hatches + [h["_poly"] for h in holes if h["filled_by"] is not None]

        # rooms: declared spaces and inferred regions, reconciled rather than one or the other
        rooms, room_source = [], "none"
        inference = {"status": "not_attempted"}
        sp = [e for e in elems if e["ifc_class"] == "IfcSpace" and e.get("_poly") is not None]
        declared = [{"id": f"space_{e['step_id']}", "step_id": e["step_id"], "global_id": e.get("global_id"),
                     "name": e["label"], "source": "IfcSpace",
                     "area_m2": round(e["_poly"].area, 2),
                     "polygon": poly_coords(e["_poly"]),
                     "geometry": poly_geometry(e["_poly"]),
                     "_poly": e["_poly"],
                     "_stated": e.get("_storey_method") not in ("inferred_from_elevation", "inferred_clamped"),
                     "_z": (e["bbox_min"][2], e["bbox_max"][2]) if e.get("bbox_min") and e.get("bbox_max") else None,
                     "storey_assignment": e["_storey_method"],
                     **({"storey_stated": e["storey_stated"]} if "storey_stated" in e else {})}
                    for e in sp]
        inferred = []
        if walls and infers_rooms.get(i, i) != i:
            owner = infers_rooms[i]
            inference = {"status": "level_inferred_on_another_storey",
                         "storey_index": owner, "storey_name": storeys[owner]["name"]}
        elif walls:
            bnds = unary_union(enclosing).bounds
            for k, r in enumerate(infer_rooms(enclosing, bnds, diagnostics=inference, **room_opts)):
                inferred.append({"id": f"room_{i}_{k}", "step_id": None,
                                 "name": f"Room {k+1}", "source": "inferred_from_wall_geometry",
                                 "area_m2": r["area_m2"],
                                 "raster_area_m2": r["raster_area_m2"],
                                 "polygon": poly_coords(r["polygon"]),
                                 "geometry": poly_geometry(r["polygon"]),
                                 "_poly": r["polygon"]})
        elif not declared:
            inference = {"status": "no_barriers"}
        # the walls say which of two overlapping declared spaces is the room: see disjoint_declared
        declared, set_aside = disjoint_declared(declared, regions=[r["_poly"] for r in inferred])
        rooms, reconciliation = reconcile_rooms(declared, inferred, ways=[h["_poly"] for h in ways])
        rooms, given_way = disjoint_rooms(rooms)
        if any(r.get("trimmed_by_rooms") for r in rooms) or given_way:
            reconciliation["trimmed_to_stay_disjoint"] = sum(1 for r in rooms if r.get("trimmed_by_rooms"))
            if given_way:
                reconciliation["dropped_to_stay_disjoint"] = len(given_way)
        if set_aside:
            reconciliation["declared_set_aside"] = len(set_aside)
        if any(r.get("carved_by") for r in declared):
            reconciliation["declared_carved"] = sum(1 for r in declared if r.get("carved_by"))
        # Two rooms must not cover the same floor, or a door stepping out of its wall lands in
        # whichever of them comes first. Recorded per storey so a regression cannot pass unseen.
        overlaps = []
        for x in range(len(rooms)):
            for y in range(x + 1, len(rooms)):
                first, second = rooms[x]["_poly"], rooms[y]["_poly"]
                if not first.intersects(second):
                    continue
                shared = first.intersection(second).area
                if shared > .5:
                    overlaps.append({"a": rooms[x]["id"], "b": rooms[y]["id"],
                                     "area_m2": round(shared, 2)})
        sources = sorted({r["source"] for r in rooms})
        room_source = "+".join(sources) if sources else "none"
        # the inference record describes the inference attempt alone, whether or not the file
        # also declares spaces on this storey
        if section_info.get('status') == 'no_section_band':
            inference = {'status': 'no_section_band', 'reason': section_info.get('reason')}
        elif not section_info['barriers_complete']:
            inference.update(status='incomplete_section_barriers', partial_regions=len(inferred))
        elif not section_info.get('barriers_all_sectioned', True):
            inference['barriers_from_full_height_silhouette'] = sum(
                1 for f in section_info.get('fallbacks', []) if f['affects_room_barrier'])

        # element records + which room each sits in
        recs = []
        for e in elems:
            if e["ifc_class"] in ("IfcSpace",):
                continue
            poly = e.get("_poly")
            rid, room_by = None, None
            section_poly = visible_by_id.get(e['step_id'], {}).get('_poly')
            if section_poly is not None and rooms:
                c = section_poly.centroid
                for r in rooms:
                    if r["_poly"].contains(c):
                        rid, room_by = r["id"], "section"
                        break
            if rid is None and poly is not None and rooms and not e["ifc_class"].startswith(NOT_CONTENTS):
                # Furniture, fittings and equipment mostly stand below the cut, so their whole
                # footprint places them. Walls, openings and structure bound rooms instead.
                c = poly.centroid
                for r in rooms:
                    if r["_poly"].contains(c):
                        rid, room_by = r["id"], "footprint"
                        break
            rec = {k: v for k, v in e.items() if not k.startswith("_")}
            rec["footprint"] = poly_coords(poly) if poly is not None else []
            rec["footprint_geometry"] = poly_geometry(poly)
            rec["footprint_method"] = e.get("_poly_method", "none")
            rec["orientation_deg"] = orientation_deg(poly) if poly is not None else None
            rec["storey_index"] = i
            rec["storey_assignment"] = e["_storey_method"]
            rec["room"] = rid
            rec["room_by"] = room_by
            recs.append(rec)
            if poly is not None and e["ifc_class"] not in NON_FABRIC:
                all_polys.append((i, e["ifc_class"], poly))

        # which rooms connect to which: the edge a scene graph is built on, and the one the
        # IFC never states
        hole_of = {h["filled_by"]: h["_poly"] for h in holes if h["filled_by"] is not None}
        doors = [{**e, "_hole": hole_of.get(e["step_id"])} for e in visible if e["ifc_class"] == "IfcDoor"]
        storey_key = st.get("global_id") or f"{model}-storey-{i}"
        outside = outside_region(enclosing)
        at_grade = i in grade
        standable = None
        if not at_grade:
            # somewhere to stand is a slab or roof whose top is on this floor's physical level: a
            # balcony a step down, or a flat roof a step or two up
            surfaces = [e["_poly"] for e in floors if abs(e["bbox_max"][2] - st["elevation_m"]) < SAME_LEVEL_M]
            # so is the floor that furniture, fittings or a railing stand on, though the file may model
            # no slab under it (model_192 has chairs, tables and toilets beyond a door, and no slab)
            surfaces += [e["_poly"] for e in data["elements"]
                         if e.get("_poly") is not None and e.get("bbox_min")
                         and abs(e["bbox_min"][2] - st["elevation_m"]) <= STANDING_M
                         and not e["ifc_class"].startswith(NOT_CONTENTS) and e["ifc_class"] not in NON_FABRIC
                         and e["step_id"] not in roles]
            standable = unary_union(surfaces) if surfaces else None
        slabs = [e["_poly"] for e in elems if e["ifc_class"].startswith("IfcSlab") and e.get("_poly") is not None]
        railings = [e["_poly"] for e in elems if e["ifc_class"].startswith("IfcRailing") and e.get("_poly") is not None]
        outdoor, outdoor_counts = outdoor_spaces(slabs, railings, rooms, outside, storey_id=storey_key)
        # a door onto a balcony leads to the balcony, not to the street
        # a door's own doorway must not stop the search for the rooms it joins; a hatch does
        blocked = unary_union([w.buffer(OUTSIDE_TOLERANCE_M) for w in walls + hatches]) if walls else None
        if rooms or outdoor:
            connections, connection_counts = door_connections(rooms + outdoor, doors, storey_id=storey_key,
                                                              outside=outside, blocked=blocked,
                                                              at_grade=at_grade, standable=standable)
            # an archway or doorless doorway is crossed like a door, and is an open passage
            through, through_counts = door_connections(
                rooms + outdoor, [{"_poly": h["_poly"], "_hole": h["_poly"], "global_id": h["global_id"],
                                   "step_id": h["step_id"], "width_m": h["width_m"]} for h in ways],
                storey_id=storey_key, outside=outside, blocked=blocked, at_grade=at_grade,
                standable=standable, noun="unfilled_opening")
            connections += through
            connection_counts = {**connection_counts, **through_counts}
        else:
            # a second record of a level whose rooms sit on the other one: that storey's cut
            # holds these doors too and connects them there
            through = []
            connections, connection_counts = [], {"doors_on_a_storey_without_rooms": len(doors)} if doors else {}
        connection_counts = {**connection_counts, **outdoor_counts}
        if not sectioned:
            # An open passage claims nothing is built between two rooms. This storey has no
            # section, so there is no evidence either way and every pair of abutting spaces
            # would otherwise become a way through.
            passages, passage_counts = [], {'open_passages_without_a_section': len(rooms)}
        elif not walls:
            # nothing at all stands at this cut, though rooms do: the cut is no evidence either way
            passages, passage_counts = [], {'open_passages_without_barriers_at_the_cut': len(rooms)}
        else:
            # a door is a way through of its own, wherever its leaf is drawn
            door_outlines = [by_step[e["step_id"]]["_poly"] for e in doors
                             if by_step.get(e["step_id"], {}).get("_poly") is not None]
            passages, passage_counts = open_passages(rooms, across + door_outlines)
            joined = {frozenset((e["a"], e["b"])) for e in through if e.get("b")}
            passages = [e for e in passages if frozenset((e["a"], e["b"])) not in joined]
        if passages and not section_info.get('barriers_complete', False):
            # Barriers were excluded from this section, so something may stand in the gap that
            # the cut never saw. The passage is still reported, with what it rests on.
            for passage in passages:
                passage['barrier_evidence'] = 'incomplete_section_barriers'
        owned_barriers = [e["_poly"] for e in elems
                          if e["ifc_class"].startswith(BARRIERS) and e.get("_poly") is not None]
        beyond, outside_counts = exterior_passages(rooms, across, storey_id=storey_key,
                                                   outside=outside, at_any_height=owned_barriers,
                                                   at_grade=at_grade, standable=standable)
        connections += passages + beyond
        connection_counts = {**connection_counts, **passage_counts, **outside_counts}
        if at_grade and outdoor:
            # a terrace at ground level leads onto the ground where its railing leaves a way down;
            # a balcony above it is a dead end, as the walkability rule has it
            down, down_counts = terrace_ways_out(outdoor, railings, across, rooms, storey_id=storey_key)
            connections += down
            connection_counts = {**connection_counts, **down_counts}
        storey_rooms[i] = rooms
        # which elements bound each room at the cut, the IFC's space boundaries read from geometry
        boundaries, boundary_counts = room_boundaries(
            rooms, [e for e in visible if e.get("_is_barrier") and e.get("_poly") is not None]
            + [{"step_id": h["step_id"], "ifc_class": "IfcOpeningElement", "_poly": h["_poly"]}
               for h in holes if h["filled_by"] is None and not h.get("same_as")])
        vector = storey_vector(elems, data.get("relations") or [], connections, st["elevation_m"],
                               next_level.get(i), elements_by_id, extensions)

        # spatial relations derived from footprints, in the LiteReality manner
        srel = section_adjacencies(visible)
        projections = []
        for e in visible:
            rid = next((r['id'] for r in rooms if r['_poly'].contains(e['_poly'].centroid)), None)
            projections.append({'step_id': e['step_id'], 'ifc_class': e['ifc_class'],
                                'owner_storey_index': e['owner_storey_index'],
                                'geometry': poly_geometry(e['_poly']), 'area_m2': e['_poly'].area,
                                'projection': e['section_projection'], 'room': rid})
            projections[-1]['is_room_barrier'] = e['_is_barrier']
            if rid:
                srel.append({'type': 'in_room', 'source': e['step_id'], 'target': rid,
                             'source_of_truth': 'section_centroid_in_room_polygon'})

        srec = {
            "format_version": 4, "room_inference": inference,
            'geometry_quality': section_quality(section_info),
            "room_reconciliation": reconciliation, "room_overlaps": overlaps,
            "section_projection": section_info, "projections": projections,
            "room_connections": connections, "connection_counts": connection_counts,
            "outdoor_spaces": [{k: v for k, v in o.items() if not k.startswith("_")} for o in outdoor],
            "n_visible_elements": len(visible),
            "step_id": st.get("step_id"), "global_id": st.get("global_id"),
            "storey_index": i, "name": st["name"], "elevation_m": st["elevation_m"],
            "height_m": vector["height_m"], "height_from": vector["height_from"], "at_grade": at_grade,
            "storey_source": st["source"], "room_source": room_source,
            "n_rooms": len(rooms), "n_elements": len(recs),
            "classes": class_counts(recs),
            "rooms": [{**{k: v for k, v in r.items() if not k.startswith("_")},
                       "floor_to_floor_m": vector["height_m"],
                       "bounded_share": r.get("_bounded_share")} for r in rooms],
            "room_boundaries": boundaries,
            "declared_spaces_set_aside": set_aside,
            "opening_holes": [{**{k: v for k, v in h.items() if not k.startswith("_")},
                               "geometry": poly_geometry(h["_poly"])} for h in holes],
            "vector": {k: vector[k] for k in ("walls", "openings", "slabs", "counts")},
            "elements": recs,
            "spatial_relations": srel,
        }
        with open(os.path.join(out, f"storey_{i:02d}.json"), "w") as fh:
            json.dump(srec, fh, indent=1)
        building["storeys"].append({k: v for k, v in srec.items()
                                    if k not in ("elements", "rooms", "spatial_relations", "projections", "vector",
                                                 "room_boundaries", "opening_holes")})
        storey_records.append(srec)
        sectioned = {e["step_id"] for e in visible}
        below = [e for e in elems if e["ifc_class"].startswith(("IfcSlab", "IfcRailing"))
                 and e.get("_poly") is not None and e["step_id"] not in sectioned]
        render_plan(srec, rooms, visible, os.path.join(out, f"storey_{i:02d}.svg"), model,
                    below=below, outdoor=outdoor)

    # stairs and lifts join rooms across storeys, so this is settled after every storey is built
    vertical, vertical_counts = vertical_connections(
        storey_rooms, data["elements"], storeys, relations=data.get("relations") or ())
    building["vertical_connections"] = vertical
    building["graph_counts"] = {
        "rooms": sum(s["n_rooms"] for s in building["storeys"]),
        "rooms_declared": sum(1 for i in storey_rooms for r in storey_rooms[i] if r["source"] == "IfcSpace"),
        "rooms_inferred": sum(1 for i in storey_rooms for r in storey_rooms[i] if r["source"] != "IfcSpace"),
        "connected_by_door": sum(s.get("connection_counts", {}).get("connected_by_door", 0) for s in building["storeys"]),
        "exterior_doors": sum(s.get("connection_counts", {}).get("exterior_door", 0) for s in building["storeys"]),
        "open_passages": sum(s.get("connection_counts", {}).get("open_passage", 0) for s in building["storeys"]),
        "open_passages_to_outside": sum(s.get("connection_counts", {}).get("open_passage_to_outside", 0)
                                        for s in building["storeys"]),
        "outdoor_spaces": sum(len(s.get("outdoor_spaces", [])) for s in storey_records),
        **vertical_counts}
    with open(os.path.join(out, "building.json"), "w") as fh:
        json.dump(building, fh, indent=1)
    render_bev(all_polys, storeys, os.path.join(out, "bev.svg"), model)
    if verbose:
        print(f"  {model}: {len(storeys)} storeys, {building['n_elements']} elements "
              f"({data['schema']}, storeys from {data['storey_source']})")
    return building


# rendering
def _frame(polys, pad=1.5, width=1400):
    xs, ys = [], []
    for p in polys:
        if p is None or p.is_empty:
            continue
        a, b, c, d = p.bounds
        xs += [a, c]
        ys += [b, d]
    if not xs:
        return None
    minx, maxx, miny, maxy = min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad
    w, h = maxx - minx, maxy - miny
    s = width / w
    return minx, miny, maxx, maxy, s, width, max(h * s, 10)


def _path(poly, fr):
    minx, miny, maxx, maxy, s, W, H = fr
    geoms = poly.geoms if isinstance(poly, MultiPolygon) else [poly]
    d = []
    for g in geoms:
        for ring in (g.exterior, *g.interiors):
            pts = [f"{(x-minx)*s:.1f},{(maxy-y)*s:.1f}" for x, y in ring.coords]
            if len(pts) >= 4:
                d.append("M" + " L".join(pts) + " Z")
    return " ".join(d)


def render_plan(srec, rooms, elems, path, model, *, below=(), outdoor=()):
    """One storey as a drawing: the section at the cut, with what stands below it drawn lightly.

        A plan shows what the cut passes through and, more faintly, what is below it. A 1.1 m railing
        and its slab never reach a 1.2 m cut, so without the lighter layer a balcony is missing from
        the drawing even though the file holds every part of it.
    """
    polys = [e["_poly"] for e in elems if e.get("_poly") is not None
             and e["ifc_class"] not in NON_FABRIC]
    polys += [r["_poly"] for r in rooms] + [o["_poly"] for o in outdoor]
    polys += [e["_poly"] for e in below if e.get("_poly") is not None]
    fr = _frame(polys)
    if fr is None:
        _write(path, "<svg xmlns='http://www.w3.org/2000/svg'/>")
        return
    minx, miny, maxx, maxy, s, W, H = fr
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" '
         f'font-family="Helvetica,Arial,sans-serif">',
         f'<rect width="{W:.0f}" height="{H:.0f}" fill="#FFFFFF"/>']
    # below the cut: slabs as a faint dashed outline, drawn first so everything else sits on top
    for e in below:
        if e["ifc_class"].startswith("IfcSlab"):
            o.append(f'<path fill-rule="evenodd" d="{_path(e["_poly"], fr)}" fill="none" stroke="#C9C4B8" '
                     f'stroke-width="0.8" stroke-dasharray="4 3"/>')
    for sp in outdoor:
        o.append(f'<path fill-rule="evenodd" d="{_path(sp["_poly"], fr)}" fill="#E6F0E1" stroke="#9DB890" '
                 f'stroke-width="1" stroke-dasharray="5 3"/>')
    for r in rooms:
        o.append(f'<path fill-rule="evenodd" d="{_path(r["_poly"], fr)}" fill="#F3F1EC" stroke="#DDD8CE" '
                 f'stroke-width="1"/>')
    for e in below:
        if e["ifc_class"].startswith("IfcRailing"):
            o.append(f'<path fill-rule="evenodd" d="{_path(e["_poly"], fr)}" fill="#8F8A80" stroke="#8F8A80" '
                     f'stroke-width="0.6" fill-opacity="0.8"/>')
    order = {c: i for i, c in enumerate(DRAW_ORDER)}
    # openings, grids and annotation are voids or references, not building fabric: they
    # stay in the JSON because they carry meaning, but drawing them obscures the plan.
    for e in sorted([e for e in elems if e.get("_poly") is not None
                     and e["ifc_class"] not in NON_FABRIC],
                    key=lambda e: order.get(e["ifc_class"], -1)):
        fill, stroke, sw = STYLE.get(e["ifc_class"], DEFAULT_STYLE)
        o.append(f'<path fill-rule="evenodd" d="{_path(e["_poly"], fr)}" fill="{fill}" stroke="{stroke}" '
                 f'stroke-width="{sw}" fill-opacity="0.95"/>')
    for sp in outdoor:
        c = sp["_poly"].representative_point()
        o.append(f'<text x="{(c.x-minx)*s:.1f}" y="{(maxy-c.y)*s:.1f}" font-size="10" '
                 f'fill="#5E7A52" text-anchor="middle">{_esc(sp["name"])}</text>')
        o.append(f'<text x="{(c.x-minx)*s:.1f}" y="{(maxy-c.y)*s+11:.1f}" font-size="8" '
                 f'fill="#7D9670" text-anchor="middle">{sp["area_m2"]:.1f} m²</text>')
    for r in rooms:
        c = r["_poly"].centroid
        if r["_poly"].area > 4:
            o.append(f'<text x="{(c.x-minx)*s:.1f}" y="{(maxy-c.y)*s:.1f}" font-size="11" '
                     f'fill="#55534B" text-anchor="middle">{_esc(r["name"])}</text>')
            o.append(f'<text x="{(c.x-minx)*s:.1f}" y="{(maxy-c.y)*s+12:.1f}" font-size="8.5" '
                     f'fill="#8B8880" text-anchor="middle">{r["area_m2"]:.1f} m²</text>')
    o.append(f'<text x="14" y="22" font-size="14" fill="#1C1B19">{_esc(model)}: '
             f'{_esc(srec["name"])} (+{srec["elevation_m"]} m)</text>')
    o.append(f'<text x="14" y="40" font-size="10" fill="#8B8880">'
             f'{srec.get("n_visible_elements", srec["n_elements"])} visible elements, {srec["n_rooms"]} rooms '
             f'({_esc(srec["room_source"])})</text>')
    band = (srec.get('section_projection') or {}).get('band_m')
    if band:
        lo, hi = band
        o.append(f'<text x="14" y="56" font-size="10" fill="#8B8880">'
                 f'Height section {lo:.3f} to {hi:.3f} m; {srec["n_elements"]} owned elements</text>')
    elif srec.get('section_projection'):
        o.append(f'<text x="14" y="56" font-size="10" fill="#8B8880">'
                 f'No height section: {_esc(str(srec["section_projection"].get("reason")))}</text>')
    o.append("</svg>")
    _write(path, "\n".join(o))


def render_bev(all_polys, storeys, path, model):
    fr = _frame([p for _, _, p in all_polys])
    if fr is None:
        _write(path, "<svg xmlns='http://www.w3.org/2000/svg'/>")
        return
    minx, miny, maxx, maxy, s, W, H = fr
    n = max(len(storeys), 1)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" '
         f'font-family="Helvetica,Arial,sans-serif">',
         f'<rect width="{W:.0f}" height="{H:.0f}" fill="#FFFFFF"/>']
    for i, cls, p in sorted(all_polys, key=lambda t: t[0]):
        # lower storeys darker, upper storeys lighter, so stacking reads as height
        t = i / max(n - 1, 1)
        g = int(58 + t * 130)
        op = 0.85 - t * 0.45
        o.append(f'<path fill-rule="evenodd" d="{_path(p, fr)}" fill="rgb({g},{g-4},{g-10})" '
                 f'fill-opacity="{op:.2f}" stroke="none"/>')
    o.append(f'<text x="14" y="22" font-size="14" fill="#1C1B19">{_esc(model)}: '
             f"bird's-eye view</text>")
    o.append(f'<text x="14" y="40" font-size="10" fill="#8B8880">{n} storeys stacked; '
             f'darker is lower</text>')
    o.append("</svg>")
    _write(path, "\n".join(o))


def _write(path, text):
    with open(path, "w") as fh:
        fh.write(text)


def _esc(t):
    return (str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


if __name__ == "__main__":
    models = sys.argv[1:] or ["model_119"]
    os.makedirs(OUT_DIR, exist_ok=True)
    for m in models:
        build(m)
