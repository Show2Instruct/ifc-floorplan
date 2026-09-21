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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fp.extract import extract, footprint, poly_coords, poly_geometry, infer_rooms, NON_FABRIC
from fp.projections import barrier_polygons, full_footprints, section_elements, BARRIERS
from fp.connectivity import (OUTSIDE_TOLERANCE_M, door_connections, exterior_passages,
                             open_passages, outdoor_spaces, outside_region, vertical_connections)

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


def class_counts(recs):
    """Element count per IFC class, largest first, ties broken by class name.

    Leaving ties to set iteration made the key order depend on Python's per-process
    string hashing, so two identical runs wrote different bytes."""
    return dict(sorted(Counter(r["ifc_class"] for r in recs).items(),
                       key=lambda kv: (-kv[1], kv[0])))


CONTAINED_IN_SPACE = .8     # of an inferred region's own area, before it counts as that space


def reconcile_rooms(declared, inferred, *, contained=CONTAINED_IN_SPACE):
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

        A region no declared space contains is a room the file never declared, and is kept whole with
        its own geometry and area. Containment is measured as a share of the inferred region's area,
        so a declared space bleeding a few centimetres over a wall centreline does not swallow the
        room next door.

        Returns (rooms, notes) with every room's origin recorded.
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
        else:
            region['overlap_with_declared_space'] = round(share, 3)
            kept.append(region)
            notes['kept_outside_declared_spaces'] += 1
    rooms = []
    for room in declared:
        parts = room.pop('_parts')
        if len(parts) <= 1:
            rooms.append(room)
            notes['declared'] += 1
            notes['absorbed_inferred_duplicate'] += len(parts)
            continue
        # walls divide what the file called one space: the parts are the rooms
        for part in parts:
            part['subdivides_declared_space'] = room['step_id']
            part['name'] = f"{room['name']} part {len(rooms) + 1}" if room.get('name') else part['name']
            rooms.append(part)
        notes['declared_subdivided'] += 1
        notes['subdivision_parts'] += len(parts)
    rooms.extend(kept)
    return rooms, dict(notes)


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

    building = {"format_version": 3, "model": model, "source_ifc": os.path.relpath(data["file"], ROOT),
                "representation": "owned elements have full-height silhouettes; projections and SVGs use height sections",
                "geometry_errors": data.get("geometry_errors", []),
                "schema": data["schema"], "unit_scale_to_m": data["unit_scale_to_m"],
                "storey_source": data["storey_source"],
                "n_elements": len(data["elements"]),
                "n_elements_with_geometry": sum(1 for e in data["elements"] if e["has_geometry"]),
                "storeys": [], "relations": data["relations"]}
    if provenance:
        building["provenance"] = provenance

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

        # rooms: declared spaces and inferred regions, reconciled rather than one or the other
        rooms, room_source = [], "none"
        inference = {"status": "not_attempted"}
        sp = [e for e in elems if e["ifc_class"] == "IfcSpace" and e.get("_poly") is not None]
        declared = [{"id": f"space_{e['step_id']}", "step_id": e["step_id"],
                     "name": e["label"], "source": "IfcSpace",
                     "area_m2": round(e["_poly"].area, 2),
                     "polygon": poly_coords(e["_poly"]),
                     "geometry": poly_geometry(e["_poly"]),
                     "_poly": e["_poly"]} for e in sp]
        inferred = []
        if walls:
            bnds = unary_union(walls).bounds
            for k, r in enumerate(infer_rooms(walls, bnds, diagnostics=inference, **room_opts)):
                inferred.append({"id": f"room_{i}_{k}", "step_id": None,
                                 "name": f"Room {k+1}", "source": "inferred_from_wall_geometry",
                                 "area_m2": r["area_m2"],
                                 "raster_area_m2": r["raster_area_m2"],
                                 "polygon": poly_coords(r["polygon"]),
                                 "geometry": poly_geometry(r["polygon"]),
                                 "_poly": r["polygon"]})
        elif not declared:
            inference = {"status": "no_barriers"}
        rooms, reconciliation = reconcile_rooms(declared, inferred)
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
            rid = None
            section_poly = visible_by_id.get(e['step_id'], {}).get('_poly')
            if section_poly is not None and rooms:
                c = section_poly.centroid
                for r in rooms:
                    if r["_poly"].contains(c):
                        rid = r["id"]
                        break
            rec = {k: v for k, v in e.items() if not k.startswith("_")}
            rec["footprint"] = poly_coords(poly) if poly is not None else []
            rec["footprint_geometry"] = poly_geometry(poly)
            rec["footprint_method"] = e.get("_poly_method", "none")
            rec["orientation_deg"] = orientation_deg(poly) if poly is not None else None
            rec["storey_index"] = i
            rec["storey_assignment"] = e["_storey_method"]
            rec["room"] = rid
            recs.append(rec)
            if poly is not None and e["ifc_class"] not in NON_FABRIC:
                all_polys.append((i, e["ifc_class"], poly))

        # which rooms connect to which: the edge a scene graph is built on, and the one the
        # IFC never states
        doors = [e for e in visible if e["ifc_class"] == "IfcDoor"]
        storey_key = st.get("global_id") or f"{model}-storey-{i}"
        outside = outside_region(walls)
        slabs = [e["_poly"] for e in elems if e["ifc_class"].startswith("IfcSlab") and e.get("_poly") is not None]
        railings = [e["_poly"] for e in elems if e["ifc_class"].startswith("IfcRailing") and e.get("_poly") is not None]
        outdoor, outdoor_counts = outdoor_spaces(slabs, railings, rooms, outside, storey_id=storey_key)
        # a door onto a balcony leads to the balcony, not to the street
        blocked = unary_union([w.buffer(OUTSIDE_TOLERANCE_M) for w in walls]) if walls else None
        connections, connection_counts = door_connections(rooms + outdoor, doors, storey_id=storey_key,
                                                          outside=outside, blocked=blocked)
        connection_counts = {**connection_counts, **outdoor_counts}
        passages, passage_counts = open_passages(rooms, walls)
        owned_barriers = [e["_poly"] for e in elems
                          if e["ifc_class"].startswith(BARRIERS) and e.get("_poly") is not None]
        beyond, outside_counts = exterior_passages(rooms, walls, storey_id=storey_key,
                                                   outside=outside, at_any_height=owned_barriers)
        connections += passages + beyond
        connection_counts = {**connection_counts, **passage_counts, **outside_counts}
        storey_rooms[i] = rooms

        # spatial relations derived from footprints, in the LiteReality manner
        srel = []
        keyed = [(e['step_id'], e['_poly'], e['ifc_class']) for e in visible]
        for a in range(len(keyed)):
            for b in range(a + 1, len(keyed)):
                ia, pa, ca = keyed[a]
                ib, pb, cb = keyed[b]
                if pa is None or pb is None:
                    continue
                if ca.startswith("IfcWall") != cb.startswith("IfcWall"):
                    continue
                if pa.distance(pb) < 0.05:
                    srel.append({"type": "adjacent_to", "source": ia, "target": ib,
                                 "source_of_truth": "section_footprint_proximity<5cm"})
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
            "format_version": 3, "room_inference": inference,
            "room_reconciliation": reconciliation,
            "section_projection": section_info, "projections": projections,
            "room_connections": connections, "connection_counts": connection_counts,
            "outdoor_spaces": [{k: v for k, v in o.items() if not k.startswith("_")} for o in outdoor],
            "n_visible_elements": len(visible),
            "step_id": st.get("step_id"), "global_id": st.get("global_id"),
            "storey_index": i, "name": st["name"], "elevation_m": st["elevation_m"],
            "storey_source": st["source"], "room_source": room_source,
            "n_rooms": len(rooms), "n_elements": len(recs),
            "classes": class_counts(recs),
            "rooms": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rooms],
            "elements": recs,
            "spatial_relations": srel,
        }
        with open(os.path.join(out, f"storey_{i:02d}.json"), "w") as fh:
            json.dump(srec, fh, indent=1)
        building["storeys"].append({k: v for k, v in srec.items()
                                    if k not in ("elements", "rooms", "spatial_relations", "projections")})
        storey_records.append(srec)
        sectioned = {e["step_id"] for e in visible}
        below = [e for e in elems if e["ifc_class"].startswith(("IfcSlab", "IfcRailing"))
                 and e.get("_poly") is not None and e["step_id"] not in sectioned]
        render_plan(srec, rooms, visible, os.path.join(out, f"storey_{i:02d}.svg"), model,
                    below=below, outdoor=outdoor)

    # stairs and lifts join rooms across storeys, so this is settled after every storey is built
    vertical, vertical_counts = vertical_connections(
        storey_rooms, [e for e in data["elements"] if e.get("_poly") is not None], storeys)
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
