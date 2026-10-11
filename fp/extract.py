#!/usr/bin/env python
"""Read an IFC file into per-storey floor plan data.

IFC2X3, IFC4 and IFC4X3 go through the same code path, using inverse attributes and
ifcopenshell.util rather than schema-specific entity names. Anything with a shape is
extracted whatever its class, and its property sets, quantities, material and type come with
it. Rooms are taken from IfcSpace where the file has them and inferred from the walls where
it does not. Lengths are converted to metres with the file's own unit scale, so a model drawn
in millimetres and one drawn in metres come out the same.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element as ue
import ifcopenshell.util.unit as uu
from shapely.geometry import Polygon, MultiPolygon
import shapely
from shapely.ops import unary_union

# Classes that are spatial structure or annotation rather than building fabric.
NON_FABRIC = {"IfcAnnotation", "IfcGrid", "IfcVirtualElement", "IfcOpeningElement",
              "IfcSpace", "IfcSite", "IfcBuilding", "IfcBuildingStorey", "IfcProject"}
MAX_TRIS_EXACT = 4000          # above this, fall back to a convex hull footprint
PLANAR_THICKNESS_M = 0.06      # nominal thickness given to zero-thickness panels


# geometry helpers
def _tri_polygons(verts_xy, faces):
    polys = []
    for a, b, c in faces:
        p = Polygon((verts_xy[a], verts_xy[b], verts_xy[c]))
        if p.is_valid and p.area > 1e-9:
            polys.append(p)
    return polys


def _union(polys):
    """Merge projected triangles.

        Dense meshes, such as detailed door leaves, can make GEOS fail on nearly coincident
        edges, so retry on a 1 micrometre grid before giving up.
    """
    try:
        return unary_union(polys).buffer(0), "union"
    except shapely.errors.GEOSException:
        try:
            return shapely.union_all(polys, grid_size=1e-6).buffer(0), "union_snapped"
        except shapely.errors.GEOSException:
            return None, "none"


def footprint(verts, faces, *, z_lo=None, z_hi=None, diagnostics=None):
    """2D outline of an element seen from above, in metres.

        Triangles are projected to XY and unioned, so an L-shaped wall stays L-shaped. Very dense
        meshes fall back to a convex hull, and the method used is reported. Height bounds select
        whole faces; they do not cut triangles that cross them.
    """
    v = verts
    info = diagnostics if diagnostics is not None else {}
    info.update(height_scope="full_height", reason=None)
    if not np.isfinite(v).all():
        raise ValueError("cannot project non-finite vertices")
    if z_lo is not None and z_hi is not None:
        keep = (v[:, 2] >= z_lo - 1e-6) & (v[:, 2] <= z_hi + 1e-6)
        if keep.sum() >= 3:
            info.update(height_scope="band_vertex_subset", reason="crossing triangles are not cut")
            idx = np.where(keep)[0]
            remap = {o: i for i, o in enumerate(idx)}
            faces = [f for f in faces if all(x in remap for x in f)]
            faces = [(remap[a], remap[b], remap[c]) for a, b, c in faces]
            v = v[idx]
        else:
            info.update(height_scope="full_height_fallback", reason="fewer_than_three_vertices_in_band")
    if len(v) < 3:
        return None, "none"
    xy = v[:, :2]
    if faces and len(faces) <= MAX_TRIS_EXACT:
        polys = _tri_polygons(xy, faces)
        if polys:
            u, method = _union(polys)
            if u is not None and not u.is_empty and u.area > 1e-9:
                return u, method
    # Door and window panels are often modelled with zero thickness. They project to a line and
    # would vanish from the plan, so give them the thickness they occupy in reality.
    try:
        from shapely.geometry import MultiPoint, LineString
        pts = MultiPoint([tuple(p) for p in xy])
        hull = pts.convex_hull
        # Collinear points come back as a zero-area Polygon rather than a LineString,
        # so degeneracy is tested by area, not by geometry type.
        if hull.is_empty or hull.area < 1e-6:
            b = hull.buffer(PLANAR_THICKNESS_M / 2, cap_style=2)
            if not b.is_empty and b.area > 1e-9:
                return b, "planar_buffered"
    except Exception:
        pass
    try:
        from shapely.geometry import MultiPoint
        h = MultiPoint([tuple(p) for p in xy]).convex_hull
        if h.geom_type == "Polygon" and h.area > 1e-9:
            return h, "hull"
    except Exception:
        pass
    return None, "none"


def poly_coords(p, ndigits=3):
    """Serialise a (Multi)Polygon as its exterior rings only. Holes are dropped, so an
    outline with a hole reads as filled; poly_geometry keeps them."""
    if p is None:
        return []
    geoms = p.geoms if isinstance(p, MultiPolygon) else [p]
    out = []
    for g in geoms:
        ring = [[round(x, ndigits), round(y, ndigits)] for x, y in g.exterior.coords]
        if len(ring) >= 4:
            out.append(ring)
    return out


def poly_geometry(p, ndigits=None):
    """Serialise a (Multi)Polygon with its holes, as GeoJSON MultiPolygon coordinates.

        Full precision by default: rounding thin wall and door rings to millimetres can make them
        self-intersect. poly_coords stays a compact display outline, not authoritative geometry.
    """
    if p is None or p.is_empty:
        return []
    geoms = p.geoms if isinstance(p, MultiPolygon) else [p]
    out = []
    for g in geoms:
        coord = (lambda value: float(value)) if ndigits is None else (lambda value: round(value, ndigits))
        rings = [[[coord(x), coord(y)] for x, y in ring.coords]
                 for ring in (g.exterior, *g.interiors)]
        if len(rings[0]) >= 4:
            out.append([rings[0]] + [r for r in rings[1:] if len(r) >= 4])
    return out


def storey_from_placement(e, max_hops=16):
    """The storey an element is placed relative to, read from its placement chain.

        Following ObjectPlacement.PlacementRelTo upwards usually reaches the IfcBuildingStorey the
        element was authored into. This is only a fallback: a door can be placed relative to a lower
        storey than the one it is contained in and stands on, so the relationships are read first.
        Guarded against cycles and against placements shared by several objects.
    """
    p = getattr(e, "ObjectPlacement", None)
    seen = set()
    hops = 0
    while p is not None and hops < max_hops:
        hops += 1
        pid = getattr(p, "id", lambda: None)()
        if pid in seen:
            break
        seen.add(pid)
        try:
            for placed in getattr(p, "PlacesObject", None) or []:
                if placed.is_a("IfcBuildingStorey"):
                    return placed.id()
        except Exception:
            pass
        p = getattr(p, "PlacementRelTo", None)
    return None


CONTAINED, AGGREGATED, NESTED = "IfcRelContainedInSpatialStructure", "IfcRelAggregates", "IfcRelNests"
VOIDS, FILLS = "IfcRelVoidsElement", "IfcRelFillsElement"


def storey_from_relationships(e, max_hops=16):
    """The storey the file's relationship tables put an element on, and the route taken.

    Follows how IFC places an element in the spatial hierarchy: containment in a storey,
    or in a space aggregated into a storey; aggregation or nesting into a parent that is
    itself placed; an opening's host element; a door or window's opening. Returns
    (storey step id, route), the route naming the relationships followed, such as
    "IfcRelVoidsElement > IfcRelContainedInSpatialStructure". Returns (None, reason)
    when the relationships stop above storey level or say nothing.
    """
    current, route, seen = e, [], set()
    while current is not None and len(route) < max_hops and current.id() not in seen:
        seen.add(current.id())
        contained = getattr(current, "ContainedInStructure", None) or ()
        if contained:
            structure, via = contained[0].RelatingStructure, route + [CONTAINED]
            while structure is not None and structure.is_a("IfcSpace") and len(via) < max_hops:
                structure, via = ue.get_aggregate(structure), via + [AGGREGATED]
            if structure is not None and structure.is_a("IfcBuildingStorey"):
                return structure.id(), " > ".join(via)
            return None, f"contained in {structure.is_a() if structure is not None else 'nothing'}"
        parent, link = ue.get_aggregate(current), AGGREGATED
        if parent is None:
            parent, link = ue.get_nest(current), NESTED
        if parent is not None:
            if parent.is_a("IfcBuildingStorey"):
                return parent.id(), " > ".join(route + [link])
            if parent.is_a("IfcSpatialStructureElement") and not parent.is_a("IfcSpace"):
                return None, f"aggregated into {parent.is_a()}"
            current, route = parent, route + [link]
            continue
        voids = getattr(current, "VoidsElements", None) or ()
        if voids:
            current, route = voids[0].RelatingBuildingElement, route + [VOIDS]
            continue
        fills = getattr(current, "FillsVoids", None) or ()
        if fills:
            current, route = fills[0].RelatingOpeningElement, route + [FILLS]
            continue
        return None, "no relationship"
    return None, "relationship chain too long or cyclic"


# storeys
def storeys_of(f, scale, elem_z):
    """Storeys from IfcBuildingStorey where the file has them, otherwise clustered from geometry.

        Some files have no usable storey assignment. The fallback groups elements by the height of
        their base and marks itself as inferred.
    """
    st = []
    for s in f.by_type("IfcBuildingStorey"):
        elev = getattr(s, "Elevation", None)
        st.append({"step_id": s.id(), "global_id": getattr(s, "GlobalId", None),
                   "name": s.Name or s.LongName or f"Storey {s.id()}",
                   "elevation_m": round(float(elev) * scale, 3) if elev is not None else None,
                   "source": "IfcBuildingStorey"})
    st = [s for s in st if s["elevation_m"] is not None]
    st.sort(key=lambda s: s["elevation_m"])
    if len(st) >= 1:
        return st, "IfcBuildingStorey"

    if not elem_z:
        return [], "none"
    zs = np.array(sorted(elem_z))
    band = 1.2
    groups, cur = [], [zs[0]]
    for z in zs[1:]:
        if z - cur[-1] <= band:
            cur.append(z)
        else:
            groups.append(cur)
            cur = [z]
    groups.append(cur)
    st = [{"step_id": None, "global_id": None, "name": f"Level {i}",
           "elevation_m": round(float(np.median(g)), 3), "source": "inferred_from_geometry"}
          for i, g in enumerate(groups)]
    return st, "inferred_from_geometry"


# rooms
def infer_rooms(barrier_polys, bounds, *, cell=0.12, min_area=1.5, erode=0, diagnostics=None):
    """Rooms as enclosed free space between barriers.

        Used when a storey has no IfcSpace. Barriers are rasterised, free space is flood-filled from
        outside the building, and every enclosed pocket left over becomes a room.

        Two things this has to get right. Doors and windows count as barriers, because each fills an
        opening cut through a wall and without them every room drains through its doorway and nothing
        encloses. And a cell counts as blocked when it touches a barrier rather than when its centre
        falls inside one, so a wall thinner than a cell still holds.
    """
    if not math.isfinite(cell) or cell <= 0 or not math.isfinite(min_area) or min_area <= 0 or erode < 0:
        raise ValueError("cell and minimum area must be finite and positive; erosion must be nonnegative")
    info = diagnostics if diagnostics is not None else {}
    info.update(status="no_barriers", cell_m=cell, minimum_area_m2=min_area, erosion_steps=erode)
    if not barrier_polys:
        return []
    from shapely.geometry import box
    from shapely.prepared import prep

    minx, miny, maxx, maxy = bounds
    if not all(math.isfinite(x) for x in bounds):
        raise ValueError("room bounds must be finite")
    W = int(math.ceil((maxx - minx) / cell)) + 3
    H = int(math.ceil((maxy - miny) / cell)) + 3
    if W * H > 6_000_000 or W < 3 or H < 3:
        info.update(status="grid_limit", grid_width=W, grid_height=H, grid_cells=W * H)
        return []
    info.update(status="no_enclosed_regions", grid_width=W, grid_height=H, grid_cells=W * H)

    # a half-cell buffer closes hairline joints between abutting barriers without
    # closing real openings, which are an order of magnitude wider
    exact = unary_union(barrier_polys)
    u = exact.buffer(cell * 0.5)
    pu = prep(u)

    grid = np.zeros((H, W), dtype=bool)
    for i in range(H):
        y0 = miny + i * cell
        for j in range(W):
            x0 = minx + j * cell
            if pu.intersects(box(x0, y0, x0 + cell, y0 + cell)):
                grid[i, j] = True

    free = ~grid

    # Rooms joined by a wide opening flood into one another and come out as one region. Eroding
    # the free space pinches off the narrow necks, so the cores can be labelled separately and
    # grown back. A doorway-width neck breaks; an open-plan room stays whole.
    # Off by default: on 12 models it was slightly worse overall (80.9% to 80.1% recall), since
    # splitting merged rooms gains about what destroying small ones loses. Use --erode to retry.
    def _erode(mask, n):
        m = mask.copy()
        for _ in range(n):
            e = m.copy()
            e[1:, :] &= m[:-1, :]
            e[:-1, :] &= m[1:, :]
            e[:, 1:] &= m[:, :-1]
            e[:, :-1] &= m[:, 1:]
            e[0, :] = e[-1, :] = e[:, 0] = e[:, -1] = False
            m = e
        return m

    def _label(mask):
        lab = np.zeros(mask.shape, dtype=np.int32)
        cur = 0
        for i in range(H):
            for j in range(W):
                if mask[i, j] and lab[i, j] == 0:
                    cur += 1
                    q = deque([(i, j)])
                    lab[i, j] = cur
                    while q:
                        y, x = q.popleft()
                        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and lab[ny, nx] == 0:
                                lab[ny, nx] = cur
                                q.append((ny, nx))
        return lab, cur

    from collections import deque
    # exterior first: whatever the flood reaches from the border is outside the building
    outside = np.zeros_like(free)
    q = deque()
    for i in range(H):
        for j in (0, W - 1):
            if free[i, j] and not outside[i, j]:
                outside[i, j] = True
                q.append((i, j))
    for j in range(W):
        for i in (0, H - 1):
            if free[i, j] and not outside[i, j]:
                outside[i, j] = True
                q.append((i, j))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and free[ny, nx] and not outside[ny, nx]:
                outside[ny, nx] = True
                q.append((ny, nx))

    interior = free & ~outside
    if not interior.any():
        return []

    cores = _erode(interior, erode)
    lab, ncore = _label(cores)
    if ncore == 0:                       # eroded everything away: fall back to whole regions
        lab, ncore = _label(interior)
        if ncore == 0:
            return []

    # grow the cores back out over the interior, nearest core wins
    grown = lab.copy()
    q = deque((i, j) for i in range(H) for j in range(W) if grown[i, j] > 0)
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and interior[ny, nx] and grown[ny, nx] == 0:
                grown[ny, nx] = grown[y, x]
                q.append((ny, nx))

    rooms = []
    for k in range(1, ncore + 1):
        cells = np.argwhere(grown == k)
        area = len(cells) * cell * cell
        if area < min_area:
            continue
        squares = [box(minx + j2 * cell, miny + i2 * cell,
                       minx + (j2 + 1) * cell, miny + (i2 + 1) * cell)
                   for i2, j2 in cells]
        poly = unary_union(squares).buffer(cell * .55).buffer(-cell * .55)
        if poly.is_empty:
            continue
        if isinstance(poly, MultiPolygon):
            poly = max(poly.geoms, key=lambda g: g.area)
        poly = _to_wall_faces(poly, exact, cell)
        rooms.append({"polygon": poly, "area_m2": round(poly.area, 2),
                      "raster_area_m2": round(area, 2)})
    rooms.sort(key=lambda r: -r["area_m2"])
    info.update(status="completed" if rooms else "below_minimum_area", rooms=len(rooms))
    return rooms


# main extraction
HAIRLINE_M = .01    # half the widest joint between abutting barriers that growth may not pass


def _to_wall_faces(poly, barriers, cell):
    """Put a rasterised room back on the wall faces it really has.

        The raster is deliberately pessimistic: barriers grow by half a cell and a cell blocks when it
        merely touches one, so thin walls cannot leak. That is right for deciding what encloses what
        and wrong for measuring it, costing about 0.15 m off every side. On a 1 m corridor that is a
        third of the area.

        So the raster decides which region is enclosed and the geometry decides where its edge lies.
        The region grows back past the pessimistic margin and is trimmed against the barriers
        themselves. Growth stops at the first real barrier, and where it crosses a thin partition the
        trim separates the pieces again and keeps the one the raster found.

        Abutting barriers rarely meet exactly. A door leaf and its frame, or two walls at a joint,
        leave gaps of a few millimetres, and growth passes through them onto the far face of the
        wall: into the recess behind a door leaf, or along the outside of an external wall. There it
        takes space from the room next door, and a door then has the same room on both sides. So
        the grown region is opened by a hairline first, which cuts those necks and lets the pieces
        separate like any other.
    """
    grown = poly.buffer(cell * 1.6).difference(barriers)
    if grown.is_empty:
        return poly
    grown = grown.buffer(-HAIRLINE_M, join_style=2).buffer(HAIRLINE_M, join_style=2)
    if grown.is_empty:
        return poly
    if isinstance(grown, MultiPolygon):
        grown = max(grown.geoms, key=lambda g: g.intersection(poly).area)
    return grown if grown.area >= poly.area else poly


def extract(path, *, verbose=False, threads=4):
    f = ifcopenshell.open(path)
    scale = uu.calculate_unit_scale(f)          # file units -> metres

    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)

    shapes, geometry_errors = {}, []
    it = ifcopenshell.geom.iterator(settings, f, threads)
    if it.initialize():
        while True:
            sh = it.get()
            try:
                # ifcopenshell.geom already normalises vertices to SI metres; only raw
                # IFC attributes (Elevation, quantities) need the file's unit scale.
                v = np.array(sh.geometry.verts, dtype=float).reshape(-1, 3)
                fc = np.array(sh.geometry.faces, dtype=int).reshape(-1, 3)
                if not np.isfinite(v).all():
                    raise ValueError("non-finite vertices")
                if len(v):
                    shapes[sh.id] = (v, [tuple(x) for x in fc])
            except Exception as ex:
                geometry_errors.append({"step_id": sh.id, "error": str(ex)})
            if not it.next():
                break

    # Storey membership, in order of authority: the relationship tables, then the placement chain,
    # then the elevation fallback in build.py. The placement chain alone put 226 doors and windows
    # in 19 models a storey or more too low, where the relationships matched their real height.
    in_storey, storey_src = {}, {}
    for e in f.by_type("IfcProduct"):
        if e.is_a() in ("IfcSite", "IfcBuilding", "IfcBuildingStorey"):
            continue
        try:
            sid, route = storey_from_relationships(e)
        except Exception:
            sid, route = None, None
        if sid is None:
            sid, route = storey_from_placement(e), "ObjectPlacement"
        if sid is not None:
            in_storey[e.id()] = sid
            storey_src[e.id()] = route

    elem_z = [v[:, 2].min() for sid, (v, _) in shapes.items()
              if f.by_id(sid).is_a() not in NON_FABRIC]
    storeys, storey_source = storeys_of(f, scale, elem_z)

    # elements
    elements = []
    for e in f.by_type("IfcProduct"):
        cls = e.is_a()
        if cls in ("IfcSite", "IfcBuilding", "IfcBuildingStorey", "IfcProject"):
            continue
        sid = e.id()
        geo = shapes.get(sid)
        rec = {
            "step_id": sid,
            "global_id": getattr(e, "GlobalId", None),
            "ifc_class": cls,
            "name": e.Name,
            "object_type": getattr(e, "ObjectType", None),
            "description": getattr(e, "Description", None),
            "has_geometry": geo is not None,
        }
        # type + material + every property set and quantity, whatever the schema
        try:
            t = ue.get_type(e)
            rec["type_name"] = t.Name if t is not None else None
            rec["type_class"] = t.is_a() if t is not None else None
        except Exception:
            rec["type_name"] = rec["type_class"] = None
        try:
            rec["property_sets"] = ue.get_psets(e)
        except Exception:
            rec["property_sets"] = {}
        try:
            mat = ue.get_material(e)
            rec["material"] = getattr(mat, "Name", None) if mat is not None else None
        except Exception:
            rec["material"] = None
        # IfcSpace usually carries the name a person would use in LongName ("Lobby") while
        # Name holds a number ("1"), so LongName is preferred where it exists.
        rec["long_name"] = getattr(e, "LongName", None)
        rec["label"] = (rec["long_name"] or rec["name"] or rec["type_name"] or rec["object_type"]
                        or f"{cls}#{sid}")

        if geo is not None:
            v, fc = geo
            lo, hi = v.min(axis=0), v.max(axis=0)
            rec["bbox_min"] = [round(float(x), 3) for x in lo]
            rec["bbox_max"] = [round(float(x), 3) for x in hi]
            rec["centroid"] = [round(float((lo[i] + hi[i]) / 2), 3) for i in range(3)]
            rec["dimensions"] = [round(float(hi[i] - lo[i]), 3) for i in range(3)]
            rec["_verts"], rec["_faces"] = v, fc
        rec["storey_step_id"] = in_storey.get(sid)
        rec["storey_link"] = storey_src.get(sid)
        elements.append(rec)

    # relations read from the file itself
    rel = []
    o2h = {}
    for r in f.by_type("IfcRelVoidsElement"):
        try:
            o2h[r.RelatedOpeningElement.id()] = r.RelatingBuildingElement.id()
        except Exception:
            pass
    for r in f.by_type("IfcRelFillsElement"):
        try:
            o = r.RelatingOpeningElement.id()
            if o in o2h:
                rel.append({"type": "hosts", "source": o2h[o],
                            "target": r.RelatedBuildingElement.id(),
                            "via_opening": o, "source_of_truth": "IfcRelVoids/FillsElement"})
        except Exception:
            pass
    # every opening and the element it is cut through, filled or not: an archway or a hatch is a
    # hole in its wall that no door or window closes
    for o, h in o2h.items():
        rel.append({"type": "voids", "source": h, "target": o, "source_of_truth": "IfcRelVoidsElement"})
    for r in f.by_type("IfcRelAggregates"):
        try:
            p = r.RelatingObject
            if p.is_a("IfcElement"):
                for c in r.RelatedObjects:
                    rel.append({"type": "part_of", "source": c.id(), "target": p.id(),
                                "source_of_truth": "IfcRelAggregates"})
        except Exception:
            pass

    return {"file": path, "schema": f.schema, "unit_scale_to_m": scale,
            "storeys": storeys, "storey_source": storey_source,
            "elements": elements, "relations": rel,
            "spaces": [s.id() for s in f.by_type("IfcSpace")],
            "geometry_errors": geometry_errors}, f
