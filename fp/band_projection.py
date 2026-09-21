"""Project a mesh onto the plan after clipping it to a height band.

The band is cut as a solid between two heights instead of intersecting a single plane, so
vertical prisms keep their interiors and their holes. Lengths and tolerances are in metres.
"""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize_full, unary_union

from fp.extract import footprint, _union

SNAP_M = 1e-6
EPS_M = 1e-9
MAX_BAND_TRIANGLES = 100_000


def clip_polygon(points, height, above):
    """Sutherland-Hodgman clipping of a convex 3D face against one z plane."""
    result = []
    if not len(points):
        return result
    previous = np.asarray(points[-1], dtype=float)
    previous_inside = previous[2] >= height if above else previous[2] <= height
    for point in points:
        point = np.asarray(point, dtype=float)
        inside = point[2] >= height if above else point[2] <= height
        if inside != previous_inside:
            t = (height - previous[2]) / (point[2] - previous[2])
            cut = previous + t * (point - previous)
            cut[2] = height
            result.append(cut)
        if inside:
            result.append(point)
        previous, previous_inside = point, inside
    return result


def plane_segment(triangle, height):
    """One oriented segment where a triangle crosses the plane.

    Direction follows z cross face normal, so a solid with outward faces winds positively seen
    from above and a cavity winds negatively. Coplanar faces contribute their projected area.
    """
    if (triangle[:, 2].min() > height or triangle[:, 2].max() < height
            or np.all(triangle[:, 2] == height)):
        return None
    hits = []
    for a, b in zip(triangle, np.roll(triangle, -1, axis=0)):
        da, db = a[2] - height, b[2] - height
        if abs(da) <= EPS_M:
            hits.append(a[:2])
        if da * db < 0:
            hits.append((a + da / (da - db) * (b - a))[:2])
    unique = sorted({tuple(np.rint(point / SNAP_M) * SNAP_M) for point in hits})
    if len(unique) < 2:
        return None
    # One triangle gives one segment. Take the extremes in case roundoff leaves several
    # near-identical points at an endpoint.
    a, b = max(((a, b) for a in unique for b in unique),
               key=lambda pair: sum((x-y)**2 for x, y in zip(*pair)))
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    direction = np.array([-normal[1], normal[0]])
    if np.dot(np.subtract(b, a), direction) < 0:
        a, b = b, a
    return LineString([a, b]) if a != b else None


def cap_polygons(segments):
    """Close the cut contours into filled faces, using nonzero winding.

    Parity cannot tell a cavity from a solid island sitting inside it. Face orientation can:
    equal orientations add, opposite ones subtract. Mixed signs mean the source shells disagree,
    so the caller is told rather than handed a cancelled-out hole.
    """
    if not segments:
        return [], {'faces': 0, 'kept_faces': 0, 'cuts': 0, 'dangles': 0, 'invalid_rings': 0,
                    'unbalanced_vertices': 0, 'mixed_winding_signs': False,
                    'fill_rule': 'nonzero_oriented_winding'}
    polygons, cuts, dangles, invalid = polygonize_full(unary_union(segments))
    faces = list(polygons.geoms)
    info = {'faces': len(faces), 'cuts': len(cuts.geoms), 'dangles': len(dangles.geoms),
            'invalid_rings': len(invalid.geoms), 'fill_rule': 'nonzero_oriented_winding'}
    # Polygonizing nodes and merges the lines, which loses their direction and how many times
    # they coincide, so count the winding from the original segments.
    edges = np.array([segment.coords for segment in segments], dtype=float)
    balance = {}
    for start, end in edges:
        for point, increment in ((start, 1), (end, -1)):
            key = tuple(point)
            balance[key] = balance.get(key, 0) + increment
    info['unbalanced_vertices'] = sum(value != 0 for value in balance.values())
    start, end = edges[:, 0], edges[:, 1]
    kept = []
    windings = []
    for face in faces:
        point = face.representative_point()
        px, py = point.x, point.y
        cross = ((end[:, 0] - start[:, 0]) * (py - start[:, 1])
                 - (end[:, 1] - start[:, 1]) * (px - start[:, 0]))
        upward = (start[:, 1] <= py) & (end[:, 1] > py) & (cross > 0)
        downward = (start[:, 1] > py) & (end[:, 1] <= py) & (cross < 0)
        winding = int(np.count_nonzero(upward) - np.count_nonzero(downward))
        windings.append(winding)
        if winding != 0:
            kept.append(face)
    info['mixed_winding_signs'] = any(w > 0 for w in windings) and any(w < 0 for w in windings)
    info['winding_range'] = [min(windings, default=0), max(windings, default=0)]
    info['kept_faces'] = len(kept)
    return kept, info


def plane_segments(vertices, faces, height, *, diagnostics=None):
    """Every contour where the mesh crosses one plane.

    An edge shared by a face above and a face below is a real cut even when neither face
    crosses on its own. Edges that only touch the plane from one side are not. Snapped
    coordinates also join vertices duplicated across separate face buffers.
    """
    segments, contact_edges = [], {}
    triangles = vertices[faces]
    dz = triangles[:, :, 2] - height
    relevant = (dz.min(axis=1) <= EPS_M) & (dz.max(axis=1) >= -EPS_M)
    for triangle, distances in zip(triangles[relevant], dz[relevant]):
        zero = np.abs(distances) <= EPS_M
        triangle = triangle.copy()
        triangle[zero, 2] = height
        if (distances < -EPS_M).any() and (distances > EPS_M).any():
            segment = plane_segment(triangle, height)
            if segment is not None:
                segments.append(segment)
        elif zero.sum() == 2:
            points = sorted({tuple(np.rint(p / SNAP_M) * SNAP_M) for p in triangle[zero, :2]})
            if len(points) == 2:
                key = tuple(points)
                side = 1 if distances[~zero][0] > 0 else -1
                segment = plane_segment(triangle, height)
                direction = 1 if tuple(segment.coords[0]) == key[0] else -1
                counts = contact_edges.setdefault(key, {-1: [], 1: []})
                counts[side].append(direction)
    inconsistent = 0
    for edge, sides in contact_edges.items():
        if not sides[-1] or not sides[1]:
            continue
        # A real shared cut edge has matching directions on both sides. Keep the multiplicity
        # for coincident shells, but count two faces meeting at one edge once. Opposite signs
        # are a defect in the mesh.
        if sorted(sides[-1]) != sorted(sides[1]):
            inconsistent += 1
        for direction in sides[-1]:
            segments.append(LineString(edge if direction > 0 else edge[::-1]))
    if diagnostics is not None:
        diagnostics['inconsistent_contact_edges'] = inconsistent
    return segments


def project_band(vertices, faces, low, high):
    """Return (polygon, diagnostics) for the part of a mesh inside [low, high].

    Bounds are inclusive, so a horizontal face resting on one of them contributes its area.
    Nothing outside the band is substituted in. Where the cut contours do not close, the
    result is no polygon and a reason, never a partial area.
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces)
    if not math.isfinite(low) or not math.isfinite(high) or low >= high:
        raise ValueError('band bounds must be finite and strictly increasing')
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError('vertices must be finite Nx3 coordinates')
    if faces.size == 0:
        faces = np.empty((0, 3), dtype=int)
    if (faces.ndim != 2 or faces.shape[1] != 3
            or not np.issubdtype(faces.dtype, np.integer)
            or (faces.size and (faces.min() < 0 or faces.max() >= len(vertices)))):
        raise ValueError('faces must be integer Mx3 indices within vertices')
    if not len(vertices):
        return None, {'status': 'empty_geometry'}
    if vertices[:, 2].max() < low or vertices[:, 2].min() > high:
        return None, {'status': 'outside_band'}
    if not len(faces):
        return None, {'status': 'missing_faces'}
    triangles = vertices[faces]
    intersects = ((triangles[:, :, 2].max(axis=1) >= low - EPS_M)
                  & (triangles[:, :, 2].min(axis=1) <= high + EPS_M))
    relevant_faces = faces[intersects]
    # A sloping face can touch a bound along one edge, and floating point noise in the IFC can
    # put that edge a fraction of a nanometre inside the band. It has no plan area, so do not
    # report the tangent line as a broken cap.
    contact = None
    if vertices[:, 2].max() <= low + EPS_M:
        contact = low
    elif vertices[:, 2].min() >= high - EPS_M:
        contact = high
    if contact is not None:
        if len(relevant_faces) > MAX_BAND_TRIANGLES:
            return None, {'status': 'unsupported_dense_contact', 'triangles': len(faces),
                          'band_triangles': len(relevant_faces), 'triangle_budget': MAX_BAND_TRIANGLES}
        parts = []
        for face in relevant_faces:
            triangle = vertices[face]
            if np.all(np.abs(triangle[:, 2] - contact) <= EPS_M):
                polygon = Polygon(triangle[:, :2])
                if polygon.is_valid and polygon.area > EPS_M:
                    parts.append(polygon)
        result, method = _union(parts) if parts else (None, None)
        return result, {'status': 'boundary_contact_area' if result is not None else 'boundary_contact_no_area',
                        'method': method, 'z_tolerance_m': EPS_M}
    if vertices[:, 2].min() >= low and vertices[:, 2].max() <= high:
        poly, method = footprint(vertices, [tuple(f) for f in faces])
        return poly, {'status': 'fully_inside_band', 'method': method}
    if len(relevant_faces) > MAX_BAND_TRIANGLES:
        return None, {'status': 'unsupported_dense_cut', 'triangles': len(faces),
                      'band_triangles': len(relevant_faces), 'triangle_budget': MAX_BAND_TRIANGLES}
    projected = []
    segments, segment_info = {}, {}
    for plane in (low, high):
        segment_info[plane] = {}
        segments[plane] = plane_segments(vertices, relevant_faces, plane,
                                         diagnostics=segment_info[plane])
    for face in relevant_faces:
        triangle = vertices[face]
        clipped = clip_polygon(clip_polygon(triangle, low, True), high, False)
        if len(clipped) >= 3:
            polygon = Polygon([point[:2] for point in clipped])
            if polygon.is_valid and polygon.area > EPS_M:
                projected.append(polygon)
    triangle_parts = len(projected)
    contours = {}
    for plane, contour in segments.items():
        caps, info = cap_polygons(contour)
        info.update(segment_info[plane])
        projected.extend(caps)
        contours[str(plane)] = info
    diagnostic = {'contours': contours, 'cut_segments': sum(map(len, segments.values())),
                  'cap_snap_m': SNAP_M, 'projected_triangle_parts': triangle_parts,
                  'source_triangles': len(faces), 'band_triangles': len(relevant_faces),
                  'triangle_budget': MAX_BAND_TRIANGLES}
    if any(info[key] for info in contours.values() for key in ('cuts', 'dangles', 'invalid_rings')):
        return None, {'status': 'incomplete_cut_contours', **diagnostic}
    if any(info[key] for info in contours.values()
           for key in ('unbalanced_vertices', 'mixed_winding_signs', 'inconsistent_contact_edges')):
        return None, {'status': 'inconsistent_cut_orientation', **diagnostic}
    if not projected:
        return None, {'status': 'no_projected_area', **diagnostic}
    result, method = _union(projected)
    return result, {'status': 'clipped_band' if result is not None else 'union_failed',
                    'method': method, **diagnostic}
