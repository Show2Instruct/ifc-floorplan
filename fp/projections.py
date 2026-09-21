"""Whole-element silhouettes, and the geometry a floor plan's cut actually passes through."""
import math
from collections import Counter

from fp.band_projection import project_band
from fp.extract import footprint, NON_FABRIC

# Columns bound a room as firmly as a wall does. Leave them out and rooms leak through the
# gaps between curtain-wall panels, and frames that define rooms with columns produce nothing.
BARRIERS = ('IfcWall', 'IfcCurtainWall', 'IfcDoor', 'IfcWindow', 'IfcPlate', 'IfcColumn')
MIN_CUT_M = .3          # below this, a section describes the level above more than this one
EMPTY_STATUSES = {'outside_band', 'boundary_contact_no_area', 'empty_geometry'}


def barrier_roles(data, classes=BARRIERS):
    """Assembly parts inherit their parent's barrier role through IFC aggregation."""
    by_id = {e['step_id']: e for e in data['elements']}
    children = {}
    for relation in data.get('relations', []):
        if relation['type'] == 'part_of':
            children.setdefault(relation['target'], []).append(relation['source'])
    roles = {sid for sid, e in by_id.items() if e['ifc_class'].startswith(tuple(classes))}
    pending = list(roles)
    while pending:
        parent = pending.pop()
        for child in children.get(parent, []):
            if child in by_id and child not in roles and by_id[child]['ifc_class'] not in NON_FABRIC:
                roles.add(child)
                pending.append(child)
    return roles, children


def barrier_polygons(visible, data):
    """Barrier outlines for room inference: walls, plus whatever fills the openings in them."""
    return [e['_poly'] for e in visible if e['_is_barrier'] and e.get('_poly') is not None], {}


def full_footprints(data, assignments):
    """One whole-element silhouette, independent of storey ownership and cut height."""
    for element in data['elements']:
        if element['step_id'] not in assignments or '_verts' not in element:
            continue
        poly, method = footprint(element['_verts'], element['_faces'])
        element['_poly'], element['_poly_method'] = poly, method
        element['footprint_projection'] = {'height_scope': 'full_height',
                                           'purpose': 'whole_element_silhouette', 'reason': None}


def section_band(storeys, index, cut_height, cut_thickness, *, band_base=None, min_cut=MIN_CUT_M):
    """Where to cut this storey's plan, kept below the next declared level.

    A plan is normally cut about 1.2 m above the floor. Structural and datum levels break that:
    where they stand a few centimetres apart, a 1.2 m cut reads the storey above and reports its
    rooms on a level holding only beams. The cut then drops to the middle of the gap, and where
    that leaves too little height the storey gets no section instead of a copy of the level above.

    With `band_base` the barriers come from everything between that height and the cut, rather
    than from a thin slice at it.

    Returns (low, high, cut height above the storey, note).
    """
    elevation = storeys[index]['elevation_m']
    gap = storeys[index + 1]['elevation_m'] - elevation if index + 1 < len(storeys) else None
    note, top = None, cut_height
    if gap is not None and gap <= cut_height + cut_thickness:
        top = gap / 2
        if top < min_cut or top <= cut_thickness:
            return None, None, top, (f'levels_too_close_for_a_section; the next level is {gap:.3f} m '
                                     f'above, leaving {top:.3f} m (minimum {min_cut} m)')
        note = f'cut_lowered_to_{top:.3f}_m_to_stay_below_the_next_level, which is {gap:.3f} m above'
    if band_base is None:
        return elevation + top, elevation + top + cut_thickness, top, note
    if top - band_base < cut_thickness:
        return None, None, top, (f'band_base_{band_base} m leaves no height below a cut at {top:.3f} m')
    return elevation + band_base, elevation + top, top, note


def section_elements(data, assignments, index, *, cut_height=1.2, cut_thickness=.01,
                     band_base=None, barrier_classes=BARRIERS, barriers_only=False):
    if (not math.isfinite(cut_height) or cut_height < 0 or not math.isfinite(cut_thickness)
            or cut_thickness <= 0):
        raise ValueError('cut height must be finite and nonnegative; cut thickness must be finite and positive')
    storeys = data['storeys']
    low, high, effective_height, note = section_band(storeys, index, cut_height, cut_thickness,
                                                     band_base=band_base)
    if low is None:
        return [], {'mode': 'horizontal_height_band', 'status': 'no_section_band', 'reason': note,
                    'cut_height_above_storey_m': None, 'band_m': None, 'bounds_inclusive': True,
                    'counts': {}, 'excluded': [], 'fallbacks': [], 'warnings': [note],
                    'barrier_roles': 'class or inherited IFC assembly role',
                    'barrier_classes': list(barrier_classes),
                    'available_barrier_meshes_projected': False, 'barriers_complete': False,
                    'barriers_all_sectioned': False}
    visible, exclusions, fallbacks, counts = [], [], [], Counter()
    roles, children = barrier_roles(data, barrier_classes)
    for element in data['elements']:
        cls = element['ifc_class']
        is_barrier = element['step_id'] in roles
        if cls in NON_FABRIC or (barriers_only and not is_barrier):
            continue
        owner = assignments.get(element['step_id'], (None,))[0]
        if '_verts' not in element:
            if owner == index or (is_barrier and owner is None):
                if children.get(element['step_id']):
                    counts['assembly_geometry_delegated_to_parts'] += 1
                else:
                    counts['missing_mesh'] += 1
                    exclusions.append({'step_id': element['step_id'], 'ifc_class': cls,
                                       'owner_storey_index': owner,
                                       'projection': {'status': 'missing_mesh',
                                                      'height_relevance': 'unknown_without_geometry'},
                                       'affects_room_barrier': is_barrier})
            continue
        verts = element['_verts']
        if verts[:, 2].max() < low or verts[:, 2].min() > high:
            if owner == index:
                counts['owned_outside_band'] += 1
            continue
        poly, info = project_band(verts, element['_faces'], low, high)
        counts[info['status']] += 1
        if poly is None:
            if info['status'] in EMPTY_STATUSES:
                continue
            # A barrier the projector cannot rebuild must not leave a hole in the wall it fills,
            # or the rooms on both sides drain into one another. Where it belongs to this storey,
            # its whole-element silhouette stands in and is recorded as a fallback.
            silhouette = element.get('_poly')
            if owner == index and silhouette is not None:
                counts['full_height_fallback'] += 1
                fallbacks.append({'step_id': element['step_id'], 'ifc_class': cls,
                                  'owner_storey_index': owner, 'projection': info,
                                  'affects_room_barrier': is_barrier})
                visible.append({**element, '_poly': silhouette,
                                '_poly_method': element.get('_poly_method'),
                                'owner_storey_index': owner, '_is_barrier': is_barrier,
                                'section_projection': {**info, 'fallback': 'full_height_silhouette'}})
                continue
            exclusions.append({'step_id': element['step_id'], 'ifc_class': cls,
                               'owner_storey_index': owner, 'projection': info,
                               'affects_room_barrier': is_barrier})
            continue
        visible.append({**element, '_poly': poly, '_poly_method': info.get('method'),
                        'owner_storey_index': owner, 'section_projection': info, '_is_barrier': is_barrier})
    warnings = []
    if index + 1 < len(storeys) and high > storeys[index+1]['elevation_m']:
        warnings.append('cut_crosses_next_declared_level; review structural or closely spaced levels')
    if note:
        warnings.append(note)
    diagnostic = {'mode': 'horizontal_height_band', 'status': 'sectioned',
                  'cut_height_above_storey_m': effective_height,
                  'requested_cut_height_m': cut_height, 'band_base_m': band_base,
                  'barrier_classes': list(barrier_classes),
                  'band_m': [low, high], 'bounds_inclusive': True, 'counts': dict(counts),
                  'excluded': exclusions, 'fallbacks': fallbacks,
                  'barriers_all_sectioned': not any(f['affects_room_barrier'] for f in fallbacks),
                  'warnings': warnings,
                  'barrier_roles': 'class or inherited IFC assembly role',
                  'available_barrier_meshes_projected': not any(e['affects_room_barrier']
                      and e['projection']['status'] != 'missing_mesh' for e in exclusions),
                  'barriers_complete': not any(e['affects_room_barrier'] for e in exclusions)}
    return visible, diagnostic
