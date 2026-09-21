"""One-to-one correspondence between pipeline and reference-plan storeys."""
from collections import defaultdict
import math


METHOD = 'unique IFC identity first; unique rounded elevation fallback; never reuse a storey'


def pair_storeys(ours, theirs):
    """Return reference-list-index -> pipeline-list-index and match diagnostics.

    Repeated elevations are common for structural/datum levels. An elevation
    dictionary silently overwrites those levels. IFC identity takes priority;
    fallback is allowed only when the remaining elevation group is unambiguous
    and at least one side has no identity. Distinct explicit identities do not
    become the same storey merely because they share an elevation.
    """
    def gid(s):
        return s.get('global_id') or s.get('gid')

    def elevation(s):
        value = s.get('elevation_m', s.get('elevation'))
        return round(value, 2) if isinstance(value, (int, float)) and math.isfinite(value) else None

    def grouped(rows, key, excluded=()):
        groups = defaultdict(list)
        for index, row in enumerate(rows):
            if index not in excluded and (value := key(row)) is not None:
                groups[value].append(index)
        return groups

    a, b = grouped(ours, gid), grouped(theirs, gid)
    pairs = {b[key][0]: a[key][0] for key in a.keys() & b.keys()
             if len(a[key]) == len(b[key]) == 1}
    identity_count = len(pairs)
    a = grouped(ours, elevation, pairs.values())
    b = grouped(theirs, elevation, pairs)
    for key in a.keys() & b.keys():
        if len(a[key]) == len(b[key]) == 1:
            ai, bi = a[key][0], b[key][0]
            if not gid(ours[ai]) or not gid(theirs[bi]):
                pairs[bi] = ai
    return pairs, {'identity_matches': identity_count,
                   'elevation_fallback_matches': len(pairs) - identity_count,
                   'unmatched_ours': sorted(set(range(len(ours))) - set(pairs.values())),
                   'unmatched_theirs': sorted(set(range(len(theirs))) - set(pairs))}
