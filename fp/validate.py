#!/usr/bin/env python
"""Validation measures for the generated floor plans, each reported with its counts.

There is no ground-truth floor plan for these models, so the pipeline is checked against what the
IFC file itself records. The quantity and held-out room checks use references that did not
produce the values being compared. These are internal consistency checks, not accuracy against a
surveyed building. The document named in DEFINITIONS_DOC defines every measure: what is counted,
what it is divided by, what is left out and what it cannot show.

  footprint_area       plan footprint against the declared footprint or slab area
  volume               footprint area x height against the declared volume
  storey_links         elements whose storey the file states, by route
  placement_agreement  placement-chain storey against the relationship-table storey
  elevation_fallback   storey guessed from height, scored against the stated storey
  storey_assignment    elements with geometry placed on a storey (complete by design)
  geometry_extraction  elements with a representation that produced geometry
  room_inference       rooms inferred from barriers against IfcSpace, at IoU 0.1 and 0.5

Every rate carries its numerator and denominator. Pooled values count elements, rooms or matches
across the whole run, with per-model values beside them. legacy_1_sep repeats the formulas behind
an earlier set of results so old and new numbers can be compared; it is not a measure to quote.
Declared quantities are converted to SI with their own unit (fp.quantities).

A run writes into its own directory, like run_pipeline.py: validation.json beside run.json,
source.zip and environment-freeze.txt. Without --models or --n every input is validated.

    python -m fp.validate --models model_100 model_119 --out runs/<name>
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fp import runs
from fp.build import assign_storeys, storey_bands
from fp.extract import extract, infer_rooms, storey_from_placement
from fp.projections import BARRIERS, barrier_polygons, full_footprints, section_elements
from fp.quantities import declared_quantity

DEFINITIONS_VERSION = 6
DEFINITIONS_DOC = "docs/validation-metrics.md"

BARRIER = BARRIERS
# Only prismatic classes are compared on volume: footprint x height means nothing for a stair
# or a curved roof. Prefix match, so IfcWallStandardCase and similar subtypes are included.
PRISM = ("IfcWall", "IfcColumn", "IfcSlab", "IfcBeam", "IfcFooting", "IfcPlate")
# Which declared quantity compares with a projected footprint depends on the class. Slabs can
# record surface area, which need not equal plan area on a slope. Walls record
# GrossFootprintArea where the authoring tool wrote one. Length x Width is deliberately not a
# fallback: for a slab "Width" is total layer thickness, and for a wall it is nominal length
# ignoring openings, so neither describes the solid seen from above.
AREA_KEYS = ("GrossFootprintArea", "NetFootprintArea")
# NetArea before GrossArea for slabs: GrossArea counts the slab as if solid, while our
# footprint excludes stair voids and shafts cut through it. On 78 slabs, NetArea cut the share
# diverging by more than 10% from 46.2% to 30.2%. What remains is bimodal rather than noisy,
# most slabs matching exactly and a few differing a lot, which points at a categorical cause
# such as sloped or multi-part slabs.
SLAB_KEYS = ("NetArea", "GrossArea")
VOL_KEYS = ("GrossVolume", "NetVolume")
MIN_DECLARED_AREA_M2 = 0.01
MIN_DECLARED_VOLUME_M3 = 0.01
MIN_HEIGHT_M = 0.01
ERROR_THRESHOLDS = (0.05, 0.10)
IOU_THRESHOLDS = (0.1, 0.5)
LEGACY_IOU = 0.1


# declared quantities
def declared_footprint_area(e):
    """Plan area as the authoring tool recorded it, converted to square metres.

    Returns the quantity record from fp.quantities. Only status "ok" carries a usable
    value_si. Any other status (no comparable quantity, a missing or unsupported unit, an
    ambiguous or invalid quantity) excludes the element and is counted, rather than
    comparing the footprint against a number that measures something else or sits in an
    unknown unit."""
    cls = e.is_a()
    keys = SLAB_KEYS if ("Slab" in cls or "Roof" in cls or "Covering" in cls) else AREA_KEYS
    return declared_quantity(e, keys, "AREAUNIT")


def declared_volume(e):
    """Volume as the authoring tool recorded it, converted to cubic metres. Same record and
    exclusion rule as declared_footprint_area."""
    return declared_quantity(e, VOL_KEYS, "VOLUMEUNIT")


def quantity_or_error(lookup, element):
    try:
        return lookup(element)
    except Exception as ex:
        return {"status": "error", "reason": f"{type(ex).__name__}: {ex}"}


# counting helpers
def ratio(numerator, denominator):
    """A rate that always carries its counts; value is None when nothing was eligible."""
    return {"numerator": numerator, "denominator": denominator,
            "value": round(numerator / denominator, 4) if denominator else None}


def median_or_none(values, ndigits=4):
    values = list(values)
    return round(statistics.median(values), ndigits) if values else None


def error_summary(errors):
    """Relative errors pooled over the items given: count, median, share within thresholds."""
    out = {"n": len(errors), "median": median_or_none(errors)}
    for t in ERROR_THRESHOLDS:
        out[f"within_{round(t * 100)}pct"] = ratio(sum(1 for x in errors if x <= t), len(errors))
    return out


def sorted_counts(counter):
    """Largest first, ties by name, so output order never depends on insertion or hashing."""
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


# rooms
def iou(a, b):
    """Intersection over union, or None when the geometry operation fails."""
    try:
        union = a.union(b).area
        overlap = a.intersection(b).area / union if union > 0 else 0.0
        return overlap if math.isfinite(overlap) and 0 <= overlap <= 1 else None
    except Exception:
        return None


def iou_matrix(truth, inferred):
    return [[iou(t, p) for p in inferred] for t in truth]


def match_rooms(matrix, threshold):
    """Maximise match count, then total IoU, over eligible one-to-one pairs.

    Scores are invariant to row/column permutations, including exact IoU ties. Several
    equally optimal pair identities may exist. Integer weights represent the supplied
    binary floats exactly, avoiding floating-point tie errors in weighted matching.
    Returns (truth index, inferred index, iou), ordered by indices for readable output.
    """
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("IoU threshold must be finite and between 0 and 1")
    if matrix and any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError("IoU matrix must be rectangular")
    candidates = []
    for ti, row in enumerate(matrix):
        for pi, value in enumerate(row):
            if value is None:
                continue
            value = float(value)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("IoU values must be finite and between 0 and 1, or None")
            if value > 0 and value >= threshold:
                numerator, denominator = value.as_integer_ratio()
                candidates.append((ti, pi, numerator, denominator))
    if not candidates:
        return []
    # IEEE float denominators are powers of two; their maximum is a common multiple.
    common = max(denominator for _, _, _, denominator in candidates)
    graph = nx.Graph()
    for ti, pi, numerator, denominator in candidates:
        graph.add_edge((0, ti), (1, pi), weight=numerator * (common // denominator))
    matched = nx.max_weight_matching(graph, maxcardinality=True)
    out = []
    for a, b in matched:
        truth, inferred = (a, b) if a[0] == 0 else (b, a)
        ti, pi = truth[1], inferred[1]
        out.append((ti, pi, float(matrix[ti][pi])))
    return sorted(out)


def legacy_match(matrix, threshold=LEGACY_IOU):
    """The earlier rule, kept only to reproduce results saved before it changed.

        Truth rooms in list order, each taking the unused inferred room with the highest IoU, matched
        if IoU > threshold. Its result can change with the order the truth rooms are listed in.
    """
    used, out = set(), []
    for ti, row in enumerate(matrix):
        best, bi = 0.0, None
        for pi, v in enumerate(row):
            v = v or 0.0
            if pi not in used and v > best:
                best, bi = v, pi
        if bi is not None and best > threshold:
            used.add(bi)
            out.append((ti, bi, best))
    return out


def room_test(truth, inferred):
    """Compare one storey's inferred rooms with its IfcSpace footprints."""
    matrix = iou_matrix(truth, inferred)
    try:
        overlap = max(0.0, math.fsum(t.area for t in truth) - unary_union(truth).area)
    except Exception:
        overlap = None
    record = {"n_truth": len(truth), "n_inferred": len(inferred),
              "area_truth_m2": math.fsum(t.area for t in truth),
              "area_inferred_m2": math.fsum(p.area for p in inferred),
              "truth_overlap_m2": None if overlap is None else round(overlap, 3),
              "iou_failures": sum(1 for row in matrix for v in row if v is None),
              "by_iou_threshold": {}}
    for threshold in IOU_THRESHOLDS:
        matches = match_rooms(matrix, threshold)
        record["by_iou_threshold"][str(threshold)] = {
            "matched": len(matches), "iou_sum": math.fsum(v for *_, v in matches)}
    old = legacy_match(matrix)
    record["legacy_1_sep"] = {
        "matched": len(old),
        "mean_iou": round(statistics.mean(v for *_, v in old), 3) if old else 0.0,
        "area_truth": round(sum(t.area for t in truth), 1),
        "area_inferred": round(sum(p.area for p in inferred), 1) if inferred else 0.0}
    return record


def infer_and_test_rooms(truth, barriers, options):
    """Keep eligible reference rooms in recall even when inference cannot complete.

    Precision still describes emitted rooms only. The completion rate and failure
    details must accompany it: a failed storey cannot be interpreted as a successful
    negative prediction or silently removed from the evaluation sample.
    """
    diagnostics = {}
    try:
        bounds = unary_union(barriers).bounds if barriers else (0., 0., 1., 1.)
        inferred = [room["polygon"] for room in infer_rooms(
            barriers, bounds, diagnostics=diagnostics, **options)]
        failed = diagnostics.get("status") == "grid_limit"
    except Exception as ex:
        inferred, failed = [], True
        diagnostics.update(status="error", error=f"{type(ex).__name__}: {ex}")
    return {**room_test(truth, inferred), "status": "failed" if failed else "completed",
            "inference_diagnostics": diagnostics}


# storeys
def elevation_guess(z, bands):
    """The storey the elevation fallback picks for an element whose base is at height z:
    the band containing z, else the nearest end. Same rule as build.assign_storeys.
    Returns (storey index, whether the nearest-end clamp was used)."""
    for i, (lo, hi) in enumerate(bands):
        if lo <= z < hi:
            return i, False
    return (0 if z < bands[0][0] else len(bands) - 1), True


def storey_measures(f, data, assign, storeys, bands):
    elements = data["elements"]
    with_geometry = [e for e in elements if e["has_geometry"]]
    index = {s["step_id"]: i for i, s in enumerate(storeys) if s["step_id"] is not None}

    routes = Counter(e["storey_link"] or "none" for e in elements)
    declared_all = sum(1 for e in elements if e["storey_link"])
    declared_geometry = [e for e in with_geometry if e["storey_link"]]

    # the two relationship tables, read independently of the placement chain
    contained, aggregated = {}, {}
    for rel, parent, children, table in (
            ("IfcRelContainedInSpatialStructure", "RelatingStructure", "RelatedElements", contained),
            ("IfcRelAggregates", "RelatingObject", "RelatedObjects", aggregated)):
        for r in f.by_type(rel):
            s = getattr(r, parent, None)
            if s is not None and s.is_a("IfcBuildingStorey"):
                for child in getattr(r, children, None) or ():
                    table.setdefault(child.id(), s.id())

    both = agree = 0
    disagreements = []
    for e in elements:
        placed = storey_from_placement(f.by_id(e["step_id"]))
        stated = contained.get(e["step_id"], aggregated.get(e["step_id"]))
        if placed is None or stated is None:
            continue
        both += 1
        if placed == stated:
            agree += 1
        elif len(disagreements) < 20:
            disagreements.append({"step_id": e["step_id"], "ifc_class": e["ifc_class"],
                                  "placement_storey": placed, "relationship_storey": stated})

    # held out: hide the stated storey, guess from height, score against what was stated
    tested = correct = clamped = 0
    for e in declared_geometry:
        truth = index.get(e["storey_step_id"])
        if truth is None:
            continue
        guess, was_clamped = elevation_guess(e["bbox_min"][2], bands)
        tested += 1
        correct += guess == truth
        clamped += was_clamped

    # assign_storeys labels each stated storey with its route, and fallbacks with their method
    methods = Counter()
    for e in with_geometry:
        if e["step_id"] not in assign:
            methods["unassigned"] += 1
            continue
        _, how = assign[e["step_id"]]
        methods[how] += 1

    has_representation = {e["step_id"]: getattr(f.by_id(e["step_id"]), "Representation", None) is not None
                          for e in elements}
    representable = [e for e in elements if has_representation[e["step_id"]]]
    missing = Counter(e["ifc_class"] for e in representable if not e["has_geometry"])

    # the earlier formulas: all IfcProduct as denominator, placement chain then containment only
    old_declared, products = {}, 0
    for el in f.by_type("IfcProduct"):
        products += 1
        sid = storey_from_placement(el)
        if sid is not None:
            old_declared[el.id()] = sid
    for el_id, storey_id in contained.items():
        old_declared.setdefault(el_id, storey_id)
    by_step = {e["step_id"]: e for e in elements}
    old_hit = old_miss = 0
    for el_id, storey_id in old_declared.items():
        e = by_step.get(el_id)
        if storey_id not in index or not e or not e.get("bbox_min"):
            continue
        guess, _ = elevation_guess(e["bbox_min"][2], bands)
        old_hit += guess == index[storey_id]
        old_miss += guess != index[storey_id]

    return {
        "storey_links": {"all_elements": ratio(declared_all, len(elements)),
                         "elements_with_geometry": ratio(len(declared_geometry), len(with_geometry)),
                         "by_route": sorted_counts(routes)},
        "placement_agreement": {**ratio(agree, both), "disagreements": disagreements},
        "elevation_fallback": {**ratio(correct, tested), "clamped": clamped},
        "storey_assignment": {**ratio(len(with_geometry) - methods.get("unassigned", 0),
                                      len(with_geometry)),
                              "by_method": sorted_counts(methods)},
        "geometry_extraction": {**ratio(sum(1 for e in representable if e["has_geometry"]),
                                        len(representable)),
                                "missing_by_class": sorted_counts(missing),
                                "geometry_without_representation": sum(
                                    1 for e in elements if e["has_geometry"]
                                    and not has_representation[e["step_id"]])},
        "_legacy": {"storey_n": old_hit + old_miss, "storey_correct": old_hit,
                    "storey_products": products, "storey_exact_linked": len(old_declared)},
    }


# one model
def validate(path, settings):
    """All measures for one model, or None when the model has no storeys at all."""
    data, f = extract(str(path), threads=settings["threads"])
    storeys = data["storeys"]
    if not storeys:
        return None
    assign = assign_storeys(data)
    zs = [e["bbox_min"][2] for e in data["elements"] if e.get("bbox_min")]
    bands = storey_bands(storeys, min(zs) if zs else 0., max(zs) if zs else 0.)

    full_footprints(data, assign)

    area_errors, area_by_class, area_status = [], defaultdict(list), Counter()
    area_comparison_status = Counter()
    for e in data["elements"]:
        if e.get("_poly") is None:
            continue
        q = quantity_or_error(declared_footprint_area, f.by_id(e["step_id"]))
        area_status[q["status"]] += 1
        declared = q.get("value_si") if q["status"] == "ok" else None
        if declared is None:
            area_comparison_status[f"quantity_{q['status']}"] += 1
            continue
        if declared <= MIN_DECLARED_AREA_M2:
            area_comparison_status["below_minimum_declared_area"] += 1
            continue
        area_comparison_status["compared"] += 1
        err = abs(e["_poly"].area - declared) / declared
        area_errors.append(err)
        area_by_class[e["ifc_class"]].append(err)

    volume_errors, volume_status, volume_comparison_status = [], Counter(), Counter()
    for e in data["elements"]:
        if e.get("_poly") is None or not e["ifc_class"].startswith(PRISM):
            continue
        q = quantity_or_error(declared_volume, f.by_id(e["step_id"]))
        volume_status[q["status"]] += 1
        declared = q.get("value_si") if q["status"] == "ok" else None
        if declared is None:
            volume_comparison_status[f"quantity_{q['status']}"] += 1
            continue
        if declared <= MIN_DECLARED_VOLUME_M3:
            volume_comparison_status["below_minimum_declared_volume"] += 1
            continue
        height = e["bbox_max"][2] - e["bbox_min"][2]
        if height <= MIN_HEIGHT_M:
            volume_comparison_status["below_minimum_height"] += 1
            continue
        volume_comparison_status["compared"] += 1
        volume_errors.append(abs(e["_poly"].area * height - declared) / declared)

    storey = storey_measures(f, data, assign, storeys, bands)
    legacy_storey = storey.pop("_legacy")

    room_options = {"cell": settings["cell"], "min_area": settings["min_room_area"],
                    "erode": settings["erode"]}
    storey_tests = []
    for i, st in enumerate(storeys):
        elems = [e for e in data["elements"]
                 if e["step_id"] in assign and assign[e["step_id"]][0] == i]
        spaces = [e for e in elems if e["ifc_class"] == "IfcSpace" and e.get("_poly") is not None]
        if len(spaces) < 2:
            continue
        truth = [s["_poly"] for s in spaces]
        # A projection failure must retain this storey's available references in recall,
        # just like an inference failure. Geometry visibility does not require ownership
        # on this storey, and absence of barriers does not make the reference ineligible.
        try:
            visible, projection = section_elements(
                data, assign, i, cut_height=settings.get("cut_height", 1.2),
                cut_thickness=settings.get("cut_thickness", .01),
                band_base=settings.get("band_base"),
                barrier_classes=tuple(settings.get("barrier_classes") or BARRIERS),
                barriers_only=True)
        except Exception as ex:
            projection = {"status": "error", "barriers_complete": False,
                          "error": f"{type(ex).__name__}: {ex}"}
            result = {**room_test(truth, []), "status": "failed",
                      "inference_diagnostics": {"status": "section_projection_error",
                                                "error": projection["error"]}}
        else:
            if projection["barriers_complete"]:
                barriers, _ = barrier_polygons(visible, data)
                result = infer_and_test_rooms(truth, barriers, room_options)
            else:
                result = {**room_test(truth, []), "status": "failed",
                          "inference_diagnostics": {
                              "status": projection.get("status") if projection.get("status") == "no_section_band"
                              else "incomplete_section_barriers",
                              "reason": projection.get("reason")}}
        storey_tests.append({"storey_index": i, "storey": st["name"],
                             "section_projection": projection, **result})

    footprint_candidates = [e for e in data["elements"]
                            if e["has_geometry"] and e["step_id"] in assign]
    missing_footprints = Counter(e["ifc_class"] for e in footprint_candidates
                                 if e.get("_poly") is None)
    inference_failures = sum(test["status"] == "failed" for test in storey_tests)
    # A storey scores again once a failed barrier section falls back to its whole-element
    # silhouette, so the fallback must be visible next to the score it restores.
    def fallback_barriers(test):
        return sum(1 for f in test["section_projection"].get("fallbacks", []) if f["affects_room_barrier"])
    barrier_fallbacks = sum(fallback_barriers(t) for t in storey_tests)
    fallback_storeys = sum(1 for t in storey_tests if fallback_barriers(t))

    area_n, volume_n = len(area_errors), len(volume_errors)
    return {
        "schema": data["schema"], "storey_source": data["storey_source"],
        "elements": len(data["elements"]),
        "footprint_area": {**error_summary(area_errors),
                           "by_class": {k: {"n": len(v), "median": median_or_none(v)}
                                        for k, v in sorted(area_by_class.items(),
                                                           key=lambda kv: (-len(kv[1]), kv[0]))},
                           "quantity_status": sorted_counts(area_status),
                           "comparison_status": sorted_counts(area_comparison_status)},
        "volume": {**error_summary(volume_errors), "quantity_status": sorted_counts(volume_status),
                   "comparison_status": sorted_counts(volume_comparison_status)},
        **storey,
        "footprint_extraction": {
            **ratio(len(footprint_candidates) - sum(missing_footprints.values()),
                    len(footprint_candidates)),
            "missing_by_class": sorted_counts(missing_footprints)},
        "room_inference": {"storeys_tested": len(storey_tests),
                           "inference_failures": inference_failures,
                           "storeys_with_fallback_barriers": fallback_storeys,
                           "barrier_fallbacks": barrier_fallbacks,
                           'planar_surface_approximations': sum(len(t['section_projection'].get(
                               'planar_surface_approximations', [])) for t in storey_tests),
                           'orientation_repairs': sum(len(t['section_projection'].get(
                               'orientation_repairs', [])) for t in storey_tests),
                           "completion": ratio(len(storey_tests) - inference_failures, len(storey_tests)),
                           "storeys": storey_tests},
        "legacy_1_sep": {
            "area_n": area_n, "area_median_rel_err": median_or_none(area_errors),
            "area_within_5pct": round(sum(1 for r in area_errors if r <= .05) / area_n, 4) if area_n else None,
            "area_within_10pct": round(sum(1 for r in area_errors if r <= .10) / area_n, 4) if area_n else None,
            "vol_n": volume_n, "vol_median_rel_err": median_or_none(volume_errors),
            "vol_within_10pct": round(sum(1 for r in volume_errors if r <= .10) / volume_n, 4) if volume_n else None,
            **legacy_storey,
            "elements_with_geometry": storey["storey_assignment"]["denominator"],
            "elements_placed": storey["storey_assignment"]["numerator"]},
        "_area_errors": area_errors, "_volume_errors": volume_errors,
    }


# pooling
def pooled_ratio(rows, *path):
    numerator = denominator = 0
    for r in rows:
        block = r
        for key in path:
            block = block[key]
        numerator += block["numerator"]
        denominator += block["denominator"]
    return ratio(numerator, denominator)


def pooled_counts(rows, *path):
    total = Counter()
    for r in rows:
        block = r
        for key in path:
            block = block[key]
        total.update(block)
    return sorted_counts(total)


def legacy_summary(rows):
    """The 1 Sep summary formulas, unchanged, for comparison only."""
    L = [r["legacy_1_sep"] for r in rows]
    area_n = sum(x["area_n"] for x in L)
    vol_n = sum(x["vol_n"] for x in L)
    w5 = sum(x["area_within_5pct"] * x["area_n"] for x in L if x["area_within_5pct"] is not None)
    w10 = sum(x["area_within_10pct"] * x["area_n"] for x in L if x["area_within_10pct"] is not None)
    v10 = sum(x["vol_within_10pct"] * x["vol_n"] for x in L if x["vol_within_10pct"] is not None)
    tests = [t for r in rows for t in r["room_inference"]["storeys"]
             if t.get("status", "completed") == "completed"]
    old = [t["legacy_1_sep"] for t in tests]
    tested = sum(x["storey_n"] for x in L)
    correct = sum(x["storey_correct"] for x in L)
    products = sum(x["storey_products"] for x in L)
    linked = sum(x["storey_exact_linked"] for x in L)
    geometry = sum(x["elements_with_geometry"] for x in L)
    placed = sum(x["elements_placed"] for x in L)
    truth_rooms = sum(t["n_truth"] for t in tests)
    return {
        "models": len(rows),
        "footprint_area": {
            "elements_compared": area_n,
            "within_5_percent": round(w5 / area_n, 4) if area_n else None,
            "within_10_percent": round(w10 / area_n, 4) if area_n else None,
            "median_relative_error": median_or_none(
                x["area_median_rel_err"] for x in L if x["area_median_rel_err"] is not None)},
        "storey_assignment": {
            "elements_tested": tested, "correct": correct,
            "accuracy": round(correct / tested, 4) if tested else None,
            "products": products, "exact_from_file": linked,
            "exact_share": round(linked / max(products, 1), 4)},
        "room_inference": {
            "storeys_tested": len(tests), "truth_rooms": truth_rooms,
            "inferred_rooms": sum(t["n_inferred"] for t in tests),
            "matched": sum(x["matched"] for x in old),
            "recall": round(sum(x["matched"] for x in old) / max(truth_rooms, 1), 4),
            "mean_iou_of_matched": round(statistics.mean(x["mean_iou"] for x in old if x["matched"]), 3)
            if any(x["matched"] for x in old) else 0.0,
            "area_ratio": round(sum(x["area_inferred"] for x in old)
                                / max(sum(x["area_truth"] for x in old), 1e-9), 3)},
        "completeness": {"elements_with_geometry": geometry, "placed": placed,
                         "rate": round(placed / geometry, 4) if geometry else None},
        "volume": {
            "elements_compared": vol_n,
            "within_10_percent": round(v10 / max(vol_n, 1), 4),
            "median_relative_error": median_or_none(
                x["vol_median_rel_err"] for x in L if x["vol_median_rel_err"] is not None)},
    }


def summarise(rows, area_errors, volume_errors):
    tests = [t for r in rows for t in r["room_inference"]["storeys"]]
    truth_rooms = sum(t["n_truth"] for t in tests)
    inferred_rooms = sum(t["n_inferred"] for t in tests)
    area_truth = math.fsum(t["area_truth_m2"] for t in tests)
    by_threshold = {}
    for threshold in map(str, IOU_THRESHOLDS):
        matched = sum(t["by_iou_threshold"][threshold]["matched"] for t in tests)
        iou_sum = math.fsum(t["by_iou_threshold"][threshold]["iou_sum"] for t in tests)
        by_threshold[threshold] = {"recall": ratio(matched, truth_rooms),
                                   "precision": ratio(matched, inferred_rooms),
                                   "mean_iou_of_matches": round(iou_sum / matched, 4) if matched else None}
    return {
        "models": len(rows),
        "footprint_area": {**error_summary(area_errors),
                           "median_of_model_medians": median_or_none(
                               r["footprint_area"]["median"] for r in rows
                               if r["footprint_area"]["median"] is not None),
                           "quantity_status": pooled_counts(rows, "footprint_area", "quantity_status"),
                           "comparison_status": pooled_counts(rows, "footprint_area", "comparison_status")},
        "volume": {**error_summary(volume_errors),
                   "median_of_model_medians": median_or_none(
                       r["volume"]["median"] for r in rows if r["volume"]["median"] is not None),
                   "quantity_status": pooled_counts(rows, "volume", "quantity_status"),
                   "comparison_status": pooled_counts(rows, "volume", "comparison_status")},
        "storey_links": {"all_elements": pooled_ratio(rows, "storey_links", "all_elements"),
                         "elements_with_geometry": pooled_ratio(rows, "storey_links",
                                                                "elements_with_geometry"),
                         "by_route": pooled_counts(rows, "storey_links", "by_route")},
        "placement_agreement": pooled_ratio(rows, "placement_agreement"),
        "elevation_fallback": {**pooled_ratio(rows, "elevation_fallback"),
                               "clamped": sum(r["elevation_fallback"]["clamped"] for r in rows)},
        "storey_assignment": {**pooled_ratio(rows, "storey_assignment"),
                              "by_method": pooled_counts(rows, "storey_assignment", "by_method")},
        "geometry_extraction": {**pooled_ratio(rows, "geometry_extraction"),
                                "missing_by_class": pooled_counts(rows, "geometry_extraction",
                                                                  "missing_by_class"),
                                "geometry_without_representation": sum(
                                    r["geometry_extraction"]["geometry_without_representation"]
                                    for r in rows)},
        "footprint_extraction": {**pooled_ratio(rows, "footprint_extraction"),
                                 "missing_by_class": pooled_counts(rows, "footprint_extraction",
                                                                   "missing_by_class")},
        "room_inference": {
            "storeys_tested": len(tests),
            "inference_failures": sum(r["room_inference"]["inference_failures"] for r in rows),
            "storeys_with_fallback_barriers": sum(r["room_inference"].get("storeys_with_fallback_barriers", 0)
                                                  for r in rows),
            "barrier_fallbacks": sum(r["room_inference"].get("barrier_fallbacks", 0) for r in rows),
            'planar_surface_approximations': sum(r['room_inference'].get('planar_surface_approximations', 0) for r in rows),
            'orientation_repairs': sum(r['room_inference'].get('orientation_repairs', 0) for r in rows),
            "completion": pooled_ratio(rows, "room_inference", "completion"),
            "truth_rooms_on_failed_storeys": sum(t["n_truth"] for t in tests
                                                  if t["status"] == "failed"),
            "truth_rooms": truth_rooms, "inferred_rooms": inferred_rooms,
            "area_ratio": round(math.fsum(t["area_inferred_m2"] for t in tests) / area_truth, 4)
            if area_truth else None,
            "storeys_with_overlapping_truth": sum(1 for t in tests if (t["truth_overlap_m2"] or 0) > 0.01),
            "iou_failures": sum(t["iou_failures"] for t in tests),
            "by_iou_threshold": by_threshold},
        "legacy_1_sep": legacy_summary(rows),
    }


def print_report(summary):
    def rate(r):
        if r["value"] is None:
            return f"n/a (0/{r['denominator']})"
        return f"{r['value']:.1%} ({r['numerator']}/{r['denominator']})"
    fa, vo, ri = summary["footprint_area"], summary["volume"], summary["room_inference"]
    lines = [
        f"models validated      {summary['models']}",
        f"footprint area        n={fa['n']} median={fa['median']} "
        f"within 5% {rate(fa['within_5pct'])} within 10% {rate(fa['within_10pct'])}",
        f"volume                n={vo['n']} median={vo['median']} within 10% {rate(vo['within_10pct'])}",
        f"storey stated         {rate(summary['storey_links']['all_elements'])} of elements, "
        f"{rate(summary['storey_links']['elements_with_geometry'])} of elements with geometry",
        f"placement agreement   {rate(summary['placement_agreement'])}",
        f"elevation fallback    {rate(summary['elevation_fallback'])} (held out)",
        f"geometry produced     {rate(summary['geometry_extraction'])} of elements with a representation",
        f"room inference        {ri['storeys_tested']} storeys, {ri['truth_rooms']} IfcSpace rooms, "
        f"{ri['inferred_rooms']} inferred",
        f"  barrier fallbacks   {ri.get('barrier_fallbacks', 0)} barriers on "
        f"{ri.get('storeys_with_fallback_barriers', 0)} storeys kept a full-height silhouette",
    ]
    for threshold, block in ri["by_iou_threshold"].items():
        lines.append(f"  IoU >= {threshold:<4}         recall {rate(block['recall'])} "
                     f"precision {rate(block['precision'])} mean IoU {block['mean_iou_of_matches']}")
    print("\n".join(lines))


# run
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    runs.input_arguments(ap)
    ap.add_argument("--include-ineligible", action="store_true",
                    help="also validate manifest rows marked ineligible")
    args = ap.parse_args(argv)
    command = [sys.executable, str(Path(__file__).resolve()),
               *(sys.argv[1:] if argv is None else argv)]
    try:
        out, record = runs.start_run(args, "validation", command=command)
    except (ValueError, OSError) as ex:
        ap.error(str(ex))

    settings = record["settings"]
    rows, area_errors, volume_errors = [], [], []
    entries = {entry['model']: entry for entry in record['inputs']}
    for row, result in runs.model_results(out, record, include_ineligible=args.include_ineligible):
        if row['status'] == 'completed':
            entry = entries[row['model']]
            area_errors.extend(result.pop('_area_errors'))
            volume_errors.extend(result.pop('_volume_errors'))
            rows.append({'model': entry['model'], 'input_ifc': entry['path'],
                         'input_ifc_sha256': entry['ifc_sha256'], **result})
    rows.sort(key=lambda row: row['model'])

    summary = summarise(rows, area_errors, volume_errors)
    statuses = Counter(row["status"] for row in record["models"])
    summary["model_execution"] = {
        "selected": len(record["inputs"]), "by_status": sorted_counts(statuses),
        "completion": ratio(statuses["completed"], statuses["completed"] + statuses["failed"])}
    runs.atomic_json(out / "validation.json", {
        "definitions_version": DEFINITIONS_VERSION, "definitions": DEFINITIONS_DOC,
        "dataset": record["dataset"], "code_sha256": record["code_sha256"],
        "settings": {**settings, "iou_thresholds": list(IOU_THRESHOLDS),
                     "error_thresholds": list(ERROR_THRESHOLDS), "area_keys": list(AREA_KEYS),
                     "slab_keys": list(SLAB_KEYS), "volume_keys": list(VOL_KEYS),
                     "prism_classes": list(PRISM), "barrier_classes": list(BARRIER),
                     "min_declared_area_m2": MIN_DECLARED_AREA_M2,
                     "min_declared_volume_m3": MIN_DECLARED_VOLUME_M3, "min_height_m": MIN_HEIGHT_M},
        "summary": summary, "per_model": rows})
    runs.finish_run(out, record)
    print()
    print_report(summary)
    print(f"\n{record['status']} -> {out}")
    return 1 if record["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
