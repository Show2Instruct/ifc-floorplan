#!/usr/bin/env python
"""Check the parametric layer of a run against the geometry it was read from.

Each wall is rebuilt from its axis and thickness and compared with the footprint the IFC gives it.
Each opening is checked to sit on its host wall: its centre should lie within half the wall's
thickness of the axis, and its offset within the wall's length. These are the properties a model
rebuilt from the plan depends on, so they are measured over the whole run rather than assumed.

    python -m fp.verify_vector --run runs/<run> --out reviews/<name>
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fp.vectorize import rebuilt_wall
from fp import runs

from fp.vectorize import ON_AXIS_M as ON_AXIS_SLACK_M   # the slack the layer itself places openings with


def geometry_of(parts):
    """A stored outline, GeoJSON MultiPolygon coordinates, back into a shape."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    if not parts:
        return None
    try:
        shape = unary_union([Polygon(part[0], part[1:]) for part in parts])
    except Exception:
        return None
    return shape if shape.is_valid else shape.buffer(0)


def verify(run):
    counts, ious, straight_ious, problems = Counter(), [], [], []
    models = sorted(os.listdir(os.path.join(run, "models")))
    for model in models:
        with open(os.path.join(run, "models", model, "building.json")) as fh:
            building = json.load(fh)
        children = {}
        for relation in building.get("relations") or []:
            if relation.get("type") == "part_of":
                children.setdefault(relation["target"], []).append(relation["source"])
        storeys = []
        for record in building["storeys"]:
            with open(os.path.join(run, "models", model, f"storey_{record['storey_index']:02d}.json")) as fh:
                storeys.append((record, json.load(fh)))
        # a wall standing through two storeys hosts openings on both, so hosts are looked up
        # across the building
        building_walls = {w["id"]: w for _, storey in storeys for w in (storey.get("vector") or {}).get("walls", [])}
        # the parts of a wall standing through several floors sit on the floors they stand on
        building_footprints = {}
        for _, storey in storeys:
            for element in storey.get("elements", []):
                if element.get("footprint_geometry"):
                    building_footprints[element["step_id"]] = element["footprint_geometry"]
        for record, storey in storeys:
            vector = storey.get("vector")
            counts["storeys"] += 1
            if vector is None:
                counts["storeys_without_a_vector_layer"] += 1
                continue
            counts["storeys_with_a_height"] += storey.get("height_m") is not None
            footprints = {}
            for element in storey.get("elements", []):
                shape = element.get("footprint_geometry")
                if shape:
                    footprints[element["step_id"]] = geometry_of(shape)
            walls = {w["id"]: w for w in vector["walls"]}
            for wall in vector["walls"]:
                counts["walls"] += 1
                counts["walls_straight"] += wall["straight"]
                original = footprints.get(wall["step_id"])
                if original is None and wall.get("composed_from_parts"):
                    # a wall built from parts is checked against the parts themselves
                    from shapely.ops import unary_union
                    pieces = [footprints.get(c) or geometry_of(building_footprints[c])
                              for c in children.get(wall["step_id"], ()) if c in footprints or c in building_footprints]
                    original = unary_union(pieces) if pieces else None
                    counts["walls_checked_against_their_parts"] += original is not None
                # a wall drawn through openings at its ends is compared where its geometry stands
                rebuilt = rebuilt_wall(wall, body=True)
                counts["walls_extended_over_end_openings"] += bool(wall.get("body_m"))
                if original is None or original.is_empty or rebuilt is None:
                    counts["walls_not_comparable"] += 1
                    continue
                # The parameters describe the wall's extent; a cavity inside it is reported by
                # solid_fill, so the extent is what they are checked against.
                extent = original.convex_hull
                iou = rebuilt.intersection(extent).area / rebuilt.union(extent).area
                ious.append(iou)
                if wall["straight"]:
                    straight_ious.append(iou)
                    if iou < .9:
                        problems.append({"model": model, "storey": record["name"], "wall": wall["id"],
                                         "iou": round(iou, 3), "problem": "straight wall rebuilds poorly"})
            for opening in vector["openings"]:
                counts["openings"] += 1
                host = walls.get(opening["host_wall"])
                if opening["offset_m"] is None:
                    counts["openings_not_placed"] += 1
                    continue
                counts["openings_placed"] += 1
                counts[f"placed_from_{opening.get('placed_from')}"] += 1
                if host is None:
                    host = building_walls.get(opening["host_wall"])
                    if host is None:
                        counts["openings_whose_host_is_not_in_the_run"] += 1
                        continue
                    counts["openings_on_a_host_from_another_storey"] += 1
                limit = host["thickness_m"] / 2 + ON_AXIS_SLACK_M
                within_length = -ON_AXIS_SLACK_M <= opening["offset_m"] <= host["length_m"] + ON_AXIS_SLACK_M
                if opening["off_axis_m"] <= limit and within_length:
                    counts["openings_on_their_host_wall"] += 1
                else:
                    counts["openings_off_their_host_wall"] += 1
                    if len(problems) < 200:
                        problems.append({"model": model, "storey": record["name"], "opening": opening["id"],
                                         "off_axis_m": opening["off_axis_m"], "limit_m": round(limit, 3),
                                         "offset_m": opening["offset_m"], "wall_length_m": host["length_m"],
                                         "problem": "opening not on its host wall"})
            for slab in vector["slabs"]:
                counts["slabs"] += 1

    def share(part, whole):
        return round(counts[part] / counts[whole], 4) if counts[whole] else None

    summary = {
        "run": run, "models": len(models), "counts": dict(counts),
        "wall_rebuild_iou": {"median": round(statistics.median(ious), 4) if ious else None,
                             "within_0_9": round(sum(i >= .9 for i in ious) / len(ious), 4) if ious else None},
        "straight_wall_rebuild_iou": {
            "median": round(statistics.median(straight_ious), 4) if straight_ious else None,
            "within_0_9": round(sum(i >= .9 for i in straight_ious) / len(straight_ious), 4) if straight_ious else None},
        "share_of_walls_straight": share("walls_straight", "walls"),
        "share_of_openings_placed": share("openings_placed", "openings"),
        "share_of_placed_openings_on_their_host": (
            round(counts["openings_on_their_host_wall"]
                  / (counts["openings_on_their_host_wall"] + counts["openings_off_their_host_wall"]), 4)
            if counts["openings_on_their_host_wall"] + counts["openings_off_their_host_wall"] else None),
        "share_of_storeys_with_a_height": share("storeys_with_a_height", "storeys"),
        "problems": problems[:200],
    }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True, help="new, empty output directory")
    args = parser.parse_args(argv)
    out = Path(runs.safe_output(args.out))
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output already contains files; choose a new directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    summary = verify(args.run)
    runs.atomic_json(out / "vector_check.json", summary)
    for key in ("share_of_walls_straight", "wall_rebuild_iou", "straight_wall_rebuild_iou",
                "share_of_openings_placed", "share_of_placed_openings_on_their_host",
                "share_of_storeys_with_a_height"):
        print(f"  {key:42s} {summary[key]}")
    print("  counts:", summary["counts"])
    print(f"-> {out / 'vector_check.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
