#!/usr/bin/env python
"""Turn a pipeline run into scene graphs, one `*.graph.json` per building.

A graph has a building node, a storey node per level, a space node per room, `contains` edges
down the hierarchy and `connected_by_door`, `open_passage` and `vertically_connected` edges across
it, in the vocabulary of the BIM-GNI floor-plan graphs, so a graph drawn by hand in that
vocabulary and one made here compare directly.

Nothing here judges anything. Every node and edge records where it came from: `ifc` for a
declared space, `inferred` for one found between walls. A graph can always be traced back to the
geometry that produced it.

    python -m fp.export_graph --run runs/<run> --out graphs/<name>
    python -m fp.export_graph --run runs/<run> --models model_187 --elements --out graphs/<name>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fp import runs
from fp.vectorize import rebuilt_wall

PROVENANCE = {"IfcSpace": "ifc"}
DEFAULT_STOREY_HEIGHT_M = 3.
# Elements worth carrying as nodes: what bounds a room, what opens it, what holds it up, and
# what stands in it. Slabs and coverings are left out, being the floor the room already is.
ELEMENT_LAYERS = {"IfcWall": "wall", "IfcWallStandardCase": "wall", "IfcCurtainWall": "wall",
                  "IfcDoor": "door", "IfcWindow": "window", "IfcColumn": "column",
                  "IfcBeam": "beam", "IfcStair": "stair", "IfcStairFlight": "stair",
                  "IfcRamp": "ramp", "IfcRailing": "railing", "IfcFurniture": "furniture",
                  "IfcTransportElement": "lift"}


def outline_of(room):
    """The room's exterior ring, for drawing its floor plate."""
    polygon = room.get("geometry") or room.get("polygon")
    if not polygon:
        return None
    ring = polygon[0][0] if isinstance(polygon[0][0][0], (list, tuple)) else polygon[0]
    return [[round(x, 3), round(y, 3)] for x, y in ring]


def centre_of(ring):
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return [round(sum(xs) / len(xs), 3), round(sum(ys) / len(ys), 3)]


def placed_by_height(room):
    """Whether a declared space's storey comes from the height of its base, not from the file."""
    return (room.get("storey_assignment") or "").startswith(("inferred", "moved"))


def graph_of(model, run_models, *, with_elements=False):
    with open(os.path.join(run_models, model, "building.json")) as fh:
        building = json.load(fh)
    seen_exterior = set()
    nodes = [{"id": model, "layer": "building", "label": model, "provenance": "ifc",
              "schema": building.get("schema")}]
    edges, counts = [], Counter()
    exterior_doors = 0
    room_storey = {}
    storey_ids = {storey["step_id"]: storey.get("global_id") or f"{model}-storey-{index}"
                  for index, storey in enumerate(building["storeys"]) if storey.get("step_id") is not None}
    for index, storey in enumerate(building["storeys"]):
        storey_id = storey.get("global_id") or f"{model}-storey-{index}"
        height_from = "default"
        if "height_m" in storey:
            # measured to the next physical level, or at the top from the storey's own walls
            height = storey["height_m"] if storey["height_m"] is not None else DEFAULT_STOREY_HEIGHT_M
            if storey["height_m"] is not None:
                height_from = storey.get("height_from") or "next_level"
        else:
            # runs made before storey heights were recorded
            following = building["storeys"][index + 1]["elevation_m"] if index + 1 < len(building["storeys"]) else None
            height = round(following - storey["elevation_m"], 3) if following is not None else DEFAULT_STOREY_HEIGHT_M
            if following is not None:
                height_from = "next_level"
            if height <= .1:                  # datum levels stacked on one another
                height, height_from = DEFAULT_STOREY_HEIGHT_M, "default"
        nodes.append({"id": storey_id, "layer": "storey", "label": storey["name"],
                      "elevation": storey["elevation_m"], "height": height, "height_from": height_from,
                      "parent": model,
                      "provenance": "ifc" if storey.get("global_id") else "inferred"})
        edges.append({"a": model, "b": storey_id, "relation": "contains", "provenance": "ifc"})
        counts["storeys"] += 1
        with open(os.path.join(run_models, model, f"storey_{storey.get('storey_index', index):02d}.json")) as fh:
            record = json.load(fh)
        for room in record["rooms"]:
            provenance = PROVENANCE.get(room["source"], "inferred")
            node = {"id": room["id"], "layer": "space", "label": room["name"],
                    "area": room["area_m2"], "parent": storey_id, "provenance": provenance,
                    "global_id": room.get("global_id")}
            node['geometry_quality'] = record.get('geometry_quality')
            if room.get("subdivides_declared_space") is not None:
                # the file named one space where walls divide it; say so rather than hide it
                node["subdivides_declared_space"] = room["subdivides_declared_space"]
                node["declared_space_global_id"] = room.get("declared_space_global_id")
                node["provenance"] = "inferred_within_declared_space"
            if room.get("storey_stated") is not None:
                # the file names another storey, but the space stands on this one
                node["storey_stated"] = storey_ids.get(room["storey_stated"])
            ring = outline_of(room)
            if ring:
                node["centroid"] = centre_of(ring)
                node["outline"] = ring
                # a room sits on its storey and stands to the next one: the graph is placed in
                # the building's own coordinates, so a wrong edge shows as a line through a wall
                node["centroid3d"] = node["centroid"] + [round(storey["elevation_m"] + .1, 3)]
                node["height"] = height
            nodes.append(node)
            edges.append({"a": storey_id, "b": room["id"], "relation": "contains",
                          "provenance": "geometry" if placed_by_height(room) else provenance})
            room_storey[room["id"]] = storey_id
            counts[f"rooms_{provenance}"] += 1
        if with_elements:
            vector = record.get("vector") or {}
            parameters = {item["step_id"]: item for item in vector.get("walls", []) + vector.get("openings", [])}
            walls = {item["step_id"] for item in vector.get("walls", [])}
            archways = {item["step_id"] for item in vector.get("openings", []) if item.get("kind") == "opening"}
            for element in record["elements"]:
                # a panel that hosts a door is a wall in the parametric layer, and a node like one;
                # an opening left unfilled is a node of its own, since no door or window stands for it
                layer = ELEMENT_LAYERS.get(element["ifc_class"]) or ("wall" if element["step_id"] in walls else None) \
                    or ("opening" if element["step_id"] in archways else None)
                if layer is None:
                    continue
                params = parameters.get(element["step_id"])
                bbox, centroid = None, element.get("centroid")
                if element.get("bbox_min"):
                    bbox = [element["bbox_min"], element["bbox_max"]]
                elif params and params.get("axis") and None not in (params.get("base_m"), params.get("height_m")):
                    # a wall built from parts has no box of its own; its parameters give one
                    minx, miny, maxx, maxy = rebuilt_wall(params, body=True).bounds
                    low = storey["elevation_m"] + params["base_m"]
                    bbox = [[minx, miny, low], [maxx, maxy, low + params["height_m"]]]
                    centroid = [(minx + maxx) / 2, (miny + maxy) / 2, low + params["height_m"] / 2]
                if bbox is None or centroid is None:
                    continue
                node = {"id": f"e{element['step_id']}", "layer": layer,
                        "ifc_class": element["ifc_class"], "global_id": element.get("global_id"),
                        "label": element.get("label") or element["ifc_class"],
                        "parent": element.get("room") or storey_id,
                        "provenance": "ifc",
                        "centroid3d": [round(v, 3) for v in centroid],
                        "bbox": bbox,
                        "material": element.get("material")}
                if params:
                    node["parameters"] = {k: v for k, v in params.items()
                                          if k not in ("id", "step_id", "global_id", "ifc_class")}
                nodes.append(node)
                edges.append({"a": node["parent"], "b": node["id"], "relation": "contains",
                              "provenance": "ifc"})
                counts[f"elements_{layer}"] += 1
                if "room_boundaries" not in record and element.get("room") and layer == "wall":
                    # runs made before room boundaries were recorded
                    edges.append({"a": element["room"], "b": node["id"], "relation": "bounded_by",
                                  "provenance": "geometry"})
                    counts["bounded_by"] += 1
            present = {n["id"] for n in nodes}
            for boundary in record.get("room_boundaries", []):
                target = f"e{boundary['element']}"
                if boundary["room"] in present and target in present:
                    edges.append({"a": boundary["room"], "b": target, "relation": "bounded_by",
                                  "length": boundary["length_m"], "provenance": "geometry"})
                    counts["bounded_by"] += 1
        for space in record.get("outdoor_spaces", []):
            ring = outline_of(space)
            node = {"id": space["id"], "layer": "outdoor", "label": space["name"],
                    "area": space["area_m2"], "parent": storey_id, "provenance": "geometry",
                    "source": space["source"]}
            if ring:
                node.update(centroid=centre_of(ring), outline=ring,
                            centroid3d=centre_of(ring) + [round(storey["elevation_m"] + .1, 3)],
                            height=height)
            nodes.append(node)
            edges.append({"a": storey_id, "b": space["id"], "relation": "contains", "provenance": "geometry"})
            counts["outdoor_spaces"] += 1
        for link in record.get("room_connections", []):
            exterior_doors += link.get('type') == 'exterior_door'
            if link["b"] is None:
                continue
            if link.get("exterior") and link["b"] not in seen_exterior:
                # the outside is a node: a building has to be enterable in the graph
                seen_exterior.add(link["b"])
                nodes.append({"id": link["b"], "layer": "exterior", "label": "outside",
                              "parent": storey_id, "provenance": "geometry",
                              "centroid3d": None})
                counts["exterior_nodes"] += 1
            edge = {"a": link["a"], "b": link["b"], "relation": link["type"],
                    "storey": storey_id, "provenance": "geometry",
                    "width": link.get("width"), "door": link.get("door"),
                    "source_of_truth": link.get("source_of_truth")}
            if link.get("opening") is not None:
                edge["opening"] = link["opening"]          # the unfilled opening walked through
            if link.get("barrier_evidence"):
                edge["barrier_evidence"] = link["barrier_evidence"]
            edges.append(edge)
            counts[link["type"]] += 1
    for link in building.get("vertical_connections", []):
        edges.append({"a": link["a"], "b": link["b"], "relation": "vertically_connected",
                      "kind": link.get("kind"), "width": link.get("width"), "element": link.get("element"),
                      "provenance": "geometry", "source_of_truth": link.get("source_of_truth")})
        counts["vertically_connected"] += 1
    if with_elements:
        # a wall hosts the doors and windows that fill its openings, and the openings it leaves
        # unfilled, as the IFC states
        present = {node["id"] for node in nodes}
        for relation in building.get("relations") or []:
            if relation.get("type") not in ("hosts", "voids"):
                continue
            host, filling = f"e{relation['source']}", f"e{relation['target']}"
            if host in present and filling in present:
                edges.append({"a": host, "b": filling, "relation": "hosts", "provenance": "ifc",
                              "source_of_truth": relation.get("source_of_truth")})
                counts["hosts"] += 1
    # every edge must end on a node, or the graph cannot be walked
    present = {node["id"] for node in nodes}
    counts["dangling_edge_endpoints"] = sum((edge["a"] not in present) + (edge["b"] not in present)
                                            for edge in edges)
    return {"model": model, "source": "pipeline", "annotator": None,
            "provenance": building.get("provenance"),
            "counts": {**counts, "exterior_doors": exterior_doors,
                       "nodes": len(nodes), "edges": len(edges)},
            "nodes": nodes, "edges": edges,
            "note": "Derived from geometry, not judged by a person. Room provenance is 'ifc' for "
                    "a declared IfcSpace and 'inferred' for a region found between barriers."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="a pipeline run directory")
    parser.add_argument("--models", nargs="+", help="selected models (default: all in the run)")
    parser.add_argument("--out", required=True, help="new, empty output directory")
    parser.add_argument("--elements", action="store_true",
                        help="also carry walls, doors, windows, columns and furniture as nodes")
    args = parser.parse_args(argv)

    run_models = os.path.join(args.run, "models")
    models = args.models or sorted(d for d in os.listdir(run_models)
                                   if os.path.isdir(os.path.join(run_models, d)))
    out = runs.safe_output(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output already contains files; choose a new directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    total = Counter()
    for model in models:
        graph = graph_of(model, run_models, with_elements=args.elements)
        runs.atomic_json(out / f"{model}.graph.json", graph)
        total.update(graph["counts"])
    runs.atomic_json(out / "summary.json", {"run": args.run, "models": len(models),
                                            "totals": dict(total)})
    print(f"{len(models)} graph(s) -> {out}")
    for key, value in sorted(total.items()):
        print(f"  {key:<34} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
