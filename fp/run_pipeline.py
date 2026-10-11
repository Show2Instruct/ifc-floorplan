#!/usr/bin/env python
"""Run the floor-plan pipeline on chosen inputs, with a record of the run.

Each run writes into a new directory of its own and never touches the dataset:

    <out>/run.json                what ran: command, settings, input fingerprints, per-model
                                  status, failures and output fingerprints
    <out>/source.zip              the pipeline sources that produced the output
    <out>/environment-freeze.txt  installed package versions
    <out>/models/<model>/         building.json, storey_NN.json/.svg, bev.svg

Inputs are chosen on the command line rather than by editing module globals, so different corpus
layouts go through the same entry point:

    python -m fp.run_pipeline --models model_119 --out runs/model_119

    python -m fp.run_pipeline --manifest pairs.json --dataset resbim --models 0 500 \
        --out runs/resbim-0-500

A model that fails is recorded with its error and the run carries on. Manifest rows marked
ineligible, such as a file whose IFC holds no building elements, are recorded as skipped unless
--include-ineligible is given, so coverage still accounts for them.

Exit status is 0 when no model failed (skips remain explicit) and 1 when any failed.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fp import runs
from fp.build import build
from fp.projections import BARRIERS


def summarise(building):
    storeys = building["storeys"]
    warnings = [{"storey_index": s["storey_index"], "reason": "room_grid_limit"}
                for s in storeys if s.get("room_inference", {}).get("status") == "grid_limit"]
    if building.get("geometry_errors"):
        warnings.append({"reason": "geometry_read_errors", "count": len(building["geometry_errors"])})
    warnings.extend({'storey_index': s['storey_index'], 'reason': 'section_projection_exclusions',
                     'count': len(s['section_projection']['excluded'])}
                    for s in storeys if s.get('section_projection', {}).get('excluded'))
    warnings.extend({'storey_index': s['storey_index'], 'reason': reason}
                    for s in storeys for reason in s.get('section_projection', {}).get('warnings', []))
    return {"storeys": len(storeys), "elements": building["n_elements"],
            "elements_with_geometry": building["n_elements_with_geometry"],
            "rooms": sum(s["n_rooms"] for s in storeys),
            "room_sources": sorted({s["room_source"] for s in storeys}),
            "storey_source": building["storey_source"], "warnings": warnings}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    runs.input_arguments(ap)
    ap.add_argument("--include-ineligible", action="store_true",
                    help="also build manifest rows marked ineligible")
    args = ap.parse_args(argv)
    command = [sys.executable, str(Path(__file__).resolve()),
               *(sys.argv[1:] if argv is None else argv)]
    try:
        out, record = runs.start_run(args, "floorplans", command=command)
    except (ValueError, OSError) as ex:
        ap.error(str(ex))

    for _row, _result in runs.model_results(out, record, include_ineligible=args.include_ineligible):
        pass

    runs.finish_run(out, record)
    counts = {}
    for r in record["models"]:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"\n{record['status']}: {counts} -> {out}")
    return 1 if record["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
