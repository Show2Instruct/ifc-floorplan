"""Explicit input selection, source snapshots and durable run provenance.

A run chooses its inputs on the command line and writes into a new directory of its own, so any
output can be traced back to the code, inputs and settings that made it:

    run.json                command, timestamps, settings, input fingerprints, per-model status,
                            failures and output fingerprints
    source.zip              the pipeline sources that ran
    environment-freeze.txt  installed package versions

Inputs must sit under the data directory, which is treated as read-only, and outputs may never be
written there.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# ROOT bounds inputs and outputs. It is a separate name from SOURCE_ROOT so tests can point it
# at a scratch workspace while the sources are still read from here.
from fp.projections import BARRIERS

SOURCE_ROOT = Path(__file__).resolve().parents[1]
ROOT = SOURCE_ROOT
GNI_DIR = "data/gni-bim/2025_BIMfundamentals"


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def safe_output(path):
    root = Path(ROOT).resolve()
    path = Path(path).resolve()
    if not path.is_relative_to(root) or path.is_relative_to(root / "data"):
        raise ValueError(f"output must be inside the workspace and outside data/: {path}")
    return path


def input_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ifc", type=Path, nargs="+", help="explicit IFC file(s)")
    group.add_argument("--input-dir", type=Path, help="directory scanned recursively for IFC files")
    group.add_argument("--manifest", type=Path,
                       help="JSON manifest with a 'pairs' list of model, ifc and ifc_sha256, "
                            "e.g. results/resbim/dataset_manifest.json")
    parser.add_argument("--dataset",
                        help="dataset label stored with the run; required unless the inputs are "
                             "the default GNI directory")
    parser.add_argument("--models", nargs="+", help="selected model identities")
    parser.add_argument("--model-list", type=Path, help="file with one model identity per line")
    parser.add_argument("--n", type=int, help="first N inputs, sorted by identity, after selection")
    parser.add_argument("--out", type=Path,
                        help="new, empty output directory (default results/runs/<operation>-<time>)")
    parser.add_argument("--threads", type=int, default=4, help="IFC geometry threads (default 4)")
    parser.add_argument("--cell", type=float, default=0.12,
                        help="room raster cell size in metres (default 0.12)")
    parser.add_argument("--min-room-area", type=float, default=1.5,
                        help="smallest inferred room in m2 (default 1.5)")
    parser.add_argument("--erode", type=int, default=0,
                        help="room erosion steps (default 0, off; FP_ERODE is not consulted)")
    parser.add_argument("--cut-height", type=float, default=1.2,
                        help="section height above each declared/inferred level in metres (default 1.2)")
    parser.add_argument("--cut-thickness", type=float, default=.01,
                        help="height interval projected for sections in metres (default 0.01)")
    parser.add_argument("--band-base", type=float,
                        help="take barriers from this height above the storey up to the cut height, "
                             "instead of a thin slice at the cut; a plan shows what stands below "
                             "its cut plane (metres)")
    parser.add_argument("--barrier-classes", nargs="+",
                        help="IFC classes that enclose a room (default: walls, curtain walls, "
                             "doors, windows, plates)")


def dataset_label(args):
    if args.dataset:
        return args.dataset
    if args.manifest or args.input_dir or args.ifc:
        raise ValueError("--dataset is required when inputs are not the default GNI directory")
    return "gni"


def select_inputs(args):
    root = Path(ROOT).resolve()
    data = root / "data"
    if args.manifest:
        manifest_path = Path(args.manifest).resolve()
        manifest = json.loads(manifest_path.read_text())
        base = Path(manifest.get("path_root", "."))
        if not base.is_absolute():
            base = (manifest_path.parent / base).resolve()
        entries = []
        for row in manifest["pairs"]:
            model = row.get("model", row.get("identity"))
            if model is None:
                raise ValueError("manifest row without a model identity")
            value = row["ifc"]
            path = value["path"] if isinstance(value, dict) else value
            entries.append({
                "model": str(model), "path": (base / path).resolve(),
                "expected_sha256": row.get("ifc_sha256")
                or (value.get("sha256") if isinstance(value, dict) else None),
                "eligible": row.get("eligible_for_structural_pipeline_pilot"),
                "quality_flags": list(row.get("quality_flags") or [])})
    else:
        if args.ifc:
            paths = [Path(p).resolve() for p in args.ifc]
        else:
            directory = Path(args.input_dir).resolve() if args.input_dir else root / GNI_DIR
            paths = sorted(p.resolve() for p in directory.rglob("*.ifc"))
        entries = [{"model": p.stem, "path": p} for p in paths]

    entries.sort(key=lambda e: e["model"])
    names = [e["model"] for e in entries]
    if len(set(names)) != len(names):
        raise ValueError("duplicate model identities; use a manifest with unique names")
    requested = set(args.models or [])
    if args.model_list:
        requested.update(line.strip() for line in Path(args.model_list).read_text().splitlines()
                         if line.strip() and not line.lstrip().startswith("#"))
    if requested:
        if missing := requested - set(names):
            raise ValueError(f"selected models absent from inputs: {sorted(missing)}")
        entries = [e for e in entries if e["model"] in requested]
    if args.n is not None:
        if args.n < 1:
            raise ValueError("--n must be positive")
        entries = entries[:args.n]
    if not entries:
        raise ValueError("no IFC inputs selected")

    for e in entries:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", e["model"]):
            raise ValueError(f"unsafe model identity: {e['model']!r}")
        p = e["path"]
        if not p.is_relative_to(data) or not p.is_file():
            raise ValueError(f"input must be an existing IFC file under data/: {p}")
        digest = sha256(p)
        expected = e.pop("expected_sha256", None)
        if expected and expected != digest:
            raise ValueError(f"input fingerprint differs from manifest: {e['model']}")
        e.update(path=str(p.relative_to(root)), ifc_sha256=digest, size_bytes=p.stat().st_size)
    return entries


def settings(args):
    cut_height, cut_thickness = getattr(args, 'cut_height', 1.2), getattr(args, 'cut_thickness', .01)
    if (args.threads < 1 or not math.isfinite(args.cell) or args.cell <= 0
            or not math.isfinite(args.min_room_area) or args.min_room_area <= 0 or args.erode < 0):
        raise ValueError("threads, cell and minimum room area must be positive; "
                         "erosion must be zero or more")
    if not math.isfinite(cut_height) or cut_height < 0 or not math.isfinite(cut_thickness) or cut_thickness <= 0:
        raise ValueError('cut height must be finite and nonnegative; cut thickness must be finite and positive')
    band_base = getattr(args, 'band_base', None)
    classes = tuple(getattr(args, 'barrier_classes', None) or BARRIERS)
    if band_base is not None and (not math.isfinite(band_base) or band_base < 0
                                  or band_base >= cut_height):
        raise ValueError('band base must be finite, nonnegative and below the cut height')
    if not all(isinstance(c, str) and c.startswith('Ifc') for c in classes):
        raise ValueError('barrier classes must be IFC class names')
    return {"threads": args.threads, "cell": args.cell,
            "min_room_area": args.min_room_area, "erode": args.erode,
            "cut_height": cut_height, "cut_thickness": cut_thickness,
            "band_base": band_base, "barrier_classes": list(classes)}


def start_run(args, operation, command=None):
    dataset = dataset_label(args)
    chosen = select_inputs(args)
    config = settings(args)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    root = Path(ROOT).resolve()
    out = safe_output(args.out or root / "results" / "runs" / f"{operation}-{stamp}")
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"output directory is not empty; use a new --out: {out}")
    out.mkdir(parents=True, exist_ok=True)

    here = Path(__file__).resolve().parent
    sources = sorted(here.glob("*.py")) + sorted(here.glob("*.html"))
    source_bytes = {str(p.relative_to(SOURCE_ROOT)): p.read_bytes() for p in sources}
    fingerprints = {name: hashlib.sha256(content).hexdigest() for name, content in source_bytes.items()}
    code_id = hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest()
    with zipfile.ZipFile(out / "source.zip", "x", compression=zipfile.ZIP_DEFLATED) as z:
        for name, content in source_bytes.items():
            z.writestr(name, content)
    packages = dict(sorted((d.metadata["Name"], d.version)
                           for d in importlib.metadata.distributions() if d.metadata["Name"]))
    (out / "environment-freeze.txt").write_text("".join(f"{k}=={v}\n" for k, v in packages.items()))

    record = {"format_version": 1, "run_id": f"{operation}-{stamp}", "operation": operation,
              "status": "running", "started_utc": utc(), "finished_utc": None,
              "dataset": dataset, "command": command or [sys.executable, *sys.argv],
              "output_dir": str(out.relative_to(root)), "settings": config, "inputs": chosen,
              "code_sha256": code_id, "source_files": fingerprints,
              "source_archive_sha256": sha256(out / "source.zip"),
              "environment": {"python": sys.version, "executable": sys.executable,
                              "platform": platform.platform(), "packages": packages},
              "models": [], "failures": []}
    atomic_json(out / "run.json", record)
    return out, record


def model_provenance(run, entry):
    """What produced one model's output.

    Deliberately carries no run id or timestamp: repeating a run with the same code,
    settings and inputs then reproduces every output file byte for byte.
    """
    return {"dataset": run["dataset"], "code_sha256": run["code_sha256"],
            "settings": run["settings"], "input_ifc": entry["path"],
            "input_ifc_sha256": entry["ifc_sha256"]}


def finish_run(out, record):
    record.update(finished_utc=utc(),
                  status="completed_with_failures" if record["failures"] else "completed")
    record["output_files"] = {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*"))
                              if p.is_file() and p.name != "run.json"}
    atomic_json(out / "run.json", record)
