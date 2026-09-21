# ifc-floorplan

Turns an IFC building model into per-storey floor plans: rooms with their outlines and areas,
the walls, doors and windows that bound them, and the connections between rooms. Each storey is
written as JSON for other programs to read and as an SVG so a person can check that the JSON
describes the building it claims to.

Built and tested on the 208 models of the BIM GNI 2025 fundamentals corpus and on 1,029 ResBIM
models. IFC2X3, IFC4 and IFC4X3 go through the same code path.

## Install

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
mkdir -p data
cp /path/to/model.ifc data/
python -m fp.run_pipeline --ifc data/model.ifc --dataset mine --out runs/first
```

Inputs are read from `data/` inside the repository and treated as read-only. Outputs may not be
written there.

Every run writes into a new directory and never touches the input:

```
runs/first/run.json                command, settings, input and output fingerprints, failures
runs/first/source.zip              the code that produced the output
runs/first/environment-freeze.txt  installed package versions
runs/first/models/<model>/building.json, storey_00.json, storey_00.svg, bev.svg
```

A model that fails is recorded with its error and the run carries on.

## How the plan is cut

This is the part worth reading if you want to reuse it. `fp/band_projection.py` does the cut and
`fp/projections.py` decides where to cut and what belongs in it.

1. Every element arrives as a triangle mesh in metres, from IfcOpenShell with world coordinates.
2. Each element also gets a whole-element silhouette, projected from above. It is the fallback in
   step 5 and it is what the footprint checks measure.
3. The cut sits 1.2 m above the storey, 1 cm thick. If the next storey is no higher than the cut,
   the cut drops to half the gap, and if that leaves less than 0.3 m the storey gets no section at
   all. Without this, thin structural and datum levels report the rooms of the floor above.
4. Each element is clipped to the band rather than intersected with a single plane. Triangles are
   clipped against both planes and projected, and the cut caps are rebuilt from the intersection
   contours, kept by nonzero winding rather than parity. Parity cannot tell a cavity inside a wall
   from a solid island inside that cavity; face orientation can, so hollow walls keep their holes.
5. An element whose section cannot be rebuilt does not silently leave a hole in the wall it fills.
   Where it belongs to this storey its whole-element silhouette stands in and the substitution is
   recorded. Without this, one unsectionable window drains every room on the storey into its
   neighbours.
6. Nothing is approximated silently. Unclosed contours, disagreeing shell orientations and meshes
   over the triangle budget return no polygon and a reason, which lands in `section_projection` in
   the storey JSON.

Walls, curtain walls, doors, windows, plates and columns count as room barriers, and assembly
parts inherit the role of their parent. Doors and windows have to be barriers: each fills an
opening cut through a wall, so without them the wall has a hole at every doorway and nothing
encloses.

## How rooms are found

Where the file declares `IfcSpace`, that is the room. Where it does not, rooms are inferred: the
barriers are rasterised at 12 cm, free space is flood-filled from outside the building, and every
enclosed pocket left over is a room. A cell blocks when it touches a barrier rather than when its
centre falls inside one, so a wall thinner than a cell still holds.

The raster is deliberately pessimistic, which costs about 0.15 m off every side, so each room is
then put back on the wall faces: the region grows past that margin and is trimmed against the
barriers themselves. On a 1 m wide corridor the difference is about a third of the area.

Declared and inferred rooms are reconciled rather than one or the other. An inferred region inside
a declared space is the same room read twice and is absorbed; several regions inside one space mean
walls divide it, so the parts become the rooms; a region no space contains is a room the file never
declared and is kept whole.

## Connections between rooms

An IFC file says which rooms exist but not that a door joins the kitchen to the hallway.
`fp/connectivity.py` derives those edges from the geometry:

| edge | meaning |
|---|---|
| `connected_by_door` | two rooms either side of a door |
| `exterior_door` | a door with a room on one side only |
| `open_passage` | two rooms sharing boundary with nothing built across it |
| `vertically_connected` | two rooms on different storeys joined by a stair, ramp or lift |

Balconies and terraces are picked up separately: slab that open ground reaches, bordered by a
railing and next to a room. They sit below the 1.2 m cut, so nothing else would find them.

## Checking the output

```bash
python -m fp.verify_run --run runs/first --out runs/first-check
python -m fp.validate --models model_119 --out runs/validation
```

`validate.py` compares the output with what the IFC file itself declares: footprint areas and
volumes against declared quantities, storey assignment against the file's own tables, and rooms
inferred from walls against held-out `IfcSpace` outlines. See
[docs/validation-metrics.md](docs/validation-metrics.md) for what each measure counts.

These are consistency checks. Both sides come from the same file, so they show that the geometry
is read the way the authoring tool meant it, not that the plan matches a surveyed building.

## Tests

```bash
python -m unittest discover -s tests
```

136 tests. Eight of them need the BIM GNI or ResBIM corpus and skip when it is absent.

## Known limits

- The cut works on triangulated geometry, not a parametric IFC section, so an opening exists only
  if the mesh has it.
- A 1 cm band through a sloping or curved element gives a thin sliver, so roof and slab quantities
  do not compare cleanly.
- The full-height fallback overstates that element's section. It is flagged per element rather than
  hidden.
- A storey whose levels sit too close together produces no rooms at all, by design.
- Room names come from `IfcSpace`. Inferred rooms are numbered, because the file never named them.
