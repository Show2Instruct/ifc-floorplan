# ifc-floorplan

Turns an IFC building model into per-storey floor plans: rooms with their outlines and areas,
the walls, doors and windows that bound them, the ways between rooms, and walls and openings in
the parametric form a BIM tool rebuilds from. Each storey is written as JSON for other programs to
read and as an SVG so a person can check that the JSON describes the building it claims to. A
second step turns a run into scene graphs.

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
python -m fp.export_graph --run runs/first --elements --out graphs/first
```

Inputs are read from `data/` inside the repository and treated as read-only. Outputs may not be
written there. Every run writes into a new directory and never touches the input:

```
runs/first/run.json                command, settings, input and output fingerprints, failures
runs/first/source.zip              the code that produced the output
runs/first/environment-freeze.txt  installed package versions
runs/first/models/<model>/building.json, storey_00.json, storey_00.svg, bev.svg
```

A model that fails is recorded with its error and the run carries on. `--workers` runs several
models at once. [docs/output-format.md](docs/output-format.md) describes every field.

## How the plan is cut

`fp/band_projection.py` does the cut and `fp/projections.py` decides where to cut and what belongs
in it.

1. Every element arrives as a triangle mesh in metres, from IfcOpenShell with world coordinates.
2. Each element also gets a whole-element silhouette, projected from above. It is the fallback in
   step 6 and it is what the footprint checks measure.
3. A file that puts several floors' walls on one storey, metres from their own floor, gets a storey
   for each floor it never declared: wall bases cluster at each floor's height, and the elements
   standing there move to it, with the move recorded on each; an element with no shape of its
   own, a curtain wall made of mullions and panels, stands where its parts do. A space the file
   puts on a storey more than 2 m from its base moves to the storey it stands on: a wall can rise
   through the floor above its own, but a space is the floor it stands on.
4. Storeys nearer than 0.6 m to each other are one physical level: a floor listed twice, or a beam
   or datum level just above or below it. The level is cut once, 1.2 m above the floor and 1 cm
   thick, and its rooms go to the storey that owns its spaces, or failing that its walls, doors and
   windows. If the next level is closer than the cut, the cut drops to half the gap.
5. Each element is clipped to the band rather than intersected with a single plane. Triangles are
   clipped against both planes and projected, and the cut caps are rebuilt from the intersection
   contours, kept by nonzero winding rather than parity. Parity cannot tell a cavity inside a wall
   from a solid island inside that cavity; face orientation can, so hollow walls keep their holes.
6. An element whose section cannot be rebuilt does not silently leave a hole in the wall it fills.
   Its whole-element silhouette stands in and the substitution is recorded, whether it belongs to
   this storey or is a wall of another that stands through it. Without this, one unsectionable
   window drains every room on the storey into its neighbours.
7. Nothing is approximated silently. Unclosed contours, disagreeing shell orientations and meshes
   over the triangle budget return no polygon and a reason, which lands in `section_projection` in
   the storey JSON.

Walls, curtain walls, doors, windows, plates and columns count as room barriers, assembly parts
inherit the role of their parent, and so does anything the file says fills an opening. Doors and
windows have to be barriers: each fills an opening cut through a wall, so without them the wall has
a hole at every doorway and nothing encloses. Every opening the file cuts through a barrier closes
its wall at the cut as well, filled or not, so a door drawn open or an archway does not let the
rooms either side flood into one.

## How rooms are found

Where the file declares `IfcSpace`, that is the room. Where it does not, rooms are inferred: the
barriers are rasterised at 12 cm, free space is flood-filled from outside the building, and every
enclosed pocket left over is a room. A cell blocks when it touches a barrier rather than when its
centre falls inside one, so a wall thinner than a cell still holds.

The raster is deliberately pessimistic, which costs about 0.15 m off every side, so each room is
then put back on the wall faces: the region grows past that margin and is trimmed against the
barriers themselves. On a 1 m wide corridor the difference is about a third of the area. Abutting
barriers rarely meet exactly, so growth may not pass a joint narrower than 2 cm: otherwise a room
slips past a door leaf into the recess behind it, and the door has the same room on both sides.

Declared spaces are made disjoint first: a space listed twice is dropped, a space the file puts on
no storey that lies over spaces it does put there is set aside as belonging to another floor, and
where a catch-all area covers the rooms inside it, those rooms keep their floor. Then declared and
inferred rooms are reconciled rather than one or the other. An inferred region inside
a declared space is the same room read twice and is absorbed; several regions inside one space mean
walls divide it, so the parts become the rooms; a region no space contains is a room the file never
declared. A region that floods across declared spaces, because a barrier is missing at the cut, is
cut back to what they leave free, so rooms on a storey never overlap.

Each element is placed in the room it stands in: by its section where the cut passes through it,
and otherwise by its footprint, so furniture below the cut still has a room. Walls, openings and
structure bound rooms rather than stand in them.

## Connections between rooms

An IFC file says which rooms exist but not that a door joins the kitchen to the hallway.
`fp/connectivity.py` derives those edges from the geometry:

| edge | meaning |
|---|---|
| `connected_by_door` | two rooms either side of a door |
| `exterior_door` | a door with a room on one side and open ground on the other |
| `open_passage` | two rooms sharing boundary with nothing built across it |
| `vertically_connected` | two rooms on different storeys joined by a stair, ramp or lift |

A connection is something a person can walk: a way through, and somewhere to stand beyond it.
At ground level, doors and unbuilt openings to the outside lead onto open ground. Above it, they
lead somewhere only onto a floor on the same level, a balcony, terrace, deck or walkway, or floor
that furniture or fittings stand on though no slab is modelled; onto open air they lead
nowhere, as the courtyard of an E-shaped building is a void from its upper storeys. The ground
level is the one at the +-0.00 datum, allowing for a raised ground floor, and a file holding only
upper floors has none.

An opening the file leaves unfilled is a way through when a person can walk it: from within a step
of the floor, at least 1.8 m high and 0.6 m wide. It is crossed like a door and gives an open
passage; a hatch is not a way through. Two rooms whose outlines run within 0.35 m of each other,
along at least 0.6 m with nothing built in between, share an open passage too.

A door is crossed along the axis that goes through it; a leaf drawn open has a footprint about as
deep as it is wide, so both axes are tried. Where a declared space is drawn into the doorway, the
room beginning past the door's own depth is the one it leads to. A stair joins the room it stands
in on each level it serves. Stairs and walls are often assemblies whose parts hold
the geometry, so both are composed from their parts before they are read. A storey with no section
asserts no open passages, since nothing is known about what stands between its rooms.

Each room also records the elements along its outline and how much of it they cover, the space
boundaries IFC calls `IfcRelSpaceBoundary`. Balconies and terraces are picked up separately: slab
that open ground reaches, bordered by a railing and next to a room. At ground level a terrace leads onto
the ground where its railing leaves a gap of at least 0.6 m; railed all round, it is a dead end, as a
balcony is.

## Walls and openings as parameters

`fp/vectorize.py` reads each storey back into the form a model is rebuilt from: walls as an axis, a
thickness, a base and a height; doors and windows by their host wall, their offset along it, their
width, sill and head, archways and hatches left unfilled among them; slabs by their top and
thickness; and the storey's height to the next level, or at the top of the building to the top of
its walls. Openings are placed from the IFC's own opening void where it has one, and whatever holds
a door or window is read as a wall, a curtain panel for instance. Every wall records how well its
footprint fills the rectangle it describes, and whether it has a cavity, so a wall that is not
straight is flagged rather than squared off. A void at a wall's end that takes the wall's whole
height leaves the wall's geometry ending where the opening begins, while the file still names that
wall as the host; the wall is read as it was drawn, running over the opening, so the opening can be
rebuilt in it. An opening stays on the wall the file names unless its void lies in another IfcWall,
as where files credit an opening to the wall joining its own; a curtain wall standing in a void
fills it, and holds it only where no wall can.

`fp/verify_vector.py` rebuilds every wall from its parameters and compares it with the footprint,
and checks that every opening sits on its host wall, including a wall that belongs to another
storey.

## Scene graphs

`fp/export_graph.py` writes a graph per building: building, storey, room, and with `--elements` the
walls, doors, windows, columns, stairs, railings, furniture and lifts, each with its 3D box.
Declared rooms and every element carry their IFC GlobalId, so the graph joins with other tools that
read the same file. Edges
are `contains` down the hierarchy, the connections above between rooms, `bounded_by` from a room
to the elements along its outline, and `hosts` from a wall to its doors and windows. Walls and
openings carry their parameters. Every edge ends on a node.

## Checking the output

```bash
python -m fp.verify_run --run runs/first --out runs/first-check
python -m fp.verify_vector --run runs/first --out runs/first-vector
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

219 tests. 10 of them need the BIM GNI or ResBIM corpus and skip when it is absent.

## Known limits

- The cut works on triangulated geometry, not a parametric IFC section, so an opening exists only
  if the mesh has it.
- A 1 cm band through a sloping or curved element gives a thin sliver, so roof and slab quantities
  do not compare cleanly.
- The full-height fallback overstates that element's section. It is flagged per element rather than
  hidden.
- A wall that is not straight keeps its exact footprint, but its axis is only the best rectangle's.
- Room names come from `IfcSpace`. Inferred rooms are numbered, because the file never named them.
