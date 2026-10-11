# Output format

Format version 4. Everything a run writes, what each field means, and how the pieces refer to one
another. Two kinds of consumer are expected: a scene graph, which needs rooms, the elements in them
and the ways between them, and a model rebuilt from the plan, which needs walls and openings in the
parametric form a BIM tool builds from.

## Units, coordinates and identifiers

- Lengths are metres, areas square metres.
- Plan coordinates are the IFC file's own world coordinates, read with IfcOpenShell's
  `use-world-coords`, so every storey of a building shares one frame. z points up.
- `elevation_m` is the storey's elevation from `IfcBuildingStorey`. Fields named `base_m`, `sill_m`,
  `head_m` and `top_m` are heights above that elevation.
- `step_id` is the element's line number in the IFC file. It is stable within that file.
- `global_id` is the IFC GlobalId. It is stable across exports of the same model and is the key to
  use when joining with another tool.

| id pattern | what it names |
|---|---|
| `space_<step_id>` | a room the file declares as an `IfcSpace` |
| `room_<storey>_<k>` | a room inferred from the walls; `_<n>` is appended when one region yields several pieces |
| `OUT-...` | a balcony or terrace |
| `EXT-<storey>` | the outside, as seen from that storey |
| `wall_<step_id>`, `opening_<step_id>`, `slab_<step_id>` | parts of the parametric layer |

## Files

```
<run>/run.json                 command, settings, input and output fingerprints, failures
<run>/source.zip               the code that produced the output
<run>/environment-freeze.txt   installed package versions
<run>/models/<model>/building.json
<run>/models/<model>/storey_NN.json
<run>/models/<model>/storey_NN.svg     the storey drawn, for checking by eye
<run>/models/<model>/bev.svg           all storeys stacked, seen from above
```

## building.json

| field | meaning |
|---|---|
| `schema`, `source_ifc`, `unit_scale_to_m` | the input file and how its units were read |
| `storeys` | one summary per storey; the full record is in `storey_NN.json` |
| `relations` | relations the IFC states: `hosts` (a wall and the door or window in its opening, with the opening's `via_opening`), `voids` (a wall and every opening cut through it, filled or not), `part_of` (an element and the assembly it belongs to) |
| `undeclared_levels` | floors that hold walls or spaces but that the file declares no storey for (ten walls, or one space, at a height more than 2 m from every declared storey), each added as a storey of its own (`storey_source` `wall_bases`), with how many walls stand on it and how many elements were moved to it from the storey the file named, an element with no shape of its own by the height of its parts |
| `vertical_connections` | rooms on different storeys joined by a stair, ramp or lift (below) |
| `graph_counts` | totals over the building |
| `provenance` | the code, input and settings that produced this file |

## storey_NN.json

| field | meaning |
|---|---|
| `name`, `elevation_m`, `storey_index`, `global_id` | the storey |
| `height_m` | distance to the next physical level above; at the top of the building, the median height of the walls standing on its floor; `null` when it has none |
| `height_from` | `next_level` or `walls`, whichever gave `height_m` |
| `at_grade` | whether the storey is the ground level, the physical level at the +-0.00 datum within 1.5 m; a file holding only upper floors has none |
| `rooms` | the storey's rooms (below) |
| `room_connections` | ways between rooms, and out of the building (below) |
| `room_boundaries` | which elements bound each room, and for how long; an opening left unfilled bounds a room as `IfcOpeningElement` |
| `opening_holes` | every opening cut through a barrier that the cut passes: the hole it leaves in its wall (`geometry`), its `host` (with `host_from` and `host_stated` when it was cut into the wall its void overlaps rather than the one named), what fills it (`filled_by`, `null` for an archway or hatch), `sill_m`, `head_m`, `width_m`, and whether a person can walk through it (`walkable`) |
| `declared_spaces_set_aside` | declared spaces left out, with the reason: repeats of another at the same height, spaces of a floor the file does not contain, zones inside a room the walls enclose (`in` names the room), or catch-all spaces the rooms inside them cover entirely |
| `outdoor_spaces` | balconies and terraces: railed floor outside the walls, next to a room |
| `vector` | walls, openings and slabs in parametric form (below) |
| `elements` | every element owned by the storey, with class, names, type, material, property sets, 3D box and plan footprint; `storey_assignment` says how it came to this storey (the IFC relations, `inferred_from_elevation` or `inferred_clamped` where the file names none, or why it was moved), and `storey_stated` the storey the file names for a moved element |
| `projections` | what the plan's cut passes through, element by element, and whether each is a room barrier |
| `section_projection` | how the cut was made: its height band, what was excluded and why, what fell back to a whole-element silhouette |
| `geometry_quality` | `sectioned`, `approximate`, `partial` or `unavailable`, with reasons, for consumers that need to filter |
| `room_inference`, `room_reconciliation`, `room_overlaps` | how the rooms were found and reconciled; `room_overlaps` is empty when rooms are disjoint, as they should be |

### Rooms

| field | meaning |
|---|---|
| `id`, `name`, `source` | `source` is `IfcSpace` or `inferred_from_wall_geometry` |
| `polygon`, `geometry` | the outline; `geometry` keeps holes, as GeoJSON MultiPolygon coordinates |
| `area_m2` | area of `geometry`, measured on the wall faces |
| `floor_to_floor_m` | the storey's `height_m`, an upper bound on the room's |
| `bounded_share` | share of the outline that runs along a wall, door or window |
| `subdivides_declared_space` | set when walls divide one declared space and the parts are the rooms |
| `trimmed_by_declared_spaces` | set when an inferred room was cut back to what the declared spaces leave free |
| `carved_by` | set on a declared space that overlapped smaller declared spaces; they keep the shared floor |
| `trimmed_by_rooms` | set on a room that still overlapped another after reconciliation; a declared room keeps the shared floor over an inferred one, otherwise the smaller room does |
| `global_id`, `declared_space_global_id` | the IfcSpace's GlobalId; on a part of a divided space, the GlobalId of the space it divides |
| `storey_assignment`, `storey_stated` | on a declared space, how it came to this storey, as for elements; `storey_stated` is set when it was moved, and is the step id of the storey the file names, or `null` where it names none. A space more than 2 m from the storey the file names moves to the storey its base stands on |

Rooms on one storey never overlap. Every opening's hole in its wall closes the room at the cut, so
a doorless doorway or a door with its leaf drawn open does not merge the rooms either side; an
archway inside one declared space does not divide it. A void that repeats another opening, the
same doorway cut again through a wall that joins there, is marked `same_as` in `opening_holes`
and counted once. When two declared spaces overlap, the walls decide which is the room: a
smaller space inside a room the walls enclose is a zone of it, and a larger space spread over
rooms the walls enclose is a catch-all that gives them their floor.

### Connections

| type | meaning |
|---|---|
| `connected_by_door` | two rooms either side of a door, or a room and a balcony; `door` is its GlobalId |
| `exterior_door` | a door from a room to the outside, onto open ground at ground level or onto a slab above it |
| `open_passage` | two rooms either side of an unfilled opening a person can walk through (`opening` is its GlobalId), or sharing at least 0.6 m of outline, at most 0.35 m apart, with nothing built across at the cut; carries `barrier_evidence` when barriers were missing from the section |
| `open_passage` with `exterior: true` | a room open to the outside with nothing built across the opening, under the same rule as doors; also a terrace at ground level whose railing leaves a gap of at least 0.6 m (steps or a gate); a terrace railed all round is a dead end, as a balcony is |
| `vertically_connected` | in `building.json`: two rooms on different storeys, with `kind` (`stair`, `ramp`, `lift`), the joining `element`, and the storeys as `storey_a`/`storey_b` (step ids, `null` in a file without IfcBuildingStorey) and `storey_index_a`/`storey_index_b` |

Every connection records `source_of_truth`, the geometric reason it exists. A door is crossed
along its wall's normal until a room is met on each side; where one room's outline reaches into
the doorway, as a space declared to the far face of the wall does, the room beginning past the
door's own depth is taken instead, and `connection_counts` says how often as
`room_reaching_into_the_doorway`.

A connection means someone can walk it: there is a way through, a door or an opening with nothing
built across, and somewhere to stand on the far side, which is another room, open ground at ground
level, or above it a floor on the same physical level: a slab such as a balcony, terrace, deck or
walkway, or floor that furniture, fittings or a railing stand on. Above the ground floor, a door or
opening onto open air with no floor beyond is a void, as the courtyard of an E-shaped building is
from its upper storeys, and is not a connection. Those are counted in `connection_counts` as
`door_onto_a_void` and `opening_onto_a_void` so they stay visible.

An opening left unfilled is walkable when its bottom is within 0.3 m of the floor, its top at
least 1.8 m above it, and it is at least 0.6 m wide. It is crossed the way a door is, and its
outcomes are counted apart, as `connected_by_unfilled_opening`, `exterior_unfilled_opening`,
`unfilled_opening_onto_a_void` and so on. One that is not walkable, a hatch, closes its wall.

### The parametric layer

This is the part a model is rebuilt from.

**Walls**

| field | meaning |
|---|---|
| `axis` | the wall's centre line, `[[x0, y0], [x1, y1]]` |
| `length_m`, `thickness_m` | along and across the axis |
| `base_m`, `height_m` | bottom above the storey, and height |
| `straight` | whether the footprint fills the rectangle the axis and thickness describe |
| `fill` | how much of that rectangle the footprint's outline covers |
| `solid_fill` | how much of it is solid; below `fill` where the wall has a cavity or separate leaves |
| `composed_from_parts` | number of parts the wall was read from, where the IfcWall itself carries no shape |
| `ifc_class` | usually `IfcWall`; anything else that holds a door or window, a curtain panel for instance, is read as a wall too |
| `body_m`, `extended_over` | set on a wall whose end an opening's void took away: a void at a wall's end that takes the wall's whole height leaves its geometry ending where the opening begins, though the wall was drawn through it. The axis then runs over those openings (`extended_over`), and `body_m` gives the stretch of the axis, from `axis[0]`, where the wall's own geometry stands. A pier, no longer than one and a half times its thickness, may be read along either side for this |

**Openings** (doors, windows, and openings left unfilled)

| field | meaning |
|---|---|
| `kind` | `door`, `window`, or `opening` for an archway, doorless doorway or hatch cut through a wall and left unfilled |
| `host_wall` | the `wall_...` that holds it, from the IFC's own relations |
| `host_from`, `host_stated` | set to `geometry` and the wall the file names when the opening does not lie in that wall: at a join, files credit an opening to the wall beside it, and it goes on the wall its void overlaps instead. That is an IfcWall: a curtain wall standing in a void fills it, and holds it only where no wall can. A void standing in a gap of the straight wall named, on that wall's line, stays with it |
| `offset_m` | distance along the host's axis, from `axis[0]`, to the opening's centre |
| `width_m` | width along the axis |
| `sill_m`, `head_m` | bottom and top above the storey |
| `off_axis_m` | distance of the opening's centre from the axis; within half the wall's thickness when placed correctly |
| `placed_from` | `opening_void` where the IFC's opening element gave the hole, else `filling_element` |
| `rooms` | the room pairs the door or opening connects |

A void that repeats an opening already found, the same doorway cut again through a wall that joins
there, is counted once (`counts.voids_repeating_another_opening`); a repeat covers half of the
other's hole at the same height, so a void right above a door on the floor below is an opening
of its own.

**Slabs**: `top_m`, `thickness_m`, `area_m2`, with the outline in the element record.

To rebuild a storey: place each wall as a straight extrusion of its axis, `thickness_m` wide, from
`elevation_m + base_m` for `height_m`; cut each opening into its host wall at `offset_m` along the
axis, `width_m` wide, from `sill_m` to `head_m`, and fill it with its door or window; add slabs and
rooms from their outlines. An opening at a wall's end that took the end away lies within the wall
too, as the wall runs over it (`extended_over`); `fp/verify_vector.py` counts the few that still lie
off their host, where the file names a wall the opening does not touch. Walls marked
`straight: false` keep their exact footprint in the element record and should be rebuilt from that
instead.

## Scene graphs

`fp/export_graph.py` writes one `<model>.graph.json` per building:

- nodes: `building`, `storey` (with `elevation`, `height` and `height_from`, which is `default`
  where the storey had no height and 3 m was assumed), `space` (a room, with outline, area,
  `centroid3d`, `height`, provenance, and `storey_stated` when the file names another storey),
  `outdoor`, `exterior`, and with `--elements` the walls, doors, windows, openings left unfilled,
  columns, beams, stairs, railings, furniture and lifts, each with its 3D box, GlobalId, material
  and, for walls and openings, the parametric fields above;
- edges: `contains` down the hierarchy, with provenance `geometry` where a space's storey comes
  from the height of its base rather than from the file; `connected_by_door`, `exterior_door`,
  `open_passage` and `vertically_connected` between rooms; `bounded_by` from a room to the
  elements along its outline; `hosts` from a wall to the door, window or unfilled opening in it.

Every edge ends on a node; `counts.dangling_edge_endpoints` is reported and is zero.
