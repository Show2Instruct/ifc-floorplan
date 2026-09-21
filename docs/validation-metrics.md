# Validation measures

What `fp/validate.py` counts, what it divides by, and what each measure cannot show.

There is no ground-truth floor plan for these models, so every measure compares the pipeline with
what the IFC file itself records. The quantity and held-out room checks use references that did not
produce the values being compared, which makes them useful, but both sides still come from one
file. None of this is accuracy against a surveyed building.

Every rate is reported with its numerator and denominator. Pooled values count elements, rooms or
matches across the whole run, with per-model values beside them.

## footprint_area

Does the whole-element plan silhouette have the area the authoring tool declared?

- Items: elements with a footprint and a usable declared area.
- Reference: classes named Slab, Roof or Covering use `NetArea`, else `GrossArea`. Everything else
  uses `GrossFootprintArea`, else `NetFootprintArea`. Converted with the quantity's own unit, else
  the project's area unit.
- Excluded and counted: no comparable quantity, a duplicated quantity name, a missing or
  unsupported unit, a non-numeric or non-positive value, declared area at or below 0.01 m².
- Cannot show: shape or position, since two outlines of equal area can differ completely. A sloped
  slab's declared area may describe its surface rather than its horizontal projection.

`Length × Width` is deliberately not used as a fallback: for a slab "Width" is total layer
thickness and for a wall it is nominal length ignoring openings, so neither measures the solid seen
from above.

## volume

Does footprint area times height reproduce the declared volume?

- Items: elements whose class starts with `IfcWall`, `IfcColumn`, `IfcSlab`, `IfcBeam`,
  `IfcFooting` or `IfcPlate`, with a usable declared volume.
- Reference: `GrossVolume`, else `NetVolume`.
- Cannot show: true volume for sloped, tapered, hollow or irregular elements, where footprint times
  height is only a prism approximation. It reuses the silhouette from the area measure, so the two
  are not independent evidence.

## storey_links

The share of elements whose storey the file states, reported by the route it was read from:
containment, aggregation or nesting, an opening's host, or a filling's opening.

## placement_agreement

How often the placement chain and the relationship tables name the same storey. Where they differ,
the relationship tables are used, because the placement chain alone put doors and windows a storey
or more below the level they physically sit on.

## elevation_fallback

A held-out check of the last-resort rule: hide the stated storey, guess it from the element's base
height, and score the guess against what the file actually said.

## storey_assignment

Elements with geometry that end up on some storey. Complete by design, since the elevation fallback
always returns a storey. It is reported so the completeness is visible rather than assumed.

## geometry_extraction

Elements with a shape representation that produced usable geometry. Elements with only an axis
representation, for example, produce none and appear in no plan.

## room_inference

Rooms inferred from barriers alone, with `IfcSpace` held out, matched against those spaces at
IoU 0.1 and 0.5. Reports recall, precision and the mean IoU of matches.

Matching maximises the number of matched pairs first and total IoU second, one to one, and does not
depend on the order rooms are listed in.

- Cannot show: anything about rooms the file never declared, since the reference is the file's own
  spaces. A storey with no room-enclosing barriers, such as a structural level, produces no rooms
  and stays in the denominator.
