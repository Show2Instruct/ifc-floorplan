"""Resolve IFC physical quantities to SI, retaining their unit provenance.

IFC physical simple quantities use Unit when set; otherwise the corresponding
IfcProject.UnitsInContext entry applies. Area/volume units are independent of
length units. Missing, ambiguous and unsupported definitions are explicit errors.
"""
from __future__ import annotations

import math
import ifcopenshell.util.element as ue
import ifcopenshell.util.unit as uu

TYPES = {"AREAUNIT": ("IfcQuantityArea", "AreaValue", "SQUARE_METRE", 2),
         "VOLUMEUNIT": ("IfcQuantityVolume", "VolumeValue", "CUBIC_METRE", 3)}


class QuantityUnitError(ValueError):
    pass


def unit_scale(unit, unit_type, seen=None):
    """Use numeric IFC conversion factors, not exporter-specific unit names."""
    if unit is None:
        raise QuantityUnitError(f"missing {unit_type}")
    seen = set() if seen is None else set(seen)
    identity = unit.id() or id(unit)
    if identity in seen:
        raise QuantityUnitError("cyclic unit conversion")
    seen.add(identity)
    if getattr(unit, "UnitType", None) != unit_type:
        raise QuantityUnitError(f"expected {unit_type}, got {getattr(unit, 'UnitType', None)}")
    _, _, si_name, power = TYPES[unit_type]
    if unit.is_a("IfcSIUnit"):
        if unit.Name != si_name:
            raise QuantityUnitError(f"unsupported SI unit {unit.Name}")
        prefix = unit.Prefix
        if prefix is not None and prefix not in uu.prefixes:
            raise QuantityUnitError(f"unsupported prefix {prefix}")
        factor = uu.get_prefix_multiplier(prefix) ** power
    elif unit.is_a("IfcConversionBasedUnit"):
        dims = unit.Dimensions
        expected = (power, 0, 0, 0, 0, 0, 0)
        if dims is None or tuple(dims) != expected:
            raise QuantityUnitError("conversion unit dimensions do not match quantity")
        if getattr(unit, "ConversionOffset", 0) not in (0, 0.0, None):
            raise QuantityUnitError("offset conversion unsupported for area/volume")
        conv = unit.ConversionFactor
        if conv is None:
            raise QuantityUnitError("missing conversion factor")
        factor = float(conv.ValueComponent.wrappedValue) * unit_scale(conv.UnitComponent, unit_type, seen)
    else:
        raise QuantityUnitError(f"unsupported {unit.is_a()}")
    if not math.isfinite(factor) or factor <= 0:
        raise QuantityUnitError("unit scale must be positive and finite")
    return factor


def project_unit(f, unit_type):
    projects = f.by_type("IfcProject")
    if len(projects) != 1:
        raise QuantityUnitError(f"expected one project, found {len(projects)}")
    assignment = projects[0].UnitsInContext
    units = [u for u in (assignment.Units if assignment else [])
             if getattr(u, "UnitType", None) == unit_type]
    if len(units) != 1:
        raise QuantityUnitError(f"expected one {unit_type}, found {len(units)}")
    return units[0]


def quantity_sets(element):
    """Keep entities (get_psets' plain floats discard per-quantity units).

    An occurrence overrides a type quantity with the same set/name. This avoids
    the installed IfcOpenShell 0.8.5 verbose get_quantities KeyError on real Qto.
    """
    # Keep every same-level definition: dict.update used to silently pick the last
    # of two conflicting quantities with the same set/name, making the reference
    # depend on relationship order. Only an occurrence may override a type value.
    def collect(definitions):
        sets = {}
        for definition in definitions:
            if not definition.is_a("IfcElementQuantity"):
                continue
            quantities = sets.setdefault(definition.Name, {})
            for quantity in definition.Quantities:
                candidates = quantities.setdefault(quantity.Name, [])
                if all(candidate != quantity for candidate in candidates):
                    candidates.append(quantity)
        return sets

    result = {}
    typ = ue.get_type(element)
    if typ is not None and typ != element:
        result = collect(getattr(typ, "HasPropertySets", ()) or ())
    occurrence = []
    for rel in getattr(element, "IsDefinedBy", ()) or ():
        if rel.is_a("IfcRelDefinesByProperties"):
            definition = rel.RelatingPropertyDefinition
            definitions = definition if isinstance(definition, tuple) else (definition,)
            occurrence.extend(definitions)
    for name, quantities in collect(occurrence).items():
        result.setdefault(name, {}).update(quantities)
    return result


def declared_quantity(element, keys, unit_type):
    """Return one selected quantity record, including non-comparable status."""
    entity_type, attribute, _, _ = TYPES[unit_type]
    sets = quantity_sets(element)
    for key in keys:
        candidates = [(name, q)
                      for name, qs in sorted(sets.items(), key=lambda item: str(item[0]))
                      for q in qs.get(key, ())]
        if not candidates:
            continue
        if len(candidates) > 1:
            return {"status": "ambiguous_quantity", "name": key,
                    "reason": "multiple definitions of the preferred quantity",
                    "quantity_step_ids": sorted(q.id() for _, q in candidates)}
        set_name, q = candidates[0]
        rec = {"quantity_step_id": q.id(), "set": set_name, "name": key,
               "unit_type": unit_type, "status": "ok"}
        if not q.is_a(entity_type):
            return dict(rec, status="wrong_quantity_type", reason=q.is_a())
        try:
            value = float(getattr(q, attribute))
        except (TypeError, ValueError, OverflowError):
            return dict(rec, status="invalid_value", reason=f"{attribute} is not a number")
        rec["raw_value"] = value if math.isfinite(value) else None
        if not math.isfinite(value) or value <= 0:
            return dict(rec, status="invalid_value")
        try:
            explicit = q.Unit
            unit = explicit if explicit is not None else project_unit(element.file, unit_type)
            factor = unit_scale(unit, unit_type)
            value_si = value * factor
            if not math.isfinite(value_si) or value_si <= 0:
                return dict(rec, status="invalid_value",
                            reason="converted SI value must be positive and finite")
            rec.update(unit_source="quantity" if explicit is not None else "project",
                       unit_step_id=unit.id(), unit_name=getattr(unit, "Name", unit.is_a()),
                       unit_prefix=getattr(unit, "Prefix", None), scale_to_si=factor,
                       value_si=value_si)
        except (QuantityUnitError, TypeError, ValueError, AttributeError, OverflowError) as ex:
            rec.update(status="unit_error", reason=str(ex))
        return rec
    return {"status": "missing_quantity"}
