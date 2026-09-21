#!/usr/bin/env python
"""Checks that declared quantities are converted with their own units.

An IFC quantity takes its unit from the quantity itself when one is set, otherwise from the
project's area or volume unit. Neither follows the length unit: model_100 has metre lengths
but cubic-foot volumes. These checks build small IFC files in memory for each unit situation,
then confirm the real model_100 case through the validator.

    python tests/test_quantities.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import ifcopenshell
import ifcopenshell.guid

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fp import quantities as Q
from fp import validate as V

MODEL_100 = Path(__file__).resolve().parents[2] / "data/gni-bim/2025_BIMfundamentals/model_100.ifc"
CUBIC_FOOT = 0.028316846592
SQUARE_FOOT = 0.09290304


class Model:
    """A minimal IFC4 file: one project with the chosen units, and one wall."""

    def __init__(self, area="SQUARE_METRE", volume="CUBIC_METRE"):
        self.f = ifcopenshell.file(schema="IFC4")
        units = [self.si("LENGTHUNIT", "METRE", "MILLI")]
        if area == "SQUARE_FOOT":
            units.append(self.conversion("AREAUNIT", "SQUARE FOOT", SQUARE_FOOT,
                                         self.si("AREAUNIT", "SQUARE_METRE"), (2, 0, 0, 0, 0, 0, 0)))
        elif area:
            units.append(self.si("AREAUNIT", area))
        if volume == "CUBIC_FOOT":
            units.append(self.cubic_foot())
        elif volume:
            units.append(self.si("VOLUMEUNIT", volume))
        self.rooted("IfcProject", Name="p",
                    UnitsInContext=self.f.create_entity("IfcUnitAssignment", Units=units))
        self.wall = self.rooted("IfcWall", Name="w")

    def rooted(self, cls, **attrs):
        return self.f.create_entity(cls, GlobalId=ifcopenshell.guid.new(), **attrs)

    def si(self, unit_type, name, prefix=None):
        return self.f.create_entity("IfcSIUnit", UnitType=unit_type, Prefix=prefix, Name=name)

    def conversion(self, unit_type, name, factor, base, exponents):
        measure = "IfcAreaMeasure" if unit_type == "AREAUNIT" else "IfcVolumeMeasure"
        return self.f.create_entity(
            "IfcConversionBasedUnit",
            Dimensions=self.f.create_entity("IfcDimensionalExponents", *exponents),
            UnitType=unit_type, Name=name,
            ConversionFactor=self.f.create_entity(
                "IfcMeasureWithUnit", ValueComponent=self.f.create_entity(measure, factor),
                UnitComponent=base))

    def cubic_foot(self):
        return self.conversion("VOLUMEUNIT", "CUBIC FOOT", CUBIC_FOOT,
                               self.si("VOLUMEUNIT", "CUBIC_METRE"), (3, 0, 0, 0, 0, 0, 0))

    def volume(self, name, value, unit=None):
        return self.f.create_entity("IfcQuantityVolume", Name=name, VolumeValue=value, Unit=unit)

    def area(self, name, value, unit=None):
        return self.f.create_entity("IfcQuantityArea", Name=name, AreaValue=value, Unit=unit)

    def quantities(self, set_name, *quantities):
        qset = self.rooted("IfcElementQuantity", Name=set_name, Quantities=list(quantities))
        self.rooted("IfcRelDefinesByProperties", RelatedObjects=[self.wall],
                    RelatingPropertyDefinition=qset)

    def type_quantities(self, set_name, *quantities):
        qset = self.rooted("IfcElementQuantity", Name=set_name, Quantities=list(quantities))
        wall_type = self.rooted("IfcWallType", Name="t", PredefinedType="STANDARD",
                                HasPropertySets=[qset])
        self.rooted("IfcRelDefinesByType", RelatedObjects=[self.wall], RelatingType=wall_type)


def volume_of(model):
    return Q.declared_quantity(model.wall, V.VOL_KEYS, "VOLUMEUNIT")


def area_of(model, keys=V.AREA_KEYS):
    return Q.declared_quantity(model.wall, keys, "AREAUNIT")


class Units(unittest.TestCase):
    def ok(self, record):
        self.assertEqual(record["status"], "ok", record)
        return record

    def test_project_si_units_need_no_conversion(self):
        m = Model()
        m.quantities("Qto_WallBaseQuantities", m.volume("GrossVolume", 2.5),
                     m.area("GrossFootprintArea", 4.0))
        v = self.ok(volume_of(m))
        self.assertEqual((v["unit_source"], v["scale_to_si"], v["value_si"]), ("project", 1.0, 2.5))
        self.assertEqual(self.ok(area_of(m))["value_si"], 4.0)

    def test_project_volume_in_cubic_feet_is_converted(self):
        m = Model(volume="CUBIC_FOOT")
        m.quantities("Qto_WallBaseQuantities", m.volume("GrossVolume", 100.0))
        v = self.ok(volume_of(m))
        self.assertEqual(v["unit_source"], "project")
        self.assertAlmostEqual(v["scale_to_si"], CUBIC_FOOT, places=12)
        self.assertAlmostEqual(v["value_si"], 100.0 * CUBIC_FOOT, places=10)

    def test_project_area_in_square_feet_is_converted(self):
        m = Model(area="SQUARE_FOOT")
        m.quantities("Qto_WallBaseQuantities", m.area("GrossFootprintArea", 10.0))
        a = self.ok(area_of(m))
        self.assertAlmostEqual(a["value_si"], 10.0 * SQUARE_FOOT, places=10)

    def test_quantity_unit_overrides_the_project_unit(self):
        m = Model()
        m.quantities("Qto_WallBaseQuantities", m.volume("GrossVolume", 10.0, unit=m.cubic_foot()))
        v = self.ok(volume_of(m))
        self.assertEqual(v["unit_source"], "quantity")
        self.assertAlmostEqual(v["value_si"], 10.0 * CUBIC_FOOT, places=10)

    def test_si_prefix_scales_with_the_quantity_power(self):
        m = Model()
        m.quantities("Qto_WallBaseQuantities",
                     m.area("GrossFootprintArea", 50000.0, unit=m.si("AREAUNIT", "SQUARE_METRE", "CENTI")))
        a = self.ok(area_of(m))
        self.assertAlmostEqual(a["scale_to_si"], 1e-4, places=15)
        self.assertAlmostEqual(a["value_si"], 5.0, places=9)

    def test_missing_project_unit_is_reported_not_assumed(self):
        m = Model(volume=None)
        m.quantities("Qto_WallBaseQuantities", m.volume("GrossVolume", 3.0))
        v = volume_of(m)
        self.assertEqual(v["status"], "unit_error")
        self.assertNotIn("value_si", v)

    def test_unsupported_units_are_reported(self):
        m = Model()
        wrong_power = m.conversion("VOLUMEUNIT", "odd", 2.0, m.si("VOLUMEUNIT", "CUBIC_METRE"),
                                   (2, 0, 0, 0, 0, 0, 0))
        m.quantities("Qto_WallBaseQuantities", m.volume("GrossVolume", 1.0, unit=wrong_power))
        self.assertEqual(volume_of(m)["status"], "unit_error")

        m = Model()
        m.quantities("Qto_WallBaseQuantities",
                     m.volume("GrossVolume", 1.0, unit=m.si("VOLUMEUNIT", "SQUARE_METRE")))
        self.assertEqual(volume_of(m)["status"], "unit_error")

        m = Model()
        m.quantities("Qto_WallBaseQuantities",
                     m.volume("GrossVolume", 1.0, unit=m.si("AREAUNIT", "SQUARE_METRE")))
        self.assertEqual(volume_of(m)["status"], "unit_error")

    def test_wrong_type_invalid_and_missing_quantities(self):
        m = Model()
        m.quantities("Qto_WallBaseQuantities",
                     m.f.create_entity("IfcQuantityLength", Name="GrossVolume", LengthValue=2.0))
        self.assertEqual(volume_of(m)["status"], "wrong_quantity_type")

        m = Model()
        m.quantities("Qto_WallBaseQuantities", m.volume("GrossVolume", 0.0))
        self.assertEqual(volume_of(m)["status"], "invalid_value")

        self.assertEqual(volume_of(Model())["status"], "missing_quantity")

    def test_occurrence_quantity_overrides_type_quantity(self):
        m = Model()
        m.type_quantities("Qto_WallBaseQuantities", m.volume("GrossVolume", 5.0))
        self.assertEqual(self.ok(volume_of(m))["value_si"], 5.0)
        m.quantities("Qto_WallBaseQuantities", m.volume("GrossVolume", 7.0))
        self.assertEqual(self.ok(volume_of(m))["value_si"], 7.0)

    def test_same_quantity_in_two_sets_is_ambiguous(self):
        m = Model()
        m.quantities("Qto_A", m.volume("GrossVolume", 1.0))
        m.quantities("Qto_B", m.volume("GrossVolume", 1.0))
        self.assertEqual(volume_of(m)["status"], "ambiguous_quantity")

    def test_duplicate_quantity_in_one_set_is_ambiguous(self):
        m = Model()
        m.quantities("Qto_A", m.volume("GrossVolume", 1.0), m.volume("GrossVolume", 2.0))
        self.assertEqual(volume_of(m)["status"], "ambiguous_quantity")

    def test_same_name_sets_do_not_silently_override_each_other(self):
        for first, second in ((1.0, 2.0), (2.0, 1.0)):
            with self.subTest(first=first):
                m = Model()
                m.quantities("Qto_A", m.volume("GrossVolume", first))
                m.quantities("Qto_A", m.volume("GrossVolume", second))
                self.assertEqual(volume_of(m)["status"], "ambiguous_quantity")

    def test_an_unnamed_set_does_not_break_quantity_selection(self):
        m = Model()
        m.quantities(None, m.volume("NetVolume", 2.0))
        m.quantities("Qto_A", m.volume("GrossVolume", 3.0))
        self.assertEqual(self.ok(volume_of(m))["value_si"], 3.0)

    def test_occurrence_resolves_only_its_own_type_quantity(self):
        m = Model()
        m.type_quantities("Qto_A", m.volume("GrossVolume", 1.0), m.volume("GrossVolume", 2.0))
        self.assertEqual(volume_of(m)["status"], "ambiguous_quantity")
        m.quantities("Qto_A", m.volume("GrossVolume", 3.0))
        self.assertEqual(self.ok(volume_of(m))["value_si"], 3.0)

    def test_conversion_overflow_and_underflow_are_not_comparable(self):
        for value, prefix in ((1e308, "EXA"), (1e-308, "ATTO")):
            with self.subTest(value=value):
                m = Model()
                m.quantities("Qto_A", m.volume("GrossVolume", value,
                                             unit=m.si("VOLUMEUNIT", "CUBIC_METRE", prefix)))
                result = volume_of(m)
                self.assertEqual(result["status"], "invalid_value")
                self.assertNotIn("value_si", result)

    def test_cyclic_and_offset_conversion_units_are_rejected(self):
        m = Model()
        unit = m.cubic_foot()
        unit.ConversionFactor.UnitComponent = unit
        m.quantities("Qto_A", m.volume("GrossVolume", 1.0, unit=unit))
        self.assertEqual(volume_of(m)["status"], "unit_error")
        self.assertIn("cyclic", volume_of(m)["reason"])

        m = Model()
        base = m.cubic_foot()
        offset = m.f.create_entity("IfcConversionBasedUnitWithOffset",
                                  Dimensions=base.Dimensions, UnitType=base.UnitType,
                                  Name="offset volume", ConversionFactor=base.ConversionFactor,
                                  ConversionOffset=1.0)
        m.quantities("Qto_A", m.volume("GrossVolume", 1.0, unit=offset))
        self.assertEqual(volume_of(m)["status"], "unit_error")

    def test_preferred_key_wins_whichever_set_holds_it(self):
        m = Model()
        m.quantities("Qto_A", m.area("GrossArea", 9.0))
        m.quantities("Qto_B", m.area("NetArea", 8.0))
        a = self.ok(area_of(m, V.SLAB_KEYS))
        self.assertEqual((a["name"], a["value_si"]), ("NetArea", 8.0))


@unittest.skipUnless(MODEL_100.is_file(), f"needs {MODEL_100}")
class Model100(unittest.TestCase):
    """The case that exposed the defect: metre lengths, cubic-foot volumes."""

    @classmethod
    def setUpClass(cls):
        cls.f = ifcopenshell.open(str(MODEL_100))

    def test_validator_converts_volumes_from_cubic_feet_and_keeps_square_metres(self):
        volumes = areas = 0
        for e in self.f.by_type("IfcElement"):
            v = V.declared_volume(e)
            if v["status"] == "ok":
                volumes += 1
                self.assertEqual((v["unit_source"], v["unit_name"]), ("project", "CUBIC FOOT"))
                self.assertAlmostEqual(v["value_si"], v["raw_value"] * CUBIC_FOOT, places=9)
            a = V.declared_footprint_area(e)
            if a["status"] == "ok":
                areas += 1
                self.assertEqual((a["unit_name"], a["scale_to_si"]), ("SQUARE_METRE", 1.0))
        self.assertEqual((volumes, areas), (211, 58))


if __name__ == "__main__":
    unittest.main(verbosity=2)
