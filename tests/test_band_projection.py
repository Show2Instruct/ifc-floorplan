"""Analytic solid cases for the experimental height-band projector."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fp.band_projection import project_band, plane_segments


def vertical_walls(rings, bottom=0., top=3.):
    vertices, faces = [], []
    for ring in rings:
        base = len(vertices)
        vertices.extend([(x,y,bottom) for x,y in ring])
        vertices.extend([(x,y,top) for x,y in ring])
        n = len(ring)
        for i in range(n):
            j = (i+1)%n
            faces.extend([(base+i,base+j,base+n+j),(base+i,base+n+j,base+n+i)])
    return np.asarray(vertices), faces


class BandProjection(unittest.TestCase):
    def test_empty_height_gap_between_disconnected_solids_is_not_a_failure(self):
        lower, a = vertical_walls([[(0,0),(2,0),(2,1),(0,1)]], 0., 1.)
        upper, b = vertical_walls([[(0,0),(2,0),(2,1),(0,1)]], 3., 4.)
        faces = a + [tuple(i+len(lower) for i in f) for f in b]
        polygon, info = project_band(np.vstack([lower,upper]), faces, 1.5, 2.)
        self.assertIsNone(polygon)
        self.assertEqual(info['status'], 'empty_band')

    def test_planar_surface_buffers_only_the_clipped_region(self):
        # Sloped sheet in the vertical x/z plane: whole footprint x=0..4,
        # but the selected height band only occupies x=1..2.
        vertices = np.array([(0,0,0),(4,0,4),(4,0,5),(0,0,1)], dtype=float)
        polygon, info = project_band(vertices, [(0,1,2),(0,2,3)], 2., 2.01)
        self.assertEqual(info['status'], 'sectioned_planar_surface')
        self.assertEqual(info['method'], 'planar_buffered')
        self.assertAlmostEqual(polygon.bounds[0], 1.)
        self.assertAlmostEqual(polygon.bounds[2], 2.01)
        self.assertAlmostEqual(polygon.area, 1.01 * info['nominal_thickness_m'])

    def test_closed_connected_box_with_reversed_faces_is_repaired(self):
        vertices, walls = vertical_walls([[(0,0),(2,0),(2,1),(0,1)]], 0., 3.)
        faces = walls + [(0,2,1),(0,3,2),(4,5,6),(4,6,7)]
        faces[0] = tuple(reversed(faces[0]))
        polygon, info = project_band(vertices, faces, 1., 2.)
        self.assertAlmostEqual(polygon.area, 2.)
        self.assertIn('orientation_repair', info)

    def test_disconnected_shells_with_conflicting_winding_are_not_guessed(self):
        vertices, faces = vertical_walls([[(0,0),(2,0),(2,1),(0,1)],
                                          [(3,0),(3,1),(5,1),(5,0)]])
        polygon, info = project_band(vertices, faces, 1., 2.)
        self.assertIsNone(polygon)
        self.assertEqual(info['status'], 'inconsistent_cut_orientation')

    def test_real_cut_at_horizontal_face_edge_uses_the_band_interior(self):
        import ifcopenshell
        import ifcopenshell.geom
        settings = ifcopenshell.geom.settings()
        settings.set('use-world-coords', True)
        for model, sid, low in [('model_112', 78100, 5.6), ('model_13', 1359, 5.5)]:
            path = Path(__file__).resolve().parents[2] / f'data/gni-bim/2025_BIMfundamentals/{model}.ifc'
            if not path.exists():
                self.skipTest('local IFC regression input unavailable')
            source = ifcopenshell.open(str(path))
            mesh = ifcopenshell.geom.create_shape(settings, source.by_id(sid))
            vertices = np.array(mesh.geometry.verts).reshape(-1,3)
            faces = np.array(mesh.geometry.faces).reshape(-1,3)
            polygon, info = project_band(vertices, faces, low, low+.01)
            interior, _ = project_band(vertices, faces, low+1e-6, low+.01)
            self.assertEqual(info['status'], 'clipped_band', info)
            self.assertTrue(polygon.is_valid)
            self.assertLess(polygon.symmetric_difference(interior).area, 1e-4)

    def test_inside_prism_band_has_filled_cap_not_only_thin_edges(self):
        vertices, faces = vertical_walls([[(0,0),(4,0),(4,3),(0,3)]])
        polygon, info = project_band(vertices, faces, 1., 2.)
        self.assertAlmostEqual(polygon.area, 12.)
        self.assertEqual(info['status'], 'clipped_band')

    def test_hollow_prism_keeps_its_hole(self):
        # A cavity's shell faces point inward, so its ring runs opposite to the outer one.
        vertices, faces = vertical_walls([[(0,0),(10,0),(10,10),(0,10)],
                                          [(4,4),(4,6),(6,6),(6,4)]])
        polygon, info = project_band(vertices, faces, .5, 1.5)
        self.assertTrue(polygon.is_valid)
        self.assertAlmostEqual(polygon.area, 96.)
        self.assertEqual(len(polygon.interiors), 1)
        self.assertEqual(info['projected_triangle_parts'], 0)
        self.assertEqual(sum(c['kept_faces'] for c in info['contours'].values()), 2)

    def test_concave_prism_is_not_replaced_by_a_convex_hull(self):
        vertices, faces = vertical_walls([[(0,0),(2,0),(2,1),(1,1),(1,2),(0,2)]])
        polygon, _ = project_band(vertices, faces, .5, 1.5)
        self.assertAlmostEqual(polygon.area, 3.)

    def test_sloping_wedge_shrinks_at_higher_band(self):
        vertices = np.array([(0,0,0),(2,0,0),(0,0,2),(0,1,0),(2,1,0),(0,1,2)])
        faces = [(0,1,2),(3,5,4),(0,3,4),(0,4,1),(1,4,5),(1,5,2),(2,5,3),(2,3,0)]
        polygon, _ = project_band(vertices, faces, 1., 1.5)
        self.assertAlmostEqual(polygon.area, 1., places=5)

    def test_outside_band_does_not_fall_back_to_full_height(self):
        vertices, faces = vertical_walls([[(0,0),(2,0),(2,1),(0,1)]])
        polygon, info = project_band(vertices, faces, 5., 6.)
        self.assertIsNone(polygon)
        self.assertEqual(info['status'], 'outside_band')

    def test_invalid_bounds_are_rejected(self):
        vertices, faces = vertical_walls([[(0,0),(2,0),(2,1),(0,1)]])
        for low, high in [(1.,1.), (2.,1.), (float('nan'),1.)]:
            with self.assertRaises(ValueError):
                project_band(vertices, faces, low, high)

    def test_invalid_face_indices_are_rejected_without_truncation(self):
        vertices, _ = vertical_walls([[(0,0),(2,0),(2,1),(0,1)]])
        for faces in [[(0,1,99)], [(0,1,-1)], [(0,1,2.5)], [(0,1)], [0,1,2]]:
            with self.assertRaises(ValueError):
                project_band(vertices, faces, 1., 2.)

    def test_incomplete_cut_contour_is_explicit(self):
        vertices, faces = vertical_walls([[(0,0),(2,0),(2,1),(0,1)]])
        polygon, info = project_band(vertices, faces[:-2], 1., 2.)
        self.assertIsNone(polygon)
        self.assertEqual(info['status'], 'incomplete_cut_contours')

    def test_cut_on_shared_horizontal_edges_keeps_caps(self):
        rings = [[(0,0),(2,0),(2,1),(0,1)]]
        lower, lf = vertical_walls(rings, 0., 1.)
        upper, uf = vertical_walls(rings, 1., 3.)
        faces = lf + [tuple(i+len(lower) for i in face) for face in uf]
        polygon, _ = project_band(np.vstack([lower, upper]), faces, 1., 2.)
        self.assertAlmostEqual(polygon.area, 2.)

    def test_nested_island_and_disconnected_part_are_retained(self):
        # Outer solid, a cavity inside it (reversed), a solid island inside the cavity,
        # and a disconnected solid: 100 - 36 + 4 + 1.
        vertices, faces = vertical_walls([
            [(0,0),(10,0),(10,10),(0,10)], [(2,2),(2,8),(8,8),(8,2)],
            [(4,4),(6,4),(6,6),(4,6)], [(12,0),(13,0),(13,1),(12,1)]])
        polygon, _ = project_band(vertices, faces, 1., 2.)
        self.assertAlmostEqual(polygon.area, 69.)
        self.assertEqual(len(polygon.geoms), 3)

    def test_second_solid_inside_the_first_is_not_read_as_a_cavity(self):
        # Same rings as the hollow prism, but the inner shell faces outward, so it is a
        # solid standing inside another solid. Parity would punch a hole here; oriented
        # winding fills it.
        vertices, faces = vertical_walls([[(0,0),(10,0),(10,10),(0,10)],
                                          [(4,4),(6,4),(6,6),(4,6)]])
        polygon, info = project_band(vertices, faces, .5, 1.5)
        self.assertTrue(polygon.is_valid)
        self.assertAlmostEqual(polygon.area, 100.)
        self.assertEqual(len(polygon.interiors), 0)
        self.assertFalse(any(c['mixed_winding_signs'] for c in info['contours'].values()))

    def test_triangle_order_and_winding_do_not_change_projection(self):
        vertices, faces = vertical_walls([[(0,0),(2,0),(2,1),(0,1)]])
        a, _ = project_band(vertices, faces, 1., 2.)
        b, _ = project_band(vertices, [tuple(reversed(f)) for f in reversed(faces)], 1., 2.)
        self.assertAlmostEqual(a.symmetric_difference(b).area, 0.)

    def test_dense_cut_and_missing_faces_do_not_invent_geometry(self):
        vertices, faces = vertical_walls([[(0,0),(2,0),(2,1),(0,1)]])
        with patch('fp.band_projection.MAX_BAND_TRIANGLES', 10):
            for candidate, status in [(faces * 2, 'unsupported_dense_cut'), ([], 'missing_faces')]:
                polygon, info = project_band(vertices, candidate, 1., 2.)
                self.assertIsNone(polygon)
                self.assertEqual(info['status'], status)

    def test_outside_triangles_do_not_exhaust_the_cut_budget(self):
        vertices, faces = vertical_walls([[(0,0),(4,0),(4,3),(0,3)]])
        extra = np.array([(0,0,20),(2,0,20),(0,1,20)])
        combined = np.vstack([vertices, extra])
        many_faces = faces + [(len(vertices),len(vertices)+1,len(vertices)+2)] * 5000
        with patch('fp.band_projection.MAX_BAND_TRIANGLES', 10):
            polygon, info = project_band(combined, many_faces, 1., 2.)
        self.assertAlmostEqual(polygon.area, 12.)
        self.assertEqual(info['band_triangles'], 8)

    def test_contact_edges_need_faces_on_both_sides_to_form_a_cut(self):
        vertices = np.array([(0,0,1),(2,0,1),(0,1,2),(0,1,0)])
        self.assertEqual(plane_segments(vertices, np.array([(0,1,2)]), 1.), [])
        segments = plane_segments(vertices, np.array([(0,1,2),(0,1,3)]), 1.)
        self.assertEqual(len(segments), 1)
        self.assertAlmostEqual(segments[0].length, 2.)

    def test_real_door_cut_does_not_treat_tangent_handle_edges_as_missing_caps(self):
        import ifcopenshell
        import ifcopenshell.geom
        path = Path(__file__).resolve().parents[2] / 'data/gni-bim/2025_BIMfundamentals/model_14.ifc'
        if not path.exists():
            self.skipTest('local IFC regression input unavailable')
        model = ifcopenshell.open(str(path))
        settings = ifcopenshell.geom.settings()
        settings.set('use-world-coords', True)
        for sid, height in [(122641,16.4),(122433,20.2)]:
            mesh = ifcopenshell.geom.create_shape(settings, model.by_id(sid))
            vertices = np.array(mesh.geometry.verts).reshape(-1,3)
            faces = np.array(mesh.geometry.faces).reshape(-1,3)
            polygon, info = project_band(vertices, faces, height, height+.01)
            self.assertEqual(info['status'], 'clipped_band')
            self.assertTrue(polygon.is_valid)
            self.assertGreater(polygon.area, 0.)
            self.assertTrue(all(c['dangles'] == 0 for c in info['contours'].values()))

    def test_horizontal_boundary_face_is_included(self):
        vertices = np.array([(0,0,1),(2,0,1),(2,1,1),(0,1,1)])
        polygon, _ = project_band(vertices, [(0,1,2),(0,2,3)], 1., 2.)
        self.assertAlmostEqual(polygon.area, 2.)

    def test_tangent_sloping_edge_is_not_a_broken_cut_contour(self):
        vertices = np.array([(0,0,1-1e-13),(2,0,1-1e-13),(0,1,2)])
        polygon, info = project_band(vertices, [(0,1,2)], 0., 1.)
        self.assertIsNone(polygon)
        self.assertEqual(info['status'], 'boundary_contact_no_area')

    def test_rotated_convex_boxes_match_independent_clipped_vertex_hulls(self):
        from shapely.geometry import MultiPoint
        vertices, walls = vertical_walls([[(0,0),(2,0),(2,1),(0,1)]], 0., 3.)
        faces = walls + [(0,2,1),(0,3,2),(4,5,6),(4,6,7)]
        edges = {(min(a,b), max(a,b)) for f in faces for a,b in zip(f, (*f[1:], f[0]))}
        rng = np.random.default_rng(731)
        for angle in rng.uniform(.1, 1.4, 20):
            c, s = np.cos(angle), np.sin(angle)
            rotated = vertices @ np.array([[c,0,s],[0,1,0],[-s,0,c]])
            low, high = np.quantile(rotated[:, 2], [.3, .7])
            points = [v[:2] for v in rotated if low <= v[2] <= high]
            for a, b in edges:
                va, vb = rotated[a], rotated[b]
                for z in (low, high):
                    if min(va[2], vb[2]) < z < max(va[2], vb[2]):
                        points.append((va + (z-va[2])/(vb[2]-va[2])*(vb-va))[:2])
            expected = MultiPoint(points).convex_hull
            polygon, info = project_band(rotated, faces, float(low), float(high))
            self.assertIsNotNone(polygon, info)
            self.assertLess(polygon.symmetric_difference(expected).area, 1e-5)


if __name__ == '__main__':
    unittest.main()
