"""Existing graph consumers must retain the floorplan's exterior-door counts."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fp.export_graph import graph_of


class GraphCounts(unittest.TestCase):
    def test_exterior_node_does_not_make_its_door_disappear_from_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'x'
            folder.mkdir()
            (folder/'building.json').write_text(json.dumps(dict(storeys=[dict(
                global_id='S', name='Ground', elevation_m=0)])))
            (folder/'storey_00.json').write_text(json.dumps(dict(rooms=[dict(
                id='r', name='Room', source='IfcSpace', area_m2=4)], room_connections=[dict(
                a='r', b='EXT-S', exterior=True, type='exterior_door')], elements=[])))
            result = graph_of('x', tmp)
            self.assertEqual(result['counts']['exterior_doors'], 1)
            self.assertEqual(result['counts']['exterior_door'], 1)
            self.assertEqual(result['counts']['exterior_nodes'], 1)


def write_run(tmp, building, storeys):
    folder = Path(tmp) / 'x'
    folder.mkdir()
    (folder / 'building.json').write_text(json.dumps(building))
    for index, record in enumerate(storeys):
        (folder / f'storey_{index:02d}.json').write_text(json.dumps(record))


class GraphContent(unittest.TestCase):
    def test_storey_height_comes_from_the_pipeline_and_falls_back_only_at_the_top(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_run(tmp, dict(storeys=[
                dict(global_id='S0', name='G', elevation_m=0., height_m=3.4, storey_index=0),
                dict(global_id='S1', name='R', elevation_m=3.4, height_m=None, storey_index=1)]),
                [dict(rooms=[], elements=[]), dict(rooms=[], elements=[])])
            heights = {n['id']: (n['height'], n['height_from']) for n in graph_of('x', tmp)['nodes']
                       if n['layer'] == 'storey'}
            self.assertEqual(heights, {'S0': (3.4, 'next_level'), 'S1': (3., 'default')})

    def test_an_unfilled_opening_is_a_node_its_wall_hosts(self):
        wall = dict(step_id=1, ifc_class='IfcWall', global_id='W', centroid=[3, .1, 1.5],
                    bbox_min=[0, 0, 0], bbox_max=[6, .2, 3])
        arch = dict(step_id=2, ifc_class='IfcOpeningElement', global_id='A', centroid=[2.6, .1, 1.15],
                    bbox_min=[2, 0, 0], bbox_max=[3.2, .2, 2.3])
        filled = dict(step_id=4, ifc_class='IfcOpeningElement', global_id='V', centroid=[4.4, .1, 1.],
                      bbox_min=[4, 0, 0], bbox_max=[4.9, .2, 2.1])
        vector = dict(walls=[dict(id='wall_1', step_id=1, axis=[[0, .1], [6, .1]], thickness_m=.2,
                                  base_m=0., height_m=3.)],
                      openings=[dict(id='opening_2', step_id=2, kind='opening', host_wall='wall_1', offset_m=2.6,
                                     width_m=1.2, sill_m=0., head_m=2.3)])
        with tempfile.TemporaryDirectory() as tmp:
            write_run(tmp, dict(storeys=[dict(global_id='S', name='G', elevation_m=0., height_m=3., storey_index=0)],
                                relations=[dict(type='voids', source=1, target=2), dict(type='voids', source=1, target=4)]),
                      [dict(rooms=[], elements=[wall, arch, filled], vector=vector)])
            graph = graph_of('x', tmp, with_elements=True)
        nodes = {n['id']: n for n in graph['nodes']}
        self.assertEqual(nodes['e2']['layer'], 'opening')
        self.assertNotIn('e4', nodes)                      # a filled void is its door or window
        self.assertIn(('e1', 'e2', 'hosts'), [(e['a'], e['b'], e['relation']) for e in graph['edges']])
        self.assertEqual(graph['counts']['dangling_edge_endpoints'], 0)

    def test_a_room_carries_its_ifc_global_id_and_a_part_the_space_it_divides(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_run(tmp, dict(storeys=[dict(global_id='S', name='G', elevation_m=0., height_m=3., storey_index=0)]),
                      [dict(rooms=[dict(id='space_5', name='Lobby', source='IfcSpace', area_m2=9., global_id='3xLobby'),
                                   dict(id='room_0_1', name='Office part 1', source='inferred_from_wall_geometry',
                                        area_m2=4., subdivides_declared_space=7, declared_space_global_id='3xOffice')],
                            elements=[])])
            nodes = {n['id']: n for n in graph_of('x', tmp)['nodes']}
        self.assertEqual(nodes['space_5']['global_id'], '3xLobby')
        self.assertIsNone(nodes['room_0_1']['global_id'])
        self.assertEqual(nodes['room_0_1']['declared_space_global_id'], '3xOffice')

    def test_a_space_moved_to_the_storey_it_stands_on_says_which_storey_the_file_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_run(tmp, dict(storeys=[dict(step_id=10, global_id='S3', name='3', elevation_m=13.2, height_m=4.4, storey_index=0),
                                         dict(step_id=20, global_id='S4', name='4', elevation_m=17.6, height_m=4.4, storey_index=1)]),
                      [dict(rooms=[dict(id='space_1', name='Hall', source='IfcSpace', area_m2=34., global_id='3xHall',
                                        storey_assignment='IfcRelContainedInSpatialStructure')], elements=[]),
                       dict(rooms=[dict(id='space_2', name='Hall', source='IfcSpace', area_m2=14., global_id='3xUpper',
                                        storey_assignment='moved to the storey its base stands on', storey_stated=10)],
                            elements=[])])
            graph = graph_of('x', tmp)
        nodes = {n['id']: n for n in graph['nodes']}
        contains = {e['b']: (e['a'], e['provenance']) for e in graph['edges'] if e['relation'] == 'contains'}
        self.assertEqual(nodes['space_2']['storey_stated'], 'S3')
        self.assertNotIn('storey_stated', nodes['space_1'])
        self.assertEqual(contains['space_2'], ('S4', 'geometry'))
        self.assertEqual(contains['space_1'], ('S3', 'ifc'))
        self.assertEqual(nodes['space_2']['provenance'], 'ifc')       # the space itself is the file's

    def test_walls_host_their_doors_and_carry_their_parameters(self):
        wall = dict(step_id=1, ifc_class='IfcWall', global_id='W', centroid=[3, .1, 1.5],
                    bbox_min=[0, 0, 0], bbox_max=[6, .2, 3])
        door = dict(step_id=2, ifc_class='IfcDoor', global_id='D', centroid=[2.4, .1, 1.],
                    bbox_min=[2, 0, 0], bbox_max=[2.9, .2, 2.1])
        vector = dict(walls=[dict(id='wall_1', step_id=1, axis=[[0, .1], [6, .1]], thickness_m=.2,
                                  length_m=6., base_m=0., height_m=3., straight=True)],
                      openings=[dict(id='opening_2', step_id=2, host_wall='wall_1', offset_m=2.45,
                                     width_m=.9, sill_m=0., head_m=2.1)])
        with tempfile.TemporaryDirectory() as tmp:
            write_run(tmp, dict(storeys=[dict(global_id='S', name='G', elevation_m=0., height_m=3.,
                                             storey_index=0)],
                                relations=[dict(type='hosts', source=1, target=2)]),
                      [dict(rooms=[], elements=[wall, door], vector=vector)])
            graph = graph_of('x', tmp, with_elements=True)
        nodes = {n['id']: n for n in graph['nodes']}
        self.assertEqual(nodes['e1']['global_id'], 'W')
        self.assertEqual(nodes['e1']['parameters']['thickness_m'], .2)
        self.assertEqual(nodes['e2']['parameters']['offset_m'], 2.45)
        self.assertIn(('e1', 'e2', 'hosts'), {(e['a'], e['b'], e['relation']) for e in graph['edges']})
        self.assertEqual(graph['counts']['dangling_edge_endpoints'], 0)

    def test_a_wall_built_from_parts_still_becomes_a_node(self):
        wall = dict(step_id=1, ifc_class='IfcWall', global_id='W')        # no box of its own
        vector = dict(walls=[dict(id='wall_1', step_id=1, axis=[[0, .1], [6, .1]], thickness_m=.2,
                                  length_m=6., base_m=0., height_m=3., straight=True)], openings=[])
        with tempfile.TemporaryDirectory() as tmp:
            write_run(tmp, dict(storeys=[dict(global_id='S', name='G', elevation_m=2., height_m=3.,
                                             storey_index=0)]),
                      [dict(rooms=[], elements=[wall], vector=vector)])
            nodes = {n['id']: n for n in graph_of('x', tmp, with_elements=True)['nodes']}
        self.assertIn('e1', nodes)
        self.assertEqual(nodes['e1']['bbox'][0][2], 2.)
        self.assertEqual(nodes['e1']['bbox'][1][2], 5.)


if __name__ == '__main__':
    unittest.main()
