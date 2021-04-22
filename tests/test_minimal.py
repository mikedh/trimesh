"""
test_minimal.py
--------------

Test things that should work with a *minimal* trimesh install.


"""
import os

import unittest
import trimesh

# the path of the current directory
_pwd = os.path.dirname(
    os.path.abspath(os.path.expanduser(__file__)))
# the absolute path for our reference models
_mwd = os.path.abspath(
    os.path.join(_pwd, '..', 'models'))


def get_mesh(file_name, **kwargs):
    return trimesh.load(os.path.join(_mwd, file_name),
                        **kwargs)


class MinimalTest(unittest.TestCase):

    def test_load(self):
        # files that should work with a minimal install
        file_names = ['fixed_top.ply',
                      'origin_inside.STL',
                      'shared.STL',
                      '1002_tray_bottom.STL',
                      'notenoughindices.obj',
                      'reference.obj',
                      'jacked.obj',
                      'cube_compressed.obj',
                      'cycloidal.ply',
                      'reference.ply',
                      'suzanne.ply',
                      'large_block_obb.STL',
                      '20mm-xyz-cube.stl',
                      'points_agisoft.xyz',
                      'plate_holes.STL',
                      'bunny.ply',
                      'octagonal_pocket.stl',
                      'large_block.STL',
                      'empty.obj',
                      'points_bin.ply',
                      'points_emptyascii.ply',
                      'empty.ply',
                      'pins.glb',
                      'not_convex.obj',
                      'round.stl',
                      'empty.stl',
                      'whitespace.off',
                      'two_objects.obj',
                      'nancolor.obj',
                      'wallhole.obj',
                      'monkey.glb',
                      'torus.STL',
                      'fuze_ascii.ply',
                      'octagonal_pocket.ply',
                      'angle_block.STL',
                      'cap.zip',
                      'points_cloudcompare.xyz',
                      'rabbit.obj',
                      'points_emptyface.ply',
                      'BoxTextured.glb',
                      'singlevn.obj',
                      'unit_cube.STL',
                      'simple_pole.glb',
                      'cube.OBJ',
                      'featuretype.ply',
                      'boolean.glb',
                      'box.obj',
                      'kinematic.tar.gz',
                      'tet.ply',
                      'points_ascii.ply',
                      'cubevt.obj',
                      'BoxTextured.ply',
                      'idler_riser.STL',
                      'points_ascii_with_lists.ply',
                      'cube.glb',
                      'empty_nodes.glb',
                      'quadknot.obj',
                      'noimg.obj',
                      'fuze.obj',
                      'busted.STL',
                      'groups.obj',
                      'fuze.ply',
                      'ADIS16480.STL',
                      'machinist.obj',
                      'textured_tetrahedron.obj',
                      'testplate.glb',
                      'multibody.stl',
                      'unit_sphere.STL',
                      '7_8ths_cube.stl',
                      'teapot.stl',
                      'comments.off',
                      'chair_model.obj',
                      'tube.obj',
                      'ballA.off',
                      'box.STL',
                      'joined_tetrahedra.obj',
                      'featuretype.STL',
                      'fandisk.obj',
                      'TestScene.gltf',
                      'polygonfaces.obj',
                      'ballB.off',
                      'soup.stl']

        for file_name in file_names:
            print(file_name)
            m = get_mesh(file_name)
            if isinstance(m, trimesh.Trimesh):
                assert len(m.face_adjacency.shape) == 2
                assert len(m.vertices.shape) == 2


if __name__ == '__main__':
    trimesh.util.attach_to_log()
    unittest.main()
