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
        # formats that should work with a minimal install
        ext = {'stl', 'glb', 'obj',
               'gltf', 'ply', 'off'}

        for file_name in os.listdir(_mwd):
            # only try loading if the file format matches
            kind = os.path.splitext(file_name.lower())[1][1:]
            if kind not in ext:
                continue
            print(file_name)
            m = get_mesh(file_name)
            if isinstance(m, trimesh.Trimesh):
                assert len(m.face_adjacency.shape) == 2


if __name__ == '__main__':
    trimesh.util.attach_to_log()
    unittest.main()
