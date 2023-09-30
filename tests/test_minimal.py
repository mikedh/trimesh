"""
test_minimal.py
--------------

Test things that should work with a *minimal* trimesh install.


"""
import os
import unittest

import numpy as np

import trimesh

# the path of the current directory
_pwd = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
# the absolute path for our reference models
_mwd = os.path.abspath(os.path.join(_pwd, "..", "models"))


def get_mesh(file_name, **kwargs):
    """
    Load a mesh from our models directory.
    """
    return trimesh.load(os.path.join(_mwd, file_name), **kwargs)


class MinimalTest(unittest.TestCase):
    def test_path_exc(self):
        # this should require *no deps*
        from trimesh.path import packing

        bounds, inserted = packing.rectangles_single([[1, 1], [2, 2]], size=[2, 4])
        assert inserted.all()

        extents = bounds.reshape((-1, 2)).ptp(axis=0)
        assert np.allclose(extents, [2, 3])
        assert bounds.shape == (2, 2, 2)
        density = 5.0 / np.prod(extents)
        assert density > 0.833

    def test_load(self):
        # kinds of files we should be able to
        # load even with a minimal install
        kinds = "stl ply obj off gltf glb".split()

        for file_name in os.listdir(_mwd):
            ext = os.path.splitext(file_name)[-1].lower()[1:]
            if ext not in kinds:
                continue

            m = get_mesh(file_name)
            if isinstance(m, trimesh.Trimesh):
                assert len(m.face_adjacency.shape) == 2
                assert len(m.vertices.shape) == 2

                # make sure hash changes
                initial = hash(m)
                m.faces += 0

                assert hash(m) == initial
                m.vertices[:, 0] += 1.0
                assert hash(m) != initial

    def test_load_path(self):
        # should be able to load a path and export it as a GLB
        # try with a Path3D
        path = trimesh.load_path(np.asarray([(0, 0, 0), (1, 0, 0), (1, 1, 0)]))
        assert isinstance(path, trimesh.path.Path3D)
        scene = trimesh.Scene(path)
        assert len(scene.geometry) == 1
        glb = scene.export(file_type="glb")
        assert len(glb) > 0

        # now create a Path2D
        path = trimesh.load_path(np.asarray([(0, 0), (1, 0), (1, 1)]))
        assert isinstance(path, trimesh.path.Path2D)

        # export to an SVG
        svg = path.export(file_type="svg")
        assert len(svg) > 0

        dxf = path.export(file_type="dxf")
        assert len(dxf) > 0

        scene = trimesh.Scene(path)
        assert len(scene.geometry) == 1
        glb = scene.export(file_type="glb")
        assert len(glb) > 0

    def test_load_wrap(self):
        # check to see if we give a useful message
        # when we *do not* have `lxml` installed
        try:
            import lxml  # noqa

            return
        except BaseException:
            pass

        # we have no 3DXML
        exc = None
        try:
            get_mesh("cycloidal.3DXML")
        except BaseException as E:
            exc = str(E).lower()

        # should have raised
        assert exc is not None

        # error message should have been useful
        # containing which module the user was missing
        if not any(m in exc for m in ("lxml", "networkx")):
            raise ValueError(exc)


if __name__ == "__main__":
    trimesh.util.attach_to_log()
    unittest.main()
