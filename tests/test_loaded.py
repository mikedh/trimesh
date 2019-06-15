try:
    from . import generic as g
except BaseException:
    import generic as g


class LoaderTest(g.unittest.TestCase):

    def test_remote(self):
        """
        Try loading a remote mesh using requests
        """
        # get a unit cube from localhost
        with g.serve_meshes() as address:
            mesh = g.trimesh.exchange.load.load_remote(
                url=address + '/unit_cube.STL')

        assert g.np.isclose(mesh.volume, 1.0)
        assert isinstance(mesh, g.trimesh.Trimesh)

    def test_stl(self):
        model = g.get_mesh('empty.stl')
        assert model.is_empty

    def test_3MF(self):
        # an assembly with instancing
        s = g.get_mesh('counterXP.3MF')
        # should be 2 unique meshes
        assert len(s.geometry) == 2
        # should be 6 instances around the scene
        assert len(s.graph.nodes_geometry) == 6

        # a single body 3MF assembly
        s = g.get_mesh('featuretype.3MF')
        # should be 2 unique meshes
        assert len(s.geometry) == 1
        # should be 6 instances around the scene
        assert len(s.graph.nodes_geometry) == 1

    def test_ply_dtype(self):
        # make sure all ply dtype strings are valid dtypes
        dtypes = g.trimesh.exchange.ply.dtypes
        for d in dtypes.values():
            # will raise if dtype string not valid
            g.np.dtype(d)


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
