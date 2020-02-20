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

    def test_ply_dtype(self):
        # make sure all ply dtype strings are valid dtypes
        dtypes = g.trimesh.exchange.ply.dtypes
        for d in dtypes.values():
            # will raise if dtype string not valid
            g.np.dtype(d)

    def test_meshio(self):
        try:
            import meshio # NOQA
        except BaseException:
            return
        # if meshio is importable we should be able to load this
        m = g.get_mesh('insulated.msh')
        assert len(m.faces) > 0
        assert m.area > 1e-5


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
