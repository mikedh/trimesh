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
            mesh = g.trimesh.exchange.load.load_remote(url=address + "/unit_cube.STL")

        assert g.np.isclose(mesh.volume, 1.0)
        assert isinstance(mesh, g.trimesh.Trimesh)

    def test_stl(self):
        model = g.get_mesh("empty.stl")
        assert model.is_empty

    def test_meshio(self):
        try:
            import meshio  # NOQA
        except BaseException:
            return
        # if meshio is importable we should be able to load this
        m = g.get_mesh("insulated.msh")
        assert len(m.faces) > 0
        assert m.area > 1e-5

    def test_fileobj(self):
        # make sure we don't close file objects that were passed
        # check load_mesh
        file_obj = open(g.os.path.join(g.dir_models, "featuretype.STL"), "rb")
        assert not file_obj.closed
        mesh = g.trimesh.load(file_obj=file_obj, file_type="stl")
        # should have actually loaded the mesh
        assert len(mesh.faces) == 3476
        # should not close the file object
        assert not file_obj.closed
        # clean up
        file_obj.close()

        # check load_path
        file_obj = open(g.os.path.join(g.dir_models, "2D", "wrench.dxf"), "rb")
        assert not file_obj.closed
        path = g.trimesh.load(file_obj=file_obj, file_type="dxf")
        assert g.np.isclose(path.area, 1.667, atol=1e-2)
        # should have actually loaded the path
        # should not close the file object
        assert not file_obj.closed
        # clean up
        file_obj.close()


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
