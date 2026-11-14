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
            scene = g.trimesh.exchange.load.load_remote(url=address + "/unit_cube.STL")

        mesh = scene.to_mesh()
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

    def test_meshio_reopen_tempfile(self):
        # meshio reads by file name, so the loader's tempfile must be
        # closed before `meshio.read` opens it again, or Windows raises
        # `PermissionError`, and it should be removed afterwards
        import io
        import types
        import warnings

        from trimesh.exchange import misc

        payload = b"not really a gmsh file"
        calls = []

        def read(file_name, file_format):
            with open(file_name, "rb") as f:
                calls.append((file_name, f.read(), file_format))
            return types.SimpleNamespace(
                points=g.np.eye(3), get_cells_type=lambda _: g.np.array([[0, 1, 2]])
            )

        fake = types.SimpleNamespace(extension_to_filetypes={".msh": ["gmsh"]}, read=read)
        missing = object()
        original = getattr(misc, "meshio", missing)
        misc.meshio = fake
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                result = misc.load_meshio(io.BytesIO(payload), file_type="msh")
        finally:
            if original is missing:
                del misc.meshio
            else:
                misc.meshio = original

        assert len(calls) == 1
        file_name, data, file_format = calls[0]
        assert data == payload
        assert file_format == "gmsh"
        assert not g.os.path.exists(file_name)
        assert g.np.array_equal(result["faces"], [[0, 1, 2]])

    def test_load_mesh(self):
        # test the influence of the parameter `process` for `load_mesh`
        with open(g.os.path.join(g.dir_models, "featuretype.STL"), "rb") as file_obj:
            mesh = g.trimesh.load_mesh(file_obj=file_obj, file_type="stl", process=False)
            # check that number of vertices is not being reduced
            assert len(mesh.vertices) == 10428

        with open(g.os.path.join(g.dir_models, "featuretype.STL"), "rb") as file_obj:
            mesh = g.trimesh.load_mesh(file_obj=file_obj, file_type="stl", process=True)
            # check that number of vertices is being reduced
            assert len(mesh.vertices) == 1722

    def test_fileobj(self):
        # make sure we don't close file objects that were passed
        # check load_mesh
        file_obj = open(g.os.path.join(g.dir_models, "featuretype.STL"), "rb")
        assert not file_obj.closed
        mesh = g.trimesh.load_mesh(file_obj=file_obj, file_type="stl")
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

    def test_load_bz2(self):
        mesh = g.trimesh.load(g.os.path.join(g.dir_models, "rock.obj.bz2"), force="mesh")
        assert mesh.is_volume


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
